import importlib
import torch
import torch.nn.functional as F

from mmcv.cnn import ConvModule, kaiming_init

from ..builder import HEADS, build_head, build_loss
from .htc_roi_head import HybridTaskCascadeRoIHead
from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)


def max_cls(ms_scores):
    return torch.stack(ms_scores).max(0)[0]


def mean_cls(ms_scores):
    return sum(ms_scores) / float(len(ms_scores))


def load_mod(name):
    mod = importlib.import_module('mmdet.models.roi_heads.r3_cnn_roi_head')
    return getattr(mod, name)


@HEADS.register_module()
class RecRoIHead(HybridTaskCascadeRoIHead):

    """RecRoi + ms iou only on last train/test step."""

    def __init__(self, stages, num_stages_test=None, mask_iou_head=None,
                 merge_cls_results=None, **kwargs):
        super(RecRoIHead, self).__init__(**kwargs)
        self.stages = stages
        self.num_stages_test = num_stages_test or kwargs.get('num_stages')
        assert self.num_stages_test > 0
        if mask_iou_head is not None:
            self.mask_iou_head = build_head(mask_iou_head)
        self.merge_cls_results = mean_cls
        if merge_cls_results is not None:
            self.merge_cls_results = load_mod(merge_cls_results)

    @property
    def with_mask_iou(self):
        """bool: whether the detector has Mask IoU head."""
        return hasattr(self, 'mask_iou_head') and self.mask_iou_head is not None

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(RecRoIHead, self).init_weights(pretrained)
        if self.with_mask_iou:
            self.mask_iou_head.init_weights()

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposal_list (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None, list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None, Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            gt_semantic_seg (None, list[Tensor]): semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        losses = dict()
        if self.with_semantic:
            # use semantic output obtained from gt masks
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(x, semantic_pred, gt_masks)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # get next bbox/mask index from the list
            idx = self.stages[i]

            # bbox head forward and loss
            bbox_results = \
                self._bbox_forward_train(
                    idx, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_results['bbox_targets'][0]

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[idx].refine_bboxes(
                            bbox_results['rois'], roi_labels,
                            bbox_results['bbox_pred'], pos_is_gts, img_metas)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []
                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_bboxes[j],
                                gt_bboxes_ignore[j], gt_labels[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_labels[j],
                                feats=[lvl_feat[j][None] for lvl_feat in x])
                            sampling_results.append(sampling_result)
                mask_results = self._mask_forward_train(
                    idx, x, sampling_results, gt_masks, rcnn_train_cfg,
                    semantic_feat, ret_intermediate_results=True)
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[idx].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

            # mask iou head forward and loss
            if i == self.num_stages - 1 and self.with_mask_iou:
                pos_labels = torch.cat([
                    res.pos_gt_labels for res in sampling_results])
                pos_mask_pred = mask_results['mask_pred'][
                    range(mask_results['mask_pred'].size(0)), pos_labels]
                mask_iou_pred = self.mask_iou_head(mask_results['mask_feats'],
                                                   pos_mask_pred)
                pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)),
                                                  pos_labels]

                mask_iou_targets = self.mask_iou_head.get_targets(
                    sampling_results, gt_masks, pos_mask_pred,
                    mask_results['mask_targets'], self.train_cfg[i])
                loss_mask_iou = self.mask_iou_head.loss(pos_mask_iou_pred,
                                                        mask_iou_targets)
                losses.update(loss_mask_iou)

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        img_shape = img_metas[0]['img_shape']
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages_test):
            # get bbox/mask index
            idx = self.stages[i]

            bbox_head = self.bbox_head[idx]
            bbox_results = self._bbox_forward(
                idx, x, rois, semantic_feat=semantic_feat)
            ms_scores.append(bbox_results['cls_score'])

            if i < self.num_stages_test - 1:
                bbox_label = bbox_results['cls_score'].argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label,
                                                  bbox_results['bbox_pred'],
                                                  img_metas[0])

        cls_score = self.merge_cls_results(ms_scores)
        det_bboxes, det_labels = self.bbox_head[-1].get_bboxes(
            rois,
            cls_score,
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            # init mask iou score
            mask_scores = None

            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head[-1].num_classes
                segm_result = [[] for _ in range(mask_classes)]

                # mask iou branch
                if self.with_mask_iou:
                    mask_scores = [[] for _ in range(mask_classes)]
            else:
                _bboxes = (
                    det_bboxes[:, :4] * det_bboxes.new_tensor(scale_factor)
                    if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None
                for i in range(self.num_stages_test):
                    idx = self.stages[i]
                    mask_head = self.mask_head[idx]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(
                        aug_masks,
                        [img_metas] * self.num_stages_test,
                        self.test_cfg)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)

                # mask iou branch
                if self.with_mask_iou:
                    # mask iou score
                    mask_iou_pred = self.mask_iou_head(
                        mask_feats,
                        mask_pred[range(det_labels.size(0)),
                                  det_labels])
                    mask_scores = self.mask_iou_head.get_mask_scores(
                        mask_iou_pred, det_bboxes, det_labels)

            # update ensemble in case of mask iou scores
            if mask_scores is not None:
                ms_segm_result['ensemble'] = (segm_result, mask_scores)
            else:
                ms_segm_result['ensemble'] = segm_result

        if self.with_mask:
            results = (
                ms_bbox_result['ensemble'], ms_segm_result['ensemble']
            )
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, img_feats, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        if self.with_semantic:
            semantic_feats = [
                self.semantic_head(feat)[1] for feat in img_feats
            ]
        else:
            semantic_feats = [None] * len(img_metas)

        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, semantic in zip(img_feats, img_metas, semantic_feats):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages_test):
                # get bbox/mask head idx
                idx = self.stages[i]

                bbox_head = self.bbox_head[idx]
                bbox_results = self._bbox_forward(
                    idx, x, rois, semantic_feat=semantic)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages_test - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = bbox_head.regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = self.merge_cls_results(ms_scores)
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            # int mask iou score
            mask_scores = None

            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes -
                                              1)]
                # mask iou branch
                if self.with_mask_iou:
                    mask_scores = [[] for _ in range(mask_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta, semantic in zip(img_feats, img_metas,
                                                 semantic_feats):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    mask_feats = self.mask_roi_extractor[-1](
                        x[:len(self.mask_roi_extractor[-1].featmap_strides)],
                        mask_rois)
                    if self.with_semantic:
                        semantic_feat = semantic
                        mask_semantic_feat = self.semantic_roi_extractor(
                            [semantic_feat], mask_rois)
                        if mask_semantic_feat.shape[-2:] != mask_feats.shape[
                                -2:]:
                            mask_semantic_feat = F.adaptive_avg_pool2d(
                                mask_semantic_feat, mask_feats.shape[-2:])
                        mask_feats += mask_semantic_feat
                    last_feat = None
                    for i in range(self.num_stages_test):
                        idx = self.stages[i]
                        mask_head = self.mask_head[idx]
                        if self.mask_info_flow:
                            mask_pred, last_feat = mask_head(
                                mask_feats, last_feat)
                        else:
                            mask_pred = mask_head(mask_feats)
                        aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)

                # mask iou branch
                if self.with_mask_iou:
                    # mask iou score
                    mask_iou_pred = self.mask_iou_head(
                        mask_feats,
                        mask_pred[range(det_labels.size(0)),
                                  det_labels])
                    mask_scores = self.mask_iou_head.get_mask_scores(
                        mask_iou_pred, det_bboxes, det_labels)

            # update ensemble in case of mask iou scores
            if mask_scores is not None:
                return bbox_result, (segm_result, mask_scores)
            else:
                return bbox_result, segm_result

            return bbox_result, (segm_result, mask_scores)
        else:
            return bbox_result
