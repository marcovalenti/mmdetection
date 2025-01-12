import importlib
import torch
import torch.nn.functional as F

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class R3RoIHead(CascadeRoIHead):
    """Hybrid task cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1901.07518
    """

    def __init__(self,
                 stages,
                 num_stages,
                 stage_loss_weights,
                 num_stages_test=None,
                 mask_iou_head=None,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 ret_intermediate_results=True,
                 with_mask_loop=True,
                 **kwargs): 
        super(R3RoIHead,
              self).__init__(num_stages, stage_loss_weights, **kwargs)
        assert self.with_bbox  # and self.with_mask
        assert not self.with_shared_head  # shared head is not supported

        if semantic_head is not None:
            if len(semantic_fusion) > 0:
                self.semantic_roi_extractor = build_roi_extractor(
                    semantic_roi_extractor)
            self.semantic_head = build_head(semantic_head)

        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow
        self.ret_intermediate_results = ret_intermediate_results
        self.with_mask_loop = with_mask_loop

        # stages configuration
        self.stages = stages
        self.num_stages_test = num_stages_test or num_stages
        assert self.num_stages_test > 0
        # mask iou configuration
        if mask_iou_head is not None:
            self.mask_iou_head = build_head(mask_iou_head)

        if self.bbox_head[-1].with_dis :
            self.reference_labels = self.bbox_head[-1].reference_labels
            self.classes = self.bbox_head[-1].classes

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(R3RoIHead, self).init_weights(pretrained)
        if self.with_semantic:
            self.semantic_head.init_weights()
        if self.with_mask_iou:
            self.mask_iou_head.init_weights()

    @property
    def with_mask_iou(self):
        """bool: whether the detector has Mask IoU head."""
        return hasattr(self, 'mask_iou_head') and \
            self.mask_iou_head is not None

    @property
    def with_semantic(self):
        """bool: whether the head has semantic head"""
        if hasattr(self, 'semantic_head') and self.semantic_head is not None:
            return True
        else:
            return False

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        outs = ()
        # semantic head
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None
        # bbox heads
        rois = bbox2roi([proposals])
        for i in range(self.num_stages_test):
            idx = self.stages[i]
            bbox_results = self._bbox_forward(
                idx, x, rois, semantic_feat=semantic_feat)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            mask_roi_extractor = self.mask_roi_extractor[-1]
            mask_feats = mask_roi_extractor(
                x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_semantic and 'mask' in self.semantic_fusion:
                mask_semantic_feat = self.semantic_roi_extractor(
                    [semantic_feat], mask_rois)
                mask_feats += mask_semantic_feat
            last_feat = None
            for i in range(self.num_stages):
                idx = self.stages[i]
                mask_head = self.mask_head[idx]
                if self.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)
                outs = outs + (mask_pred, )
        return outs

    def _bbox_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            rcnn_train_cfg,
                            semantic_feat=None):
        """Run forward function and calculate loss for box head in training."""
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(
            stage, x, rois, semantic_feat=semantic_feat)
        if bbox_head.with_dis:
            bbox_targets = bbox_head.get_targets(sampling_results, gt_bboxes,
                                                 gt_labels, rcnn_train_cfg, 
                                                 self.reference_labels,
                                                 self.classes)
            
            loss_bbox = bbox_head.loss(bbox_results['cls_score'],
                                   bbox_results['bbox_pred'],
                                   bbox_results['dis_pred'],
                                   rois,
                                   *bbox_targets)
        else:
            bbox_targets = bbox_head.get_targets(sampling_results, gt_bboxes,
                                                 gt_labels, rcnn_train_cfg)
        
            loss_bbox = bbox_head.loss(bbox_results['cls_score'],
                                   bbox_results['bbox_pred'],
                                   rois,
                                   *bbox_targets)
        '''
        ind_0 = bbox_targets[4] == 0
        ind_1 = bbox_targets[4] == 1
        ind_2 = bbox_targets[4] == 2
        ind_3 = bbox_targets[4] == 3
        ind_neg = bbox_targets[4] == -1
        import logging
        from mmcv.utils import print_log
        logger = logging.getLogger(__name__)
        print_log("DIS_LABEL_COUNT: num_0= {}, num_1= {}, num_2= {}, num_3= {}, num_neg= {}"  \
            .format(bbox_targets[4][ind_0].numel(), bbox_targets[4][ind_1].numel(), 
                 bbox_targets[4][ind_2].numel(), bbox_targets[4][ind_3].numel(),
                 bbox_targets[4][ind_neg].numel()), logger = logger)
        '''
        bbox_results.update(
            loss_bbox=loss_bbox,
            rois=rois,
            bbox_targets=bbox_targets,
        )
        
        return bbox_results

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            semantic_feat=None):
        """Run forward function and calculate loss for mask head in
        training."""
        idx = self.stages[stage]

        mask_roi_extractor = self.mask_roi_extractor[idx]
        mask_head = self.mask_head[idx]
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        pos_rois)

        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             pos_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat

        # mask information flow
        # forward all previous mask heads to obtain last_feat, and fuse it
        # with the normal mask feature
        if self.mask_info_flow:
            last_feat = None
            loop_idx = stage
            # check if you want internal loop for mask
            if not self.with_mask_loop:
                loop_idx = idx
            for i in range(loop_idx):
                idx = self.stages[i]
                last_feat = self.mask_head[idx](
                    mask_feats, last_feat, return_logits=False)
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
        else:
            mask_pred = mask_head(mask_feats, return_feat=False)

        mask_targets = mask_head.get_targets(sampling_results, gt_masks,
                                             rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)

        mask_results = dict(loss_mask=loss_mask)

        if self.ret_intermediate_results:
            mask_results.update(dict(
                loss_mask=loss_mask,
                mask_pred=mask_pred,
                mask_feats=mask_feats,
                mask_targets=mask_targets))

        return mask_results

    def _bbox_forward(self, stage, x, rois, semantic_feat=None):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat

        cls_score, bbox_pred, dis_pred = bbox_head(bbox_feats)
        if dis_pred != None:
            bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, dis_pred=dis_pred)
        else:
            bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    # TODO nobody call it! can we deprecate it?
    def _mask_forward_test(self, stage, x, bboxes, semantic_feat=None):
        """Mask head forward function for testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_rois = bbox2roi([bboxes])
        mask_feats = mask_roi_extractor(
            x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             mask_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat
        if self.mask_info_flow:
            last_feat = None
            last_pred = None
            for i in range(stage):
                idx = self.stages[i]
                mask_pred, last_feat = self.mask_head[idx](
                    mask_feats, last_feat)
                if last_pred is not None:
                    mask_pred = mask_pred + last_pred
                last_pred = mask_pred
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
            if last_pred is not None:
                mask_pred = mask_pred + last_pred
        else:
            mask_pred = mask_head(mask_feats)
        return mask_pred

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
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None

        for i in range(self.num_stages):
            # get next bbox/mask index from list
            idx = self.stages[i]

            self.current_stage = idx
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
            
            # bbox head forward and loss
            bbox_results = \
                self._bbox_forward_train(
                    idx, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_results['bbox_targets'][0]
            
            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)
                    
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
            
            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,
                    semantic_feat)
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

        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        if self.bbox_head[-1].with_dis:
            ms_dis_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages_test):
            idx = self.stages[i]

            bbox_head = self.bbox_head[idx]
            bbox_results = self._bbox_forward(
                idx, x, rois, semantic_feat=semantic_feat)
            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            if self.bbox_head[-1].with_dis:
                dis_pred = bbox_results['dis_pred']
            num_proposals_per_img = tuple(len(p) for p in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            if self.bbox_head[-1].with_dis:
                dis_pred = dis_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)

            if i < self.num_stages_test - 1:
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([
                    bbox_head.regress_by_class(rois[i], bbox_label[i],
                                               bbox_pred[i], img_metas[i])
                    for i in range(num_imgs)
                ])

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        if self.bbox_head[-1].with_dis:
            det_diseases = []
        for i in range(num_imgs):
            if self.bbox_head[-1].with_dis:
                det_bbox, det_label, det_dis = self.bbox_head[-1].get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    dis_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
                det_diseases.append(det_dis)
            else:
                det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
        bbox_result = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
             
        ms_bbox_result['ensemble'] = bbox_result
        if self.bbox_head[-1].with_dis:
            for i in range(num_imgs):
                diss = det_diseases[i].detach().cpu().numpy()
                labels = det_labels[i].detach().cpu().numpy()
                dis_result = [diss[labels == j] for j in range(self.bbox_head[-1].num_classes)]
            ms_dis_result['ensemble'] = [dis_result]
        if self.with_mask:
            # init mask iou score
            mask_scores = []

            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]

                # mask iou branch
                if self.with_mask_iou:
                    mask_scores = [[[] for _ in range(mask_classes)]]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i]
                    for i in range(num_imgs)
                ]
                mask_rois = bbox2roi(_bboxes)
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None

                num_bbox_per_img = tuple(len(_bbox) for _bbox in _bboxes)
                for i in range(self.num_stages_test):
                    idx = self.stages[i]

                    mask_head = self.mask_head[idx]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)

                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_bbox_per_img, 0)
                    aug_masks.append(
                        [mask.sigmoid().cpu().numpy() for mask in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                        # mask iou branch
                        if self.with_mask_iou:
                            mask_scores.append(
                                [[] for _ in range(mask_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_mask = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages_test,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_mask, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)

                        # mask iou branch
                        if self.with_mask_iou:
                            mask_iou_pred = self.mask_iou_head(
                                mask_feats,
                                mask_pred[i][
                                    range(det_labels[i].size(0)),
                                    det_labels[i]])
                            # get mask scores with mask iou head
                            mask_scores.append(
                                self.mask_iou_head.get_mask_scores(
                                    mask_iou_pred, det_bboxes[i], det_labels[i]
                                ))

            # update ensemble in case of mask iou scores
            if len(mask_scores) > 0:
                ms_segm_result['ensemble'] = list(
                    zip(segm_results, mask_scores))
            else:
                ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            #Uncomment this section only for testing the correctness
            #of the dis feature. Keep it commented during train
            #if self.bbox_head[-1].with_dis:
            #    results = list(
            #        zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble'], ms_dis_result['ensemble']))
            #else:
            #    results = list(
            #        zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
            results = list(
                    zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
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
                idx = self.stages[i]

                bbox_head = self.bbox_head[idx]
                bbox_results = self._bbox_forward(
                    idx, x, rois, semantic_feat=semantic)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = bbox_head.regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
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
            # init mask iou score
            mask_scores = None

            if det_bboxes.shape[0] == 0:
                segm_result = [[
                    [] for _ in range(self.mask_head[-1].num_classes)]]

                # mask iou branch
                mask_classes = self.mask_head[-1].num_classes
                if self.with_mask_iou:
                    mask_scores = [[[] for _ in range(mask_classes)]]
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
                segm_result = (segm_result, mask_scores)

            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]
