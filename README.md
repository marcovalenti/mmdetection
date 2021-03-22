# Recursively Refined R-CNN: Self RoI Re-Balancing on Instance Segmentation

## Installation

See [MMDetection](https://mmdetection.readthedocs.io/) and
[MMCV](https://github.com/open-mmlab/mmcv) official pages to know the
installation procedure for mmdetection and mmcv libraries.

Our code has been designed with the following version of the libraries:

- MMDetection - commit [eb66e0d0](https://github.com/open-mmlab/mmdetection/commit/eb66e0d0)
- MMCV - commit [1830347](https://github.com/open-mmlab/mmcv/commit/1830347)

**Note**: In order to run R<sup>3</sup>-CNN training, it is required to install our
[mmcv library](https://github.com/IMPLabUniPr/mmcv/tree/r3_cnn) version
(not the official one!).


## Models

Most important files.

| Name      | Description |
| :-------: | :---------: |
| [FCC BBox Head](mmdet/models/roi_heads/bbox_heads/fcc_bbox_head.py) | It contains our brand new lightweight R<sup>3</sup>-CNN-L (advanced) bbox head. |
| [Mask Head](mmdet/models/roi_heads/mask_heads/r3_cnn_mask_head.py) | It contains R<sup>3</sup>-CNN mask head. |
| [FCC Mask IoU Head](mmdet/models/roi_heads/mask_heads/fcc_maskiou_head.py) | It contains our brand new Mask IoU segmentation head for R<sup>3</sup>-CNN-L (advanced) model. |
| [R<sup>3</sup>-CNN RoI Head](mmdet/models/roi_heads/r3_cnn_roi_head.py) | It contains R<sup>3</sup>-CNN RoI Head whose task is to manage bbox, segmentation and Mask IoU heads. |


## Experiments

The following configuration files have been used to run each experiment:


#### Table 1

| Row | Model  |
| :-: | :----: |
| 1   | [Mask (1x)](configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py) |
| 2   | [Mask (3x)](configs/mask_rcnn/mask_rcnn_r50_fpn_3x_coco.py) |
| 3   | [HTC](configs/htc/htc_without_semantic_r50_fpn_1x_coco.py)  |
| :--: | :----: |
| 4   | [R<sup>3</sup>-CNN (naive)](configs/r3_cnn/r3_cnn-exp_1-row_4.py) |
| 5   | [R<sup>3</sup>-CNN (deeper)](configs/r3_cnn/r3_cnn-exp_1-row_5.py) |
| 6   | [R<sup>3</sup>-CNN (deeper)](configs/r3_cnn/r3_cnn-exp_1-row_6.py) |


#### Table 2

| Row | Model  | Lt  | H   | Le  |
| :-: | :----: | :-: | :-: | :-: |
| 1   | [Mask)](configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py) | 1   | 1   | 1   |
| :--: | :----: | :--: | :--: | :--: |
| 2   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_2-row_2.py) | 3   | 1   | 1  |
| 3   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_2-row_3.py) | 3   | 1   | 2  |
| 4   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_2-row_4.py) | 3   | 1   | 3  |
| 5   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_2-row_5.py) | 3   | 1   | 4  |
| 6   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_2-row_6.py) | 3   | 1   | 5  |


#### Table 3

| Row | Model  | Lt  | H   |
| :--: | :----: | :--: | :--: |
| 1   | [Mask)](configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py) | 1   | 1   |
| :--: | :----: | :--: | :--: |
| 2   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_3-row_2.py) | 1   | 1   |
| 3   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_3-row_3.py) | 2   | 1   |
| 4   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_3-row_4.py) | 3   | 1   |
| 5   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_3-row_5.py) | 4   | 1   |


#### Table 4

| Row | Model             | Backbone  |
| :--: | :---------------: | :-------: |
| 1   | [Mask](configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py)                         | r50-FPN   |
| 2   | [HTC](configs/htc/htc_without_semantic_r50_fpn_1x_coco.py)                     | r50-FPN   |
| 3   | [R<sup>3</sup>-CNN-L](configs/r3_cnn/r3_cnn-exp_4-row_3.py)                    | r50-FPN   |
| :--: | :----: | :------: | r50-FPN   |
| 4   | [GRoIE](configs/groie/mask_rcnn_r50_fpn_groie_1x_coco.py)                      | r50-FPN   |
| 5   | [R<sup>3</sup>-CNN-L + GRoIE](configs/r3_cnn/r3_cnn-exp_4-row_5.py)            | r50-FPN   |
| :--: | :----: | :------: | r50-FPN   |
| 6   | [GC-Net](configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py)              | r50-FPN   |
| 7   | [HTC + GC-Net](configs/gcnet/htc_without_semantic_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py)      | r50-FPN   |
| 8   | [R<sup>3</sup>-CNN-L + GC-Net](configs/r3_cnn/r3_cnn-exp_4-row_8.py)           | r50-FPN   |
| :--: | :----: | :------: | r50-FPN   |
| 9   | [DCN](configs/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py)                    | r50-FPN   |
| 10  | [HTC + DCN](configs/dcn/htc_without_semantic_r50_fpn_dconv_c3-c5_1x_coco.py)   | r50-FPN   |
| 11  | [R<sup>3</sup>-CNN-L + DCN](configs/r3_cnn/r3_cnn-exp_4-row_11.py)             | r50-FPN   |
