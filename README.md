# Recursively Refined R-CNN: Self RoI Re-Balancing on Instance Segmentation

## Installation

See [MMDetection](https://mmdetection.readthedocs.io/) and
[MMCV](https://github.com/open-mmlab/mmcv) official pages to know the
installation procedure for mmdetection and mmcv libraries.

Our code has been designed with the following version of the libraries:

- MMDetection - commit [eb66e0d0](https://github.com/open-mmlab/mmdetection/commit/f07de13b)
- MMCV - commit [1830347](https://github.com/open-mmlab/mmcv/commit/1830347)

**Note**: In order to run R<sup>3</sup>-CNN training, it is required to install our
[mmcv library](https://github.com/IMPLabUniPr/mmcv/tree/r3_cnn) version
(not the official one!).


## Models

Most important files.

| Name      | Description |
| :-------: | :---------: |
| [FCC BBox Head](mmdet/models/roi_heads/bbox_heads/fcc_bbox_head.py) | It contains our brand new R<sup>3</sup>-CNN bbox head with FCC Lighter v1 and v2, corresponding to Fig. 4(b, c). |
| [FCC BBox Head](mmdet/models/roi_heads/bbox_heads/fcc2_bbox_head.py) | It contains our brand new R<sup>3</sup>-CNN bbox head with FCC Advanced v1 and v2, corresponding to Fig. 4(d, e). |
| [Mask Head](mmdet/models/roi_heads/mask_heads/r3_cnn_mask_head.py) | It contains R<sup>3</sup>-CNN mask head. |
| [FCC Mask IoU Head](mmdet/models/roi_heads/mask_heads/fcc_maskiou_head.py) | It contains our brand new Mask IoU segmentation head for R<sup>3</sup>-CNN model with FCC Lighter v1 and v2, corresponding to Fig. 4(b, c). |
| [FCC Mask IoU Head](mmdet/models/roi_heads/mask_heads/fcc_maskiou_head.py) | It contains our brand new Mask IoU segmentation head for R<sup>3</sup>-CNN model with FCC Advanced v1 and v2, corresponding to Fig. 4(d, e). |
| [R<sup>3</sup>-CNN RoI Head](mmdet/models/roi_heads/r3_roi_head.py) | It contains R<sup>3</sup>-CNN RoI Head whose task is to manage bbox, segmentation and Mask IoU heads. |


## Experiments

The following configuration files have been used to run each experiment:


#### Table 2

| Row | Model  |
| :-: | :----: |
| 1   | [Mask (1x)](configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py) |
| 2   | [Mask (3x)](configs/mask_rcnn/mask_rcnn_r50_fpn_3x_coco.py) |
| 3   | [HTC](configs/htc/htc_without_semantic_r50_fpn_1x_coco.py)  |
| :--: | :----: |
| 4   | [R<sup>3</sup>-CNN (naive)](configs/r3_cnn/r3_cnn-exp_1-row_4.py) |
| 5   | [R<sup>3</sup>-CNN (deeper)](configs/r3_cnn/r3_cnn-exp_1-row_5.py) |


#### Table 3

| Row | Model  | Lt  | H   | Alt. |
| :--: | :----: | :--: | :--: |
| 1  | [HTC](configs/htc/htc_without_semantic_r50_fpn_1x_coco.py) | 3   | 3   | abc   |
| :--: | :----: | :--: | :--: |
| 2   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_4-row_2.py) | 2   | 2   | ab    |
| 3   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_1-row_4.py) | 3   | 2   | abb   |
| 4   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_4-row_4.py) | 3   | 2   | aab   |
| 5   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_4-row_5.py) | 3   | 2   | aba   |
| 6   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_4-row_6.py) | 4   | 2   | aabb  |
| 7   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_4-row_7.py) | 4   | 2   | abab  |
| 8   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_4-row_8.py) | 5   | 2   | aabbb |


#### Table 4

| Row | Model  | Lt  | H   | L2C | NL<sub>b</sub> | NL<sub>a</sub>
| :--: | :----: | :--: | :--: | :----: | :--: | :--: |
| 1  | [HTC](configs/htc/htc_without_semantic_r50_fpn_1x_coco.py)            | 3   | 3   |            |   |   |
| :--: | :----: | :--: | :--: | :----: | :--: | :--: |
| 2   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_1-row_4.py)            | 3   | 1   |            |   |   |
| 3   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_5-row_3.py)            | 3   | 1   | 7x7        |   |   |
| 4   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_5-row_3-fcc2_2.py)     | 3   | 1   | 7x3 -> 3x7 |   |   |
| 5   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_5-row_4.py)            | 3   | 1   | 7x7        | v |   |
| 6   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_5-row_3-fcc2_8.py)     | 3   | 1   | 7x3 -> 3x7 | v |   |
| 7   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_5-row_5.py)            | 3   | 1   | 7x7        | v | v |
| :--: | :----: | :--: | :--: | :----: | :--: | :--: |
| 8   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_5-row_6.py)            | 4   | 2   |            |   |   |
| 9   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_5-row_7.py)            | 4   | 2   | 7x7        |   |   |
| 10  | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_5-row_6-fcc2_2.py)     | 4   | 2   | 7x3 -> 3x7 |   |   |
| 11  | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_5-row_8.py)            | 4   | 2   | 7x7        | v |   |
| 12  | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_5-row_9.py)            | 4   | 2   | 7x3 -> 3x7 | v |   |


#### Table 5

| Row | Model  | Lt  | H   | M<sub>IoU</sub> | L2C | NL<sub>b</sub> | NL<sub>a</sub>
| :--: | :----: | :--: | :--: | :----: | :--: | :--: |
| 1  | [HTC](configs/htc/htc_without_semantic_r50_fpn_1x_coco.py)            | 3   | 3   |   |            |   |   |
| :--: | :----: | :--: | :--: | :----: | :--: | :--: | :--: |
| 2   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_6-row_2_rect.py)       | 3   | 1   |   |            |   |   |
| 3   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_6-row_3_rect.py)       | 3   | 1   | v |            |   |   |
| 4   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_6-row_4_rect.py)       | 3   | 1   | v | 7x7        |   |   |
| 5   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_6-row_4_2_rect.py)     | 3   | 1   | v | 7x3 -> 3x7 |   |   |
| 6   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_6-row_5_rect.py)       | 3   | 1   | v | 7x7        | v |   |
| 7   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_6-row_6_rect.py)       | 3   | 1   | v | 7x7        | v | v |
| :--: | :----: | :--: | :--: | :----: | :--: | :--: | :--: |
| 8   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_6-row_7_rect.py)       | 4   | 2   |   |            |   |   |
| 9   | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_6-row_8.py)            | 4   | 2   | v |            |   |   |
| 10  | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_6-row_9.py)            | 4   | 2   | v | 7x7        |   |   |
| 11  | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_6-row_10.py)           | 4   | 2   | v | 7x7        | v |   |
| 12  | [R<sup>3</sup>-CNN](configs/r3_cnn/r3_cnn-exp_6-row_11.py)           | 4   | 2   | v | 7x7        | v | v |


#### Table 6

| Model  |
| :----: |
| [baseline](configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) |
| [conv 3x3](configs/groie/faster_rcnn_r50_fpn_groie_1x_coco-pre-v1.py) |
| [conv 5x5](configs/groie/faster_rcnn_r50_fpn_groie_1x_coco-pre-v2.py) |
| [conv 7x7](configs/groie/faster_rcnn_r50_fpn_groie_1x_coco-pre-v3.py) |
| [conv 7x3 -> 3x7](configs/groie/faster_rcnn_r50_fpn_groie_1x_coco-v15.py) |
| [Non-local 1x1](configs/groie/faster_rcnn_r50_fpn_groie_1x_coco-pre-v4.py) |
| [Non-local 3x3](configs/groie/faster_rcnn_r50_fpn_groie_1x_coco-pre-v5.py) |


#### Table 7

| Model  |
| :----: |
| [baseline](configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) |
| [conv 3x3](configs/groie/faster_rcnn_r50_fpn_groie_1x_coco-post-v1.py) |
| [conv 5x5](configs/groie/faster_rcnn_r50_fpn_groie_1x_coco-post-v2.py) |
| [conv 7x7](configs/groie/faster_rcnn_r50_fpn_groie_1x_coco-post-v3.py) |
| [conv 7x3 -> 3x7](configs/groie/faster_rcnn_r50_fpn_groie_1x_coco-post-v4.py) |
| [Non-local 1x1](configs/groie/faster_rcnn_r50_fpn_groie_1x_coco-post-v5.py) |
| [Non-local 3x3](configs/groie/faster_rcnn_r50_fpn_groie_1x_coco-post-v6.py) |


#### Table 8

| Model  |
| :----: |
| [baseline](configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) |
| [GRoIE](configs/groie/faster_rcnn_r50_fpn_groie_1x_coco-all-v1.py) |


#### Table 9

| Row | Model             | Backbone  |
| :--: | :---------------: | :-------: |
| 1   | [Mask](configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py)                         | r50-FPN   |
| 2   | [HTC](configs/htc/htc_without_semantic_r50_fpn_1x_coco.py)                     | r50-FPN   |
| 3   | [SBR-CNN](configs/r3_cnn/r3_cnn-exp_8-row_3_4_rect.py)                         | r50-FPN   |
| :--: | :----: | :------: | r50-FPN   |
| 6   | [GC-Net](configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py)              | r50-FPN   |
| 7   | [HTC + GC-Net](configs/gcnet/htc_without_semantic_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py)      | r50-FPN   |
| 8   | [SBR-CNN + GC-Net](configs/r3_cnn/r3_cnn-exp_8-row_7_rect.py)                  | r50-FPN   |
| :--: | :----: | :------: | r50-FPN   |
| 9   | [DCN](configs/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py)                    | r50-FPN   |
| 10  | [HTC + DCN](configs/dcn/htc_without_semantic_r50_fpn_dconv_c3-c5_1x_coco.py)   | r50-FPN   |
| 11  | [SBR-CNN + DCN](configs/r3_cnn/r3_cnn-exp_8-row_10_rect.py)                    | r50-FPN   |
