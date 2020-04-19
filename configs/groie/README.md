# GRoIE

## A novel Region of Interest Extraction Layer for Instance Segmentation

By Leonardo Rossi, Akbar Karimi and Andrea Prati from
[IMPLab](http://implab.ce.unipr.it/).

We provide config files to reproduce the results in the paper for
"*A novel Region of Interest Extraction Layer for Instance Segmentation*"
on COCO object detection.

## Introduction

This paper is motivated by the need to overcome to the limitations of existing
RoI extractors which select only one (the best) layer from FPN.

Our intuition is that all the layers of FPN retain useful information.

Therefore, the proposed layer (called Generic RoI Extractor - **GRoIE**)
introduces non-local building blocks and attention mechanisms to boost the
performance.

## Results and models
The results on COCO 2017v minival (5k images) are shown in the below table.

### Module-wise ablation analysis

| Backbone  | Model  | Module          | Type      | Lr schd | box AP | mask AP | Config file                               |
| :-------: | :----: | :-------------: | :-------: | :-----: | :----: | :-----: | :---------------------------------------: |
| R-50-FPN  | Faster |                 | baseline  |   1x    |  36.5  |         | [model](./faster_rcnn_r50_fpn_1x-orig.py) |
| R-50-FPN  | Faster |                 | random    |   1x    |  34.8  |         | [model](./faster_rcnn_r50_fpn_1x-v1.py)   |
| R-50-FPN  | Faster | Aggregation     | sum       |   1x    |  36.8  |         | [model](./faster_rcnn_r50_fpn_1x-v2.py)   |
| R-50-FPN  | Faster | Aggregation     | sum+      |   1x    |  36.0  |         | [model](./faster_rcnn_r50_fpn_1x-v3.py)   |
| R-50-FPN  | Faster | Aggregation     | concat    |   1x    |  36.1  |         | [model](./faster_rcnn_r50_fpn_1x-v4.py)   |
| R-50-FPN  | Faster | Pre-processing  | conv 1x1  |   1x    |  36.2  |         | [model](./faster_rcnn_r50_fpn_1x-v5.py)   |
| R-50-FPN  | Faster | Pre-processing  | conv 3x3  |   1x    |  37.0  |         | [model](./faster_rcnn_r50_fpn_1x-v6.py)   |
| R-50-FPN  | Faster | Pre-processing  | conv 5x5  |   1x    |  37.2  |         | [model](./faster_rcnn_r50_fpn_1x-v7.py)   |
| R-50-FPN  | Faster | Pre-processing  | Non-local |   1x    |  36.5  |         | [model](./faster_rcnn_r50_fpn_1x-v8.py)   |
| R-50-FPN  | Faster | Pre-processing  | Attention |   1x    |  36.4  |         | [model](./faster_rcnn_r50_fpn_1x-v9.py)   |
| R-50-FPN  | Faster | Post-processing | conv 1x1  |   1x    |  36.0  |         | [model](./faster_rcnn_r50_fpn_1x-v10.py)  |
| R-50-FPN  | Faster | Post-processing | conv 3x3  |   1x    |  36.6  |         | [model](./faster_rcnn_r50_fpn_1x-v11.py)  |
| R-50-FPN  | Faster | Post-processing | conv 5x5  |   1x    |  36.6  |         | [model](./faster_rcnn_r50_fpn_1x-v12.py)  |
| R-50-FPN  | Faster | Post-processing | Non-local |   1x    |  36.7  |         | [model](./faster_rcnn_r50_fpn_1x-v13.py)  |
| R-50-FPN  | Faster | Post-processing | Attention |   1x    |  36.8  |         | [model](./faster_rcnn_r50_fpn_1x-v14.py)  |

### Application of GRoIE to different architectures

| Backbone  | Model            | Lr schd | box AP | mask AP | Config file                                                   |
| :-------: | :--------------: | :-----: | :----: | :-----: | :-----------------------------------------------------------: |
| R-50-FPN  | Faster Original  |   1x    |  36.5  |         | [model](./faster_rcnn_r50_fpn_1x-orig.py)                     |
| R-50-FPN  | + GRoIE          |   1x    |  37.5  |         | [model](./faster_rcnn_r50_fpn_1x-groie.py)                    |
| R-50-FPN  | Grid R-CNN       |   1x    |  39.1  |         | [model](./grid_rcnn_gn_head_r50_fpn_1x-orig.py)               |
| R-50-FPN  | + GRoIE          |   1x    |  39.8  |         | [model](./grid_rcnn_gn_head_r50_fpn_1x-groie.py)              |
| R-50-FPN  | Mask R-CNN       |   1x    |  37.3  |  34.1   | [model](./mask_rcnn_r50_fpn_1x-orig.py)                       |
| R-50-FPN  | + GRoIE          |   1x    |  38.4  |  35.8   | [model](./mask_rcnn_r50_fpn_1x-groie.py)                      |
| R-50-FPN  | GC-Net           |   1x    |  39.5  |  35.9   | [model](./mask_rcnn_r4_gcb_c3-c5_r50_fpn_syncbn_1x-orig.py)   |
| R-50-FPN  | + GRoIE          |   1x    |  40.3  |  37.2   | [model](./mask_rcnn_r4_gcb_c3-c5_r50_fpn_syncbn_1x-groie.py)  |
| R-101-FPN | GC-Net           |   1x    |  41.4  |  37.4   | [model](./mask_rcnn_r4_gcb_c3-c5_r101_fpn_syncbn_1x-orig.py)  |
| R-101-FPN | + GRoIE          |   1x    |  42.2  |  38.5   | [model](./mask_rcnn_r4_gcb_c3-c5_r101_fpn_syncbn_1x-groie.py) |
