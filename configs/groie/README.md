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
You can find
[here](https://drive.google.com/drive/folders/19ssstbq_h0Z1cgxHmJYFO8s1arf3QJbT)
the trained models.

### Module-wise ablation analysis

| Backbone  | Model  | Module          | Type      | Lr schd | box AP | Model                                                                       | Config file                                     |
| :-------: | :----: | :-------------: | :-------: | :-----: | :----: | :-------------------------------------------------------------------------: | :---------------------------------------------: |
| R-50-FPN  | Faster |                 | baseline  |   1x    |  36.5  | [model](https://drive.google.com/open?id=1HEt58kSrTrvgSKT_wObii1kg4NTTZLN_) | [config file](./faster_rcnn_r50_fpn_1x-orig.py) |
| R-50-FPN  | Faster |                 | random    |   1x    |  34.8  | [model](https://drive.google.com/open?id=1LepNXzIu7BMjKInOtvJMwpuZGApLz3Bn) | [config file](./faster_rcnn_r50_fpn_1x-v1.py)   |
| R-50-FPN  | Faster | Aggregation     | sum       |   1x    |  36.8  |                                                                             | [config file](./faster_rcnn_r50_fpn_1x-v2.py)   |
| R-50-FPN  | Faster | Aggregation     | sum+      |   1x    |  36.0  | [model](https://drive.google.com/open?id=1RCQriewvFHvvtUJlKuGURUYECSiFOqL0) | [config file](./faster_rcnn_r50_fpn_1x-v3.py)   |
| R-50-FPN  | Faster | Aggregation     | concat    |   1x    |  36.1  | [model](https://drive.google.com/open?id=1icrvxPIYQgxTnSSEKAXFYvWcTc48ZSKo) | [config file](./faster_rcnn_r50_fpn_1x-v4.py)   |
| R-50-FPN  | Faster | Pre-processing  | conv 1x1  |   1x    |  36.2  | [model](https://drive.google.com/open?id=1XLXNJoPHuZIQLVyzmmx_6boBs8Fq4PkT) | [config file](./faster_rcnn_r50_fpn_1x-v5.py)   |
| R-50-FPN  | Faster | Pre-processing  | conv 3x3  |   1x    |  37.0  | [model](https://drive.google.com/open?id=1ZPzXkHD6bRjB4PHnbhUef42qGM9PYN_2) | [config file](./faster_rcnn_r50_fpn_1x-v6.py)   |
| R-50-FPN  | Faster | Pre-processing  | conv 5x5  |   1x    |  37.2  | [model](https://drive.google.com/open?id=1BvJSOZRD53pC6EuBTGlP02FWqI4XuZAQ) | [config file](./faster_rcnn_r50_fpn_1x-v7.py)   |
| R-50-FPN  | Faster | Pre-processing  | Non-local |   1x    |  36.5  | [model](https://drive.google.com/open?id=10xGIodxVSYLZAN1tsbaMcxnJuC2D5Ofp) | [config file](./faster_rcnn_r50_fpn_1x-v8.py)   |
| R-50-FPN  | Faster | Pre-processing  | Attention |   1x    |  36.4  | [model](https://drive.google.com/open?id=1gIvVU7xBzCnWPtzQZeZj5wTG7nCpBBP-) | [config file](./faster_rcnn_r50_fpn_1x-v9.py)   |
| R-50-FPN  | Faster | Post-processing | conv 1x1  |   1x    |  36.0  | [model](https://drive.google.com/open?id=1fys66CMdgcL5hdDJS-Z4UALj4s-X0CMM) | [config file](./faster_rcnn_r50_fpn_1x-v10.py)  |
| R-50-FPN  | Faster | Post-processing | conv 3x3  |   1x    |  36.6  | [model](https://drive.google.com/open?id=1Y55IZMlpCUejQ0nkT5j8qpIViLVTU8cj) | [config file](./faster_rcnn_r50_fpn_1x-v11.py)  |
| R-50-FPN  | Faster | Post-processing | conv 5x5  |   1x    |  36.6  | [model](https://drive.google.com/open?id=1acYY0pjwNG_dWkiEpXPrwmk2eWNFo489) | [config file](./faster_rcnn_r50_fpn_1x-v12.py)  |
| R-50-FPN  | Faster | Post-processing | Non-local |   1x    |  36.7  | [model](https://drive.google.com/open?id=1sU0BayjZdvznJSTl8UkDSwH8iMGszEq6) | [config file](./faster_rcnn_r50_fpn_1x-v13.py)  |
| R-50-FPN  | Faster | Post-processing | Attention |   1x    |  36.8  | [model](https://drive.google.com/open?id=1bV8K9Exp09wvOXapjCmEXuKicnaeUFPS) | [config file](./faster_rcnn_r50_fpn_1x-v14.py)  |

### Application of GRoIE to different architectures

| Backbone  | Model            | Lr schd | box AP | mask AP | Model                                                                       | Config file                                                         |
| :-------: | :--------------: | :-----: | :----: | :-----: | :-------------------------------------------------------------------------: | :-----------------------------------------------------------------: |
| R-50-FPN  | Faster Original  |   1x    |  36.5  |         | [model](https://drive.google.com/open?id=1HEt58kSrTrvgSKT_wObii1kg4NTTZLN_) | [config file](./faster_rcnn_r50_fpn_1x-orig.py)                     |
| R-50-FPN  | + GRoIE          |   1x    |  37.5  |         |                                                                             | [config file](./faster_rcnn_r50_fpn_1x-groie.py)                    |
| R-50-FPN  | Grid R-CNN       |   1x    |  39.1  |         | [model](https://drive.google.com/open?id=1JG7legc2_oomIg_1pGGPjB7JtpUcVmtz) | [config file](./grid_rcnn_gn_head_r50_fpn_1x-orig.py)               |
| R-50-FPN  | + GRoIE          |   1x    |  39.8  |         | [model](https://drive.google.com/open?id=1aMZwFx80pAPWIPw6hfh0dHrNEOauLE6i) | [config file](./grid_rcnn_gn_head_r50_fpn_1x-groie.py)              |
| R-50-FPN  | Mask R-CNN       |   1x    |  37.3  |  34.1   | [model](https://drive.google.com/open?id=1z5XgXOtZQpDcMKNE_87YHRQc-guCo7v7) | [config file](./mask_rcnn_r50_fpn_1x-orig.py)                       |
| R-50-FPN  | + GRoIE          |   1x    |  38.4  |  35.8   |                                                                             | [config file](./mask_rcnn_r50_fpn_1x-groie.py)                      |
| R-50-FPN  | GC-Net           |   1x    |  39.5  |  35.9   | [model](https://drive.google.com/open?id=1C3rgU9960ooflBBYv4O9bOMrmkWWWvsr) | [config file](./mask_rcnn_r4_gcb_c3-c5_r50_fpn_syncbn_1x-orig.py)   |
| R-50-FPN  | + GRoIE          |   1x    |  40.3  |  37.2   |                                                                             | [config file](./mask_rcnn_r4_gcb_c3-c5_r50_fpn_syncbn_1x-groie.py)  |
| R-101-FPN | GC-Net           |   1x    |  41.4  |  37.4   | [model](https://drive.google.com/open?id=16aL2Nrpnntkbo5R9-wnwTd3OQ3v_HQ0x) | [config file](./mask_rcnn_r4_gcb_c3-c5_r101_fpn_syncbn_1x-orig.py)  |
| R-101-FPN | + GRoIE          |   1x    |  42.2  |  38.5   | [model](https://drive.google.com/open?id=1XeFwFYjkZXWsMaLLZO6bC3jt31cdi4Kq) | [config file](./mask_rcnn_r4_gcb_c3-c5_r101_fpn_syncbn_1x-groie.py) |
