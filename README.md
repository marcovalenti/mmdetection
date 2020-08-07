# GRoIE

See [GRoIE](./configs/groie) configurations and models.

To quick test them:

```
git clone https://github.com/IMPLabUniPr/mmdetection-groie.git
cd mmdetection-groie
# download inside ./data/coco the COCO 2017 minival (5k images)
pip install pytorch
pip install -e .
python tools/test.py configs/groie/mask_rcnn_r4_gcb_c3-c5_r101_fpn_syncbn_1x-groie.py checkpoints/mask_rcnn_r4_gcb_c3-c5_r101_fpn_syncbn_1x-groie.pth --json_out checkpoints/mask_rcnn_r4_gcb_c3-c5_r101_fpn_syncbn_1x-groie-test
python tools/coco_eval.py checkpoints/mask_rcnn_r4_gcb_c3-c5_r101_fpn_syncbn_1x-groie-test.bbox.json --types bbox --ann data/coco/annotations/instances_val2017.json
python tools/coco_eval.py checkpoints/mask_rcnn_r4_gcb_c3-c5_r101_fpn_syncbn_1x-groie-test.segm.json --types segm --ann data/coco/annotations/instances_val2017.json
```

**Note**: For latest version and/or pre-trained models, you can find them on
[mmdetection master branch](https://github.com/open-mmlab/mmdetection/tree/master/configs/groie).


## Citation

If you use this work or benchmark in your research, please cite this project.

```
@misc{rossi2020novel,
    title={A novel Region of Interest Extraction Layer for Instance Segmentation},
    author={Leonardo Rossi and Akbar Karimi and Andrea Prati},
    year={2020},
    eprint={2004.13665},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Contact

The implementation of GROI is currently maintained by
[Leonardo Rossi](https://github.com/hachreak/).
