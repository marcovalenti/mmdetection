_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# model settings
model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='GenericRoIExtractor',
            aggregation='sum',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            post_cfg=dict(
                type='ConvModule',
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                inplace=False,
            ),
        )))
