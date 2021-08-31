_base_ = 'r3_cnn-exp_8-row_3_rect.py'
# model settings
model = dict(
    roi_head=dict(
        mask_roi_extractor=dict(
            type='GenericRoIExtractor',
            aggregation='sum',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            pre_cfg=dict(
                type='ConvModule',
                in_channels=256,
                out_channels=256,
                kernel_size=7,
                padding=3,
                inplace=False,
            ),
            post_cfg=dict(
                type='NonLocal2d',
                in_channels=256,
                reduction=2,
                use_scale=True,
                kernel_size=7,
                padding=3
            ),
        )))
