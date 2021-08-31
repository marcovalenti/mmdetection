_base_ = './r3_cnn-exp_5-row_3-fcc2.py'
model = dict(
    roi_head=dict(
        with_mask_loop=False,
        bbox_head=[
            dict(
                type='Shared2FCC2BBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                nl_stages=(True, False, 0),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0),
                convwise=(
                    {'kernel_size': (7, 3), 'padding': 1},
                    {'kernel_size': (3, 7), 'padding': 3}
                ),
                post_reduce=False,
                nl_cfg=dict(
                    type='NonLocal2d',
                    reduction=2,
                    use_scale=True,
                    kernel_size=7,
                    padding=3,
                ),
            )
        ],
    )
)
