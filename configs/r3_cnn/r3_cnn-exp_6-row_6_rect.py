_base_ = './r3_cnn-exp_6-row_4_rect.py'
# model settings
model = dict(
    roi_head=dict(
        mask_iou_head=dict(
            nl_stages=(True, True),
            nl_cfg=dict(
                type='NonLocal2d',
                reduction=2,
                use_scale=True,
                kernel_size=7,
                padding=3,
            ),
        )
    ),
)
