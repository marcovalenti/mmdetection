_base_ = './r3_cnn-exp_6-row_10.py'
# model settings
model = dict(
    roi_head=dict(
        mask_iou_head=dict(
            nl_stages=(True, True),
        )
    )
)
