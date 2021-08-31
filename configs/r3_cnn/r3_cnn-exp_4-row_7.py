_base_ = './r3_cnn-exp_4-row_6.py'
# model settings
model = dict(
    roi_head=dict(
        stages=(0, 1, 0, 1),
    )
)
