_base_ = './r3_cnn-exp_1-row_5.py'
model = dict(
    roi_head=dict(
        stages=[0, 0, 1],
    )
)
