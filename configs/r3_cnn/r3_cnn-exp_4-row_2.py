_base_ = './r3_cnn-exp_1-row_5.py'
model = dict(
    roi_head=dict(
        num_stages=2,
        stages=[0, 1],
    )
)
