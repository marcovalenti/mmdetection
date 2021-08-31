_base_ = './r3_cnn-exp_5-row_3-fcc2_8.py'
model = dict(
    roi_head=dict(
        with_mask_loop=True,
    )
)
