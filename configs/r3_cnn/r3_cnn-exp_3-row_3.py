_base_ = './r3_cnn-exp_1-row_4.py'
# model settings
model = dict(
    roi_head=dict(
        num_stages=2,
        stage_loss_weights=[1, 0.75]))
