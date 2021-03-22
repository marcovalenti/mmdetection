_base_ = './r3_cnn-exp_1-row_4.py'
# model settings
model = dict(
    roi_head=dict(
        stages=[0, 0, 0, 0],
        num_stages_test=4))
dist_params = dict(backend='nccl', port=20424)
