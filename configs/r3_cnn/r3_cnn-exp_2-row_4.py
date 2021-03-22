_base_ = './r3_cnn-exp_1-row_4.py'
# model settings
model = dict(
    roi_head=dict(
        num_stages_test=3))
dist_params = dict(backend='nccl', port=20423)
