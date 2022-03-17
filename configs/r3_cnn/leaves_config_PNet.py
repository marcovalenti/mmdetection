_base_ = './leaves_config.py' 

model = dict(
    #PathNet
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5))
