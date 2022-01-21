_base_ = './leaves_config.py' 
# GC-net
model = dict(
    backbone=dict(
        #norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        plugins=[
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 4),
                stages=(False, True, True, True),
                position='after_conv3')
        ]
    )
)
