_base_ = './r3_cnn-exp_4-row_3.py'
# dcn
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    )
)
