_base_ = 'r3_cnn-exp_8-row_3_4_rect.py'
# dcn
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    )
)
