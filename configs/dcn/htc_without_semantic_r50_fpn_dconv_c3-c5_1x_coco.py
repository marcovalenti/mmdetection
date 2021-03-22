_base_ = '../htc/htc_without_semantic_r50_fpn_1x_coco.py'
# dcn
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    )
)
