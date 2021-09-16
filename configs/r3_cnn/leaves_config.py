# _base_ = [
#     '../_base_/models/mask_rcnn_r50_fpn.py',
#     '../_base_/datasets/leaves_instance.py',
#     '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
# ]

# model = dict( 
#     roi_head=dict(
#         bbox_head=dict(
#             # num_classes=16
#             num_classes=2
#         ),
#         mask_head=dict(
#             # num_classes=2
#             num_classes=2
#         )
#     )
# )
_base_ = [
    '../htc/htc_without_semantic_r50_fpn_1x_coco_leaves.py'
]

model = dict(
    roi_head=dict(
        type='R3RoIHead',
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.75, 0.5],
        stages=[0, 0, 0],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=16,
                #num_classes=11,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=[
            dict(
                type='R3MaskHead',
                with_conv_res=True,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=16,
                #num_classes=11,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ]))
