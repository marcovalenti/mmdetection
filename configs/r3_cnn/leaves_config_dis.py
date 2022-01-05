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
                type='Shared2FCBBoxHeadLeaves',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                #num_classes=16,
                num_classes=10,
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
                               loss_weight=1.0),  
                #loss_dis=dict(type='CrossEntropyLoss',
                #              use_sigmoid=False,
                #              loss_weight=1.0,
                             #class_weight=[0.42, 0.14, 0.43, 0.01]),
                #              class_weight=[0.33, 0.29, 0.33, 0.05]),
                loss_dis=dict(type='FocalLoss',
                              use_sigmoid=True,
                              gamma=1.5,
                              alpha=0.96,
                              loss_weight=1.0),
                #loss_dis=dict(type='FocalLoss',
                #              use_sigmoid=False,
                #              gamma=1.0,
                #              alpha=1.0,
                #              loss_weight=1.0,
                              #class_weight=[0.42, 0.14, 0.43, 0.01]),
                #              class_weight=[0.33, 0.29, 0.33, 0.05]),
                #reference_labels= dict([('grappolo_vite', 7), 
                #             ('foglia_vite', 6),
                #             ('oidio_tralci', 11)]),
                #classes = ['accartocciamento_fogliare', 'black_rot_foglia', 'black_rot_grappolo' ,
		        #           'botrite_foglia', 'botrite_grappolo', 'carie_bianca_grappolo',
		        #           'foglia_vite', 'grappolo_vite', 'malattia_esca', 'oidio_foglia',
		        #           'oidio_grappolo', 'oidio_tralci', 'peronospora_foglia', 'peronospora_grappolo',
		        #           'red_blotch_foglia', 'virosi_pinot_grigio'],
                reference_labels= dict([('grappolo_vite', 5), 
                                        ('foglia_vite', 4)]),
                classes = ['black_rot_foglia', 'black_rot_grappolo' ,
		           'botrite_foglia', 'botrite_grappolo',
		           'foglia_vite', 'grappolo_vite', 'oidio_foglia',
		           'oidio_grappolo', 'peronospora_foglia', 'peronospora_grappolo'],
	        dis_selector= 1)	#0 consider only diseases, 1 only grapes and leaves, 2 for both cases
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
                #num_classes=16,
                num_classes=10,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ]))

valid_classes = ('black_rot_foglia', 'black_rot_grappolo' ,
		 'botrite_foglia', 'botrite_grappolo',
		 'foglia_vite', 'grappolo_vite', 'oidio_foglia',
		 'oidio_grappolo', 'peronospora_foglia', 'peronospora_grappolo')

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(classes=valid_classes),
    val=dict(classes=valid_classes),
    test=dict(classes=valid_classes))

optimizer = dict (type = 'SGD', lr = 0.0025 /2 , momentum = 0.9, weight_decay = 0.0001)
