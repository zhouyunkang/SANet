_base_ = ['../../_base_/default_runtime.py']
#export CUBLAS_WORKSPACE_CONFIG=":4096:8"
#export PYTHONPATH=/home/zyk/code/mmaction2-main/mmaction2-main:$PYTHONPATH
# CUDA_VISIBLE_DEVICES='0,1' bash tools/dist_train.sh configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics400-rgb.py 2 --work-dir /nfs/zyk/UniformerV2_result/EK-6/work_dir83 --seed 220 --deterministic
num_frames = 16
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='UniFormerV2',
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        t_size=num_frames,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.,
        temporal_downsample=False,
        no_lmhra=True,
        double_lmhra=True,
        return_list=[8, 9, 10, 11],
        n_layers=4,
        n_dim=768,
        n_head=12,
        mlp_factor=4.,
        drop_path_rate=0.,
        frozen_stages = 12,

        mlp_dropout=[0.5, 0.5, 0.5, 0.5],
        clip_pretrained=True,
        pretrained='ViT-B/16'),#'ViT-B/16'
    cls_head=dict(
        type='UniFormerHead',
        dropout_ratio=0.5,
        num_classes=6,
        in_channels=768,
        average_clips='prob'
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'VideoDataset'
data_root = ''
data_root_val = ''
ann_file_train = '/home/zyk/Dataset/Ekman6/shuff/train.csv'
ann_file_val = '/home/zyk/Dataset/Ekman6/shuff/val.csv'
ann_file_test = '/home/zyk/Dataset/Ekman6/shuff/test.csv'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=1),
    dict(type='DecordDecode'),

    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='TenCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,

        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')#AccMetric
test_evaluator = dict(type='AccMetric')
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
base_lr = 2e-4
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=20, norm_type=2))
model_wrapper_cfg = dict(type='MMDistributedDataParallel',find_unused_parameters=True)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.8,
        by_epoch=True,
        begin=0,
        end=200,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=100,
        eta_min_ratio=0.8,
        by_epoch=True,
        begin=5,
        end=200,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=40), logger=dict(interval=100))
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=256)
