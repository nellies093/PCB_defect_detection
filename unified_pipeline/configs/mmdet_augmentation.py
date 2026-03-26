"""
MMDetection augmentation config fragment for PCB defect detection.

Usage example:
python tools/train.py <base_config.py> \
  --cfg-options custom_imports.imports=['unified_pipeline.configs.mmdet_augmentation'] \
                custom_imports.allow_failed_imports=False \
                train_dataloader.dataset.pipeline='${train_pipeline_pcb}'

If your shell cannot parse the pipeline string, copy the train_pipeline_pcb list
into your model config directly.
"""

# Recommend image scales for small PCB defects.
SCALES_PCB = [(640, 640), (768, 768), (896, 896), (1024, 1024)]

# Common train pipeline for RetinaNet / Faster R-CNN / Cascade R-CNN.
train_pipeline_pcb = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomChoiceResize', scales=SCALES_PCB, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        max_rotate_degree=8.0,
        max_translate_ratio=0.06,
        scaling_ratio_range=(0.9, 1.2),
        max_shear_degree=1.5,
        border=(0, 0),
        border_val=(114, 114, 114),
    ),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=20,
        contrast_range=(0.85, 1.15),
        saturation_range=(0.8, 1.2),
        hue_delta=8,
    ),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(2, 2), keep_empty=False),
    dict(type='PackDetInputs'),
]

# DETR / Deformable DETR can reuse the same augmentation policy.
train_pipeline_detr_pcb = train_pipeline_pcb

# Validation/test pipeline.
test_pipeline_pcb = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'),
]
