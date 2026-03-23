_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'

classes = (
    'missing_hole',
    'mouse_bite',
    'open_circuit',
    'short',
    'spur',
    'spurious_copper',
)
num_classes = len(classes)
metainfo = dict(classes=classes)

data_root = 'data/'

model = dict(
    bbox_head=dict(num_classes=num_classes),
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations_json/train.json',
        data_prefix=dict(img='train/images/'),
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations_json/val.json',
        data_prefix=dict(img='val/images/'),
    ),
)

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations_json/test.json',
        data_prefix=dict(img='test/images/'),
    ),
)

val_evaluator = dict(ann_file=data_root + 'annotations_json/val.json')
test_evaluator = dict(ann_file=data_root + 'annotations_json/test.json')
