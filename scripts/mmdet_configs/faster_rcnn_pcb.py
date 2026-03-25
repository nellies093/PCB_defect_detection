_base_ = "mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"

dataset_type = "CocoDataset"
data_root = "data/"
metainfo = {
    "classes": (
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper",
    )
}
num_classes = 6

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations_json/train.json",
        data_prefix=dict(img="train/images/"),
        metainfo=metainfo,
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations_json/val.json",
        data_prefix=dict(img="val/images/"),
        metainfo=metainfo,
    ),
)
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations_json/test.json",
        data_prefix=dict(img="test/images/"),
        metainfo=metainfo,
    ),
)

model = dict(roi_head=dict(bbox_head=dict(num_classes=num_classes)))
val_evaluator = dict(ann_file=data_root + "annotations_json/val.json")
test_evaluator = dict(ann_file=data_root + "annotations_json/test.json")

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=24, val_interval=1)
default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=2, max_keep_ckpts=3))

optim_wrapper = dict(optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.0001))
