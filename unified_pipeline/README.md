# Unified PCB Dataset Pipeline

## 1) Build merged dataset (6 sources, 80/10/10)

```bash
python unified_pipeline/build_unified_dataset.py \
  --root D:/CS/DeepLearning/midterm/PCB_defect_detection \
  --output D:/CS/DeepLearning/midterm/PCB_defect_detection/dataset_unified_6sources \
  --seed 42
```

Output structure:

- `train/images/{mouse_bite,open_circuit,mising_hole,short,spur,spurious_copper}`
- `train/annotations/txt/{...}`
- `train/annotations/xml/{...}`
- same for `val`, `test`
- `annotations/json/annotations_all.json` (single global JSON for full dataset)
- `coco_annotations/{train,val,test}.json` (for MMDetection)
- `yolo_format/{train,val,test}/{images,labels}` + `yolo_format/data.yaml` (for YOLO)

## 2) Plot statistics

```bash
python unified_pipeline/plot_dataset_stats.py \
  --dataset D:/CS/DeepLearning/midterm/PCB_defect_detection/dataset_unified_6sources
```

Creates:

- `plots/bar_images_per_defect.png`
- `plots/scatter_bbox_wh.png`

### Plot before merge (per source dataset)

```bash
python unified_pipeline/plot_premerge_dataset_stats.py \
  --root D:/CS/DeepLearning/midterm/PCB_defect_detection \
  --output D:/CS/DeepLearning/midterm/PCB_defect_detection/dataset_premerge_stats
```

Creates (for each source dataset):

- `dataset_premerge_stats/<source>/class_image_count_bar.png`
- `dataset_premerge_stats/<source>/bbox_scatter_wh.png`
- `dataset_premerge_stats/<source>/image_size_top15_bar.png`
- `dataset_premerge_stats/premerge_summary.json`

Label mapping notes written to JSON include:

- DsPCBDS+ 9->6 map: `SH->short`, `SP->spur`, `SC->spurious_copper`, `OP->open_circuit`, `MB->mouse_bite`, `HB->mising_hole`
- Dropped in current merge pipeline: `CS`, `CFO`, `BMFO`

## 3) Train YOLO + evaluate metrics

YOLO augmentation preset:

- `unified_pipeline/configs/yolo_augmentation.yaml`

```bash
pip install ultralytics
pip install pyyaml
python unified_pipeline/train_yolo.py \
  --dataset D:/CS/DeepLearning/midterm/PCB_defect_detection/dataset_unified_6sources \
  --model yolo11m.pt --epochs 100 --batch 16 --imgsz 640 \
  --aug-config unified_pipeline/configs/yolo_augmentation.yaml \
  --metrics-out reports/metrics_yolo.csv \
  --metrics-json reports/metrics_yolo.json
```

Metrics:

- recall, precision
- box loss, classify loss, dfl loss (from `results.csv`)
- mAP50, mAP50-95
- additional benchmark columns exported: accuracy, f1, model_size_mb, trainable_params, fps, inference_time_ms, train_time_sec, train_samples_per_sec, sec_per_epoch

## 4) Train MMDetection models

MMDetection augmentation preset fragment:

- `unified_pipeline/configs/mmdet_augmentation.py`

Supported models:

- retinanet
- faster_rcnn
- cascade_rcnn
- detr
- deformable_detr

```bash
python unified_pipeline/train_mmdet.py \
  --dataset D:/CS/DeepLearning/midterm/PCB_defect_detection/dataset_unified_6sources \
  --mmdet-root D:/path/to/mmdetection \
  --model all \
  --metrics-out reports/metrics_mmdet.csv \
  --metrics-dir reports/metrics_mmdet
```

Example direct MMDetection use with augmentation preset:

```bash
cd D:/path/to/mmdetection
python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
  --cfg-options \
    train_dataloader.dataset.data_root=D:/CS/DeepLearning/midterm/PCB_defect_detection/dataset_unified_6sources/yolo_format \
    train_dataloader.dataset.ann_file=D:/CS/DeepLearning/midterm/PCB_defect_detection/dataset_unified_6sources/coco_annotations/train.json \
    train_dataloader.dataset.data_prefix.img=train/images/ \
    val_dataloader.dataset.data_root=D:/CS/DeepLearning/midterm/PCB_defect_detection/dataset_unified_6sources/yolo_format \
    val_dataloader.dataset.ann_file=D:/CS/DeepLearning/midterm/PCB_defect_detection/dataset_unified_6sources/coco_annotations/val.json \
    val_dataloader.dataset.data_prefix.img=val/images/
```

Notes:

- MMDetection logs train losses and eval metrics (mAP/precision/recall) in each model work directory.
- Some config keys are architecture-specific. If your mmdet version differs, adjust `--cfg-options` in the command.
- For augmentation in MMDetection, copy `train_pipeline_pcb` from `unified_pipeline/configs/mmdet_augmentation.py` into your target model config, then point that config to `train.py`.

## 5) Train both branches in one command

```bash
python unified_pipeline/train_dual_branch.py \
  --dataset D:/CS/DeepLearning/midterm/PCB_defect_detection/dataset_unified_6sources \
  --mmdet-root D:/path/to/mmdetection \
  --mmdet-model all
```

Outputs:

- `reports/metrics_yolo.csv`
- `reports/metrics_yolo.json`
- `reports/metrics_mmdet.csv`
- `reports/metrics_mmdet/*.json`

## 6) Plot model performance dashboard

`plot_model_performance.py` now auto-merges YOLO + MMDetection CSV rows into one file (`metrics_all_models.csv`) before plotting.

```bash
python unified_pipeline/plot_model_performance.py \
  --yolo-metrics-csv D:/CS/DeepLearning/midterm/PCB_defect_detection/reports/metrics_yolo.csv \
  --mmdet-metrics-csv D:/CS/DeepLearning/midterm/PCB_defect_detection/reports/metrics_mmdet.csv \
  --metrics-csv D:/CS/DeepLearning/midterm/PCB_defect_detection/reports/metrics_all_models.csv \
  --yolo-results D:/CS/DeepLearning/midterm/PCB_defect_detection/runs/unified_yolo/exp/results.csv \
  --yolo-confusion D:/CS/DeepLearning/midterm/PCB_defect_detection/runs/unified_yolo/exp/confusion_matrix.png \
  --output D:/CS/DeepLearning/midterm/PCB_defect_detection/reports/performance_plots
```

Optional flags:

- `--disable-auto-merge`: read only `--metrics-csv` and skip merging.
- `--disable-dedupe`: keep duplicate rows when merging.

Creates:

- `bar_accuracy.png`
- `bar_f1.png`
- `bar_map50.png`
- `bar_map50_95.png`
- `scatter_precision_recall.png`
- `curve_yolo_losses.png` (if `--yolo-results` provided)
- `confusion_matrix_yolo.png` (if `--yolo-confusion` provided)
- `performance_plot_summary.json`
