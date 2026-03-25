# PCB Defect Detection

Utility scripts and configs to train multiple defect detection models (YOLO, MMDetection, TorchVision, Transformers) on a 6-class PCB dataset.

## Kaggle Training Quickstart

Run in a Kaggle notebook with GPU enabled (Python 3.12, 2xT4):

```bash
# adjust path to where you placed the repo in your notebook
cd /kaggle/working/pcb-defect-detection

# (Optional) install training dependencies
python scripts/train_kaggle.py --install-deps --dry-run

# Train selected models
python scripts/train_kaggle.py \
  --project /kaggle/working/runs \
  --models yolo11s retinanet faster_rcnn cascade_rcnn detr deformable_detr \
  --gpus 2 \
  --yolo-device 0,1
```

Notes:
- The script auto-detects the dataset in common Kaggle input paths. Use `--data-source /kaggle/input/<dataset>/data` if it cannot find data.
- It uses your existing annotations directly:
  - YOLO: `data/<split>/labels_txt`
  - MMDetection models: `data/annotations_json/{train,val,test}.json` (COCO format)
- Train only one model, for example:
  - `python scripts/train_kaggle.py --models yolo11s`
  - `python scripts/train_kaggle.py --models detr --gpus 2`
