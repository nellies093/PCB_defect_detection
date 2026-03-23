# PCB Defect Detection

Utility scripts and configs to train multiple defect detection models (YOLO, MMDetection, TorchVision, Transformers) on a 6-class PCB dataset.

## Kaggle Training Quickstart

Run in a Kaggle notebook with GPU enabled:

```bash
# adjust path to where you placed the repo in your notebook
cd /kaggle/working/pcb-defect-detection

# (Optional) install training dependencies
python scripts/train_kaggle.py --install-deps --dry-run

# Train (YOLO + MMDetection by default)
python scripts/train_kaggle.py \
  --project /kaggle/working/runs \
  --models yolo11s retinanet faster_rcnn
```

Notes:
- The script auto-detects the dataset in common Kaggle input paths. Use `--data-source /kaggle/input/<dataset>` if it cannot find `train/val/test`.
- To train only YOLO: `python scripts/train_kaggle.py --models yolo11s`
- To train Transformers backend: `python scripts/train_kaggle.py --det-backend transformers --models detr deformable_detr`
