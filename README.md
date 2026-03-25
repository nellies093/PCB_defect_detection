# PCB Defect Detection

Utility scripts and configs to train multiple defect detection models (YOLO, MMDetection, TorchVision, Transformers) on a 6-class PCB dataset.

## Kaggle Training Quickstart

### Option A – Kaggle Notebook (recommended)

Open `notebooks/pcb_defect_train.ipynb` directly on Kaggle.  
The notebook walks through environment setup, dependency installation, dry-run
verification, and training all (or selected) models with sensible defaults for
the 2× T4 / P100 configuration.

Assumptions:
- The repository contents have been copied to `/kaggle/working/project`.
- The dataset is attached as a Kaggle input dataset or placed under
  `/kaggle/working/project/data`.

### Option B – Command line

```bash
# Repository is at /kaggle/working/project
cd /kaggle/working/project

# (Optional) install training dependencies
python scripts/train_kaggle.py --install-deps --dry-run

# Train all models on 2× T4 GPUs
python scripts/train_kaggle.py \
  --project /kaggle/working/runs \
  --models yolo11s retinanet faster_rcnn cascade_rcnn detr deformable_detr \
  --epochs 50 \
  --gpus 2 \
  --yolo-device 0,1
```

Notes:
- The script auto-detects the dataset in common Kaggle input paths.
  Use `--data-source /kaggle/input/<dataset>/data` if it cannot find data.
- It uses your existing annotations directly:
  - YOLO: `data/<split>/images/` + `data/<split>/labels_txt/`
  - MMDetection models: `data/annotations_json/{train,val,test}.json` (COCO format)
- Multi-GPU MMDetection training uses `torchrun --nproc_per_node` (PyTorch DDP).
- Train only one model, for example:
  - `python scripts/train_kaggle.py --models yolo11s`
  - `python scripts/train_kaggle.py --models detr --gpus 2`
