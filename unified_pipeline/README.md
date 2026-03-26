# Unified Pipeline – Hướng dẫn sử dụng / Usage Guide

Thư mục `unified_pipeline/` là điểm vào mới cho quá trình huấn luyện tất cả các model, tích hợp kỹ thuật **Dynamic Tensor Rematerialization (DTR)** ([Kirisame et al., ICLR 2021](https://arxiv.org/pdf/2006.09616)) để tối ưu bộ nhớ GPU trong quá trình huấn luyện.

---

## Cấu trúc file / File structure

```
unified_pipeline/
├── __init__.py    – package init; re-exports apply_dtr, dtr_context
├── dtr.py         – triển khai DTR qua gradient checkpointing của PyTorch
├── train.py       – entry-point thống nhất (YOLO + MMDet + DTR)
└── README.md      – file này
```

---

## Yêu cầu / Requirements

Cài đặt dependencies trước khi chạy (chỉ cần làm một lần):

```bash
cd /kaggle/working/pcb-defect-detection   # hoặc thư mục clone của bạn

# Cài đặt tất cả dependencies
python -m pip install -r scripts/requirements_train.txt
python -m pip install -U openmim
python -m mim install "mmcv>=2.1.0,<2.2.0"
```

Hoặc để script tự cài qua `--install-deps`:

```bash
python -m unified_pipeline.train --install-deps --dry-run
```

---

## Cách chạy / How to run

> Tất cả lệnh bên dưới chạy từ **thư mục gốc của project** (nơi chứa `unified_pipeline/`).

### 1. Xem kế hoạch huấn luyện (không chạy thực)

```bash
python -m unified_pipeline.train --dry-run
```

### 2. Huấn luyện một model

```bash
# YOLO với DTR, 1 GPU
python -m unified_pipeline.train \
  --models yolo11s \
  --dtr \
  --gpus 1 \
  --yolo-device 0

# RetinaNet với DTR, 2 GPU
python -m unified_pipeline.train \
  --models retinanet \
  --dtr \
  --gpus 2
```

### 3. Huấn luyện tất cả model (như Kaggle notebook)

```bash
python -m unified_pipeline.train \
  --project /kaggle/working/runs \
  --models yolo11s retinanet faster_rcnn cascade_rcnn detr deformable_detr \
  --epochs 50 \
  --gpus 2 \
  --yolo-device 0,1 \
  --dtr
```

### 4. Giới hạn bộ nhớ GPU (DTR budget)

```bash
# Giới hạn 4 GB bộ nhớ GPU để buộc DTR kích hoạt sớm hơn
python -m unified_pipeline.train \
  --models detr \
  --dtr \
  --dtr-budget-mb 4096
```

### 5. Chỉ định đường dẫn dataset thủ công

```bash
python -m unified_pipeline.train \
  --data-source /kaggle/input/pcb-defect-dataset/data \
  --models faster_rcnn \
  --dtr \
  --gpus 2
```

---

## Tùy chọn CLI / CLI options

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--models` | tất cả | Tên model cần train: `yolo11s`, `retinanet`, `faster_rcnn`, `cascade_rcnn`, `detr`, `deformable_detr` |
| `--epochs` | 50 | Số epoch |
| `--imgsz` | 640 | Kích thước ảnh đầu vào (YOLO) |
| `--batch` | 32 | Batch size (YOLO) |
| `--workers` | 8 | Số worker của DataLoader |
| `--gpus` | 2 | Số GPU (cho MMDet multi-GPU) |
| `--yolo-device` | `0,1` | Device string cho YOLO |
| `--dtr` | tắt | Bật Dynamic Tensor Rematerialization |
| `--dtr-budget-mb` | None | Giới hạn bộ nhớ GPU (MiB) cho DTR allocator |
| `--data-source` | tự phát hiện | Đường dẫn tới dataset root |
| `--project` | `/kaggle/working/runs` | Thư mục lưu checkpoints & logs |
| `--install-deps` | tắt | Tự cài requirements trước khi train |
| `--dry-run` | tắt | In kế hoạch rồi thoát, không train thực |

---

## DTR hoạt động như thế nào? / How DTR works

Module `dtr.py` cung cấp hai API:

### `apply_dtr(model)`

Bật gradient checkpointing trên model PyTorch bất kỳ. Thay vì lưu tất cả activations trong forward pass, các activation sẽ bị xóa và tính lại khi cần trong backward pass – tiết kiệm bộ nhớ GPU đáng kể.

```python
from unified_pipeline.dtr import apply_dtr
import torchvision.models as tv

backbone = tv.resnet50(weights=None)
backbone = apply_dtr(backbone)   # gradient checkpointing được bật
```

### `dtr_context(budget_mb=None)`

Context manager để áp dụng cài đặt allocator của PyTorch (tự động reset sau khi thoát):

```python
from unified_pipeline.dtr import dtr_context

with dtr_context(budget_mb=4096):
    # vòng lặp huấn luyện của bạn
    ...
```

---

## Chạy từ Kaggle Notebook

```python
import subprocess, sys

# Clone project (nếu chưa có)
# !git clone https://github.com/nellies093/PCB_defect_detection /kaggle/working/project

%cd /kaggle/working/project

# Cài dependencies
subprocess.run([sys.executable, "-m", "unified_pipeline.train",
                "--install-deps", "--dry-run"], check=True)

# Train tất cả với DTR
subprocess.run([sys.executable, "-m", "unified_pipeline.train",
                "--dtr", "--gpus", "2", "--yolo-device", "0,1",
                "--project", "/kaggle/working/runs"], check=True)
```
