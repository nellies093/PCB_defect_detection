# TensorRT Deployment for MMDetection Models (Kaggle)

This guide deploys trained MMDetection checkpoints to TensorRT engines using MMDeploy.

## 1) Environment (Kaggle)

Run in a Kaggle notebook with GPU enabled.

```bash
pip install -U openmim
mim install "mmengine>=0.10.0,<1.0.0"
mim install "mmcv>=2.0.0,<2.2.0"
mim install "mmdet>=3.2.0,<3.4.0"
```

Install TensorRT + MMDeploy runtime stack (version depends on your Kaggle image).

```bash
# Clone MMDeploy
cd /kaggle/working
git clone https://github.com/open-mmlab/mmdeploy.git

# Install mmdeploy python package from source tree
pip install -e /kaggle/working/mmdeploy
```

## 2) Export checkpoint -> TensorRT engine

Use the wrapper script in this repo.

```bash
cd /kaggle/working/project/pcb-defect-dataset
python scripts/export_mmdet_tensorrt.py \
  --mmdeploy-root /kaggle/working/mmdeploy \
  --model retinanet \
  --checkpoint runs/pcb_train/retinanet/latest.pth \
  --work-dir runs/tensorrt_export/retinanet
```

Repeat with `--model faster_rcnn`, `--model cascade_rcnn`, `--model detr`, or `--model deformable_detr`.

## 3) TensorRT inference

```bash
cd /kaggle/working/project/pcb-defect-dataset
python scripts/infer_tensorrt_mmdet.py \
  --model-dir runs/tensorrt_export/retinanet \
  --input data/test/images \
  --classes classes.txt \
  --output-dir runs/tensorrt_infer/retinanet \
  --score-thr 0.25
```

## Notes

- If export fails for `deformable_detr` or `cascade_rcnn`, try `retinanet` or `faster_rcnn` first.
- TensorRT export is sensitive to CUDA/TensorRT/MMDeploy version compatibility.
- The export script expects a valid checkpoint and at least one image in `data/test/images/`.
