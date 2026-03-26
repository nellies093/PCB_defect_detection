import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

from benchmark_utils import append_metrics_csv, bytes_to_mb, compute_f1, safe_float, write_metrics_json


MODEL_CONFIGS = {
    "retinanet": "configs/retinanet/retinanet_r50_fpn_1x_coco.py",
    "faster_rcnn": "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
    "cascade_rcnn": "configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py",
    "detr": "configs/detr/detr_r50_8xb2-150e_coco.py",
    "deformable_detr": "configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py",
}

CLASS_NAMES = ("mouse_bite", "open_circuit", "mising_hole", "short", "spur", "spurious_copper")

NUM_CLASS_OVERRIDES = {
    "retinanet": ["model.bbox_head.num_classes=6"],
    "faster_rcnn": ["model.roi_head.bbox_head.num_classes=6"],
    "cascade_rcnn": [
        "model.roi_head.bbox_head.0.num_classes=6",
        "model.roi_head.bbox_head.1.num_classes=6",
        "model.roi_head.bbox_head.2.num_classes=6",
    ],
    "detr": ["model.bbox_head.num_classes=6"],
    "deformable_detr": ["model.bbox_head.num_classes=6"],
}


def run(cmd, cwd: Path, capture_output: bool = False) -> subprocess.CompletedProcess:
    print("Running:", " ".join(cmd))
    if capture_output:
        return subprocess.run(cmd, cwd=str(cwd), check=True, text=True, capture_output=True)
    return subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_max_epochs_from_cfg(cfg_path: Path) -> Optional[int]:
    content = cfg_path.read_text(encoding="utf-8")
    match = re.search(r"max_epochs\s*=\s*(\d+)", content)
    if not match:
        return None
    return int(match.group(1))


def parse_coco_metrics(stdout: str) -> Dict[str, Optional[float]]:
    def pick(pattern: str) -> Optional[float]:
        m = re.search(pattern, stdout)
        if not m:
            return None
        return safe_float(m.group(1))

    map_ = pick(r"coco/bbox_mAP\s*:\s*([0-9.]+)")
    map50 = pick(r"coco/bbox_mAP_50\s*:\s*([0-9.]+)")
    mar100 = pick(r"coco/bbox_mAR_100\s*:\s*([0-9.]+)")

    if map_ is None or map50 is None:
        m_copy = re.search(r"bbox_mAP_copypaste:\s*([0-9.\s]+)", stdout)
        if m_copy:
            vals = [safe_float(v) for v in m_copy.group(1).strip().split()]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                if map_ is None:
                    map_ = vals[0]
                if map50 is None:
                    map50 = vals[1]

    return {"map": map_, "map50": map50, "mar100": mar100}


def parse_losses_from_json_logs(work_dir: Path) -> Dict[str, Optional[float]]:
    log_candidates = sorted(work_dir.glob("*.json"))
    vis_dir = work_dir / "vis_data"
    if vis_dir.exists():
        log_candidates.extend(sorted(vis_dir.glob("*.json")))

    last_loss_record = None
    for log_file in log_candidates:
        with log_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if any(k in rec for k in ("loss_bbox", "loss_cls", "loss_dfl", "loss")):
                    last_loss_record = rec

    if not last_loss_record:
        return {"box_loss": None, "cls_loss": None, "dfl_loss": None}

    return {
        "box_loss": safe_float(last_loss_record.get("loss_bbox")) or safe_float(last_loss_record.get("loss")),
        "cls_loss": safe_float(last_loss_record.get("loss_cls")),
        "dfl_loss": safe_float(last_loss_record.get("loss_dfl")),
    }


def count_images_in_coco(coco_json: Path) -> int:
    payload = json.loads(coco_json.read_text(encoding="utf-8"))
    return len(payload.get("images", []))


def count_checkpoint_params(ckpt_path: Path) -> Optional[int]:
    if not ckpt_path.exists():
        return None
    try:
        import torch
    except ImportError:
        return None

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt if isinstance(ckpt, dict) else None)
    if not isinstance(state_dict, dict):
        return None

    total = 0
    for tensor in state_dict.values():
        if hasattr(tensor, "numel"):
            total += int(tensor.numel())
    return total if total > 0 else None


def benchmark_mmdet_inference(
    mmdet_root: Path,
    cfg_abs: Path,
    ckpt_path: Path,
    image_root: Path,
    device: str,
    max_images: int,
    timed_iters: int,
) -> Dict[str, Optional[float]]:
    script = f"""
import glob
import json
import time
from pathlib import Path

from mmdet.apis import inference_detector, init_detector

cfg = {repr(str(cfg_abs))}
ckpt = {repr(str(ckpt_path))}
img_root = Path({repr(str(image_root))})
device = {repr(device)}
max_images = {int(max_images)}
timed_iters = {int(timed_iters)}

images = []
for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
    images.extend(img_root.rglob(pattern))
images = [str(p) for p in images[:max(1, max_images)]]

if not images:
    print(json.dumps({{"fps": None, "inference_time_ms": None}}))
    raise SystemExit(0)

model = init_detector(cfg, ckpt, device=device)

for _ in range(2):
    for img in images:
        inference_detector(model, img)

start = time.perf_counter()
for _ in range(max(1, timed_iters)):
    for img in images:
        inference_detector(model, img)
elapsed = time.perf_counter() - start

total = max(1, timed_iters) * len(images)
fps = total / elapsed if elapsed > 0 else None
inf_ms = (elapsed / total) * 1000.0 if elapsed > 0 else None
print(json.dumps({{"fps": fps, "inference_time_ms": inf_ms}}))
"""

    result = subprocess.run(
        ["python", "-c", script],
        cwd=str(mmdet_root),
        text=True,
        capture_output=True,
        check=True,
    )

    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        return {"fps": None, "inference_time_ms": None}

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"fps": None, "inference_time_ms": None}

    return {
        "fps": safe_float(payload.get("fps")),
        "inference_time_ms": safe_float(payload.get("inference_time_ms")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MMDetection models on unified PCB dataset")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset_unified_6sources")
    parser.add_argument("--mmdet-root", type=Path, required=True, help="Path to mmdetection repo")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()) + ["all"], default="all")
    parser.add_argument("--work-dir", type=Path, default=Path("work_dirs/unified_pcb"))
    parser.add_argument("--gpus", type=str, default="1")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for benchmark, e.g. cuda:0 or cpu")
    parser.add_argument("--benchmark-images", type=int, default=16)
    parser.add_argument("--benchmark-iters", type=int, default=4)
    parser.add_argument("--metrics-out", type=Path, default=Path("reports/metrics_mmdet.csv"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("reports/metrics_mmdet"))
    parser.add_argument("--cfg-options", nargs="*", default=[])
    args = parser.parse_args()

    coco_train = args.dataset / "coco_annotations" / "train.json"
    coco_val = args.dataset / "coco_annotations" / "val.json"
    data_root = args.dataset / "yolo_format"

    if not coco_train.exists() or not coco_val.exists():
        raise SystemExit("Missing COCO annotations. Run build_unified_dataset.py first.")

    train_samples = count_images_in_coco(coco_train)
    selected = MODEL_CONFIGS.keys() if args.model == "all" else [args.model]

    for model_name in selected:
        cfg_rel = MODEL_CONFIGS[model_name]
        cfg_abs = args.mmdet_root / cfg_rel
        model_work_dir = args.work_dir / model_name
        max_epochs = parse_max_epochs_from_cfg(cfg_abs)

        cfg_overrides = [
            f"train_dataloader.dataset.ann_file={coco_train}",
            f"train_dataloader.dataset.data_root={data_root}",
            "train_dataloader.dataset.data_prefix.img=train/images/",
            f"train_dataloader.dataset.metainfo.classes={CLASS_NAMES}",
            f"val_dataloader.dataset.ann_file={coco_val}",
            f"val_dataloader.dataset.data_root={data_root}",
            "val_dataloader.dataset.data_prefix.img=val/images/",
            f"val_dataloader.dataset.metainfo.classes={CLASS_NAMES}",
            f"test_dataloader.dataset.ann_file={coco_val}",
            f"test_dataloader.dataset.data_root={data_root}",
            "test_dataloader.dataset.data_prefix.img=val/images/",
            f"test_dataloader.dataset.metainfo.classes={CLASS_NAMES}",
            f"val_evaluator.ann_file={coco_val}",
            f"test_evaluator.ann_file={coco_val}",
        ]

        cfg_overrides.extend(NUM_CLASS_OVERRIDES.get(model_name, []))

        cfg_overrides.extend(args.cfg_options)

        train_cmd = [
            "python",
            "tools/train.py",
            str(cfg_abs),
            "--work-dir",
            str(model_work_dir),
            "--cfg-options",
            *cfg_overrides,
        ]
        train_start = time.perf_counter()
        run(train_cmd, cwd=args.mmdet_root)
        train_time_sec = time.perf_counter() - train_start

        best_ckpt = model_work_dir / "latest.pth"
        test_cmd = [
            "python",
            "tools/test.py",
            str(cfg_abs),
            str(best_ckpt),
            "--cfg-options",
            *cfg_overrides,
            "--eval",
            "bbox",
        ]
        test_result = run(test_cmd, cwd=args.mmdet_root, capture_output=True)
        metrics = parse_coco_metrics(test_result.stdout)

        precision = metrics["map"]
        recall = metrics["mar100"]
        map50 = metrics["map50"]
        map50_95 = metrics["map"]
        f1 = compute_f1(precision, recall)

        infer_bench = benchmark_mmdet_inference(
            mmdet_root=args.mmdet_root,
            cfg_abs=cfg_abs,
            ckpt_path=best_ckpt,
            image_root=data_root / "val" / "images",
            device=args.device,
            max_images=args.benchmark_images,
            timed_iters=args.benchmark_iters,
        )

        sec_per_epoch = (train_time_sec / max_epochs) if (max_epochs and max_epochs > 0) else None
        train_samples_per_sec = (float(train_samples) / train_time_sec) if train_time_sec > 0 else None
        losses = parse_losses_from_json_logs(model_work_dir)

        metrics_row = {
            "branch": "mmdet",
            "model": model_name,
            "accuracy": map50,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "model_size_mb": bytes_to_mb(best_ckpt.stat().st_size) if best_ckpt.exists() else None,
            "trainable_params": count_checkpoint_params(best_ckpt),
            "fps": infer_bench["fps"],
            "inference_time_ms": infer_bench["inference_time_ms"],
            "train_time_sec": train_time_sec,
            "train_samples": train_samples,
            "train_samples_per_sec": train_samples_per_sec,
            "sec_per_epoch": sec_per_epoch,
            "map50": map50,
            "map50_95": map50_95,
            "box_loss": losses["box_loss"],
            "cls_loss": losses["cls_loss"],
            "dfl_loss": losses["dfl_loss"],
            "notes": "accuracy uses mAP50 proxy; precision uses mAP; recall uses mAR@100 when available",
        }

        append_metrics_csv(args.metrics_out, metrics_row)
        write_metrics_json(args.metrics_dir / f"{model_name}.json", metrics_row)
        print(json.dumps(metrics_row, ensure_ascii=True, indent=2))

    print("Done. Metrics CSV:", args.metrics_out)
    print("Per-model JSON directory:", args.metrics_dir)


if __name__ == "__main__":
    main()
