import argparse
import json
import time
from pathlib import Path

from benchmark_utils import (
    append_metrics_csv,
    benchmark_callable,
    bytes_to_mb,
    compute_f1,
    count_files_recursive,
    parse_last_losses_from_results_csv,
    safe_float,
    write_metrics_json,
)


def load_aug_cfg(path: Path):
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Please install pyyaml to use --aug-config: pip install pyyaml") from exc

    if not path.exists():
        raise SystemExit(f"Augmentation config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Augmentation config must be a YAML mapping.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate YOLO on unified PCB dataset")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset_unified_6sources")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="YOLO pretrained model path/name")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/unified_yolo")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("reports/metrics_yolo.csv"),
        help="Path to append YOLO benchmark metrics CSV.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("reports/metrics_yolo.json"),
        help="Path to write YOLO benchmark metrics JSON.",
    )
    parser.add_argument("--benchmark-images", type=int, default=32, help="Number of test images for speed benchmark")
    parser.add_argument("--benchmark-iters", type=int, default=8, help="Timed iterations for speed benchmark")
    parser.add_argument(
        "--aug-config",
        type=Path,
        default=Path("unified_pipeline/configs/yolo_augmentation.yaml"),
        help="Path to YOLO augmentation YAML. Use empty path to disable.",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Please install ultralytics: pip install ultralytics") from exc

    data_yaml = args.dataset / "yolo_format" / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"Missing file: {data_yaml}")

    train_kwargs = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "val": True,
    }

    if str(args.aug_config).strip():
        aug_cfg = load_aug_cfg(args.aug_config)
        train_kwargs.update(aug_cfg)

    model = YOLO(args.model)
    train_start = time.perf_counter()
    train_result = model.train(**train_kwargs)
    train_time_sec = time.perf_counter() - train_start

    metrics = model.val(data=str(data_yaml), split="test")

    p = safe_float(getattr(metrics.box, "mp", None))
    r = safe_float(getattr(metrics.box, "mr", None))
    map50 = safe_float(getattr(metrics.box, "map50", None))
    map50_95 = safe_float(getattr(metrics.box, "map", None))
    f1 = compute_f1(p, r)

    run_dir = Path(train_result.save_dir)
    best_ckpt = run_dir / "weights" / "best.pt"
    last_ckpt = run_dir / "weights" / "last.pt"
    ckpt = best_ckpt if best_ckpt.exists() else last_ckpt

    model_size_mb = bytes_to_mb(ckpt.stat().st_size) if ckpt.exists() else None
    trainable_params = None
    try:
        trainable_params = int(sum(p_.numel() for p_ in model.model.parameters() if p_.requires_grad))
    except Exception:
        pass

    test_img_root = args.dataset / "yolo_format" / "test" / "images"
    image_candidates = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_candidates.extend(test_img_root.rglob(ext))
    image_candidates = [str(pth) for pth in image_candidates[: max(1, args.benchmark_images)]]

    fps = None
    inference_time_ms = None
    if image_candidates:
        batch_source = image_candidates

        def infer_once() -> None:
            model.predict(
                source=batch_source,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                save=False,
                stream=False,
                conf=0.001,
            )

        bench = benchmark_callable(
            infer_fn=infer_once,
            warmup_iters=2,
            timed_iters=max(1, args.benchmark_iters),
            samples_per_iter=len(batch_source),
        )
        fps = bench["fps"]
        inference_time_ms = bench["inference_time_ms"]

    train_samples = count_files_recursive(args.dataset / "yolo_format" / "train" / "images", ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"])
    sec_per_epoch = train_time_sec / max(1, args.epochs)
    train_samples_per_sec = (float(train_samples) / train_time_sec) if train_time_sec > 0 else None

    losses = parse_last_losses_from_results_csv(run_dir / "results.csv")
    metrics_row = {
        "branch": "yolo",
        "model": args.model,
        "accuracy": map50,
        "precision": p,
        "recall": r,
        "f1": f1,
        "model_size_mb": model_size_mb,
        "trainable_params": trainable_params,
        "fps": fps,
        "inference_time_ms": inference_time_ms,
        "train_time_sec": train_time_sec,
        "train_samples": train_samples,
        "train_samples_per_sec": train_samples_per_sec,
        "sec_per_epoch": sec_per_epoch,
        "map50": map50,
        "map50_95": map50_95,
        "box_loss": losses["box_loss"],
        "cls_loss": losses["cls_loss"],
        "dfl_loss": losses["dfl_loss"],
        "notes": "accuracy uses mAP50 proxy for object detection",
    }

    append_metrics_csv(args.metrics_out, metrics_row)
    write_metrics_json(args.metrics_json, metrics_row)

    print("Training finished")
    print("Main metrics:")
    print(json.dumps(metrics_row, ensure_ascii=True, indent=2))
    print("Loss values (box_loss, cls_loss, dfl_loss) are logged per epoch in results.csv under the run folder.")
    print("Run directory:", train_result.save_dir)
    print("Metrics CSV:", args.metrics_out)
    print("Metrics JSON:", args.metrics_json)


if __name__ == "__main__":
    main()
