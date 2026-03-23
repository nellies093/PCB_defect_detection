from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for PCB defect detection.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root that contains data/ and scripts/.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to YOLO data yaml. Default: <dataset-root>/scripts/yolo_data.yaml",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Pretrained checkpoint path or model name (e.g. yolov11m.pt).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use: 0, 0,1, or cpu.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Dataloader worker count.")
    parser.add_argument(
        "--project",
        type=str,
        default="runs/yolo_local_train",
        help="Directory to store training runs.",
    )
    parser.add_argument("--name", type=str, default="yolov11s", help="Run name.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after training.",
    )
    return parser.parse_args()


def resolve_weights(dataset_root: Path, custom_weights: str | None) -> str:
    if custom_weights:
        return custom_weights

    local_default = dataset_root / "scripts" / "yolov11s.pt"
    if local_default.exists():
        return str(local_default)

    # Fallback to Ultralytics registry name if local checkpoint is not found.
    return "yolov11s.pt"


def validate_paths(dataset_root: Path, data_yaml: Path) -> None:
    required = [
        dataset_root / "data" / "train" / "images",
        dataset_root / "data" / "val" / "images",
        dataset_root / "data" / "test" / "images",
        data_yaml,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        pretty = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required paths:\n{pretty}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    data_yaml = (args.data or (dataset_root / "scripts" / "yolo_data.yaml")).resolve()
    weights = resolve_weights(dataset_root, args.weights)

    validate_paths(dataset_root, data_yaml)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    model = YOLO(weights)

    print("[INFO] Start YOLO training")
    print(f"[INFO] data={data_yaml}")
    print(f"[INFO] weights={weights}")
    print(f"[INFO] device={args.device}")

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str((dataset_root / args.project).resolve()),
        name=args.name,
        resume=args.resume,
    )

    if args.validate:
        print("[INFO] Run validation")
        model.val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device)


if __name__ == "__main__":
    main()
