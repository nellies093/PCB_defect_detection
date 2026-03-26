from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCB defect model locally with Ultralytics YOLO")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root containing train/val/test")
    parser.add_argument("--data-yaml", type=Path, default=None, help="Path to YOLO data yaml")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Model checkpoint, e.g. yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, 0, 0,1")
    parser.add_argument("--project", type=Path, default=Path("runs/train_local"))
    parser.add_argument("--name", type=str, default="yolo11s_local")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def detect_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def detect_data_root(repo_root: Path, user_data_root: Path | None) -> Path:
    candidates = []
    if user_data_root is not None:
        candidates.append(user_data_root)
    candidates.append(repo_root / "data")

    for cand in candidates:
        if (cand / "train" / "images").exists() and (cand / "val" / "images").exists() and (cand / "test" / "images").exists():
            return cand.resolve()
    raise FileNotFoundError("Cannot find dataset root with train/val/test images folders")


def read_class_names(repo_root: Path, data_root: Path) -> list[str]:
    candidates = [repo_root / "classes.txt", data_root / "classes.txt", data_root.parent / "classes.txt"]
    for path in candidates:
        if path.exists():
            names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if names:
                return names
    raise FileNotFoundError("Cannot find classes.txt or file is empty")


def write_data_yaml(data_yaml: Path, data_root: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {data_root.as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    repo_root = detect_repo_root()
    data_root = detect_data_root(repo_root, args.data_root)

    data_yaml = args.data_yaml.resolve() if args.data_yaml else (repo_root / "scripts" / "yolo_data_local.yaml")
    class_names = read_class_names(repo_root, data_root)
    write_data_yaml(data_yaml, data_root, class_names)

    device = resolve_device(args.device)
    project_dir = args.project if args.project.is_absolute() else (repo_root / args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Local training configuration")
    print(f"  repo_root: {repo_root}")
    print(f"  data_root: {data_root}")
    print(f"  data_yaml: {data_yaml}")
    print(f"  model:     {args.model}")
    print(f"  device:    {device}")
    print(f"  project:   {project_dir}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(project_dir),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        resume=args.resume,
        exist_ok=True,
        amp=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
