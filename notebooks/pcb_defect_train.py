from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MMDET_MODELS = {"retinanet", "faster_rcnn", "cascade_rcnn", "detr", "deformable_detr"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PCB training flow from a Python script (Notebook equivalent).")
    parser.add_argument("--project-dir", type=Path, default=Path("/kaggle/working/project"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/runs"))
    parser.add_argument("--data-source", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11s", "retinanet", "faster_rcnn", "cascade_rcnn", "detr", "deformable_detr"],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--yolo-device", type=str, default="0,1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--install-deps", action="store_true", help="Install Python dependencies before training")
    parser.add_argument("--dry-run", action="store_true", help="Run train_kaggle.py with --dry-run only")
    return parser


def install_deps(project_dir: Path, models: list[str]) -> None:
    req_candidates = [
        project_dir / "scripts" / "requirements_train.txt",
        project_dir / "requirements_kaggle.txt",
        project_dir / "requiment.txt",
    ]
    req_file = next((p for p in req_candidates if p.exists()), None)
    if req_file is None:
        raise FileNotFoundError("No requirements file found in project root/scripts.")

    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pip"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], check=True)
    print(f"Installed dependencies from: {req_file}")

    if any(m in MMDET_MODELS for m in models):
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "openmim"], check=True)
        try:
            subprocess.run([sys.executable, "-m", "mim", "install", "mmcv>=2.1.0,<2.2.0"], check=True)
        except subprocess.CalledProcessError:
            subprocess.run([sys.executable, "-m", "pip", "install", "mmcv>=2.1.0,<2.2.0"], check=True)


def main() -> None:
    args = build_parser().parse_args()

    train_script = args.project_dir / "scripts" / "train_kaggle.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Cannot find training script: {train_script}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.install_deps:
        install_deps(args.project_dir, args.models)

    cmd = [
        sys.executable,
        str(train_script),
        "--models",
        *args.models,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--gpus",
        str(args.gpus),
        "--yolo-device",
        args.yolo_device,
        "--workers",
        str(args.workers),
        "--project",
        str(args.output_dir),
    ]

    if args.data_source is not None:
        cmd.extend(["--data-source", str(args.data_source)])
    if args.dry_run:
        cmd.append("--dry-run")

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(args.project_dir), check=True)


if __name__ == "__main__":
    main()
