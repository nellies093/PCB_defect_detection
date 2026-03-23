from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import train_all_models


KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def is_kaggle() -> bool:
    return KAGGLE_INPUT.exists() and KAGGLE_WORKING.exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kaggle-friendly training entrypoint for all PCB defect detection models."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Project root that contains scripts/ and classes.txt. "
        "Defaults to the repository root or a Kaggle working copy.",
    )
    parser.add_argument(
        "--data-source",
        type=Path,
        default=None,
        help=(
            "Optional path to a read-only dataset (e.g. /kaggle/input/pcb-defect-dataset). "
            "If provided and dataset-root/data is missing, a symlink will be created."
        ),
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Directory for training outputs. Defaults to /kaggle/working/runs on Kaggle.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=train_all_models.ALL_MODELS,
        choices=train_all_models.ALL_MODELS,
        help="Models to train.",
    )
    parser.add_argument(
        "--det-backend",
        type=str,
        choices=["mmdet", "torchvision", "transformers"],
        default="mmdet",
        help="Backend for non-YOLO detectors.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for all models.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for YOLO.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for YOLO/torchvision.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Training device. Examples: 0, 0,1, cpu.",
    )
    parser.add_argument(
        "--mmdet-lr",
        type=float,
        default=0.004,
        help="Base learning rate override for MMDetection configs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each model from latest checkpoint if possible.",
    )
    parser.add_argument(
        "--export-map-table",
        action="store_true",
        help="Export mAP comparison CSV/Markdown after training.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install training dependencies from scripts/requirements_train.txt before running.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not execute training.",
    )
    return parser.parse_args()


def _has_split_dirs(path: Path) -> bool:
    return all((path / split / "images").exists() for split in ("train", "val", "test"))


def guess_dataset_root(user_root: Path | None) -> Path:
    if user_root:
        return user_root.resolve()

    repo_root = Path(__file__).resolve().parents[1]
    candidates: Iterable[Path] = [repo_root]

    if is_kaggle():
        kaggle_repo = KAGGLE_WORKING / repo_root.name
        kaggle_input_repo = KAGGLE_INPUT / repo_root.name
        candidates = [
            kaggle_repo,
            kaggle_input_repo,
            KAGGLE_INPUT / "pcb-defect-detection",
            KAGGLE_INPUT / "pcb_defect_detection",
            KAGGLE_INPUT / "pcb-defect-dataset",
            repo_root,
        ]

    for path in candidates:
        if (path / "scripts").exists():
            return path.resolve()

    return repo_root.resolve()


def guess_data_source() -> Path | None:
    if not is_kaggle():
        return None

    candidates = [
        KAGGLE_INPUT / "pcb-defect-detection",
        KAGGLE_INPUT / "pcb_defect_detection",
        KAGGLE_INPUT / "pcb-defect-dataset",
        KAGGLE_INPUT / "pcb-defect-midterm",
    ]
    for path in candidates:
        if _has_split_dirs(path / "data"):
            return (path / "data").resolve()
        if _has_split_dirs(path):
            return path.resolve()
    return None


def ensure_data_dir(dataset_root: Path, data_source: Path | None) -> None:
    data_dir = dataset_root / "data"
    if data_dir.exists():
        return
    if data_source is None:
        raise FileNotFoundError(
            f"Missing dataset at {data_dir}. "
            "Provide --data-source pointing to a directory with train/val/test images."
        )

    if not _has_split_dirs(data_source):
        raise FileNotFoundError(
            f"data_source must contain train/val/test splits: {data_source}"
        )

    data_source = data_source.resolve()
    dataset_root = dataset_root.resolve()

    allowed_roots = [dataset_root]
    if KAGGLE_INPUT.exists():
        allowed_roots.append(KAGGLE_INPUT.resolve())

    if not any(
        data_source.is_relative_to(root) for root in allowed_roots if root.exists()
    ):
        allowed_list = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(
            f"data_source must live under one of: {allowed_list}. Got: {data_source}"
        )

    try:
        data_dir.symlink_to(data_source, target_is_directory=True)
    except OSError as exc:
        raise OSError(
            f"Failed to create symlink {data_dir} -> {data_source}. "
            "On Kaggle you can instead copy the data to a writable location "
            "or set --dataset-root to a path that already contains data/."
        ) from exc
    else:
        print(f"[INFO] Created symlink: {data_dir} -> {data_source}")


def resolve_project_dir(user_project: str | None) -> str:
    if user_project:
        project = Path(user_project)
    elif is_kaggle():
        project = KAGGLE_WORKING / "runs"
    else:
        project = Path("runs/pcb_train_kaggle")

    project.mkdir(parents=True, exist_ok=True)
    return str(project.resolve())


def install_requirements(dataset_root: Path) -> None:
    requirements = dataset_root / "scripts" / "requirements_train.txt"
    if not requirements.exists():
        print(f"[WARN] requirements file not found: {requirements}")
        return

    dataset_root = dataset_root.resolve()
    if not requirements.resolve().is_relative_to(dataset_root):
        raise ValueError(
            f"Refusing to install from unexpected path: {requirements}. "
            f"Must be inside {dataset_root}"
        )

    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
    print(f"[INFO] Installing dependencies: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()

    dataset_root = guess_dataset_root(args.dataset_root)
    data_source = args.data_source or guess_data_source()
    project_dir = resolve_project_dir(args.project)

    ensure_data_dir(dataset_root, data_source)

    if args.install_deps:
        install_requirements(dataset_root)

    kaggle_args = argparse.Namespace(
        dataset_root=dataset_root,
        models=args.models,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=project_dir,
        det_backend=args.det_backend,
        mmdet_lr=args.mmdet_lr,
        resume=args.resume,
        dry_run=args.dry_run,
        export_map_table=args.export_map_table,
    )

    exit_code = train_all_models.train_models(kaggle_args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
