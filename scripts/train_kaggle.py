from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_MODELS = (
    "yolo11s",
    "retinanet",
    "faster_rcnn",
    "cascade_rcnn",
    "detr",
    "deformable_detr",
)

MMDET_MODELS = {"retinanet", "faster_rcnn", "cascade_rcnn", "detr", "deformable_detr"}


@dataclass
class Paths:
    repo_root: Path
    scripts_dir: Path
    data_root: Path
    yolo_yaml: Path
    coco_ann_dir: Path
    out_dir: Path


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def detect_data_root(repo_root: Path, data_source: Path | None) -> Path:
    candidates: list[Path] = []
    if data_source:
        candidates.append(data_source)
    candidates.extend(
        [
            repo_root / "data",
            Path("/kaggle/input/pcb-defect-dataset/data"),
            Path("/kaggle/input/pcb-defect-detection/data"),
            Path("/kaggle/input/pcb-defect/data"),
        ]
    )

    for cand in candidates:
        if (
            (cand / "train").exists()
            and (cand / "val").exists()
            and (cand / "test").exists()
            and (cand / "annotations_json").exists()
        ):
            return cand.resolve()
    raise FileNotFoundError("Could not detect dataset root containing train/val/test + annotations_json")


def ensure_yolo_yaml(yolo_yaml: Path, data_root: Path) -> None:
    classes_path = data_root.parent / "classes.txt"
    if not classes_path.exists():
        classes_path = data_root / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError("Missing classes.txt to auto-generate YOLO yaml")

    class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
    yolo_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


def install_dependencies(scripts_dir: Path, with_mmdet: bool) -> None:
    req = scripts_dir / "requirements_train.txt"
    if req.exists():
        run_cmd([sys.executable, "-m", "pip", "install", "-U", "pip"])
        run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req)])
    if with_mmdet:
        # mmcv often requires mim on GPU environments.
        run_cmd([sys.executable, "-m", "pip", "install", "-U", "openmim"])
        run_cmd([sys.executable, "-m", "mim", "install", "mmcv>=2.1.0,<2.2.0"])


def train_yolo(paths: Paths, epochs: int, imgsz: int, batch: int, device: str, workers: int) -> None:
    # Build device argument: comma-separated becomes list notation for YOLO multi-GPU
    device_arg = f"[{device}]" if "," in device else device
    cmd = [
        "yolo",
        "train",
        "model=yolo11s.pt",
        f"data={paths.yolo_yaml.as_posix()}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"device={device_arg}",
        f"workers={workers}",
        f"project={paths.out_dir.as_posix()}",
        "name=yolo11s",
        "exist_ok=True",
        "plots=True",
        "amp=True",
    ]
    run_cmd(cmd, cwd=paths.repo_root)



def mmdet_cfg_path(paths: Paths, model_name: str) -> Path:
    return paths.scripts_dir / "mmdet_configs" / f"{model_name}_pcb.py"


def train_mmdet(paths: Paths, model_name: str, epochs: int, gpus: int) -> None:
    cfg = mmdet_cfg_path(paths, model_name)
    if not cfg.exists():
        raise FileNotFoundError(f"Missing config: {cfg}")
    work_dir = paths.out_dir / model_name
    work_dir.mkdir(parents=True, exist_ok=True)

    # Pass absolute data_root so the config works regardless of cwd.
    # Trailing slash is required by MMDetection's CocoDataset path joining.
    data_root_posix = paths.data_root.as_posix().rstrip("/") + "/"
    cfg_options = [
        f"train_dataloader.dataset.data_root={data_root_posix}",
        f"val_dataloader.dataset.data_root={data_root_posix}",
        f"test_dataloader.dataset.data_root={data_root_posix}",
        f"val_evaluator.ann_file={data_root_posix}annotations_json/val.json",
        f"test_evaluator.ann_file={data_root_posix}annotations_json/test.json",
        f"train_cfg.max_epochs={epochs}",
        f"default_hooks.checkpoint.interval={max(1, min(5, epochs))}",
    ]

    if gpus > 1:
        # MMDetection 3.x dropped the legacy --gpus flag; distributed training
        # requires PyTorch DDP launched via torchrun with --launcher pytorch.
        launcher = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(gpus),
            "-m", "mmdet.tools.train",
            str(cfg),
            "--work-dir", str(work_dir),
            "--launcher", "pytorch",
            "--cfg-options",
        ] + cfg_options
    else:
        launcher = [
            "mim",
            "train",
            "mmdet",
            str(cfg),
            "--work-dir",
            str(work_dir),
            "--cfg-options",
        ] + cfg_options
    run_cmd(launcher, cwd=paths.repo_root)


def parse_models(values: Iterable[str]) -> list[str]:
    models: list[str] = []
    for v in values:
        key = v.strip().lower()
        if key not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{v}'. Supported: {', '.join(SUPPORTED_MODELS)}")
        models.append(key)
    # Keep order but remove duplicates
    dedup: list[str] = []
    seen = set()
    for m in models:
        if m not in seen:
            dedup.append(m)
            seen.add(m)
    return dedup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PCB models on Kaggle (Python 3.12 + 2xT4)")
    parser.add_argument("--data-source", type=Path, default=None, help="Dataset root containing data/train|val|test")
    parser.add_argument("--project", type=Path, default=Path("/kaggle/working/runs"))
    parser.add_argument("--models", nargs="+", default=list(SUPPORTED_MODELS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--yolo-device", type=str, default="0,1")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    data_root = detect_data_root(repo_root, args.data_source)
    yolo_yaml = scripts_dir / "yolo_data.yaml"
    ensure_yolo_yaml(yolo_yaml, data_root)

    if data_root != repo_root / "data":
        local_data = repo_root / "data"
        if local_data.exists():
            print(f"[INFO] Using external dataset: {data_root}")
        else:
            # Optional symlink to simplify relative config references in notebook runs.
            try:
                os.symlink(data_root, local_data, target_is_directory=True)
                print(f"[INFO] Created symlink: {local_data} -> {data_root}")
            except OSError:
                shutil.copytree(data_root, local_data, dirs_exist_ok=True)
                print(f"[INFO] Copied dataset into: {local_data}")

    paths = Paths(
        repo_root=repo_root,
        scripts_dir=scripts_dir,
        data_root=data_root,
        yolo_yaml=yolo_yaml,
        coco_ann_dir=data_root / "annotations_json",
        out_dir=args.project.resolve(),
    )
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    if args.install_deps:
        install_dependencies(scripts_dir=scripts_dir, with_mmdet=any(m in MMDET_MODELS for m in models))

    print("[INFO] Training plan")
    print(f"  repo_root: {paths.repo_root}")
    print(f"  data_root: {paths.data_root}")
    print(f"  project:   {paths.out_dir}")
    print(f"  models:    {models}")

    if args.dry_run:
        return

    for model_name in models:
        print(f"\n[INFO] Start model: {model_name}")
        if model_name == "yolo11s":
            train_yolo(
                paths=paths,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.yolo_device,
                workers=args.workers,
            )
        else:
            train_mmdet(paths=paths, model_name=model_name, epochs=args.epochs, gpus=args.gpus)


if __name__ == "__main__":
    main()
