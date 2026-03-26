from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
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


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def is_kaggle_runtime() -> bool:
    return Path("/kaggle").exists() or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def print_runtime_summary(project_dir: Path, checkpoint_root: Path) -> None:
    print("[INFO] Runtime summary")
    print(f"  python:          {sys.version.split()[0]}")
    print(f"  kaggle_runtime:  {is_kaggle_runtime()}")
    print(f"  project_dir:     {project_dir}")
    print(f"  checkpoint_root: {checkpoint_root}")
    if project_dir.exists():
        usage = shutil.disk_usage(project_dir)
        free_gb = usage.free / (1024 ** 3)
        print(f"  free_disk_gb:    {free_gb:.2f}")


def cleanup_before_train(paths: Paths, models: list[str], checkpoint_root: Path) -> None:
    print("[INFO] Cleanup before training...")
    removed_files = 0
    removed_dirs = 0

    targets = [
        paths.out_dir,
        paths.repo_root / "plots",
        paths.repo_root / "reports",
    ]

    for base in targets:
        if not base.exists():
            continue

        for pyc in base.rglob("*.pyc"):
            try:
                pyc.unlink()
                removed_files += 1
            except OSError:
                pass

        for pyo in base.rglob("*.pyo"):
            try:
                pyo.unlink()
                removed_files += 1
            except OSError:
                pass

        for tmp in base.rglob("*.tmp"):
            try:
                tmp.unlink()
                removed_files += 1
            except OSError:
                pass

        for d in sorted(base.rglob("__pycache__"), reverse=True):
            try:
                shutil.rmtree(d, ignore_errors=True)
                removed_dirs += 1
            except OSError:
                pass

        for d in sorted(base.rglob(".ipynb_checkpoints"), reverse=True):
            try:
                shutil.rmtree(d, ignore_errors=True)
                removed_dirs += 1
            except OSError:
                pass

    # Remove stale artifacts from previous runs for selected models.
    for model_name in models:
        for stale in paths.out_dir.glob(f"{model_name}*"):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
                removed_dirs += 1

    stale_saved = paths.out_dir / "saved_models"
    if stale_saved.exists():
        shutil.rmtree(stale_saved, ignore_errors=True)
        removed_dirs += 1

    for scope in ("hourly", "final"):
        scope_dir = checkpoint_root / scope
        for model_name in models:
            stale_scope = scope_dir / model_name
            if stale_scope.exists():
                shutil.rmtree(stale_scope, ignore_errors=True)
                removed_dirs += 1

    print(f"[INFO] Cleanup completed: removed_files={removed_files}, removed_dirs={removed_dirs}")


def resolve_model_output_dir(paths: Paths, model_name: str) -> Path | None:
    direct = paths.out_dir / model_name
    if direct.exists() and direct.is_dir():
        return direct

    candidates = [p for p in paths.out_dir.glob(f"{model_name}*") if p.is_dir()]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def snapshot_model_artifacts(paths: Paths, model_name: str, dest_root: Path, reason: str) -> Path | None:
    src_dir = resolve_model_output_dir(paths, model_name)
    if src_dir is None:
        print(f"[WARN] No output directory found for model '{model_name}' to snapshot ({reason}).")
        return None

    dest_dir = dest_root / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)

    manifest = {
        "model": model_name,
        "reason": reason,
        "timestamp": now_tag(),
        "source_dir": str(src_dir),
        "snapshot_dir": str(dest_dir),
    }
    (dest_dir / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[INFO] Snapshot saved ({reason}): {dest_dir}")
    return dest_dir


def save_completed_model(paths: Paths, model_name: str, saved_models_root: Path) -> Path | None:
    src_dir = resolve_model_output_dir(paths, model_name)
    if src_dir is None:
        print(f"[WARN] Cannot save completed model '{model_name}': output directory not found.")
        return None

    saved_models_root.mkdir(parents=True, exist_ok=True)
    dest_dir = saved_models_root / f"{model_name}_{now_tag()}"
    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
    print(f"[INFO] Model artifacts stored: {dest_dir}")
    return dest_dir


def periodic_checkpoint_worker(
    stop_event: threading.Event,
    paths: Paths,
    model_name: str,
    checkpoint_root: Path,
    interval_seconds: int,
) -> None:
    if interval_seconds <= 0:
        return

    while not stop_event.wait(timeout=interval_seconds):
        try:
            snapshot_model_artifacts(
                paths=paths,
                model_name=model_name,
                dest_root=checkpoint_root / "hourly",
                reason="hourly",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Hourly checkpoint failed for {model_name}: {exc}")


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
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/kaggle/working/checkpoints"))
    parser.add_argument("--checkpoint-every-minutes", type=int, default=60)
    parser.add_argument("--no-cleanup-before-train", action="store_true")
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

    checkpoint_root = args.checkpoint_root.resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    print_runtime_summary(project_dir=paths.out_dir, checkpoint_root=checkpoint_root)

    if not args.no_cleanup_before_train:
        cleanup_before_train(paths=paths, models=models, checkpoint_root=checkpoint_root)

    if args.install_deps:
        install_dependencies(scripts_dir=scripts_dir, with_mmdet=any(m in MMDET_MODELS for m in models))

    print("[INFO] Training plan")
    print(f"  repo_root: {paths.repo_root}")
    print(f"  data_root: {paths.data_root}")
    print(f"  project:   {paths.out_dir}")
    print(f"  ckpt_root: {checkpoint_root}")
    print(f"  models:    {models}")

    if args.dry_run:
        return

    checkpoint_interval_seconds = max(1, args.checkpoint_every_minutes) * 60
    saved_models_root = paths.out_dir / "saved_models"

    for model_name in models:
        print(f"\n[INFO] Start model: {model_name}")
        stop_event = threading.Event()
        checkpoint_thread = threading.Thread(
            target=periodic_checkpoint_worker,
            args=(stop_event, paths, model_name, checkpoint_root, checkpoint_interval_seconds),
            daemon=True,
        )
        checkpoint_thread.start()

        try:
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

            snapshot_model_artifacts(
                paths=paths,
                model_name=model_name,
                dest_root=checkpoint_root / "final",
                reason="model_completed",
            )
            save_completed_model(paths=paths, model_name=model_name, saved_models_root=saved_models_root)
        finally:
            stop_event.set()
            checkpoint_thread.join(timeout=5)


if __name__ == "__main__":
    main()
