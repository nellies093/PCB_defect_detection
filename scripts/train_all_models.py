from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


YOLO_MODEL_KEY = "yolo11s"
MMDET_MODELS = {
    "retinanet": "retinanet_pcb.py",
    "faster_rcnn": "faster_rcnn_pcb.py",
    "cascade_rcnn": "cascade_rcnn_pcb.py",
    "detr": "detr_pcb.py",
    "deformable_detr": "deformable_detr_pcb.py",
}
TORCHVISION_MODELS = {"retinanet", "faster_rcnn"}
TRANSFORMERS_MODELS = {"detr", "deformable_detr"}
ALL_MODELS = [YOLO_MODEL_KEY, *MMDET_MODELS.keys()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PCB defect detectors sequentially with YOLO + MMDetection or torchvision backends."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root that contains data/, scripts/, and classes.txt",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Models to train in sequence",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for all models")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for YOLO")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for YOLO")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Training device. Examples: 0, 0,1, cpu",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader workers passed to both YOLO and MMDetection",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/pcb_train",
        help="Output root for training runs",
    )
    parser.add_argument(
        "--det-backend",
        type=str,
        choices=["mmdet", "torchvision", "transformers"],
        default="mmdet",
        help=(
            "Backend for non-YOLO detectors. "
            "torchvision: retinanet/faster_rcnn, transformers: detr/deformable_detr"
        ),
    )
    parser.add_argument(
        "--mmdet-lr",
        type=float,
        default=0.004,
        help="Base learning rate override for MMDetection configs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each model from latest checkpoint if possible",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not execute",
    )
    parser.add_argument(
        "--export-map-table",
        action="store_true",
        help="Export mAP comparison CSV/Markdown after training",
    )
    return parser.parse_args()


def run_command(cmd: List[str], cwd: Path, dry_run: bool) -> int:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"\n[RUN] {rendered}")
    print(f"[CWD] {cwd}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    return proc.returncode


def validate_paths(dataset_root: Path) -> tuple[Path, Path, Path]:
    data_dir = dataset_root / "data"
    yolo_yaml = dataset_root / "scripts" / "yolo_data.yaml"
    mmdet_cfg_dir = dataset_root / "scripts" / "mmdet_configs"

    required = [
        data_dir / "annotations_json" / "train.json",
        data_dir / "annotations_json" / "val.json",
        data_dir / "annotations_json" / "test.json",
        data_dir / "train" / "images",
        data_dir / "val" / "images",
        data_dir / "test" / "images",
        yolo_yaml,
        mmdet_cfg_dir,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        pretty = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required dataset/config paths:\n{pretty}")

    return data_dir, yolo_yaml, mmdet_cfg_dir


def train_yolo11s(args: argparse.Namespace, dataset_root: Path, yolo_yaml: Path) -> int:
    project_path = dataset_root / args.project
    print("\n[RUN] ultralytics.YOLO('yolo11s.pt').train(...)")
    print(f"[CWD] {dataset_root}")
    if args.dry_run:
        return 0

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[FAIL] ultralytics import failed: {exc}")
        return 1

    model = YOLO("yolo11s.pt")
    model.train(
        data=str(yolo_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(project_path),
        name="yolo11s",
        resume=args.resume,
    )
    return 0


def build_mmdet_cmd(
    args: argparse.Namespace,
    dataset_root: Path,
    mmdet_cfg_dir: Path,
    model_name: str,
) -> List[str]:
    config_path = mmdet_cfg_dir / MMDET_MODELS[model_name]
    work_dir = dataset_root / args.project / model_name

    cmd = [
        sys.executable,
        "-m",
        "mim",
        "train",
        "mmdet",
        str(config_path),
        "--work-dir",
        str(work_dir),
        "--cfg-options",
        f"work_dir={work_dir}",
        f"train_cfg.max_epochs={args.epochs}",
        f"optim_wrapper.optimizer.lr={args.mmdet_lr}",
        f"train_dataloader.num_workers={args.workers}",
        f"val_dataloader.num_workers={max(1, args.workers // 2)}",
        f"test_dataloader.num_workers={max(1, args.workers // 2)}",
    ]

    if args.device.lower() == "cpu":
        cmd += ["--launcher", "none"]

    if args.resume:
        cmd += ["--resume", "auto"]

    return cmd


def build_torchvision_cmd(
    args: argparse.Namespace,
    dataset_root: Path,
    model_name: str,
) -> List[str]:
    run_name = f"torchvision_{model_name}"
    cmd = [
        sys.executable,
        str(dataset_root / "scripts" / "train_torchvision.py"),
        "--dataset-root",
        str(dataset_root),
        "--model",
        model_name,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        args.device,
        "--project",
        args.project,
        "--name",
        run_name,
    ]
    if args.resume:
        cmd.append("--resume")
    return cmd


def build_transformers_cmd(
    args: argparse.Namespace,
    dataset_root: Path,
    model_name: str,
) -> List[str]:
    run_name = f"transformers_{model_name}"
    cmd = [
        sys.executable,
        str(dataset_root / "scripts" / "train_transformers.py"),
        "--dataset-root",
        str(dataset_root),
        "--model",
        model_name,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        args.device,
        "--project",
        args.project,
        "--name",
        run_name,
    ]
    if args.resume:
        cmd.append("--resume")
    return cmd


def train_models(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root.resolve()
    _, yolo_yaml, mmdet_cfg_dir = validate_paths(dataset_root)

    failures: list[tuple[str, int]] = []
    for model_name in args.models:
        print(f"\n========== Training: {model_name} ==========")
        if model_name == YOLO_MODEL_KEY:
            code = train_yolo11s(args, dataset_root, yolo_yaml)
        else:
            if args.det_backend == "torchvision":
                if model_name not in TORCHVISION_MODELS:
                    print(
                        f"[FAIL] torchvision backend does not support '{model_name}'. "
                        "Use --det-backend mmdet for cascade_rcnn/detr/deformable_detr."
                    )
                    failures.append((model_name, 2))
                    continue
                cmd = build_torchvision_cmd(args, dataset_root, model_name)
            elif args.det_backend == "transformers":
                if model_name not in TRANSFORMERS_MODELS:
                    print(
                        f"[FAIL] transformers backend does not support '{model_name}'. "
                        "Use --det-backend mmdet or torchvision for this model."
                    )
                    failures.append((model_name, 2))
                    continue
                cmd = build_transformers_cmd(args, dataset_root, model_name)
            else:
                cmd = build_mmdet_cmd(args, dataset_root, mmdet_cfg_dir, model_name)
            code = run_command(cmd, cwd=dataset_root, dry_run=args.dry_run)

        if code != 0:
            failures.append((model_name, code))
            print(f"[FAIL] {model_name} exited with code {code}")
        else:
            print(f"[OK] {model_name} finished successfully")

    print("\n========== Summary ==========")
    if args.export_map_table:
        export_script = dataset_root / "scripts" / "export_map_comparison.py"
        cmd = [
            sys.executable,
            str(export_script),
            "--project-dir",
            str(dataset_root / args.project),
        ]
        print("\n========== Export mAP Table ==========")
        export_code = run_command(cmd, cwd=dataset_root, dry_run=args.dry_run)
        if export_code != 0:
            print(f"[WARN] mAP export exited with code {export_code}")

    if not failures:
        print("All selected models finished successfully.")
        return 0

    for model_name, code in failures:
        print(f"- {model_name}: exit code {code}")
    return 1


def main() -> None:
    args = parse_args()
    exit_code = train_models(args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
