from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

MODEL_CONFIGS = {
    "retinanet": "retinanet_pcb.py",
    "faster_rcnn": "faster_rcnn_pcb.py",
    "cascade_rcnn": "cascade_rcnn_pcb.py",
    "detr": "detr_pcb.py",
    "deformable_detr": "deformable_detr_pcb.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MMDetection checkpoint to TensorRT engine via MMDeploy."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root that contains scripts/, data/, and classes.txt",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MODEL_CONFIGS.keys()),
        default="retinanet",
        help="Model key for scripts/mmdet_configs/<name>.py",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Override MMDetection config path. If omitted, use --model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained MMDetection checkpoint (.pth)",
    )
    parser.add_argument(
        "--test-image",
        type=Path,
        default=None,
        help="Image path used by MMDeploy export sanity pass. Default: first image in data/test/images/",
    )
    parser.add_argument(
        "--mmdeploy-root",
        type=Path,
        required=True,
        help="Path to cloned mmdeploy repository (must contain tools/deploy.py)",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default="configs/mmdet/detection/detection_tensorrt-fp16_dynamic-320x320-1344x1344.py",
        help="Path under mmdeploy root for deployment config",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("runs/tensorrt_export"),
        help="Export output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for export. Example: cuda or cpu",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command only, do not execute",
    )
    return parser.parse_args()


def resolve_model_config(dataset_root: Path, args: argparse.Namespace) -> Path:
    if args.model_config is not None:
        return args.model_config.resolve()
    return (dataset_root / "scripts" / "mmdet_configs" / MODEL_CONFIGS[args.model]).resolve()


def find_default_test_image(dataset_root: Path) -> Path:
    image_dir = dataset_root / "data" / "test" / "images"
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images = sorted(image_dir.glob(suffix))
        if images:
            return images[0].resolve()
    raise FileNotFoundError(f"No test image found in {image_dir}")


def validate_paths(
    mmdeploy_root: Path,
    deploy_config: Path,
    model_config: Path,
    checkpoint: Path,
    test_image: Path,
) -> Path:
    deploy_tool = (mmdeploy_root / "tools" / "deploy.py").resolve()
    required = [deploy_tool, deploy_config, model_config, checkpoint, test_image]
    missing = [p for p in required if not p.exists()]
    if missing:
        pretty = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(f"Missing required paths:\n{pretty}")
    return deploy_tool


def render_cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    mmdeploy_root = args.mmdeploy_root.resolve()

    model_config = resolve_model_config(dataset_root, args)
    deploy_config = (mmdeploy_root / args.deploy_config).resolve()
    checkpoint = args.checkpoint.resolve()
    test_image = args.test_image.resolve() if args.test_image else find_default_test_image(dataset_root)
    work_dir = (dataset_root / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    deploy_tool = validate_paths(mmdeploy_root, deploy_config, model_config, checkpoint, test_image)

    cmd = [
        sys.executable,
        str(deploy_tool),
        str(deploy_config),
        str(model_config),
        str(checkpoint),
        str(test_image),
        "--work-dir",
        str(work_dir),
        "--device",
        args.device,
        "--dump-info",
    ]

    print("[RUN]", render_cmd(cmd))
    if args.dry_run:
        raise SystemExit(0)

    code = subprocess.run(cmd, check=False, cwd=str(mmdeploy_root)).returncode
    raise SystemExit(code)


if __name__ == "__main__":
    main()
