import argparse
import subprocess
from pathlib import Path


def run(cmd):
    print("Running:", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train both branches (YOLO + MMDetection) and export unified metrics")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mmdet-root", type=Path, required=True)

    parser.add_argument("--yolo-model", type=str, default="yolo11m.pt")
    parser.add_argument("--yolo-epochs", type=int, default=100)
    parser.add_argument("--yolo-batch", type=int, default=16)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-device", type=str, default="0")

    parser.add_argument("--mmdet-model", choices=["retinanet", "faster_rcnn", "cascade_rcnn", "detr", "deformable_detr", "all"], default="all")
    parser.add_argument("--mmdet-device", type=str, default="cuda:0")

    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)

    yolo_csv = args.report_dir / "metrics_yolo.csv"
    yolo_json = args.report_dir / "metrics_yolo.json"
    mmdet_csv = args.report_dir / "metrics_mmdet.csv"
    mmdet_json_dir = args.report_dir / "metrics_mmdet"

    run(
        [
            "python",
            "unified_pipeline/train_yolo.py",
            "--dataset",
            args.dataset,
            "--model",
            args.yolo_model,
            "--epochs",
            args.yolo_epochs,
            "--batch",
            args.yolo_batch,
            "--imgsz",
            args.yolo_imgsz,
            "--device",
            args.yolo_device,
            "--metrics-out",
            yolo_csv,
            "--metrics-json",
            yolo_json,
        ]
    )

    run(
        [
            "python",
            "unified_pipeline/train_mmdet.py",
            "--dataset",
            args.dataset,
            "--mmdet-root",
            args.mmdet_root,
            "--model",
            args.mmdet_model,
            "--device",
            args.mmdet_device,
            "--metrics-out",
            mmdet_csv,
            "--metrics-dir",
            mmdet_json_dir,
        ]
    )

    print("Dual-branch training completed.")
    print("YOLO metrics:", yolo_csv, yolo_json)
    print("MMDet metrics:", mmdet_csv, mmdet_json_dir)


if __name__ == "__main__":
    main()
