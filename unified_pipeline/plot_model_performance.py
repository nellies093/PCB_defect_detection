import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def safe_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


METRIC_COLUMNS = [
    "branch",
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "model_size_mb",
    "trainable_params",
    "fps",
    "inference_time_ms",
    "train_time_sec",
    "train_samples",
    "train_samples_per_sec",
    "sec_per_epoch",
    "map50",
    "map50_95",
    "box_loss",
    "cls_loss",
    "dfl_loss",
    "notes",
]


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    out = {k: row.get(k, "") for k in METRIC_COLUMNS}
    out["branch"] = (out.get("branch") or "").strip()
    out["model"] = (out.get("model") or "").strip()
    return out


def merge_metrics_csv(
    yolo_csv: Path,
    mmdet_csv: Path,
    output_csv: Path,
    dedupe: bool,
) -> Dict[str, int]:
    yolo_rows = [normalize_row(r) for r in read_csv_rows(yolo_csv)]
    mmdet_rows = [normalize_row(r) for r in read_csv_rows(mmdet_csv)]
    merged = yolo_rows + mmdet_rows

    if dedupe:
        by_key: Dict[str, Dict[str, str]] = {}
        for r in merged:
            key = f"{r.get('branch','')}::{r.get('model','')}"
            by_key[key] = r
        merged = list(by_key.values())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(merged)

    return {
        "yolo_rows": len(yolo_rows),
        "mmdet_rows": len(mmdet_rows),
        "merged_rows": len(merged),
    }


def load_metrics_rows(metrics_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not metrics_csv.exists():
        return rows

    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "branch": r.get("branch"),
                    "model": r.get("model"),
                    "accuracy": safe_float(r.get("accuracy")),
                    "f1": safe_float(r.get("f1")),
                    "map50": safe_float(r.get("map50")),
                    "map50_95": safe_float(r.get("map50_95")),
                    "precision": safe_float(r.get("precision")),
                    "recall": safe_float(r.get("recall")),
                    "box_loss": safe_float(r.get("box_loss")),
                    "cls_loss": safe_float(r.get("cls_loss")),
                    "dfl_loss": safe_float(r.get("dfl_loss")),
                }
            )
    return rows


def plot_metric_bars(rows: List[Dict[str, object]], metric_key: str, title: str, out_path: Path) -> None:
    labels = []
    values = []
    for r in rows:
        val = r.get(metric_key)
        if val is None:
            continue
        labels.append(f"{r.get('branch')}:{r.get('model')}")
        values.append(val)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5.5))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_key)
    ax.tick_params(axis="x", rotation=35)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_precision_recall(rows: List[Dict[str, object]], out_path: Path) -> None:
    xs = []
    ys = []
    labels = []
    for r in rows:
        p = r.get("precision")
        rc = r.get("recall")
        if p is None or rc is None:
            continue
        xs.append(p)
        ys.append(rc)
        labels.append(f"{r.get('branch')}:{r.get('model')}")

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xs, ys, s=50, alpha=0.8)
    for x, y, lb in zip(xs, ys, labels):
        ax.annotate(lb, (x, y), fontsize=8, xytext=(5, 3), textcoords="offset points")
    ax.set_title("Precision vs Recall")
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_yolo_loss_curves(yolo_results_csv: Path, out_path: Path) -> None:
    if not yolo_results_csv.exists():
        return

    epochs = []
    box_loss = []
    cls_loss = []
    dfl_loss = []

    with yolo_results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            epochs.append(idx)
            box_loss.append(safe_float(row.get("train/box_loss")))
            cls_loss.append(safe_float(row.get("train/cls_loss")))
            dfl_loss.append(safe_float(row.get("train/dfl_loss")))

    if not epochs:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if any(v is not None for v in box_loss):
        ax.plot(epochs, [v if v is not None else float("nan") for v in box_loss], label="box_loss")
    if any(v is not None for v in cls_loss):
        ax.plot(epochs, [v if v is not None else float("nan") for v in cls_loss], label="cls_loss")
    if any(v is not None for v in dfl_loss):
        ax.plot(epochs, [v if v is not None else float("nan") for v in dfl_loss], label="dfl_loss")

    ax.set_title("YOLO Train Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training/validation performance dashboard")
    parser.add_argument("--metrics-csv", type=Path, default=Path("reports/metrics_all_models.csv"))
    parser.add_argument("--yolo-metrics-csv", type=Path, default=Path("reports/metrics_yolo.csv"))
    parser.add_argument("--mmdet-metrics-csv", type=Path, default=Path("reports/metrics_mmdet.csv"))
    parser.add_argument(
        "--disable-auto-merge",
        action="store_true",
        help="Disable automatic merge of YOLO/MMDet CSV into --metrics-csv",
    )
    parser.add_argument(
        "--disable-dedupe",
        action="store_true",
        help="Keep duplicate rows when auto-merging CSV files",
    )
    parser.add_argument("--yolo-results", type=Path, default=None, help="Path to YOLO results.csv (optional)")
    parser.add_argument("--yolo-confusion", type=Path, default=None, help="Path to YOLO confusion_matrix.png (optional)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = args.output.resolve() if args.output else (repo_root / "plots" / "performance")
    out_dir.mkdir(parents=True, exist_ok=True)

    merge_info = None
    metrics_csv_path = args.metrics_csv.resolve()
    if not args.disable_auto_merge:
        merge_info = merge_metrics_csv(
            yolo_csv=args.yolo_metrics_csv.resolve(),
            mmdet_csv=args.mmdet_metrics_csv.resolve(),
            output_csv=metrics_csv_path,
            dedupe=not args.disable_dedupe,
        )

    rows = load_metrics_rows(metrics_csv_path)
    if not rows:
        raise SystemExit(f"No rows found in metrics CSV: {metrics_csv_path}")

    plot_metric_bars(rows, "accuracy", "Accuracy (mAP50 proxy)", out_dir / "bar_accuracy.png")
    plot_metric_bars(rows, "f1", "F1-score", out_dir / "bar_f1.png")
    plot_metric_bars(rows, "map50", "mAP@50", out_dir / "bar_map50.png")
    plot_metric_bars(rows, "map50_95", "mAP@50-95", out_dir / "bar_map50_95.png")
    plot_precision_recall(rows, out_dir / "scatter_precision_recall.png")

    if args.yolo_results is not None:
        plot_yolo_loss_curves(args.yolo_results.resolve(), out_dir / "curve_yolo_losses.png")

    copied_conf = False
    if args.yolo_confusion is not None:
        copied_conf = copy_if_exists(args.yolo_confusion.resolve(), out_dir / "confusion_matrix_yolo.png")

    summary = {
        "metrics_csv": str(metrics_csv_path),
        "row_count": len(rows),
        "plots": sorted([p.name for p in out_dir.glob("*.png")]),
        "confusion_matrix_included": copied_conf,
        "auto_merge": not args.disable_auto_merge,
        "merge_info": merge_info,
        "note": "For MMDetection confusion matrix, provide external confusion image or add dedicated evaluator script.",
    }
    (out_dir / "performance_plot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved performance plots to: {out_dir}")


if __name__ == "__main__":
    main()
