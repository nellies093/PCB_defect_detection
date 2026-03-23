from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


ALL_MODELS = [
    "yolo11n",
    "retinanet",
    "faster_rcnn",
    "cascade_rcnn",
    "detr",
    "deformable_detr",
]


@dataclass
class ModelMetric:
    model: str
    map_50_95: Optional[float]
    map_50: Optional[float]
    run_dir: str
    source: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export mAP comparison table across trained PCB detection models."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("runs") / "pcb_train",
        help="Directory containing model run outputs",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        help="Models to include in the comparison",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV file path (default: <project-dir>/map_comparison.csv)",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown file path (default: <project-dir>/map_comparison.md)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any model metrics are missing",
    )
    return parser.parse_args()


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def choose_latest(paths: Iterable[Path]) -> Optional[Path]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def choose_latest_dir_with_prefix(parent: Path, prefix: str) -> Optional[Path]:
    if not parent.exists():
        return None
    candidates = [
        path for path in parent.iterdir() if path.is_dir() and path.name.startswith(prefix)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_last_csv_row(csv_path: Path) -> Optional[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        rows = [row for row in reader]
    if not rows:
        return None
    return rows[-1]


def find_first_float(row: dict[str, str], keys: list[str]) -> Optional[float]:
    for key in keys:
        if key in row:
            value = safe_float(row.get(key))
            if value is not None:
                return value
    return None


def parse_yolo_metrics(project_dir: Path) -> ModelMetric:
    yolo_run_dir = choose_latest_dir_with_prefix(project_dir, "yolo11n")
    if yolo_run_dir is None:
        return ModelMetric(
            model="yolo11n",
            map_50_95=None,
            map_50=None,
            run_dir="",
            source="",
            note="Run directory not found",
        )

    results_csv = yolo_run_dir / "results.csv"
    if not results_csv.exists():
        return ModelMetric(
            model="yolo11n",
            map_50_95=None,
            map_50=None,
            run_dir=str(yolo_run_dir),
            source="",
            note="results.csv not found",
        )

    last_row = read_last_csv_row(results_csv)
    if not last_row:
        return ModelMetric(
            model="yolo11n",
            map_50_95=None,
            map_50=None,
            run_dir=str(yolo_run_dir),
            source=str(results_csv),
            note="results.csv has no rows",
        )

    map_50_95 = find_first_float(
        last_row,
        [
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95",
            "mAP50-95",
        ],
    )
    map_50 = find_first_float(
        last_row,
        [
            "metrics/mAP50(B)",
            "metrics/mAP50",
            "mAP50",
        ],
    )

    note = "ok" if map_50_95 is not None or map_50 is not None else "mAP columns missing"
    return ModelMetric(
        model="yolo11n",
        map_50_95=map_50_95,
        map_50=map_50,
        run_dir=str(yolo_run_dir),
        source=str(results_csv),
        note=note,
    )


def extract_latest_mmdet_map(log_path: Path) -> tuple[Optional[float], Optional[float]]:
    latest_step = -1
    latest_map_50_95: Optional[float] = None
    latest_map_50: Optional[float] = None

    with log_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue

            step = int(payload.get("epoch", payload.get("iter", -1)))
            map_50_95 = safe_float(
                payload.get("coco/bbox_mAP", payload.get("bbox_mAP"))
            )
            map_50 = safe_float(
                payload.get("coco/bbox_mAP_50", payload.get("bbox_mAP_50"))
            )

            if map_50_95 is None and map_50 is None:
                continue

            if step >= latest_step:
                latest_step = step
                if map_50_95 is not None:
                    latest_map_50_95 = map_50_95
                if map_50 is not None:
                    latest_map_50 = map_50

    return latest_map_50_95, latest_map_50


def parse_mmdet_metrics(project_dir: Path, model_name: str) -> ModelMetric:
    model_dir = project_dir / model_name
    if not model_dir.exists():
        return ModelMetric(
            model=model_name,
            map_50_95=None,
            map_50=None,
            run_dir="",
            source="",
            note="Run directory not found",
        )

    candidates = []
    vis_scalars = model_dir / "vis_data" / "scalars.json"
    if vis_scalars.exists():
        candidates.append(vis_scalars)
    candidates.extend(model_dir.glob("*.log.json"))

    log_file = choose_latest(candidates)
    if log_file is None:
        return ModelMetric(
            model=model_name,
            map_50_95=None,
            map_50=None,
            run_dir=str(model_dir),
            source="",
            note="No MMDetection json logs found",
        )

    map_50_95, map_50 = extract_latest_mmdet_map(log_file)
    note = "ok" if map_50_95 is not None or map_50 is not None else "mAP keys not found"
    return ModelMetric(
        model=model_name,
        map_50_95=map_50_95,
        map_50=map_50,
        run_dir=str(model_dir),
        source=str(log_file),
        note=note,
    )


def parse_torchvision_metrics(project_dir: Path, model_name: str) -> ModelMetric | None:
    tv_dir = project_dir / f"torchvision_{model_name}"
    summary_path = tv_dir / "summary.json"
    metrics_path = tv_dir / "metrics.json"

    if not summary_path.exists() and not metrics_path.exists():
        return None

    map_50_95: Optional[float] = None
    map_50: Optional[float] = None
    note = "ok"
    source = ""

    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                last = payload[-1]
                map_50_95 = safe_float(last.get("map_50_95"))
                map_50 = safe_float(last.get("map_50"))
                source = str(metrics_path)
        except json.JSONDecodeError:
            note = "metrics.json invalid JSON"

    if map_50_95 is None and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            map_50_95 = safe_float(summary.get("best_map_50_95"))
            source = str(summary_path)
        except json.JSONDecodeError:
            note = "summary.json invalid JSON"

    if map_50_95 is None and map_50 is None and note == "ok":
        note = "mAP keys not found"

    return ModelMetric(
        model=model_name,
        map_50_95=map_50_95,
        map_50=map_50,
        run_dir=str(tv_dir),
        source=source,
        note=note,
    )


def parse_transformers_metrics(project_dir: Path, model_name: str) -> ModelMetric | None:
    tf_dir = project_dir / f"transformers_{model_name}"
    summary_path = tf_dir / "summary.json"
    metrics_path = tf_dir / "metrics.json"

    if not summary_path.exists() and not metrics_path.exists():
        return None

    map_50_95: Optional[float] = None
    map_50: Optional[float] = None
    note = "ok"
    source = ""

    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                last = payload[-1]
                map_50_95 = safe_float(last.get("map_50_95"))
                map_50 = safe_float(last.get("map_50"))
                source = str(metrics_path)
        except json.JSONDecodeError:
            note = "metrics.json invalid JSON"

    if map_50_95 is None and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            map_50_95 = safe_float(summary.get("best_map_50_95"))
            source = str(summary_path)
        except json.JSONDecodeError:
            note = "summary.json invalid JSON"

    if map_50_95 is None and map_50 is None and note == "ok":
        note = "mAP keys not found"

    return ModelMetric(
        model=model_name,
        map_50_95=map_50_95,
        map_50=map_50,
        run_dir=str(tf_dir),
        source=source,
        note=note,
    )


def metric_sort_key(item: ModelMetric) -> tuple[int, float]:
    if item.map_50_95 is None:
        return (1, float("inf"))
    return (0, -item.map_50_95)


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def write_csv(output_csv: Path, rows: list[ModelMetric]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["model", "mAP50-95", "mAP50", "run_dir", "source", "note"])
        for row in rows:
            writer.writerow(
                [
                    row.model,
                    "" if row.map_50_95 is None else f"{row.map_50_95:.6f}",
                    "" if row.map_50 is None else f"{row.map_50:.6f}",
                    row.run_dir,
                    row.source,
                    row.note,
                ]
            )


def write_markdown(output_md: Path, rows: list[ModelMetric]) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# mAP Comparison",
        "",
        "| Model | mAP50-95 | mAP50 | Note |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.model} | {format_metric(row.map_50_95)} | {format_metric(row.map_50)} | {row.note} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console_table(rows: list[ModelMetric]) -> None:
    print("\nModel comparison (sorted by mAP50-95):")
    print(f"{'Model':<18} {'mAP50-95':>10} {'mAP50':>10}  Note")
    print("-" * 60)
    for row in rows:
        print(
            f"{row.model:<18} {format_metric(row.map_50_95):>10} {format_metric(row.map_50):>10}  {row.note}"
        )


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    output_csv = args.output_csv.resolve() if args.output_csv else project_dir / "map_comparison.csv"
    output_md = args.output_md.resolve() if args.output_md else project_dir / "map_comparison.md"

    rows: list[ModelMetric] = []
    for model_name in args.models:
        if model_name == "yolo11n":
            rows.append(parse_yolo_metrics(project_dir))
        else:
            tf_metric = parse_transformers_metrics(project_dir, model_name)
            if tf_metric is not None:
                rows.append(tf_metric)
                continue
            tv_metric = parse_torchvision_metrics(project_dir, model_name)
            if tv_metric is not None:
                rows.append(tv_metric)
            else:
                rows.append(parse_mmdet_metrics(project_dir, model_name))

    rows.sort(key=metric_sort_key)

    write_csv(output_csv, rows)
    write_markdown(output_md, rows)
    print_console_table(rows)
    print(f"\nSaved CSV: {output_csv}")
    print(f"Saved Markdown: {output_md}")

    has_missing = any(row.map_50_95 is None and row.map_50 is None for row in rows)
    if args.strict and has_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
