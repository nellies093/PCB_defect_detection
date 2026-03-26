import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from build_unified_dataset import CLASS_TO_ID, DEFECTS
from build_unified_dataset import collect_deeppcb
from build_unified_dataset import collect_dspcbds_yolo
from build_unified_dataset import collect_hripcb
from build_unified_dataset import collect_pcb_dataset_xml
from build_unified_dataset import collect_pcb_defect_dataset
from build_unified_dataset import collect_pku_pcb


def gather_by_source(root: Path) -> Dict[str, list]:
    return {
        "DeepPCB": collect_deeppcb(root),
        "DsPCBDS+": collect_dspcbds_yolo(root),
        "HRIPCB_UPDATE": collect_hripcb(root),
        "PCB_DATASET": collect_pcb_dataset_xml(root),
        "pcb-defect-dataset": collect_pcb_defect_dataset(root),
        "PKU_PCB": collect_pku_pcb(root),
    }


def per_source_stats(records: list) -> Dict[str, object]:
    image_count = len(records)
    total_boxes = 0
    image_count_per_class = Counter()
    box_count_per_class = Counter()
    bbox_wh_points = []
    image_size_counter = Counter()

    for rec in records:
        image_size_counter[f"{rec.width}x{rec.height}"] += 1
        classes_in_image = set()
        for b in rec.boxes:
            total_boxes += 1
            box_count_per_class[b.cls_name] += 1
            classes_in_image.add(b.cls_name)
            bbox_wh_points.append((b.xmax - b.xmin, b.ymax - b.ymin))

        for cls_name in classes_in_image:
            image_count_per_class[cls_name] += 1

    bbox_areas = [w * h for w, h in bbox_wh_points]

    return {
        "images": image_count,
        "boxes": total_boxes,
        "avg_boxes_per_image": (float(total_boxes) / image_count) if image_count else 0.0,
        "image_count_per_class": {c: int(image_count_per_class.get(c, 0)) for c in DEFECTS},
        "box_count_per_class": {c: int(box_count_per_class.get(c, 0)) for c in DEFECTS},
        "bbox_summary": {
            "count": len(bbox_wh_points),
            "width_mean": mean([x[0] for x in bbox_wh_points]) if bbox_wh_points else None,
            "height_mean": mean([x[1] for x in bbox_wh_points]) if bbox_wh_points else None,
            "area_mean": mean(bbox_areas) if bbox_areas else None,
            "area_p90": sorted(bbox_areas)[int(0.9 * (len(bbox_areas) - 1))] if bbox_areas else None,
        },
        "image_size_top10": dict(image_size_counter.most_common(10)),
        "_bbox_points": bbox_wh_points,
        "_image_size_counter": image_size_counter,
    }


def plot_class_distribution(source_name: str, stats: Dict[str, object], out_dir: Path) -> None:
    labels = DEFECTS
    values = [stats["image_count_per_class"].get(c, 0) for c in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values)
    ax.set_title(f"{source_name}: image count per class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, str(val), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "class_image_count_bar.png", dpi=220)
    plt.close(fig)


def plot_bbox_scatter(source_name: str, stats: Dict[str, object], out_dir: Path) -> None:
    points: List[Tuple[float, float]] = stats["_bbox_points"]
    fig, ax = plt.subplots(figsize=(8, 6))
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=8, alpha=0.35)
    ax.set_title(f"{source_name}: bbox width-height scatter")
    ax.set_xlabel("BBox width (pixel)")
    ax.set_ylabel("BBox height (pixel)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "bbox_scatter_wh.png", dpi=220)
    plt.close(fig)


def plot_image_size_distribution(source_name: str, stats: Dict[str, object], out_dir: Path) -> None:
    counter: Counter = stats["_image_size_counter"]
    if not counter:
        return

    top_items = counter.most_common(15)
    labels = [x[0] for x in top_items]
    values = [x[1] for x in top_items]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.65), 5.5))
    bars = ax.bar(labels, values)
    ax.set_title(f"{source_name}: image-size distribution (top 15)")
    ax.set_xlabel("Image size (W x H)")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=35)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, str(val), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "image_size_top15_bar.png", dpi=220)
    plt.close(fig)


def build_mapping_notes() -> Dict[str, object]:
    ds9_to_6 = {
        "SH": "short",
        "SP": "spur",
        "SC": "spurious_copper",
        "OP": "open_circuit",
        "MB": "mouse_bite",
        "HB": "mising_hole",
    }
    dropped = ["CS", "CFO", "BMFO"]

    six_label_alias = {
        "open": "open_circuit",
        "open-circuit": "open_circuit",
        "mousebite": "mouse_bite",
        "pin-hole": "mising_hole",
        "missing_hole": "mising_hole",
        "spurious": "spurious_copper",
        "spurious-copper": "spurious_copper",
    }

    return {
        "target_classes": DEFECTS,
        "dspcbds_9_to_6": ds9_to_6,
        "dspcbds_dropped_classes": dropped,
        "six_label_alias_normalization": six_label_alias,
        "note": "For 6-label datasets, classes are effectively 1-to-1 mapped after alias normalization.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-source dataset stats before merge")
    parser.add_argument("--root", type=Path, required=True, help="Workspace root containing source datasets")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for pre-merge stats (default: <repo-root>/plots/premerge_stats)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    out_root = args.output.resolve() if args.output else repo_root / "plots" / "premerge_stats"
    out_root.mkdir(parents=True, exist_ok=True)

    by_source = gather_by_source(root)

    merged_summary = {
        "root": str(root),
        "sources": {},
        "label_mapping": build_mapping_notes(),
    }

    for source_name, records in by_source.items():
        source_dir = out_root / source_name
        source_dir.mkdir(parents=True, exist_ok=True)

        stats = per_source_stats(records)
        plot_class_distribution(source_name, stats, source_dir)
        plot_bbox_scatter(source_name, stats, source_dir)
        plot_image_size_distribution(source_name, stats, source_dir)

        stats.pop("_bbox_points", None)
        stats.pop("_image_size_counter", None)
        merged_summary["sources"][source_name] = stats

    (out_root / "premerge_summary.json").write_text(
        json.dumps(merged_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved pre-merge stats at: {out_root}")


if __name__ == "__main__":
    main()
