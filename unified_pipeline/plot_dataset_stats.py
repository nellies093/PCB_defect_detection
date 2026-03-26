import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def load_summary(summary_path: Path):
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_coco(coco_path: Path):
    return json.loads(coco_path.read_text(encoding="utf-8"))


def merge_coco_splits(split_paths: list[Path]) -> dict:
    merged = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    categories_set = set()
    for p in split_paths:
        data = load_coco(p)
        merged["images"].extend(data.get("images", []))
        merged["annotations"].extend(data.get("annotations", []))
        for cat in data.get("categories", []):
            key = (cat.get("id"), cat.get("name"))
            if key not in categories_set:
                categories_set.add(key)
                merged["categories"].append(cat)

    return merged


def resolve_dataset_dir(dataset_arg: Path | None) -> Path:
    if dataset_arg is not None:
        return dataset_arg.resolve()

    script_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    candidates = [
        cwd / "dataset_unified_6sources",
        cwd / "data",
        script_root / "dataset_unified_6sources",
        script_root / "data",
    ]

    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    return script_root


def resolve_summary_path(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / "dataset_summary.json",
        dataset_dir / "dataset_info.json",
        dataset_dir / "summary.json",
        dataset_dir.parent / "dataset_info.json",
        dataset_dir.parent / "summary.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find summary JSON from dataset directory: {dataset_dir}. "
        "Expected one of: dataset_summary.json, dataset_info.json, summary.json."
    )


def resolve_coco_data(dataset_dir: Path) -> dict:
    single_candidates = [
        dataset_dir / "annotations" / "json" / "annotations_all.json",
        dataset_dir / "annotations_json" / "annotations_all.json",
    ]
    for p in single_candidates:
        if p.exists():
            return load_coco(p)

    split_base_dirs = [
        dataset_dir / "annotations_json",
        dataset_dir / "annotations" / "json",
    ]
    for base in split_base_dirs:
        split_paths = [base / "train.json", base / "val.json", base / "test.json"]
        if all(p.exists() for p in split_paths):
            return merge_coco_splits(split_paths)

    raise FileNotFoundError(
        f"Cannot find COCO JSON under dataset directory: {dataset_dir}. "
        "Expected annotations_all.json or train/val/test json files."
    )


def plot_bar(summary: dict, out_dir: Path) -> None:
    counts = summary.get("image_count_per_defect", {}) or summary.get("image_count_per_class", {})
    if not counts and summary.get("classes"):
        counts = {name: 0 for name in summary["classes"]}

    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values)
    ax.set_title("So luong anh theo tung loi")
    ax.set_xlabel("Loai loi")
    ax.set_ylabel("So luong anh")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "bar_images_per_defect.png", dpi=180)
    plt.close(fig)


def plot_split_counts(summary: dict, out_dir: Path) -> None:
    counts = summary.get("split_counts", {}) or summary.get("split_image_counts", {})
    if not counts:
        return

    labels = ["train", "val", "test"]
    values = [int(counts.get(k, 0)) for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values)
    ax.set_title("So luong anh theo split")
    ax.set_xlabel("Split")
    ax.set_ylabel("So luong anh")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "split_image_counts_bar.png", dpi=180)
    plt.close(fig)


def plot_bbox_scatter(coco: dict, out_dir: Path) -> None:
    widths = []
    heights = []

    for ann in coco.get("annotations", []):
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            widths.append(bbox[2])
            heights.append(bbox[3])

    plt.figure(figsize=(8, 6))
    plt.scatter(widths, heights, s=10, alpha=0.35)
    plt.title("Phan tan chieu rong va chieu cao bounding box")
    plt.xlabel("Bounding box width")
    plt.ylabel("Bounding box height")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_bbox_wh.png", dpi=180)
    plt.close()


def plot_bbox_hexbin(coco: dict, out_dir: Path) -> None:
    widths = []
    heights = []

    for ann in coco.get("annotations", []):
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) >= 4:
            widths.append(bbox[2])
            heights.append(bbox[3])

    if not widths:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(widths, heights, gridsize=45, mincnt=1, cmap="viridis")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("So luong bbox")
    ax.set_title("Mat do chieu rong va chieu cao bounding box")
    ax.set_xlabel("Bounding box width")
    ax.set_ylabel("Bounding box height")
    fig.tight_layout()
    fig.savefig(out_dir / "bbox_hexbin_wh.png", dpi=180)
    plt.close(fig)


def plot_image_size_distribution(coco: dict, out_dir: Path) -> None:
    size_counts = Counter()
    for img in coco.get("images", []):
        w = img.get("width")
        h = img.get("height")
        if w and h:
            size_counts[f"{w}x{h}"] += 1

    if not size_counts:
        return

    top_items = size_counts.most_common(15)
    labels = [k for k, _ in top_items]
    values = [v for _, v in top_items]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), 5))
    bars = ax.bar(labels, values)
    ax.set_title("Phan bo kich thuoc anh (top 15)")
    ax.set_xlabel("Kich thuoc anh")
    ax.set_ylabel("So luong anh")
    ax.tick_params(axis="x", rotation=35)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "image_size_top15_bar.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dataset statistics for unified dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to dataset folder. If omitted, auto-detects dataset_unified_6sources or data folder.",
    )
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(args.dataset)
    summary_path = resolve_summary_path(dataset_dir)
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    coco = resolve_coco_data(dataset_dir)

    plot_bar(summary, out_dir)
    plot_split_counts(summary, out_dir)
    plot_bbox_scatter(coco, out_dir)
    plot_bbox_hexbin(coco, out_dir)
    plot_image_size_distribution(coco, out_dir)

    print(f"Dataset dir: {dataset_dir}")
    print(f"Summary file: {summary_path}")
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
