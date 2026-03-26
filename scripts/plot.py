from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_class_names(classes_file: Path) -> List[str]:
    if not classes_file.exists():
        raise FileNotFoundError(f"Missing classes file: {classes_file}")
    return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_labels_dir(split_dir: Path) -> Path:
    labels_dir = split_dir / "labels"
    if labels_dir.exists():
        return labels_dir
    labels_txt_dir = split_dir / "labels_txt"
    if labels_txt_dir.exists():
        return labels_txt_dir
    raise FileNotFoundError(f"Missing labels dir in: {split_dir}")


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])


def collect_plot_data(dataset_root: Path, class_names: Sequence[str]) -> Tuple[List[int], List[Tuple[float, float]], Counter]:
    data_root = dataset_root / "data"
    image_count_per_class = [0 for _ in class_names]
    bbox_points: List[Tuple[float, float]] = []
    size_counts: Counter = Counter()

    for split in ("train", "val", "test"):
        split_dir = data_root / split
        images_dir = split_dir / "images"
        labels_dir = find_labels_dir(split_dir)

        for img_path in list_images(images_dir):
            with Image.open(img_path) as img:
                w, h = img.size
            size_counts[f"{w}x{h}"] += 1

            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            classes_in_image = set()
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    bw = float(parts[3])
                    bh = float(parts[4])
                except ValueError:
                    continue
                if 0 <= cls < len(class_names):
                    classes_in_image.add(cls)
                    bbox_points.append((bw, bh))

            for cls in classes_in_image:
                image_count_per_class[cls] += 1

    return image_count_per_class, bbox_points, size_counts


def plot_defect_bar(class_names: Sequence[str], counts: Sequence[int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, counts)
    ax.set_title("Image Count by Defect Class")
    ax.set_xlabel("Defect Class")
    ax.set_ylabel("Image Count")
    ax.tick_params(axis="x", rotation=25)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, val, str(val), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_bbox_scatter(points: Sequence[Tuple[float, float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=7, alpha=0.35)
    ax.set_title("BBox Size Scatter (YOLO normalized)")
    ax.set_xlabel("bbox width")
    ax.set_ylabel("bbox height")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_image_size_bar(size_counts: Counter, out_path: Path) -> None:
    ordered = sorted(size_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [x[0] for x in ordered]
    values = [x[1] for x in ordered]

    fig_width = max(10, min(24, len(labels) * 0.7))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    bars = ax.bar(labels, values)
    ax.set_title("Image Count by Image Size")
    ax.set_xlabel("Image Size (W x H)")
    ax.set_ylabel("Image Count")
    ax.tick_params(axis="x", rotation=40)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, str(val), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate only plots for PCB dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir.resolve() if args.out_dir else repo_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = read_class_names(dataset_root / "classes.txt")
    class_counts, bbox_points, size_counts = collect_plot_data(dataset_root, class_names)

    out1 = out_dir / "defect_image_count_bar.png"
    out2 = out_dir / "bbox_size_scatter.png"
    out3 = out_dir / "image_size_count_bar.png"

    plot_defect_bar(class_names, class_counts, out1)
    plot_bbox_scatter(bbox_points, out2)
    plot_image_size_bar(size_counts, out3)

    print(f"Saved plot: {out1}")
    print(f"Saved plot: {out2}")
    print(f"Saved plot: {out3}")


if __name__ == "__main__":
    main()
