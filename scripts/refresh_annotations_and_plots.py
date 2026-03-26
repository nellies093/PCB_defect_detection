from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    raise FileNotFoundError(f"Missing labels directory in: {split_dir} (expected labels or labels_txt)")


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])


def yolo_to_coco_bbox(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    abs_w = max(0.0, bw * width)
    abs_h = max(0.0, bh * height)
    x = xc * width - abs_w / 2.0
    y = yc * height - abs_h / 2.0
    x = max(0.0, min(x, width - 1))
    y = max(0.0, min(y, height - 1))
    if x + abs_w > width:
        abs_w = max(0.0, width - x)
    if y + abs_h > height:
        abs_h = max(0.0, height - y)
    return x, y, abs_w, abs_h


def parse_yolo_line(line: str) -> Tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cls = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])
        return cls, xc, yc, bw, bh
    except ValueError:
        return None


def build_coco_for_split(
    split: str,
    data_root: Path,
    class_names: Sequence[str],
) -> Tuple[Dict, List[Tuple[float, float]], Counter]:
    split_dir = data_root / split
    images_dir = split_dir / "images"
    labels_dir = find_labels_dir(split_dir)

    images = []
    annotations = []
    bbox_scatter_points: List[Tuple[float, float]] = []
    image_count_per_class: Counter = Counter()
    ann_id = 1

    image_paths = list_images(images_dir)
    for img_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as img:
            width, height = img.size

        images.append(
            {
                "id": img_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        seen_classes_in_image = set()
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parsed = parse_yolo_line(raw_line)
            if parsed is None:
                continue
            cls, xc, yc, bw, bh = parsed
            if cls < 0 or cls >= len(class_names):
                continue

            x, y, abs_w, abs_h = yolo_to_coco_bbox(xc, yc, bw, bh, width, height)
            if abs_w <= 0 or abs_h <= 0:
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls + 1,  # COCO category ids are 1-based in this project
                    "bbox": [round(x, 2), round(y, 2), round(abs_w, 2), round(abs_h, 2)],
                    "area": round(abs_w * abs_h, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            bbox_scatter_points.append((bw, bh))
            seen_classes_in_image.add(cls)

        for cls in seen_classes_in_image:
            image_count_per_class[cls] += 1

    categories = [
        {"id": i + 1, "name": name, "supercategory": "pcb_defect"}
        for i, name in enumerate(class_names)
    ]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    return coco, bbox_scatter_points, image_count_per_class


def plot_defect_image_count_bar(class_names: Sequence[str], counts: Sequence[int], out_path: Path) -> None:
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


def plot_bbox_size_scatter(points: Sequence[Tuple[float, float]], out_path: Path) -> None:
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


def plot_image_size_count_bar(data_root: Path, out_path: Path) -> None:
    size_counts: Counter = Counter()
    for split in ("train", "val", "test"):
        for image_path in list_images(data_root / split / "images"):
            with Image.open(image_path) as img:
                w, h = img.size
            size_counts[f"{w}x{h}"] += 1

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
    parser = argparse.ArgumentParser(
        description="Refresh COCO annotations from YOLO labels and generate dataset plots."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing classes.txt and data/",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    data_root = root / "data"
    ann_dir = data_root / "annotations_json"
    plot_dir = root / "plots"
    ann_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    class_names = read_class_names(root / "classes.txt")

    overall_scatter: List[Tuple[float, float]] = []
    overall_img_count_per_class: Counter = Counter()

    for split in ("train", "val", "test"):
        coco, split_points, split_counts = build_coco_for_split(split, data_root, class_names)
        (ann_dir / f"{split}.json").write_text(json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8")
        overall_scatter.extend(split_points)
        overall_img_count_per_class.update(split_counts)

    class_image_counts = [overall_img_count_per_class.get(i, 0) for i in range(len(class_names))]
    plot_defect_image_count_bar(class_names, class_image_counts, plot_dir / "defect_image_count_bar.png")
    plot_bbox_size_scatter(overall_scatter, plot_dir / "bbox_size_scatter.png")
    plot_image_size_count_bar(data_root, plot_dir / "image_size_count_bar.png")

    print(f"Updated annotations: {ann_dir / 'train.json'}")
    print(f"Updated annotations: {ann_dir / 'val.json'}")
    print(f"Updated annotations: {ann_dir / 'test.json'}")
    print(f"Saved plot: {plot_dir / 'defect_image_count_bar.png'}")
    print(f"Saved plot: {plot_dir / 'bbox_size_scatter.png'}")
    print(f"Saved plot: {plot_dir / 'image_size_count_bar.png'}")


if __name__ == "__main__":
    main()
