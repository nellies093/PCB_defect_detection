import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Keep folder names exactly as requested by user.
DEFECTS = [
    "mouse_bite",
    "open_circuit",
    "mising_hole",
    "short",
    "spur",
    "spurious_copper",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(DEFECTS)}


NAME_ALIASES = {
    "mousebite": "mouse_bite",
    "mouse_bite": "mouse_bite",
    "open": "open_circuit",
    "open-circuit": "open_circuit",
    "open_circuit": "open_circuit",
    "pin-hole": "mising_hole",
    "pinhole": "mising_hole",
    "missing_hole": "mising_hole",
    "mising_hole": "mising_hole",
    "short": "short",
    "spur": "spur",
    "copper": "spurious_copper",
    "spurious": "spurious_copper",
    "spurious-copper": "spurious_copper",
    "spurious_copper": "spurious_copper",
}


# DsPCBDS+ category abbreviations found in COCO annotations.
DSPCBDS_CATEGORY_MAP = {
    "SH": "short",
    "SP": "spur",
    "SC": "spurious_copper",
    "OP": "open_circuit",
    "MB": "mouse_bite",
    "HB": "mising_hole",
}

# For DsPCBDS+ YOLO labels, ids are expected to follow SH,SP,SC,OP,MB,HB,CS,CFO,BMFO.
DSPCBDS_YOLO_ID_MAP = {
    0: "short",
    1: "spur",
    2: "spurious_copper",
    3: "open_circuit",
    4: "mouse_bite",
    5: "mising_hole",
}


@dataclass
class Box:
    cls_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class ImageRecord:
    source: str
    image_path: Path
    width: int
    height: int
    boxes: List[Box]


def normalize_class(name: str) -> Optional[str]:
    key = name.strip().lower().replace(" ", "_")
    return NAME_ALIASES.get(key)


def clamp_box(xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    xmin = max(0.0, min(float(width - 1), xmin))
    ymin = max(0.0, min(float(height - 1), ymin))
    xmax = max(0.0, min(float(width), xmax))
    ymax = max(0.0, min(float(height), ymax))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def parse_voc_xml(xml_path: Path) -> Optional[Tuple[int, int, List[Box]]]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None

    size = root.find("size")
    if size is None:
        return None

    try:
        width = int(float(size.findtext("width", default="0")))
        height = int(float(size.findtext("height", default="0")))
    except ValueError:
        return None

    if width <= 0 or height <= 0:
        return None

    boxes: List[Box] = []
    for obj in root.findall("object"):
        raw_name = obj.findtext("name", default="")
        cls_name = normalize_class(raw_name)
        if cls_name is None:
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        try:
            xmin = float(bnd.findtext("xmin", default="0"))
            ymin = float(bnd.findtext("ymin", default="0"))
            xmax = float(bnd.findtext("xmax", default="0"))
            ymax = float(bnd.findtext("ymax", default="0"))
        except ValueError:
            continue

        clamped = clamp_box(xmin, ymin, xmax, ymax, width, height)
        if clamped is None:
            continue
        boxes.append(Box(cls_name=cls_name, xmin=clamped[0], ymin=clamped[1], xmax=clamped[2], ymax=clamped[3]))

    return width, height, boxes


def parse_yolo_txt(txt_path: Path, width: int, height: int, id_map: Optional[Dict[int, str]] = None) -> List[Box]:
    boxes: List[Box] = []
    if not txt_path.exists():
        return boxes

    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except ValueError:
            continue

        if id_map is not None:
            cls_name = id_map.get(cls_id)
        else:
            if cls_id < 0 or cls_id >= len(DEFECTS):
                cls_name = None
            else:
                cls_name = DEFECTS[cls_id]

        if cls_name is None:
            continue

        xmin = (cx - bw / 2.0) * width
        ymin = (cy - bh / 2.0) * height
        xmax = (cx + bw / 2.0) * width
        ymax = (cy + bh / 2.0) * height

        clamped = clamp_box(xmin, ymin, xmax, ymax, width, height)
        if clamped is None:
            continue
        boxes.append(Box(cls_name=cls_name, xmin=clamped[0], ymin=clamped[1], xmax=clamped[2], ymax=clamped[3]))

    return boxes


def get_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None


def collect_deeppcb(root: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for split in ["train", "val", "test"]:
        img_dir = root / "DeepPCB" / "dataset" / split / "images"
        ann_dir = root / "DeepPCB" / "dataset" / split / "annotations"
        if not img_dir.exists() or not ann_dir.exists():
            continue

        for xml_path in ann_dir.glob("*.xml"):
            image_name = xml_path.stem + ".jpg"
            image_path = img_dir / image_name
            if not image_path.exists():
                continue
            parsed = parse_voc_xml(xml_path)
            if parsed is None:
                continue
            width, height, boxes = parsed
            if not boxes:
                continue
            records.append(ImageRecord(source="DeepPCB", image_path=image_path, width=width, height=height, boxes=boxes))
    return records


def collect_dspcbds_yolo(root: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    base = root / "DsPCBDS+" / "Data_YOLO"
    for split in ["train", "val"]:
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_path in img_dir.glob("*.jpg"):
            txt_path = lbl_dir / f"{img_path.stem}.txt"
            size = get_image_size(img_path)
            if size is None:
                continue
            width, height = size
            boxes = parse_yolo_txt(txt_path, width=width, height=height, id_map=DSPCBDS_YOLO_ID_MAP)
            if not boxes:
                continue
            records.append(ImageRecord(source="DsPCBDS+", image_path=img_path, width=width, height=height, boxes=boxes))
    return records


def collect_hripcb(root: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    base = root / "HRIPCB_UPDATE"
    # This dataset in this workspace only has images and no labels for many files.
    for split in ["train", "val", "test"]:
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_path in img_dir.glob("*.jpg"):
            txt_path = lbl_dir / f"{img_path.stem}.txt"
            if not txt_path.exists():
                continue
            size = get_image_size(img_path)
            if size is None:
                continue
            width, height = size
            boxes = parse_yolo_txt(txt_path, width=width, height=height)
            if not boxes:
                continue
            records.append(ImageRecord(source="HRIPCB_UPDATE", image_path=img_path, width=width, height=height, boxes=boxes))
    return records


def collect_pcb_dataset_xml(root: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    img_base = root / "PCB_DATASET" / "images"
    ann_base = root / "PCB_DATASET" / "Annotations"
    if not img_base.exists() or not ann_base.exists():
        return records

    for xml_path in ann_base.glob("*/*.xml"):
        class_dir = xml_path.parent.name
        cls_guess = normalize_class(class_dir)
        parsed = parse_voc_xml(xml_path)
        if parsed is None:
            continue
        width, height, boxes = parsed

        if not boxes and cls_guess is not None:
            # Fallback for files missing object tags.
            pass

        stem = xml_path.stem
        img_dir = img_base / class_dir
        image_candidates = [img_dir / f"{stem}.jpg", img_dir / f"{stem}.png", img_dir / f"{stem}.jpeg"]
        image_path = next((p for p in image_candidates if p.exists()), None)
        if image_path is None:
            continue

        if not boxes and cls_guess is not None:
            continue

        if boxes:
            records.append(ImageRecord(source="PCB_DATASET", image_path=image_path, width=width, height=height, boxes=boxes))
    return records


def collect_pcb_defect_dataset(root: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    base = root / "pcb-defect-dataset" / "img"
    for split in ["train", "val", "test"]:
        img_root = base / split / "images"
        lbl_root = base / split / "labels"
        if not img_root.exists() or not lbl_root.exists():
            continue

        for class_dir in img_root.iterdir():
            if not class_dir.is_dir():
                continue
            cls_name = normalize_class(class_dir.name)
            if cls_name is None:
                continue
            label_dir_name = class_dir.name if (lbl_root / class_dir.name).exists() else "spurious"
            label_class_dir = lbl_root / label_dir_name
            for img_path in class_dir.glob("*.jpg"):
                txt_path = label_class_dir / f"{img_path.stem}.txt"
                if not txt_path.exists():
                    continue
                size = get_image_size(img_path)
                if size is None:
                    continue
                width, height = size
                boxes = parse_yolo_txt(txt_path, width=width, height=height)
                if not boxes:
                    continue
                records.append(ImageRecord(source="pcb-defect-dataset", image_path=img_path, width=width, height=height, boxes=boxes))
    return records


def collect_pku_pcb(root: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    base = root / "PKU_PCB"
    for split in ["train", "valid", "test"]:
        img_dir = base / split / "images"
        xml_dir = base / split / "Annotations_XML"
        lbl_dir = base / split / "labels"

        if xml_dir.exists():
            for xml_path in xml_dir.glob("*.xml"):
                parsed = parse_voc_xml(xml_path)
                if parsed is None:
                    continue
                width, height, boxes = parsed
                if not boxes:
                    continue
                image_name = xml_path.stem + ".jpg"
                image_path = img_dir / image_name
                if not image_path.exists():
                    continue
                records.append(ImageRecord(source="PKU_PCB", image_path=image_path, width=width, height=height, boxes=boxes))

        # Optional fallback to YOLO labels in test split.
        if img_dir.exists() and lbl_dir.exists():
            for img_path in img_dir.glob("*.jpg"):
                txt_path = lbl_dir / f"{img_path.stem}.txt"
                if not txt_path.exists():
                    continue
                size = get_image_size(img_path)
                if size is None:
                    continue
                width, height = size
                boxes = parse_yolo_txt(txt_path, width=width, height=height)
                if not boxes:
                    continue
                records.append(ImageRecord(source="PKU_PCB", image_path=img_path, width=width, height=height, boxes=boxes))
    return records


def dedupe_records(records: List[ImageRecord]) -> List[ImageRecord]:
    # Keep record with more boxes when duplicate image path appears.
    best: Dict[str, ImageRecord] = {}
    for r in records:
        key = str(r.image_path.resolve())
        old = best.get(key)
        if old is None or len(r.boxes) > len(old.boxes):
            best[key] = r
    return list(best.values())


def ensure_output_tree(out_dir: Path) -> None:
    for split in ["train", "val", "test"]:
        for defect in DEFECTS:
            (out_dir / split / "images" / defect).mkdir(parents=True, exist_ok=True)
            (out_dir / split / "annotations" / "txt" / defect).mkdir(parents=True, exist_ok=True)
            (out_dir / split / "annotations" / "xml" / defect).mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations" / "json").mkdir(parents=True, exist_ok=True)
    (out_dir / "coco_annotations").mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (out_dir / "yolo_format" / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "yolo_format" / split / "labels").mkdir(parents=True, exist_ok=True)


def write_txt_annotation(path: Path, boxes: List[Box], width: int, height: int, class_filter: Optional[str] = None) -> None:
    lines: List[str] = []
    for b in boxes:
        if class_filter is not None and b.cls_name != class_filter:
            continue
        cls_id = CLASS_TO_ID[b.cls_name]
        cx = ((b.xmin + b.xmax) / 2.0) / width
        cy = ((b.ymin + b.ymax) / 2.0) / height
        bw = (b.xmax - b.xmin) / width
        bh = (b.ymax - b.ymin) / height
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_xml_annotation(path: Path, image_filename: str, width: int, height: int, boxes: List[Box], class_filter: Optional[str] = None) -> None:
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = "images"
    ET.SubElement(ann, "filename").text = image_filename

    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    for b in boxes:
        if class_filter is not None and b.cls_name != class_filter:
            continue
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = b.cls_name
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(round(b.xmin)))
        ET.SubElement(bnd, "ymin").text = str(int(round(b.ymin)))
        ET.SubElement(bnd, "xmax").text = str(int(round(b.xmax)))
        ET.SubElement(bnd, "ymax").text = str(int(round(b.ymax)))

    tree = ET.ElementTree(ann)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def split_records(records: List[ImageRecord], seed: int = 42) -> Dict[str, List[ImageRecord]]:
    random.seed(seed)
    shuffled = records[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def build_dataset(root: Path, out_dir: Path, seed: int) -> None:
    ensure_output_tree(out_dir)

    all_records: List[ImageRecord] = []
    all_records.extend(collect_deeppcb(root))
    all_records.extend(collect_dspcbds_yolo(root))
    all_records.extend(collect_hripcb(root))
    all_records.extend(collect_pcb_dataset_xml(root))
    all_records.extend(collect_pcb_defect_dataset(root))
    all_records.extend(collect_pku_pcb(root))

    all_records = dedupe_records(all_records)
    all_records = [r for r in all_records if any(b.cls_name in CLASS_TO_ID for b in r.boxes)]

    splits = split_records(all_records, seed=seed)

    coco_global = {
        "info": {"description": "Unified PCB dataset from 6 sources", "version": "1.0"},
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": c} for i, c in enumerate(DEFECTS)],
    }

    next_image_id = 1
    next_ann_id = 1
    per_defect_image_count = defaultdict(int)

    split_coco = {
        split: {
            "info": {"description": f"Unified PCB {split}"},
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": c} for i, c in enumerate(DEFECTS)],
        }
        for split in ["train", "val", "test"]
    }

    for split_name, split_records_list in splits.items():
        for idx, rec in enumerate(split_records_list):
            unique_name = f"{split_name}_{idx:07d}_{rec.source}_{rec.image_path.name}"
            yolo_image_path = out_dir / "yolo_format" / split_name / "images" / unique_name
            shutil.copy2(rec.image_path, yolo_image_path)

            all_txt_path = out_dir / "yolo_format" / split_name / "labels" / f"{Path(unique_name).stem}.txt"
            write_txt_annotation(all_txt_path, rec.boxes, rec.width, rec.height, class_filter=None)

            image_id = next_image_id
            next_image_id += 1

            coco_image = {
                "id": image_id,
                "file_name": unique_name,
                "width": rec.width,
                "height": rec.height,
                "source": rec.source,
                "split": split_name,
            }
            coco_global["images"].append(coco_image)
            split_coco[split_name]["images"].append(coco_image)

            present_classes = {b.cls_name for b in rec.boxes}
            for cls_name in present_classes:
                per_defect_image_count[cls_name] += 1
                img_target = out_dir / split_name / "images" / cls_name / unique_name
                shutil.copy2(rec.image_path, img_target)

                txt_target = out_dir / split_name / "annotations" / "txt" / cls_name / f"{Path(unique_name).stem}.txt"
                xml_target = out_dir / split_name / "annotations" / "xml" / cls_name / f"{Path(unique_name).stem}.xml"
                write_txt_annotation(txt_target, rec.boxes, rec.width, rec.height, class_filter=cls_name)
                write_xml_annotation(xml_target, unique_name, rec.width, rec.height, rec.boxes, class_filter=cls_name)

            for b in rec.boxes:
                cat_id = CLASS_TO_ID[b.cls_name]
                bbox_w = b.xmax - b.xmin
                bbox_h = b.ymax - b.ymin
                ann = {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [round(b.xmin, 3), round(b.ymin, 3), round(bbox_w, 3), round(bbox_h, 3)],
                    "area": round(bbox_w * bbox_h, 3),
                    "iscrowd": 0,
                }
                next_ann_id += 1
                coco_global["annotations"].append(ann)
                split_coco[split_name]["annotations"].append(ann)

    (out_dir / "annotations" / "json" / "annotations_all.json").write_text(
        json.dumps(coco_global, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for split_name in ["train", "val", "test"]:
        (out_dir / "coco_annotations" / f"{split_name}.json").write_text(
            json.dumps(split_coco[split_name], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    yolo_yaml = {
        "path": str((out_dir / "yolo_format").resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(DEFECTS)},
        "nc": len(DEFECTS),
    }
    (out_dir / "yolo_format" / "data.yaml").write_text(json.dumps(yolo_yaml, indent=2), encoding="utf-8")

    summary = {
        "total_images": len(coco_global["images"]),
        "total_annotations": len(coco_global["annotations"]),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "image_count_per_defect": dict(sorted(per_defect_image_count.items())),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified PCB dataset from 6 source datasets.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Workspace root that contains all dataset folders.")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "dataset_unified_6sources", help="Output dataset folder.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for 80/10/10 split.")
    args = parser.parse_args()

    build_dataset(root=args.root, out_dir=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
