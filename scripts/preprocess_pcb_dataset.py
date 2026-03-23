from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class PipelineConfig:
	target_size: int = 640
	apply_clahe: bool = True
	clahe_clip_limit: float = 2.0
	clahe_tile_size: int = 8
	denoise_mode: str = "bilateral"
	bilateral_d: int = 5
	bilateral_sigma_color: int = 60
	bilateral_sigma_space: int = 60
	median_kernel_size: int = 3
	min_bbox_wh_norm: float = 0.003
	train_augment: bool = True
	rare_ratio: float = 0.60
	max_aug_per_image: int = 1
	aug_prob_hflip: float = 0.5
	aug_prob_rotate: float = 0.5
	aug_rotate_deg: float = 10.0
	aug_prob_brightness_contrast: float = 0.8
	aug_brightness_delta: float = 0.15
	aug_contrast_delta: float = 0.15
	aug_prob_gamma: float = 0.3
	aug_gamma_delta: float = 0.15
	aug_prob_noise: float = 0.3
	aug_noise_std: float = 8.0


def read_class_names(classes_path: Path) -> List[str]:
	if not classes_path.exists():
		raise FileNotFoundError(f"Missing classes file: {classes_path}")
	return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_images(images_dir: Path) -> List[Path]:
	if not images_dir.exists():
		return []
	return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])


def parse_yolo_label_file(
	label_path: Path,
	class_count: int,
	stats: Dict[str, int],
) -> List[Tuple[int, float, float, float, float]]:
	boxes: List[Tuple[int, float, float, float, float]] = []
	if not label_path.exists():
		stats["missing_label_files"] += 1
		return boxes

	for raw_line in label_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line:
			continue
		parts = line.split()
		if len(parts) != 5:
			stats["invalid_label_lines"] += 1
			continue

		try:
			class_id = int(float(parts[0]))
			xc = float(parts[1])
			yc = float(parts[2])
			bw = float(parts[3])
			bh = float(parts[4])
		except ValueError:
			stats["invalid_label_lines"] += 1
			continue

		if class_id < 0 or class_id >= class_count:
			stats["invalid_class_id"] += 1
			continue

		boxes.append((class_id, xc, yc, bw, bh))
	return boxes


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, float, float, float]:
	x1 = float(np.clip(x1, 0.0, width - 1.0))
	y1 = float(np.clip(y1, 0.0, height - 1.0))
	x2 = float(np.clip(x2, 0.0, width - 1.0))
	y2 = float(np.clip(y2, 0.0, height - 1.0))
	if x2 < x1:
		x1, x2 = x2, x1
	if y2 < y1:
		y1, y2 = y2, y1
	return x1, y1, x2, y2


def yolo_to_xyxy(xc: float, yc: float, bw: float, bh: float, width: int, height: int) -> Tuple[float, float, float, float]:
	x1 = (xc - bw / 2.0) * width
	y1 = (yc - bh / 2.0) * height
	x2 = (xc + bw / 2.0) * width
	y2 = (yc + bh / 2.0) * height
	return x1, y1, x2, y2


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, float, float, float]:
	bw = max(0.0, x2 - x1)
	bh = max(0.0, y2 - y1)
	xc = x1 + bw / 2.0
	yc = y1 + bh / 2.0
	return xc / width, yc / height, bw / width, bh / height


def sanitize_boxes(
	boxes: Sequence[Tuple[int, float, float, float, float]],
	width: int,
	height: int,
	min_bbox_wh_norm: float,
	stats: Dict[str, int],
) -> List[Tuple[int, float, float, float, float]]:
	cleaned: List[Tuple[int, float, float, float, float]] = []
	for class_id, xc, yc, bw, bh in boxes:
		x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, width, height)
		x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, width, height)
		xc_n, yc_n, bw_n, bh_n = xyxy_to_yolo(x1, y1, x2, y2, width, height)
		if bw_n <= 0 or bh_n <= 0:
			stats["removed_zero_or_negative_bbox"] += 1
			continue
		if bw_n < min_bbox_wh_norm or bh_n < min_bbox_wh_norm:
			stats["removed_tiny_bbox"] += 1
			continue
		cleaned.append((class_id, xc_n, yc_n, bw_n, bh_n))
	return cleaned


def save_yolo_label_file(label_path: Path, boxes: Sequence[Tuple[int, float, float, float, float]], dry_run: bool) -> None:
	lines = [f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for cid, xc, yc, bw, bh in boxes]
	if dry_run:
		return
	label_path.parent.mkdir(parents=True, exist_ok=True)
	label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def apply_clahe_rgb(image: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	l_chan, a_chan, b_chan = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
	l_chan = clahe.apply(l_chan)
	merged = cv2.merge((l_chan, a_chan, b_chan))
	return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise_image(image: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
	if cfg.denoise_mode == "bilateral":
		return cv2.bilateralFilter(
			image,
			d=cfg.bilateral_d,
			sigmaColor=cfg.bilateral_sigma_color,
			sigmaSpace=cfg.bilateral_sigma_space,
		)
	if cfg.denoise_mode == "median":
		kernel = cfg.median_kernel_size if cfg.median_kernel_size % 2 == 1 else cfg.median_kernel_size + 1
		kernel = max(3, kernel)
		return cv2.medianBlur(image, kernel)
	return image


def letterbox_and_remap(
	image: np.ndarray,
	boxes: Sequence[Tuple[int, float, float, float, float]],
	target_size: int,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
	height, width = image.shape[:2]
	scale = min(target_size / width, target_size / height)
	new_w = int(round(width * scale))
	new_h = int(round(height * scale))

	resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
	canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
	pad_x = (target_size - new_w) // 2
	pad_y = (target_size - new_h) // 2
	canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

	mapped: List[Tuple[int, float, float, float, float]] = []
	for class_id, xc, yc, bw, bh in boxes:
		x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, width, height)
		x1 = x1 * scale + pad_x
		y1 = y1 * scale + pad_y
		x2 = x2 * scale + pad_x
		y2 = y2 * scale + pad_y
		x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, target_size, target_size)
		xc_n, yc_n, bw_n, bh_n = xyxy_to_yolo(x1, y1, x2, y2, target_size, target_size)
		mapped.append((class_id, xc_n, yc_n, bw_n, bh_n))

	return canvas, mapped


def preprocess_image(image: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
	result = image
	if cfg.apply_clahe:
		result = apply_clahe_rgb(result, cfg.clahe_clip_limit, cfg.clahe_tile_size)
	result = denoise_image(result, cfg)
	return result


def random_brightness_contrast(image: np.ndarray, rng: random.Random, cfg: PipelineConfig) -> np.ndarray:
	alpha = 1.0 + rng.uniform(-cfg.aug_contrast_delta, cfg.aug_contrast_delta)
	beta = 255.0 * rng.uniform(-cfg.aug_brightness_delta, cfg.aug_brightness_delta)
	output = image.astype(np.float32) * alpha + beta
	return np.clip(output, 0, 255).astype(np.uint8)


def random_gamma(image: np.ndarray, rng: random.Random, cfg: PipelineConfig) -> np.ndarray:
	gamma = 1.0 + rng.uniform(-cfg.aug_gamma_delta, cfg.aug_gamma_delta)
	gamma = max(0.3, gamma)
	inv_gamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
	return cv2.LUT(image, table)


def random_noise(image: np.ndarray, rng: random.Random, cfg: PipelineConfig) -> np.ndarray:
	noise = rng.gauss(0.0, cfg.aug_noise_std)
	arr = image.astype(np.float32)
	arr += np.random.normal(loc=noise, scale=cfg.aug_noise_std, size=arr.shape)
	return np.clip(arr, 0, 255).astype(np.uint8)


def horizontal_flip(
	image: np.ndarray,
	boxes: Sequence[Tuple[int, float, float, float, float]],
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
	flipped = cv2.flip(image, 1)
	new_boxes = [(cid, 1.0 - xc, yc, bw, bh) for cid, xc, yc, bw, bh in boxes]
	return flipped, new_boxes


def rotate_image_and_boxes(
	image: np.ndarray,
	boxes: Sequence[Tuple[int, float, float, float, float]],
	angle_deg: float,
	min_bbox_wh_norm: float,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
	height, width = image.shape[:2]
	center = (width / 2.0, height / 2.0)
	mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
	rotated = cv2.warpAffine(
		image,
		mat,
		(width, height),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_REFLECT_101,
	)

	remapped: List[Tuple[int, float, float, float, float]] = []
	for cid, xc, yc, bw, bh in boxes:
		x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, width, height)
		pts = np.array(
			[
				[x1, y1, 1.0],
				[x2, y1, 1.0],
				[x2, y2, 1.0],
				[x1, y2, 1.0],
			],
			dtype=np.float32,
		)
		rot = (mat @ pts.T).T
		rx1 = float(np.min(rot[:, 0]))
		ry1 = float(np.min(rot[:, 1]))
		rx2 = float(np.max(rot[:, 0]))
		ry2 = float(np.max(rot[:, 1]))
		rx1, ry1, rx2, ry2 = clamp_box(rx1, ry1, rx2, ry2, width, height)
		xc_n, yc_n, bw_n, bh_n = xyxy_to_yolo(rx1, ry1, rx2, ry2, width, height)
		if bw_n >= min_bbox_wh_norm and bh_n >= min_bbox_wh_norm:
			remapped.append((cid, xc_n, yc_n, bw_n, bh_n))

	return rotated, remapped


def apply_train_augmentation(
	image: np.ndarray,
	boxes: Sequence[Tuple[int, float, float, float, float]],
	cfg: PipelineConfig,
	rng: random.Random,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
	aug_img = image.copy()
	aug_boxes = list(boxes)

	if rng.random() < cfg.aug_prob_hflip:
		aug_img, aug_boxes = horizontal_flip(aug_img, aug_boxes)

	if rng.random() < cfg.aug_prob_rotate:
		angle = rng.uniform(-cfg.aug_rotate_deg, cfg.aug_rotate_deg)
		aug_img, aug_boxes = rotate_image_and_boxes(aug_img, aug_boxes, angle, cfg.min_bbox_wh_norm)

	if rng.random() < cfg.aug_prob_brightness_contrast:
		aug_img = random_brightness_contrast(aug_img, rng, cfg)

	if rng.random() < cfg.aug_prob_gamma:
		aug_img = random_gamma(aug_img, rng, cfg)

	if rng.random() < cfg.aug_prob_noise:
		aug_img = random_noise(aug_img, rng, cfg)

	return aug_img, aug_boxes


def count_class_image_occurrence(labels_dir: Path, class_count: int) -> List[int]:
	counts = [0 for _ in range(class_count)]
	for label_path in labels_dir.glob("*.txt"):
		classes_in_file = set()
		for line in label_path.read_text(encoding="utf-8").splitlines():
			parts = line.strip().split()
			if len(parts) != 5:
				continue
			try:
				class_id = int(float(parts[0]))
			except ValueError:
				continue
			if 0 <= class_id < class_count:
				classes_in_file.add(class_id)
		for class_id in classes_in_file:
			counts[class_id] += 1
	return counts


def classes_in_label_file(label_path: Path) -> set[int]:
	result: set[int] = set()
	for line in label_path.read_text(encoding="utf-8").splitlines():
		parts = line.strip().split()
		if len(parts) != 5:
			continue
		try:
			result.add(int(float(parts[0])))
		except ValueError:
			continue
	return result


def build_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
	return PipelineConfig(
		target_size=args.target_size,
		apply_clahe=not args.disable_clahe,
		clahe_clip_limit=args.clahe_clip_limit,
		clahe_tile_size=args.clahe_tile_size,
		denoise_mode=args.denoise,
		min_bbox_wh_norm=args.min_bbox_wh_norm,
		train_augment=not args.disable_train_augment,
		rare_ratio=args.rare_ratio,
		max_aug_per_image=args.max_aug_per_image,
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="PCB preprocessing pipeline for YOLO datasets")
	parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1])
	parser.add_argument("--output-root", type=Path, default=None)
	parser.add_argument("--target-size", type=int, default=640)
	parser.add_argument("--min-bbox-wh-norm", type=float, default=0.003)
	parser.add_argument("--denoise", type=str, default="bilateral", choices=["none", "bilateral", "median"])
	parser.add_argument("--disable-clahe", action="store_true")
	parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
	parser.add_argument("--clahe-tile-size", type=int, default=8)
	parser.add_argument("--disable-train-augment", action="store_true")
	parser.add_argument("--rare-ratio", type=float, default=0.60)
	parser.add_argument("--max-aug-per-image", type=int, default=1)
	parser.add_argument("--max-images-per-split", type=int, default=0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--dry-run", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	rng = random.Random(args.seed)
	np.random.seed(args.seed)

	dataset_root = args.dataset_root.resolve()
	output_root = args.output_root.resolve() if args.output_root else dataset_root / "preprocessed_dataset"
	config = build_pipeline_config(args)

	classes = read_class_names(dataset_root / "classes.txt")
	class_count = len(classes)

	stats: DefaultDict[str, int] = defaultdict(int)
	if not args.dry_run:
		output_root.mkdir(parents=True, exist_ok=True)

	for split in ("train", "val", "test"):
		images_dir = dataset_root / split / "images"
		labels_dir = dataset_root / split / "labels_txt"
		out_images = output_root / split / "images"
		out_labels = output_root / split / "labels_txt"

		if not args.dry_run:
			out_images.mkdir(parents=True, exist_ok=True)
			out_labels.mkdir(parents=True, exist_ok=True)

		image_paths = discover_images(images_dir)
		if args.max_images_per_split > 0:
			image_paths = image_paths[: args.max_images_per_split]

		for image_path in image_paths:
			stats["images_seen"] += 1
			label_path = labels_dir / f"{image_path.stem}.txt"

			image = cv2.imread(str(image_path))
			if image is None:
				stats["images_unreadable"] += 1
				continue

			boxes = parse_yolo_label_file(label_path, class_count=class_count, stats=stats)
			boxes = sanitize_boxes(
				boxes,
				width=image.shape[1],
				height=image.shape[0],
				min_bbox_wh_norm=config.min_bbox_wh_norm,
				stats=stats,
			)

			processed = preprocess_image(image, config)
			letterboxed, remapped_boxes = letterbox_and_remap(processed, boxes, target_size=config.target_size)
			remapped_boxes = sanitize_boxes(
				remapped_boxes,
				width=config.target_size,
				height=config.target_size,
				min_bbox_wh_norm=config.min_bbox_wh_norm,
				stats=stats,
			)

			out_img_path = out_images / f"{image_path.stem}.jpg"
			out_lbl_path = out_labels / f"{image_path.stem}.txt"
			if not args.dry_run:
				cv2.imwrite(str(out_img_path), letterboxed)
			save_yolo_label_file(out_lbl_path, remapped_boxes, dry_run=args.dry_run)
			stats["images_written"] += 1
			stats["labels_written"] += 1
			stats["bboxes_written"] += len(remapped_boxes)

	if config.train_augment and not args.dry_run:
		train_images_dir = output_root / "train" / "images"
		train_labels_dir = output_root / "train" / "labels_txt"
		if train_images_dir.exists() and train_labels_dir.exists():
			train_class_counts = count_class_image_occurrence(train_labels_dir, class_count)
			max_count = max(train_class_counts) if train_class_counts else 0
			rare_threshold = int(max_count * config.rare_ratio)
			rare_classes = {idx for idx, value in enumerate(train_class_counts) if value < rare_threshold}

			for label_path in sorted(train_labels_dir.glob("*.txt")):
				if not rare_classes:
					break
				present = classes_in_label_file(label_path)
				if not (present & rare_classes):
					continue

				image_path = train_images_dir / f"{label_path.stem}.jpg"
				image = cv2.imread(str(image_path))
				if image is None:
					continue

				boxes = parse_yolo_label_file(label_path, class_count, stats)
				for aug_index in range(config.max_aug_per_image):
					aug_img, aug_boxes = apply_train_augmentation(image, boxes, config, rng)
					aug_boxes = sanitize_boxes(
						aug_boxes,
						width=aug_img.shape[1],
						height=aug_img.shape[0],
						min_bbox_wh_norm=config.min_bbox_wh_norm,
						stats=stats,
					)
					if not aug_boxes:
						continue

					aug_name = f"{label_path.stem}_aug{aug_index + 1}"
					aug_img_path = train_images_dir / f"{aug_name}.jpg"
					aug_lbl_path = train_labels_dir / f"{aug_name}.txt"
					if not args.dry_run:
						cv2.imwrite(str(aug_img_path), aug_img)
					save_yolo_label_file(aug_lbl_path, aug_boxes, dry_run=args.dry_run)
					stats["augmented_images_written"] += 1
					stats["augmented_bboxes_written"] += len(aug_boxes)

	if not args.dry_run:
		shutil.copy2(dataset_root / "classes.txt", output_root / "classes.txt")
		yaml_path = output_root / "yolo_data.yaml"
		yaml_text = (
			"path: .\n"
			"train: train/images\n"
			"val: val/images\n"
			"test: test/images\n\n"
			"names:\n"
		)
		for idx, class_name in enumerate(classes):
			yaml_text += f"  {idx}: {class_name}\n"
		yaml_path.write_text(yaml_text, encoding="utf-8")

	report = {
		"generated_at_utc": datetime.now(timezone.utc).isoformat(),
		"dataset_root": str(dataset_root),
		"output_root": str(output_root),
		"config": {
			"dataset_root": str(dataset_root),
			"output_root": str(output_root),
			"target_size": args.target_size,
			"min_bbox_wh_norm": args.min_bbox_wh_norm,
			"denoise": args.denoise,
			"disable_clahe": args.disable_clahe,
			"clahe_clip_limit": args.clahe_clip_limit,
			"clahe_tile_size": args.clahe_tile_size,
			"disable_train_augment": args.disable_train_augment,
			"rare_ratio": args.rare_ratio,
			"max_aug_per_image": args.max_aug_per_image,
			"max_images_per_split": args.max_images_per_split,
			"seed": args.seed,
			"dry_run": args.dry_run,
		},
		"stats": dict(stats),
		"classes": classes,
	}

	if not args.dry_run:
		report_path = output_root / "preprocess_report.json"
		report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
		print(f"Saved report: {report_path}")

	print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()