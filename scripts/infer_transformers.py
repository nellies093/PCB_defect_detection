from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    DeformableDetrForObjectDetection,
    DetrForObjectDetection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference using DETR/Deformable DETR checkpoints trained via transformers."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="Image path or directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/transformers_infer"),
        help="Directory to save visualized predictions",
    )
    parser.add_argument("--score-thr", type=float, default=0.25, help="Score threshold")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args()


def list_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    images: List[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(sorted(path.glob(suffix)))
    return images


def build_model(checkpoint: dict, num_labels: int, id2label: dict[int, str], label2id: dict[str, int]):
    model_name = str(checkpoint.get("model_name", "detr"))
    hf_model_name = str(checkpoint.get("hf_model_name", "facebook/detr-resnet-50"))

    image_processor = AutoImageProcessor.from_pretrained(hf_model_name)
    if model_name == "deformable_detr":
        model = DeformableDetrForObjectDetection.from_pretrained(
            hf_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    else:
        model = DetrForObjectDetection.from_pretrained(
            hf_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    model.load_state_dict(checkpoint["model"])
    return model, image_processor


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: List[str],
    score_thr: float,
) -> np.ndarray:
    canvas = image.copy()
    for box, label, score in zip(boxes, labels, scores):
        if float(score) < score_thr:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        label_idx = int(label)
        class_name = class_names[label_idx] if 0 <= label_idx < len(class_names) else str(label_idx)
        caption = f"{class_name}:{float(score):.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            caption,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    class_names = list(checkpoint.get("class_names", []))
    if not class_names:
        raise SystemExit("Checkpoint is missing class_names metadata.")

    id2label = {i: n for i, n in enumerate(class_names)}
    label2id = {n: i for i, n in id2label.items()}
    model, image_processor = build_model(
        checkpoint=checkpoint,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
    )

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    images = list_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found at: {input_path}")

    with torch.no_grad():
        for image_path in images:
            pil = Image.open(image_path).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

            encoding = image_processor(images=pil, return_tensors="pt")
            pixel_values = encoding["pixel_values"].to(device)
            pixel_mask = encoding.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device)

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            target_sizes = torch.tensor([[pil.height, pil.width]], device=device)
            result = image_processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=args.score_thr,
            )[0]

            vis = draw_detections(
                image=bgr,
                boxes=result["boxes"].detach().cpu().numpy(),
                labels=result["labels"].detach().cpu().numpy(),
                scores=result["scores"].detach().cpu().numpy(),
                class_names=class_names,
                score_thr=args.score_thr,
            )

            out_path = output_dir / image_path.name
            cv2.imwrite(str(out_path), vis)
            print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
