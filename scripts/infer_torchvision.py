from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference using torchvision detection checkpoints."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="Image path or directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/torchvision_infer"),
        help="Directory for visualized predictions",
    )
    parser.add_argument("--score-thr", type=float, default=0.25, help="Score threshold")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    return parser.parse_args()


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:
    if model_name == "faster_rcnn":
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    if model_name == "retinanet":
        model = retinanet_resnet50_fpn_v2(weights=None)
        in_channels = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=None,
        )
        return model

    raise ValueError(f"Unsupported model in checkpoint: {model_name}")


def list_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    out: List[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        out.extend(sorted(path.glob(suffix)))
    return out


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
        class_name = class_names[label_idx - 1] if 1 <= label_idx <= len(class_names) else str(label_idx)
        text = f"{class_name}:{float(score):.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (50, 220, 50), 2)
        cv2.putText(
            canvas,
            text,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 220, 50),
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

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model_name = str(checkpoint.get("model_name", "faster_rcnn"))
    class_names = list(checkpoint.get("class_names", []))
    num_classes = len(class_names) + 1

    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    image_paths = list_images(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found at: {input_path}")

    with torch.no_grad():
        for image_path in image_paths:
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                print(f"[WARN] Skip unreadable image: {image_path}")
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
            output = model([image_tensor])[0]

            vis = draw_detections(
                image=bgr,
                boxes=output["boxes"].detach().cpu().numpy(),
                labels=output["labels"].detach().cpu().numpy(),
                scores=output["scores"].detach().cpu().numpy(),
                class_names=class_names,
                score_thr=args.score_thr,
            )

            out_path = output_dir / image_path.name
            cv2.imwrite(str(out_path), vis)
            print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
