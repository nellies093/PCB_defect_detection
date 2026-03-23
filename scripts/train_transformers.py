from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoImageProcessor,
    DeformableDetrForObjectDetection,
    DetrForObjectDetection,
)


HF_MODEL_NAMES = {
    "detr": "facebook/detr-resnet-50",
    "deformable_detr": "SenseTime/deformable-detr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DETR / Deformable DETR via Hugging Face Transformers on COCO-format PCB dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing data/ and classes.txt",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["detr", "deformable_detr"],
        default="detr",
        help="Transformers detector type",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--project",
        type=str,
        default="runs/pcb_train",
        help="Output root directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run directory name. Default: transformers_<model>",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in run directory",
    )
    return parser.parse_args()


def load_class_names(path: Path) -> List[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [n for n in names if n]


class CocoDetectionForDetr(Dataset):
    def __init__(
        self,
        images_dir: Path,
        ann_file: Path,
        image_processor: AutoImageProcessor,
        catid_to_label: Dict[int, int] | None = None,
    ) -> None:
        payload = json.loads(ann_file.read_text(encoding="utf-8"))
        self.images_dir = images_dir
        self.image_processor = image_processor

        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        categories = payload.get("categories", [])

        self.images: Dict[int, Dict[str, Any]] = {int(i["id"]): i for i in images}
        self.image_ids = sorted(self.images.keys())

        self.ann_by_image: Dict[int, List[Dict[str, Any]]] = {image_id: [] for image_id in self.image_ids}
        for ann in annotations:
            image_id = int(ann["image_id"])
            if image_id in self.ann_by_image:
                self.ann_by_image[image_id].append(ann)

        cat_ids = sorted(int(c["id"]) for c in categories)
        if catid_to_label is None:
            self.catid_to_label = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        else:
            self.catid_to_label = dict(catid_to_label)

        self.label_to_catid = {label: cat_id for cat_id, label in self.catid_to_label.items()}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        image_path = self.images_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        anns = self.ann_by_image.get(image_id, [])
        coco_anns: List[Dict[str, Any]] = []
        for ann in anns:
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            if float(w) <= 1 or float(h) <= 1:
                continue
            coco_anns.append(
                {
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "category_id": int(self.catid_to_label[int(ann["category_id"])]),
                    "area": float(ann.get("area", float(w) * float(h))),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )

        encoding = self.image_processor(
            images=image,
            annotations={"image_id": image_id, "annotations": coco_anns},
            return_tensors="pt",
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def collate_fn(batch: List[Dict[str, Any]], image_processor: AutoImageProcessor) -> Dict[str, Any]:
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }


def build_model(
    model_name: str,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
):
    hf_name = HF_MODEL_NAMES[model_name]
    image_processor = AutoImageProcessor.from_pretrained(hf_name)

    if model_name == "detr":
        model = DetrForObjectDetection.from_pretrained(
            hf_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    else:
        model = DeformableDetrForObjectDetection.from_pretrained(
            hf_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    return model, image_processor, hf_name


def move_labels_to_device(labels: List[Dict[str, torch.Tensor]], device: torch.device) -> List[Dict[str, torch.Tensor]]:
    moved: List[Dict[str, torch.Tensor]] = []
    for label in labels:
        moved.append({k: v.to(device) for k, v in label.items()})
    return moved


def cxcywh_to_xyxy_abs(boxes: torch.Tensor, image_hw: torch.Tensor) -> torch.Tensor:
    h, w = float(image_hw[0].item()), float(image_hw[1].item())
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = (cx - 0.5 * bw) * w
    y1 = (cy - 0.5 * bh) * h
    x2 = (cx + 0.5 * bw) * w
    y2 = (cy + 0.5 * bh) * h
    return torch.stack((x1, y1, x2, y2), dim=-1)


def train_one_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = move_labels_to_device(batch["labels"], device)

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_map(
    model,
    image_processor: AutoImageProcessor,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    try:
        MeanAveragePrecision = importlib.import_module(
            "torchmetrics.detection.mean_ap"
        ).MeanAveragePrecision
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "torchmetrics is not installed. Run: pip install torchmetrics"
        ) from exc

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        target_sizes = torch.stack([label["orig_size"] for label in labels]).to(device)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.0,
        )

        preds: List[Dict[str, torch.Tensor]] = []
        targets: List[Dict[str, torch.Tensor]] = []

        for pred, target in zip(results, labels):
            preds.append(
                {
                    "boxes": pred["boxes"].detach().cpu(),
                    "scores": pred["scores"].detach().cpu(),
                    "labels": pred["labels"].detach().cpu(),
                }
            )

            tgt_boxes = cxcywh_to_xyxy_abs(target["boxes"].detach().cpu(), target["orig_size"].detach().cpu())
            targets.append(
                {
                    "boxes": tgt_boxes,
                    "labels": target["class_labels"].detach().cpu(),
                }
            )

        metric.update(preds, targets)

    result = metric.compute()
    map_50_95 = float(result["map"].item())
    map_50 = float(result["map_50"].item())
    return map_50_95, map_50


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("epoch_*.pth"))
    return candidates[-1] if candidates else None


def validate_paths(dataset_root: Path) -> None:
    required = [
        dataset_root / "classes.txt",
        dataset_root / "data" / "annotations_json" / "train.json",
        dataset_root / "data" / "annotations_json" / "val.json",
        dataset_root / "data" / "train" / "images",
        dataset_root / "data" / "val" / "images",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        pretty = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(f"Missing required dataset files:\n{pretty}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    validate_paths(dataset_root)

    class_names = load_class_names(dataset_root / "classes.txt")
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in id2label.items()}
    num_labels = len(class_names)

    model, image_processor, hf_name = build_model(
        model_name=args.model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = CocoDetectionForDetr(
        images_dir=dataset_root / "data" / "train" / "images",
        ann_file=dataset_root / "data" / "annotations_json" / "train.json",
        image_processor=image_processor,
    )
    val_dataset = CocoDetectionForDetr(
        images_dir=dataset_root / "data" / "val" / "images",
        ann_file=dataset_root / "data" / "annotations_json" / "val.json",
        image_processor=image_processor,
        catid_to_label=train_dataset.catid_to_label,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=lambda b: collate_fn(b, image_processor),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch),
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        collate_fn=lambda b: collate_fn(b, image_processor),
    )

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    run_name = args.name or f"transformers_{args.model}"
    output_dir = (dataset_root / args.project / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    start_epoch = 1
    best_map = -1.0
    metrics_history: List[Dict[str, Any]] = []

    if args.resume:
        ckpt_path = latest_checkpoint(output_dir)
        if ckpt_path is not None:
            checkpoint = torch.load(str(ckpt_path), map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_map = float(checkpoint.get("best_map_50_95", -1.0))
            print(f"[INFO] Resumed from {ckpt_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        map_50_95, map_50 = evaluate_map(model, image_processor, val_loader, device)
        lr_scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "map_50_95": map_50_95,
            "map_50": map_50,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        }
        metrics_history.append(row)
        print(
            f"[EPOCH {epoch:03d}] loss={train_loss:.4f} "
            f"mAP50-95={map_50_95:.4f} mAP50={map_50:.4f} "
            f"lr={row['lr']:.6f} time={elapsed:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "best_map_50_95": best_map,
            "model_name": args.model,
            "hf_model_name": hf_name,
            "class_names": class_names,
            "label_to_catid": train_dataset.label_to_catid,
        }
        torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pth")

        if map_50_95 >= best_map:
            best_map = map_50_95
            checkpoint["best_map_50_95"] = best_map
            torch.save(checkpoint, output_dir / "best.pth")

    (output_dir / "metrics.json").write_text(json.dumps(metrics_history, indent=2), encoding="utf-8")

    summary = {
        "model": args.model,
        "hf_model_name": hf_name,
        "best_map_50_95": best_map,
        "epochs": args.epochs,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Training complete. Best mAP50-95={best_map:.4f}")
    print(f"[OK] Saved: {output_dir}")


if __name__ == "__main__":
    main()
