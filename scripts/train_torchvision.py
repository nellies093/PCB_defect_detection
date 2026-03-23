from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms import functional as TF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PCB detectors using torchvision (Faster R-CNN / RetinaNet)."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root that contains data/ and classes.txt",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["faster_rcnn", "retinanet"],
        default="faster_rcnn",
        help="Torchvision detection architecture",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
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
        help="Run name. Default uses model name",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet/COCO pretraining",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in output directory",
    )
    return parser.parse_args()


def load_class_names(path: Path) -> List[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name]


class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        ann_file: Path,
        catid_to_label: Dict[int, int] | None = None,
    ) -> None:
        try:
            COCO = importlib.import_module("pycocotools.coco").COCO
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "pycocotools is not installed. Run: pip install pycocotools"
            ) from exc

        self.images_dir = images_dir
        self.coco = COCO(str(ann_file))
        self.image_ids = sorted(self.coco.getImgIds())

        categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat_ids = sorted(cat["id"] for cat in categories)

        if catid_to_label is None:
            self.catid_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}
        else:
            self.catid_to_label = dict(catid_to_label)

        self.label_to_catid = {label: cat_id for cat_id, label in self.catid_to_label.items()}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = self.images_dir / image_info["file_name"]

        image = Image.open(image_path).convert("RGB")
        image_tensor = TF.convert_image_dtype(TF.pil_to_tensor(image), dtype=torch.float32)

        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in anns:
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            if w <= 1 or h <= 1:
                continue
            x1 = float(x)
            y1 = float(y)
            x2 = float(x + w)
            y2 = float(y + h)
            boxes.append([x1, y1, x2, y2])
            labels.append(self.catid_to_label[int(ann["category_id"])])
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            areas_tensor = torch.tensor(areas, dtype=torch.float32)
            crowd_tensor = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
            crowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": crowd_tensor,
        }
        return image_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if model_name == "faster_rcnn":
        weights = "DEFAULT" if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    if model_name == "retinanet":
        weights = "DEFAULT" if pretrained else None
        model = retinanet_resnet50_fpn_v2(weights=weights)
        in_channels = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=None,
        )
        return model

    raise ValueError(f"Unsupported torchvision model: {model_name}")


def move_targets_to_device(targets: List[Dict[str, torch.Tensor]], device: torch.device) -> List[Dict[str, torch.Tensor]]:
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        total_loss += float(losses.detach().item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_coco_map(
    model: nn.Module,
    loader: DataLoader,
    dataset: CocoDetectionDataset,
    device: torch.device,
) -> Tuple[float, float]:
    try:
        COCOeval = importlib.import_module("pycocotools.cocoeval").COCOeval
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "pycocotools is not installed. Run: pip install pycocotools"
        ) from exc

    model.eval()
    results: List[Dict[str, Any]] = []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].detach().cpu()
            labels = output["labels"].detach().cpu()
            scores = output["scores"].detach().cpu()

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.tolist()
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(dataset.label_to_catid.get(int(label), int(label))),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )

    if not results:
        return 0.0, 0.0

    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    map_50_95 = float(evaluator.stats[0])
    map_50 = float(evaluator.stats[1])
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
    num_classes = len(class_names) + 1

    train_dataset = CocoDetectionDataset(
        images_dir=dataset_root / "data" / "train" / "images",
        ann_file=dataset_root / "data" / "annotations_json" / "train.json",
    )
    val_dataset = CocoDetectionDataset(
        images_dir=dataset_root / "data" / "val" / "images",
        ann_file=dataset_root / "data" / "annotations_json" / "val.json",
        catid_to_label=train_dataset.catid_to_label,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch),
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        collate_fn=collate_fn,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    run_name = args.name or args.model
    output_dir = (dataset_root / args.project / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, num_classes=num_classes, pretrained=not args.no_pretrained).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

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
        map_50_95, map_50 = evaluate_coco_map(model, val_loader, val_dataset, device)
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
            "class_names": class_names,
            "label_to_catid": train_dataset.label_to_catid,
        }
        torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pth")

        if map_50_95 >= best_map:
            best_map = map_50_95
            checkpoint["best_map_50_95"] = best_map
            torch.save(checkpoint, output_dir / "best.pth")

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_history, indent=2), encoding="utf-8"
    )

    summary = {
        "model": args.model,
        "best_map_50_95": best_map,
        "epochs": args.epochs,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Training complete. Best mAP50-95={best_map:.4f}")
    print(f"[OK] Saved: {output_dir}")


if __name__ == "__main__":
    main()
