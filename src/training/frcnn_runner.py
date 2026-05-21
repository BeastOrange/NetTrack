"""
Faster R-CNN (torchvision R50-FPN) training loop using the YOLO-format dataset.

This is a thin training runner so the comparison row in the matrix uses the
exact same data split and image size as the YOLOv8 variants. We keep it
intentionally minimal: SGD, step LR, fixed image size, COCO-style targets.

Outputs `<project>/frcnn/best.pt` (state_dict) plus a `metrics.json` with
val mAP@0.5 and mAP@0.5:0.95 per epoch (computed via torchmetrics).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import yaml
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from tqdm import tqdm


@dataclass
class DatasetSpec:
    root: Path
    split: str
    nc: int


class YoloDirDataset(Dataset):
    """Reads YOLO-format <class_id cx cy w h> labels into torchvision-style targets."""

    def __init__(self, spec: DatasetSpec, imgsz: int) -> None:
        self.images_dir = spec.root / "images" / spec.split
        self.labels_dir = spec.root / "labels" / spec.split
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Missing images dir: {self.images_dir}")
        self.image_paths = sorted(p for p in self.images_dir.glob("*.jpg"))
        if not self.image_paths:
            raise RuntimeError(f"No images in {self.images_dir}")
        self.imgsz = imgsz

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        image_path = self.image_paths[idx]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as im:
            im = im.convert("RGB")
            orig_w, orig_h = im.size
            im = im.resize((self.imgsz, self.imgsz), Image.BILINEAR)
            tensor = torchvision.transforms.functional.to_tensor(im)

        boxes: list[list[float]] = []
        labels: list[int] = []

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = (float(x) for x in parts)
                cx *= self.imgsz
                cy *= self.imgsz
                bw *= self.imgsz
                bh *= self.imgsz
                x1 = max(0.0, cx - bw / 2)
                y1 = max(0.0, cy - bh / 2)
                x2 = min(float(self.imgsz), cx + bw / 2)
                y2 = min(float(self.imgsz), cy + bh / 2)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                boxes.append([x1, y1, x2, y2])
                # torchvision detector reserves class 0 for background.
                labels.append(int(cls_id) + 1)

        if boxes:
            target_boxes = torch.tensor(boxes, dtype=torch.float32)
            target_labels = torch.tensor(labels, dtype=torch.int64)
        else:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target_labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": target_boxes,
            "labels": target_labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
        }
        return tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def _load_data_yaml(path: Path) -> tuple[Path, int]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", path.parent)).expanduser().resolve()
    nc = int(cfg["nc"])
    return root, nc


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.to(device)
    model.eval()
    for images, targets in tqdm(loader, desc="val", leave=False):
        images = [im.to(device) for im in images]
        outputs = model(images)
        preds = [
            {
                "boxes": o["boxes"].detach(),
                "scores": o["scores"].detach(),
                "labels": o["labels"].detach(),
            }
            for o in outputs
        ]
        gts = [
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device),
            }
            for t in targets
        ]
        metric.update(preds, gts)
    out = metric.compute()
    return {
        "map50": float(out["map_50"].item()),
        "map5095": float(out["map"].item()),
    }


def run_frcnn_training(
    data_yaml: Path,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str,
    project: Path,
    seed: int,
    workers: int,
) -> None:
    torch.manual_seed(seed)
    dev = torch.device(device)

    root, nc = _load_data_yaml(data_yaml)

    train_ds = YoloDirDataset(DatasetSpec(root=root, split="train", nc=nc), imgsz=imgsz)
    val_ds = YoloDirDataset(DatasetSpec(root=root, split="val", nc=nc), imgsz=imgsz)
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=workers, collate_fn=collate_fn, pin_memory=True,
    )

    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        num_classes=91,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc + 1)
    model.to(dev)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)

    out_dir = project / "frcnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    history: list[dict] = []
    best_map50 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for images, targets in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
            images = [im.to(dev) for im in images]
            targets = [{k: v.to(dev) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
        lr_sched.step()
        train_loss = running / max(1, n_batches)

        val_metrics = evaluate(model, val_loader, dev)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        logger.info(
            f"[frcnn] epoch={epoch} loss={train_loss:.4f} "
            f"map50={val_metrics['map50']:.4f} map5095={val_metrics['map5095']:.4f}"
        )

        if val_metrics["map50"] > best_map50:
            best_map50 = val_metrics["map50"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    torch.save(model.state_dict(), out_dir / "last.pt")
    logger.info(f"[frcnn] done. best map@0.5={best_map50:.4f}, artefacts in {out_dir}")
