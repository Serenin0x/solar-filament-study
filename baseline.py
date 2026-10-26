"""Solar filament baseline v1.0: one-file training, evaluation, and inference.

The local PQ implementation follows the standard IoU > 0.5 definition. Its
agreement with the organizer's private evaluator has not yet been verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


VERSION = "1.0"
SPLIT_RATIOS = (0.70, 0.15, 0.15)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        channels = [width * (2**i) for i in range(4)]
        self.encoders = nn.ModuleList(
            [ConvBlock(a, b) for a, b in zip([1] + channels[:-1], channels)]
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)
        self.decoders = nn.ModuleList()
        incoming = channels[-1] * 2
        for skip_channels in reversed(channels):
            self.decoders.append(ConvBlock(incoming + skip_channels, skip_channels))
            incoming = skip_channels
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.head(x)


def decode_annotation(annotation: dict, height: int, width: int) -> np.ndarray:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        if not segmentation:
            raise ValueError(f"Empty polygon in annotation {annotation.get('id')}")
        if isinstance(segmentation[0], (int, float)):
            segmentation = [segmentation]
        rles = mask_utils.frPyObjects(segmentation, height, width)
        encoded = mask_utils.merge(rles) if isinstance(rles, list) else rles
    elif isinstance(segmentation["counts"], list):
        encoded = mask_utils.frPyObjects(segmentation, height, width)
    else:
        encoded = segmentation
    decoded = mask_utils.decode(encoded)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


class CompetitionData:
    def __init__(self, root: Path):
        self.root = root
        self.train_dir = root / "train" / "train_images"
        self.test_dir = root / "test" / "test_images"
        self.annotation_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        if not self.train_dir.is_dir() or not self.annotation_path.is_file():
            raise FileNotFoundError(f"Expected train/train_images and annotation JSON under {root}")
        raw_bytes = self.annotation_path.read_bytes()
        self.annotation_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        raw = json.loads(raw_bytes)
        self.records = defaultdict(list)
        self.by_record = defaultdict(list)
        for image_record in raw["images"]:
            name = Path(image_record["file_name"]).name
            self.records[name].append(image_record)
        for annotation in raw["annotations"]:
            self.by_record[str(annotation["image_id"])].append(annotation)
        if not self.records:
            raise ValueError("No image records found")
        self.paths = self._paths(self.train_dir, self.records)
        missing = [name for name in self.records if name not in self.paths]
        if missing:
            raise FileNotFoundError(f"Cannot resolve {len(missing)} training images: {missing[:3]}")
        for name, records in self.records.items():
            for record in records:
                if str(record["id"]) not in self.by_record:
                    self.by_record[str(record["id"])] = []
            self.records[name] = sorted(records, key=lambda record: str(record["id"]))

    @staticmethod
    def _paths(directory: Path, names) -> dict[str, Path]:
        files = {path.name: path for path in directory.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png")}
        by_stem = defaultdict(list)
        for path in files.values():
            by_stem[path.stem].append(path)
        paths = {}
        for name in names:
            if name in files:
                paths[name] = files[name]
            elif len(by_stem[Path(name).stem]) == 1:
                paths[name] = by_stem[Path(name).stem][0]
        return paths

    def instance_masks(self, record: dict, height: int, width: int) -> list[np.ndarray]:
        return [decode_annotation(ann, height, width) for ann in self.by_record[str(record["id"])]]


def split_group(name: str) -> str:
    # Observations from the same calendar month remain in one subset.
    stem = Path(name).stem
    return stem[:6] if re.match(r"^\d{14}", stem) else stem


def get_splits(data: CompetitionData, out: Path, seed: int) -> dict:
    manifest = out / "split.json"
    names = set(data.records)
    if manifest.exists():
        splits = json.loads(manifest.read_text())
        if splits["annotation_sha256"] != data.annotation_sha256:
            raise ValueError("Annotation JSON changed; do not reuse this split")
        if splits["seed"] != seed:
            raise ValueError("Seed differs from saved split; use the recorded seed")
        assigned = [name for subset in ("train", "val", "test") for name in splits[subset]]
        if len(assigned) != len(names) or set(assigned) != names:
            raise ValueError("Saved split does not match current image set")
        return splits

    groups = defaultdict(list)
    for name in sorted(names):
        groups[split_group(name)].append(name)
    group_keys = sorted(groups)
    if len(group_keys) < 3:
        raise ValueError("At least three observation groups are needed for train/val/test")
    random.Random(seed).shuffle(group_keys)
    target_train = len(names) * SPLIT_RATIOS[0]
    target_val = len(names) * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1])
    splits = {"train": [], "val": [], "test": []}
    for group in group_keys:
        if len(splits["train"]) < target_train:
            part = "train"
        elif len(splits["train"]) + len(splits["val"]) < target_val:
            part = "val"
        else:
            part = "test"
        splits[part].extend(groups[group])
    if any(not splits[key] for key in ("train", "val", "test")):
        raise ValueError("Grouped split left an empty subset; inspect observation groups")
    for key in ("train", "val", "test"):
        splits[key].sort()
    splits.update({
        "version": VERSION,
        "seed": seed,
        "grouping": "same calendar month from filename; fallback full stem",
        "annotation_sha256": data.annotation_sha256,
        "train_record_policy": "first record by string-sorted image record ID",
    })
    out.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(splits, indent=2) + "\n")
    return splits


class TrainingImages(Dataset):
    def __init__(self, data: CompetitionData, names: list[str], patch: int):
        self.data, self.names, self.patch = data, names, patch

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        image = np.asarray(Image.open(self.data.paths[name]).convert("L"), dtype=np.uint8)
        height, width = image.shape
        record = self.data.records[name][0]
        mask = np.zeros((height, width), dtype=bool)
        for instance in self.data.instance_masks(record, height, width):
            mask |= instance
        patch = self.patch
        image = np.pad(image, ((0, max(0, patch - height)), (0, max(0, patch - width))))
        mask = np.pad(mask, ((0, max(0, patch - height)), (0, max(0, patch - width))))
        height, width = image.shape
        if mask.any() and np.random.random() < 0.75:
            positive = np.argwhere(mask)
            y, x = positive[np.random.randint(len(positive))]
            top = int(np.clip(y - np.random.randint(patch), 0, height - patch))
            left = int(np.clip(x - np.random.randint(patch), 0, width - patch))
        else:
            top = np.random.randint(height - patch + 1)
            left = np.random.randint(width - patch + 1)
        image = image[top : top + patch, left : left + patch]
        mask = mask[top : top + patch, left : left + patch]
        if np.random.random() < 0.5:
            image, mask = np.fliplr(image), np.fliplr(mask)
        if np.random.random() < 0.5:
            image, mask = np.flipud(image), np.flipud(mask)
        rotations = np.random.randint(4)
        image, mask = np.rot90(image, rotations), np.rot90(mask, rotations)
        x = torch.from_numpy(image.copy()).float().unsqueeze(0) / 255.0
        y = torch.from_numpy(mask.copy()).float().unsqueeze(0)
        return x, y


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(1, 2, 3))
    dice_loss = 1 - ((2 * intersection + 1) / (
        probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1
    )).mean()
    return bce + dice_loss


@torch.inference_mode()
def predict_probability(model: nn.Module, image: np.ndarray, patch: int,
                        device: torch.device) -> np.ndarray:
    height, width = image.shape
    probability = np.empty((height, width), dtype=np.float32)
    for top in range(0, height, patch):
        for left in range(0, width, patch):
            tile = image[top : top + patch, left : left + patch]
            tile_height, tile_width = tile.shape
            tile = np.pad(tile, ((0, patch - tile_height), (0, patch - tile_width)))
            tensor = torch.from_numpy(tile.copy()).float()[None, None].to(device) / 255.0
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                output = model(tensor).sigmoid()
            probability[top : top + tile_height, left : left + tile_width] = (
                output[0, 0, :tile_height, :tile_width].float().cpu().numpy()
            )
    return probability


def instance_labels(probability: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    connectivity = ndimage.generate_binary_structure(2, 2)
    labels, _ = ndimage.label(probability >= threshold, structure=connectivity)
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    clean = keep[labels]
    return ndimage.label(clean, structure=connectivity)[0].astype(np.int32)


def score_record(predicted: np.ndarray, truth: list[np.ndarray]) -> dict[str, float]:
    count = int(predicted.max())
    predicted_areas = np.bincount(predicted.ravel(), minlength=count + 1)
    ground_union = np.zeros(predicted.shape, dtype=bool)
    candidates = []
    for ground_id, mask in enumerate(truth):
        ground_union |= mask
        ground_area = int(mask.sum())
        if not ground_area:
            continue
        overlap = np.bincount(predicted[mask], minlength=count + 1)
        for predicted_id in np.flatnonzero(overlap[1:]) + 1:
            intersection = overlap[predicted_id]
            iou = intersection / (ground_area + predicted_areas[predicted_id] - intersection)
            if iou > 0.5:
                candidates.append((float(iou), ground_id, int(predicted_id)))
    used_ground, used_predicted, matched_iou = set(), set(), 0.0
    for iou, ground_id, predicted_id in sorted(candidates, reverse=True):
        if ground_id not in used_ground and predicted_id not in used_predicted:
            used_ground.add(ground_id)
            used_predicted.add(predicted_id)
            matched_iou += iou
    true_positive = len(used_ground)
    denominator = true_positive + 0.5 * (count - true_positive) + 0.5 * (len(truth) - true_positive)
    pq = matched_iou / denominator if denominator else 1.0
    predicted_union = predicted > 0
    intersection = int(np.logical_and(predicted_union, ground_union).sum())
    total = int(predicted_union.sum() + ground_union.sum())
    union = total - intersection
    return {
        "pq": pq,
        "dice": 2 * intersection / total if total else 1.0,
        "iou": intersection / union if union else 1.0,
    }


def evaluate(model: nn.Module, data: CompetitionData, names: list[str],
             patch: int, threshold: float, min_area: int,
             device: torch.device) -> dict:
    model.eval()
    image_scores, record_scores = [], []
    for index, name in enumerate(names, 1):
        image = np.asarray(Image.open(data.paths[name]).convert("L"), dtype=np.uint8)
        height, width = image.shape
        probability = predict_probability(model, image, patch, device)
        predicted = instance_labels(probability, threshold, min_area)
        current = [score_record(predicted, data.instance_masks(record, height, width))
                   for record in data.records[name]]
        record_scores.extend(current)
        image_scores.append({key: float(np.mean([item[key] for item in current]))
                             for key in ("pq", "dice", "iou")})
        if index % 25 == 0 or index == len(names):
            print(f"Evaluated {index}/{len(names)} images", flush=True)
    return {
        "images": len(names),
        "annotation_records": len(record_scores),
        "pq_image_mean": float(np.mean([item["pq"] for item in image_scores])),
        "pq_record_mean": float(np.mean([item["pq"] for item in record_scores])),
        "dice_image_mean": float(np.mean([item["dice"] for item in image_scores])),
        "iou_image_mean": float(np.mean([item["iou"] for item in image_scores])),
        "metric_note": "Local standard PQ (IoU > 0.5); official scorer not yet audited",
    }


def load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = UNet(checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def train(args, data: CompetitionData, device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("Full baseline training needs a CUDA GPU; use 'smoke' for a CPU code check")
    if args.patch % 16:
        raise ValueError("--patch must be divisible by 16")
    if min(args.epochs, args.batch_size, args.patch, args.width, args.val_every) < 1 or args.workers < 0:
        raise ValueError("Epochs, batch size, patch, width, validation interval and workers are invalid")
    if args.lr <= 0 or not 0 < args.threshold < 1 or args.min_area < 1:
        raise ValueError("Learning rate and minimum area must be positive; threshold must be in (0, 1)")
    splits = get_splits(data, args.out, args.seed)
    seed_everything(args.seed)
    model = UNet(args.width).to(device)
    print("Environment:", {name: importlib.metadata.version(name)
                           for name in ("torch", "numpy", "Pillow", "pycocotools", "scipy")})
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Image split: {[len(splits[key]) for key in ('train', 'val', 'test')]}")
    dataset = TrainingImages(data, splits["train"], args.patch)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, worker_init_fn=worker_seed,
                        generator=torch.Generator().manual_seed(args.seed), pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best = (-1.0, -1.0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (image, target) in enumerate(loader, 1):
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                loss = loss_fn(model(image), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            if step % 50 == 0 or step == len(loader):
                print(f"Epoch {epoch}/{args.epochs}, batch {step}/{len(loader)}, "
                      f"mean loss {total_loss / step:.4f}", flush=True)
        if epoch % args.val_every and epoch != args.epochs:
            history.append({"epoch": epoch, "train_loss": total_loss / len(loader),
                            "validation": None})
            continue
        validation = evaluate(model, data, splits["val"], args.patch,
                              args.threshold, args.min_area, device)
        record = {"epoch": epoch, "train_loss": total_loss / len(loader),
                  "validation": validation}
        history.append(record)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"epoch": epoch, "train_loss": record["train_loss"],
                          "val_local_pq": validation["pq_image_mean"],
                          "val_dice": validation["dice_image_mean"]}), flush=True)
        score = (validation["pq_image_mean"], validation["dice_image_mean"])
        if score > best:
            best = score
            torch.save({"version": VERSION, "model": model.state_dict(),
                        "width": args.width, "patch": args.patch,
                        "threshold": args.threshold, "min_area": args.min_area,
                        "seed": args.seed, "annotation_sha256": data.annotation_sha256,
                        "epoch": epoch, "validation": validation}, args.out / "baseline.pt")
    print(f"Best local validation PQ (image mean): {best[0]:.4f}; "
          f"weights: {args.out / 'baseline.pt'}")
    if not getattr(args, "no_visualize", False):
        try:
            visualize_checkpoint(args, data, device)
        except Exception as error:
            raise RuntimeError(
                f"Training finished and the best checkpoint is saved in {args.out}, "
                "but result visualization failed. Run 'visualize' to retry without training."
            ) from error


def checked_model(args, data: CompetitionData, device: torch.device):
    model, checkpoint = load_model(args.out / "baseline.pt", device)
    if checkpoint["annotation_sha256"] != data.annotation_sha256:
        raise ValueError("Checkpoint was trained on a different annotation JSON")
    return model, checkpoint


def evaluate_checkpoint(args, data: CompetitionData, device: torch.device) -> None:
    model, checkpoint = checked_model(args, data, device)
    splits = get_splits(data, args.out, checkpoint["seed"])
    results = evaluate(model, data, splits[args.split], checkpoint["patch"],
                       checkpoint["threshold"], checkpoint["min_area"], device)
    results.update({"split": args.split, "checkpoint_epoch": checkpoint["epoch"]})
    path = args.out / f"{args.split}-metrics.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def plotting():
    # Plotting is optional for smoke checks and runs using --no-visualize.
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise RuntimeError("Result visualization requires matplotlib") from error
    return plt


def history_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = next((raw[key] for key in ("history", "epochs", "records")
                    if isinstance(raw.get(key), (list, dict))), raw)
    if isinstance(raw, dict):
        records = []
        for epoch, record in raw.items():
            if isinstance(record, dict):
                item = dict(record)
                if "epoch" not in item:
                    try:
                        item["epoch"] = int(epoch)
                    except (TypeError, ValueError):
                        continue
                records.append(item)
        return records
    if isinstance(raw, list):
        return [record for record in raw if isinstance(record, dict)]
    raise ValueError("history.json must contain a list or dictionary of epoch records")


def finite_number(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def plot_training(out: Path, checkpoint: dict, plt, destination: Path | None = None) -> None:
    records = history_records(out / "history.json")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    series = [(axes[0], "train_loss", "Training loss", "#333333")]
    for axis, key, label, color in series:
        points = [(finite_number(item.get("epoch")), finite_number(item.get(key)))
                  for item in records]
        points = sorted((x, y) for x, y in points if x is not None and y is not None)
        if points:
            axis.plot(*zip(*points), marker="o", label=label, color=color)
        else:
            axis.text(0.5, 0.5, "Training loss is not available in history.json",
                      ha="center", va="center", transform=axis.transAxes, fontsize=9)
        axis.set_ylabel("BCE + soft Dice loss")
        axis.set_title("Training loss")
    for key, label, color in (("pq_image_mean", "Local PQ", "#a14832"),
                              ("dice_image_mean", "Dice", "#527739")):
        points = []
        for item in records:
            validation = item.get("validation")
            if not isinstance(validation, dict):
                continue
            x, y = finite_number(item.get("epoch")), finite_number(validation.get(key))
            if x is not None and y is not None:
                points.append((x, y))
        if points:
            axes[1].plot(*zip(*sorted(points)), marker="o", label=label, color=color)
    if axes[1].lines:
        axes[1].legend(loc="best")
    else:
        axes[1].text(0.5, 0.5, "Validation metrics are not available in history.json",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Image-mean score")
    axes[1].set_title("Local validation metrics; not official Kaggle scores")
    best_epoch = finite_number(checkpoint.get("epoch"))
    for axis in axes:
        if best_epoch is not None:
            axis.axvline(best_epoch, color="#777777", linestyle="--", linewidth=1)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    epoch_note = f"; selected checkpoint: epoch {int(best_epoch)}" if best_epoch is not None else ""
    figure.suptitle(f"Baseline {checkpoint.get('version', VERSION)}{epoch_note}")
    try:
        figure.savefig(destination if destination is not None else out / "training.png", dpi=180)
    finally:
        plt.close(figure)


def instance_overlay(labels: np.ndarray, plt) -> np.ndarray:
    # A uint8 lookup keeps full-resolution RGBA overlays small in CPU-only sessions.
    palette = (plt.get_cmap("tab20")(np.arange(20)) * 255).astype(np.uint8)
    colors = palette[(np.maximum(labels, 1) - 1) % 20]
    colors[:, :, 3] = np.where(labels > 0, 166, 0).astype(np.uint8)
    return colors


def visualize_checkpoint(args, data: CompetitionData, device: torch.device) -> None:
    """Overwrite two result images using the saved best model and validation split."""
    model, checkpoint = checked_model(args, data, device)
    if not (args.out / "split.json").is_file():
        raise FileNotFoundError("split.json is required to visualize the saved validation split")
    splits = get_splits(data, args.out, checkpoint["seed"])
    names = sorted(splits["val"])
    if not names:
        raise ValueError("The saved validation split is empty")
    # Evenly spaced sorted names are deterministic and independent of prediction quality.
    positions = np.linspace(0, len(names) - 1, min(3, len(names)), dtype=int)
    selected = [names[int(position)] for position in positions]
    plt = plotting()
    figure, axes = plt.subplots(len(selected), 3, figsize=(15, 5 * len(selected)),
                                squeeze=False)
    epoch = checkpoint.get("epoch", "unknown")
    figure.suptitle(
        f"Baseline {checkpoint.get('version', VERSION)}: validation examples | "
        f"best epoch {epoch} | threshold {checkpoint['threshold']:g} | "
        f"minimum area {checkpoint['min_area']} px", fontsize=13)
    training_stage = args.out / ".training.tmp.png"
    predictions_stage = args.out / ".predictions.tmp.png"
    try:
        plot_training(args.out, checkpoint, plt, destination=training_stage)
        for row, name in enumerate(selected):
            image = np.asarray(Image.open(data.paths[name]).convert("L"), dtype=np.uint8)
            height, width = image.shape
            record = data.records[name][0]
            truth = data.instance_masks(record, height, width)
            truth_labels = np.zeros(image.shape, dtype=np.int32)
            for instance_id, mask in enumerate(truth, 1):
                truth_labels[mask] = instance_id
            probability = predict_probability(model, image, checkpoint["patch"], device)
            predicted = instance_labels(probability, checkpoint["threshold"], checkpoint["min_area"])
            scores = score_record(predicted, truth)
            for axis in axes[row]:
                axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none")
                axis.set_axis_off()
            axes[row, 1].imshow(instance_overlay(truth_labels, plt), interpolation="none")
            axes[row, 2].imshow(instance_overlay(predicted, plt), interpolation="none")
            axes[row, 0].set_title(f"{name}\nOriginal ({width} x {height})", fontsize=10)
            axes[row, 1].set_title(
                f"Human annotation: record {record['id']}\n{len(truth)} instances", fontsize=10)
            metric_note = ("Both masks empty; local empty-case convention = 1"
                           if not truth and not predicted.any() else
                           f"Local PQ {scores['pq']:.3f} | Dice {scores['dice']:.3f}")
            axes[row, 2].set_title(
                f"Model prediction: {int(predicted.max())} instances\n"
                f"{metric_note}", fontsize=10)
        figure.text(0.5, 0.015,
                    "Selection: up to 3 evenly spaced sorted validation filenames; first annotation record by string-sorted ID.\n"
                    "Scores refer to the displayed annotation only; other annotators may disagree. "
                    "Colors identify instances within each panel, not matched identities.\n"
                    "Overlapping annotation pixels show one display color; scoring retains all instance masks.\n"
                    "Local evaluation; not official Kaggle scores. The saved checkpoint and numerical results are unchanged.",
                    ha="center", va="bottom", fontsize=9)
        figure.subplots_adjust(top=0.92, bottom=0.10, hspace=0.18, wspace=0.03)
        figure.savefig(predictions_stage, dpi=220)
        # Replace the published images only after both new figures were generated.
        training_stage.replace(args.out / "training.png")
        predictions_stage.replace(args.out / "predictions.png")
    finally:
        plt.close(figure)
        training_stage.unlink(missing_ok=True)
        predictions_stage.unlink(missing_ok=True)
    print(f"Result figures: {args.out / 'training.png'}, {args.out / 'predictions.png'}")
    print("Visualization uses the saved best checkpoint and deterministic validation examples; no training.")


def predict_test(args, data: CompetitionData, device: torch.device) -> None:
    model, checkpoint = checked_model(args, data, device)
    if not data.test_dir.is_dir():
        raise FileNotFoundError(data.test_dir)
    paths = sorted(path for path in data.test_dir.iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        raise ValueError("No test images found")
    output = args.out / "submission.csv"
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filament_id", "segmentation_rle"))
        for index, path in enumerate(paths, 1):
            image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            if image.shape != (2048, 2048):
                raise ValueError(f"Competition submission requires 2048x2048: {path}")
            probability = predict_probability(model, image, checkpoint["patch"], device)
            labels = instance_labels(probability, checkpoint["threshold"], checkpoint["min_area"])
            for instance_id in range(1, int(labels.max()) + 1):
                encoded = mask_utils.encode(np.asfortranarray((labels == instance_id).astype(np.uint8)))
                writer.writerow((f"{path.stem}_{instance_id}", encoded["counts"].decode("ascii")))
            if index % 25 == 0 or index == len(paths):
                print(f"Predicted {index}/{len(paths)} test images", flush=True)
    print(f"Submission CSV: {output}")


def smoke(device: torch.device) -> None:
    seed_everything(42)
    model = UNet(width=4).to(device)
    image = torch.rand(1, 1, 32, 32, device=device)
    target = torch.zeros_like(image)
    target[:, :, 8:20, 10:24] = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    prediction = model(image)
    assert prediction.shape == target.shape
    loss = loss_fn(prediction, target)
    loss.backward()
    optimizer.step()
    gt = np.zeros((32, 32), dtype=bool)
    gt[8:20, 10:24] = True
    labels = ndimage.label(gt, structure=ndimage.generate_binary_structure(2, 2))[0]
    assert abs(score_record(labels, [gt])["pq"] - 1) < 1e-9
    print(f"Smoke check passed on {device}; parameter count: {sum(p.numel() for p in model.parameters()):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("smoke", "train", "evaluate", "predict", "visualize"):
        sub = subparsers.add_parser(action)
        if action != "smoke":
            sub.add_argument("--data", type=Path, required=True, help="MAGFiLO_1.0_Kaggle_2026 directory")
            sub.add_argument("--out", type=Path, default=Path("baseline-output"))
        if action == "train":
            sub.add_argument("--epochs", type=int, default=8)
            sub.add_argument("--batch-size", type=int, default=2)
            sub.add_argument("--patch", type=int, default=512)
            sub.add_argument("--width", type=int, default=16)
            sub.add_argument("--lr", type=float, default=1e-3)
            sub.add_argument("--seed", type=int, default=2026)
            sub.add_argument("--workers", type=int, default=0)
            sub.add_argument("--val-every", type=int, default=2)
            sub.add_argument("--threshold", type=float, default=0.5)
            sub.add_argument("--min-area", type=int, default=32)
            sub.add_argument("--no-visualize", action="store_true",
                             help="Skip result figures for disposable diagnostic runs")
        if action == "evaluate":
            sub.add_argument("--split", choices=("train", "val", "test"), default="test")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.action == "smoke":
        smoke(device)
        return
    data = CompetitionData(args.data)
    if args.action == "train":
        train(args, data, device)
    elif args.action == "evaluate":
        evaluate_checkpoint(args, data, device)
    elif args.action == "visualize":
        visualize_checkpoint(args, data, device)
    else:
        predict_test(args, data, device)


if __name__ == "__main__":
    main()
