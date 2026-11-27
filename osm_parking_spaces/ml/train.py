"""Train a stall-mask network. Checkpoints resume; metrics append to metrics.jsonl.

`checkpoint.pt` is the last epoch and carries the optimiser, so the same command resumes.
`best.pt` is the epoch with the highest validation Dice on whole validation tiles.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from osm_parking_spaces.errors import ParkingError
from osm_parking_spaces.ml.augment import lot_weights, sample_chip
from osm_parking_spaces.ml.dataset import ANGLED_LOT, load_tile
from osm_parking_spaces.ml.device import pick_device
from osm_parking_spaces.ml.model import build_model, encoder_param_names


def train(
    data_dir: Path,
    out_dir: Path,
    epochs: int = 20,
    chip: int = 128,
    batch_size: int = 4,
    device_name: str = "auto",
    max_steps: int | None = None,
    crops_per_lot: int = 4,
    seed: int = 0,
    arch: str = "unet",
    encoder_weights: Path | None = None,
    lr: float = 1e-3,
    encoder_lr: float = 1e-4,
    oversample: float = 3.0,
    rotate: bool = True,
) -> dict:
    import torch

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise ParkingError(f"No manifest at {manifest_path}. Run dataset build first.")
    manifest = json.loads(manifest_path.read_text())
    tiles = manifest.get("tiles_detail") or []
    if not tiles:
        raise ParkingError("The dataset manifest has no tiles.")
    train_items = [item for item in tiles if item.get("split", "train") != "val"] or tiles
    val_items = [item for item in tiles if item.get("split") == "val"]
    device = pick_device(device_name)
    torch.manual_seed(seed)
    model = build_model(arch, encoder_weights=encoder_weights).to(device)
    optimiser = _optimiser(model, lr, encoder_lr, torch)
    start_epoch = 0
    best_dice = -1.0
    checkpoint_path = out_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved.get("arch", "unet") != arch:
            raise ParkingError(
                f"{checkpoint_path} is a {saved.get('arch', 'unet')} run, not {arch}. "
                "Use a new --out directory."
            )
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        start_epoch = int(saved["epoch"]) + 1
        best_dice = float(saved.get("best_dice", -1.0))
    train_cache = {item["file"]: load_tile(data_dir / item["file"]) for item in train_items}
    val_cache = {item["file"]: load_tile(data_dir / item["file"]) for item in val_items}
    weights = lot_weights(
        [item.get("angled_fraction", 0.0) for item in train_items], ANGLED_LOT, oversample
    )
    total_epochs = start_epoch + epochs
    metrics_path = out_dir / "metrics.jsonl"
    last = {"epoch": start_epoch - 1, "train_loss": None}
    for epoch in range(start_epoch, total_epochs):
        scale = 0.5 * (1.0 + math.cos(math.pi * epoch / max(total_epochs, 1)))
        for group in optimiser.param_groups:
            group["lr"] = group["base_lr"] * (0.05 + 0.95 * scale)
        loss = _epoch(
            model,
            optimiser,
            [train_cache[item["file"]] for item in train_items],
            weights,
            chip,
            batch_size,
            device,
            max_steps,
            crops_per_lot,
            seed + epoch,
            rotate,
        )
        val_dice = _validation_dice(model, list(val_cache.values()), device) if val_items else None
        record = {
            "epoch": epoch,
            "train_loss": loss,
            "val_dice": val_dice,
            "lr": optimiser.param_groups[-1]["lr"],
            "device": str(device),
        }
        if val_dice is not None and val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {"arch": arch, "epoch": epoch, "model": model.state_dict(), "val_dice": val_dice},
                out_dir / "best.pt",
            )
            record["best"] = True
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        torch.save(
            {
                "arch": arch,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "best_dice": best_dice,
            },
            checkpoint_path,
        )
        last = record
        if max_steps is not None:
            break
    return last


def _optimiser(model, lr: float, encoder_lr: float, torch):
    pretrained = encoder_param_names(model)
    fast = [p for n, p in model.named_parameters() if n not in pretrained]
    slow = [p for n, p in model.named_parameters() if n in pretrained]
    groups = [{"params": fast, "lr": lr, "base_lr": lr}]
    if slow:
        groups.append({"params": slow, "lr": encoder_lr, "base_lr": encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def _epoch(
    model,
    optimiser,
    tiles,
    weights,
    chip,
    batch_size,
    device,
    max_steps,
    crops_per_lot,
    seed,
    rotate,
):
    import torch
    from torch import nn

    model.train()
    rng = np.random.default_rng(seed)
    count = len(tiles) * crops_per_lot
    if max_steps is not None:
        count = min(count, max_steps * batch_size)
    picks = rng.choice(len(tiles), size=count, replace=True, p=weights)
    bce = nn.BCEWithLogitsLoss()
    total = 0.0
    steps = 0
    for start in range(0, count, batch_size):
        batch = [
            sample_chip(*tiles[int(index)], chip=chip, rng=rng, rotate=rotate)
            for index in picks[start : start + batch_size]
        ]
        total += _step(
            model,
            optimiser,
            bce,
            [item[0] for item in batch],
            [item[1] for item in batch],
            device,
            torch,
        )
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return total / max(steps, 1)


def _step(model, optimiser, bce, batch_x, batch_y, device, torch):
    images = torch.from_numpy(np.stack(batch_x)).to(device)
    masks = torch.from_numpy(np.stack(batch_y)).to(device)
    optimiser.zero_grad(set_to_none=True)
    logits = model(images)
    loss = bce(logits, masks) + _dice_loss(logits, masks, torch)
    loss.backward()
    optimiser.step()
    return float(loss.detach().cpu())


def _validation_dice(model, tiles, device, threshold: float = 0.5):
    """Mean per-tile Dice of whole validation tiles at one threshold."""
    import torch

    model.eval()
    scores = []
    with torch.no_grad():
        for image, mask in tiles:
            tensor = torch.from_numpy(
                np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))[None]
            ).to(device)
            probs = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
            scores.append(dice_score(probs >= threshold, mask))
    model.train()
    return float(np.mean(scores)) if scores else None


def dice_score(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    denom = int(pred.sum()) + int(truth.sum())
    if denom == 0:
        return 1.0
    return 2.0 * int((pred & truth).sum()) / denom


def _dice_loss(logits, target, torch):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + 1) / (denom + 1)
    return 1 - dice.mean()
