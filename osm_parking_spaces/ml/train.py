"""Train the stall U-Net. Checkpoints resume; metrics append to metrics.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from osm_parking_spaces.errors import ParkingError
from osm_parking_spaces.ml.dataset import load_tile
from osm_parking_spaces.ml.device import pick_device
from osm_parking_spaces.ml.model import build_model


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
) -> dict:
    import torch
    from torch import nn

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
    device = pick_device(device_name)
    torch.manual_seed(seed)
    model = build_model().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_epoch = 0
    checkpoint_path = out_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        start_epoch = int(saved["epoch"]) + 1
    bce = nn.BCEWithLogitsLoss()
    metrics_path = out_dir / "metrics.jsonl"
    last = {"epoch": start_epoch - 1, "train_loss": None}
    for epoch in range(start_epoch, start_epoch + epochs):
        train_tiles = [item for item in tiles if item.get("split", "train") != "val"] or tiles
        val_tiles = [item for item in tiles if item.get("split") == "val"]
        loss = _epoch(
            model,
            optimiser,
            bce,
            train_tiles,
            data_dir,
            chip,
            batch_size,
            device,
            max_steps,
            crops_per_lot,
            seed + epoch,
        )
        val_dice = _validation_dice(model, val_tiles, data_dir, chip, device) if val_tiles else None
        record = {
            "epoch": epoch,
            "train_loss": loss,
            "val_dice": val_dice,
            "device": str(device),
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optimiser": optimiser.state_dict()},
            checkpoint_path,
        )
        last = record
        if max_steps is not None:
            break
    return last


def _epoch(
    model,
    optimiser,
    bce,
    tiles,
    data_dir,
    chip,
    batch_size,
    device,
    max_steps,
    crops_per_lot,
    seed,
):
    import torch

    model.train()
    rng = np.random.default_rng(seed)
    samples = []
    limit = None if max_steps is None else max_steps * batch_size
    for index in rng.permutation(len(tiles)):
        image, mask = load_tile(data_dir / tiles[int(index)]["file"])
        for _crop in range(crops_per_lot):
            samples.append(_random_crop(image, mask, chip, rng))
            if limit is not None and len(samples) >= limit:
                break
        if limit is not None and len(samples) >= limit:
            break
    total = 0.0
    steps = 0
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        total += _step(
            model,
            optimiser,
            bce,
            [item[0] for item in chunk],
            [item[1] for item in chunk],
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


def _validation_dice(model, tiles, data_dir, chip, device):
    import torch

    model.eval()
    scores = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for item in tiles:
            image, mask = load_tile(data_dir / item["file"])
            crop_x, crop_y = _random_crop(image, mask, chip, rng)
            images = torch.from_numpy(crop_x[None, ...]).to(device)
            target = torch.from_numpy(crop_y[None, ...]).to(device)
            logits = model(images)
            scores.append(float(1 - _dice_loss(logits, target, torch)))
    model.train()
    return sum(scores) / len(scores) if scores else None


def _dice_loss(logits, target, torch):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + 1) / (denom + 1)
    return 1 - dice.mean()


def _random_crop(image, mask, chip: int, rng) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape[:2]
    if height < chip or width < chip:
        canvas = np.zeros((max(height, chip), max(width, chip), 3), np.uint8)
        canvas[:height, :width] = image
        mask_canvas = np.zeros((canvas.shape[0], canvas.shape[1]), np.float32)
        mask_canvas[:height, :width] = mask
        image, mask = canvas, mask_canvas
        height, width = mask.shape
    y0 = int(rng.integers(0, height - chip + 1))
    x0 = int(rng.integers(0, width - chip + 1))
    crop = image[y0 : y0 + chip, x0 : x0 + chip]
    label = mask[y0 : y0 + chip, x0 : x0 + chip].astype(np.float32)
    tensor = np.transpose(crop.astype(np.float32) / 255.0, (2, 0, 1))
    return tensor, label[None, ...]
