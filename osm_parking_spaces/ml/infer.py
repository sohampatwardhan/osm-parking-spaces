"""Run the stall U-Net and turn the mask into shared-corner stalls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from osm_parking_spaces.errors import ParkingError
from osm_parking_spaces.ml.device import pick_device
from osm_parking_spaces.ml.model import build_model
from osm_parking_spaces.ml.postprocess import stalls_from_mask


def detect_stalls_ml(
    image_bgr: np.ndarray,
    lot: Polygon,
    weights: Path,
    gsd_ft: float = 1.0,
    device_name: str = "auto",
):
    import torch

    path = Path(weights)
    if not path.exists():
        raise ParkingError(f"Model weights not found: {path}")
    device = pick_device(device_name)
    model = build_model()
    saved = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    model.to(device)
    model.eval()
    mask = predict_mask(model, image_bgr, device)
    return stalls_from_mask(mask, lot, gsd_ft=gsd_ft)


def predict_mask(model, image_bgr: np.ndarray, device) -> np.ndarray:
    import torch

    height, width = image_bgr.shape[:2]
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    padded = np.pad(image_bgr, ((0, pad_h), (0, pad_w), (0, 0)))
    tensor = torch.from_numpy(np.transpose(padded.astype(np.float32) / 255.0, (2, 0, 1)))
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return (probs[:height, :width] >= 0.5).astype(np.uint8)


def export_coreml(weights: Path, out_path: Path, chip: int = 128) -> None:
    import torch

    try:
        import coremltools as ct
    except ImportError as exc:
        raise ParkingError(
            "coremltools is not installed. Install the coreml extra: "
            'python3 -m pip install -e ".[ml,coreml]"'
        ) from exc
    device = torch.device("cpu")
    model = build_model()
    saved = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    model.eval()
    example = torch.rand(1, 3, chip, chip)
    traced = torch.jit.trace(model, example)
    converted = ct.convert(traced, inputs=[ct.TensorType(name="image", shape=example.shape)])
    converted.save(str(out_path))
