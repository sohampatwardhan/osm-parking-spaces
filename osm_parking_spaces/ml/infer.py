"""Run the stall U-Net and turn the mask into shared-corner stalls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from osm_parking_spaces.errors import ParkingError
from osm_parking_spaces.ml.device import pick_device
from osm_parking_spaces.ml.model import build_model
from osm_parking_spaces.ml.postprocess import stalls_from_mask


def load_model(weights: Path, device):
    """Return (model, pixel threshold) from a checkpoint. v3 files carry no arch or threshold."""
    import torch

    path = Path(weights)
    if not path.exists():
        raise ParkingError(f"Model weights not found: {path}")
    saved = torch.load(path, map_location=device, weights_only=False)
    model = build_model(saved.get("arch", "unet"))
    model.load_state_dict(saved["model"])
    model.to(device)
    model.eval()
    return model, float(saved.get("threshold", 0.5))


def detect_stalls_ml(
    image_bgr: np.ndarray,
    lot: Polygon,
    weights: Path,
    gsd_ft: float = 1.0,
    device_name: str = "auto",
):
    """v3: threshold the mask and turn components into stalls."""
    device = pick_device(device_name)
    model, threshold = load_model(weights, device)
    probs = predict_probs(model, image_bgr, device)
    mask = (probs >= threshold).astype(np.uint8)
    return stalls_from_mask(mask, lot, gsd_ft=gsd_ft)


def detect_stalls_hybrid(
    image_bgr: np.ndarray,
    lot: Polygon,
    weights: Path,
    gsd_ft: float = 1.0,
    device_name: str = "auto",
):
    """v4: classical stalls checked, extended and trimmed by the learned probability map."""
    from osm_parking_spaces.detect import detect_stalls
    from osm_parking_spaces.ml.fusion import FusionConfig, fuse

    device = pick_device(device_name)
    model, threshold = load_model(weights, device)
    probs = predict_probs(model, image_bgr, device, tta=True)
    config = FusionConfig.load_for(weights, pixel=threshold)
    return fuse(detect_stalls(image_bgr, lot, gsd_ft=gsd_ft), probs, lot, gsd_ft, config)


def predict_probs(model, image_bgr: np.ndarray, device, tta: bool = False) -> np.ndarray:
    """Sigmoid probabilities, shape (H, W). `tta` averages the eight flips and quarter-turns."""
    import torch

    tensor = torch.from_numpy(np.transpose(image_bgr.astype(np.float32) / 255.0, (2, 0, 1)))
    tensor = tensor.unsqueeze(0).to(device)
    views = [(k, flip) for k in range(4) for flip in (False, True)] if tta else [(0, False)]
    total = None
    with torch.no_grad():
        for k, flip in views:
            view = torch.rot90(tensor, k, dims=(2, 3))
            if flip:
                view = torch.flip(view, dims=(3,))
            out = torch.sigmoid(model(_pad16(view)))[..., : view.shape[2], : view.shape[3]]
            if flip:
                out = torch.flip(out, dims=(3,))
            out = torch.rot90(out, -k, dims=(2, 3))
            total = out if total is None else total + out
    probs = (total / len(views))[0, 0].detach().cpu().numpy()
    return probs[: image_bgr.shape[0], : image_bgr.shape[1]].astype(np.float32)


def _pad16(tensor):
    import torch.nn.functional as F

    height, width = tensor.shape[-2:]
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return tensor


def predict_mask(model, image_bgr: np.ndarray, device, threshold: float = 0.5) -> np.ndarray:
    return (predict_probs(model, image_bgr, device) >= threshold).astype(np.uint8)


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
    model, _threshold = load_model(weights, device)
    example = torch.rand(1, 3, chip, chip)
    traced = torch.jit.trace(model, example)
    converted = ct.convert(traced, inputs=[ct.TensorType(name="image", shape=example.shape)])
    converted.save(str(out_path))
