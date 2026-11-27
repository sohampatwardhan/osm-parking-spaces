"""Fetch and read the pretrained aerial encoder (SatlasPretrain `Aerial_SwinB_SI`).

Licence: ODC-BY 1.0, Allen Institute for AI. See docs/licences.md. The file is downloaded on
request, checked against a SHA-256, and never committed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

from osm_parking_spaces.errors import ParkingError
from osm_parking_spaces.osm_api import USER_AGENT

ENCODER_URL = "https://huggingface.co/allenai/satlas-pretrain/resolve/main/aerial_swinb_si.pth"
ENCODER_SHA256 = "9c524bb5719401bd4d1b165fa198f6baac3006c53071185f2c2638424a933dba"
ENCODER_NAME = "aerial_swinb_si.pth"
# Keys in the Satlas file look like backbone.backbone.features.1.0.norm1.weight.
_PREFIX = "backbone.backbone.features."


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_encoder(directory: Path, url: str = ENCODER_URL, sha256: str = ENCODER_SHA256) -> Path:
    """Download the weights into `directory` unless a file with the right hash is there."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / ENCODER_NAME
    if target.exists() and sha256_of(target) == sha256:
        return target
    partial = target.with_suffix(".part")
    try:
        with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=120) as r:
            if r.status_code != 200:
                raise ParkingError(f"Encoder download failed: HTTP {r.status_code}")
            with partial.open("wb") as handle:
                for block in r.iter_content(1 << 20):
                    handle.write(block)
    except requests.RequestException as exc:
        raise ParkingError(f"Encoder download failed: {exc}") from exc
    if sha256_of(partial) != sha256:
        partial.unlink(missing_ok=True)
        raise ParkingError("Encoder download did not match the expected SHA-256")
    partial.replace(target)
    return target


def load_encoder_state(path: Path) -> dict:
    """The Swin-v2-Base `features` tensors from a Satlas checkpoint. Heads and FPN are ignored."""
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = {k[len(_PREFIX) :]: v for k, v in saved.items() if k.startswith(_PREFIX)}
    if not state:
        raise ParkingError(
            f"{path} has no {_PREFIX}* tensors. Is it the Satlas aerial Swin-B file?"
        )
    return state
