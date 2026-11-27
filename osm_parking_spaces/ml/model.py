"""Stall-interior networks: a small U-Net from scratch, and a Swin-UNet on a pretrained encoder.

Inputs are float BGR images in 0..1, shape (N, 3, H, W), as the OpenCV loader gives them.
Output is one logit per pixel.
"""

from __future__ import annotations

from pathlib import Path

ARCHS = ("unet", "swin-unet")


def build_model(arch: str = "unet", base: int = 8, encoder_weights: Path | None = None):
    """Build a network. torch is imported lazily so the classical install does not need it.

    `encoder_weights` is the Satlas `aerial_swinb_si.pth` file. It is read only for the
    `swin-unet` encoder and only when given. A saved checkpoint already holds every tensor.
    """
    if arch == "unet":
        return _build_unet(base)
    if arch == "swin-unet":
        return _build_swin_unet(encoder_weights)
    raise ValueError(f"Unknown architecture {arch!r}. Use one of {ARCHS}.")


def encoder_param_names(model) -> set[str]:
    """Names of the pretrained-encoder parameters, so training can give them a lower rate."""
    return {name for name, _ in model.named_parameters() if name.startswith("encoder.")}


def _build_swin_unet(encoder_weights: Path | None):
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torchvision.models import swin_v2_b

    from osm_parking_spaces.ml.encoder import load_encoder_state

    def block(cin: int, cout: int):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class SwinUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = swin_v2_b().features
            if encoder_weights is not None:
                state = load_encoder_state(Path(encoder_weights))
                self.encoder.load_state_dict(state, strict=True)
            # Full-resolution stem keeps one-pixel paint that a stride-4 transformer loses.
            self.stem1 = block(3, 32)
            self.stem2 = block(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.dec16 = block(1024 + 512, 256)
            self.dec8 = block(256 + 256, 128)
            self.dec4 = block(128 + 128, 96)
            self.dec2 = block(96 + 64, 64)
            self.dec1 = block(64 + 32, 32)
            self.head = nn.Conv2d(32, 1, 1)

        def forward(self, x):
            height, width = x.shape[-2:]
            pad_h = (32 - height % 32) % 32
            pad_w = (32 - width % 32) % 32
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            s1 = self.stem1(x)
            s2 = self.stem2(self.pool(s1))
            rgb = x[:, [2, 1, 0]]  # Satlas expects RGB
            feats = []
            y = rgb
            for index, layer in enumerate(self.encoder):
                y = layer(y)
                if index in (1, 3, 5, 7):
                    feats.append(y.permute(0, 3, 1, 2))
            f4, f8, f16, f32 = feats

            def up(t, ref):
                return F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)

            d16 = self.dec16(torch.cat([up(f32, f16), f16], dim=1))
            d8 = self.dec8(torch.cat([up(d16, f8), f8], dim=1))
            d4 = self.dec4(torch.cat([up(d8, f4), f4], dim=1))
            d2 = self.dec2(torch.cat([up(d4, s2), s2], dim=1))
            d1 = self.dec1(torch.cat([up(d2, s1), s1], dim=1))
            logits = self.head(d1)
            return logits[..., :height, :width]

    return SwinUNet()


def _build_unet(base: int = 8):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class StallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = Block(3, base)
            self.enc2 = Block(base, base * 2)
            self.enc3 = Block(base * 2, base * 4)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = Block(base * 4, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.dec3 = Block(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.dec2 = Block(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.dec1 = Block(base * 2, base)
            self.head = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)

    return StallUNet()
