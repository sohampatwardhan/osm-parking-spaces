# Design: learned stall detector

Tracked by [issue #3](https://github.com/sohampatwardhan/osm-parking-spaces/issues/3).

## Why this model

At 1 ft per pixel a stall is about 9 by 18 pixels. An instance-segmentation head or an oriented-box detector usually starts from an ImageNet backbone. Those weights are research-licensed and are not used here. A small U-Net trained from scratch predicts a stall-interior mask. Neighbours are rasterised with a one-pixel gap so connected components stay separate. Each component becomes a rotated rectangle, then rectangles in the same row are snapped to one pitch and one depth and written with shared divider corners. That reuses the classical export and evaluation path without copying a pretrained encoder.

## Dataset

```
Overpass / API 0.6 (read-only)
        │  amenity=parking + amenity=parking_space in New Jersey
        ▼
drop holdout ways and anything intersecting their buffer
drop lots below the stall-count or coverage filter
        │
        ▼
Natural2020 GetMap (EPSG:6527, 1 ft)
raster stall mask (gap between neighbours)
        │
        ▼
chips on disk + manifest.json (train / val by lot id)
```

The split is by parking-lot way id, not by chip, so chips from one lot stay in one side. The two ShopRite lots are never written.

Coverage is the area of the stall polygons divided by the area of the parent parking polygon. Lots outside the configured band are treated as incomplete or as a single polygon drawn over the whole lot.

## Network

Three-level U-Net, one input channel group (RGB), one logit channel, no pretrained tensors. Loss is binary cross-entropy plus a soft Dice term. Chips are fixed-size crops of the lot tile.

`pick_device()` returns `mps` when `torch.backends.mps.is_available()`, otherwise `cpu`.

Checkpoints store the epoch, model state, optimiser state, and the best validation dice. `metrics.jsonl` gets one record per epoch. Starting `train` again with the same `--out` directory resumes from `checkpoint.pt`.

Core ML export is a separate command. It imports coremltools only when asked.

## Inference

The full lot image is padded to a multiple of 16 and run in one forward pass when it fits, otherwise in tiles. The sigmoid mask is thresholded. Connected components that are stall-sized become oriented rectangles. Components in one row (similar angle, similar across-row position) are replaced by a lattice at the median pitch and median depth so corners are shared. The lattice is clipped to the parking polygon and emitted as the same `Stall` objects the classical detector uses.

## CLI

```
osm-parking-spaces dataset build --out DIR [--max-lots N]
osm-parking-spaces train --data DIR --out DIR [--epochs N] [--device auto|cpu|mps]
osm-parking-spaces export-coreml --weights FILE --out FILE
osm-parking-spaces detect --detector {classical,ml} --weights FILE --way ID --out-dir DIR
osm-parking-spaces evaluate --detected FILE --truth FILE
```

`--detector` defaults to `classical`. `--weights` is required for `ml`.

## Packaging

Base dependencies stay the wheels used by the classical tool. PyTorch is the `[ml]` extra. coremltools is the `[coreml]` extra so CI does not download it. `data/` and `runs/` are gitignored. Weights are not committed.

## Licences

See `docs/licences.md`. Imagery is public domain. Labels are ODbL. The network is trained from scratch, so no ImageNet licence applies. A checkpoint is derived from ODbL geometries and should be shared under ODbL if it is distributed. Code stays MIT.

## Decisions

| Decision | Why |
| --- | --- |
| Stall mask, not a box detector | Box detectors want a pretrained backbone. A mask plus the existing row snap stays small and licence-clean. |
| Train from scratch | ImageNet weights are not cleared for an ODbL geometry product. |
| Holdout by way id and by buffer intersection | A neighbouring lot that overlaps the ShopRite pavement must not leak into training. |
| Lot-level split | Random chips would put the same painted row in train and val. |
| Resume from checkpoint | The long run is on the owner's Mac, not in CI. |
