# Design: hybrid stall detector (v4)

Tracked by [issue #4](https://github.com/sohampatwardhan/osm-parking-spaces/issues/4).

## Why a hybrid

The v2 line detector is good on straight rows and gap-fills hidden bays, but it misses whole angled rows and sometimes runs a box into the aisle. The v3 mask network could not learn the layout from 421 lots, so it found only the straight outer rows. A mask is still useful as a second opinion: it says whether a box sits on a stall, where a row continues, and where a bay ends.

```
image ──► classical v2 (rows, stalls) ──┐
   │                                    ▼
   └──► Swin-UNet probability map ──► fusion ──► shared-node stalls ──► GeoJSON / .osc / preview
```

## Model

A U-Net decoder on a Swin-v2-Base encoder.

- Encoder: torchvision `swin_v2_b` `features`, loaded from the SatlasPretrain `Aerial_SwinB_SI` checkpoint. Input is RGB divided by 255, as the Satlas documentation says for high-resolution imagery. Our images are BGR, so channels are swapped.
- Feature maps at strides 4, 8, 16 and 32 feed the decoder through skip connections.
- A small convolutional stem at strides 1 and 2, trained from scratch, keeps one-pixel paint detail that a stride-4 transformer loses.
- Output is one logit per pixel (stall interior).
- Loss is binary cross-entropy plus soft Dice. The encoder uses a lower learning rate than the decoder.
- The small U-Net from v3 stays as `arch="unet"` for CI and for back-compatibility.
- Inference runs the whole lot (padded to a multiple of 32) with eight-way flip and quarter-turn test-time augmentation, and averages the probabilities.

## Data and training

The builder is the v3 builder with fixes: the Overpass query is split into a grid of sub-boxes with retry and back-off and an on-disk cache, stalls are paired with lots through an STRtree, lots are shuffled with a fixed seed, tiles outside Natural2020 coverage (blank white) are dropped, and each tile also stores the lot polygon, the OSM stall polygons and the fraction of stalls drawn as angled bays. The holdout exclusion is unchanged.

Training uses:

- Arbitrary-angle rotation (reflect padding), flips, and mild brightness and contrast jitter. The crop is cut larger and rotated, then centre-cropped.
- Chips of 256 px.
- Lots whose angled fraction is at least 0.25 are drawn three times as often.
- Best-validation checkpointing: `best.pt` keeps the epoch with the highest full-tile validation Dice. `checkpoint.pt` is the resumable last epoch.
- A pixel-threshold sweep on the validation tiles. The chosen threshold is written to the checkpoint.

The split is the same lot-level split as v3: way id divisible by five is validation.

## Fusion

All steps work in pixel space on `Stall` objects, and rows are the classical `row_id`.

1. Score. The score of a stall is the mean probability inside the polygon shrunk by 1.5 px.
2. Confirm or reject. Below `reject` the stall is dropped. At or above `keep` it stays. In between it stays only when the row's median score is at least `keep`.
3. Trim. For each row the stall depth axis is sampled across the stall width, from 4 ft before the top edge to 4 ft after the bottom edge. The longest run above the pixel threshold gives a near and a far end. The row median of each end is applied to every stall in the row as one translation, so shared corners stay equal. Shifts are limited to 6 ft and the stall must stay inside the stall depth range.
4. Fill. For each row the end stall is translated by the row pitch, once at each end and into interior gaps wider than 1.6 pitches. The candidate is kept when its mean probability is at least `fill`, it is mostly inside the lot and it does not overlap a kept stall. Its near corners are the neighbour's far corners.
5. Seed (optional). Mask pixels not covered by kept stalls are turned into rows by the v3 post-processing, with a quadrilateral fit that keeps angled bays as parallelograms. Seeds are kept when their mean probability is at least `seed`, and then go through steps 3 and 4.
6. Final non-maximum suppression against overlaps.

Parameters are chosen on the validation lots by running the fusion on each validation tile with its true lot polygon and scoring against the OSM stall polygons at IoU >= 0.5. The test lots are not used for tuning.

## Evaluation

`osm-parking-spaces evaluate-way` runs a detector on one way, writes the three outputs and an `eval.png`, and scores the GeoJSON against the hand-labelled truth. The score matches `evaluate`. The mean offset is the centroid distance in feet. The v3 driver measured it in pixels on WGS84 code and printed nonsense, and that is fixed.

## Licences

The encoder weights are SatlasPretrain `Aerial_SwinB_SI` (Allen Institute for AI), released under ODC-BY 1.0. Its pretraining imagery is NAIP (public domain). Its labels include OSM (ODbL) and other open sources. Recorded in `docs/licences.md`. The checkpoint we train from it is still a produced work from ODbL geometries.

## Decisions

| Decision | Why |
| --- | --- |
| Keep v2 as the proposer | It is the only detector that scores well on rows, and its output already has shared corners. |
| Swin-v2-Base from Satlas | It is the only aerial-imagery encoder with a permissive (ODC-BY) licence and NAIP-only pretraining. ImageNet weights are research-licensed. |
| Row-level trim and fill | One translation per row keeps shared nodes and avoids one noisy stall dragging a box. |
| Tune on validation lots | The two ShopRite lots are the test set and must not influence thresholds. |
| Seeded rows are optional | They can add false positives. Validation decides whether they stay on. |
