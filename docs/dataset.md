# Dataset split

Lots are assigned by parking-way id:

- `way_id % 5 == 0` is validation
- every other id is training
- if a small download would otherwise have no validation lot, the last lot is moved to validation
- ways `1329737820` and `1329737821` are rejected before the split, as is any lot that intersects a 150 m buffer around those two polygons

Tiles are one Natural2020 mosaic and one stall mask per accepted lot (`tiles/<way_id>.npz`). Chips are cropped at train time. Imagery and checkpoints are not committed.

## Sample collected on a CPU build machine

This is not the statewide set. Command:

```bash
osm-parking-spaces dataset --out data/nj-sample --max-lots 8 \
  --bbox 40.30,-74.80,40.50,-74.30
```

| | |
| --- | --- |
| Lots | 8 |
| Stalls | 162 |
| Tiles | 8 |
| Train lots | 6 |
| Val lots | 2 |
| Disk | about 0.5 MB |

Way ids written: 262302355 (val), 889089306, 889089307, 924968281, 980739577, 1000374050 (val), 1107580731, 1107580767. Neither holdout id is in that list.

A 30-epoch CPU resume on that sample (6 + 24 epochs) lowered train loss from about 1.37 to about 0.55. The checkpoint is about 1.5 MB and was not committed. Held-out detection with that checkpoint is not useful yet: way 1329737820 scored 0 / 0 / 0 (25 boxes, no IoU ≥ 0.5 match) and way 1329737821 scored 0.03 / 0.01 / 0.01 (one match). That is a tiny CPU sample, not a Mac GPU training run.
