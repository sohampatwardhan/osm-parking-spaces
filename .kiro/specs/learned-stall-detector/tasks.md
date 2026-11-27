# Tasks: learned stall detector

See `requirements.md` and `design.md`. Issue #3.

- [ ] Dataset builder: read-only OSM labels, Natural2020 tiles, stall masks, lot-level split, count manifest.
- [ ] Holdout filter for ways 1329737820 and 1329737821 plus a buffer, with an offline test.
- [ ] Quality filter on stall count and coverage of the parent parking polygon.
- [ ] From-scratch U-Net, MPS with CPU fallback, resumable checkpoint, `metrics.jsonl`.
- [ ] Optional Core ML export behind the `[coreml]` extra.
- [ ] Mask post-processing into shared-node stalls. `--detector {classical,ml}` and `--weights`.
- [ ] `docs/licences.md` and README commands for dataset, train, evaluate, and detect.
- [ ] CI: rasterisation, holdout, post-processing, and a few CPU training steps on a fixture. No live network.
- [ ] If the environment allows: a real New Jersey sample and a short CPU training run, metrics labelled as that run. Do not commit the weights or the imagery tiles.
