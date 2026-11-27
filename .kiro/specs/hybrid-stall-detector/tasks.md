# Tasks: hybrid stall detector

See `requirements.md` and `design.md`. Issue #4.

- [ ] Dataset builder: split and cached Overpass, retry and back-off, STRtree pairing, seeded shuffle, blank-tile filter, lot and stall polygons and angled fraction in each tile.
- [ ] Swin-UNet with the SatlasPretrain aerial encoder. Download with a SHA-256 check. `docs/licences.md` entry.
- [ ] Training: rotation and flip augmentation, angled-lot oversampling, 256 px chips, best-validation checkpoint, threshold sweep on validation.
- [ ] Fusion: confirm or reject, trim, fill, seed. `--detector hybrid`.
- [ ] Evaluation: mean offset fix and the `evaluate-way` command.
- [ ] Tune fusion parameters on the validation lots.
- [ ] Synthetic tests for fusion, augmentation, oversampling, the offset and the dataset helpers. No network.
- [ ] Run on the Mac. Score both test ways next to v2 and v3. Commit example outputs and `docs/FINDINGS.md`. Do not commit datasets or checkpoints.
