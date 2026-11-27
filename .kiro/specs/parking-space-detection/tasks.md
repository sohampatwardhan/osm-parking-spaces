# Tasks: parking-space detection

Implement after this spec is on `main`. See `requirements.md` and `design.md`.

- [ ] Package `osm_parking_spaces` with a `detect` CLI (`--way` or `--geojson`, `--out-dir`).
- [ ] Read a closed OSM way from API 0.6, or one GeoJSON polygon. Reject anything else.
- [ ] Fetch and mosaic Natural2020 in EPSG:6527 at 1 ft/pixel. Reproject with pyproj.
- [ ] Detect stalls (paint mask, primary and residual orientations, row pitch, clip, dedupe).
- [ ] Write `stalls.geojson`, create-only `stalls.osc`, and `preview.png`.
- [ ] Unit tests on committed fixtures, with no network. Mark live WMS/OSM tests opt-in.
- [ ] CI: ruff + pytest on Ubuntu and macOS.
- [ ] Run ways `1329737820` and `1329737821`. Commit GeoJSON, `.osc`, and preview PNGs under `examples/`.
- [ ] README: install, command, NJOGIS attribution, OSM ODbL note, and the no-upload rule.
