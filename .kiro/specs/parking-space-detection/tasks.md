# Tasks: parking-space detection

Implement after this spec is on `main`. See `requirements.md` and `design.md`.

- [x] Package `osm_parking_spaces` with a `detect` CLI (`--way` or `--geojson`, `--out-dir`).
- [x] Read a closed OSM way from API 0.6, or one GeoJSON polygon. Reject anything else.
- [x] Fetch and mosaic Natural2020 in EPSG:6527 at 1 ft/pixel. Reproject with pyproj.
- [x] Detect stalls (paint mask, primary and residual orientations, row pitch, clip, dedupe).
- [x] Write `stalls.geojson`, create-only `stalls.osc`, and `preview.png`.
- [x] Unit tests on committed fixtures, with no network. Mark live WMS/OSM tests opt-in.
- [x] CI: ruff + pytest on Ubuntu and macOS.
- [x] Run ways `1329737820` and `1329737821`. Commit GeoJSON, `.osc`, and preview PNGs under `examples/`.
- [x] README: install, command, NJOGIS attribution, OSM ODbL note, and the no-upload rule.
