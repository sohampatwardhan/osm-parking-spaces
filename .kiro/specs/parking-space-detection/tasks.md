# Tasks: parking-space detection

See `requirements.md` and `design.md`.

- [x] Package `osm_parking_spaces` with a `detect` CLI (`--way` or `--geojson`, `--out-dir`).
- [x] Read a closed OSM way from API 0.6, or one GeoJSON polygon. Reject anything else.
- [x] Fetch and mosaic Natural2020 in EPSG:6527 at 1 ft/pixel. Reproject with pyproj.
- [x] Write `stalls.geojson`, create-only `stalls.osc`, and `preview.png`.
- [x] CI: ruff + pytest on Ubuntu and macOS. Unit tests stay offline unless marked `live`.
- [x] README: install, command, NJOGIS attribution, OSM ODbL note, and the no-upload rule.
- [ ] Replace lot-wide rectangles with per-row orientation, parallelogram stalls, pitch-and-depth snapping, and gap fill for stalls hidden by cars.
- [ ] Reject hatched islands, arrows, and rows with fewer than three consistent dividers.
- [ ] Share divider-corner nodes in GeoJSON and in the create-only `.osc`.
- [ ] Hand-label ground truth for ways 1329737820 and 1329737821 from the native imagery.
- [ ] Add `evaluate` (IoU ≥ 0.5, one-to-one, precision / recall / F1 / mean centroid offset) and an offline test.
- [ ] Re-run both ShopRite lots. Commit GeoJSON, `.osc`, preview PNG, and a true-positive / false-positive / missed overlay.
