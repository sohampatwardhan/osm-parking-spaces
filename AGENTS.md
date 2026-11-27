# Repository rules

This repo builds a Python CLI that proposes OpenStreetMap `amenity=parking_space` polygons. Specs live in `.kiro/specs/parking-space-detection/`. Steering lives in `.kiro/steering/`.

## Mapping safety

- Never upload to OpenStreetMap. Never create a changeset. Never read or require OSM credentials.
- The only OSM HTTP call is a read-only GET of a way (API 0.6). Send a User-Agent that names this project.
- OsmChange output is create-only, with negative ids, `amenity=parking_space`, and `source=NJOGIS Natural Color Imagery 2020`.
- Do not add further OSM tags unless the spec changes. Measured width and length may appear in GeoJSON properties only.
- Stall polygons are candidates. The README must say that a person reviews them before upload.
- Keep the NJOGIS public-domain attribution and the OSM ODbL note in the README.

## Runtime

- Target arm64 macOS and Linux. Dependencies must be pip wheels. No GDAL or rasterio.
- Default imagery is the NJOGIS Natural2020 WMS at native 1 ft resolution in EPSG:6527.
- Reproject with pyproj. Inputs and GeoJSON outputs are WGS84.

## Tests and CI

- Unit tests use files in `tests/fixtures/`. They must pass with no network.
- Live WMS or OSM tests use the `live` pytest marker and stay out of the default run.
- `ruff check` and `pytest` should pass before a change is merged. CI runs both on Ubuntu and macOS.
