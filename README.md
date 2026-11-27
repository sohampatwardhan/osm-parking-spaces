# osm-parking-spaces

Detect painted parking stalls in aerial orthoimagery and write OpenStreetMap
[`amenity=parking_space`](https://wiki.openstreetmap.org/wiki/Tag:amenity%3Dparking_space)
polygons.

Given one parking-area polygon (an OSM way id, or a GeoJSON polygon), the tool
downloads public-domain NJOGIS 2020 imagery, finds stall-sized gaps between
painted lines, and writes three local files:

- `stalls.geojson` — WGS84 polygons
- `stalls.osc` — a create-only OsmChange document (negative ids)
- `preview.png` — stalls drawn on the imagery, with the lot outline

The `.osc` file is for a person to review in an editor such as JOSM. **This
tool never uploads to OpenStreetMap and never asks for OSM credentials.**
Treat every stall as a candidate. Rows with faint paint, cars covering the
lines, and angled or curved bays are often missed or only roughly placed.

## Install

Python 3.11 or newer. The dependencies publish wheels for Apple Silicon
(`macosx_arm64`) and for Linux, and the install does not need GDAL.

```bash
python3 -m pip install -e .
```

## Run

ShopRite Heritage Plaza, Monmouth Junction, New Jersey:

```bash
osm-parking-spaces detect --way 1329737820 --out-dir examples/way-1329737820
osm-parking-spaces detect --way 1329737821 --out-dir examples/way-1329737821
```

Or a local polygon in WGS84:

```bash
osm-parking-spaces detect --geojson lot.geojson --out-dir out/
```

`--gsd-ft` defaults to `1`, the native pixel size of this imagery (1 US survey
foot in EPSG:6527, NAD83(2011) / New Jersey ftUS). Outputs are reprojected to
WGS84. Adjacent stalls in a row share divider corners in both the GeoJSON
and the `.osc` file.

## Ground truth

`tests/ground_truth/way-1329737820.geojson` and
`tests/ground_truth/way-1329737821.geojson` were labelled by hand from the
NJOGIS Natural2020 mosaic at native resolution. Each polygon was traced along
the painted stall dividers and the row ends, including bays with a vehicle in
them when the row's pitch continues. They are not a copy of the detector
output. Please audit them. Where the paint is faint, a corner can be off by
a foot or two.

## Evaluate

```bash
osm-parking-spaces evaluate \
  --detected examples/way-1329737820/stalls.geojson \
  --truth tests/ground_truth/way-1329737820.geojson
```

Matching is one-to-one at IoU ≥ 0.5. The command prints precision, recall, F1,
and the mean centroid offset of the matches in feet.

`examples/way-<id>/eval.png` shows the same comparison on the imagery: green
is a match, red is a detection with no match, orange is a ground-truth stall
the detector missed.

Each stall way is tagged:

- `amenity=parking_space`
- `source=NJOGIS Natural Color Imagery 2020`

GeoJSON features also include `width_ft` and `length_ft` for review. Those
measurements are not written as OSM tags.

## Imagery and OSM data

Imagery is the New Jersey Office of GIS (NJOGIS) 2020 natural-color
orthoimagery, layer `Natural2020`, from
<https://img.nj.gov/imagerywms/Natural2020>. NJOGIS publishes this imagery in
the public domain, and it is approved for tracing into OpenStreetMap. The
`source` tag on each stall names it.

Parking-area polygons fetched from the OSM API are © OpenStreetMap
contributors, and are available under the
[Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
See <https://www.openstreetmap.org/copyright>. The tool only sends a read-only
GET for the way you name. Stall geometry comes from the imagery, clipped to
that polygon.

## Learned detector

A second detector, tracked in
[issue #3](https://github.com/sohampatwardhan/osm-parking-spaces/issues/3),
trains a small U-Net from scratch on other New Jersey lots. It predicts a
stall-interior mask. Connected components are snapped into rows with shared
corners and written with the same GeoJSON, create-only `.osc`, and preview as
the classical detector. There are no ImageNet weights. Licences are in
[docs/licences.md](docs/licences.md).

The ShopRite ways `1329737820` and `1329737821`, and any lot that touches a
150 m buffer around them, are left out of the dataset.

Install the extra, then on an Apple Silicon Mac (`mps` is used when it
exists; otherwise training stays on CPU):

```bash
python3 -m pip install -e ".[ml]"
```

Build a dataset. The default bbox is New Jersey. `--max-lots` caps the
download. A few dozen lots are a few megabytes to a few tens of megabytes
and take several minutes. A statewide build is on the order of a few
hundred megabytes and under an hour if Overpass and the WMS stay up.

```bash
osm-parking-spaces dataset \
  --out data/nj \
  --max-lots 40 \
  --bbox 38.92,-75.58,41.36,-73.89
```

Train. Checkpoints and `metrics.jsonl` go to the output directory. Run the
same command again to resume. Thirty epochs of this small network on a few
hundred lots is on the order of 15–40 minutes on an M5 Pro.

```bash
osm-parking-spaces train --data data/nj --out runs/stall-unet --epochs 30 --device auto
```

Evaluate the held-out lots, then detect with the learned weights. Fetching
imagery and scoring both lots is about one to two minutes. One `detect` call
is a few seconds after the tile arrives.

```bash
osm-parking-spaces detect --detector ml --weights runs/stall-unet/checkpoint.pt \
  --way 1329737820 --out-dir out/ml-1329737820
osm-parking-spaces evaluate \
  --detected out/ml-1329737820/stalls.geojson \
  --truth tests/ground_truth/way-1329737820.geojson
```

Repeat for way `1329737821`. Optional Core ML export:

```bash
python3 -m pip install -e ".[ml,coreml]"
osm-parking-spaces export-coreml --weights runs/stall-unet/checkpoint.pt --out runs/stall-unet/model.mlpackage
```

## Tests

```bash
python3 -m pip install -e ".[dev]"
ruff check .
pytest
```

Unit tests use committed fixtures and do not touch the network. Live checks
against the OSM API and the WMS are marked `live` and stay off unless you run
`pytest -m live`.

## License

MIT. See [LICENSE](LICENSE).
