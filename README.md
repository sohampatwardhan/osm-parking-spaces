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
