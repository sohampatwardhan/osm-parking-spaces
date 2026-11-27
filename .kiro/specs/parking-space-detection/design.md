# Design: parking-space detection

## Pipeline

```
way id or GeoJSON
        │
        ▼
parking polygon (WGS84)
        │  pyproj → EPSG:6527
        ▼
WMS GetMap tiles (1 ft / pixel) → one mosaic + pixel georeferencing
        │
        ▼
per-row lattices (pitch + depth snapped, gaps filled, angled quads)
        │  clip to the lot, suppress hatches, drop overlaps
        ▼
shared-corner topology
        │
        ▼
GeoJSON (WGS84)   OsmChange (create only, shared nodes)   preview PNG
```

Nothing in this pipeline writes to OSM. The `.osc` file is for a person to review and upload with their own editor.

## Imagery and CRS

GetCapabilities for Natural2020 lists EPSG:6527 with `resx=1` (US survey feet). That is the native grid, so the default ground sample distance is **1 ft/pixel** in EPSG:6527. Requesting a finer size only interpolates. Web Mercator is available but would resample the same grid, so it is not used.

WMS 1.1.1 `GetMap` uses `SRS=EPSG:6527` and an `x,y` bbox. Pixel `(0,0)` is the north-west corner of the requested bbox. Tiles are at most 1500 pixels on a side and are mosaicked before detection.

OSM nodes and GeoJSON coordinates stay in EPSG:4326. Transforms go through pyproj (`always_xy=True`). Output rings are closed and oriented counter-clockwise. Shared corners are rounded to 1e-7 degrees once, then reused, so GeoJSON coordinates and `.osc` node positions are the same point.

## Detection

A learned model is a poor fit: there is no training set shipped with the tool, the package must stay a small wheel-only install, and a mapper needs to see why a box was drawn. Global "one dominant angle, then rectangles" missed the chevron bays and let boxes drift into aisles. Rows are detected on their own.

1. **Paint mask.** Grayscale minus a Gaussian (σ ≈ 2.5 px). Keep pixels that are both brighter than that local background and brighter than a fixed asphalt floor. Mask to the lot polygon, eroded by about 2 ft so the curb is not a stall line.
2. **Candidate rows at every orientation.** Every 3° from 0° to 180°, open the mask with a thin line kernel (~10–12 ft) and rotate so those lines are vertical. A band of vertical energy whose height is about one stall is a candidate row. Several angles are kept. Nearby angles that describe the same paint are resolved later by overlap, not by keeping only the global winner.
3. **Per-row pitch.** Inside a band, the column autocorrelation supplies the pitch. A peak between 7.5 and 12.5 ft is a stall pitch. A stronger peak below about 6.5 ft is hatch or a crosswalk, and the band is rejected. Divider peaks vote for the phase of a regular lattice at that pitch. The lattice runs from the first supported divider to the last, which is the painted end of the row.
4. **Occluded stalls.** A missing divider is inserted when the gap is an integer multiple of the pitch (typically 2× or 3×) and the gap still lies inside the row's painted extent. The stall is emitted even when a vehicle covers the paint. A wide empty stretch with no row energy is a break, not a hidden stall, and is not filled.
5. **Depth and angled geometry.** The top and bottom ends of the visible dividers are each fit with a straight line. Those lines are the aisle edge and the back of the row. Every lattice divider is cut by those two lines, and each stall is the quad between neighboring dividers. The quad is a rectangle when the end lines are perpendicular to the dividers, and a parallelogram when they are not (chevron / angled bays). Depth is that cut, not the raw band height, so the box stops at the aisle edge. Width is the pitch, not the noisy distance between two broken segments.
6. **False-bay suppression.** A row is kept only when it has at least three divider lines, the divider angles agree, the pitch is in the stall range, and the paint is not a dense hatch (short marks at a few feet apart) or an isolated arrow. Hatched islands and turn arrows fail those tests.
7. **Cleanup.** Clip each quad to the lot. Drop a stall that loses most of its area to the clip. Suppress duplicates across angles when IoU is high.

Width and depth limits are US commercial-stall assumptions, not measurements to write back as OSM tags. GeoJSON may carry `width_ft` and `length_ft` as review aids. The `.osc` file does not.

## Shared corners

Stalls in one row are ordered along the pitch. The two corners on the shared divider are one pair of coordinates, used by both polygons. Export builds a node table keyed by rounded longitude and latitude:

- GeoJSON rings repeat those rounded positions, so a shared corner is byte-identical in both features.
- The `.osc` file creates the node once, with a negative id, and both ways reference it.
- Ids stay negative. There is still a single `<create>` block and no upload code.

## Outputs

| File | Contents |
| --- | --- |
| `stalls.geojson` | WGS84 polygons. Properties: `amenity`, `source`, `width_ft`, `length_ft`. Shared corners use identical coordinates. |
| `stalls.osc` | `<osmChange>` with a single `<create>`. Negative ids. Shared divider nodes. Tags `amenity=parking_space` and `source=NJOGIS Natural Color Imagery 2020` only. |
| `preview.png` | Imagery, yellow lot outline, green stall polygons. |

Evaluation adds `eval.png` beside a detection run: true positives, false positives, and missed ground-truth stalls in different colors. That file is produced by `evaluate`, not by `detect`.

## Evaluation

Ground truth lives in `tests/ground_truth/way-<id>.geojson`. It was traced by hand from the native-resolution Natural2020 mosaic of each ShopRite lot, not copied from the detector.

```
osm-parking-spaces evaluate \
  --detected examples/way-1329737820/stalls.geojson \
  --truth tests/ground_truth/way-1329737820.geojson \
  --image examples/way-1329737820/preview.png
```

Matching is one-to-one, greedy by intersection-over-union, with a minimum IoU of 0.5. Unmatched detections are false positives. Unmatched ground-truth stalls are false negatives. Precision, recall, and F1 use those counts. Mean centroid offset is the mean distance, in feet, between matched pairs only.

The default pytest run loads the committed example GeoJSON and the committed ground truth. It does not call the network. Live WMS and OSM checks stay under the `live` marker.

## CLI

```
osm-parking-spaces detect --way WAY_ID --out-dir DIR
osm-parking-spaces detect --geojson FILE --out-dir DIR
osm-parking-spaces evaluate --detected FILE --truth FILE [--preview-out FILE]
```

Optional `--gsd-ft` overrides the 1 ft pixel size. The WMS URL and layer have defaults and can be overridden for tests.

## Packaging and CI

Runtime dependencies are limited to wheels that publish `macosx_arm64` and manylinux builds: `opencv-python-headless`, `numpy`, `shapely`, `pyproj`, `requests`. Images are read and written with OpenCV, so GDAL/rasterio is not required.

CI runs ruff and pytest on `ubuntu-latest` and `macos-latest` (arm64). Default pytest excludes tests marked `live`.

## Decisions

| Decision | Why |
| --- | --- |
| Per-row orientation, not one lot-wide angle | Chevron bays on the ShopRite lots are not the dominant axis. Keeping every qualified row, then dropping overlaps, lets those bays through. |
| Parallelograms from divider lines cut by the row end lines | Angled stalls are not rectangles in the aisle frame. Snapping both ends to the fitted end lines also stops boxes at the aisle. |
| Lattice at the measured pitch, including empty steps | A car hides a divider but rarely hides the whole row. The pitch is known from the visible dividers. |
| Hatch test on the short-period autocorrelation | Islands and crosswalks repeat every few feet. Stall paint repeats every 8–12 ft. |
| Shared nodes | Adjacent bays physically share a painted line. One node per corner is the usual OSM topology and keeps the create-only file valid. |
| Hand-labelled ground truth, IoU 0.5 | The owner can audit the labels. IoU 0.5 is a one-to-one match that still fails a box shifted by a full stall. |
| `source` tag only, no `width`/`orientation` tags | The foot grid is coarse. Those measurements stay in GeoJSON for review. |
| EPSG:6527 at 1 ft | Native resolution advertised by the WMS. Avoids a resample. |
| Create-only `.osc`, negative ids, no upload | Safe to open in an editor. No changeset and no credentials. |

## Limitations

- Curved end islands and parallel curb stalls are not modeled.
- A row with too few visible dividers to measure a pitch is still missed.
- The end-line fit can sit a foot or two off faded paint.
- The width model is US commercial parking.
- Results are candidates for a human to check before anyone uploads them.
