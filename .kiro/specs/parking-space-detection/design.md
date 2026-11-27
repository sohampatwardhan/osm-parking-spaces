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
stall rectangles in pixel space
        │  clip to the lot, drop overlaps
        ▼
GeoJSON (WGS84)   OsmChange (create only)   preview PNG
```

Nothing in this pipeline writes to OSM. The `.osc` file is for a person to review and upload with their own editor.

## Imagery and CRS

GetCapabilities for Natural2020 lists EPSG:6527 with `resx=1` (US survey feet). That is the native grid, so the default ground sample distance is **1 ft/pixel** in EPSG:6527. Requesting a finer size only interpolates. Web Mercator is available but would resample the same grid, so it is not used.

WMS 1.1.1 `GetMap` uses `SRS=EPSG:6527` and an `x,y` bbox. Pixel `(0,0)` is the north-west corner of the requested bbox. Tiles are at most 1500 pixels on a side and are mosaicked before detection.

OSM nodes and GeoJSON coordinates stay in EPSG:4326. Transforms go through pyproj (`always_xy=True`). Output rings are closed and oriented counter-clockwise.

## Detection

A learned model is a poor fit here: there is no labeled set, the package must stay a small wheel-only install, and a mapper needs to see why a box was drawn. The issue's Hough-line sketch is the right family. On this imagery a raw Hough or LSD pass is too noisy (cars, arrows, and 1 px paint all produce segments), so the implementation uses the spacing of the paint instead of pairing raw segments.

1. **Paint mask.** Grayscale minus a Gaussian (σ ≈ 2.5 px). Keep pixels that are both brighter than that local background and brighter than a fixed asphalt floor. Mask to the lot polygon, eroded by about 2 ft so the curb is not a stall line.
2. **Orientation.** Every 3° from 0° to 180°, open the mask with a thin line kernel (~12 ft). Rotate so those lines are vertical and score the column profile by its autocorrelation at 8–12 ft (stall width) against 15–22 ft (stall depth). The depth axis is rejected so the long head-of-row lines are not treated as dividers. The best angle is the primary stall direction.
3. **Secondary direction.** Subtract the primary opening from the paint mask and score what remains. Chevron bays on the test lots sit near 90° from the main rows, so a perpendicular secondary angle is allowed. At most one secondary angle is used, and only its own residual paint is searched.
4. **Rows.** In the rotated mask, group vertical-line energy into bands. Bands taller than one stall are split at an interior valley.
5. **Bays.** Inside a band, local column peaks are chained at the measured pitch (about 9–10 ft). A single missing divider (gap ≈ 2 pitches, with some paint still in between) is filled, which recovers a stall whose line is hidden by a car. Each step of the chain becomes a rectangle. Depth is the painted extent of that bay, clamped to 12–24 ft.
6. **Cleanup.** Clip to the lot. Drop a stall that loses most of its area to the clip, is smaller than about 80 sq ft, or overlaps a kept stall by more than about half of the smaller area.

Width and depth limits are US commercial-stall assumptions, not measurements to write back as OSM tags. GeoJSON may carry `width_ft` and `length_ft` as review aids. The `.osc` file does not.

## Outputs

| File | Contents |
| --- | --- |
| `stalls.geojson` | WGS84 polygons. Properties: `amenity`, `source`, `width_ft`, `length_ft`. |
| `stalls.osc` | `<osmChange>` with a single `<create>`. Negative node and way ids. Tags `amenity=parking_space` and `source=NJOGIS Natural Color Imagery 2020` only. |
| `preview.png` | Imagery, yellow lot outline, green stall polygons. |

Adjacent stalls do not share nodes. Corner coordinates are independent, which keeps the create-only file valid when clipping trims one bay and not its neighbor.

## CLI

```
osm-parking-spaces detect --way WAY_ID --out-dir DIR
osm-parking-spaces detect --geojson FILE --out-dir DIR
```

Optional `--gsd-ft` overrides the 1 ft pixel size. The WMS URL and layer have defaults and can be overridden for tests.

## Packaging and CI

Runtime dependencies are limited to wheels that publish `macosx_arm64` and manylinux builds: `opencv-python-headless`, `numpy`, `shapely`, `pyproj`, `requests`. Images are read and written with OpenCV, so GDAL/rasterio is not required.

CI runs ruff and pytest on `ubuntu-latest` and `macos-latest` (arm64). Default pytest excludes tests marked `live`.

## Decisions

| Decision | Why |
| --- | --- |
| Morphological opening + pitch, not Hough pairing alone | 1 ft pixels make paint a 1 px ridge. Autocorrelation at the known stall width rejects car edges (~6 ft) and crosswalk stripes (~2–4 ft). |
| EPSG:6527 at 1 ft | Native resolution advertised by the WMS. Avoids a resample. |
| Create-only `.osc`, negative ids, no shared nodes | Safe to open in an editor. No changeset, no credentials, no upload code path. |
| `source` tag only, no `width`/`orientation` tags | The foot grid and occluding cars make those measurements too coarse to assert on OSM. |
| Secondary orientation on residual paint | The ShopRite lots have a main grid plus chevron bays. Searching the leftover paint picks up the second direction without re-detecting the first. |
| Offline fixtures | Unit tests must not depend on WMS or OSM availability. |

## Limitations

- Painted lines hidden by vehicles, faded rows, and curved end islands are missed or shortened.
- A stall can sit a few feet off the paint, or bleed into the aisle, when the row band is taller than the bay.
- Chevron and angled sections are only partially recovered.
- Hatched islands and arrows can still produce a false bay.
- The width model is US commercial parking. It will not fit every lot.
- Results are candidates for a human to check before anyone uploads them.
