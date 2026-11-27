# Requirements: parking-space detection

Source: [issue #1](https://github.com/sohampatwardhan/osm-parking-spaces/issues/1).

The tool finds painted parking stalls inside one parking-area polygon and writes OpenStreetMap `amenity=parking_space` candidates. It does not upload anything.

## Assumptions

- A stall is a painted bay about 7.5–12.5 ft wide and 12–24 ft deep (typical US commercial layout), either perpendicular or angled to the aisle. Curved end islands and parallel (curb-side) bays are out of scope.
- NJOGIS Natural2020 is served at a native ground sample distance of 1 US survey foot in NAD83(2011) / New Jersey (ftUS), EPSG:6527.
- The input polygon is the paved lot (`amenity=parking`), including drive aisles. Stall polygons are clipped to it.
- Ground truth for the two ShopRite lots was drawn by hand from that imagery. It is an audit target, not detector output.

## Acceptance criteria

1. WHEN the user passes an OSM way id, the tool SHALL download that way read-only from the OSM API 0.6 and use its closed node ring as the parking-area polygon.
2. WHEN the user passes a GeoJSON file containing one Polygon (bare, a Feature, or a single-feature FeatureCollection), the tool SHALL use that polygon and SHALL NOT call the OSM API.
3. IF the way is not closed or the GeoJSON does not contain exactly one polygon, THEN the tool SHALL exit with an error and write no output files.
4. The tool SHALL request the NJOGIS WMS layer `Natural2020` at `https://img.nj.gov/imagerywms/Natural2020` in EPSG:6527 at 1 US survey foot per pixel, and SHALL reproject between that CRS and WGS84 (EPSG:4326).
5. WHEN the lot is larger than one WMS tile, the tool SHALL mosaic tiles and detect stalls on the mosaic.
6. WHEN a lot has rows at different orientations, including chevron or angled bays, the tool SHALL estimate orientation per row and SHALL emit a stall polygon whose divider edges follow that row. The polygon SHALL be a parallelogram when the row's end lines are not perpendicular to the dividers.
7. WHEN visible dividers in a row establish a pitch, the tool SHALL fill gaps of one or more missing stalls where that pitch continues between the row's painted ends, including stalls hidden by vehicles.
8. The tool SHALL snap every stall in a row to that row's measured pitch and measured depth. Depth SHALL stop at the row's back line or aisle edge, so a stall does not run into the drive aisle or shift onto the neighboring bay.
9. The tool SHALL reject rows explained by hatched islands, arrows, or other non-stall markings. A kept row SHALL have a consistent divider orientation, a pitch inside the stall-width range, and at least three divider lines.
10. The tool SHALL write a WGS84 GeoJSON FeatureCollection of the detected stall polygons. Adjacent stalls in a row SHALL reuse the same coordinates for each shared divider corner.
11. The tool SHALL write a create-only OsmChange (`.osc`) document. Every created way SHALL be tagged `amenity=parking_space` and `source=NJOGIS Natural Color Imagery 2020`. Element ids SHALL be negative. Adjacent stalls SHALL reference one shared node per shared corner. The document SHALL NOT contain `modify` or `delete` blocks.
12. The tool SHALL NOT upload to OSM, open a changeset, or require OSM credentials.
13. The tool SHALL write a preview PNG with the imagery, the lot outline, and the stall polygons.
14. WHEN no stalls are detected, the tool SHALL still write an empty FeatureCollection, a create-only `.osc` with no ways, and a preview of the lot outline.
15. The repository SHALL contain hand-labelled ground-truth GeoJSON for OSM ways 1329737820 and 1329737821, traced from Natural2020 at native resolution.
16. `osm-parking-spaces evaluate` SHALL match detections to that ground truth one-to-one at IoU ≥ 0.5 and SHALL report precision, recall, F1, and the mean centroid offset of the matches.
17. The published package SHALL install on Linux and on arm64 macOS with pip, using wheels only (no GDAL or other system GIS libraries).
18. Unit tests SHALL use committed fixtures and SHALL NOT call the network. Live WMS or OSM checks SHALL be opt-in and off in CI. The default test run SHALL compute precision, recall, and F1 from the committed ground truth and the committed example detections.
