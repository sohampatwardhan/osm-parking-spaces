# Requirements: parking-space detection

Source: [issue #1](https://github.com/sohampatwardhan/osm-parking-spaces/issues/1).

The tool finds painted parking stalls inside one parking-area polygon and writes OpenStreetMap `amenity=parking_space` candidates. It does not upload anything.

## Assumptions

- A stall is a painted bay about 7.5–12.5 ft wide and 12–24 ft deep (typical US commercial layout). Compact, parallel, and curved bays are out of scope for the first version.
- NJOGIS Natural2020 is served at a native ground sample distance of 1 US survey foot in NAD83(2011) / New Jersey (ftUS), EPSG:6527.
- The input polygon is the paved lot (`amenity=parking`), including drive aisles. Stall polygons are clipped to it.

## Acceptance criteria

1. WHEN the user passes an OSM way id, the tool SHALL download that way read-only from the OSM API 0.6 and use its closed node ring as the parking-area polygon.
2. WHEN the user passes a GeoJSON file containing one Polygon (bare, a Feature, or a single-feature FeatureCollection), the tool SHALL use that polygon and SHALL NOT call the OSM API.
3. IF the way is not closed or the GeoJSON does not contain exactly one polygon, THEN the tool SHALL exit with an error and write no output files.
4. The tool SHALL request the NJOGIS WMS layer `Natural2020` at `https://img.nj.gov/imagerywms/Natural2020` in EPSG:6527 at 1 US survey foot per pixel, and SHALL reproject between that CRS and WGS84 (EPSG:4326).
5. WHEN the lot is larger than one WMS tile, the tool SHALL mosaic tiles and detect stalls on the mosaic.
6. The tool SHALL write a WGS84 GeoJSON FeatureCollection of the detected stall polygons.
7. The tool SHALL write a create-only OsmChange (`.osc`) document. Every created way SHALL be tagged `amenity=parking_space` and `source=NJOGIS Natural Color Imagery 2020`. Element ids SHALL be negative. The document SHALL NOT contain `modify` or `delete` blocks.
8. The tool SHALL NOT upload to OSM, open a changeset, or require OSM credentials.
9. The tool SHALL write a preview PNG with the imagery, the lot outline, and the stall polygons.
10. WHEN no stalls are detected, the tool SHALL still write an empty FeatureCollection, a create-only `.osc` with no ways, and a preview of the lot outline.
11. The published package SHALL install on Linux and on arm64 macOS with pip, using wheels only (no GDAL or other system GIS libraries).
12. Unit tests SHALL use committed fixtures and SHALL NOT call the network. Live WMS or OSM checks SHALL be opt-in and off in CI.
