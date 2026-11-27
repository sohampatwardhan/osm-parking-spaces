# Structure

```
.kiro/specs/parking-space-detection/   requirements, design, tasks
.kiro/steering/                        product, tech, structure
osm_parking_spaces/                    CLI, OSM/GeoJSON IO, WMS, detection, export
tests/                                 offline tests and small fixtures
tests/fixtures/                        cropped imagery, polygon, sample OSM XML
tests/ground_truth/                    hand-traced stall GeoJSON for the ShopRite lots
examples/way-<id>/                     stalls.geojson, stalls.osc, preview.png, eval.png
.github/workflows/ci.yml
```

Detection thresholds and the imagery CRS live in the library, not in the CLI parser. The CLI only parses arguments, calls the pipeline, and writes the three output files.

Do not add an upload command, a changeset call, or an OSM write token.
