# Tech

- Python 3.11+.
- Runtime libraries must publish wheels for arm64 macOS and for manylinux: `opencv-python-headless`, `numpy`, `shapely`, `pyproj`, `requests`.
- Do not add GDAL, rasterio, or other packages that need a system GIS install.
- Imagery: NJOGIS WMS `https://img.nj.gov/imagerywms/Natural2020`, layer `Natural2020`, WMS 1.1.1, SRS EPSG:6527, 1 US survey foot per pixel.
- Parking polygons: OSM API 0.6 `GET /api/0.6/way/{id}/full` (read-only, with a identifying User-Agent) or a local GeoJSON file.
- Outputs: WGS84 GeoJSON, create-only OsmChange 0.6, PNG preview.
- Lint with ruff. Test with pytest. Tests do not touch the network unless marked `live`.
- CI runs on `ubuntu-latest` and `macos-latest`.
