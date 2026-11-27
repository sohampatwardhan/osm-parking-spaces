# Tech

- Python 3.11+.
- Runtime libraries must publish wheels for arm64 macOS and for manylinux: `opencv-python-headless`, `numpy`, `shapely`, `pyproj`, `requests`.
- Do not add GDAL, rasterio, or other packages that need a system GIS install.
- Imagery: NJOGIS WMS `https://img.nj.gov/imagerywms/Natural2020`, layer `Natural2020`, WMS 1.1.1, SRS EPSG:6527, 1 US survey foot per pixel.
- Parking polygons: OSM API 0.6 `GET /api/0.6/way/{id}/full` (read-only, with an identifying User-Agent) or a local GeoJSON file.
- Learned detector ([issue #3](https://github.com/sohampatwardhan/osm-parking-spaces/issues/3)): optional `[ml]` extra (PyTorch). A from-scratch U-Net predicts a stall mask. Device is `mps` when available, else CPU. Training labels are OSM `amenity=parking_space` (ODbL) on Natural2020 only. Ways 1329737820 and 1329737821 and lots that intersect a buffer around them are held out. No ImageNet weights. Core ML export is a separate `[coreml]` extra.
- Outputs: WGS84 GeoJSON, create-only OsmChange 0.6, PNG preview. The learned detector uses the same outputs.
- Lint with ruff. Test with pytest. Tests do not touch the network unless marked `live`.
- CI runs on `ubuntu-latest` and `macos-latest`.
