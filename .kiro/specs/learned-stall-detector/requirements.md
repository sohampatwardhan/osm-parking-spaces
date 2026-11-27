# Requirements: learned stall detector

Source: [issue #3](https://github.com/sohampatwardhan/osm-parking-spaces/issues/3), a sub-issue of #1.

A learned detector sits beside the classical one. It predicts individual stalls inside an `amenity=parking` polygon. It does not replace the GeoJSON, create-only OsmChange, preview, or evaluation path, and it never uploads to OSM.

## Assumptions

- Imagery is only NJOGIS Natural2020 (public domain), at native 1 US survey foot per pixel in EPSG:6527. Google and Apple imagery are not used.
- Training labels are existing OSM `amenity=parking_space` polygons (ODbL), read-only from Overpass or the OSM API 0.6.
- Ways 1329737820 and 1329737821, and every lot that intersects a buffer around them, are held out of training and validation.
- Weights are trained from scratch. No ImageNet or other research-only checkpoint is loaded.
- The owner's machine is an Apple Silicon Mac. CI runs on CPU.

## Acceptance criteria

1. WHEN the user builds a dataset, the tool SHALL fetch New Jersey parking-lot polygons and the `amenity=parking_space` ways inside them, plus the matching Natural2020 tile, and SHALL rasterise stall interiors to a mask.
2. The dataset builder SHALL drop a lot whose way id is 1329737820 or 1329737821, and SHALL drop any lot that intersects a buffer around those two lots. A unit test SHALL prove both cases with fixtures and no network.
3. WHEN a lot has fewer stalls than the configured minimum, or the stalls cover too little or too much of the parent `amenity=parking` polygon, the builder SHALL drop that lot.
4. The builder SHALL write a lot-level train/validation split that does not contain the held-out lots, and SHALL record lot, stall, and tile counts.
5. The model SHALL be a stall-interior network trained from scratch in PyTorch. Training and inference SHALL use the `mps` device when it is available and SHALL fall back to CPU.
6. WHERE Core ML export is requested and coremltools is installed, the tool SHALL write a Core ML package. The base install SHALL NOT require coremltools.
7. WHEN the user detects with `--detector ml` and a weights path, the predicted mask SHALL be turned into stall polygons that go through the same shared-node GeoJSON, create-only `.osc`, and preview as the classical detector.
8. WHEN the user detects with `--detector classical`, behaviour SHALL stay the classical detector.
9. The tool SHALL NOT upload to OSM, open a changeset, or require OSM credentials.
10. `docs/licences.md` and the README SHALL name the licence of the imagery, the OSM labels, the code, and the trained weights.
11. Unit tests SHALL cover rasterisation, holdout exclusion, mask post-processing, and a few CPU training steps on a tiny fixture. Default tests SHALL NOT call the network.
12. CI SHALL stay green on `ubuntu-latest` and `macos-latest`.
