# Requirements: hybrid stall detector (v4)

Source: [issue #4](https://github.com/sohampatwardhan/osm-parking-spaces/issues/4), a sub-issue of #3.

The v3 learned detector (a from-scratch U-Net, `--detector ml`) scored F1 0.37 and 0.18 on the two held-out ShopRite lots, against 0.64 and 0.63 for the v2 classical detector. It found only the straight outer rows and missed nearly every angled bay. v4 fuses the two. The classical detector proposes rows and stalls, and a better segmentation model checks, extends and trims them.

## Assumptions

- Imagery is only NJOGIS Natural2020 (public domain) at native 1 US survey foot per pixel. Google and Apple imagery are not used.
- Training labels are OSM `amenity=parking_space` polygons (ODbL), read-only.
- Ways 1329737820 and 1329737821, and every lot that intersects a 150 m buffer around them, stay out of training and validation. They are the test set.
- The owner's machine is an Apple Silicon Mac (PyTorch MPS). CI runs on CPU and does no heavy compute.
- Nothing is ever uploaded to OSM.

## Acceptance criteria

1. The classical v2 detector SHALL keep its role. It proposes the rows and stalls. Its behaviour under `--detector classical` SHALL NOT change.
2. WHEN the user detects with `--detector hybrid` and a weights path, the tool SHALL compute a stall probability map and SHALL fuse it with the classical stalls by the rules in 3 to 6.
3. The fusion SHALL reject a classical stall whose mean model probability is below a reject threshold, and SHALL keep one at or above a keep threshold. A stall between the two SHALL be kept only when its row's median score reaches the keep threshold.
4. The fusion SHALL add stalls beyond the ends of a row, and into interior gaps, at that row's own pitch while the model probability supports them, up to a configured limit. A new stall SHALL share its divider corners with its neighbour and SHALL NOT overlap a kept stall.
5. The fusion SHALL pull a row's near and far edges back when the model probability along the stall depth shows the stall runs into an aisle. The same shift SHALL apply to every stall in the row so shared corners stay shared.
6. The fusion MAY add stalls from model components that no classical stall covers. WHERE it does, those stalls SHALL be rows with shared corners, with angled bays kept as parallelograms.
7. The output SHALL go through the same shared-node GeoJSON, create-only `.osc` and preview as the other detectors.
8. The segmentation model SHALL be trained with arbitrary-angle rotation and flip augmentation, oversampling of lots with angled bays, chips of at least 256 px, and a pretrained encoder.
9. The encoder weights SHALL have a licence compatible with ODbL output (permissive, with attribution). Their source, licence and training data SHALL be recorded in `docs/licences.md`. Weights trained under non-commercial or research-only terms SHALL NOT be used.
10. Training SHALL keep the best validation checkpoint. Thresholds SHALL be chosen on the validation lots only. The two test lots SHALL NOT be used for any choice.
11. The evaluation harness SHALL keep one-to-one matching at IoU >= 0.5 and SHALL report precision, recall, F1 and the mean centroid offset in feet. The offset SHALL be correct whether the polygons are in pixels or in WGS84.
12. The dataset builder SHALL retry Overpass and WMS requests with back-off, SHALL split the Overpass query so a statewide build completes, SHALL drop tiles that fall outside the imagery's coverage, and SHALL keep the holdout exclusion.
13. Unit tests SHALL cover the fusion rules, the angled-bay oversampling, the augmentation and the offset fix with small synthetic fixtures, and SHALL NOT call the network or train a large model.
14. Datasets, checkpoints and Core ML packages SHALL NOT be committed. The README and `docs/FINDINGS.md` SHALL say how to regenerate them.
15. CI SHALL stay green on `ubuntu-latest` and `macos-latest`.
