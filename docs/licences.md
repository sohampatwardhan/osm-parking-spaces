# Licences

The classical tool is MIT. See the repository `LICENSE`.

## Imagery

NJOGIS Natural2020 orthoimagery, layer `Natural2020`, from
<https://img.nj.gov/imagerywms/Natural2020>. The New Jersey Office of GIS
publishes this imagery in the public domain. No other imagery, including
Google or Apple, is fetched or used.

## Training labels

`amenity=parking` and `amenity=parking_space` geometries come from
OpenStreetMap, © OpenStreetMap contributors, under the
[Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
The dataset builder only sends read-only Overpass and OSM API requests.

Ways 1329737820 and 1329737821, and any lot that intersects a 150 m buffer
around them, are not written into the training or validation tiles.

## Model weights

Two networks exist. Both are defined in this repository.

- `unet` (v3, `--detector ml`): trained from scratch. No pretrained tensors.
- `swin-unet` (v4, `--detector hybrid`): a U-Net decoder on a Swin-v2-Base encoder that starts from a pretrained checkpoint. The decoder and the one- and two-pixel-stride stem start from random values.

### Pretrained encoder

| | |
| --- | --- |
| Checkpoint | SatlasPretrain `Aerial_SwinB_SI` (`aerial_swinb_si.pth`) |
| Source | <https://huggingface.co/allenai/satlas-pretrain> (Allen Institute for AI). Code and loader: <https://github.com/allenai/satlaspretrain_models> |
| Weights licence | ODC-BY 1.0 (Open Data Commons Attribution). It is permissive. It asks for attribution and has no non-commercial or share-alike clause, so it is compatible with ODbL output. |
| Pretraining data | SatlasPretrain. The aerial model uses NAIP (USGS, public domain) with labels from OpenStreetMap (ODbL), Microsoft Buildings (ODbL), WorldCover (CC-BY 4.0) and new Allen AI annotations (ODC-BY). No Google or Apple imagery. |
| SHA-256 | `9c524bb5719401bd4d1b165fa198f6baac3006c53071185f2c2638424a933dba` |
| Use here | Only the `backbone.backbone.features.*` tensors (the Swin-v2-Base encoder) are loaded. The detection heads and the FPN in the file are ignored. |

Attribution: SatlasPretrain, Allen Institute for AI (Bastani et al., ICCV 2023), ODC-BY 1.0.

The code that loads it uses `torchvision` (BSD-3-Clause), which is part of the `[ml]` extra. The `satlaspretrain-models` package (Apache-2.0) is not needed. Weights are downloaded on demand by `osm-parking-spaces fetch-encoder` into `runs/pretrained/` and are not committed.

No ImageNet checkpoint, and no weights released under non-commercial or research-only terms, are loaded. PyTorch is BSD-3-Clause.

### Trained checkpoints

A saved checkpoint is derived from ODbL geometries (the stall masks), from public-domain imagery and from the ODC-BY encoder. If you distribute a checkpoint, share it under the ODbL as a produced work of the OSM database, and keep the ODC-BY attribution above. Stall polygons written for OpenStreetMap are ODbL contributions, reviewed by a person before upload. This tool never uploads them. Checkpoints are not committed to this repository.

## Core ML

`coremltools`, used only by `export-coreml`, is BSD-3-Clause. It is an
optional extra and is not required to train or detect.
