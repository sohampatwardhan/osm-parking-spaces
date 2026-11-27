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

The stall network is a U-Net defined in this repository and trained from
scratch. No ImageNet checkpoint, and no other research-only pretrained file,
is loaded. PyTorch itself is BSD-3-Clause.

A saved checkpoint is derived from ODbL geometries (the stall masks) and from
public-domain imagery. If you distribute a checkpoint, share it under the
ODbL as a produced work of the OSM database. Stall polygons written for
OpenStreetMap are ODbL contributions, reviewed by a person before upload.
This tool never uploads them.

## Core ML

`coremltools`, used only by `export-coreml`, is BSD-3-Clause. It is an
optional extra and is not required to train or detect.
