# Product

`osm-parking-spaces` is a command-line tool for mappers. It looks at public-domain aerial imagery inside an existing `amenity=parking` polygon and proposes individual `amenity=parking_space` polygons.

The user reviews the GeoJSON, the preview, and a create-only OsmChange file, then uploads with their own OSM account if the stalls look right. This tool never uploads and never asks for OSM credentials.

The first site is the ShopRite Heritage Plaza lots in Monmouth Junction, New Jersey (OSM ways 1329737820 and 1329737821), using NJOGIS 2020 natural-color orthoimagery.
