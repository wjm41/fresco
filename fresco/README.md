# Library

This directory contains most of the useful functions for running FRESCO, which can be accessed as part of the `fresco` module. An overview of the functions is given below.

## File descriptions

| name  | description |
| --- | --- |
| `featurise.py` | functions for generating pharmacophore distance distributions from a list of RDKit molecules|
| `model.py` | functions for fitting KDEs on pharmacophore distance distributions |
| `score_2body.py` | functions for scoring molecules with pharmacophore KDEs |
| `filter.py`| functions for filtering molecules by physicochemical properties |
| `butina_selection.py`| functions for selecting molecules from a list via clustering and sampling of cluster centroids |