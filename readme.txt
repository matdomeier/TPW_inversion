## TPW_inversion
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20127221.svg)](https://doi.org/10.5281/zenodo.20127221)

This is a collection of data, python codes and jupyter notebooks supporting the paper "Quadrupolar sea-level fluctuations reveal epsiodes of rapid polar wander" by Mathew Domeier, Leandro C. Gallo, Chloé Marcilly and Trond H. Torsvik. The contents include the following:

1) This readme file.

2) plate_model: this subdirectory contains the rotation file (PHAB25_zero-longitude-Africa.rot) and static continental polygons (static_polygons.gpml) from the plate model that is used as the basis for the flooding maps of Marcilly et al. (2022). The plate model is modified to provide a zero-longitude-Africa reference frame (plate_ID=998; this should be used by default in the following codes). The flooding maps themselves are included as PHAB25_exposed_land.gpml. Two additional .gpml files provide polygon definitions for active major foreland basins (foreland_basins.gpml) and incipient passive margins (passive_margins.gpml). The latter are defined as ~300 km wide belts along newly formed passive margins (from 10 Myr prior to continental breakup to 40 Myr following breakup).

3) obs_grids: this is a subdirectory containing .csv files reporting the flooding map data of Marcilly et al. (2022) as a binary state (0=submerged, 1=exposed) referenced to an h3 grid (resolution 3). A .csv file is provided for each 10 Myr between 320 and 0 Ma. The first column of these files are the h3 cell references. The in_margin and in_basin columns denote if a given point occurs along an incipient passive margin or within an active foreland basin according to the polygon definitions provided in the plate model files above. 

4) methods.py: is a python file containing auxiliary codes supporting the jupyter notebooks

5) permutation_tests.ipynb: performs permutation tests to establish the statistical significance of the results to be computed in the invert_data.ipynb notebook.

6) invert_data.ipynb: applies logistic regression to perform TPW inversions on the supplied flooding map data (in obs_grids).

7) reconstruct_inversions.ipynb: uses the results of the invert_data.ipynb notebook and the supplied plate model to generate paleogeographic maps of the data and inversion results.

8) validation_experiments.ipynb: runs synthetic tests to evaluate the performance of the method on synthetic test data. 

9) environment.yml: .yml file describing the packages necessary to run all the codes in this repository.
    
