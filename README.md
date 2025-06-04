# AAR_disequilibrium
This repository contains the code used to reproduce the simulations, data analysis, and figures of
- Yang, W., Mackintosh, A.N., Chu, W., & Li, Y., (2025): Global Glacier Climate Disequilibrium: Committed Mass Loss and Sea Level Rise.

We quantify global glacier disequilibrium in the present-day climate (2014-2023) using a parameterization approach, leveraging global ice thickness, geodetic mass balance, debris cover, and frontal ablation data. This approach estimates the area perturbation required to bring the present-day glacier into equilibrium based on a fractional disequilibrium ratio of the current glacier accumulation area ratio to its equilibrium value. The mass loss for the adjustment can then be estimated using area-volume scaling. All simulations are based on a hybrid of [Python Glacier Evolution Model (PyGEM v0.2.5)](https://github.com/PyGEM-Community/PyGEM/releases/tag/v0.2.0), developed by David Rounce and collaborators, and [Open Global Glacier Model (OGGM v1.6.0)](https://github.com/OGGM/oggm/releases/tag/v1.6.0), developed by the OGGM community.

The files include:
- [`README.md`](README.md) — Description of the repository
- ['data'](data) - The documentation of the data. Download the large data file (ERA5_MCMC_ba1_2014_2023_corrected.nc) from [Google Drive](https://drive.google.com/file/d/1dm7v9OQjZxV-C2maQvJgCwW4ov6IOO27/view?usp=sharing).
- ['code'](code) - The documentation of the code for running simulations, analyzing the data, and creating figures and tables.

## Overview of the code
- Run the PyGEM script `run_simulation.py` and `pygem_input.py`. <br>
  This script replaces the original `run_simulation` file in PyGEM and automatically performs glacier climate disequilibrium calculations using both the parameterization approach and the equilibrium experiment.

- `process_disequilibrium.py`. <br>
  Compiles the output of the PyGEM runs of several gdirs into one file.
  
- `process_disequilibrium_errors.py`. <br>
  Uses the nearest neighbour interpolation to estimate results for the failed glaciers.
  
- `process_disequilibrium_by_region.py`, `process_disequilibrium_by_area.py`, `process_disequilibrium_lat_lon_mean.py`, and `process_disequilibrium_griddata.py`. <br>
  Analyze the results based on RGI regions, glacier area, and 2°×2° grid resolution.

- `wgms_disequilibrium.py`. <br>
  Calculate glacier climate disequilbirium based on the WGMS observations.

- `Figure_*.py` and `Table_*.py`. <br>
  Create the figures and tables

## Contact

If you have any questions, please contact:

**Dr. Weilin Yang**  
School of Earth, Atmosphere and Environment, Monash University  <br>
✉️ weilinyang.yang@monash.edu
