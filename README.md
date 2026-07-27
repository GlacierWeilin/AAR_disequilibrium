# AAR_disequilibrium
This repository contains the code used to reproduce the simulations, data analysis, and figures of
- Yang, W., Mackintosh, A.N., Chu, W., & Li, Y., (2025): Committed glacier loss underscores need for enhanced monitoring of large marine-terminating glaciers.

We quantify global glacier climate-imbalance index and committed glacier mass loss under current climatic conditions (2014-2023) and under 80 different global warming levels using a simulation-based parameterization approach, leveraging global ice thickness, geodetic mass balance, debris cover, and frontal ablation data. This approach estimates the area perturbation required to bring the present-day glacier into equilibrium based on a fractional glacier-imbalance ratio of the current glacier accumulation area ratio to its equilibrium value. The steady-state volume can then be estimated using area-volume scaling. All simulations are based on a coupled model of [Python Glacier Evolution Model (PyGEM v1.1.0)](https://github.com/PyGEM-Community/PyGEM/releases/tag/v1.1.0), developed by David Rounce and collaborators, and [Open Global Glacier Model (OGGM v1.6.3)](https://github.com/OGGM/oggm/releases/tag/v1.6.3), developed by the OGGM community.

The files include:
- [`README.md`](README.md) — Description of the repository
- ['data'](data) - The documentation of the data.
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

- `Loibl_snowline_ELA.py`, `run_Loibl_AAR.py`, `Loibl_AAR.py`, and `compile_Loibl_results.ipynb`. <br>
  Calculate glacier climate disequilbirium based on transient snowline altitude estimates.

- `Figure_*.py` and `Table_*.py`. <br>
  Create the figures and tables

## Contact

If you have any questions, please contact:

**Dr. Weilin Yang**  
School of Earth, Atmosphere and Environment, Monash University  <br>
✉️ weilinyang.yang@monash.edu
