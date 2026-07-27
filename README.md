# AAR_disequilibrium
This repository contains the code used to reproduce the simulations, data analysis, and figures of
- Yang, W., Mackintosh, A.N., Chu, W., & Li, Y., (2025): Committed glacier loss underscores need for enhanced monitoring of large marine-terminating glaciers.

We quantify global glacier climate-imbalance index and committed glacier mass loss under current climatic conditions (2014-2023) and under 80 different global warming levels using a simulation-based parameterization approach, leveraging global ice thickness, geodetic mass balance, debris cover, and frontal ablation data. This approach estimates the area perturbation required to bring the present-day glacier into equilibrium based on a fractional glacier-imbalance ratio of the current glacier accumulation area ratio to its equilibrium value. The steady-state volume can then be estimated using area-volume scaling. All simulations are based on a coupled model of [Python Glacier Evolution Model (PyGEM v1.1.0)](https://github.com/PyGEM-Community/PyGEM/releases/tag/v1.1.0), developed by David Rounce and collaborators, and [Open Global Glacier Model (OGGM v1.6.3)](https://github.com/OGGM/oggm/releases/tag/v1.6.3), developed by the OGGM community.

## Overview of the code

The workflow is organized into three main directories:

### `code_pygem_none`

This directory contains the PyGEM simulations without dynamic glacier-geometry evolution (`option_dynamics=None`) and the calculation of multiyear mean accumulation-area ratios (AARs).

- `Run_simulation.py` and `Run_simulation_ERA5.py` run PyGEM for the GCM–SSP experiments and the ERA5-based current-climate experiment, respectively.
- `CalDisequilibriumPyGEM.py` calculates annual AARs from the simulated equilibrium-line altitudes and the initial glacier geometry, and then derives the mean AAR for each selected climate period.
- `Run_disequilibrium.py` and `Run_disequilibrium_ERA5.py` apply these calculations to all glaciers in batches and compile the glacier-level results.
- `submit_*.sh` files submit the corresponding simulations and calculations on an HPC system.

### `code_pygem_oggm`

This directory contains the PyGEM–OGGM simulations with dynamic glacier-geometry evolution (`option_dynamics=OGGM`) used to estimate the steady-state AAR (AAR₀) associated with each glacier.

- `Run_simulation.py` and `Run_simulation_ERA5.py` run the dynamic PyGEM–OGGM simulations for the GCM–SSP experiments and ERA5, exporting both glacier-wide and elevation-bin outputs.
- `CalDisequilibriumPyGEM.py` calculates annual AARs from the evolving glacier geometry and estimates AAR₀ from the linear relationship between simulated AAR and glacier-wide total mass balance.
- `Run_AAR_steady.py` and `Run_AAR_steady_ERA5.py` apply the AAR₀ calculation to all glaciers in batches and compile the glacier-level results.
- `submit_*.sh` files submit the corresponding simulations and calculations on an HPC system.

### `code_analysis`

This directory compiles the PyGEM outputs, calculates glacier-climate imbalance and committed glacier changes, performs statistical analyses and observational comparisons, and generates the figures.

- `Compile_results.ipynb` combines the batch and regional PyGEM outputs into analysis-ready datasets, while `Count_failed.ipynb` summarizes missing or failed glacier calculations.
- `Calculate_regional_stats_median_a.py` and `Calculate_regional_mass_median_a.py` calculate regional and glacier-level statistics using the regional median-α treatment for marine-terminating glaciers lacking frontal-ablation observations.
- `Calculate_regional_stats_median_k.py` and `Calculate_regional_mass_median_k.py` perform the corresponding calculations using the regional median frontal-ablation scaling parameter *k*.
- `Calculate_global_*.ipynb`, `Calculate_MT_*.py`, and `Calculate_size_terminus_type_*.py` aggregate results globally and by marine-terminating glacier group, glacier-size class, and terminus type.
- `Calculate_griddata.ipynb` aggregates glacier-level results to a 2° × 2° grid.
- `Lowess_fit.py` implements the GlacierMIP3-style LOWESS quantile fitting. `Lowess_fit_mass*.py` and `Lowess_fit_stats*.py` apply the fits to global, regional, glacier-size, and terminus-type results.
- `Calculate_GlacierMIP3_this_study.ipynb` compares the results with GlacierMIP3 and evaluates the effect of adding this study to the GlacierMIP3 ensemble.
- `wgms_disequilibrium.py`, `wgms_test.py`, and `Compare_wgms_hugonnet_mb.py` derive and evaluate observation-based glacier-climate imbalance estimates using WGMS and geodetic mass-balance data.
- `Loibl_snowline_ELA.py`, `Run_Loibl_AAR.py`, `Loibl_snowline_disequilibrium.py`, and `compile_Loibl_results.ipynb` derive glacier-climate imbalance estimates from transient snowline-altitude observations in High Mountain Asia.
- `Figure_*.py` creates the main, supplementary, and graphical-abstract figures.
- `submit_*.sh` files submit the analysis and LOWESS calculations on an HPC system.

Most scripts currently contain user- and system-specific paths, RGI-region selections, climate-model lists, and HPC settings. These values should be updated before running the workflow in a different environment.

## Overview of the data

The input, reference, and processed data are organized into four main directories:

### `data`

This directory contains the external observational datasets and auxiliary inputs used for model evaluation and observation-based calculations.

- `temp_ch_ipcc_ar6_isimip3b.csv` provides the global temperature changes associated with the GCM–SSP experiments, obtained from GlacierMIP3.
- `00_rgi60_regions/`, `rgi62_stats.h5`, and the glacier-ID lookup tables provide glacier-inventory attributes, regional boundaries, and links between glacier identifiers, obtained from RGI 6.2.
- `DOI-WGMS-FoG-2026-02-10/` contains the WGMS Fluctuations of Glaciers database, while the `WGMS_*.csv` files contain the extracted, processed, and model-comparison results used in the WGMS analysis.
- `TSLA-HMA-2021-12-filtered.nc`, `ELA_1985_2021_Loibl.nc`, and `AAR_1985_2021_Loibl.csv` contain the transient snowline, equilibrium-line altitude, and derived AAR data for High Mountain Asia.
- `MB_1985_2021_Dussaillant.csv` and `Loibl_Dussaillant_disequilibrium.csv` contain the mass-balance data and the resulting observation-based glacier-climate imbalance estimates.
- `wgms-amce-2025-02b/` and `WGMS_Hugonnet_2000_2020_comparison.csv` provide annual glacier mass-change data and the comparison between glaciological and geodetic mass-balance estimates.

### `frontalablation_data`

This directory contains the frontal-ablation observations used in the PyGEM calibration framework and in the assessment of observational coverage for marine-terminating glaciers, Obtained from [PyGEM-OGGM community](https://pygem.readthedocs.io/en/latest/).

### `GlacierMIP3`

This directory contains the published and processed [GlacierMIP3](https://www.science.org/doi/10.1126/science.adu4675) data used to compare the simulation-based parameterization results with equilibrium experiments.

- `fig1b_scatter_by_temp_model.csv` and the corresponding median files contain GlacierMIP3 steady-state glacier-mass estimates for individual models and global warming levels.
- `lowess_fit_rel_2020_101yr_avg_steady_state_Feb12_2024*.csv` contains the GlacierMIP3 LOWESS fits for the multi-model ensemble and individual glacier models.
- `table_S3.csv` provides the GlacierMIP3 regional reference glacier mass in 2020.
- `GlacierMIP3_add_this_study*.csv` contains the recalculated ensemble results after adding the estimates from this study.
- `temp_ch_ipcc_ar6_isimip3b.csv` provides the temperature changes used to align the experiments with global warming levels.

### `pygem_oggm`

This directory contains the compiled and post-processed results generated in this study.

- `PyGEM_glacier_stats_*.nc` contains glacier-level AAR, steady-state AAR, α, glacier geometry, and related attributes compiled by RGI region.
- `PyGEM_glacier_mass_*.csv` contains steady-state area, volume, mass, and committed mass-change estimates aggregated globally, regionally, by glacier-size class, and by terminus type.
- Files containing `MT` report results for marine-terminating glaciers, including separate results for glaciers inside and outside the Antarctic and Subantarctic and for glaciers with or without frontal-ablation observations.
- Files ending in `median_a` use the regional median-α approach for marine-terminating glaciers lacking observations; files ending in `median_k` instead use the regional median frontal-ablation scaling parameter *k*.
- Files ending in `lowess_fit.csv` contain the fitted median and the 17th–83rd percentile range across the global climate experiments.
- `PyGEM_faileds.csv` records glaciers for which the required simulations or calculations were unsuccessful or unavailable.

The contents of `pygem_oggm` are generated products rather than independent input datasets. Their filenames retain the aggregation group, treatment of missing frontal-ablation observations, and fitted variable to make the source of each result explicit.



## Contact

If you have any questions, please contact:

**Dr. Weilin Yang**  
School of Earth, Atmosphere and Environment, Monash University  <br>
✉️ weilinyang.yang@monash.edu
