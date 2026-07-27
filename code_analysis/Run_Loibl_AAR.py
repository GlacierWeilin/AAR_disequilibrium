#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 19:52:40 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import shutil
import os
import numpy as np
import pandas as pd
import xarray as xr
import logging

from oggm import cfg, workflow, entity_task, global_task
from oggm.workflow import execute_entity_task

log = logging.getLogger(__name__)
@entity_task(log)
def compute_AAR_from_ELA(gdir, ela=None, year_range=np.arange(1985, 2021)):
    """Compute annual AAR time series based on ELA.
    
    Parameters
    ----------
    gdir : oggm.GlacierDirectory
        The glacier directory to process
    ela : dict or pd.Series
        Dictionary or Series with {rgi_id: {year: ela_value}}
    year_range : iterable
        Years to compute AAR for (default 1985-2021)
    
    Returns
    -------
    pd.Series
        AAR time series indexed by years
    """
    
    rgi_id = gdir.rgi_id
    
    AARs = pd.Series(index=year_range, dtype=float)
    
    path = gdir.get_filepath('model_flowlines')
    if os.path.exists(path):
        fls = gdir.read_pickle('model_flowlines')
        fl = fls[0]

        mask = fl.thick > 0
        widths = np.sum(fl.widths_m[mask])

        for year in year_range:
        
            _ela = (ela.sel(RGIId=rgi_id, Year=year)).item()
        
            if np.isnan(_ela):
                AARs.loc[year] = np.nan
                continue

            mask = fl.surface_h > _ela
            acc_area = np.sum(fl.widths_m[mask])
            AAR = acc_area / widths

            if AAR <= 0 or AAR >= 1:
                AAR = np.nan

            AARs.loc[year] = AAR

    return AARs
    
@global_task(log)
def compile_AAR_from_ELA(gdirs, filesuffix='', path=True, csv=True, ela=None, year_range=np.arange(1985, 2021)):
    """Compiles a table of AAR.

    The file is stored in a hdf file (not csv) per default. Use pd.read_hdf
    to open it.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location (file
        extension matters)
    csv: bool
        Set to store the data in csv instead of hdf.
    ela: DataFrame
        annual ela of each glacier
    year_range : iterable
        Years to compute AAR for (default 1985-2021)
    """

    out_df = execute_entity_task(compute_AAR_from_ELA, gdirs, ela=ela, year_range=np.arange(1985, 2021))

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.nan)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    #out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            fpath = os.path.join(cfg.PATHS['working_dir'],
                                 'AAR_1985_2021_Loibl' + filesuffix)
            if csv:
                out.to_csv(fpath + '.csv')
            else:
                out.to_hdf(fpath + '.hdf', key='df')
        else:
            ext = os.path.splitext(path)[-1]
            if ext.lower() == '.csv':
                out.to_csv(path)
            elif ext.lower() == '.hdf':
                out.to_hdf(path, key='df')
    return out
    
@global_task(log)
def compile_AAR_from_ELA(gdirs, filesuffix='', path=True, csv=True, ela=None, year_range=np.arange(1985, 2021)):
    """Compiles a table of AAR.

    The file is stored in a hdf file (not csv) per default. Use pd.read_hdf
    to open it.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location (file
        extension matters)
    csv: bool
        Set to store the data in csv instead of hdf.
    ela: DataFrame
        annual ela of each glacier
    year_range : iterable
        Years to compute AAR for (default 1985-2021)
    """

    out_df = execute_entity_task(compute_AAR_from_ELA, gdirs, ela=ela, year_range=np.arange(1985, 2021))

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.nan)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    #out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            fpath = os.path.join(cfg.PATHS['working_dir'],
                                 'AAR_1985_2021_Loibl' + filesuffix)
            if csv:
                out.to_csv(fpath + '.csv')
            else:
                out.to_hdf(fpath + '.hdf', key='df')
        else:
            ext = os.path.splitext(path)[-1]
            if ext.lower() == '.csv':
                out.to_csv(path)
            elif ext.lower() == '.hdf':
                out.to_hdf(path, key='df')
    return out

for i in range(1,7):
    cfg.initialize(logging_level='WARNING')
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['dl_verify'] = False
    cfg.PARAMS['has_internet'] = False
    cfg.PATHS['dl_cache_dir'] = '/g/data/rd53/OGGM/download_cache/'
    cfg.PATHS['working_dir'] = '/scratch/k10/wy2165/Loibl/'
    
    filepath = '/g/data/rd53/wy2165/disequilibrium/data/';
    data = xr.open_dataset(filepath + 'ELA_1985_2021_Loibl.nc')
    gdf_sel = (data['RGIId'].values).tolist()
    
    base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2025.6/elev_bands/ERA5/per_glacier_spinup/'
    if i!=6:
        gdirs = workflow.init_glacier_directories(gdf_sel[5000*(i-1):5000*i], from_prepro_level=5, prepro_border=160, prepro_base_url=base_url)
    else:
        gdirs = workflow.init_glacier_directories(gdf_sel[5000*(i-1):], from_prepro_level=5, prepro_border=160, prepro_base_url=base_url)
        
    ela = data['ELA']
    compile_AAR_from_ELA(gdirs, filesuffix='_' + str(i), path=True, csv=True, ela=ela);
    
    folder_path = cfg.PATHS['working_dir'] + 'per_glacier'
    shutil.rmtree(folder_path)
    
