#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Weilin Yang (weilinyang.yang@monash.edu; ywlcwc@gmail.com)
"""

import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as st
from pathlib import Path

from oggm import cfg, entity_task, global_task
from oggm.exceptions import InvalidWorkflowError

log = logging.getLogger(__name__)
@entity_task(log)
def cal_disequilibrium_ERA5(gdir,
                            pygem_prms = None,
                            region = None,
                            gcm = None, 
                            calibration = None, 
                            bias_adj = None, 
                            sets = None,
                            startyear = None,
                            endyear = None,
                            overwrite_gdir = True):
    """
    
    Glacier climate disequilibrium analysis based on simulated ELA and mass balance

    disequilibrium = AAR_mean/AAR_steady
    Glacier area at steady-state: Area = Area_init * disequilibrium
    Glacier volume: Volume = c * Area**r

    """
    
    # RGIID
    rgiid = gdir.rgi_id
    if region < 10:
        glac_no = rgiid[7:]
    else:
        glac_no = rgiid[6:]
    

    # Read the existing glacier_disequilibrium.json if it exists, otherwise initialize empty dict
    fp = gdir.get_filepath('glacier_disequilibrium')
    
    if os.path.exists(fp):
        with open(fp, 'w') as f:
            f.write('')
        simu = {}
    else:
        simu = {}
        
    # save rgiid
    simu['rgi_id'] = rgiid
    
    if gdir.has_file('model_flowlines'):
        # model_flowlines
        fls = gdir.read_pickle('model_flowlines');
        fl = fls[0];
        tag = 1;
    else:
        tag = 0;
        
    ####### load PyGEM outputs
    simpath = pygem_prms['root'] + '/Output/simulations_none'
    base_dir = simpath + '/' + str(region).zfill(2) + '/'
    
    fp = Path(
        f'{base_dir}{gcm}/stats/'
        f'{glac_no}_{gcm}_{calibration}'
        f'_ba{bias_adj}_{sets}sets_'
        f'{startyear}_{endyear}_all.nc'
    )
    
    if fp.exists():
        data = xr.open_dataset(fp)
        exist = 1
    else:
        exist = 0
    
    AAR_mean = np.nan
    
    if exist == 1 and tag == 1:
        
       annual_ela = data['glac_ELA_annual'] # m
       
       elas = annual_ela.sel(year=slice(startyear, endyear))
       elas = elas.values.squeeze()
       
       valid = ~(np.isnan(elas))
       n_nan = np.sum(~valid)

       elas = elas[valid]
       
       #if n_nan >= 5: # 5
       if n_nan == endyear - startyear + 1:

           AAR_mean = 0
                   
       else:
           aars = []
           mask = fl.thick > 0
           widths = np.sum(fl.widths_m[mask])
           for ela in elas:
               mask = fl.surface_h > ela
               acc_area = np.sum(fl.widths_m[mask])

               aars.append(acc_area / widths)

           aars = np.array(aars)
           mask = (aars >= 0) & (aars <= 1)

           aars = aars[mask]
           if len(aars) != 0:
               AAR_mean = np.nanmean(aars)
           else:
               AAR_mean = np.nan
       
       data.close()
        

    # save outputs
    # the glacier disequilibrium for each period
    column_name = f'{gcm.lower()}_{startyear}-{endyear}_AAR_mean'
    simu[column_name] = AAR_mean
                 
    # Check if the file already exists and raise an error if overwrite is False
    if gdir.has_file('glacier_disequilibrium') and not overwrite_gdir:
        raise InvalidWorkflowError(
            'glacier_disequilibrium.json already exists for this glacier. '
            'Set overwrite_gdir=True to overwrite a previous calibration.'
        )
    
    # Write the observational data dictionary to inv_obs.json in the glacier directory
    gdir.write_json(simu, 'glacier_disequilibrium');


log = logging.getLogger(__name__)
@entity_task(log)
def cal_disequilibrium(gdir, 
                       pygem_prms = None,
                       region = None,
                       gcm = None, 
                       ssp = None, 
                       calibration = None, 
                       bias_adj = None, 
                       sets = None, 
                       startyears = None,
                       endyears = None,
                       overwrite_gdir = True):
    """
    
    Glacier climate disequilibrium analysis based on simulated ELA and mass balance

    disequilibrium = AAR_mean/AAR_steady
    Glacier area at steady-state: Area = Area_init * disequilibrium
    Glacier volume: Volume = c * Area**r

    """
    
    # RGIID
    rgiid = gdir.rgi_id
    if region < 10:
        glac_no = rgiid[7:]
    else:
        glac_no = rgiid[6:]
    

    # Read the existing glacier_disequilibrium.json if it exists, otherwise initialize empty dict
    fp = gdir.get_filepath('glacier_disequilibrium')
    
    if os.path.exists(fp):
        with open(fp, 'w') as f:
            f.write('')
        simu = {}
    else:
        simu = {}
        
    # save rgiid
    simu['rgi_id'] = rgiid
    
    if gdir.has_file('model_flowlines'):
        # model_flowlines
        fls = gdir.read_pickle('model_flowlines');
        fl = fls[0];
        tag = 1;
    else:
        tag = 0;
        
    ####### load PyGEM outputs
    simpath = pygem_prms['root'] + '/Output/simulations_none'
    base_dir = simpath + '/' + str(region).zfill(2) + '/'
    
    if ssp == 'ssp126':
        starts = startyears
        ends = endyears
    else:
        starts = startyears[4:]
        ends = endyears[4:]
        
    fp = Path(
        f'{base_dir}{gcm}/{ssp}/stats/'
        f'{glac_no}_{gcm}_{ssp}_{calibration}'
        f'_ba{bias_adj}_{sets}sets_'
        f'{startyears[0]}_{endyears[-1]}_all.nc'
    )
    
    if fp.exists():
        data = xr.open_dataset(fp)
        exist = 1
    else:
        exist = 0
    
    for i, year in enumerate(starts):
        
        AAR_mean = np.nan
        
        if exist == 1 and tag == 1:
            
           annual_ela = data['glac_ELA_annual'] # m
           
           elas = annual_ela.sel(year=slice(starts[i], ends[i]))
           elas = elas.values.squeeze()
           
           valid = ~(np.isnan(elas))
           n_nan = np.sum(~valid)

           elas = elas[valid]
           
           #if n_nan >= 10: # 10
           if n_nan == 20:

               AAR_mean = 0
                       
           else:
               aars = []
               mask = fl.thick > 0
               widths = np.sum(fl.widths_m[mask])
               for ela in elas:
                   mask = fl.surface_h > ela
                   acc_area = np.sum(fl.widths_m[mask])

                   aars.append(acc_area / widths)

               aars = np.array(aars)
               mask = (aars >= 0) & (aars <= 1)

               aars = aars[mask]
               if len(aars) != 0:
                   AAR_mean = np.nanmean(aars)
               else:
                   AAR_mean = np.nan
   
        # save outputs
        # the glacier disequilibrium for each period
        if starts[i] < 2021:
            column_name = f'{gcm.lower()}_{starts[i]}-{ends[i]}_hist_AAR_mean'
            simu[column_name] = AAR_mean
           
        else:
            column_name = f'{gcm.lower()}_{starts[i]}-{ends[i]}_{ssp}_AAR_mean'
            simu[column_name] = AAR_mean
       
    if exist == 1:
        data.close()
                 
    # Check if the file already exists and raise an error if overwrite is False
    if gdir.has_file('glacier_disequilibrium') and not overwrite_gdir:
        raise InvalidWorkflowError(
            'glacier_disequilibrium.json already exists for this glacier. '
            'Set overwrite_gdir=True to overwrite a previous calibration.'
        )
    
    # Write the observational data dictionary to glacier_disequilibrium.json in the glacier directory
    gdir.write_json(simu, 'glacier_disequilibrium');


@entity_task(log)
def glacier_disequilibrium(gdir):
    """Gather as much statistics as possible about this glacier.

    It can be used to do result diagnostics and other stuffs. If the data
    necessary for a statistic is not available (e.g.: flowlines length) it
    will simply be ignored.

    Parameters
    ----------
    """
    
    if not gdir.has_file('glacier_disequilibrium'):
            raise InvalidWorkflowError(
                '`glacier_disequilibrium.json` not found for this glacier.'
            )

    # Read glacier_disequilibrium data (must exist)
    df = gdir.read_json('glacier_disequilibrium')
    
    odf = pd.Series(df)
    
    return odf

@global_task(log)
def compile_disequilibrium(gdirs, filesuffix='', path=True):
    """Gather as much statistics as possible about a list of glaciers.

    It can be used to do result diagnostics and other stuffs. If the data
    necessary for a statistic is not available (e.g.: flowlines length) it
    will simply be ignored.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    """
    from oggm.workflow import execute_entity_task

    out_df = execute_entity_task(glacier_disequilibrium, gdirs)

    out = pd.DataFrame(out_df).set_index('rgi_id')

    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('glacier_disequilibrium' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)
    return out