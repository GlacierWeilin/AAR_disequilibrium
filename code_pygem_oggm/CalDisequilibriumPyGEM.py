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
def cal_AAR_steady(gdir,
                   pygem_prms = None,
                   region = None,
                   gcm = None,
                   ssp = None,
                   calibration = None,
                   bias_adj = None,
                   sets = None,
                   startyear = None,
                   endyear = None,
                   benchmark = None,
                   overwrite_gdir = True):
    """
    
    Calculated glacier steady-state AAR based on RGI glacier geometry, simulated ELA and mass balance.

    """
    
    # RGIID
    rgiid = gdir.rgi_id
    if region < 10:
        glac_no = rgiid[7:]
    else:
        glac_no = rgiid[6:]
    
    # Read the existing glacier_AAR_steady.json if it exists, otherwise initialize empty dict
    fp = gdir.get_filepath('glacier_AAR_steady')
    
    if os.path.exists(fp):
        with open(fp, 'w') as f:
            f.write('')
        simu = {}
    else:
        simu = {}
        
    # save rgiid
    simu['rgi_id'] = rgiid
    
    ####### load PyGEM outputs
    simpath = pygem_prms['root'] + '/Output/simulations_oggm'
    base_dir = simpath + '/' + str(region).zfill(2) + '/'

    AAR_steady = np.nan
    
    if gcm == 'ERA5':
        fstats = Path(
            f'{base_dir}{gcm}/stats/'
            f'{glac_no}_{gcm}_{calibration}'
            f'_ba{bias_adj}_{sets}sets_'
            f'{startyear}_{endyear}_all.nc'
        )
        
        fbinned = Path(
            f'{base_dir}{gcm}/binned/'
            f'{glac_no}_{gcm}_{calibration}'
            f'_ba{bias_adj}_{sets}sets_'
            f'{startyear}_{endyear}_binned.nc'
        )
    else:
        fstats = Path(
            f'{base_dir}{gcm}/{ssp}/stats/'
            f'{glac_no}_{gcm}_{ssp}_{calibration}'
            f'_ba{bias_adj}_{sets}sets_'
            f'{startyear}_{endyear}_all.nc'
        )
        
        fbinned = Path(
            f'{base_dir}{gcm}/{ssp}/binned/'
            f'{glac_no}_{gcm}_{ssp}_{calibration}'
            f'_ba{bias_adj}_{sets}sets_'
            f'{startyear}_{endyear}_binned.nc'
        )
    
    if fstats.exists() and fbinned.exists():
        
        # Annual area, total mass balance, ELA
        stats = xr.open_dataset(fstats)
        
        # glacier area
        annual_area = stats['glac_area_annual'] # m2
        areas = annual_area.sel(year=slice(startyear, endyear))
        
        # glacier mass
        annual_mass = stats['glac_mass_annual'] # kg
        mass = annual_mass.sel(year=slice(startyear, endyear))

        simu[f'area_{startyear:d}'] = areas[0][0].values.item()
        simu[f'area_{benchmark:d}'] = areas[0].sel(year=benchmark).item()
        
        simu[f'mass_{startyear:d}'] = mass[0][0].values.item()
        simu[f'mass_{benchmark:d}'] = mass[0].sel(year=benchmark).values.item()
        
        # glacier mass balance
        annual_mb = stats['glac_massbaltotal'].groupby('time.year').sum('time') # m3 w.e.
        mbs = annual_mb.sel(year=slice(startyear, endyear))

        areas = areas.values.squeeze()
        mbs = mbs.values.squeeze()
        
        # glacier ELA
        annual_ela = stats['glac_ELA_annual'] # m
        elas = annual_ela.sel(year=slice(startyear, endyear))
        elas = elas.values.squeeze()
        
        # Binned glacier areas and surface elevations
        binned = xr.open_dataset(fbinned)
        bin_surface_h_initial = binned['bin_surface_h_initial']
        bin_surface_h_initial = bin_surface_h_initial.values.squeeze()
        
        # bin ice thickness
        bin_thick_annual = binned['bin_thick_annual']        
        bin_thick_annual = bin_thick_annual.sel(year=slice(startyear, endyear))
        bin_thick_annual = bin_thick_annual.values.squeeze()
        bin_thick_annual_init = bin_thick_annual[:,0]
        
        bin_surface_h = (bin_surface_h_initial[:, np.newaxis] - bin_thick_annual_init[:, np.newaxis]) + bin_thick_annual
        
        # bin area
        bin_area_annual = binned['bin_area_annual']
        bin_area_annual = bin_area_annual.sel(year=slice(startyear, endyear))
        bin_area_annual = bin_area_annual.values.squeeze()
        
        valid = (~np.isnan(elas) & ~np.isnan(mbs) & ~np.isnan(areas) & (areas != 0))
        n_nan = np.sum(~valid)
        
        mbs = mbs[valid]
        areas = areas[valid]
        elas = elas[valid]

        mbs = mbs / areas * 1000 # mm w.e.

        bin_thick_annual = bin_thick_annual[:,valid]
        bin_surface_h = bin_surface_h[:,valid]
        bin_area_annual = bin_area_annual[:,valid]
        
        if n_nan < endyear - startyear + 1:
            
            aars = []
            for i, ela in enumerate(elas):
                bin_thick = bin_thick_annual[:,i]
                surface_h = bin_surface_h[:,i]
                bin_area = bin_area_annual[:,i]

                is_glacier = bin_thick > 0
                area = np.sum(bin_area[is_glacier])
                
                above_ela = surface_h > ela
                acc_area = np.sum(bin_area[above_ela])

                if area > 0:
                    aars.append(acc_area / area)
                else:
                    aars.append(np.nan)

            aars = np.array(aars)
            mask = (np.isfinite(aars) & (aars >= 0) & (aars <= 1))

            aars = aars[mask]
            mbs = mbs[mask]

            if len(aars) >= 5:
                # calculate AAR0 by linear regression
                slope, intercept, r_value, p_value, std_err = st.linregress(mbs, aars);
                
                if intercept >= 0.05 and intercept <= 0.95:
                    AAR_steady = intercept
            
        stats.close()
        binned.close()
        
    else:
        simu[f'area_{startyear:d}'] = np.nan
        simu[f'area_{benchmark:d}'] = np.nan
        
        simu[f'mass_{startyear:d}'] = np.nan
        simu[f'mass_{benchmark:d}'] = np.nan
        
    # save outputs
    simu['AAR_steady'] = AAR_steady
                 
    # Check if the file already exists and raise an error if overwrite is False
    if gdir.has_file('glacier_AAR_steady') and not overwrite_gdir:
        raise InvalidWorkflowError(
            'glacier_AAR_steady.json already exists for this glacier. '
            'Set overwrite_gdir=True to overwrite a previous calibration.'
        )
    
    # Write the observational data dictionary to glacier_AAR.json in the glacier directory
    gdir.write_json(simu, 'glacier_AAR_steady');

@entity_task(log)
def glacier_AAR_steady(gdir):
    """Gather as much statistics as possible about this glacier.

    It can be used to do result diagnostics and other stuffs. If the data
    necessary for a statistic is not available (e.g.: flowlines length) it
    will simply be ignored.

    Parameters
    ----------
    """
    
    if not gdir.has_file('glacier_AAR_steady'):
            raise InvalidWorkflowError(
                '`glacier_AAR_steady.json` not found for this glacier.'
            )

    # Read glacier_disequilibrium data (must exist)
    df = gdir.read_json('glacier_AAR_steady')
    
    odf = pd.Series(df)
    
    return odf

@global_task(log)
def compile_AAR_steady(gdirs, filesuffix='', path=True):
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

    out_df = execute_entity_task(glacier_AAR_steady, gdirs)

    out = pd.DataFrame(out_df).set_index('rgi_id')

    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('glacier_AAR_steady' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)
    return out