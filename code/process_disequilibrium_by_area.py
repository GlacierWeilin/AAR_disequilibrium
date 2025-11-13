#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:31:13 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import xarray as xr
from scipy.stats import median_abs_deviation

import pygem.pygem_input as pygem_prms

def calc_stats_AAR(output):
    """
    Calculate stats for a given variable

    Parameters
    ----------
    vn : str
        variable name
    ds : xarray output_ds_allset
        output_ds_allset of output with all ensemble simulations

    Returns
    -------
    stats : np.array
        Statistics related to a given variable
    """
    if output.ndim == 2:
        output_ds_all = output[:,0];
        stats = None
        stats = np.nanmean(output_ds_all) # 'mean'
        stats = np.append(stats, np.nanstd(output_ds_all)) # 'std'
        stats = np.append(stats, np.nanpercentile(output_ds_all, 5)) # '5%'
        stats = np.append(stats, np.nanpercentile(output_ds_all, 17)) # '17%'
        stats = np.append(stats, np.nanmedian(output_ds_all)) # 'median'
        stats = np.append(stats, np.nanpercentile(output_ds_all, 83)) # '83%'
        stats = np.append(stats, np.nanpercentile(output_ds_all, 95)) # '95%'
        stats = np.append(stats, median_abs_deviation(output_ds_all, nan_policy='omit')) # Compute the median absolute deviation of the output_ds_all
    
        stats = np.append(stats, np.sum(~np.isnan(output_ds_all))) # n
        stats = np.append(stats, np.nansum(output_ds_all)) # sum
        output_ds_all  = output[:,1]; output_ds_all = output_ds_all*output_ds_all;
        stats = np.append(stats, np.sqrt(np.nansum(output_ds_all))/stats[8])
        stats = np.append(stats, np.sqrt(np.nansum(output_ds_all)))
    elif output.ndim == 1:
        output_ds_all = output[:];
        stats = None
        stats = np.nanmean(output_ds_all) # 'mean'
        stats = np.append(stats, np.nanstd(output_ds_all)) # 'std'
        stats = np.append(stats, np.nanpercentile(output_ds_all, 5)) # '5%'
        stats = np.append(stats, np.nanpercentile(output_ds_all, 17)) # '17%'
        stats = np.append(stats, np.nanmedian(output_ds_all)) # 'median'
        stats = np.append(stats, np.nanpercentile(output_ds_all, 83)) # '83%'
        stats = np.append(stats, np.nanpercentile(output_ds_all, 95)) # '95%'
        stats = np.append(stats, median_abs_deviation(output_ds_all, nan_policy='omit')) # Compute the median absolute deviation of the output_ds_all
    
        stats = np.append(stats, np.sum(~np.isnan(output_ds_all))) # n
        stats = np.append(stats, np.nansum(output_ds_all)) # sum
        output_ds_all  = np.zeros(np.shape(output_ds_all)); output_ds_all = output_ds_all*output_ds_all;
        stats = np.append(stats, np.sqrt(np.nansum(output_ds_all))/stats[8])
        stats = np.append(stats, np.sqrt(np.nansum(output_ds_all)))
        
    return stats

#%% ===== Regional analysis =====

# prepare output
ds = xr.Dataset();
    
# Attributes
ds.attrs['Source'] = 'PyGEMv0.2.5 developed by David Rounce (drounce@alaska.edu)'
ds.attrs['Further developed by'] = 'Weilin Yang (weilinyang.yang@monash.edu)'
ds.attrs['Code reviewed by'] = 'Wenchao Chu (peterchuwenchao@foxmail.com)'

# Coordinates
ds.coords['dim'] = ('dim', np.arange(12))
ds['dim'].attrs['description'] = '0-mean, 1-std, 2-5%, 3-17%, 4-median, 5-83%, 6-95%, 7-mad, \
    8-n, 9-sum(std), 10-mean_std(std), 11-sum_std(std)'

ds.coords['variables'] = ('variables', np.arange(14))
ds['variables'].attrs['description'] = '0-ELA_mean, 1-AAR_mean, 2-AAR, \
    3-a, 4-dA, 5-dV, 6-ELA_steady, 7-THAR, 8-dV_bwl, 9-dV_eff, 10-SLR, 11-area, 12-area_2020, 13-volume_2020'

ds.coords['area_dim'] = ('area_dim', np.arange(4))
ds['area_dim'].attrs['area_dim'] = '0-area_0, 1-area_1, 2-area_10, 3-area_100'
ds['by_area'] = (('area_dim', 'variables','dim'), np.zeros([4,14,12])*np.nan)

# %% ===== all ======
output_ds_all = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/Output/ERA5_MCMC_ba1_2014_2023_corrected.nc');

find_id = np.where(output_ds_all['area_2020'].values[:,0]/1e6<=1)[0]
i=0
ds['by_area'].values[i,0,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_mean'].values[find_id,:]);
ds['by_area'].values[i,1,:] = calc_stats_AAR(output_ds_all['parameterization_AAR_mean'].values[find_id,:]);
ds['by_area'].values[i,2,:] = calc_stats_AAR(output_ds_all['parameterization_AAR'].values[find_id,:]);
ds['by_area'].values[i,3,:] = calc_stats_AAR(output_ds_all['parameterization_a'].values[find_id,:]);
ds['by_area'].values[i,4,:] = calc_stats_AAR(output_ds_all['parameterization_dA'].values[find_id,:]);
ds['by_area'].values[i,5,:] = calc_stats_AAR(output_ds_all['parameterization_dV'].values[find_id,:]);
ds['by_area'].values[i,6,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_steady'].values[find_id,:]);
ds['by_area'].values[i,7,:] = calc_stats_AAR(output_ds_all['parameterization_THAR'].values[find_id,:]);
ds['by_area'].values[i,8,:] = calc_stats_AAR(output_ds_all['parameterization_dV_bwl'].values[find_id,:]);
ds['by_area'].values[i,9,:] = ds['by_area'].values[i,5,:] - ds['by_area'].values[i,8,:];
ds['by_area'].values[i,10,:] = abs(ds['by_area'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);
ds['by_area'].values[i,11,:] = calc_stats_AAR(output_ds_all['Area'].values[find_id]);
ds['by_area'].values[i,12,:] = calc_stats_AAR(output_ds_all['area_2020'].values[find_id,:]);
ds['by_area'].values[i,13,:] = calc_stats_AAR(output_ds_all['volume_2020'].values[find_id,:]);

find_id = np.where((output_ds_all['area_2020'].values[:,0]/1e6>1) & (output_ds_all['area_2020'].values[:,0]/1e6<=10))[0]
i=1
ds['by_area'].values[i,0,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_mean'].values[find_id,:]);
ds['by_area'].values[i,1,:] = calc_stats_AAR(output_ds_all['parameterization_AAR_mean'].values[find_id,:]);
ds['by_area'].values[i,2,:] = calc_stats_AAR(output_ds_all['parameterization_AAR'].values[find_id,:]);
ds['by_area'].values[i,3,:] = calc_stats_AAR(output_ds_all['parameterization_a'].values[find_id,:]);
ds['by_area'].values[i,4,:] = calc_stats_AAR(output_ds_all['parameterization_dA'].values[find_id,:]);
ds['by_area'].values[i,5,:] = calc_stats_AAR(output_ds_all['parameterization_dV'].values[find_id,:]);
ds['by_area'].values[i,6,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_steady'].values[find_id,:]);
ds['by_area'].values[i,7,:] = calc_stats_AAR(output_ds_all['parameterization_THAR'].values[find_id,:]);
ds['by_area'].values[i,8,:] = calc_stats_AAR(output_ds_all['parameterization_dV_bwl'].values[find_id,:]);
ds['by_area'].values[i,9,:] = ds['by_area'].values[i,5,:] - ds['by_area'].values[i,8,:];
ds['by_area'].values[i,10,:] = abs(ds['by_area'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);
ds['by_area'].values[i,11,:] = calc_stats_AAR(output_ds_all['Area'].values[find_id]);
ds['by_area'].values[i,12,:] = calc_stats_AAR(output_ds_all['area_2020'].values[find_id,:]);
ds['by_area'].values[i,13,:] = calc_stats_AAR(output_ds_all['volume_2020'].values[find_id,:]);

find_id = np.where((output_ds_all['area_2020'].values[:,0]/1e6>10) & (output_ds_all['area_2020'].values[:,0]/1e6<=100))[0]
i=2
ds['by_area'].values[i,0,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_mean'].values[find_id,:]);
ds['by_area'].values[i,1,:] = calc_stats_AAR(output_ds_all['parameterization_AAR_mean'].values[find_id,:]);
ds['by_area'].values[i,2,:] = calc_stats_AAR(output_ds_all['parameterization_AAR'].values[find_id,:]);
ds['by_area'].values[i,3,:] = calc_stats_AAR(output_ds_all['parameterization_a'].values[find_id,:]);
ds['by_area'].values[i,4,:] = calc_stats_AAR(output_ds_all['parameterization_dA'].values[find_id,:]);
ds['by_area'].values[i,5,:] = calc_stats_AAR(output_ds_all['parameterization_dV'].values[find_id,:]);
ds['by_area'].values[i,6,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_steady'].values[find_id,:]);
ds['by_area'].values[i,7,:] = calc_stats_AAR(output_ds_all['parameterization_THAR'].values[find_id,:]);
ds['by_area'].values[i,8,:] = calc_stats_AAR(output_ds_all['parameterization_dV_bwl'].values[find_id,:]);
ds['by_area'].values[i,9,:] = ds['by_area'].values[i,5,:] - ds['by_area'].values[i,8,:];
ds['by_area'].values[i,10,:] = abs(ds['by_area'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);
ds['by_area'].values[i,11,:] = calc_stats_AAR(output_ds_all['Area'].values[find_id]);
ds['by_area'].values[i,12,:] = calc_stats_AAR(output_ds_all['area_2020'].values[find_id,:]);
ds['by_area'].values[i,13,:] = calc_stats_AAR(output_ds_all['volume_2020'].values[find_id,:]);

find_id = np.where(output_ds_all['area_2020'].values[:,0]/1e6>100)[0]
i=3
ds['by_area'].values[i,0,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_mean'].values[find_id,:]);
ds['by_area'].values[i,1,:] = calc_stats_AAR(output_ds_all['parameterization_AAR_mean'].values[find_id,:]);
ds['by_area'].values[i,2,:] = calc_stats_AAR(output_ds_all['parameterization_AAR'].values[find_id,:]);
ds['by_area'].values[i,3,:] = calc_stats_AAR(output_ds_all['parameterization_a'].values[find_id,:]);
ds['by_area'].values[i,4,:] = calc_stats_AAR(output_ds_all['parameterization_dA'].values[find_id,:]);
ds['by_area'].values[i,5,:] = calc_stats_AAR(output_ds_all['parameterization_dV'].values[find_id,:]);
ds['by_area'].values[i,6,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_steady'].values[find_id,:]);
ds['by_area'].values[i,7,:] = calc_stats_AAR(output_ds_all['parameterization_THAR'].values[find_id,:]);
ds['by_area'].values[i,8,:] = calc_stats_AAR(output_ds_all['parameterization_dV_bwl'].values[find_id,:]);
ds['by_area'].values[i,9,:] = ds['by_area'].values[i,5,:] - ds['by_area'].values[i,8,:];
ds['by_area'].values[i,10,:] = abs(ds['by_area'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);
ds['by_area'].values[i,11,:] = calc_stats_AAR(output_ds_all['Area'].values[find_id]);
ds['by_area'].values[i,12,:] = calc_stats_AAR(output_ds_all['area_2020'].values[find_id,:]);
ds['by_area'].values[i,13,:] = calc_stats_AAR(output_ds_all['volume_2020'].values[find_id,:]);

#%% To file
path = '/Users/wyan0065/Desktop/PyGEM/calving/Output/' + 'results_by_area.nc';
enc_var = {'dtype': 'float32'}
encoding = {v: enc_var for v in ds.data_vars}
ds.to_netcdf(path, encoding=encoding);