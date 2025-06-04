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
    ds : xarray dataset
        dataset of output with all ensemble simulations

    Returns
    -------
    stats : np.array
        Statistics related to a given variable
    """
    if output.ndim == 2:
        data = output[:,0];
        stats = None
        stats = np.nanmean(data) # 'mean'
        stats = np.append(stats, np.nanstd(data)) # 'std'
        stats = np.append(stats, np.nanpercentile(data, 5)) # '5%'
        stats = np.append(stats, np.nanpercentile(data, 17)) # '17%'
        stats = np.append(stats, np.nanmedian(data)) # 'median'
        stats = np.append(stats, np.nanpercentile(data, 83)) # '83%'
        stats = np.append(stats, np.nanpercentile(data, 95)) # '95%'
        stats = np.append(stats, median_abs_deviation(data, nan_policy='omit')) # Compute the median absolute deviation of the data
    
        stats = np.append(stats, np.sum(~np.isnan(data))) # n
        stats = np.append(stats, np.nansum(data)) # sum
        data  = output[:,1]; data = data*data;
        stats = np.append(stats, np.sqrt(np.nansum(data))/stats[8])
        stats = np.append(stats, np.sqrt(np.nansum(data)))
    elif output.ndim == 1:
        data = output[:];
        stats = None
        stats = np.nanmean(data) # 'mean'
        stats = np.append(stats, np.nanstd(data)) # 'std'
        stats = np.append(stats, np.nanpercentile(data, 5)) # '5%'
        stats = np.append(stats, np.nanpercentile(data, 17)) # '17%'
        stats = np.append(stats, np.nanmedian(data)) # 'median'
        stats = np.append(stats, np.nanpercentile(data, 83)) # '83%'
        stats = np.append(stats, np.nanpercentile(data, 95)) # '95%'
        stats = np.append(stats, median_abs_deviation(data, nan_policy='omit')) # Compute the median absolute deviation of the data
    
        stats = np.append(stats, np.sum(~np.isnan(data))) # n
        stats = np.append(stats, np.nansum(data)) # sum
        data  = np.zeros(np.shape(data)); data = data*data;
        stats = np.append(stats, np.sqrt(np.nansum(data))/stats[8])
        stats = np.append(stats, np.sqrt(np.nansum(data)))
        
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

ds.coords['glac_dim'] = ('glac_dim', np.arange(3))
ds['glac_dim'].attrs['description'] = '0-area,1-area_2020,2-volume_2020'

ds.coords['variables'] = ('variables', np.arange(11))
ds['variables'].attrs['description'] = '0-ELA_mean, 1-AAR_mean, 2-AAR, \
    3-a, 4-dA, 5-dV, 6-ELA_steady, 7-THAR, 8-dV_bwl, 9-dV_eff, 10-SLR'

ds.coords['region'] = ('region', np.arange(20))
ds['region'].attrs['description'] = '0-all, 1-19-01Region'
ds.coords['falseortrue'] = ('falseortrue', np.arange(2))
ds['falseortrue'].attrs['description'] = '0-False, 1-True'

# glac_results
ds['glac_region'] = (('region', 'glac_dim','dim'), np.zeros([20,3,12])*np.nan)
ds['glac_region'].attrs['description'] = 'AAR results of each region (linear regression)'
ds['glac_icecap'] = (('falseortrue', 'glac_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['glac_icecap'].attrs['description'] = 'AAR results of ice caps and glaciers (linear regression)'
ds['glac_debris'] = (('falseortrue', 'glac_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['glac_debris'].attrs['description'] = 'AAR results of debris covered and no debris coverd glaciers (linear regression)'
ds['glac_tidewater'] = (('falseortrue', 'glac_dim','dim'), np.zeros([2,3,12])*np.nan)
ds['glac_tidewater'].attrs['description'] = 'AAR results of marine and land terminating glaciers (linear regression)'

# Results by linear regression
ds['intercept_region'] = (('region', 'variables','dim'), np.zeros([20,11,12])*np.nan)
ds['intercept_region'].attrs['description'] = 'AAR results of each region (linear regression)'
ds['intercept_icecap'] = (('falseortrue', 'variables','dim'), np.zeros([2,11,12])*np.nan)
ds['intercept_icecap'].attrs['description'] = 'AAR results of ice caps and glaciers (linear regression)'
ds['intercept_debris'] = (('falseortrue', 'variables','dim'), np.zeros([2,11,12])*np.nan)
ds['intercept_debris'].attrs['description'] = 'AAR results of debris covered and no debris coverd glaciers (linear regression)'
ds['intercept_tidewater'] = (('falseortrue', 'variables','dim'), np.zeros([2,11,12])*np.nan)
ds['intercept_tidewater'].attrs['description'] = 'AAR results of marine and land terminating glaciers (linear regression)'

# Equilibrium experiments
ds['equil_region'] = (('region', 'variables','dim'), np.zeros([20,11,12])*np.nan)
ds['equil_region'].attrs['description'] = 'AAR results of each region (equilibrium experiment)'
ds['equil_icecap'] = (('falseortrue', 'variables','dim'), np.zeros([2,11,12])*np.nan)
ds['equil_icecap'].attrs['description'] = 'AAR results of ice caps and glaciers (equilibrium experiment)'
ds['equil_debris'] = (('falseortrue', 'variables','dim'), np.zeros([2,11,12])*np.nan)
ds['equil_debris'].attrs['description'] = 'AAR results of debris covered and no debris coverd glaciers (equilibrium experiment)'
ds['equil_tidewater'] = (('falseortrue', 'variables','dim'), np.zeros([2,11,12])*np.nan)
ds['equil_tidewater'].attrs['description'] = 'AAR results of marine and land terminating glaciers (equilibrium experiment)'

# parameterization results
ds['parameterization_region'] = (('region', 'variables','dim'), np.zeros([20,11,12])*np.nan)
ds['parameterization_region'].attrs['description'] = 'AAR results of each region'
ds['parameterization_icecap'] = (('falseortrue', 'variables','dim'), np.zeros([2,11,12])*np.nan)
ds['parameterization_icecap'].attrs['description'] = 'AAR results of ice caps and glaciers'
ds['parameterization_debris'] = (('falseortrue', 'variables','dim'), np.zeros([2,11,12])*np.nan)
ds['parameterization_debris'].attrs['description'] = 'AAR results of debris covered and no debris coverd glaciers'
ds['parameterization_tidewater'] = (('falseortrue', 'variables','dim'), np.zeros([2,11,12])*np.nan)
ds['parameterization_tidewater'].attrs['description'] = 'AAR results of marine and land terminating glaciers'

# %% ===== all ======
output_ds_all = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/Output/ERA5_MCMC_ba1_2014_2023_corrected.nc');
i=0

# glac_results
ds['glac_region'].values[i,0,:] = calc_stats_AAR(output_ds_all['Area'].values);
ds['glac_region'].values[i,1,:] = calc_stats_AAR(output_ds_all['area_2020'].values);
ds['glac_region'].values[i,2,:] = calc_stats_AAR(output_ds_all['volume_2020'].values);

# intercept_results
# 0-intercept_ELA_mean, 1-intercept_AAR_mean, 2-intercept_AAR, \
# 3-intercept_a, 4-intercept_dA, 5-intercept_dV, 6-intercept_ELA_steady, 7-intercept_THAR, \
# 8-intercept_dV_bwl, 9-intercept_dV_eff, 10-SLR
ds['intercept_region'].values[i,0,:] = calc_stats_AAR(output_ds_all['intercept_ELA_mean'].values);
ds['intercept_region'].values[i,1,:] = calc_stats_AAR(output_ds_all['intercept_AAR_mean'].values);
ds['intercept_region'].values[i,2,:] = calc_stats_AAR(output_ds_all['intercept_AAR'].values);
ds['intercept_region'].values[i,3,:] = calc_stats_AAR(output_ds_all['intercept_a'].values);
ds['intercept_region'].values[i,4,:] = calc_stats_AAR(output_ds_all['intercept_dA'].values);
ds['intercept_region'].values[i,5,:] = calc_stats_AAR(output_ds_all['intercept_dV'].values);
ds['intercept_region'].values[i,6,:] = calc_stats_AAR(output_ds_all['intercept_ELA_steady'].values);
ds['intercept_region'].values[i,7,:] = calc_stats_AAR(output_ds_all['intercept_THAR'].values);
ds['intercept_region'].values[i,8,:] = calc_stats_AAR(output_ds_all['intercept_dV_bwl'].values);
ds['intercept_region'].values[i,9,:] = ds['intercept_region'].values[i,5,:] - ds['intercept_region'].values[i,8,:];
ds['intercept_region'].values[i,10,:] = abs(ds['intercept_region'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);

# equil_results
ds['equil_region'].values[i,0,:] = calc_stats_AAR(output_ds_all['equil_ELA_mean'].values);
ds['equil_region'].values[i,1,:] = calc_stats_AAR(output_ds_all['equil_AAR_mean'].values);
ds['equil_region'].values[i,2,:] = calc_stats_AAR(output_ds_all['equil_AAR'].values);
ds['equil_region'].values[i,3,:] = calc_stats_AAR(output_ds_all['equil_a'].values);
ds['equil_region'].values[i,4,:] = calc_stats_AAR(output_ds_all['equil_dA'].values);
ds['equil_region'].values[i,5,:] = calc_stats_AAR(output_ds_all['equil_dV'].values);
ds['equil_region'].values[i,6,:] = calc_stats_AAR(output_ds_all['equil_ELA_steady'].values);
ds['equil_region'].values[i,7,:] = calc_stats_AAR(output_ds_all['equil_THAR'].values);
ds['equil_region'].values[i,8,:] = calc_stats_AAR(output_ds_all['equil_dV_bwl'].values);
ds['equil_region'].values[i,9,:] = ds['equil_region'].values[i,5,:] - ds['equil_region'].values[i,8,:];
ds['equil_region'].values[i,10,:] = abs(ds['equil_region'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);

# parameterization_results
ds['parameterization_region'].values[i,0,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_mean'].values);
ds['parameterization_region'].values[i,1,:] = calc_stats_AAR(output_ds_all['parameterization_AAR_mean'].values);
ds['parameterization_region'].values[i,2,:] = calc_stats_AAR(output_ds_all['parameterization_AAR'].values);
ds['parameterization_region'].values[i,3,:] = calc_stats_AAR(output_ds_all['parameterization_a'].values);
ds['parameterization_region'].values[i,4,:] = calc_stats_AAR(output_ds_all['parameterization_dA'].values);
ds['parameterization_region'].values[i,5,:] = calc_stats_AAR(output_ds_all['parameterization_dV'].values);
ds['parameterization_region'].values[i,6,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_steady'].values);
ds['parameterization_region'].values[i,7,:] = calc_stats_AAR(output_ds_all['parameterization_THAR'].values);
ds['parameterization_region'].values[i,8,:] = calc_stats_AAR(output_ds_all['parameterization_dV_bwl'].values);
ds['parameterization_region'].values[i,9,:] = ds['parameterization_region'].values[i,5,:] - ds['parameterization_region'].values[i,8,:];
ds['parameterization_region'].values[i,10,:] = abs(ds['parameterization_region'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);

#%% ===== Each region (01 region) =====
for i in range(1,20):
    # glac_results
    find_id = np.where(output_ds_all['O1Region'].values==i)[0];
    
    # glac_results
    ds['glac_region'].values[i,0,:] = calc_stats_AAR(output_ds_all['Area'].values[find_id]);
    ds['glac_region'].values[i,1,:] = calc_stats_AAR(output_ds_all['area_2020'].values[find_id,:]);
    ds['glac_region'].values[i,2,:] = calc_stats_AAR(output_ds_all['volume_2020'].values[find_id,:]);

    # intercept_results
    ds['intercept_region'].values[i,0,:] = calc_stats_AAR(output_ds_all['intercept_ELA_mean'].values[find_id,:]);
    ds['intercept_region'].values[i,1,:] = calc_stats_AAR(output_ds_all['intercept_AAR_mean'].values[find_id,:]);
    ds['intercept_region'].values[i,2,:] = calc_stats_AAR(output_ds_all['intercept_AAR'].values[find_id,:]);
    ds['intercept_region'].values[i,3,:] = calc_stats_AAR(output_ds_all['intercept_a'].values[find_id,:]);
    ds['intercept_region'].values[i,4,:] = calc_stats_AAR(output_ds_all['intercept_dA'].values[find_id,:]);
    ds['intercept_region'].values[i,5,:] = calc_stats_AAR(output_ds_all['intercept_dV'].values[find_id,:]);
    ds['intercept_region'].values[i,6,:] = calc_stats_AAR(output_ds_all['intercept_ELA_steady'].values[find_id,:]);
    ds['intercept_region'].values[i,7,:] = calc_stats_AAR(output_ds_all['intercept_THAR'].values[find_id,:]);
    ds['intercept_region'].values[i,8,:] = calc_stats_AAR(output_ds_all['intercept_dV_bwl'].values[find_id,:]);
    ds['intercept_region'].values[i,9,:] = ds['intercept_region'].values[i,5,:] - ds['intercept_region'].values[i,8,:];
    ds['intercept_region'].values[i,10,:] = abs(ds['intercept_region'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);
    
    # equil_results
    ds['equil_region'].values[i,0,:] = calc_stats_AAR(output_ds_all['equil_ELA_mean'].values[find_id,:]);
    ds['equil_region'].values[i,1,:] = calc_stats_AAR(output_ds_all['equil_AAR_mean'].values[find_id,:]);
    ds['equil_region'].values[i,2,:] = calc_stats_AAR(output_ds_all['equil_AAR'].values[find_id,:]);
    ds['equil_region'].values[i,3,:] = calc_stats_AAR(output_ds_all['equil_a'].values[find_id,:]);
    ds['equil_region'].values[i,4,:] = calc_stats_AAR(output_ds_all['equil_dA'].values[find_id,:]);
    ds['equil_region'].values[i,5,:] = calc_stats_AAR(output_ds_all['equil_dV'].values[find_id,:]);
    ds['equil_region'].values[i,6,:] = calc_stats_AAR(output_ds_all['equil_ELA_steady'].values[find_id,:]);
    ds['equil_region'].values[i,7,:] = calc_stats_AAR(output_ds_all['equil_THAR'].values[find_id,:]);
    ds['equil_region'].values[i,8,:] = calc_stats_AAR(output_ds_all['equil_dV_bwl'].values[find_id,:]);
    ds['equil_region'].values[i,9,:] = ds['equil_region'].values[i,5,:] - ds['equil_region'].values[i,8,:];
    ds['equil_region'].values[i,10,:] = abs(ds['equil_region'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);
    
    # parameterization_results
    ds['parameterization_region'].values[i,0,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_mean'].values[find_id,:]);
    ds['parameterization_region'].values[i,1,:] = calc_stats_AAR(output_ds_all['parameterization_AAR_mean'].values[find_id,:]);
    ds['parameterization_region'].values[i,2,:] = calc_stats_AAR(output_ds_all['parameterization_AAR'].values[find_id,:]);
    ds['parameterization_region'].values[i,3,:] = calc_stats_AAR(output_ds_all['parameterization_a'].values[find_id,:]);
    ds['parameterization_region'].values[i,4,:] = calc_stats_AAR(output_ds_all['parameterization_dA'].values[find_id,:]);
    ds['parameterization_region'].values[i,5,:] = calc_stats_AAR(output_ds_all['parameterization_dV'].values[find_id,:]);
    ds['parameterization_region'].values[i,6,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_steady'].values[find_id,:]);
    ds['parameterization_region'].values[i,7,:] = calc_stats_AAR(output_ds_all['parameterization_THAR'].values[find_id,:]);
    ds['parameterization_region'].values[i,8,:] = calc_stats_AAR(output_ds_all['parameterization_dV_bwl'].values[find_id,:]);
    ds['parameterization_region'].values[i,9,:] = ds['parameterization_region'].values[i,5,:] - ds['parameterization_region'].values[i,8,:];
    ds['parameterization_region'].values[i,10,:] = abs(ds['parameterization_region'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);

#%% ===== is ice cap =====
for i in range(0,2):
    find_id = np.where(output_ds_all['is_icecap'].values==i)[0];
    n = len(find_id);
    if n!=0:
        
        # glac_results
        ds['glac_icecap'].values[i,0,:] = calc_stats_AAR(output_ds_all['Area'].values[find_id]);
        ds['glac_icecap'].values[i,1,:] = calc_stats_AAR(output_ds_all['area_2020'].values[find_id,:]);
        ds['glac_icecap'].values[i,2,:] = calc_stats_AAR(output_ds_all['volume_2020'].values[find_id,:]);
        
        # intercept_results
        ds['intercept_icecap'].values[i,0,:] = calc_stats_AAR(output_ds_all['intercept_ELA_mean'].values[find_id,:]);
        ds['intercept_icecap'].values[i,1,:] = calc_stats_AAR(output_ds_all['intercept_AAR_mean'].values[find_id,:]);
        ds['intercept_icecap'].values[i,2,:] = calc_stats_AAR(output_ds_all['intercept_AAR'].values[find_id,:]);
        ds['intercept_icecap'].values[i,3,:] = calc_stats_AAR(output_ds_all['intercept_a'].values[find_id,:]);
        ds['intercept_icecap'].values[i,4,:] = calc_stats_AAR(output_ds_all['intercept_dA'].values[find_id,:]);
        ds['intercept_icecap'].values[i,5,:] = calc_stats_AAR(output_ds_all['intercept_dV'].values[find_id,:]);
        ds['intercept_icecap'].values[i,6,:] = calc_stats_AAR(output_ds_all['intercept_ELA_steady'].values[find_id,:]);
        ds['intercept_icecap'].values[i,7,:] = calc_stats_AAR(output_ds_all['intercept_THAR'].values[find_id,:]);
        ds['intercept_icecap'].values[i,8,:] = calc_stats_AAR(output_ds_all['intercept_dV_bwl'].values[find_id,:]);
        ds['intercept_icecap'].values[i,9,:] = ds['intercept_icecap'].values[i,5,:] - ds['intercept_icecap'].values[i,8,:];
        ds['intercept_icecap'].values[i,10,:] = abs(ds['intercept_icecap'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);
    
        # equil_results
        ds['equil_icecap'].values[i,0,:] = calc_stats_AAR(output_ds_all['equil_ELA_mean'].values[find_id,:]);
        ds['equil_icecap'].values[i,1,:] = calc_stats_AAR(output_ds_all['equil_AAR_mean'].values[find_id,:]);
        ds['equil_icecap'].values[i,2,:] = calc_stats_AAR(output_ds_all['equil_AAR'].values[find_id,:]);
        ds['equil_icecap'].values[i,3,:] = calc_stats_AAR(output_ds_all['equil_a'].values[find_id,:]);
        ds['equil_icecap'].values[i,4,:] = calc_stats_AAR(output_ds_all['equil_dA'].values[find_id,:]);
        ds['equil_icecap'].values[i,5,:] = calc_stats_AAR(output_ds_all['equil_dV'].values[find_id,:]);
        ds['equil_icecap'].values[i,6,:] = calc_stats_AAR(output_ds_all['equil_ELA_steady'].values[find_id,:]);
        ds['equil_icecap'].values[i,7,:] = calc_stats_AAR(output_ds_all['equil_THAR'].values[find_id,:]);
        ds['equil_icecap'].values[i,8,:] = calc_stats_AAR(output_ds_all['equil_dV_bwl'].values[find_id,:]);
        ds['equil_icecap'].values[i,9,:] = ds['equil_icecap'].values[i,5,:] - ds['equil_icecap'].values[i,8,:];
        ds['equil_icecap'].values[i,10,:] = abs(ds['equil_icecap'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);
        
        # parameterization_results
        ds['parameterization_icecap'].values[i,0,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_mean'].values[find_id,:]);
        ds['parameterization_icecap'].values[i,1,:] = calc_stats_AAR(output_ds_all['parameterization_AAR_mean'].values[find_id,:]);
        ds['parameterization_icecap'].values[i,2,:] = calc_stats_AAR(output_ds_all['parameterization_AAR'].values[find_id,:]);
        ds['parameterization_icecap'].values[i,3,:] = calc_stats_AAR(output_ds_all['parameterization_a'].values[find_id,:]);
        ds['parameterization_icecap'].values[i,4,:] = calc_stats_AAR(output_ds_all['parameterization_dA'].values[find_id,:]);
        ds['parameterization_icecap'].values[i,5,:] = calc_stats_AAR(output_ds_all['parameterization_dV'].values[find_id,:]);
        ds['parameterization_icecap'].values[i,6,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_steady'].values[find_id,:]);
        ds['parameterization_icecap'].values[i,7,:] = calc_stats_AAR(output_ds_all['parameterization_THAR'].values[find_id,:]);
        ds['parameterization_icecap'].values[i,8,:] = calc_stats_AAR(output_ds_all['parameterization_dV_bwl'].values[find_id,:]);
        ds['parameterization_icecap'].values[i,9,:] = ds['parameterization_icecap'].values[i,5,:] - ds['parameterization_icecap'].values[i,8,:];
        ds['parameterization_icecap'].values[i,10,:] = abs(ds['parameterization_icecap'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);

#%% ===== is debris covered glacier =====
for i in range(0,2):
    find_id = np.where(output_ds_all['is_debris'].values==i)[0];
    n = len(find_id);
    if n!=0:
        # glac_results
        ds['glac_debris'].values[i,0,:] = calc_stats_AAR(output_ds_all['Area'].values[find_id]);
        ds['glac_debris'].values[i,1,:] = calc_stats_AAR(output_ds_all['area_2020'].values[find_id,:]);
        ds['glac_debris'].values[i,2,:] = calc_stats_AAR(output_ds_all['volume_2020'].values[find_id,:]);
        
        # intercept_results
        ds['intercept_debris'].values[i,0,:] = calc_stats_AAR(output_ds_all['intercept_ELA_mean'].values[find_id,:]);
        ds['intercept_debris'].values[i,1,:] = calc_stats_AAR(output_ds_all['intercept_AAR_mean'].values[find_id,:]);
        ds['intercept_debris'].values[i,2,:] = calc_stats_AAR(output_ds_all['intercept_AAR'].values[find_id,:]);
        ds['intercept_debris'].values[i,3,:] = calc_stats_AAR(output_ds_all['intercept_a'].values[find_id,:]);
        ds['intercept_debris'].values[i,4,:] = calc_stats_AAR(output_ds_all['intercept_dA'].values[find_id,:]);
        ds['intercept_debris'].values[i,5,:] = calc_stats_AAR(output_ds_all['intercept_dV'].values[find_id,:]);
        ds['intercept_debris'].values[i,6,:] = calc_stats_AAR(output_ds_all['intercept_ELA_steady'].values[find_id,:]);
        ds['intercept_debris'].values[i,7,:] = calc_stats_AAR(output_ds_all['intercept_THAR'].values[find_id,:]);
        ds['intercept_debris'].values[i,8,:] = calc_stats_AAR(output_ds_all['intercept_dV_bwl'].values[find_id,:]);
        ds['intercept_debris'].values[i,9,:] = ds['intercept_debris'].values[i,5,:] - ds['intercept_debris'].values[i,8,:];
        ds['intercept_debris'].values[i,10,:] = abs(ds['intercept_debris'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);
    
        # equil_results
        ds['equil_debris'].values[i,0,:] = calc_stats_AAR(output_ds_all['equil_ELA_mean'].values[find_id,:]);
        ds['equil_debris'].values[i,1,:] = calc_stats_AAR(output_ds_all['equil_AAR_mean'].values[find_id,:]);
        ds['equil_debris'].values[i,2,:] = calc_stats_AAR(output_ds_all['equil_AAR'].values[find_id,:]);
        ds['equil_debris'].values[i,3,:] = calc_stats_AAR(output_ds_all['equil_a'].values[find_id,:]);
        ds['equil_debris'].values[i,4,:] = calc_stats_AAR(output_ds_all['equil_dA'].values[find_id,:]);
        ds['equil_debris'].values[i,5,:] = calc_stats_AAR(output_ds_all['equil_dV'].values[find_id,:]);
        ds['equil_debris'].values[i,6,:] = calc_stats_AAR(output_ds_all['equil_ELA_steady'].values[find_id,:]);
        ds['equil_debris'].values[i,7,:] = calc_stats_AAR(output_ds_all['equil_THAR'].values[find_id,:]);
        ds['equil_debris'].values[i,8,:] = calc_stats_AAR(output_ds_all['equil_dV_bwl'].values[find_id,:]);
        ds['equil_debris'].values[i,9,:] = ds['equil_debris'].values[i,5,:] - ds['equil_debris'].values[i,8,:];
        ds['equil_debris'].values[i,10,:] = abs(ds['equil_debris'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);
        
        # parameterization_results
        ds['parameterization_debris'].values[i,0,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_mean'].values[find_id,:]);
        ds['parameterization_debris'].values[i,1,:] = calc_stats_AAR(output_ds_all['parameterization_AAR_mean'].values[find_id,:]);
        ds['parameterization_debris'].values[i,2,:] = calc_stats_AAR(output_ds_all['parameterization_AAR'].values[find_id,:]);
        ds['parameterization_debris'].values[i,3,:] = calc_stats_AAR(output_ds_all['parameterization_a'].values[find_id,:]);
        ds['parameterization_debris'].values[i,4,:] = calc_stats_AAR(output_ds_all['parameterization_dA'].values[find_id,:]);
        ds['parameterization_debris'].values[i,5,:] = calc_stats_AAR(output_ds_all['parameterization_dV'].values[find_id,:]);
        ds['parameterization_debris'].values[i,6,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_steady'].values[find_id,:]);
        ds['parameterization_debris'].values[i,7,:] = calc_stats_AAR(output_ds_all['parameterization_THAR'].values[find_id,:]);
        ds['parameterization_debris'].values[i,8,:] = calc_stats_AAR(output_ds_all['parameterization_dV_bwl'].values[find_id,:]);
        ds['parameterization_debris'].values[i,9,:] = ds['parameterization_debris'].values[i,5,:] - ds['parameterization_debris'].values[i,8,:];
        ds['parameterization_debris'].values[i,10,:] = abs(ds['parameterization_debris'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);

#%% ===== is tidewater glacier =====
for i in range(0,2):
    find_id = np.where(output_ds_all['is_tidewater'].values==i)[0];
    n = len(find_id);
    if n!=0:
        
        # glac_results
        ds['glac_tidewater'].values[i,0,:] = calc_stats_AAR(output_ds_all['Area'].values[find_id]);
        ds['glac_tidewater'].values[i,1,:] = calc_stats_AAR(output_ds_all['area_2020'].values[find_id,:]);
        ds['glac_tidewater'].values[i,2,:] = calc_stats_AAR(output_ds_all['volume_2020'].values[find_id,:]);
        
        # intercept_results
        ds['intercept_tidewater'].values[i,0,:] = calc_stats_AAR(output_ds_all['intercept_ELA_mean'].values[find_id,:]);
        ds['intercept_tidewater'].values[i,1,:] = calc_stats_AAR(output_ds_all['intercept_AAR_mean'].values[find_id,:]);
        ds['intercept_tidewater'].values[i,2,:] = calc_stats_AAR(output_ds_all['intercept_AAR'].values[find_id,:]);
        ds['intercept_tidewater'].values[i,3,:] = calc_stats_AAR(output_ds_all['intercept_a'].values[find_id,:]);
        ds['intercept_tidewater'].values[i,4,:] = calc_stats_AAR(output_ds_all['intercept_dA'].values[find_id,:]);
        ds['intercept_tidewater'].values[i,5,:] = calc_stats_AAR(output_ds_all['intercept_dV'].values[find_id,:]);
        ds['intercept_tidewater'].values[i,6,:] = calc_stats_AAR(output_ds_all['intercept_ELA_steady'].values[find_id,:]);
        ds['intercept_tidewater'].values[i,7,:] = calc_stats_AAR(output_ds_all['intercept_THAR'].values[find_id,:]);
        ds['intercept_tidewater'].values[i,8,:] = calc_stats_AAR(output_ds_all['intercept_dV_bwl'].values[find_id,:]);
        ds['intercept_tidewater'].values[i,9,:] = ds['intercept_tidewater'].values[i,5,:] - ds['intercept_tidewater'].values[i,8,:];
        ds['intercept_tidewater'].values[i,10,:] = abs(ds['intercept_tidewater'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);
    
        # equil_results
        ds['equil_tidewater'].values[i,0,:] = calc_stats_AAR(output_ds_all['equil_ELA_mean'].values[find_id,:]);
        ds['equil_tidewater'].values[i,1,:] = calc_stats_AAR(output_ds_all['equil_AAR_mean'].values[find_id,:]);
        ds['equil_tidewater'].values[i,2,:] = calc_stats_AAR(output_ds_all['equil_AAR'].values[find_id,:]);
        ds['equil_tidewater'].values[i,3,:] = calc_stats_AAR(output_ds_all['equil_a'].values[find_id,:]);
        ds['equil_tidewater'].values[i,4,:] = calc_stats_AAR(output_ds_all['equil_dA'].values[find_id,:]);
        ds['equil_tidewater'].values[i,5,:] = calc_stats_AAR(output_ds_all['equil_dV'].values[find_id,:]);
        ds['equil_tidewater'].values[i,6,:] = calc_stats_AAR(output_ds_all['equil_ELA_steady'].values[find_id,:]);
        ds['equil_tidewater'].values[i,7,:] = calc_stats_AAR(output_ds_all['equil_THAR'].values[find_id,:]);
        ds['equil_tidewater'].values[i,8,:] = calc_stats_AAR(output_ds_all['equil_dV_bwl'].values[find_id,:]);
        ds['equil_tidewater'].values[i,9,:] = ds['equil_tidewater'].values[i,5,:] - ds['equil_tidewater'].values[i,8,:];
        ds['equil_tidewater'].values[i,10,:] = abs(ds['equil_tidewater'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);
        
        # parameterization_results
        ds['parameterization_tidewater'].values[i,0,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_mean'].values[find_id,:]);
        ds['parameterization_tidewater'].values[i,1,:] = calc_stats_AAR(output_ds_all['parameterization_AAR_mean'].values[find_id,:]);
        ds['parameterization_tidewater'].values[i,2,:] = calc_stats_AAR(output_ds_all['parameterization_AAR'].values[find_id,:]);
        ds['parameterization_tidewater'].values[i,3,:] = calc_stats_AAR(output_ds_all['parameterization_a'].values[find_id,:]);
        ds['parameterization_tidewater'].values[i,4,:] = calc_stats_AAR(output_ds_all['parameterization_dA'].values[find_id,:]);
        ds['parameterization_tidewater'].values[i,5,:] = calc_stats_AAR(output_ds_all['parameterization_dV'].values[find_id,:]);
        ds['parameterization_tidewater'].values[i,6,:] = calc_stats_AAR(output_ds_all['parameterization_ELA_steady'].values[find_id,:]);
        ds['parameterization_tidewater'].values[i,7,:] = calc_stats_AAR(output_ds_all['parameterization_THAR'].values[find_id,:]);
        ds['parameterization_tidewater'].values[i,8,:] = calc_stats_AAR(output_ds_all['parameterization_dV_bwl'].values[find_id,:]);
        ds['parameterization_tidewater'].values[i,9,:] = ds['parameterization_tidewater'].values[i,5,:] - ds['parameterization_tidewater'].values[i,8,:];
        ds['parameterization_tidewater'].values[i,10,:] = abs(ds['parameterization_tidewater'].values[i,9,:] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000);

#%% To file
path = '/Users/wyan0065/Desktop/PyGEM/calving/Output/' + 'results_by_region.nc';
enc_var = {'dtype': 'float32'}
encoding = {v: enc_var for v in ds.data_vars}
ds.to_netcdf(path, encoding=encoding);