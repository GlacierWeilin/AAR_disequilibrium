#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:13:05 2025

@author: wyan0065
"""

import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import median_abs_deviation

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

file_fp = '/Users/wyan0065/Desktop/PyGEM/calving/Output/';
data = xr.open_dataset(file_fp+'ERA5_MCMC_ba1_2014_2023_corrected.nc', decode_times=False);
frontal_obs = pd.read_csv(file_fp + '/tidewater_has_obs.csv')
obs_19 = frontal_obs[frontal_obs['RGIId'].str.startswith('RGI60-19')]
obs_other = frontal_obs[~frontal_obs['RGIId'].str.startswith('RGI60-19')]
# Global
re_a     = calc_stats_AAR(data['parameterization_dV'].values)
re_k     = calc_stats_AAR(data['intercept_dV'].values)
volume   = calc_stats_AAR(data['volume_2020'].values)

re_global = pd.Series(dtype='float64')
re_global['number']       = re_a[8]
re_global['area']         = 100.0
re_global['volume']       = 100.0
re_global['median_a']     = re_a[9]/volume[9] * 100
re_global['median_a_mad'] = re_global['median_a'] * np.sqrt((re_a[11]/re_a[9])**2+
                                                            (volume[11]/volume[9])**2)
re_global['median_k']     = re_k[9]/volume[9] * 100
re_global['median_k_mad'] = re_global['median_k'] * np.sqrt((re_k[11]/re_k[9])**2+
                                                            (volume[11]/volume[9])**2)

#%% land-terminating
find = np.where(data['is_tidewater'].values != 1)[0]
re_a_find     = calc_stats_AAR(data['parameterization_dV'].values[find])
re_k_find     = calc_stats_AAR(data['intercept_dV'].values[find])
volume_find   = calc_stats_AAR(data['volume_2020'].values[find])

re_land = pd.Series(dtype='float64')
re_land['number']       = re_a_find[8]
re_land['area']         = np.nansum(data['area_2020'].values[find,0]) / np.nansum(data['area_2020'].values[:,0]) * 100
re_land['volume']       = np.nansum(data['volume_2020'].values[find,0]) / np.nansum(data['volume_2020'].values[:,0]) * 100
re_land['median_a']     = re_a_find[9]/volume_find[9] * 100
re_land['median_a_mad'] = re_land['median_a'] * np.sqrt((re_a_find[11]/re_a_find[9])**2+
                                                        (volume_find[11]/volume_find[9])**2)
re_land['median_k']     = re_k_find[9]/volume_find[9] * 100
re_land['median_k_mad'] = re_land['median_k'] * np.sqrt((re_k_find[11]/re_k_find[9])**2+
                                                        (volume_find[11]/volume_find[9])**2)

#%% marine-terminating
find = np.where(data['is_tidewater'].values == 1)[0]
re_a_find     = calc_stats_AAR(data['parameterization_dV'].values[find])
re_k_find     = calc_stats_AAR(data['intercept_dV'].values[find])
volume_find   = calc_stats_AAR(data['volume_2020'].values[find])

re_marine = pd.Series(dtype='float64')
re_marine['number']       = re_a_find[8]
re_marine['area']         = np.nansum(data['area_2020'].values[find,0]) / np.nansum(data['area_2020'].values[:,0]) * 100
re_marine['volume']       = np.nansum(data['volume_2020'].values[find,0]) / np.nansum(data['volume_2020'].values[:,0]) * 100
re_marine['median_a']     = re_a_find[9]/volume_find[9] * 100
re_marine['median_a_mad'] = re_marine['median_a'] * np.sqrt((re_a_find[11]/re_a_find[9])**2+
                                                        (volume_find[11]/volume_find[9])**2)
re_marine['median_k']     = re_k_find[9]/volume_find[9] * 100
re_marine['median_k_mad'] = re_marine['median_k'] * np.sqrt((re_k_find[11]/re_k_find[9])**2+
                                                            (volume_find[11]/volume_find[9])**2)

#%% land-terminating
find = np.where((data['is_tidewater'].values != 1) & (data['O1Region'].values != 19))[0]
re_a_find     = calc_stats_AAR(data['parameterization_dV'].values[find])
re_k_find     = calc_stats_AAR(data['intercept_dV'].values[find])
volume_find   = calc_stats_AAR(data['volume_2020'].values[find])

re_land = pd.Series(dtype='float64')
re_land['number']       = re_a_find[8]
re_land['area']         = np.nansum(data['area_2020'].values[find,0]) / np.nansum(data['area_2020'].values[:,0]) * 100
re_land['volume']       = np.nansum(data['volume_2020'].values[find,0]) / np.nansum(data['volume_2020'].values[:,0]) * 100
re_land['median_a']     = re_a_find[9]/volume_find[9] * 100
re_land['median_a_mad'] = re_land['median_a'] * np.sqrt((re_a_find[11]/re_a_find[9])**2+
                                                        (volume_find[11]/volume_find[9])**2)
re_land['median_k']     = re_k_find[9]/volume_find[9] * 100
re_land['median_k_mad'] = re_land['median_k'] * np.sqrt((re_k_find[11]/re_k_find[9])**2+
                                                        (volume_find[11]/volume_find[9])**2)

#%% marine-terminating
find = np.where((data['is_tidewater'].values == 1) & (data['O1Region'].values != 19))[0]
re_a_find     = calc_stats_AAR(data['parameterization_dV'].values[find])
re_k_find     = calc_stats_AAR(data['intercept_dV'].values[find])
volume_find   = calc_stats_AAR(data['volume_2020'].values[find])

re_marine = pd.Series(dtype='float64')
re_marine['number']       = re_a_find[8]
re_marine['area']         = np.nansum(data['area_2020'].values[find,0]) / np.nansum(data['area_2020'].values[:,0]) * 100
re_marine['volume']       = np.nansum(data['volume_2020'].values[find,0]) / np.nansum(data['volume_2020'].values[:,0]) * 100
re_marine['median_a']     = re_a_find[9]/volume_find[9] * 100
re_marine['median_a_mad'] = re_marine['median_a'] * np.sqrt((re_a_find[11]/re_a_find[9])**2+
                                                        (volume_find[11]/volume_find[9])**2)
re_marine['median_k']     = re_k_find[9]/volume_find[9] * 100
re_marine['median_k_mad'] = re_marine['median_k'] * np.sqrt((re_k_find[11]/re_k_find[9])**2+
                                                            (volume_find[11]/volume_find[9])**2)

#%% land-terminating
find = np.where((data['is_tidewater'].values != 1) & (data['O1Region'].values == 19))[0]
re_a_find     = calc_stats_AAR(data['parameterization_dV'].values[find])
re_k_find     = calc_stats_AAR(data['intercept_dV'].values[find])
volume_find   = calc_stats_AAR(data['volume_2020'].values[find])

re_land = pd.Series(dtype='float64')
re_land['number']       = re_a_find[8]
re_land['area']         = np.nansum(data['area_2020'].values[find,0]) / np.nansum(data['area_2020'].values[:,0]) * 100
re_land['volume']       = np.nansum(data['volume_2020'].values[find,0]) / np.nansum(data['volume_2020'].values[:,0]) * 100
re_land['median_a']     = re_a_find[9]/volume_find[9] * 100
re_land['median_a_mad'] = re_land['median_a'] * np.sqrt((re_a_find[11]/re_a_find[9])**2+
                                                        (volume_find[11]/volume_find[9])**2)
re_land['median_k']     = re_k_find[9]/volume_find[9] * 100
re_land['median_k_mad'] = re_land['median_k'] * np.sqrt((re_k_find[11]/re_k_find[9])**2+
                                                        (volume_find[11]/volume_find[9])**2)

#%% marine-terminating
find = np.where((data['is_tidewater'].values == 1) & (data['O1Region'].values == 19))[0]
re_a_find     = calc_stats_AAR(data['parameterization_dV'].values[find])
re_k_find     = calc_stats_AAR(data['intercept_dV'].values[find])
volume_find   = calc_stats_AAR(data['volume_2020'].values[find])

re_marine = pd.Series(dtype='float64')
re_marine['number']       = re_a_find[8]
re_marine['area']         = np.nansum(data['area_2020'].values[find,0]) / np.nansum(data['area_2020'].values[:,0]) * 100
re_marine['volume']       = np.nansum(data['volume_2020'].values[find,0]) / np.nansum(data['volume_2020'].values[:,0]) * 100
re_marine['median_a']     = re_a_find[9]/volume_find[9] * 100
re_marine['median_a_mad'] = re_marine['median_a'] * np.sqrt((re_a_find[11]/re_a_find[9])**2+
                                                        (volume_find[11]/volume_find[9])**2)
re_marine['median_k']     = re_k_find[9]/volume_find[9] * 100
re_marine['median_k_mad'] = re_marine['median_k'] * np.sqrt((re_k_find[11]/re_k_find[9])**2+
                                                            (volume_find[11]/volume_find[9])**2)

#%% marine_terminating 0-18
mask = np.isin(data['RGIId'].values,obs_other['RGIId'].values)
find = np.where(mask)[0]
re_a_find     = calc_stats_AAR(data['parameterization_dV'].values[find])
re_k_find     = calc_stats_AAR(data['intercept_dV'].values[find])
volume_find   = calc_stats_AAR(data['volume_2020'].values[find])

re_marine = pd.Series(dtype='float64')
re_marine['number']       = re_a_find[8]
re_marine['area']         = np.nansum(data['area_2020'].values[find,0]) / np.nansum(data['area_2020'].values[:,0]) * 100
re_marine['volume']       = np.nansum(data['volume_2020'].values[find,0]) / np.nansum(data['volume_2020'].values[:,0]) * 100
re_marine['median_a']     = re_a_find[9]/volume_find[9] * 100
re_marine['median_a_mad'] = re_marine['median_a'] * np.sqrt((re_a_find[11]/re_a_find[9])**2+
                                                        (volume_find[11]/volume_find[9])**2)
re_marine['median_k']     = re_k_find[9]/volume_find[9] * 100
re_marine['median_k_mad'] = re_marine['median_k'] * np.sqrt((re_k_find[11]/re_k_find[9])**2+
                                                        (volume_find[11]/volume_find[9])**2)

#%% marine-terminating 19
mask = np.isin(data['RGIId'].values,obs_19['RGIId'].values)
find = np.where(mask)[0]
re_a_find     = calc_stats_AAR(data['parameterization_dV'].values[find])
re_k_find     = calc_stats_AAR(data['intercept_dV'].values[find])
volume_find   = calc_stats_AAR(data['volume_2020'].values[find])

re_marine = pd.Series(dtype='float64')
re_marine['number']       = re_a_find[8]
re_marine['area']         = np.nansum(data['area_2020'].values[find,0]) / np.nansum(data['area_2020'].values[:,0]) * 100
re_marine['volume']       = np.nansum(data['volume_2020'].values[find,0]) / np.nansum(data['volume_2020'].values[:,0]) * 100
re_marine['median_a']     = re_a_find[9]/volume_find[9] * 100
re_marine['median_a_mad'] = re_marine['median_a'] * np.sqrt((re_a_find[11]/re_a_find[9])**2+
                                                        (volume_find[11]/volume_find[9])**2)
re_marine['median_k']     = re_k_find[9]/volume_find[9] * 100
re_marine['median_k_mad'] = re_marine['median_k'] * np.sqrt((re_k_find[11]/re_k_find[9])**2+
                                                        (volume_find[11]/volume_find[9])**2)



