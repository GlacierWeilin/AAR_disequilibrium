#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:43:09 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
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
        if (np.isnan(data)).all():
            stats = np.zeros(12) * np.nan
        else:
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
            data = output[:,1]; data = data*data;
            stats = np.append(stats, np.sqrt(np.nansum(data))/stats[8])
            stats = np.append(stats, np.sqrt(np.nansum(data)))

    elif output.ndim == 1:
        data = output[:];
        stats = None
        if (np.isnan(data)).all():
            stats = np.zeros(12) * np.nan
        else:
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

def regional_avg(n=None, data=None, analysis='', path=''):
    """Calculate the regional values with the spatial resolution of 2.0°*2.0°

    Parameters
    ----------
    n : float
        Spatial resolution. The default is 0.25 (ERA5).
    data: array
        ERA5_MCMC_ba1_2014_2023_corrected.nc
    analysis: str
        'all', 'icecap', 'debris', 'tidewater', '-icecap', '-debris', '-tidewater'
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location (file
        extension matters)
    """
    
    latbnd = np.arange(-90-n/2, 90+n, n)
    lat    = int(180/n+1)
    lon    = int(360/n)
    if n==0.25:
        lonbnd = np.arange(0-n/2, 360+n/2, n)
    else:
        lonbnd = np.arange(-180, 180+n/2, n)
    
    # prepare output
    ds = xr.Dataset();
    
    # Attributes
    ds.attrs['Source'] = 'PyGEMv0.2.5 developed by David Rounce (drounce@alaska.edu)'
    ds.attrs['Further developed by'] = 'Weilin Yang (weilinyang.yang@monash.edu)'
    ds.attrs['Code reviewed by'] = 'Wenchao Chu (peterchuwenchao@foxmail.com)'
    
    # Coordinates
    ds.coords['dim'] = ('dim', np.arange(12))
    ds['dim'].attrs['description'] = '0-mean, 1-std, 2-2.5%, 3-25%, 4-median, 5-75%, 6-97.5%, 7-mad, \
        8-n, 9-sum(std), 10-mean_std(std), 11-sum_std(std)'
    
    ds.coords['latitude'] = np.arange(-90, 90+n, n)
    ds['latitude'].attrs['long_name'] = 'latitude'
    ds['latitude'].attrs['units'] = 'degrees_north'
    if n==0.25:
        ds.coords['longitude'] = np.arange(0, 360, n)
        ds['longitude'].attrs['long_name'] = 'longitude'
        ds['longitude'].attrs['units'] = 'degrees_east'
    else:
        ds.coords['longitude'] = np.arange(-180+n/2, 180, n)
        ds['longitude'].attrs['long_name'] = 'longitude'
        ds['longitude'].attrs['units'] = 'degrees_east'
    
    # glac_results
    ds['area'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['area'].attrs['description'] = 'RGI area (m2)'
    ds['area_2020'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['area_2020'].attrs['description'] = 'simulated glacier area in 2020 (m2)'
    ds['volume_2020'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['volume_2020'].attrs['description'] = 'simulated glacier volume in 2020 (m3)'
    
    # intercept_results
    # 0-intercept_ELA_mean, 1-intercept_AAR_mean, 2-intercept_AAR, \
    # 3-intercept_a, 4-intercept_dA, 5-intercept_dV, 6-intercept_ELA_steady, 7-intercept_THAR, \
    # 8-intercept_dV_bwl, 9-intercept_dV_eff, 10-SLR
    ds['intercept_ELA_mean'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['intercept_ELA_mean'].attrs['description'] = 'grid ELA_Mean (linear regression)'
    ds['intercept_AAR_mean'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['intercept_AAR_mean'].attrs['description'] = 'grid AAR_mean (linear regression)'
    ds['intercept_AAR'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['intercept_AAR'].attrs['description'] = 'grid AAR (linear regression)'
    ds['intercept_a'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['intercept_a'].attrs['description'] = 'grid a (linear regression)'
    ds['intercept_dA'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['intercept_dA'].attrs['description'] = 'grid dA (linear regression)'
    ds['intercept_dV'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['intercept_dV'].attrs['description'] = 'grid dV (linear regression)'
    ds['intercept_ELA_steady'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['intercept_ELA_steady'].attrs['description'] = 'grid ELA_steady (linear regression)'
    ds['intercept_THAR'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['intercept_THAR'].attrs['description'] = 'grid THAR (linear regression)'
    ds['intercept_dV_bwl'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['intercept_dV_bwl'].attrs['description'] = 'grid dV_bwl (linear regression)'
    ds['intercept_dV_eff'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['intercept_dV_eff'].attrs['description'] = 'grid dV_eff (linear regression)'
    
    # equil_results
    ds['equil_ELA_mean'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['equil_ELA_mean'].attrs['description'] = 'grid ELA_Mean (equilibrium experiment)'
    ds['equil_AAR_mean'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['equil_AAR_mean'].attrs['description'] = 'grid AAR_mean (equilibrium experiment)'
    ds['equil_AAR'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['equil_AAR'].attrs['description'] = 'grid AAR (equilibrium experiment)'
    ds['equil_a'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['equil_a'].attrs['description'] = 'grid a (equilibrium experiment)'
    ds['equil_dA'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['equil_dA'].attrs['description'] = 'grid dA (equilibrium experiment)'
    ds['equil_dV'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['equil_dV'].attrs['description'] = 'grid dV (equilibrium experiment)'
    ds['equil_ELA_steady'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['equil_ELA_steady'].attrs['description'] = 'grid ELA_steady (equilibrium experiment)'
    ds['equil_THAR'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['equil_THAR'].attrs['description'] = 'grid THAR (equilibrium experiment)'
    ds['equil_dV_bwl'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['equil_dV_bwl'].attrs['description'] = 'grid dV_bwl (equilibrium experiment)'
    ds['equil_dV_eff'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['equil_dV_eff'].attrs['description'] = 'grid dV_eff (equilibrium experiment)'
    
    # parameterization_results
    ds['parameterization_ELA_mean'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['parameterization_ELA_mean'].attrs['description'] = 'grid ELA_Mean'
    ds['parameterization_AAR_mean'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['parameterization_AAR_mean'].attrs['description'] = 'grid AAR_mean'
    ds['parameterization_AAR'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['parameterization_AAR'].attrs['description'] = 'grid AAR'
    ds['parameterization_a'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['parameterization_a'].attrs['description'] = 'grid a'
    ds['parameterization_dA'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['parameterization_dA'].attrs['description'] = 'grid dA'
    ds['parameterization_dV'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['parameterization_dV'].attrs['description'] = 'grid dV'
    ds['parameterization_ELA_steady'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['parameterization_ELA_steady'].attrs['description'] = 'grid ELA_steady'
    ds['parameterization_THAR'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['parameterization_THAR'].attrs['description'] = 'grid THAR'
    ds['parameterization_dV_bwl'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['parameterization_dV_bwl'].attrs['description'] = 'grid dV_bwl'
    ds['parameterization_dV_eff'] = (('latitude', 'longitude','dim'), np.zeros([lat,lon,12])*np.nan)
    ds['parameterization_dV_eff'].attrs['description'] = 'grid dV_eff'
    
    for i in range(0, lat):
        bottom = latbnd[i]; up = latbnd[i+1]
        for j in range(0, lon):
            left = lonbnd[j]; right = lonbnd[j+1];
            
            if analysis == 'all':
                find_id = np.where((data['CenLat']>=bottom) & (data['CenLat']<up) & (data['CenLon']>=left) & (data['CenLon']<right))[0];
            elif analysis == 'icecap':
                find_id = np.where((data['CenLat']>=bottom) & (data['CenLat']<up) & (data['CenLon']>=left) & (data['CenLon']<right) & \
                                   (data['is_icecap']==1))[0];
            elif analysis == '-icecap':
                find_id = np.where((data['CenLat']>=bottom) & (data['CenLat']<up) & (data['CenLon']>=left) & (data['CenLon']<right) & \
                                   (data['is_icecap']==0))[0];
            elif analysis == 'debris':
                find_id = np.where((data['CenLat']>=bottom) & (data['CenLat']<up) & (data['CenLon']>=left) & (data['CenLon']<right) & \
                                   (data['is_debris']==1))[0];
            elif analysis == '-debris':
                find_id = np.where((data['CenLat']>=bottom) & (data['CenLat']<up) & (data['CenLon']>=left) & (data['CenLon']<right) & \
                                   (data['is_debris']==0))[0];
            elif analysis == 'tidewater':
                find_id = np.where((data['CenLat']>=bottom) & (data['CenLat']<up) & (data['CenLon']>=left) & (data['CenLon']<right) & \
                                   (data['is_tidewater']==1))[0];
            elif analysis == '-tidewater':
                find_id = np.where((data['CenLat']>=bottom) & (data['CenLat']<up) & (data['CenLon']>=left) & (data['CenLon']<right) & \
                                   (data['is_tidewater']==0))[0];
            
            if sum(find_id)!=0:
                
                # glac_results
                ds['area'].values[i,j,:]        = calc_stats_AAR(data['Area'].values[find_id])
                ds['area_2020'].values[i,j,:]   = calc_stats_AAR(data['area_2020'].values[find_id,:])
                ds['volume_2020'].values[i,j,:] = calc_stats_AAR(data['volume_2020'].values[find_id,:])
                
                # intercept_results
                ds['intercept_ELA_mean'].values[i,j,:]   = calc_stats_AAR(data['intercept_ELA_mean'].values[find_id,:])
                ds['intercept_AAR_mean'].values[i,j,:]   = calc_stats_AAR(data['intercept_AAR_mean'].values[find_id,:])
                ds['intercept_AAR'].values[i,j,:]        = calc_stats_AAR(data['intercept_AAR'].values[find_id,:])
                ds['intercept_a'].values[i,j,:]          = calc_stats_AAR(data['intercept_a'].values[find_id,:])
                ds['intercept_dA'].values[i,j,:]         = calc_stats_AAR(data['intercept_dA'].values[find_id,:])
                ds['intercept_dV'].values[i,j,:]         = calc_stats_AAR(data['intercept_dV'].values[find_id,:])
                ds['intercept_ELA_steady'].values[i,j,:] = calc_stats_AAR(data['intercept_ELA_steady'].values[find_id,:])
                ds['intercept_THAR'].values[i,j,:]       = calc_stats_AAR(data['intercept_THAR'].values[find_id,:])
                ds['intercept_dV_bwl'].values[i,j,:]     = calc_stats_AAR(data['intercept_dV_bwl'].values[find_id,:])
                ds['intercept_dV_eff'].values[i,j,:]     = ds['intercept_dV'].values[i,j,:] - ds['intercept_dV_bwl'].values[i,j,:];
                
                # random_results
                ds['equil_ELA_mean'].values[i,j,:]   = calc_stats_AAR(data['equil_ELA_mean'].values[find_id,:])
                ds['equil_AAR_mean'].values[i,j,:]   = calc_stats_AAR(data['equil_AAR_mean'].values[find_id,:])
                ds['equil_AAR'].values[i,j,:]        = calc_stats_AAR(data['equil_AAR'].values[find_id,:])
                ds['equil_a'].values[i,j,:]          = calc_stats_AAR(data['equil_a'].values[find_id,:])
                ds['equil_dA'].values[i,j,:]         = calc_stats_AAR(data['equil_dA'].values[find_id,:])
                ds['equil_dV'].values[i,j,:]         = calc_stats_AAR(data['equil_dV'].values[find_id,:])
                ds['equil_ELA_steady'].values[i,j,:] = calc_stats_AAR(data['equil_ELA_steady'].values[find_id,:])
                ds['equil_THAR'].values[i,j,:]       = calc_stats_AAR(data['equil_THAR'].values[find_id,:])
                ds['equil_dV_bwl'].values[i,j,:]     = calc_stats_AAR(data['equil_dV_bwl'].values[find_id,:])
                ds['equil_dV_eff'].values[i,j,:]     = ds['equil_dV'].values[i,j,:] - ds['equil_dV_bwl'].values[i,j,:];
            
                # parameterization_results
                ds['parameterization_ELA_mean'].values[i,j,:]   = calc_stats_AAR(data['parameterization_ELA_mean'].values[find_id,:])
                ds['parameterization_AAR_mean'].values[i,j,:]   = calc_stats_AAR(data['parameterization_AAR_mean'].values[find_id,:])
                ds['parameterization_AAR'].values[i,j,:]        = calc_stats_AAR(data['parameterization_AAR'].values[find_id,:])
                ds['parameterization_a'].values[i,j,:]          = calc_stats_AAR(data['parameterization_a'].values[find_id,:])
                ds['parameterization_dA'].values[i,j,:]         = calc_stats_AAR(data['parameterization_dA'].values[find_id,:])
                ds['parameterization_dV'].values[i,j,:]         = calc_stats_AAR(data['parameterization_dV'].values[find_id,:])
                ds['parameterization_ELA_steady'].values[i,j,:] = calc_stats_AAR(data['parameterization_ELA_steady'].values[find_id,:])
                ds['parameterization_THAR'].values[i,j,:]       = calc_stats_AAR(data['parameterization_THAR'].values[find_id,:])
                ds['parameterization_dV_bwl'].values[i,j,:]     = calc_stats_AAR(data['parameterization_dV_bwl'].values[find_id,:])
                ds['parameterization_dV_eff'].values[i,j,:]     = ds['parameterization_dV'].values[i,j,:] - ds['parameterization_dV_bwl'].values[i,j,:];
                
    # To file
    path = path + 'results_' + str(n) + '_' + analysis + '.nc';
    enc_var = {'dtype': 'float32'}
    encoding = {v: enc_var for v in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding);                

#%% ===== Main codes =====

path = '/Users/wyan0065/Desktop/PyGEM/calving/Output/';
fn = 'ERA5_MCMC_ba1_2014_2023_corrected.nc';
data = xr.open_dataset(path+fn);

n=2;
if n==0.25:
    find_id = np.where(data['CenLon'].values<-n/2);
    data['CenLon'].values[find_id] = data['CenLon'].values[find_id] + 360;
else:
    pass

analysis = 'all';
regional_avg(n=n, data=data, analysis=analysis, path=path)
#analysis = 'icecap';
#regional_avg(n=n, data=data, analysis=analysis, path=path)
#analysis = '-icecap';
#regional_avg(n=n, data=data, analysis=analysis, path=path)
#analysis = 'debris';
#regional_avg(n=n, data=data, analysis=analysis, path=path)
#analysis = '-debris';
#regional_avg(n=n, data=data, analysis=analysis, path=path)
#analysis = 'tidewater';
#regional_avg(n=n, data=data, analysis=analysis, path=path)
#analysis = '-tidewater';
#regional_avg(n=n, data=data, analysis=analysis, path=path)
