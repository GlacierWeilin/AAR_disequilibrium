#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:44:15 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import xarray as xr

def latlon_avg(n=None, data=None, path=''):
    """Calculate the regional values with the spatial resolution of 2.0°*2.0°

    Parameters
    ----------
    n : float
        Spatial resolution. The default is 0.25 (ERA5).
    data: array
        ERA5_MCMC_ba1_2014_2023_corrected.nc
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
    ds.coords['dim'] = ('dim', np.arange(2))
    ds['dim'].attrs['description'] = '0-mean, 1-std'

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
    
    ds['intercept_lat_AAR_mean'] = (('latitude', 'dim'), np.zeros([lat, 2])*np.nan)
    ds['intercept_lat_AAR_mean'].attrs['description'] = 'latitude average AAR'
    ds['intercept_lat_AAR'] = (('latitude', 'dim'), np.zeros([lat, 2])*np.nan)
    ds['intercept_lat_AAR'].attrs['description'] = 'latitude average AAR'
    ds['intercept_lat_a'] = (('latitude', 'dim'), np.zeros([lat, 2])*np.nan)
    ds['intercept_lat_a'].attrs['description'] = 'latitude average climate imbalance'
    ds['intercept_lon_AAR_mean'] = (('longitude', 'dim'), np.zeros([lon, 2])*np.nan)
    ds['intercept_lon_AAR_mean'].attrs['description'] = 'longitude average AAR'
    ds['intercept_lon_AAR'] = (('longitude', 'dim'), np.zeros([lon, 2])*np.nan)
    ds['intercept_lon_AAR'].attrs['description'] = 'longitude average AAR'
    ds['intercept_lon_a'] = (('longitude', 'dim'), np.zeros([lon, 2])*np.nan)
    ds['intercept_lon_a'].attrs['description'] = 'longitude average climate imbalance'
    
    ds['equil_lat_AAR_mean'] = (('latitude', 'dim'), np.zeros([lat, 2])*np.nan)
    ds['equil_lat_AAR_mean'].attrs['description'] = 'latitude average AAR'
    ds['equil_lat_AAR'] = (('latitude', 'dim'), np.zeros([lat, 2])*np.nan)
    ds['equil_lat_AAR'].attrs['description'] = 'latitude average AAR'
    ds['equil_lat_a'] = (('latitude', 'dim'), np.zeros([lat, 2])*np.nan)
    ds['equil_lat_a'].attrs['description'] = 'latitude average climate imbalance'
    ds['equil_lon_AAR_mean'] = (('longitude', 'dim'), np.zeros([lon, 2])*np.nan)
    ds['equil_lon_AAR_mean'].attrs['description'] = 'longitude average AAR'
    ds['equil_lon_AAR'] = (('longitude', 'dim'), np.zeros([lon, 2])*np.nan)
    ds['equil_lon_AAR'].attrs['description'] = 'longitude average AAR'
    ds['equil_lon_a'] = (('longitude', 'dim'), np.zeros([lon, 2])*np.nan)
    ds['equil_lon_a'].attrs['description'] = 'longitude average climate imbalance'
    
    ds['parameterization_lat_AAR_mean'] = (('latitude', 'dim'), np.zeros([lat, 2])*np.nan)
    ds['parameterization_lat_AAR_mean'].attrs['description'] = 'latitude average AAR'
    ds['parameterization_lat_AAR'] = (('latitude', 'dim'), np.zeros([lat, 2])*np.nan)
    ds['parameterization_lat_AAR'].attrs['description'] = 'latitude average AAR'
    ds['parameterization_lat_a'] = (('latitude', 'dim'), np.zeros([lat, 2])*np.nan)
    ds['parameterization_lat_a'].attrs['description'] = 'latitude average climate imbalance'
    ds['parameterization_lon_AAR_mean'] = (('longitude', 'dim'), np.zeros([lon, 2])*np.nan)
    ds['parameterization_lon_AAR_mean'].attrs['description'] = 'longitude average AAR'
    ds['parameterization_lon_AAR'] = (('longitude', 'dim'), np.zeros([lon, 2])*np.nan)
    ds['parameterization_lon_AAR'].attrs['description'] = 'longitude average AAR'
    ds['parameterization_lon_a'] = (('longitude', 'dim'), np.zeros([lon, 2])*np.nan)
    ds['parameterization_lon_a'].attrs['description'] = 'longitude average climate imbalance'
    
    for i in range(0, lat):
        bottom = latbnd[i]; up = latbnd[i+1]
        find_id = np.where((data['CenLat']>=bottom) & (data['CenLat']<up))[0];
        if sum(find_id)!=0:
            ds['intercept_lat_AAR_mean'].values[i,0] = np.nanmean(data['intercept_AAR_mean'].values[find_id,0]);
            ds['intercept_lat_AAR_mean'].values[i,1] = np.nanstd(data['intercept_AAR_mean'].values[find_id,0]);
            ds['intercept_lat_AAR'].values[i,0] = np.nanmean(data['intercept_AAR'].values[find_id,0]);
            ds['intercept_lat_AAR'].values[i,1] = np.nanstd(data['intercept_AAR'].values[find_id,0]);
            ds['intercept_lat_a'].values[i,0] = np.nanmean(data['intercept_a'].values[find_id,0]);
            ds['intercept_lat_a'].values[i,1] = np.nanstd(data['intercept_a'].values[find_id,0]);
            
            ds['equil_lat_AAR_mean'].values[i,0] = np.nanmean(data['equil_AAR_mean'].values[find_id,0]);
            ds['equil_lat_AAR_mean'].values[i,1] = np.nanstd(data['equil_AAR_mean'].values[find_id,0]);
            ds['equil_lat_AAR'].values[i,0] = np.nanmean(data['equil_AAR'].values[find_id,0]);
            ds['equil_lat_AAR'].values[i,1] = np.nanstd(data['equil_AAR'].values[find_id,0]);
            ds['equil_lat_a'].values[i,0] = np.nanmean(data['equil_a'].values[find_id,0]);
            ds['equil_lat_a'].values[i,1] = np.nanstd(data['equil_a'].values[find_id,0]);
            
            ds['parameterization_lat_AAR_mean'].values[i,0] = np.nanmean(data['parameterization_AAR_mean'].values[find_id,0]);
            ds['parameterization_lat_AAR_mean'].values[i,1] = np.nanstd(data['parameterization_AAR_mean'].values[find_id,0]);
            ds['parameterization_lat_AAR'].values[i,0] = np.nanmean(data['parameterization_AAR'].values[find_id,0]);
            ds['parameterization_lat_AAR'].values[i,1] = np.nanstd(data['parameterization_AAR'].values[find_id,0]);
            ds['parameterization_lat_a'].values[i,0] = np.nanmean(data['parameterization_a'].values[find_id,0]);
            ds['parameterization_lat_a'].values[i,1] = np.nanstd(data['parameterization_a'].values[find_id,0]);
    
    for j in range(0, lon):
        left = lonbnd[j]; right = lonbnd[j+1];
        find_id = np.where( (data['CenLon']>=left) & (data['CenLon']<right))[0];
        if sum(find_id)!=0:
            ds['intercept_lon_AAR_mean'].values[j,0] = np.nanmean(data['intercept_AAR_mean'].values[find_id,0]);
            ds['intercept_lon_AAR_mean'].values[j,1] = np.nanstd(data['intercept_AAR_mean'].values[find_id,0]);
            ds['intercept_lon_AAR'].values[j,0] = np.nanmean(data['intercept_AAR'].values[find_id,0]);
            ds['intercept_lon_AAR'].values[j,1] = np.nanstd(data['intercept_AAR'].values[find_id,0]);
            ds['intercept_lon_a'].values[j,0] = np.nanmean(data['intercept_a'].values[find_id,0]);
            ds['intercept_lon_a'].values[j,1] = np.nanstd(data['intercept_a'].values[find_id,0]);
            
            ds['equil_lon_AAR_mean'].values[j,0] = np.nanmean(data['equil_AAR_mean'].values[find_id,0]);
            ds['equil_lon_AAR_mean'].values[j,1] = np.nanstd(data['equil_AAR_mean'].values[find_id,0]);
            ds['equil_lon_AAR'].values[j,0] = np.nanmean(data['equil_AAR'].values[find_id,0]);
            ds['equil_lon_AAR'].values[j,1] = np.nanstd(data['equil_AAR'].values[find_id,0]);
            ds['equil_lon_a'].values[j,0] = np.nanmean(data['equil_a'].values[find_id,0]);
            ds['equil_lon_a'].values[j,1] = np.nanstd(data['equil_a'].values[find_id,0]);
            
            ds['parameterization_lon_AAR_mean'].values[j,0] = np.nanmean(data['parameterization_AAR_mean'].values[find_id,0]);
            ds['parameterization_lon_AAR_mean'].values[j,1] = np.nanstd(data['parameterization_AAR_mean'].values[find_id,0]);
            ds['parameterization_lon_AAR'].values[j,0] = np.nanmean(data['parameterization_AAR'].values[find_id,0]);
            ds['parameterization_lon_AAR'].values[j,1] = np.nanstd(data['parameterization_AAR'].values[find_id,0]);
            ds['parameterization_lon_a'].values[j,0] = np.nanmean(data['parameterization_a'].values[find_id,0]);
            ds['parameterization_lon_a'].values[j,1] = np.nanstd(data['parameterization_a'].values[find_id,0]);
    
    # To file
    path = path + 'results_' + str(n) + '_latlon_mean.nc';
    enc_var = {'dtype': 'float32'}
    encoding = {v: enc_var for v in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding);

path = '/Users/wyan0065/Desktop/PyGEM/calving/Output/';
fn = 'ERA5_MCMC_ba1_2014_2023_corrected.nc';
data = xr.open_dataset(path+fn);

n=2;
if n==0.25:
    find_id = np.where(data['CenLon'].values<-n/2);
    data['CenLon'].values[find_id] = data['CenLon'].values[find_id] + 360;
else:
    pass

latlon_avg(n=n, data=data, path=path)