#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:45:09 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import pygem.pygem_modelsetup as modelsetup
import pygem.pygem_input as pygem_prms

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid defined by WGS84
    (From https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7)
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = np.deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    (from https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7)
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
#                    from numpy import meshgrid, deg2rad, gradient, cos
#                    from xarray import DataArray

    xlon, ylat = np.meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = np.deg2rad(np.gradient(ylat, axis=0))
    dlon = np.deg2rad(np.gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * np.cos(np.deg2rad(ylat))

    area = dy * dx

    xda = xr.DataArray(
        area,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda

ds_temp = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/climate_data/ERA5/ERA5_temp_monthly.nc')
ds_prec = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/climate_data/ERA5/ERA5_totalprecip_monthly.nc')
temp_vn = 't2m'
prec_vn = 'tp'
lat_vn = 'latitude'
lon_vn = 'longitude'
time_vn = 'time'
startyear = 2014
endyear = 2023

dates_table = modelsetup.datesmodelrun(startyear=startyear, endyear=endyear, spinupyears=0)
start_idx = (np.where(pd.Series(ds_temp[time_vn]).apply(lambda x: x.strftime('%Y-%m')) == 
                      dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[0]))[0][0]
end_idx = (np.where(pd.Series(ds_temp[time_vn]).apply(lambda x: x.strftime('%Y-%m')) == 
                    dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[dates_table.shape[0] - 1]))[0][0]

temp_all = ds_temp[temp_vn][start_idx:end_idx+1, :, :].values
prec_all = ds_prec[prec_vn][start_idx:end_idx+1, :, :].values
temp_all = temp_all - 273.15
prec_all = prec_all * dates_table['daysinmonth'].values[:,np.newaxis,np.newaxis]

# Global average must be weighted by area
# area dataArray
da_area = area_grid(ds_temp[lat_vn].values, ds_temp[lon_vn].values)
latlon_areas = da_area.values

# total area
total_area = da_area.sum(['latitude','longitude']).values

# temperature weighted by grid-cell area
temp_global_mean_monthly = (temp_all*latlon_areas[np.newaxis,:,:]).sum((1,2)) / total_area
temp_global_mean_annual = temp_global_mean_monthly.reshape(-1,12).mean(axis=1)

# precipitation weighted by grid-cell area
prec_global_mean_monthly = (prec_all*latlon_areas[np.newaxis,:,:]).sum((1,2)) / total_area
prec_global_sum_annual = prec_global_mean_monthly.reshape(-1,12).sum(axis=1)

shuffled_yr = pd.read_csv(pygem_prms.main_directory+'/shuffled_year.csv', index_col=0)
shuffled_yr = shuffled_yr['2014-2023'].values
shuffled_yr = shuffled_yr-2014
temp = np.zeros(5000)
prec = np.zeros(5000)

for i in range(0,5000):
    yr = shuffled_yr[i]
    temp[i] = temp_global_mean_annual[yr]
    prec[i] = prec_global_sum_annual[yr]

#%%
plt.rcParams.update({'lines.linewidth':0.5})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth':0.5})
plt.rcParams.update({'axes.titlepad':3})
plt.rcParams.update({'axes.titlesize':8})
plt.rcParams.update({'axes.labelpad':2})
plt.rcParams.update({'xtick.major.pad':2})
plt.rcParams.update({'ytick.major.pad':2})
plt.rcParams.update({'xtick.major.width':0.5})
plt.rcParams.update({'ytick.major.width':0.5})
plt.rcParams.update({'xtick.major.size':1.5})
plt.rcParams.update({'ytick.major.size':1.5})

fig = plt.figure(figsize=(3.54, 3.1), dpi=600)
gs = GridSpec(2, 25, figure=fig, wspace=0, hspace=0.1)
plt.subplots_adjust(left=0.11, right=0.96, top=0.97, bottom=0.08)

ax1_1 = fig.add_subplot(gs[1, 0:4], xlim=(2014,2023), ylim=(1.052,1.098))
ax1_1.set_xticks([2014,2023])
ax1_1.set_yticks(np.arange(1.06,1.095,0.01))
ax1_1.set(ylabel='GMP (m year$^{-1}$)')

ax1_2 = fig.add_subplot(gs[1, 4:14], xlim=(1,100), ylim=(1.052,1.098))
ax1_2.set_yticks([])
ax1_2.set_yticklabels([])
ax1_2.set_xticks([20,40,60,80,100])
ax1_2.spines['right'].set_visible(False)
ax1_2.set_xlabel('Year', loc='right')

ax1_3 = fig.add_subplot(gs[1, 15:25], xlim=(4900,5000), ylim=(1.052,1.098))
ax1_3.set_yticks([])
ax1_3.set_yticklabels([])
ax1_3.set_xticks([4900,4920,4940,4960,4980,5000])
ax1_3.set_xticklabels(['','4920','','4960','','5000'])
ax1_3.spines['left'].set_visible(False)

ax2_1 = fig.add_subplot(gs[0, 0:4], xlim=(2014,2023), ylim=(14.45,15.05))
ax2_1.set_xticks([2014,2023])
ax2_1.set_xticklabels([])
ax2_1.set_yticks(np.arange(14.5,15.1,0.1))
ax2_1.set(ylabel='GMT (°C)')

ax2_2 = fig.add_subplot(gs[0, 4:14], xlim=(1,100), ylim=(14.45,15.05))
ax2_2.set_xticks([20,40,60,80,100])
ax2_2.set_yticks([])
ax2_2.set_yticklabels([])
ax2_2.set_xticklabels([])
ax2_2.spines['right'].set_visible(False)

ax2_3 = fig.add_subplot(gs[0, 15:25], xlim=(4900,5000), ylim=(14.45,15.05))
ax2_3.set_yticks([])
ax2_3.set_yticklabels([])
ax2_3.set_xticklabels([])
ax2_3.set_xticks([4900,4920,4940,4960,4980,5000])
ax2_3.spines['left'].set_visible(False)

ax1_1.plot(np.arange(2014,2024,1), prec_global_sum_annual,linewidth=1.5, color='#489FE3')
ax1_2.plot(np.arange(1,101,1), prec[0:100],linewidth=1, color='#489FE3')
ax1_3.plot(np.arange(4900,5000,1), prec[4900:],linewidth=1, color='#489FE3')

ax2_1.plot(np.arange(2014,2024,1), temp_global_mean_annual,linewidth=1.5, color='orange')
ax2_2.plot(np.arange(1,101,1), temp[0:100],linewidth=1, color='orange')
ax2_3.plot(np.arange(4900,5000,1), temp[4900:],linewidth=1, color='orange')

d = 0.02
kwargs = dict(transform = ax1_2.transAxes, color='k', linewidth=0.5, clip_on=False)
ax1_2.plot((1-d,1+d),(-d,d), **kwargs)
ax1_2.plot((1-d,1+d),(1-d,1+d), **kwargs)

kwargs.update(transform = ax1_3.transAxes)
ax1_3.plot((-d,+d),(-d,d), **kwargs)
ax1_3.plot((-d,+d),(1-d,1+d), **kwargs)

kwargs = dict(transform = ax2_2.transAxes, color='k', linewidth=0.5, clip_on=False)
ax2_2.plot((1-d,1+d),(-d,d), **kwargs)
ax2_2.plot((1-d,1+d),(1-d,1+d), **kwargs)

kwargs.update(transform = ax2_3.transAxes)
ax2_3.plot((-d,+d),(-d,d), **kwargs)
ax2_3.plot((-d,+d),(1-d,1+d), **kwargs)

#plt.tight_layout()
out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/manuscript/' + 'figure_S3.png'
plt.savefig(out_pdf, dpi=600)

plt.show()