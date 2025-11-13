#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:01:01 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import xarray as xr
from scipy.stats import median_abs_deviation
from brokenaxes import brokenaxes
import matplotlib.pyplot as plt

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
        stats = np.append(stats, np.nanpercentile(data, 2.5)) # min
        stats = np.append(stats, np.nanpercentile(data, 25)) # '25%'
        stats = np.append(stats, np.nanmedian(data)) # 'median'
        stats = np.append(stats, np.nanpercentile(data, 75)) # '75%'
        stats = np.append(stats, np.nanpercentile(data, 97.5)) # max
        stats = np.append(stats, median_abs_deviation(data, nan_policy='omit')) # Compute the median absolute deviation of the data
    elif output.ndim == 1:
        data = output[:];
        stats = None
        stats = np.nanmean(data) # 'mean'
        stats = np.append(stats, np.nanstd(data)) # 'std'
        stats = np.append(stats, np.nanpercentile(data, 2.5)) # min
        stats = np.append(stats, np.nanpercentile(data, 25)) # '25%'
        stats = np.append(stats, np.nanmedian(data)) # 'median'
        stats = np.append(stats, np.nanpercentile(data, 75)) # '75%'
        stats = np.append(stats, np.nanpercentile(data, 97.5)) # max
        stats = np.append(stats, median_abs_deviation(data, nan_policy='omit')) # Compute the median absolute deviation of the data
        
    return stats

data = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/Output/ERA5_MCMC_ba1_2014_2023_corrected.nc');

ds = xr.Dataset();

# Coordinates
ds.coords['dim'] = ('dim', np.arange(8))
ds['dim'].attrs['description'] = '0-mean, 1-std, 2-2.5%, 3-25%, 4-median, 5-75%, 6-97.5%, 7-mad'

# glac_results
ds['time_region'] = (('region', 'dim'), np.zeros([20,8])*np.nan)

i=0
ds['time_region'].values[i,:] = calc_stats_AAR(data['equil_time'].values*10);

for i in range(1,20):
    find_id = np.where(data['O1Region'].values==i)[0];
    ds['time_region'].values[i,:] = calc_stats_AAR(data['equil_time'].values[find_id]*10);

#%%
plt.rcParams.update({'lines.linewidth':0.5})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth':0.5})
plt.rcParams.update({'axes.titlepad':3})
plt.rcParams.update({'axes.titlesize':7})
plt.rcParams.update({'axes.labelpad':2})
plt.rcParams.update({'xtick.major.pad':2})
plt.rcParams.update({'ytick.major.pad':2})
plt.rcParams.update({'xtick.major.width':0.5})
plt.rcParams.update({'ytick.major.width':0.5})
plt.rcParams.update({'xtick.major.size':1.5})
plt.rcParams.update({'ytick.major.size':1.5})

plt.rcParams.update({'legend.borderaxespad':0.4})
plt.rcParams.update({'legend.fontsize':7})
plt.rcParams.update({'legend.title_fontsize':7})
plt.rcParams.update({'legend.frameon':True})
plt.rcParams.update({'legend.handlelength': 1.5})
plt.rcParams.update({'legend.handletextpad': 0.5})
plt.rcParams.update({'legend.labelspacing': 0.3})
plt.rcParams.update({'legend.framealpha':1})
plt.rcParams.update({'legend.fancybox':False})

fig = plt.figure(figsize=(3.54, 2.3), dpi=600)

ax = plt.axes([0.135,0.12,0.83,0.85], xlim=(0,5000), ylim=(-1,20))
ax.set_xlabel('Year', fontsize=7)
ax.set_ylabel('RGI region', fontsize=7)
arr = np.arange(1, 20)
arr_with_parentheses = [f'{x}' for x in arr]
y = np.concatenate((np.array(['Global']),arr_with_parentheses))

ax.errorbar(np.zeros(20), np.arange(0,20,1), xerr=ds['time_region'].values[:,6], lolims=False, uplims=True,
            fmt='o', color='orange', markersize=0, capsize=2)
ax.errorbar(np.zeros(20), np.arange(0,20,1), xerr=ds['time_region'].values[:,2], lolims=True, uplims=False,
            fmt='o', color='orange', markersize=0, capsize=2)

ax.barh(y, ds['time_region'].values[:,5], color='orange', edgecolor='none')
ax.barh(y, ds['time_region'].values[:,3], color='white', edgecolor='white')

ax.scatter(ds['time_region'].values[:,4], np.arange(0,20,1),marker='|',color='#489FE3', edgecolors='none', s=10, zorder=10)
ax.scatter(ds['time_region'].values[:,0], np.arange(0,20,1),color='#489FE3', edgecolors='none', s=10, zorder=10)

ax.invert_yaxis()

# legend
axl = plt.axes([0.8, 0.2, 0.3, 0.3], xlim=(0,1.5), ylim=(0,1))

axl.errorbar(0.05, 0.5, yerr=0.45,fmt='o', color='orange', markersize=0, capsize=2)

axl.barh(0.6, 0.1, color='orange', edgecolor='none')
axl.barh(1.2, 0.1, color='white', edgecolor='white')

axl.scatter(0.05,0.45,marker='_',color='#489FE3', edgecolors='none', s=10, zorder=10)
axl.scatter(0.05,0.6,color='#489FE3', edgecolors='none', s=10, zorder=10)

axl.text(0.15, 0.05, '2.5%', ha='left', va='center')
axl.text(0.15, 0.2, '25%', ha='left', va='center')
axl.text(0.15, 0.45, 'median', ha='left', va='center')
axl.text(0.15, 0.6, 'mean', ha='left', va='center')
axl.text(0.15, 0.75, '75%', ha='left', va='center')
axl.text(0.15, 0.95, '97.5%', ha='left', va='center')

axl.axis('off')

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/manuscript/' + 'figure_S6.png'
plt.savefig(out_pdf, dpi=600)

plt.show()

