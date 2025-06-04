#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:15:36 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

experiment = 'parameterization_'
file_fp = '/Users/wyan0065/Desktop/PyGEM/calving/Output/';
grid_data = xr.open_dataset(file_fp+'results_2_all.nc', decode_times=False);
# Regional results
# 0-ELA_mean, 1-AAR_mean, 2-AAR, \
# 3-a, 4-dA, 5-dV, 6-ELA_steady, 7-THAR, 8-dV_bwl, 9-dV_eff, 10-SLR
ELA  = grid_data[experiment+'ELA_steady'].values[:,:,0]
THAR = grid_data[experiment+'THAR'].values[:,:,0]

ELA  = np.flip(ELA, axis=0)
THAR = np.flip(THAR, axis=0)

lonmin = -179.5; lonmax = 179.5;
latmin = -90; latmax = 90;
extents = [lonmin, lonmax, latmin, latmax]

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

fig = plt.figure(figsize=(4.71, 5.5), dpi=600)
gs = GridSpec(2, 1, figure=fig, hspace=0.01, height_ratios=[1,1])
plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=-0.03)

#proj = ccrs.Robinson()
proj = ccrs.PlateCarree()
regions_shp='/Users/wyan0065/Desktop/PyGEM/calving/RGI/rgi60/00_rgi60_regions/00_rgi60_O1Regions.shp'
shape_feature = ShapelyFeature(Reader(regions_shp).geometries(), ccrs.PlateCarree(), edgecolor='white', alpha=1,
                               facecolor='None', linewidth=0.5)

## ======================================================================== figure a ========================================================================
ax1 = fig.add_subplot(gs[0, 0], projection=proj)
ax1.set_global()
ax1.set_title(r'Steady-state ELA ($\mathbf{ELA}_0$; m)', fontweight='bold',loc='center', pad=2)
ax1.text(0, 1.008, 'a', fontsize=8, fontweight='bold', transform=ax1.transAxes);

ax1.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='gainsboro'))
ax1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='dimgrey'), alpha=0.5)

ax1.add_feature(shape_feature)

ax1.set_xticks(np.arange(-180, 180 + 60, 60), crs=proj)
ax1.set_xticks(np.arange(-180, 180 + 30, 30), minor=True, crs=proj)
ax1.set_yticks(np.arange(-90, 90 + 30, 30), crs=proj)
ax1.set_yticks(np.arange(-90, 90 + 15, 15), minor=True, crs=proj)
ax1.xaxis.set_major_formatter(LongitudeFormatter())
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.tick_params(axis='both', which='major', length=2, width=0.5, color='black', labelcolor='black', pad=1)
ax1.tick_params(axis='both', which='minor', length=1, width=0.5, color='black')
ax1.spines['geo'].set_edgecolor('black') 

im1  = ax1.imshow(ELA, extent=extents, transform=ccrs.PlateCarree(), alpha=1,
                  cmap='viridis_r', vmin=0, vmax=6000)

char1 = fig.colorbar(im1, ax=ax1, ticks=np.arange(0,7000,1000), extend='both',
                     shrink=0.6, aspect=30, pad=0.08, orientation='horizontal')
char1.ax.tick_params(direction='in', size=2, width=0.5, labelsize=7)

## ======================================================================== figure b ========================================================================
ax2 = fig.add_subplot(gs[1, 0], projection=proj)
ax2.set_global()

ax2.set_title(r'Steady-state THAR ($\mathbf{THAR}_0$)', fontweight='bold',loc='center', pad=2)
ax2.text(0, 1.008, 'b', fontsize=8, fontweight='bold', transform=ax2.transAxes);

ax2.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='gainsboro'))
ax2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='dimgrey'), alpha=0.5)

ax2.add_feature(shape_feature)

ax2.set_xticks(np.arange(-180, 180 + 60, 60), crs=proj)
ax2.set_xticks(np.arange(-180, 180 + 30, 30), minor=True, crs=proj)
ax2.set_yticks(np.arange(-90, 90 + 30, 30), crs=proj)
ax2.set_yticks(np.arange(-90, 90 + 15, 15), minor=True, crs=proj)
ax2.xaxis.set_major_formatter(LongitudeFormatter())
ax2.yaxis.set_major_formatter(LatitudeFormatter())
ax2.tick_params(axis='both', which='major', length=2, width=0.5, color='black', labelcolor='black', pad=1)
ax2.tick_params(axis='both', which='minor', length=1, width=0.5, color='black')
ax2.spines['geo'].set_edgecolor('black') 

col_bounds = np.linspace(0.36,0.57,7)
col_bounds = np.append(col_bounds, np.linspace(0.57,0.80,7))
cb = []
cb_val = np.linspace(1, 0, len(col_bounds))
for j in range(len(cb_val)):
    cb.append(mpl.cm.RdBu_r(cb_val[j])) #'RdYlBu_r'
cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), 
                                                                          cb)), N=1000)

norm = mpl.colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))
im2  = ax2.imshow(THAR, extent=extents, transform=ccrs.PlateCarree(), alpha=1,
                  norm=norm, cmap='RdBu_r')

char2 = fig.colorbar(im2, ax=ax2, ticks=[0.36,0.43,0.5,0.57,0.63,0.69,0.75,0.8], extend='both',
                     shrink=0.6, aspect=30, pad=0.08, orientation='horizontal')
char2.ax.tick_params(direction='in', size=2, width=0.5, labelsize=7)

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/manuscript/' + 'figure_S11.png'
plt.savefig(out_pdf, dpi=600)

plt.show()