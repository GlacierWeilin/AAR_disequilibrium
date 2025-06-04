#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:44:08 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

#%% Figure 1: AAR_steady, and α
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import scipy.stats

#%% data
experiment = 'parameterization_'
# Grid data
# 0-mean, 1-std, 2-5%, 3-17%, 4-median, 5-83%, 6-95%, 7-mad, \
# 8-n, 9-sum(std), 10-mean_std(std), 11-sum_std(std)
file_fp = '/Users/wyan0065/Desktop/PyGEM/calving/Output/';
grid_data = xr.open_dataset(file_fp+'results_2_all.nc', decode_times=False);
AAR_mean = grid_data[experiment+'AAR_mean'].values[:,:,0]
AAR      = grid_data[experiment+'AAR'].values[:,:,0]
#a        = grid_data[experiment+'dA'].values[:,:,9] / grid_data['area_2020'].values[:,:,9] + 1
a      = grid_data[experiment+'a'].values[:,:,0]

AAR_mean = np.flip(AAR_mean, axis=0)
AAR      = np.flip(AAR, axis=0)
a        = np.flip(a, axis=0)

lonmin = -179.5; lonmax = 179.5;
latmin = -90; latmax = 90;
extents = [lonmin, lonmax, latmin, latmax]

# Regional results
# 0-ELA_mean, 1-AAR_mean, 2-AAR, \
# 3-a, 4-dA, 5-dV, 6-ELA_steady, 7-THAR, 8-dV_bwl, 9-dV_eff, 10-SLR
region_data     = xr.open_dataset(file_fp+'results_by_region.nc', decode_times=False);
region_AAR_mean = region_data[experiment+'region'].values[:,1,0]
region_AAR      = region_data[experiment+'region'].values[:,2,0]
#region_a        = region_data[experiment+'region'].values[:,4,9] / region_data['glac_region'].values[:,1,9] + 1
region_a      = region_data[experiment+'region'].values[:,3,0]

region_AAR_mean = np.round(region_AAR_mean, 2)
region_AAR      = np.round(region_AAR, 2)

# latlon mean
latlon_mean = xr.open_dataset(file_fp+'results_2_latlon_mean.nc', decode_times=False);
lat = latlon_mean['latitude']
lon = latlon_mean['longitude']
lat_AAR_mean = latlon_mean[experiment+'lat_AAR_mean'].values[:,0]
lon_AAR_mean = latlon_mean[experiment+'lon_AAR_mean'].values[:,0]
lat_AAR = latlon_mean[experiment+'lat_AAR'].values[:,0]
lon_AAR = latlon_mean[experiment+'lon_AAR'].values[:,0]
lat_AAR_mean_std = latlon_mean[experiment+'lat_AAR_mean'].values[:,1]
lon_AAR_mean_std = latlon_mean[experiment+'lon_AAR_mean'].values[:,1]
lat_AAR_std = latlon_mean[experiment+'lat_AAR'].values[:,1]
lon_AAR_std = latlon_mean[experiment+'lon_AAR'].values[:,1]
lat_a = latlon_mean[experiment+'lat_a'].values[:,0]
lon_a = latlon_mean[experiment+'lon_a'].values[:,0]
lat_a_std = latlon_mean[experiment+'lat_a'].values[:,1]
lon_a_std = latlon_mean[experiment+'lon_a'].values[:,1]

# Histogram
data = xr.open_dataset(file_fp+'ERA5_MCMC_ba1_2014_2023_corrected.nc', decode_times=False);
compile_AAR_mean = data[experiment+'AAR_mean'].values[:,0]
compile_AAR = data[experiment+'AAR'].values[:,0]
compile_a = data[experiment+'a'].values[:,0]

# WGMS
wgms_data  = pd.read_csv(file_fp + '/WGMS_disequilibrium_RGI.csv')
wgms_AAR_mean = wgms_data['wgms_AAR_mean'].values
wgms_AAR = wgms_data['intercept'].values
wgms_a   = wgms_data['a'].values

# previous publications
repub_AAR_mean = [[0.44, 0.492, 0.34], [0.02, 0.142, 0.03]] # mean with 1 sigma
repub_AAR = [[0.57, 0.579, 0.55241, 0.559], [0.01, 0.087, 0.09939, 0.09]] # mean with 1 sigma
repub_a   = [[0.44/0.57, 49.2/57.9, 0.68],[(0.57*0.02-0.44*0.01)/0.57/0.57, 0.22, 0.12]] # x=a±p, y=b±q; x/y; a/b; (b*p-a*q)/b^2
repub_label = ['Bahr et al.$^{13}$', 'Dyurgerov et al.$^{12}$', 'Mernild et al.$^{11}$', 'Kern and László$^{22}$']

#%% default parameters for plotting

plt.rcParams.update({'lines.linewidth':0.5})
plt.rcParams.update({'axes.linewidth':0.5})
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'font.sans-serif': 'Arial'})

#                        1,     2,    3,     4,     5,   6,   7,   8,    9,  10,    11,    12,  13,    14,    15,  16,   17,    18,   19
text_lon = np.array([-177, -132.5, -127, -88.5,   -56, -25,  -8,33.5,   37, 177,  -4.5,  33.5, 107,  66.5, 103.5, 140,  -60,   162, -177])
text_lat = np.array([  48,     33, 83.5,    55,    57,  57,83.5,  57,   69,  87,    38,    29,  44,    24,    24, -23,  -27, -35.5, -47.5])
text_loc = np.array(['lt',   'lt', 'rt',  'lt','lt','lt','lt','rb', 'lb','rt',  'lt',  'lt','lt',  'lt','rt','rb', 'lt',  'rt', 'lt'])

point_lon= np.array([-177+8, -132.5+8, -127-3, -88.5+8, -56+8, -24+8, -8+8, 33.5-3, 37+8, 177-3, -4.5+12, 33.5+12, 107+12, 66.5+12, 103.5-3, 140-3, -60+12, 162-3, -177+12])
point_lat= np.array([48-3, 33-3, 83.5-3, 55-3, 57-3, 57-3, 83.5-3, 57+3, 69+3.5, 87-3, 38-3, 29-3, 44-3, 24-3, 24-3, -23+3, -27-3, -35.5-3, -47.5-3])

proj = ccrs.PlateCarree()
fig_width_inch = 7
fig = plt.figure(figsize=(fig_width_inch, 8.7), dpi=600)
box_fig = fig.get_window_extent()

regions_shp='/Users/wyan0065/Desktop/PyGEM/calving/RGI/rgi60/00_rgi60_regions/00_rgi60_O1Regions.shp'
shape_feature = ShapelyFeature(Reader(regions_shp).geometries(), ccrs.PlateCarree(), edgecolor='white', alpha=1,
                               facecolor='None', linewidth=0.5)
#%% plot
## ======================================================================== figure_1a ======================================================================== 
ax_a = fig.add_subplot([0.12,0.515,0.7,0.7], projection=proj)
box_window1 = pd.Series([2469.6, 1234.7999999999997, 504.0, 3897.8999999999996], index=['width', 'height', 'x0', 'y0']);
ax_a.set_global()
ax_a.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='gainsboro'))
ax_a.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='dimgrey', alpha=0.5))
ax_a.set_title('Glacier disequilibrium (α)', fontsize=7, fontweight='bold',loc='center', pad=3)
ax_a.text(0, 1.02, 'a', fontsize=7, fontweight='bold', transform=ax_a.transAxes);
ax_a.spines['geo'].set_edgecolor('black')
ax_a.add_feature(shape_feature)

ax_a.text(-150, 0, 'Global\n      ', fontsize=7, alpha=1, color='black', ha='center', va='center', transform=ax_a.transData,
         bbox={'facecolor':'white', 'pad': 1, 'linewidth': 0.2});

for i in range(0, len(text_lon)):
    if text_loc[i] == 'lt':
        ha='left'; va='top';
    elif text_loc[i] == 'rt':
        ha='right'; va='top';
    elif text_loc[i] == 'rb':
        ha='right'; va='bottom';
    elif text_loc[i] == 'lb':
        ha='left'; va='bottom';
    
    ax_a.text(text_lon[i], text_lat[i], str(i+1)+'   ', fontsize=6, alpha=1, color='black', ha=ha, va=va, transform=ax_a.transData,
             bbox={'facecolor':'white', 'pad': 1, 'linewidth': 0.2});

ax_a.set_xticks(np.arange(-180, 180 + 60, 60), crs=proj)
ax_a.set_xticks(np.arange(-180, 180 + 30, 30), minor=True, crs=proj)
ax_a.set_yticks(np.arange(-90, 90 + 30, 30), crs=proj)
ax_a.set_yticks(np.arange(-90, 90 + 15, 15), minor=True, crs=proj)
ax_a.xaxis.set_major_formatter(LongitudeFormatter())
ax_a.yaxis.set_major_formatter(LatitudeFormatter())
ax_a.tick_params(axis='both', which='major', length=2, width=0.5, color='black', labelcolor='black', pad=37)
ax_a.tick_params(axis='both', which='minor', length=1, width=0.5, color='black')

col_bounds = np.linspace(0.4,1.0,7)
col_bounds = np.append(col_bounds, np.linspace(1.0,1.2,7))
cb = []
cb_val = np.linspace(1, 0, len(col_bounds))
for j in range(len(cb_val)):
    cb.append(mpl.cm.RdBu_r(cb_val[j])) #'RdYlBu_r'
cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), 
                                                                          cb)), N=1000)

norm = mpl.colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))
im1  = ax_a.imshow(a, extent=extents, transform=ccrs.PlateCarree(), alpha=0.8,
                  norm=norm, cmap=cmap_cus, zorder=2)
ax_a.scatter(-150, -4, c=region_a[0], s=10,
            norm=norm, cmap=cmap_cus, zorder=3)
ax_a.scatter(point_lon, point_lat, c=region_a[1:20], s=7,
            norm=norm, cmap=cmap_cus, zorder=3)

cbar1 = fig.colorbar(im1, ax=ax_a, ticks=np.linspace(0.4, 1.2, 9), extend='both',
                     shrink=box_window1['height']/(box_fig.height*0.7), aspect=25, pad=0.01, orientation='vertical') # horizontal

cbar1.ax.tick_params(direction='in', size=2, width=0.5, labelsize=6, pad=1.5, labelcolor='black')

## ================================================================= figure_1a: latitude mean =================================================================
ax_a1 = fig.add_subplot([0.05,box_window1.y0/box_fig.height,0.07,box_window1['height']/box_fig.height], facecolor='None')
ax_a1.plot(lat_a, lat, color='royalblue');
ax_a1.plot(lat_a-lat_a_std, lat, linewidth=0);
ax_a1.plot(lat_a+lat_a_std, lat, linewidth=0);
ax_a1.fill_betweenx(lat, lat_a-lat_a_std, lat_a+lat_a_std, color='lightsteelblue', alpha=1)

ax_a1.axvline(x=region_a[0], color='dimgrey', linestyle='--', label='Global mean')

ax_a1.set_xlim(0,1.5)
ax_a1.set_ylim(-90,90)
ax_a1.yaxis.set_major_formatter(plt.NullFormatter())
ax_a1.set_xticks(np.arange(0.5, 1.5, 0.5))
ax_a1.set_xticks(np.arange(0, 1.5, 0.25), minor=True)
ax_a1.set_yticks(np.arange(-90, 90 + 30, 30))
ax_a1.set_yticks(np.arange(-90, 90 + 15, 15), minor=True)
ax_a1.tick_params(axis='both', which='major', length=2, width=0.5, color='black',labelcolor='black', pad=1.5)
ax_a1.tick_params(axis='both', which='minor', length=1, width=0.5, color='black')
ax_a1.spines[['top', 'bottom', 'right', 'left']].set_edgecolor('black')

## ================================================================= figure_1a: longitude mean =================================================================
ax_a2_height = box_fig.width*0.07/box_fig.height
ax_a2 = fig.add_subplot([0.12, box_window1.y0/box_fig.height-ax_a2_height, box_window1['width']/box_fig.width, ax_a2_height], facecolor='None')
ax_a2.plot(lon, lon_a);
ax_a2.plot(lon, lon_a, color='royalblue', label='Mean');
ax_a2.plot(lon, lon_a-lon_a_std, linewidth=0);
ax_a2.plot(lon, lon_a+lon_a_std, linewidth=0);
ax_a2.fill_between(lon, lon_a-lon_a_std, lon_a+lon_a_std, color='lightsteelblue', alpha=1, label=r'1$\sigma$')

ax_a2.axhline(y=region_a[0], color='dimgrey', linestyle='--', label='Global mean')

ax_a2.set_ylim(0,1.5)
ax_a2.set_xlim(-180,180)
ax_a2.xaxis.set_major_formatter(plt.NullFormatter())
ax_a2.set_xticks(np.arange(-180, 180 + 60, 60))
ax_a2.set_xticks(np.arange(-180, 180 + 30, 30), minor=True)
ax_a2.set_yticks(np.arange(0.5, 1.5, 0.5))
ax_a2.set_yticks(np.arange(0, 1.5, 0.25), minor=True)
ax_a2.tick_params(axis='both', which='major', length=2, width=0.5, color='black',labelcolor='black', pad=1.5,
                  left=False, right=True, labelleft=False, labelright=True)
ax_a2.tick_params(axis='both', which='minor', length=1, width=0.5, color='black', left=False, right=True)
ax_a2.spines[['top', 'bottom', 'right', 'left']].set_edgecolor('black')

handles, labels = ax_a2.get_legend_handles_labels()
labels[2] = 'Global\nmean'
ax_a2.legend([handles[0],handles[2], handles[1]],[labels[0], labels[2], labels[1]], loc='best', bbox_to_anchor=(0.005,0.8), ncols=1, fontsize=6, frameon=False,
             borderpad=0.3, handlelength=1.5, labelspacing=0.3, handletextpad=0.4, columnspacing=-2);

## ================================================================= figure_1b: Histogram =================================================================
ax_b = fig.add_subplot([0.77,box_window1.y0/box_fig.height,0.19,box_window1['height']/box_fig.height], facecolor='None')
ax_b.text(0, 1.02, 'b', fontsize=7, fontweight='bold', transform=ax_b.transAxes);

## ============== Our study ==============
ax_b.hist(compile_a, density=True, bins = np.linspace(0.1, 3, 30), alpha=1,
           histtype='stepfilled', color='lightsteelblue', edgecolor='none', orientation='horizontal', label='Histogram');

# Normal
compile_a_normx = np.linspace(compile_a.min(), compile_a.max(), 1000)
compile_a_normy = scipy.stats.norm.pdf(compile_a_normx, compile_a.mean(), compile_a.std())
ax_b.plot(compile_a_normy, compile_a_normx, color='royalblue', linestyle='-', label='Gaussian DIST')

# Mean and Median
ax_b.axhline(y=np.median(compile_a), color='royalblue', linestyle='--', label='Median')

ax_b.errorbar(max(compile_a_normy), compile_a.mean(), fmt='o', yerr=compile_a.std(), capsize=1, elinewidth=0.5, capthick=0.5,
              label=r'Mean with 1$\sigma$', c='royalblue', markersize=3)

ax_b.invert_xaxis()
ax_b.set_yticks(np.arange(0, 3.5, 0.5))
ax_b.set_yticks(np.arange(0, 3.5, 0.25), minor=True)
ax_b.set_ylim(0, 3)

## ============== tick_params ==============
ax_b.tick_params(axis='both', which='major', length=2, width=0.5, color='black',labelcolor='black', pad=1.5,
                  left=False, right=True, labelleft=False, labelright=True, bottom=False, labelbottom=False)
ax_b.tick_params(axis='both', which='minor', length=1, width=0.5, color='black', left=False, right=True, bottom=False)
ax_b.spines[['top', 'bottom', 'right', 'left']].set_edgecolor('black')

ax_b.legend(loc='upper left', ncols=1, fontsize=6, markerscale=0.7, frameon=False, borderpad=0.3, 
            labelspacing=0.3, handletextpad=0.4, title='This study:', title_fontsize=6, alignment='left');

## ================================================================= figure_1b: WGMS =================================================================
ax_b1 = fig.add_subplot([0.77,box_window1.y0/box_fig.height,0.19,box_window1['height']/box_fig.height], facecolor='None')

ax_b1.hist(wgms_a, density=True, bins = np.linspace(0.1, 3, 30), alpha=0.3,
           histtype='stepfilled', color='orange', edgecolor='none', orientation='horizontal', label='Histogram');

# Normal
wgms_a_normx = np.linspace(wgms_a.min(), wgms_a.max(), 1000)
wgms_a_normy = scipy.stats.norm.pdf(compile_a_normx, wgms_a.mean(), wgms_a.std())
ax_b1.plot(wgms_a_normy, compile_a_normx, color='orange', linestyle='-', label='Gaussian DIST')

# Mean and Median
#ax_b1.axhline(y=wgms_a.mean(), color='darkorange', linestyle='--', label='Mean')
ax_b1.axhline(y=np.median(wgms_a), color='darkorange', linestyle='--', label='Median')

ax_b1.errorbar(max(wgms_a_normy), wgms_a.mean(), fmt='o', yerr=wgms_a.std(), capsize=1, elinewidth=0.5, capthick=0.5,
               label=r'Mean with 1$\sigma$', c='darkorange', markersize=3)

ax_b1.invert_xaxis()
ax_b1.yaxis.set_major_locator(plt.NullLocator())
ax_b1.xaxis.set_major_locator(plt.NullLocator())
ax_b1.set_ylim(0, 3)

ax_b1.legend(loc='best', bbox_to_anchor=(0.38,0.43,0.3,0.3), ncols=1, fontsize=6, markerscale=0.7, frameon=False, borderpad=0.3,
             labelspacing=0.3, handletextpad=0.4, title='WGMS:', title_fontsize=6, alignment='left');

## ================================================================= figure_1b: previous publications =================================================================
ax_b2 = fig.add_subplot([0.77,box_window1.y0/box_fig.height,0.19,box_window1['height']/box_fig.height], facecolor='None')

repub_x = np.linspace(0.6, 0.7, 3)
marker = ['v', '^', 's', 'd']
for i in np.array([0,1,2]):
    ax_b2.errorbar(repub_x[i], repub_a[0][i], fmt=marker[i], yerr=repub_a[1][i], capsize=1, elinewidth=0.5, capthick=0.5,
                   label=repub_label[i], c='black', markersize=3)
    
ax_b2.yaxis.set_major_locator(plt.NullLocator())
ax_b2.xaxis.set_major_locator(plt.NullLocator())
ax_b2.set_ylim(0, 3)

#ax_b2.legend(loc='best', bbox_to_anchor=(0.6,-0.3,0.3,0.3), ncols=1, fontsize=6, markerscale=0.7, frameon=False, borderpad=0.3,
#             labelspacing=0.5, handletextpad=0.4, title=r'Mean with 1$\sigma$', title_fontsize=6, alignment='left');

## ======================================================================= figure_1c =======================================================================
ax_c= fig.add_subplot([0.12,0.19,0.7,0.7], projection=proj)
box_window2 = pd.Series([2469.5999999999995, 1234.8000000000002, 504.0, 2201.3999999999996], index=['width', 'height', 'x0', 'y0']);
ax_c.set_global()
ax_c.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='gainsboro'))
ax_c.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='dimgrey', alpha=0.5))
ax_c.set_title(r'$\mathbf{\overline{AAR}}$ from 2014 to 2023', fontsize=7, fontweight='bold', loc='center', pad=3)
ax_c.text(0, 1.02, 'c', fontsize=7, fontweight='bold', transform=ax_c.transAxes);
ax_c.spines['geo'].set_edgecolor('black')
ax_c.add_feature(shape_feature)

ax_c.text(-150, 0, 'Global\n      ', fontsize=7, alpha=1, color='black', ha='center', va='center', transform=ax_c.transData,
         bbox={'facecolor':'white', 'pad': 1, 'linewidth': 0.2});

for i in range(0, len(text_lon)):
    if text_loc[i] == 'lt':
        ha='left'; va='top';
    elif text_loc[i] == 'rt':
        ha='right'; va='top';
    elif text_loc[i] == 'rb':
        ha='right'; va='bottom';
    elif text_loc[i] == 'lb':
        ha='left'; va='bottom';
    
    ax_c.text(text_lon[i], text_lat[i], str(i+1)+'   ', fontsize=6, alpha=1, color='black', ha=ha, va=va, transform=ax_c.transData,
             bbox={'facecolor':'white', 'pad': 1, 'linewidth': 0.2});

ax_c.set_xticks(np.arange(-180, 180 + 60, 60), crs=proj)
ax_c.set_xticks(np.arange(-180, 180 + 30, 30), minor=True, crs=proj)
ax_c.set_yticks(np.arange(-90, 90 + 30, 30), crs=proj)
ax_c.set_yticks(np.arange(-90, 90 + 15, 15), minor=True, crs=proj)
ax_c.xaxis.set_major_formatter(LongitudeFormatter())
ax_c.yaxis.set_major_formatter(LatitudeFormatter())
ax_c.tick_params(axis='both', which='major', length=2, width=0.5, color='black', labelcolor='black', pad=37)
ax_c.tick_params(axis='both', which='minor', length=1, width=0.5, color='black')

col_bounds = np.linspace(0.20,0.44,7)
col_bounds = np.append(col_bounds, np.linspace(0.44,0.68,7))
cb = []
cb_val = np.linspace(1, 0, len(col_bounds))
for j in range(len(cb_val)):
    cb.append(mpl.cm.RdBu(cb_val[j])) #'RdYlBu_r'
cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), 
                                                                          cb)), N=1000)

norm = mpl.colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))
im1  = ax_c.imshow(AAR_mean, extent=extents, transform=ccrs.PlateCarree(), alpha=0.8,
                  norm=norm, cmap=cmap_cus, zorder=2)
ax_c.scatter(-150, -4, c=region_AAR_mean[0], s=10,
            norm=norm, cmap=cmap_cus, zorder=3)
ax_c.scatter(point_lon, point_lat, c=region_AAR_mean[1:20], s=7,
            norm=norm, cmap=cmap_cus, zorder=3)

cbar1 = fig.colorbar(im1, ax=ax_c, ticks=np.arange(0.2,0.68,0.06), extend='both',
                     shrink=box_window1['height']/(box_fig.height*0.7), aspect=25, pad=0.01, orientation='vertical') # horizontal

cbar1.ax.tick_params(direction='in', size=2, width=0.5, labelsize=6, pad=1.5, labelcolor='black')

## ================================================================= figure_1c: latitude mean =================================================================
ax_c1 = fig.add_subplot([0.05,box_window2.y0/box_fig.height,0.07,box_window2['height']/box_fig.height], facecolor='None')

ax_c1.plot(lat_AAR_mean, lat, color='royalblue');
ax_c1.plot(lat_AAR_mean-lat_AAR_mean_std, lat, linewidth=0);
ax_c1.plot(lat_AAR_mean+lat_AAR_mean_std, lat, linewidth=0);
ax_c1.fill_betweenx(lat, lat_AAR_mean-lat_AAR_mean_std, lat_AAR_mean+lat_AAR_mean_std, color='lightsteelblue', alpha=1)

ax_c1.axvline(x=region_AAR_mean[0], color='dimgrey', linestyle='--', label='Global mean')

ax_c1.set_xlim(0.1,0.9)
ax_c1.set_ylim(-90,90)
ax_c1.yaxis.set_major_formatter(plt.NullFormatter())
ax_c1.set_xticks(np.arange(0.4, 0.9, 0.3))
ax_c1.set_xticks(np.arange(0.1, 0.9, 0.15), minor=True)
ax_c1.set_yticks(np.arange(-90, 90 + 30, 30))
ax_c1.set_yticks(np.arange(-90, 90 + 15, 15), minor=True)
ax_c1.tick_params(axis='both', which='major', length=2, width=0.5, color='black',labelcolor='black', pad=1.5)
ax_c1.tick_params(axis='both', which='minor', length=1, width=0.5, color='black')
ax_c1.spines[['top', 'bottom', 'right', 'left']].set_edgecolor('black')

## ================================================================= figure_1c: longitude mean =================================================================
ax_c2_height = box_fig.width*0.07/box_fig.height
ax_c2 = fig.add_subplot([0.12, box_window2.y0/box_fig.height-ax_c2_height, box_window2['width']/box_fig.width, ax_c2_height], facecolor='None')
ax_c2.plot(lon, lon_AAR_mean);
ax_c2.plot(lon, lon_AAR_mean, color='royalblue', label='Mean');
ax_c2.plot(lon, lon_AAR_mean-lon_AAR_mean_std, linewidth=0);
ax_c2.plot(lon, lon_AAR_mean+lon_AAR_mean_std, linewidth=0);
ax_c2.fill_between(lon, lon_AAR_mean-lon_AAR_mean_std, lon_AAR_mean+lon_AAR_mean_std, color='lightsteelblue', alpha=1, label=r'1$\sigma$')

ax_c2.axhline(y=region_AAR_mean[0], color='dimgrey', linestyle='--', label='Global mean')

ax_c2.set_ylim(0.1,0.9)
ax_c2.set_xlim(-180,180)
ax_c2.xaxis.set_major_formatter(plt.NullFormatter())
ax_c2.set_xticks(np.arange(-180, 180 + 60, 60))
ax_c2.set_xticks(np.arange(-180, 180 + 30, 30), minor=True)
ax_c2.set_yticks(np.arange(0.4, 0.9, 0.3))
ax_c2.set_yticks(np.arange(0.1,0.9, 0.15), minor=True)
ax_c2.tick_params(axis='both', which='major', length=2, width=0.5, color='black',labelcolor='black', pad=1.5,
                  left=False, right=True, labelleft=False, labelright=True)
ax_c2.tick_params(axis='both', which='minor', length=1, width=0.5, color='black', left=False, right=True)
ax_c2.spines[['top', 'bottom', 'right', 'left']].set_edgecolor('black')

#loc='best', bbox_to_anchor=(0.65,0.55,0.2,0.2)
handles, labels = ax_c2.get_legend_handles_labels()
labels[2] = 'Global\nmean'
ax_c2.legend([handles[0],handles[2], handles[1]],[labels[0], labels[2], labels[1]], loc='best', bbox_to_anchor=(0.005,0.8), ncols=1, fontsize=6, frameon=False,
             borderpad=0.3, handlelength=1.5, labelspacing=0.3, handletextpad=0.4, columnspacing=-2);

## ================================================================= figure_1d: Histogram =================================================================
ax_d = fig.add_subplot([0.77,box_window2.y0/box_fig.height,0.19,box_window2['height']/box_fig.height], facecolor='None')
ax_d.text(0, 1.02, 'd', fontsize=7, fontweight='bold', transform=ax_d.transAxes);

## ============== Our study ==============
ax_d.hist(compile_AAR_mean, density=True, bins = np.linspace(0.05, 0.95, 37), alpha=1,
           histtype='stepfilled', color='lightsteelblue', edgecolor='none', orientation='horizontal', label='Histogram');

# Normal
compile_AAR_normx = np.linspace(compile_AAR_mean.min(), compile_AAR_mean.max(), 1000)
compile_AAR_normy = scipy.stats.norm.pdf(compile_AAR_normx, compile_AAR_mean.mean(), compile_AAR_mean.std())
ax_d.plot(compile_AAR_normy, compile_AAR_normx, color='royalblue', linestyle='-', label='Gaussian DIST')

# Mean and Median
ax_d.axhline(y=np.median(compile_AAR_mean), color='royalblue', linestyle='--', label='Median')

ax_d.errorbar(max(compile_AAR_normy), compile_AAR_mean.mean(), fmt='o', yerr=compile_AAR_mean.std(), capsize=1, elinewidth=0.5, capthick=0.5,
              label=r'Mean with 1$\sigma$', c='royalblue', markersize=3)

ax_d.invert_xaxis()
ax_d.set_yticks(np.arange(0.1, 1, 0.1))
ax_d.set_yticks(np.arange(0.05, 1, 0.05), minor=True)
ax_d.set_ylim(0.05, 0.95)

## ============== tick_params ==============
ax_d.tick_params(axis='both', which='major', length=2, width=0.5, color='black',labelcolor='black', pad=1.5,
                  left=False, right=True, labelleft=False, labelright=True, bottom=False, labelbottom=False)
ax_d.tick_params(axis='both', which='minor', length=1, width=0.5, color='black', left=False, right=True, bottom=False)
ax_d.spines[['top', 'bottom', 'right', 'left']].set_edgecolor('black')

#ax_d.legend(loc='upper left', ncols=1, fontsize=6, markerscale=0.7, frameon=False, borderpad=0.3, 
#            labelspacing=0.3, handletextpad=0.4, title='This study:', title_fontsize=6, alignment='left');

## ================================================================= figure_1d: WGMS =================================================================
ax_d1 = fig.add_subplot([0.77,box_window2.y0/box_fig.height,0.19,box_window2['height']/box_fig.height], facecolor='None')

ax_d1.hist(wgms_AAR_mean, density=True, bins = np.linspace(0.05, 0.95, 37), alpha=0.3,
           histtype='stepfilled', color='orange', edgecolor='none', orientation='horizontal', label='Histogram');

# Normal
wgms_AAR_normx = np.linspace(wgms_AAR_mean.min(), wgms_AAR_mean.max(), 1000)
wgms_AAR_normy = scipy.stats.norm.pdf(wgms_AAR_normx, wgms_AAR_mean.mean(), wgms_AAR_mean.std())
ax_d1.plot(wgms_AAR_normy, wgms_AAR_normx, color='orange', linestyle='-', label='Gaussian DIST')

# Mean and Median
ax_d1.axhline(y=np.median(wgms_AAR_mean), color='darkorange', linestyle='--', label='Median')

ax_d1.errorbar(max(wgms_AAR_normy), wgms_AAR_mean.mean(), fmt='o', yerr=wgms_AAR_mean.std(), capsize=1, elinewidth=0.5, capthick=0.5,
               label=r'Mean with 1$\sigma$', c='darkorange', markersize=3)

ax_d1.invert_xaxis()
ax_d1.yaxis.set_major_locator(plt.NullLocator())
ax_d1.xaxis.set_major_locator(plt.NullLocator())
ax_d1.set_ylim(0.05, 0.95)

#ax_d1.legend(loc='lower left', ncols=1, fontsize=6, markerscale=0.7, frameon=False, borderpad=0.3,
#             labelspacing=0.3, handletextpad=0.4, title='WGMS:', title_fontsize=6, alignment='left');

## ================================================================= figure_1d: previous publications =================================================================
ax_d2 = fig.add_subplot([0.77,box_window2.y0/box_fig.height,0.19,box_window2['height']/box_fig.height], facecolor='None')

marker = ['v', '^', 's', 'd']
repub_x = np.linspace(0.6, 0.7, 4)
for i in np.array([0,1,2]):
    ax_d2.errorbar(repub_x[i], repub_AAR_mean[0][i], fmt=marker[i], yerr=repub_AAR_mean[1][i], capsize=1, elinewidth=0.5, capthick=0.5,
                   label=repub_label[i], c='black', markersize=3)
    
ax_d2.yaxis.set_major_locator(plt.NullLocator())
ax_d2.xaxis.set_major_locator(plt.NullLocator())
ax_d2.set_ylim(0.05, 0.95)

#ax_d2.legend(loc='best', bbox_to_anchor=(0.6,-0.3,0.3,0.3), ncols=1, fontsize=6, markerscale=0.7, frameon=False, borderpad=0.3,
#             labelspacing=0.5, handletextpad=0.4, title=r'Mean with 1$\sigma$', title_fontsize=6, alignment='left');

## ======================================================================= figure_1e =======================================================================
ax_e= fig.add_subplot([0.12,-0.135,0.7,0.7], projection=proj)
box_window3 = pd.Series([2469.5999999999995, 1234.8000000000002, 504.0, 504.8999999999998], index=['width', 'height', 'x0', 'y0']);
ax_e.set_global()
ax_e.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='gainsboro'))
ax_e.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='dimgrey', alpha=0.5))
ax_e.set_title(r'Steady-state AAR ($\mathbf{AAR}_0$)', fontsize=7, fontweight='bold', loc='center', pad=3)
ax_e.text(0, 1.02, 'e', fontsize=7, fontweight='bold', transform=ax_e.transAxes);
ax_e.spines['geo'].set_edgecolor('black')
ax_e.add_feature(shape_feature)

ax_e.text(-150, 0, 'Global\n      ', fontsize=7, alpha=1, color='black', ha='center', va='center', transform=ax_e.transData,
         bbox={'facecolor':'white', 'pad': 1, 'linewidth': 0.2});

for i in range(0, len(text_lon)):
    if text_loc[i] == 'lt':
        ha='left'; va='top';
    elif text_loc[i] == 'rt':
        ha='right'; va='top';
    elif text_loc[i] == 'rb':
        ha='right'; va='bottom';
    elif text_loc[i] == 'lb':
        ha='left'; va='bottom';
    
    ax_e.text(text_lon[i], text_lat[i], str(i+1)+'   ', fontsize=6, alpha=1, color='black', ha=ha, va=va, transform=ax_e.transData,
             bbox={'facecolor':'white', 'pad': 1, 'linewidth': 0.2});

ax_e.set_xticks(np.arange(-180, 180 + 60, 60), crs=proj)
ax_e.set_xticks(np.arange(-180, 180 + 30, 30), minor=True, crs=proj)
ax_e.set_yticks(np.arange(-90, 90 + 30, 30), crs=proj)
ax_e.set_yticks(np.arange(-90, 90 + 15, 15), minor=True, crs=proj)
ax_e.xaxis.set_major_formatter(LongitudeFormatter())
ax_e.yaxis.set_major_formatter(LatitudeFormatter())
ax_e.tick_params(axis='both', which='major', length=2, width=0.5, color='black', labelcolor='black', pad=37)
ax_e.tick_params(axis='both', which='minor', length=1, width=0.5, color='black')

col_bounds = np.linspace(0.40,0.53,7)
col_bounds = np.append(col_bounds, np.linspace(0.53,0.66,7))
cb = []
cb_val = np.linspace(1, 0, len(col_bounds))
for j in range(len(cb_val)):
    cb.append(mpl.cm.RdBu(cb_val[j])) #'RdYlBu_r'
cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), 
                                                                          cb)), N=1000)

norm = mpl.colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))
im1  = ax_e.imshow(AAR, extent=extents, transform=ccrs.PlateCarree(), alpha=0.8,
                  norm=norm, cmap=cmap_cus, zorder=2)
ax_e.scatter(-150, -4, c=region_AAR[0], s=10,
            norm=norm, cmap=cmap_cus, zorder=3)
ax_e.scatter(point_lon, point_lat, c=region_AAR[1:20], s=7,
            norm=norm, cmap=cmap_cus, zorder=3)

cbar1 = fig.colorbar(im1, ax=ax_e, ticks=[0.40,0.45,0.49,0.53,0.57,0.61,0.66], extend='both',
                     shrink=box_window1['height']/(box_fig.height*0.7), aspect=25, pad=0.01, orientation='vertical') # horizontal

cbar1.ax.tick_params(direction='in', size=2, width=0.5, labelsize=6, pad=1.5, labelcolor='black')

## ================================================================= figure_1e: latitude mean =================================================================
ax_e1 = fig.add_subplot([0.05,box_window3.y0/box_fig.height,0.07,box_window3['height']/box_fig.height], facecolor='None')

ax_e1.plot(lat_AAR, lat, color='royalblue');
ax_e1.plot(lat_AAR-lat_AAR_std, lat, linewidth=0);
ax_e1.plot(lat_AAR+lat_AAR_std, lat, linewidth=0);
ax_e1.fill_betweenx(lat, lat_AAR-lat_AAR_std, lat_AAR+lat_AAR_std, color='lightsteelblue', alpha=1)

ax_e1.axvline(x=region_AAR[0], color='dimgrey', linestyle='--', label='Global mean')

ax_e1.set_xlim(0.22,0.78)
ax_e1.set_ylim(-90,90)
ax_e1.yaxis.set_major_formatter(plt.NullFormatter())
ax_e1.set_xticks(np.arange(0.4, 0.7, 0.2))
ax_e1.set_xticks(np.arange(0.2,0.75, 0.1), minor=True)
ax_e1.set_yticks(np.arange(-90, 90 + 30, 30))
ax_e1.set_yticks(np.arange(-90, 90 + 15, 15), minor=True)
ax_e1.tick_params(axis='both', which='major', length=2, width=0.5, color='black',labelcolor='black', pad=1.5)
ax_e1.tick_params(axis='both', which='minor', length=1, width=0.5, color='black')
ax_e1.spines[['top', 'bottom', 'right', 'left']].set_edgecolor('black')

## ================================================================= figure_1e: longitude mean =================================================================
ax_e2_height = box_fig.width*0.07/box_fig.height
ax_e2 = fig.add_subplot([0.12, box_window3.y0/box_fig.height-ax_e2_height, box_window3['width']/box_fig.width, ax_e2_height], facecolor='None')
ax_e2.plot(lon, lon_AAR);
ax_e2.plot(lon, lon_AAR, color='royalblue', label='Mean');
ax_e2.plot(lon, lon_AAR-lon_AAR_std, linewidth=0);
ax_e2.plot(lon, lon_AAR+lon_AAR_std, linewidth=0);
ax_e2.fill_between(lon, lon_AAR-lon_AAR_std, lon_AAR+lon_AAR_std, color='lightsteelblue', alpha=1, label=r'1$\sigma$')

ax_e2.axhline(y=region_AAR[0], color='dimgrey', linestyle='--', label='Global mean')

ax_e2.set_ylim(0.22,0.78)
ax_e2.set_xlim(-180,180)
ax_e2.xaxis.set_major_formatter(plt.NullFormatter())
ax_e2.set_xticks(np.arange(-180, 180 + 60, 60))
ax_e2.set_xticks(np.arange(-180, 180 + 30, 30), minor=True)
ax_e2.set_yticks(np.arange(0.4, 0.7, 0.2))
ax_e2.set_yticks(np.arange(0.2,0.75, 0.1), minor=True)
ax_e2.tick_params(axis='both', which='major', length=2, width=0.5, color='black',labelcolor='black', pad=1.5,
                  left=False, right=True, labelleft=False, labelright=True)
ax_e2.tick_params(axis='both', which='minor', length=1, width=0.5, color='black', left=False, right=True)
ax_e2.spines[['top', 'bottom', 'right', 'left']].set_edgecolor('black')

#loc='best', bbox_to_anchor=(0.65,0.55,0.2,0.2)
handles, labels = ax_e2.get_legend_handles_labels()
labels[2] = 'Global\nmean'
ax_e2.legend([handles[0],handles[2], handles[1]],[labels[0], labels[2], labels[1]], loc='best', bbox_to_anchor=(0.005,0.8), ncols=1, fontsize=6, frameon=False,
             borderpad=0.3, handlelength=1.5, labelspacing=0.3, handletextpad=0.4, columnspacing=-2);

## ================================================================= figure_1f: Histogram =================================================================
ax_f = fig.add_subplot([0.77,box_window3.y0/box_fig.height,0.19,box_window3['height']/box_fig.height], facecolor='None')
ax_f.text(0, 1.02, 'f', fontsize=7, fontweight='bold', transform=ax_f.transAxes);

## ============== Our study ==============
ax_f.hist(compile_AAR, density=True, bins = np.linspace(0.05, 0.95, 37), alpha=1,
           histtype='stepfilled', color='lightsteelblue', edgecolor='none', orientation='horizontal', label='Histogram');

# Normal
compile_AAR_normx = np.linspace(compile_AAR.min(), compile_AAR.max(), 1000)
compile_AAR_normy = scipy.stats.norm.pdf(compile_AAR_normx, compile_AAR.mean(), compile_AAR.std())
ax_f.plot(compile_AAR_normy, compile_AAR_normx, color='royalblue', linestyle='-', label='Gaussian DIST')

# Mean and Median
ax_f.axhline(y=np.median(compile_AAR), color='royalblue', linestyle='--', label='Median')

ax_f.errorbar(max(compile_AAR_normy), compile_AAR.mean(), fmt='o', yerr=compile_AAR.std(), capsize=1, elinewidth=0.5, capthick=0.5,
              label=r'Mean with 1$\sigma$', c='royalblue', markersize=3)

ax_f.invert_xaxis()
ax_f.set_yticks(np.arange(0.1, 1, 0.1))
ax_f.set_yticks(np.arange(0.05, 1, 0.05), minor=True)
ax_f.set_ylim(0.05, 0.95)

## ============== tick_params ==============
ax_f.tick_params(axis='both', which='major', length=2, width=0.5, color='black',labelcolor='black', pad=1.5,
                  left=False, right=True, labelleft=False, labelright=True, bottom=False, labelbottom=False)
ax_f.tick_params(axis='both', which='minor', length=1, width=0.5, color='black', left=False, right=True, bottom=False)
ax_f.spines[['top', 'bottom', 'right', 'left']].set_edgecolor('black')

#ax_f.legend(loc='upper left', ncols=1, fontsize=6, markerscale=0.7, frameon=False, borderpad=0.3, 
#            labelspacing=0.3, handletextpad=0.4, title='This study:', title_fontsize=6, alignment='left');

## ================================================================= figure_1f: WGMS =================================================================
ax_f1 = fig.add_subplot([0.77,box_window3.y0/box_fig.height,0.19,box_window3['height']/box_fig.height], facecolor='None')

ax_f1.hist(wgms_AAR, density=True, bins = np.linspace(0.05, 0.95, 37), alpha=0.3,
           histtype='stepfilled', color='orange', edgecolor='none', orientation='horizontal', label='Histogram');

# Normal
wgms_AAR_normx = np.linspace(wgms_AAR.min(), wgms_AAR.max(), 1000)
wgms_AAR_normy = scipy.stats.norm.pdf(wgms_AAR_normx, wgms_AAR.mean(), wgms_AAR.std())
ax_f1.plot(wgms_AAR_normy, wgms_AAR_normx, color='orange', linestyle='-', label='Gaussian DIST')

# Mean and Median
ax_f1.axhline(y=np.median(wgms_AAR), color='darkorange', linestyle='--', label='Median')

ax_f1.errorbar(max(wgms_AAR_normy), wgms_AAR.mean(), fmt='o', yerr=wgms_AAR.std(), capsize=1, elinewidth=0.5, capthick=0.5,
               label=r'Mean with 1$\sigma$', c='darkorange', markersize=3)

ax_f1.invert_xaxis()
ax_f1.yaxis.set_major_locator(plt.NullLocator())
ax_f1.xaxis.set_major_locator(plt.NullLocator())
ax_f1.set_ylim(0.05, 0.95)

#ax_f1.legend(loc='lower left', ncols=1, fontsize=6, markerscale=0.7, frameon=False, borderpad=0.3,
#             labelspacing=0.3, handletextpad=0.4, title='WGMS:', title_fontsize=6, alignment='left');

## ================================================================= figure_1f: previous publications =================================================================
ax_f2 = fig.add_subplot([0.77,box_window3.y0/box_fig.height,0.19,box_window3['height']/box_fig.height], facecolor='None')

marker = ['v', '^', 's', 'd']
repub_x = np.linspace(0.6, 0.7, 4)
for i in np.array([0,1,3,2]):
    ax_f2.errorbar(repub_x[i], repub_AAR[0][i], fmt=marker[i], yerr=repub_AAR[1][i], capsize=1, elinewidth=0.5, capthick=0.5,
                   label=repub_label[i], c='black', markersize=3)
    
ax_f2.yaxis.set_major_locator(plt.NullLocator())
ax_f2.xaxis.set_major_locator(plt.NullLocator())
ax_f2.set_ylim(0.05, 0.95)

ax_f2.legend(loc='best', bbox_to_anchor=(0.46,2.46,0.3,0.3), ncols=1, fontsize=6, markerscale=0.7, frameon=False, borderpad=0.3,
             labelspacing=0.5, handletextpad=0.4, title=r'Mean with 1$\sigma$', title_fontsize=6, alignment='left');

#%% output
ax_a.text(0.02, 0.015, '1: Alaska  2: W Canada & US  3: Arctic Canada North  4: Arctic Canada South  5: Greenland Periphery  6: Iceland  7: Svalbard  8: Scandinavia  9: Russian Arctic  10: North Asia', 
          fontsize=5, transform=fig.transFigure);
ax_a.text(0.02, 0.005, '11: Central Europe  12: Caucasus & Middle East 13: Central Asia  14: South Asia West  15: South Asia East  16: Low Latitudes  17: Southern Andes  18: New Zealand  19: Antarctic & Subantarctic', 
          fontsize=5, transform=fig.transFigure);

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/manuscript/' + 'figure_1.png'
plt.savefig(out_pdf, dpi=600)
#box_window1 = ax_a.get_window_extent()

plt.show()
