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
a      = grid_data[experiment+'a'].values[:,:,0]

a        = np.flip(a, axis=0)

lonmin = -179.5; lonmax = 179.5;
latmin = -90; latmax = 90;
extents = [lonmin, lonmax, latmin, latmax]

# Regional results
# 0-ELA_mean, 1-AAR_mean, 2-AAR, \
# 3-a, 4-dA, 5-dV, 6-ELA_steady, 7-THAR, 8-dV_bwl, 9-dV_eff, 10-SLR
region_data     = xr.open_dataset(file_fp+'results_by_region.nc', decode_times=False);
region_a      = region_data[experiment+'region'].values[:,3,0]

# latlon mean
latlon_mean = xr.open_dataset(file_fp+'results_2_latlon_mean.nc', decode_times=False);
lat = latlon_mean['latitude']
lon = latlon_mean['longitude']
lat_a = latlon_mean[experiment+'lat_a'].values[:,0]
lon_a = latlon_mean[experiment+'lon_a'].values[:,0]
lat_a_std = latlon_mean[experiment+'lat_a'].values[:,1]
lon_a_std = latlon_mean[experiment+'lon_a'].values[:,1]

# Histogram
data = xr.open_dataset(file_fp+'ERA5_MCMC_ba1_2014_2023_corrected.nc', decode_times=False);
compile_a = data[experiment+'a'].values[:,0]

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
fig = plt.figure(figsize=(fig_width_inch, 3.1), dpi=600)
box_fig = fig.get_window_extent()

regions_shp='/Users/wyan0065/Desktop/PyGEM/calving/RGI/rgi60/00_rgi60_regions/00_rgi60_O1Regions.shp'
shape_feature = ShapelyFeature(Reader(regions_shp).geometries(), ccrs.PlateCarree(), edgecolor='white', alpha=1,
                               facecolor='None', linewidth=0.5)
#%% plot
## ======================================================================== figure_1a ======================================================================== 
ax_a = fig.add_subplot([0.12,0.27,0.7,0.7], projection=proj)
box_window1 = pd.Series([2469.5999999999995, 1234.8000000000002, 504.0,535.8], index=['width', 'height', 'x0', 'y0']);
ax_a.set_global()
ax_a.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='gainsboro'))
ax_a.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='dimgrey', alpha=0.5))
ax_a.set_title('Glacier disequilibrium (α)', fontsize=7, fontweight='bold',loc='center', pad=3)
ax_a.text(0, 1.02, 'A', fontsize=9, fontweight='bold', transform=ax_a.transAxes);
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
ax_b.text(0, 1.02, 'B', fontsize=9, fontweight='bold', transform=ax_b.transAxes);

## ============== Our study ==============
ax_b.hist(compile_a, density=True, bins = np.linspace(0.1, 2, 40), alpha=1,
          color='gainsboro', edgecolor='white', linewidth=0.5, orientation='horizontal', label='Histogram');

# Normal
compile_a_normx = np.linspace(compile_a.min(), compile_a.max(), 1000)
compile_a_normy = scipy.stats.norm.pdf(compile_a_normx, compile_a.mean(), compile_a.std())
ax_b.plot(compile_a_normy, compile_a_normx, color='dimgrey', linestyle='-', label='Gaussian DIST')

# Mean and Median
ax_b.axhline(y=np.median(compile_a), color='dimgrey', linestyle='--', label='Median')

ax_b.errorbar(max(compile_a_normy), compile_a.mean(), fmt='o', yerr=compile_a.std(), capsize=1, elinewidth=0.5, capthick=0.5,
              label=r'Mean with 1$\sigma$', c='dimgrey', markersize=3)


ax_b.axhline(y=1, color='red', linestyle=':', linewidth = 0.8, label='α = 1')

ax_b.invert_xaxis()
ax_b.set_yticks(np.arange(0, 2.1, 0.2))
ax_b.set_yticks(np.arange(0, 2.1, 0.1), minor=True)
ax_b.set_ylim(0, 2)

## ============== tick_params ==============
ax_b.tick_params(axis='both', which='major', length=2, width=0.5, color='black',labelcolor='black', pad=1.5,
                  left=False, right=True, labelleft=False, labelright=True, bottom=False, labelbottom=False)
ax_b.tick_params(axis='both', which='minor', length=1, width=0.5, color='black', left=False, right=True, bottom=False)
ax_b.spines[['top', 'bottom', 'right', 'left']].set_edgecolor('black')

ax_b.legend(loc='upper left', ncols=1, fontsize=6, markerscale=0.7, frameon=False, borderpad=0.3, 
            labelspacing=0.3, handletextpad=0.4, alignment='left');

#%% output
ax_a.text(0.02, 0.045, '1: Alaska  2: W Canada & US  3: Arctic Canada North  4: Arctic Canada South  5: Greenland Periphery  6: Iceland  7: Svalbard  8: Scandinavia  9: Russian Arctic  10: North Asia', 
          fontsize=5.5, transform=fig.transFigure);
ax_a.text(0.02, 0.02, '11: Central Europe  12: Caucasus & Middle East 13: Central Asia  14: South Asia West  15: South Asia East  16: Low Latitudes  17: Southern Andes  18: New Zealand  19: Antarctic & Subantarctic', 
          fontsize=5.5, transform=fig.transFigure);

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/manuscript/' + 'figure_1.png'
plt.savefig(out_pdf, dpi=600)

plt.show()
