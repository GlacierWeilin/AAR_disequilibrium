#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:15:36 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

filepath = '/Users/wyan0065/Desktop/PyGEM/calving/Output/';
obs = pd.read_csv(filepath + 'WGMS_disequilibrium_RGI.csv');

result = pd.DataFrame({
    'a': obs['a'].values,
    'AAR_mean': obs['wgms_AAR_mean'].values,
    'AAR': obs['intercept'].values
})

titles = ['Glacier disequilibrium (α)', r'$\mathbf{\overline{AAR}}$ from 2014 to 2023',
          r'Steady-state AAR ($\mathbf{AAR}_0$)']

bounds = np.array([[0.4, 0.15, 0.31], [1.0, 0.39, 0.55], [1.1, 0.63, 0.73], 
                   [0.1,0.06, 0.06]])

labels = ['a', 'b', 'c']

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

fig = plt.figure(figsize=(4.71, 6.8), dpi=600)
gs = GridSpec(3, 1, figure=fig, hspace=0.04, height_ratios=[1,1,1])
plt.subplots_adjust(left=0.06, right=1.07, top=0.99, bottom=0.01)


for i in range(3):
    proj = ccrs.PlateCarree()
    regions_shp='/Users/wyan0065/Desktop/PyGEM/calving/RGI/rgi60/00_rgi60_regions/00_rgi60_O1Regions.shp'
    shape_feature = ShapelyFeature(Reader(regions_shp).geometries(), ccrs.PlateCarree(), edgecolor='white', alpha=1,
                                   facecolor='None', linewidth=0.5)

    ax = fig.add_subplot(gs[i, 0], projection=proj)
    ax.set_global()
    ax.set_title(titles[i], fontweight='bold',loc='center', pad=2)

    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='gainsboro'))
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='dimgrey'), alpha=0.5)

    ax.add_feature(shape_feature)

    ax.set_xticks(np.arange(-180, 180 + 60, 60), crs=proj)
    ax.set_xticks(np.arange(-180, 180 + 30, 30), minor=True, crs=proj)
    ax.set_yticks(np.arange(-90, 90 + 30, 30), crs=proj)
    ax.set_yticks(np.arange(-90, 90 + 15, 15), minor=True, crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis='both', which='major', length=2, width=0.5, color='black', labelcolor='black', pad=1)
    ax.tick_params(axis='both', which='minor', length=1, width=0.5, color='black')
    ax.spines['geo'].set_edgecolor('black') 

    col_bounds = np.linspace(bounds[0,i],bounds[1,i],7)
    col_bounds = np.append(col_bounds, np.linspace(bounds[1,i],bounds[2,i],7))
    cb = []
    cb_val = np.linspace(1, 0, len(col_bounds))
    for j in range(len(cb_val)):
        if i == 0:
            cb.append(mpl.cm.RdBu_r(cb_val[j])) #'RdYlBu_r'
        else:
            cb.append(mpl.cm.RdBu(cb_val[j]))
        cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), 
                                                                          cb)), N=1000)

    norm = mpl.colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))
    name = result.columns[i]
    x = result[name].values
    im=ax.scatter(obs['lon'].values, obs['lat'].values, c=x, s=4,
                   norm=norm, cmap=cmap_cus, zorder=3, transform=ccrs.PlateCarree())
    
    char = fig.colorbar(im, ax=ax, ticks=np.arange(bounds[0,i], bounds[2,i]+bounds[3,i], bounds[3,i]), extend='both',
                        shrink=0.8, aspect=20, pad=0.03, orientation='vertical')

    char.ax.tick_params(direction='in', size=2, width=0.5, labelsize=7)
    
    ax.text(0.01, 1.03, labels[i], transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='center', ha='center')


out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/manuscript/' + 'figure_S4.png'
plt.savefig(out_pdf, dpi=600)

plt.show()