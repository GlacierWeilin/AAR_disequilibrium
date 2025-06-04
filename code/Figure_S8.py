#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:06:18 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

#%% figure_2:
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import scipy.interpolate

#%% data
experiment = 'intercept_'
# 0-mean, 1-std, 2-5%, 3-17%, 4-median, 5-83%, 6-95%, 7-mad, \
# 8-n, 9-sum(std), 10-mean_std(std), 11-sum_std(std)
file_fp = '/Users/wyan0065/Desktop/PyGEM/calving/Output/';
grid_data = xr.open_dataset(file_fp+'results_2_all.nc', decode_times=False);
area      = grid_data[experiment+'dA'].values[:,:,9]/1e6
area      = np.flip(area, axis=0)
# Regional results
# 0-ELA_mean, 1-AAR_mean, 2-AAR, \
# 3-a, 4-dA, 5-dV, 6-ELA_steady, 7-THAR, 8-dV_bwl, 9-dV_eff, 10-SLR
region_data = xr.open_dataset(file_fp+'results_by_region.nc', decode_times=False);
region_dV   = region_data[experiment+'region'].values[:,5,9]/1e9
region_bwl  = region_data[experiment+'region'].values[:,5,9]/1e9-region_data[experiment+'region'].values[:,9,9]/1e9
region_SLR = region_data[experiment+'region'].values[:,10,9]

lonmin = -179.5; lonmax = 179.5;
latmin = -90; latmax = 90;
extents = [lonmin, lonmax, latmin, latmax]

#%% default parameters for plotting

plt.rcParams.update({'lines.linewidth':0.5})
plt.rcParams.update({'axes.linewidth':0.5})
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'axes.titlesize':7})
plt.rcParams.update({'font.sans-serif': 'Arial'})

#                        0     1    2     3      4    5    6   7    8     9   10  11   12     13   14   15   16    17     18    19
point_lon = np.array([-190, -191,-150, -100,   -83, -50, -33, -5,  16,   40,  83, -5,  35, 121.5,  68, 100, 110,  -48,   153, 102])
point_lat = np.array([ -20, 63  , 41 ,  120,    68, 119,  62,110,  69,116.5,  78, 37,  26,  43.5,   4,19.5, -24,  -52,   -38, -60])

fig_width_inch=7
fig = plt.figure(figsize=(fig_width_inch,4.2))

regions_shp='/Users/wyan0065/Desktop/PyGEM/calving/RGI/rgi60/00_rgi60_regions/00_rgi60_O1Regions.shp'
shape_feature = ShapelyFeature(Reader(regions_shp).geometries(), ccrs.PlateCarree(), edgecolor='white', alpha=1,
                               facecolor='None', linewidth=0.5)

#%% plot
# ========================================================================= extent =========================================================================
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(),facecolor='None')
box_fig = fig.get_window_extent()

ax.set_global()
ax.spines['geo'].set_linewidth(0)

# ======================================================================= background =======================================================================
sub_ax = fig.add_axes([0.05,-0.04,0.88,0.88],projection=ccrs.Robinson())
sub_ax.set_extent(extents, ccrs.Geodetic())

sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='gainsboro'))
sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='dimgrey'), alpha=0.5)

sub_ax.spines['geo'].set_edgecolor('lightgrey')

col_bounds = np.linspace(-2500,0,6)
col_bounds = np.append(col_bounds, np.linspace(0,500,6))
cb = []
cb_val = np.linspace(1, 0, len(col_bounds))
for j in range(len(cb_val)):
    cb.append(mpl.cm.RdBu_r(cb_val[j])) #'RdYlBu_r's
cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), 
                                                                          cb)), N=1000)

norm = mpl.colors.Normalize(vmin=-2500, vmax=500)
_area = area.copy()
_area[area>0] = area[area>0]*30
im  = sub_ax.imshow(_area, extent=extents, transform=ccrs.PlateCarree(), alpha=0.8,
                  norm=norm, cmap=cmap_cus, zorder=2)

cbar = fig.colorbar(im, ax=sub_ax, ticks=np.array([-2500,-2000,-1500,-1000,-500,0,250,500]), extend='both',
                     shrink=0.45, aspect=30, pad=0.02, orientation='horizontal') # horizontal
cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))

cbar.ax.tick_params(direction='in', size=2, width=0.5, labelsize=7, pad=3, labelcolor='black')
cbar.ax.set_title(label=r'   Area change (km$^2$)', y=-3.1, fontsize=7)

sub_ax.add_feature(shape_feature, zorder=10)

# ========================================================================== volume ==========================================================================
ax_volume = fig.add_axes([0,-0.05,1,1.3], projection=ccrs.PlateCarree(),facecolor='None')
ax_volume.spines['geo'].set_linewidth(0)
box_volume = pd.Series([504.0, 252.0, 0.0, 70.56], index=['width', 'height', 'x0', 'y0']);

volume_fun = scipy.interpolate.interp1d([70000, 2], [46,5], kind='linear')
volume_radius = volume_fun(abs(region_dV))
ax_volume.add_patch(mpatches.Circle(xy=[point_lon[0],point_lat[0]], radius=volume_radius[0], facecolor=plt.cm.Blues(0.3), alpha=1, transform=ax.transData,
                                    zorder=30, edgecolor=plt.cm.Greys(0.8), linewidth=0.35))
for i in range(1,20):#len(area_radius)
    ax_volume.add_patch(mpatches.Circle(xy=[point_lon[i],point_lat[i]], radius=volume_radius[i], facecolor=plt.cm.Blues(0.3), alpha=1, transform=ax.transData,
                                          zorder=30, edgecolor=plt.cm.Greys(0.8), linewidth=0.35))

# =========================================================================== bwl ===========================================================================
theta2 = abs(region_bwl/region_dV * 360)+90;
ax_volume.add_patch(mpatches.Wedge(center=[point_lon[0],point_lat[0]], r=volume_radius[0], theta1=90, theta2=theta2[0], facecolor=plt.cm.Blues(0.9), alpha=0.6, transform=ax.transData,
                                   zorder=30, edgecolor=plt.cm.Blues(0.9), linewidth=0.5))
for i in range(1,20):#len(area_radius)
    if theta2[i] != 90:
        ax_volume.add_patch(mpatches.Wedge(center=[point_lon[i],point_lat[i]], r=volume_radius[i], theta1=90, theta2=theta2[i], facecolor=plt.cm.Blues(0.9), alpha=0.6, transform=ax.transData,
                                         zorder=30, edgecolor=plt.cm.Blues(0.9), linewidth=0.5))

# ========================================================================== SLR ==========================================================================
ax_volume.text(point_lon[0],point_lat[0], format(region_SLR[0], '.1f')+' mm', fontsize=8,
               ha='center', va='center', transform=ax.transData, zorder=35);

for i in range(1,20): #len(area_radius)
    if region_SLR[i] >=0.1:
        ax_volume.text(point_lon[i],point_lat[i], format(region_SLR[i], '.1f'), fontsize=6,
                       ha='center', va='center', transform=ax.transData, zorder=35);

# ========================================================================= label =========================================================================
text_label = ['Global', '(1) Alaska', '(2) W Canada & US', '(3) Arctic Canada\n North', '(4) Arctic Canada\n South', '(5) Greenland\n Periphery',\
              '(6) Iceland', '(7) Svalbard', '(8) Scandinavia', '(9) Russian\n Arctic', '(10) North Asia', \
                  '(11) Central\n Europe', '(12) Caucasus\n & Middle East', '(13) Central\n Asia', '(14) South\n Asia West',\
                      '(15) South\n Asia East', '(16) Low Latitudes', '(17) Southern Andes',\
                          '(18) New Zealand','(19) Antarctic & Subantarctic']

ax_volume.text(point_lon[0], point_lat[0] - volume_radius[0]/(box_volume['height']-76)*180, 'Global', fontsize=8, alpha=1, 
               color='black', ha='center', va='top', transform=ax.transData, bbox={'facecolor':'white', 'pad': 1, 'linewidth': 0.2});

text_fun = scipy.interpolate.interp1d([np.max(volume_radius[1:]), np.min(volume_radius[1:])], [83,91], kind='linear')
text_lat = point_lat[1:] - volume_radius[1:]/(box_volume['height'] - text_fun(volume_radius[1:]))*180;
                                          
for i in range(1,20):
    ax_volume.text(point_lon[i], text_lat[i-1], text_label[i], fontsize=6, alpha=1, 
                   color='black', ha='center', va='top', transform=ax.transData, bbox={'facecolor':'white', 'pad': 1, 'linewidth': 0.2});
    
# ========================================================================= legend =========================================================================
ax_legend = fig.add_axes([0,-0.3,1,1.3], projection=ccrs.PlateCarree(), facecolor='None')
ax_legend.spines['geo'].set_linewidth(0)

# volume
ax_legend.add_patch(mpatches.Circle(xy=[128,-118], radius=volume_fun(50/0.9), facecolor=plt.cm.Blues(0.3), alpha=1, transform=ax.transData,
                                    zorder=30, edgecolor=plt.cm.Greys(0.8), linewidth=0.35))
ax_legend.text(128,-118, '-50', fontsize=6,
               ha='center', va='center', transform=ax.transData, zorder=35);

ax_legend.add_patch(mpatches.Circle(xy=[150,-115], radius=volume_fun(5000/0.9), facecolor=plt.cm.Blues(0.3), alpha=1, transform=ax.transData,
                                    zorder=30, edgecolor=plt.cm.Greys(0.8), linewidth=0.35))
ax_legend.text(150,-115, '-5,000', fontsize=6,
               ha='center', va='center', transform=ax.transData, zorder=35);

ax_legend.text(145,-132, 'Total mass change (Gt)', fontsize=7,
               ha='center', va='center', transform=ax.transData, zorder=35);

# bwl
ax_legend.add_patch(mpatches.Circle(xy=[190,-100], radius=volume_fun(3e4), facecolor=plt.cm.Blues(0.3), alpha=1, transform=ax.transData,
                                    zorder=30, edgecolor=plt.cm.Greys(0.8), linewidth=0.35))
ax_legend.add_patch(mpatches.Wedge(center=[190,-100], r=volume_fun(3e4), theta1=90, theta2=270, facecolor=plt.cm.Blues(0.9), alpha=0.6, transform=ax.transData,
                                   zorder=30, edgecolor=plt.cm.Blues(0.9), linewidth=0.5))

ax_legend.text(178,-100, 'dV\n below\n sea\n level', fontsize=6,
               ha='center', va='center', transform=ax.transData, zorder=35);

ax_legend.text(201,-100, 'Effictive\n dV', fontsize=6,
               ha='center', va='center', transform=ax.transData, zorder=35);

#%% output
out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/manuscript/' + 'figure_S8.png'
plt.savefig(out_pdf,dpi=600,transparent=True)

plt.show()
