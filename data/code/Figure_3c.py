#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Sep 10 15:34:31 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
'''

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
from matplotlib.patches import Arc

import pygem.pygem_input as pygem_prms

def add_pie_split_lines(axis, wedges, split_ratios,
                        cs=None, lw=0.8, linestyle='--'):
    """
    """
    ratio = 1.05
    for i, w in enumerate(wedges):
        
        theta = split_ratios[i] * (w.theta2-w.theta1)
        theta = np.deg2rad(w.theta1 + theta)
       
        x0, y0 = w.center
        #x1 = x0 + w.r * np.cos(np.deg2rad(w.theta1)) * ratio
        #y1 = y0 + w.r * np.sin(np.deg2rad(w.theta1)) * ratio
        #axis.plot([x0, x1], [y0, y1], lw=lw, linestyle=linestyle, color=cs[i])
    
        x1 = x0 + w.r * np.cos(theta) * ratio
        y1 = y0 + w.r * np.sin(theta) * ratio
        axis.plot([x0, x1], [y0, y1], lw=lw, linestyle=linestyle, color=cs[i])
    
        arc = Arc(xy=w.center, width=2*w.r * ratio, height=2*w.r * ratio, angle=0,
                  theta1=w.theta1, theta2=np.rad2deg(theta),
                  lw=lw, linestyle=linestyle, color=cs[i])
        axis.add_patch(arc)
        
def autopct_volume(pct, threshold=2):
    return f'{pct:.1f}' if pct >= threshold else ''


def autopct_slr(pct, threshold=2):
    lable = ''
    if pct >= threshold:
        lable = f'{abs(pct * total[5] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000 /100):.1f}'
    return lable

#%%
data = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/Output/ERA5_MCMC_ba1_2014_2023_corrected.nc');

total = np.zeros(6,)
total[0] = np.nansum(data['area_2020'].values[:,0])
total[1] = np.nansum(data['volume_2020'].values[:,0])
total[2] = np.shape(data['area_2020'].values[:,0])[0]
total[3] = np.nansum(data['parameterization_dA'].values[:,0])
total[4] = np.nansum(data['parameterization_dV'].values[:,0])
total[5] = np.nansum(data['parameterization_dV'].values[:,0]) - np.nansum(data['parameterization_dV_bwl'].values[:,0])

data_ratios = np.zeros([4,6,2])

find_id = np.where(data['area_2020'].values[:,0]/1e6<=1)[0]
data_ratios[0,0,0]   = np.nansum(data['area_2020'].values[find_id,0])/total[0]*100
data_ratios[0,1,0]   = np.nansum(data['volume_2020'].values[find_id,0])/total[1]*100
data_ratios[0,2,0]   = np.shape(find_id)[0] / total[2]*100
data_ratios[0,3,0]   = np.nansum(data['parameterization_dA'].values[find_id,0])/total[3]*100
data_ratios[0,4,0]   = np.nansum(data['parameterization_dV'].values[find_id,0])/total[4]*100
data_ratios[0,5,0]   = (np.nansum(data['parameterization_dV'].values[find_id,0]) - 
                        np.nansum(data['parameterization_dV_bwl'].values[find_id,0]))/total[5]*100

find_id = np.where((data['area_2020'].values[:,0]/1e6<=1) & (data['is_tidewater'].values[:]))[0]
data_ratios[0,0,1]   = np.nansum(data['area_2020'].values[find_id,0])/total[0]*100 / data_ratios[0,0,0]
data_ratios[0,1,1]   = np.nansum(data['volume_2020'].values[find_id,0])/total[1]*100 / data_ratios[0,1,0]
data_ratios[0,2,1]   = np.shape(find_id)[0] / total[2]*100 / data_ratios[0,2,0]
data_ratios[0,3,1]   = np.nansum(data['parameterization_dA'].values[find_id,0])/total[3]*100 / data_ratios[0,3,0]
data_ratios[0,4,1]   = np.nansum(data['parameterization_dV'].values[find_id,0])/total[4]*100 / data_ratios[0,4,0]
data_ratios[0,5,1]   = (np.nansum(data['parameterization_dV'].values[find_id,0]) - 
                        np.nansum(data['parameterization_dV_bwl'].values[find_id,0]))/total[5]*100 / data_ratios[0,5,0]


###################
find_id = np.where((data['area_2020'].values[:,0]/1e6>1) & (data['area_2020'].values[:,0]/1e6<=10))[0]
data_ratios[1,0,0]   = np.nansum(data['area_2020'].values[find_id,0])/total[0]*100
data_ratios[1,1,0]   = np.nansum(data['volume_2020'].values[find_id,0])/total[1]*100
data_ratios[1,2,0]   = np.shape(find_id)[0] / total[2]*100
data_ratios[1,3,0]   = np.nansum(data['parameterization_dA'].values[find_id,0])/total[3]*100
data_ratios[1,4,0]   = np.nansum(data['parameterization_dV'].values[find_id,0])/total[4]*100
data_ratios[1,5,0]   = (np.nansum(data['parameterization_dV'].values[find_id,0]) - 
                        np.nansum(data['parameterization_dV_bwl'].values[find_id,0]))/total[5]*100
find_id = np.where((data['area_2020'].values[:,0]/1e6>1) & (data['area_2020'].values[:,0]/1e6<=10) & (data['is_tidewater'].values[:]))[0]
data_ratios[1,0,1]   = np.nansum(data['area_2020'].values[find_id,0])/total[0]*100 / data_ratios[1,0,0]
data_ratios[1,1,1]   = np.nansum(data['volume_2020'].values[find_id,0])/total[1]*100 / data_ratios[1,1,0]
data_ratios[1,2,1]   = np.shape(find_id)[0] / total[2]*100 / data_ratios[1,2,0]
data_ratios[1,3,1]   = np.nansum(data['parameterization_dA'].values[find_id,0])/total[3]*100 / data_ratios[1,3,0]
data_ratios[1,4,1]   = np.nansum(data['parameterization_dV'].values[find_id,0])/total[4]*100 / data_ratios[1,4,0]
data_ratios[1,5,1]   = (np.nansum(data['parameterization_dV'].values[find_id,0]) - 
                        np.nansum(data['parameterization_dV_bwl'].values[find_id,0]))/total[5]*100 / data_ratios[1,5,0]

###################
find_id = np.where((data['area_2020'].values[:,0]/1e6>10) & (data['area_2020'].values[:,0]/1e6<=100))[0]
data_ratios[2,0,0]   = np.nansum(data['area_2020'].values[find_id,0])/total[0]*100
data_ratios[2,1,0]   = np.nansum(data['volume_2020'].values[find_id,0])/total[1]*100
data_ratios[2,2,0]   = np.shape(find_id)[0] / total[2]*100
data_ratios[2,3,0]   = np.nansum(data['parameterization_dA'].values[find_id,0])/total[3]*100
data_ratios[2,4,0]   = np.nansum(data['parameterization_dV'].values[find_id,0])/total[4]*100
data_ratios[2,5,0]   = (np.nansum(data['parameterization_dV'].values[find_id,0]) - 
                        np.nansum(data['parameterization_dV_bwl'].values[find_id,0]))/total[5]*100
find_id = np.where((data['area_2020'].values[:,0]/1e6>10) & (data['area_2020'].values[:,0]/1e6<=100) & (data['is_tidewater'].values[:]))[0]
data_ratios[2,0,1]   = np.nansum(data['area_2020'].values[find_id,0])/total[0]*100 / data_ratios[2,0,0]
data_ratios[2,1,1]   = np.nansum(data['volume_2020'].values[find_id,0])/total[1]*100 / data_ratios[2,1,0]
data_ratios[2,2,1]   = np.shape(find_id)[0] / total[2]*100 / data_ratios[2,2,0]
data_ratios[2,3,1]   = np.nansum(data['parameterization_dA'].values[find_id,0])/total[3]*100 / data_ratios[2,3,0]
data_ratios[2,4,1]   = np.nansum(data['parameterization_dV'].values[find_id,0])/total[4]*100 / data_ratios[2,4,0]
data_ratios[2,5,1]   = (np.nansum(data['parameterization_dV'].values[find_id,0]) - 
                        np.nansum(data['parameterization_dV_bwl'].values[find_id,0]))/total[5]*100 / data_ratios[2,5,0]

###################
find_id = np.where(data['area_2020'].values[:,0]/1e6>100)[0]
data_ratios[3,0,0]   = np.nansum(data['area_2020'].values[find_id,0])/total[0]*100
data_ratios[3,1,0]   = np.nansum(data['volume_2020'].values[find_id,0])/total[1]*100
data_ratios[3,2,0]   = np.shape(find_id)[0] / total[2]*100
data_ratios[3,3,0]   = np.nansum(data['parameterization_dA'].values[find_id,0])/total[3]*100
data_ratios[3,4,0]   = np.nansum(data['parameterization_dV'].values[find_id,0])/total[4]*100
data_ratios[3,5,0]   = (np.nansum(data['parameterization_dV'].values[find_id,0]) - 
                        np.nansum(data['parameterization_dV_bwl'].values[find_id,0]))/total[5]*100
find_id = np.where((data['area_2020'].values[:,0]/1e6>100) & (data['is_tidewater'].values[:]))[0]
data_ratios[3,0,1]   = np.nansum(data['area_2020'].values[find_id,0])/total[0]*100 / data_ratios[3,0,0]
data_ratios[3,1,1]   = np.nansum(data['volume_2020'].values[find_id,0])/total[1]*100 / data_ratios[3,1,0]
data_ratios[3,2,1]   = np.shape(find_id)[0] / total[2]*100 / data_ratios[3,2,0]
data_ratios[3,3,1]   = np.nansum(data['parameterization_dA'].values[find_id,0])/total[3]*100 / data_ratios[3,3,0]
data_ratios[3,4,1]   = np.nansum(data['parameterization_dV'].values[find_id,0])/total[4]*100 / data_ratios[3,4,0]
data_ratios[3,5,1]   = (np.nansum(data['parameterization_dV'].values[find_id,0]) - 
                        np.nansum(data['parameterization_dV_bwl'].values[find_id,0]))/total[5]*100 / data_ratios[3,5,0]

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
plt.rcParams['legend.fontsize'] = 6

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=600)

plt.subplots_adjust(left=0.046, right=0.955, top=0.99, bottom=0.01)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
#ax.set_frame_on(False)
org_colors = ['#C93735','#F09137', '#5BBBD0','#5266B0']
alpha = 0.7
colors = [mcolors.to_rgba(c, alpha=alpha) for c in org_colors]
pctdistance = 0.8
fontsize = 10
width = '45%'

### area ratio
axins = inset_axes(ax, width=width, height=width, 
                   bbox_to_anchor=(-0.33, 0.25, 1, 1),
                   bbox_transform=ax.transAxes,
                   loc='center')

wedges = axins.pie(data_ratios[:,0,0], startangle=90, colors=colors,textprops={'fontsize': 8}, 
                   autopct=lambda pct: autopct_volume(pct), pctdistance=pctdistance)[0]

axins.text(0, 0, 'Area (%)', ha='center', va='center', 
           fontsize=fontsize, color='k')
axins.axis('equal')

split_ratios = data_ratios[:,0,1]
add_pie_split_lines(axins, wedges, split_ratios, cs=org_colors)

### volume ratio
axins = inset_axes(ax, width=width, height=width, 
                   bbox_to_anchor=(0, 0.25, 1, 1),
                   bbox_transform=ax.transAxes,
                   loc='center')

wedges = axins.pie(data_ratios[:,1,0], startangle=90, colors=colors, textprops={'fontsize': 8}, 
                   autopct=lambda pct: autopct_volume(pct), pctdistance=pctdistance)[0]
axins.text(0, 0, 'Volume (%)', ha='center', va='center', 
           fontsize=fontsize, color='k')
axins.axis('equal')

split_ratios = data_ratios[:,1,1]
add_pie_split_lines(axins, wedges, split_ratios, cs=org_colors)

axins.annotate(
    '',
    xy=(-0.05, 0.9),
    xytext=(0.25, 0.78),
    fontsize=6,
    ha='center',
    va='center',
    arrowprops=dict(arrowstyle='<-', color='black', lw=0.3, shrinkA=0)
)
axins.text(0.3, 0.7, '0.7', ha='center', va='center', fontsize=8, color='k')

### number ratio
axins = inset_axes(ax, width=width, height=width, 
                   bbox_to_anchor=(0.33, 0.25, 1, 1),
                   bbox_transform=ax.transAxes,
                   loc='center')

wedges = axins.pie(data_ratios[:,2,0], startangle=90, colors=colors, textprops={'fontsize': 8}, 
                   autopct=lambda pct: autopct_volume(pct), pctdistance=pctdistance)[0]

axins.text(0, 0, 'Number (%)', ha='center', va='center', 
           fontsize=fontsize, color='k')
axins.axis('equal')

split_ratios = data_ratios[:,2,1]
add_pie_split_lines(axins, wedges, split_ratios, cs=org_colors)

axins.annotate(
    '',
    xy=(0.04, 0.9),
    xytext=(-0.25, 0.78),
    fontsize=6,
    ha='center',
    va='center',
    arrowprops=dict(arrowstyle='<-', color='black', lw=0.3, shrinkA=0)
)
axins.text(-0.3, 0.7, '0.5', ha='center', va='center', fontsize=8, color='k')

### area change ratio
axins = inset_axes(ax, width=width, height=width, 
                   bbox_to_anchor=(-0.33, -0.25, 1, 1),
                   bbox_transform=ax.transAxes,
                   loc='center')

wedges = axins.pie(data_ratios[:,3,0], startangle=90, colors=colors, textprops={'fontsize': 8}, 
                   autopct=lambda pct: autopct_volume(pct), pctdistance=pctdistance)[0]

split_ratios = data_ratios[:,3,1]
add_pie_split_lines(axins, wedges, split_ratios, cs=org_colors)

axins.text(0, 0, 'Area change \ncontribution (%)', ha='center', va='center', 
           fontsize=fontsize, color='k')
axins.axis('equal')

### volume change ratio
axins = inset_axes(ax, width=width, height=width, 
                   bbox_to_anchor=(0, -0.25, 1, 1),
                   bbox_transform=ax.transAxes,
                   loc='center')

wedges = axins.pie(data_ratios[:,4,0], startangle=90, colors=colors, textprops={'fontsize': 8}, 
                   autopct=lambda pct: autopct_volume(pct), pctdistance=pctdistance)[0]
axins.text(0, 0, 'Volume change \ncontribution (%)', ha='center', va='center', 
           fontsize=fontsize, color='k')
axins.axis('equal')

split_ratios = data_ratios[:,4,1]
add_pie_split_lines(axins, wedges, split_ratios, cs=org_colors)

axins.annotate(
    '',
    xy=(-0.05, 0.9),
    xytext=(0.25, 0.78),
    fontsize=6,
    ha='center',
    va='center',
    arrowprops=dict(arrowstyle='<-', color='black', lw=0.3, shrinkA=0)
)
axins.text(0.3, 0.7, '0.6', ha='center', va='center', fontsize=8, color='k')

axins.annotate(
    '',
    xy=(-0.6, -0.82),
    xytext=(-0.8, -1.15),
    fontsize=6,
    ha='center',
    va='center',
    arrowprops=dict(arrowstyle='<-', linestyle='--', color=org_colors[3], lw=0.8, shrinkA=0)
)
axins.text(-0.8, -1.2, 'Marine-terminating', ha='center', va='center', fontsize=8, color=org_colors[3])

axins.annotate(
    '',
    xy=(-0.45, 0.92),
    xytext=(-0.6, 1.15),
    fontsize=6,
    ha='center',
    va='center',
    arrowprops=dict(arrowstyle='<-', linestyle='--', color=org_colors[2], lw=0.8, shrinkA=0)
)
axins.text(-0.6, 1.2, 'Marine-terminating', ha='center', va='center', fontsize=8, color=org_colors[2])

axins.annotate(
    '',
    xy=(0.57, 0.79),
    xytext=(0.8, 1.15),
    fontsize=6,
    ha='center',
    va='center',
    arrowprops=dict(arrowstyle='<-', color=colors[3], lw=0.8, shrinkA=0)
)
axins.text(0.8, 1.2, 'Land-terminating', ha='center', va='center', fontsize=8, color=colors[3])

axins.annotate(
    '',
    xy=(-0.78, 0.56),
    xytext=(-1.2, 0.85),
    fontsize=6,
    ha='center',
    va='center',
    arrowprops=dict(arrowstyle='<-', color=colors[2], lw=0.8, shrinkA=0)
)
axins.text(-1.2, 0.9, 'Land-terminating', ha='center', va='center', fontsize=8, color=colors[2])

### sea-level rise
axins = inset_axes(ax, width=width, height=width, 
                   bbox_to_anchor=(0.33, -0.25, 1, 1),
                   bbox_transform=ax.transAxes,
                   loc='center')

wedges = axins.pie(data_ratios[:,5,0], startangle=90, colors=colors, textprops={'fontsize': 8}, 
                   autopct=lambda pct: autopct_slr(pct), pctdistance=pctdistance)[0]
axins.text(0, 0, 'Sea-level rise\n(mm SLE)', ha='center', va='center', 
           fontsize=fontsize, color='k')
axins.axis('equal')

split_ratios = data_ratios[:,5,1]
add_pie_split_lines(axins, wedges, split_ratios, cs=org_colors)

axins.annotate(
    '',
    xy=(-0.05, 0.9),
    xytext=(0.25, 0.78),
    fontsize=6,
    ha='center',
    va='center',
    arrowprops=dict(arrowstyle='<-', color='black', lw=0.3, shrinkA=0)
)
axins.text(0.3, 0.7, '0.9', ha='center', va='center', fontsize=8, color='k')

ax.text(0.02, 0.96, 'C', color='k', fontweight='bold', ha='center', va='center', size=12)

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/manuscript/' + 'figure_3c.png'
plt.savefig(out_pdf, dpi=600)

plt.show()