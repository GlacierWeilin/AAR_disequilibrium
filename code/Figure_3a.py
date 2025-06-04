#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Sep 10 15:34:31 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
'''

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

data = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/Output/ERA5_MCMC_ba1_2014_2023_corrected.nc');
AAR = data['parameterization_AAR'].values[:,0]
AAR_mean = data['parameterization_AAR_mean'].values[:,0]
a = data['parameterization_a'].values[:,0]

total_area = np.sum(data['area_2020'].values[:,0]/1e6)
total = np.shape(data['area_2020'].values[:,0])[0]
area_ratio = np.zeros(4,)
number_ratio = np.zeros(4,)

find_id = np.where(data['area_2020'].values[:,0]/1e6<=1)[0]
area0 = xr.DataArray(np.column_stack((a[find_id], AAR_mean[find_id], AAR[find_id])),
                      coords=[np.arange(0,len(find_id)), ['a','AAR_mean','AAR']], dims=['samples', 'variables'])
area_ratio[0]   = np.sum(data['area_2020'].values[find_id,0]/1e6)/total_area*100
number_ratio[0] = (np.shape(find_id)[0])/total*100

find_id = np.where((data['area_2020'].values[:,0]/1e6>1) & (data['area_2020'].values[:,0]/1e6<=10))[0]
area1 = xr.DataArray(np.column_stack((a[find_id], AAR_mean[find_id], AAR[find_id])),
                     coords=[np.arange(0,len(find_id)), ['a','AAR_mean','AAR']], dims=['samples', 'variables'])
area_ratio[1]   = np.sum(data['area_2020'].values[find_id,0]/1e6)/total_area*100
number_ratio[1] = (np.shape(find_id)[0])/total*100

find_id = np.where((data['area_2020'].values[:,0]/1e6>10) & (data['area_2020'].values[:,0]/1e6<=100))[0]
area10 = xr.DataArray(np.column_stack((a[find_id], AAR_mean[find_id], AAR[find_id])),
                      coords=[np.arange(0,len(find_id)), ['a','AAR_mean','AAR']], dims=['samples', 'variables'])
area_ratio[2]   = np.sum(data['area_2020'].values[find_id,0]/1e6)/total_area*100
number_ratio[2] = (np.shape(find_id)[0])/total*100

find_id = np.where(data['area_2020'].values[:,0]/1e6>100)[0]
area100 = xr.DataArray(np.column_stack((a[find_id], AAR_mean[find_id], AAR[find_id])),
                     coords=[np.arange(0,len(find_id)), ['a','AAR_mean','AAR']], dims=['samples', 'variables'])
area_ratio[3]   = np.sum(data['area_2020'].values[find_id,0]/1e6)/total_area*100
number_ratio[3] = (np.shape(find_id)[0])/total*100

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

# 转换为 DataFrame
categories = ['area0','area1','area10','area100']
frames = []
for cat, data in zip(categories, [area0,area1,area10,area100]):
    df = pd.DataFrame(data.values, columns=data.coords['variables'].values)
    df['Group'] = cat
    frames.append(df)
data = pd.concat(frames)

fig, ax1 = plt.subplots(figsize=(3.5, 2.1), dpi=600)
ax2 = ax1.twinx()

pos1 = ax1.get_position()
ax1.set_position([pos1.x0-0.03, pos1.y0 -0.05, pos1.width + 0.04, pos1.height + 0.15])
ax2.set_position(ax1.get_position())

colors = ['#C93735','#F09137', '#5BBBD0','#5266B0']

width = 0.09
x_positions = np.array([1, 2, 3])
category_offset = 0.09

for i, category in enumerate(categories):
    subset = data[data['Group'] == category]

    aar_data = subset['AAR'].dropna()
    aar_mean_data = subset['AAR_mean'].dropna()
    a_data = subset['a'].dropna()
    
    ax1.bar(
        [x_positions[0] + i * category_offset], 
        [np.percentile(a_data, 95)],
        width=width,
        color=colors[i],
        alpha=0.7,
        edgecolor='none',
    )
    ax1.bar(
        [x_positions[0] + i * category_offset], 
        [np.percentile(a_data, 5)],
        width=width,
        color='w',
        alpha=1,
        edgecolor='w',
    )
    ax1.errorbar(
        [x_positions[0] + i * category_offset], [a_data.mean()],yerr=[a_data.std()],
        marker='o', markersize=2, capsize=1, color=colors[i], linestyle='none'
    )
    ax1.plot(
        [x_positions[0] + i * category_offset], [a_data.median()],
        marker='_', markersize=6, color=colors[i], linestyle='none'
    )
    ax1.set_ylim(0.3, 1.5)
    
    ######
    ax2.bar(
        x_positions[1:] + i * category_offset, 
        [np.percentile(aar_mean_data, 95), np.percentile(aar_data, 95)],
        width=width,
        color=colors[i],
        alpha=0.7,
        edgecolor='none',
    )
    ax2.bar(
        x_positions[1:] + i * category_offset, 
        [np.percentile(aar_mean_data, 5), np.percentile(aar_data, 5)],
        width=width,
        color='w',
        alpha=1,
        edgecolor='w',
    )

    ax2.errorbar(
        x_positions[1:] + i * category_offset, [aar_mean_data.mean(), aar_data.mean()], yerr=[aar_mean_data.std(), aar_data.std()],
        marker='o', markersize=2, capsize=1, color=colors[i], linestyle='none'
    )
    ax2.plot(
        x_positions[1:] + i * category_offset, [aar_mean_data.median(), aar_data.median()],
        marker='_', markersize=6, color=colors[i], linestyle='none'
    )

    ax2.set_xlim(0.7, 3.55)
    ax2.set_ylim(0.05, 0.95)

ax1.set_ylabel('α')
ax2.set_ylabel('AAR')

x_ticks = x_positions + (len(categories) - 1) * category_offset / 2
ax2.set_xticks(x_ticks)
ax2.set_xticklabels(['α', '$\overline{AAR}$', 'AAR$_0$'])


# legend
loc=-1.5
ax2.bar(x_positions[0] - loc, 0.38,width=width-0.02,
        color='dimgrey',alpha=0.7,edgecolor='none')
ax2.bar(x_positions[0] - loc, 0.21,width=width-0.02,
        color='w',alpha=1,edgecolor='w')
ax2.errorbar(
    x_positions[0] - loc, 0.31, yerr=0.05,
    marker='o', markersize=2, capsize=1, color='dimgrey', linestyle='none'
)
ax2.plot(
    x_positions[0] -loc, 0.27,
    marker='_', markersize=4, color='dimgrey', linestyle='none'
)

ax2.text(x_positions[2]+0.47, 0.92, 'a', color='k', fontweight='bold', ha='center', va='center', size=8)
ax2.text(x_positions[0]+1.6, 0.195, '5%', color='dimgrey', ha='left', va='bottom', size=6)
ax2.text(x_positions[0]+1.6, 0.39, '95%', color='dimgrey', ha='left', va='top', size=6)
ax2.text(x_positions[0]+1.6, 0.27, 'median', color='dimgrey', ha='left', va='center', size=6)
ax2.text(x_positions[0]+1.6, 0.31, 'mean ± 1σ', color='dimgrey', ha='left', va='center', size=6)

ax2.text(x_positions[0]+1.45, 0.172, 'Glacier area (km$^2$):', color='k', ha='left', va='center', size=6)

handles = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(len(categories))]
ax2.legend([handles[0],handles[2],handles[1],handles[3]],
                  ['<1 ({:.1f}%)'.format(number_ratio[0]),'10-100 ({:.1f}%)'.format(number_ratio[2]),
                   '1-10 ({:.1f}%)'.format(number_ratio[1]),'>100 ({:.1f}%)'.format(number_ratio[3])],
                  loc='lower left', ncol=2, columnspacing=1,frameon=False, handlelength=1, bbox_to_anchor=(0.48, -0.025), labelspacing=0.05)

axins = inset_axes(ax2, width='20%', height='20%', 
                   bbox_to_anchor=(0.41, -0.21, 0.95, 0.95),
                   bbox_transform=ax2.transAxes,
                   loc='center')

axins.pie(area_ratio, startangle=90, colors=colors, textprops={'fontsize': 6})
axins.text(0, 0, 'Area%', ha='center', va='center', fontsize=6, color='k')
axins.axis('equal')

#plt.tight_layout()
out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/manuscript/' + 'figure_3a.png'
plt.savefig(out_pdf, dpi=600)

plt.show()