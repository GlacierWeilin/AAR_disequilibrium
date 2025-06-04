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

data = data = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/Output/ERA5_MCMC_ba1_2014_2023_corrected.nc');
AAR = data['parameterization_AAR'].values[:,0]
AAR_mean = data['parameterization_AAR_mean'].values[:,0]
a = data['parameterization_a'].values[:,0]

find_id = np.where(data['is_icecap']==1)[0]
icecap = xr.DataArray(np.column_stack((a[find_id], AAR_mean[find_id], AAR[find_id])),
                      coords=[np.arange(0,len(find_id)), ['a','AAR_mean','AAR']], dims=['samples', 'variables'])

find_id = np.where(data['is_icecap']==0)[0]
glacier = xr.DataArray(np.column_stack((a[find_id], AAR_mean[find_id], AAR[find_id])),
                     coords=[np.arange(0,len(find_id)), ['a','AAR_mean','AAR']], dims=['samples', 'variables'])

find_id = np.where(data['is_debris']==1)[0]
debris = xr.DataArray(np.column_stack((a[find_id], AAR_mean[find_id], AAR[find_id])),
                      coords=[np.arange(0,len(find_id)), ['a','AAR_mean','AAR']], dims=['samples', 'variables'])

find_id = np.where(data['is_debris']==0)[0]
clean = xr.DataArray(np.column_stack((a[find_id], AAR_mean[find_id], AAR[find_id])),
                     coords=[np.arange(0,len(find_id)), ['a','AAR_mean','AAR']], dims=['samples', 'variables'])

find_id = np.where(data['is_tidewater']==1)[0]
marine = xr.DataArray(np.column_stack((a[find_id], AAR_mean[find_id], AAR[find_id])),
                         coords=[np.arange(0,len(find_id)), ['a','AAR_mean','AAR']], dims=['samples', 'variables'])

find_id = np.where(data['is_tidewater']==0)[0]
land = xr.DataArray(np.column_stack((a[find_id], AAR_mean[find_id], AAR[find_id])),
                    coords=[np.arange(0,len(find_id)), ['a','AAR_mean','AAR']], dims=['samples', 'variables'])


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
categories = ['icecap','glacier','debris', 'clean', 'marine', 'land']
frames = []
for cat, data in zip(categories, [icecap, glacier, debris, clean, marine, land]):
    df = pd.DataFrame(data.values, columns=data.coords['variables'].values)
    df['Group'] = cat
    frames.append(df)
data = pd.concat(frames)

fig, ax1 = plt.subplots(figsize=(3.5, 2.1), dpi=600)
ax2 = ax1.twinx()

pos1 = ax1.get_position()
ax1.set_position([pos1.x0-0.03, pos1.y0 -0.05, pos1.width + 0.04, pos1.height + 0.15])
ax2.set_position(ax1.get_position())

colors = ['#C93735', '#E59693', '#5266B0','#B4C3DD','#F09137', '#FAC696',]

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
    
    ax2.bar(
        x_positions[1:] + i * category_offset, 
        [np.percentile(aar_mean_data, 95), np.percentile(aar_data, 95)],
        width=width,
        color=colors[i],
        alpha=0.7,
        edgecolor='none'
    )
    ax2.bar(
        x_positions[1:] + i * category_offset, 
        [np.percentile(aar_mean_data, 5), np.percentile(aar_data, 5)],
        width=width,
        color='w',
        alpha=1,
        edgecolor='w'
    )

    ax2.errorbar(
        x_positions[1:] + i * category_offset, [aar_mean_data.mean(), aar_data.mean()], yerr=[aar_mean_data.std(), aar_data.std()],
        marker='o', markersize=2, capsize=1, color=colors[i], linestyle='none'
    )
    ax2.plot(
        x_positions[1:] + i * category_offset, [aar_mean_data.median(), aar_data.median()],
        marker='_', markersize=6, color=colors[i], linestyle='none'
    )

    ax2.set_xlim(0.8, 3.65)
    ax2.set_ylim(0.05, 0.95)
    
    
ax1.set_ylabel('α')
ax2.set_ylabel('AAR')

x_ticks = x_positions + (len(categories) - 1) * category_offset / 2
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(['α', '$\overline{AAR}$', 'AAR$_0$'])

# legend
# legend
loc=-1.7
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

ax2.text(x_positions[2]+0.57, 0.92, 'b', color='k', fontweight='bold', ha='center', va='center', size=8)
ax2.text(x_positions[0]+1.8, 0.195, '5%', color='dimgrey', ha='left', va='bottom', size=6)
ax2.text(x_positions[0]+1.8, 0.39, '95%', color='dimgrey', ha='left', va='top', size=6)
ax2.text(x_positions[0]+1.8, 0.27, 'median', color='dimgrey', ha='left', va='center', size=6)
ax2.text(x_positions[0]+1.8, 0.31, 'mean ± 1σ', color='dimgrey', ha='left', va='center', size=6)

ax2.text(x_positions[0]+1.65, 0.172, 'Glacier type:', color='k', ha='left', va='center', size=6)

handles = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(len(categories))]
                   
ax2.legend(handles,['Icecap','Glacier','Debris-cover','Clean','Marine-terminating','Land-terminating'],
                  loc='lower left', ncol=3, columnspacing=1,frameon=False, handlelength=1, bbox_to_anchor=(0.25, -0.025), labelspacing=0.05)


out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/manuscript/' + 'figure_3b.png'
plt.savefig(out_pdf, dpi=600)

plt.show()