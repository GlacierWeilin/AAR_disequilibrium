#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Oct  8 09:54:05 2025

@author: Weilin Yang (weilinyang.yang@monash.edu)
'''

import numpy as np
import pandas as pd
import xarray as xr

import scipy.stats as st
from scipy.stats import median_abs_deviation
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

file_fp = '/Users/wyan0065/Desktop/PyGEM/calving/Output/';

wgms_data = pd.read_csv(file_fp + '/WGMS_disequilibrium_comparison.csv')
wgms_obs  = wgms_data['a'].values
wgms_mod  = wgms_data['parameterization_a_median'].values

loibl_data = pd.read_csv(file_fp + '/Loibl_disequilibrium_comparison.csv')
loibl_obs  = loibl_data['a'].values
loibl_mod  = loibl_data['parameterization_a_median'].values

data = xr.open_dataset(file_fp+'ERA5_MCMC_ba1_2014_2023_corrected.nc', decode_times=False);
param = data['parameterization_a'].values[:,0]
equil = data['equil_a'].values[:,0]

# previous publications
repub_a   = [[49.2/57.9, 0.44/0.57, 0.68],[0.22, (0.57*0.02-0.44*0.01)/0.57/0.57, 0.12]] # x=a±p, y=b±q; x/y; a/b; (b*p-a*q)/b^2
repub_label = ['Dyurgerov et al.$^{18}$', 'Bahr et al.$^{19}$', 'Mernild et al.$^{20}$']

alpha = 0.01
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

fig = plt.figure(figsize=(7, 2), dpi=600)
gs = GridSpec(1, 3, figure=fig, wspace=0.02, width_ratios=[1,1,1])
plt.subplots_adjust(left=0.02, right=0.99, top=0.93, bottom=0.14)

for i in range(0,3):
    ax = fig.add_subplot(gs[0, i])
    
    if i == 0:
        x = wgms_mod;
        y = wgms_obs;
    elif i == 1:
        x = loibl_mod;
        y = loibl_obs;
    else:
        x = param;
        y = equil;
    
    n = np.shape(x);
    stat, p_value = st.ttest_rel(x, y)
    rmse = mean_squared_error(x, y) ** 0.5
    
    ax.axvline(x=1, color='black', linestyle='-', linewidth=0.5, label='α = 1')
    
    #########################x
    color = '#489FE3'
    
    ax.hist(x, bins = np.linspace(0, 2, 50),edgecolor='white', linewidth=0.5, 
            color=color, alpha=0.6, density=True, label='Histogram')
    
    # Normal
    compile_normx = np.linspace(0, 2, 1000)
    compile_normy = st.norm.pdf(compile_normx, x.mean(), x.std())
    ax.plot(compile_normx, compile_normy, color=color, linestyle='-', linewidth=1, label='Gaussian DIST')

    # Mean and Median
    ax.axvline(x=np.median(x), color=color, linestyle='--', linewidth=1, label='Median')

    ax.errorbar(x.mean(), max(compile_normy), fmt='o', xerr=x.std(), capsize=2, elinewidth=1, capthick=1,
                 c=color, markersize=2)
    
    ax.text(0.66, 0.83,
            f'This study\n'
            f'Mean: {x.mean():.2f} ± {x.std():.2f}\n'
            f'Median: {np.median(x):.2f} ± {median_abs_deviation(x):.2f}',
            color=color, fontsize=6, transform=ax.transAxes
            )
    
    #########################y
    color = '#DC6D57'
    
    ax.hist(y, bins = np.linspace(0, 2, 50),edgecolor='white', linewidth=0.5, 
            color=color, alpha=0.6, density=True)
    
    # Normal
    compile_normx = np.linspace(0, 2, 1000)
    compile_normy = st.norm.pdf(compile_normx, y.mean(), y.std())
    ax.plot(compile_normx, compile_normy, color=color, linestyle='-', linewidth=1)

    # Mean and Median
    ax.axvline(x=np.median(y), color=color, linestyle='--', linewidth=1)

    ax.errorbar(y.mean(), max(compile_normy), fmt='o', xerr=y.std(), capsize=2, elinewidth=1, capthick=1,
                c=color, markersize=2)
    
    label = ['WGMS obs', 'Snowline obs', 'Equilibrium exp']
    ax.text(0.66, 0.65,
        f'{label[i]}\n'
        f'Mean: {y.mean():.2f} ± {y.std():.2f}\n'
        f'Median: {np.median(y):.2f} ± {median_abs_deviation(y):.2f}',
        color=color, fontsize=6, transform=ax.transAxes
        )
    
    #################################
    
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.2, 1.9, 0.2))
    ax.set_xticks(np.arange(0.1, 2.0, 0.1), minor=True)
    ax.set_xlim(0, 2)
    ax.set_xlabel('Glacier disequilibrium (α)')
    if i == 0:
        ax.set_ylabel('Probability Density Function')
    else:
        ax.set_ylabel('')
    
    if p_value < alpha:
        ax.text(0.66, 0.47,
                f'n: {n[0]}\n'
                f'Stat: {stat:.2f}$^*$\n'
                f'RMSE: {rmse:.2f}',
                color='black', fontsize=6, transform=ax.transAxes
                )
    else:
        ax.text(0.66, 0.47,
                f'n: {n[0]}\n'
                f'Stat: {stat:.2f}\n'
                f'RMSE: {rmse:.2f}',
                color='black', fontsize=6, transform=ax.transAxes
                )
    
    if i == 0:
        title = 'Comparison with WGMS observation'
    elif i == 1:
        title = 'Comparison with snowline observation'
    else:
        title = 'Comparison with equilibrium experiment'
    
    if i == 1:
        ax.errorbar(-1, 1, fmt='o', xerr=x.std(), capsize=2, elinewidth=1, capthick=1,
                      label=r'Mean with 1$\sigma$', c='grey', markersize=2)
        
        legend = ax.legend(loc='upper left', ncols=1, fontsize=6, markerscale=0.7, frameon=False, borderpad=-0.1, 
                           labelspacing=0.3, handletextpad=0.4, alignment='left', title= 'Legend');
        legend.get_title().set_fontsize(6)
        
        for text in legend.get_texts()[1:]: 
            text.set_color('gray')
            
        for line in legend.get_lines()[1:]: 
            line.set_color('gray')
            line.set_markerfacecolor('gray')
            line.set_markeredgecolor('gray')
            
        for patch in legend.get_patches():
            patch.set_facecolor('gray')
            patch.set_edgecolor('white')
                

    ax.set_title(title, fontsize=7, color='black', loc='center')
    ax.text(0.03, 1.04, chr(ord('A') + i), transform=ax.transAxes, fontsize=8, ha='center', va='center', fontweight='bold', color='black')
    
ax_repub= fig.add_subplot(gs[0, 0])
ax_repub.set_xticks([])
ax_repub.set_yticks([])
ax_repub.set_xticklabels([])
ax_repub.set_yticklabels([])
ax_repub.spines['top'].set_visible(False)
ax_repub.spines['right'].set_visible(False)
ax_repub.spines['bottom'].set_visible(False)
ax_repub.spines['left'].set_visible(False)
ax_repub.set_frame_on(False)
ax_repub.set_xlim(0, 2)

repub_x = np.linspace(0.6, 0.7, 3)
marker = ['^', 'd', 's']
for i in np.array([0,1,2]):
    ax_repub.errorbar(repub_a[0][i], repub_x[i], fmt=marker[i], xerr=repub_a[1][i], capsize=1, elinewidth=0.5, capthick=0.5,
                   label=repub_label[i], c='black', markersize=2)
    
ax_repub.yaxis.set_major_locator(plt.NullLocator())
ax_repub.xaxis.set_major_locator(plt.NullLocator())
ax_repub.set_ylim(0, 3)

ax_repub.legend(loc='best', bbox_to_anchor=(0.58,0.1,0.3,0.3), ncols=1, fontsize=6, markerscale=0.7, frameon=False, borderpad=-0.1,
             labelspacing=0.3, handletextpad=0.4, title=r'Mean with 1$\sigma$', title_fontsize=6, alignment='left');

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/manuscript/' + 'figure_4.png'
plt.savefig(out_pdf, dpi=600)

plt.show()
