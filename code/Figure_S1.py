#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:34:31 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/wyan0065/Desktop/AAR-disequilibrium/manuscript/Rainbow_glacier.csv')
data = data.set_index('Year')

data_2014 = data.loc[2014:2023]
x_2014 = data_2014['SMB'].values * 1000;
y_2014 = data_2014['AAR'].values;
slope_2014, intercept_2014, r_value_2014, p_value_2014, std_err_2014 = st.linregress(x_2014, y_2014);

new_x =np.linspace(np.min(x_2014), np.max(x_2014), 100)
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

ax = plt.axes([0.1,0.12,0.85,0.8], xlim=(-3500,600), ylim=(0,0.8))
ax.set_title('RGI60-02.17733: Rainbow Glacier')
ax.set(xlabel='Mass balance (mm w.e.)')
ax.set(ylabel='AAR')

ax.plot(x_2014, y_2014, 'o', markersize=2, color='#489FE3', alpha=1)
ax.plot(new_x, new_x*slope_2014+intercept_2014, linewidth=1, color='#489FE3',label='AAR$_{0}$ = 0.57 (2014-2014)')
ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.5)
ax.axhline(y=0.574, color='grey', linestyle='--', linewidth=0.5)

ax.grid(True, linestyle='-', color='lightgray', linewidth=0.5);
legend = ax.legend(loc='upper left', title='AAR = AAR$_{0}$+k×MB');
legend.get_title().set_fontstyle('italic')
legend.get_frame().set_linewidth(0.5)

out_pdf = '/Users/wyan0065/Desktop/AAR-disequilibrium/manuscript/' + 'figure_S1.png'
plt.savefig(out_pdf, dpi=600)

plt.show()