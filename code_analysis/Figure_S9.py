import numpy as np
import pandas as pd

import scipy.stats as st
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

alpha = 0.01

plt.rcParams.update({'lines.linewidth': 0.5})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth': 0.5})
plt.rcParams.update({'axes.titlepad': 3})
plt.rcParams.update({'axes.titlesize': 7})
plt.rcParams.update({'axes.labelpad': 2})
plt.rcParams.update({'xtick.major.pad': 2})
plt.rcParams.update({'ytick.major.pad': 2})
plt.rcParams.update({'xtick.major.width': 0.5})
plt.rcParams.update({'ytick.major.width': 0.5})
plt.rcParams.update({'xtick.major.size': 1.5})
plt.rcParams.update({'ytick.major.size': 1.5})
plt.rcParams['legend.fontsize'] = 6


data_path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/data/'


# %% add data
compare_data = pd.read_csv(
    data_path + 'WGMS_Hugonnet_2000_2020_comparison.csv'
)

compare_data = compare_data[
    (compare_data['first_year'] == 2000) &
    (compare_data['last_year'] == 2020)
].copy()

compare_data = compare_data.dropna(
    subset=['dmdtda', 'wgms_annual_balance_mean']
).copy()


# %% Create plot
fig = plt.figure(figsize=(2.85, 2.26))

gs = GridSpec(1, 1, figure=fig,left=0.15, right=0.98, bottom=0.155, top=0.93)

ax = fig.add_subplot(gs[0, 0])


############################################################################ Paired scatter
x = compare_data['dmdtda'].values
y = compare_data['wgms_annual_balance_mean'].values

stat, p_value = st.ttest_rel(x, y)
wilcoxon_stat, wilcoxon_p_value = st.wilcoxon(x, y)
rmse = mean_squared_error(x, y) ** 0.5

x_median = np.median(x)
y_median = np.median(y)
x_mad = st.median_abs_deviation(x, nan_policy='omit')
y_mad = st.median_abs_deviation(y, nan_policy='omit')

if p_value < alpha:
    p_txt = f'paired t-test: {p_value:.3g}$^*$'
else:
    p_txt = f'paired t-test: {p_value:.2f}'

if wilcoxon_p_value < alpha:
    wilcoxon_p_txt = f'Wilcoxon signed-rank test: {wilcoxon_p_value:.3g}$^*$'
else:
    wilcoxon_p_txt = f'Wilcoxon signed-rank test: {wilcoxon_p_value:.2f}'

ax.scatter(
    x, y,
    s=8,
    color='grey',
    alpha=0.35,
    linewidths=0,
    zorder=5
)

plot_min = np.floor(min(np.nanmin(x), np.nanmin(y)) * 2) / 2
plot_max = np.ceil(max(np.nanmax(x), np.nanmax(y)) * 2) / 2

ax.plot(
    [plot_min, plot_max], [plot_min, plot_max],
    color='black',
    linestyle='--',
    linewidth=0.7,
    zorder=10
)

ax.axvline(x=0, color='grey', linestyle=':', linewidth=0.6, zorder=10)
ax.axhline(y=0, color='grey', linestyle=':', linewidth=0.6, zorder=10)

ax.set_xlim(plot_min, plot_max)
ax.set_ylim(plot_min, plot_max)
ax.set_xticks(np.arange(plot_min, plot_max + 0.5, 0.5))
ax.set_yticks(np.arange(plot_min, plot_max + 0.5, 0.5))

ax.set_xlabel(r'Geodetic mass balance (m w.e. yr$^{-1}$)$^{6}$', fontsize=8)
ax.set_ylabel(r'WGMS mass balance (m w.e. yr$^{-1}$)$^{16}$', fontsize=8)

ax.text(
    0.42, 0.11,
    f'n: {len(x)}\n'
    f'RMSE: {rmse:.2f}',
    color='black',
    fontsize=7,
    transform=ax.transAxes,
    ha='left',
    va='top',
    zorder=100
)

ax.text(
    0.42, 0.33,
    f'p-value:\n'
    f'{p_txt}\n'
    f'{wilcoxon_p_txt}\n',
    color='black',
    fontsize=7,
    transform=ax.transAxes,
    ha='left',
    va='top',
    zorder=100
)

ax.text(
    0.05, 0.92, 'Mean:\nMedian:',
    color='black',
    fontsize=7,
    transform=ax.transAxes,
    ha='left',
    va='top',
    zorder=100
)

ax.text(
    0.19, 0.975,
    f'Geodetic\n'
    f'{x.mean():.2f} ± {x.std():.2f}\n'
    f'{x_median:.2f} ± {x_mad:.2f}\n',
    color='#489FE3',
    fontsize=7,
    transform=ax.transAxes,
    ha='left',
    va='top',
    zorder=100
)

ax.text(
    0.43, 0.975,
    f'WGMS\n'
    f'{y.mean():.2f} ± {y.std():.2f}\n'
    f'{y_median:.2f} ± {y_mad:.2f}\n',
    color='#DC6D57',
    fontsize=7,
    transform=ax.transAxes,
    ha='left',
    va='top',
    zorder=100
)


out_pdf = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_S9.png'
plt.savefig(out_pdf, dpi=600)

plt.show()
