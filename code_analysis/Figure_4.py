import numpy as np
import pandas as pd
import xarray as xr

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
pygem_path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/pygem_oggm/'
# %% add WGMS and Loibl-Dussaillant data
wgms_data = pd.read_csv(
    data_path + 'WGMS_disequilibrium_valid.csv'
)

loibl_data = pd.read_csv(
    data_path + 'Loibl_Dussaillant_disequilibrium.csv'
)

wgms_data = wgms_data.dropna(subset=['RGIId', 'disequilibrium']).copy()
loibl_data = loibl_data.dropna(subset=['RGIId', 'disequilibrium']).copy()


# %% add PyGEM disequilibrium data
nc_file = pygem_path + 'PyGEM_global_glacier_stats_median_a.nc'

data = xr.open_dataset(nc_file)
rgi_id = data['rgi_id'].values.astype(str)
pygem_disequilibrium = data['disequilibrium'].isel(experiment=0).values

rgi_area_km2 = data['rgi_area_km2'].values

pygem_data = pd.DataFrame({
    'RGIId': rgi_id,
    'pygem_disequilibrium': pygem_disequilibrium,
    'rgi_area_km2': rgi_area_km2,
})


# %% match by RGIId
wgms_compare = wgms_data.merge(
    pygem_data,
    on='RGIId',
    how='inner'
)

loibl_compare = loibl_data.merge(
    pygem_data,
    on='RGIId',
    how='inner'
)

wgms_compare = wgms_compare.dropna(subset=['disequilibrium', 'pygem_disequilibrium']).copy()
loibl_compare = loibl_compare.dropna(subset=['disequilibrium', 'pygem_disequilibrium']).copy()


# %% Create plot
fig = plt.figure(figsize=(5.7, 2.26))

gs = GridSpec(1, 2, figure=fig,left=0.07, right=0.99, bottom=0.125, top=0.93, wspace=0.03, width_ratios=[1, 1])

ax_c = fig.add_subplot(gs[0, 0])
ax_d = fig.add_subplot(gs[0, 1], sharex=ax_c, sharey=ax_c)

ax_d.set_ylabel('')
ax_d.tick_params(labelleft=False)


############################################################################ Paired scatter
for i, ax in enumerate([ax_c, ax_d]):

    if i == 0:
        compare_data = wgms_compare
        title = 'WGMS observations$^{21}$'
    else:
        compare_data = loibl_compare
        title = 'Loibl-nowline observations$^{35}$'

    x = compare_data['pygem_disequilibrium'].values
    y = compare_data['disequilibrium'].values

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

    ax.plot(
        [0, 2.6], [0, 2.6],
        color='black',
        linestyle='--',
        linewidth=0.7,
        zorder=10
    )

    ax.axvline(x=1, color='black', linestyle=':', linewidth=0.6, zorder=100)
    ax.axhline(y=1, color='black', linestyle=':', linewidth=0.6, zorder=100)

    ax.set_xlim(0, 2.6)
    ax.set_ylim(0, 2.6)
    ax.set_xticks(np.arange(0, 2.6, 0.5))
    ax.set_yticks(np.arange(0, 2.6, 0.5))

    ax.set_xlabel('This study', fontsize=8)

    if i == 0:
        ax.set_ylabel('Observation', fontsize=8)

    ax.text(0.01, 0.99, chr(ord('A') + i), transform=ax.transAxes,
            fontsize=9, ha='left', va='top',
            fontweight='bold', color='black')
    
    ax.set_title(title, fontsize=7, color='black',
                 loc='center', fontweight='bold', pad=4)

    ax.text(
        0.46, 0.11,
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
        0.46, 0.33,
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
        f'This study\n'
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
        f'Observation\n'
        f'{y.mean():.2f} ± {y.std():.2f}\n'
        f'{y_median:.2f} ± {y_mad:.2f}\n',
        color='#DC6D57',
        fontsize=7,
        transform=ax.transAxes,
        ha='left',
        va='top',
        zorder=100
    )


out_png = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_4.png'
plt.savefig(out_png, dpi=600)

plt.show()
