import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats as st

plt.rcParams.update({'lines.linewidth': 0.5})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth': 0.5})
plt.rcParams.update({'axes.titlepad': 3})
plt.rcParams.update({'axes.titlesize': 8})
plt.rcParams.update({'axes.labelpad': 2})
plt.rcParams.update({'xtick.major.pad': 2})
plt.rcParams.update({'ytick.major.pad': 2})
plt.rcParams.update({'xtick.major.width': 0.5})
plt.rcParams.update({'ytick.major.width': 0.5})
plt.rcParams.update({'xtick.major.size': 1.5})
plt.rcParams.update({'ytick.major.size': 1.5})
plt.rcParams['legend.fontsize'] = 7


path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/pygem_oggm/'
fa_path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/frontalablation_data/analysis/'
tag = 'median_a'
regions = np.array([1, 3, 4, 5, 7, 9, 17, 19])


# %% add data
with xr.open_dataset(path + f'PyGEM_global_glacier_stats_{tag}.nc') as ds:
    rgi_area = ds['rgi_area_km2'].values
    rgi_id = ds['rgi_id'].values
    region = ds['region'].values
    is_tidewater = ds['is_tidewater'].values

area_by_rgi = pd.Series(rgi_area, index=rgi_id)

global_rgi_area_km2 = np.nansum(rgi_area)
global_mt_mask = is_tidewater == 1
global_mt_area_km2 = np.nansum(rgi_area[global_mt_mask])


# %% add frontal-ablation observation data
obs_id_l = []
missing_id_l = []
obs_calving_l = []

for reg in regions:
    obs = pd.read_csv(
        fa_path + f'{reg}-frontalablation_cal_ind.csv'
    )
    missing = pd.read_csv(
        fa_path + f'{reg}-frontalablation_cal_ind-missing.csv'
    )

    obs['region'] = reg
    missing['region'] = reg

    obs_id_l.append(obs[['RGIId', 'region']])
    missing_id_l.append(missing[['RGIId', 'region']])
    obs_calving_l.append(obs[['RGIId', 'region', 'calving_k']])

obs_id = pd.concat(obs_id_l, ignore_index=True)
missing_id = pd.concat(missing_id_l, ignore_index=True)
obs_calving = pd.concat(obs_calving_l, ignore_index=True)

obs_id = obs_id.drop_duplicates(subset=['RGIId']).copy()
missing_id = missing_id.drop_duplicates(subset=['RGIId']).copy()
obs_calving = obs_calving.drop_duplicates(subset=['RGIId']).copy()

obs_id['rgi_area_km2'] = area_by_rgi.reindex(obs_id['RGIId']).values
missing_id['rgi_area_km2'] = area_by_rgi.reindex(missing_id['RGIId']).values


# %% calculate statistics
outside_19_regions = regions[regions != 19]
class_regions = [
    None,
    outside_19_regions,
    np.array([19]),
]
x_labels = ['all', 'outside 19', '19']
x = np.arange(len(x_labels))
x_calving_labels = [str(reg) for reg in regions]
x_calving = np.arange(len(x_calving_labels))

mt_area_percent_global = []
missing_area_percent_mt = []
mt_mean_area = []
mt_std_area = []
mt_median_area = []
mt_mad_area = []
obs_mean_area = []
obs_std_area = []
obs_median_area = []
obs_mad_area = []
missing_mean_area = []
missing_std_area = []
missing_median_area = []
missing_mad_area = []
calving_k_median = []

for reg in regions:
    calving_sub = obs_calving[obs_calving['region'] == reg].copy()
    calving_k_median.append(np.nanmedian(calving_sub['calving_k'].values))

for class_region in class_regions:
    if class_region is None:
        mt_mask = global_mt_mask
        obs_sub = obs_id.copy()
        missing_sub = missing_id.copy()
    else:
        mt_mask = np.isin(region, class_region) & (is_tidewater == 1)
        obs_sub = obs_id[obs_id['region'].isin(class_region)].copy()
        missing_sub = missing_id[missing_id['region'].isin(class_region)].copy()

    mt_area_km2 = np.nansum(rgi_area[mt_mask])

    mt_area_percent_global.append(mt_area_km2 / global_rgi_area_km2 * 100)
    missing_area_percent_mt.append(
        np.nansum(missing_sub['rgi_area_km2'].values) / mt_area_km2 * 100
    )
    mt_mean_area.append(np.nanmean(rgi_area[mt_mask]))
    mt_std_area.append(np.nanstd(rgi_area[mt_mask]))
    mt_median_area.append(np.nanmedian(rgi_area[mt_mask]))
    mt_mad_area.append(st.median_abs_deviation(rgi_area[mt_mask], nan_policy='omit'))
    obs_mean_area.append(np.nanmean(obs_sub['rgi_area_km2'].values))
    obs_std_area.append(np.nanstd(obs_sub['rgi_area_km2'].values))
    obs_median_area.append(np.nanmedian(obs_sub['rgi_area_km2'].values))
    obs_mad_area.append(st.median_abs_deviation(obs_sub['rgi_area_km2'].values, nan_policy='omit'))
    missing_mean_area.append(np.nanmean(missing_sub['rgi_area_km2'].values))
    missing_std_area.append(np.nanstd(missing_sub['rgi_area_km2'].values))
    missing_median_area.append(np.nanmedian(missing_sub['rgi_area_km2'].values))
    missing_mad_area.append(st.median_abs_deviation(missing_sub['rgi_area_km2'].values, nan_policy='omit'))

mt_area_percent_global = np.array(mt_area_percent_global)
missing_area_percent_mt = np.array(missing_area_percent_mt)
mt_mean_area = np.array(mt_mean_area)
mt_std_area = np.array(mt_std_area)
mt_median_area = np.array(mt_median_area)
mt_mad_area = np.array(mt_mad_area)
obs_mean_area = np.array(obs_mean_area)
obs_std_area = np.array(obs_std_area)
obs_median_area = np.array(obs_median_area)
obs_mad_area = np.array(obs_mad_area)
missing_mean_area = np.array(missing_mean_area)
missing_std_area = np.array(missing_std_area)
missing_median_area = np.array(missing_median_area)
missing_mad_area = np.array(missing_mad_area)
calving_k_median = np.array(calving_k_median)


# %% Create plot
fig = plt.figure(figsize=(5.7, 2.26))

gs = GridSpec(1, 2, figure=fig,left=0.07, right=0.99, bottom=0.125, top=0.96, wspace=0.18, width_ratios=[1, 1])

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])


############################################################################ Figure_S15A
ax_a.bar(
    x_calving,
    calving_k_median,
    width=0.72,
    color='0.55',
    alpha=0.75,
    linewidth=0,
    zorder=10
)

ax_a.set_ylabel(r'Median calving rate $k$', fontsize=8)
ax_a.text(0.01, 0.93, 'A', transform=ax_a.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')

for i in range(len(x_calving)):
    ax_a.text(
        x_calving[i],
        calving_k_median[i],
        f'{calving_k_median[i]:.2f}',
        fontsize=6,
        ha='center',
        va='bottom'
    )


############################################################################ Figure_S15B
bar_width = 0.26

ax_b.bar(
    x - bar_width / 2,
    obs_mean_area,
    width=bar_width,
    color='#489FE3',
    alpha=0.68,
    linewidth=0,
    zorder=10,
    label='with observations'
)

for i in range(len(x)):
    ax_b.text(
        x[i] - bar_width / 2,
        obs_mean_area[i],
        f'{obs_mean_area[i]:.0f}',
        fontsize=6,
        ha='center',
        va='bottom'
    )

ax_b.bar(
    x + bar_width / 2,
    missing_mean_area,
    width=bar_width,
    color='#C93735',
    alpha=0.68,
    linewidth=0,
    zorder=10,
    label='missing observations'
)

for i in range(len(x)):
    ax_b.text(
        x[i] + bar_width / 2,
        missing_mean_area[i],
        f'{missing_mean_area[i]:.0f}',
        fontsize=6,
        ha='center',
        va='bottom'
    )

ax_b.text(0.01, 0.93, 'B', transform=ax_b.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')
ax_b.legend(
    loc='upper right',
    frameon=False,
    fontsize=7,
    handletextpad=0.4,
    borderpad=-0.1
)


ax_a.set_xticks(x_calving)
ax_a.set_xticklabels(x_calving_labels, fontsize=7)
ax_a.set_xlabel('Region', fontsize=8)
ax_a.set_xlim(-0.6, len(x_calving_labels) - 0.4)
ax_a.tick_params(axis='both', which='major', length=2, width=0.5)
ax_a.tick_params(axis='both', which='minor', length=1, width=0.5)

ax_b.set_xticks(x)
ax_b.set_xticklabels(x_labels, fontsize=7)
ax_b.set_xlabel('Region', fontsize=8)
ax_b.set_xlim(-0.6, len(x_labels) - 0.4)
ax_b.tick_params(axis='both', which='major', length=2, width=0.5)
ax_b.tick_params(axis='both', which='minor', length=1, width=0.5)

ax_b.set_ylabel('Mean glacier area (km$^2$)', fontsize=8)
ax_b.set_ylim(0, 200)

out_png = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_S16.png'
plt.savefig(out_png, dpi=600)

plt.show()
plt.close(fig)
