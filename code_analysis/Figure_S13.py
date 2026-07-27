import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
global_mt_mask = np.isin(region, regions) & (is_tidewater == 1)
global_mt_area_km2 = np.nansum(rgi_area[global_mt_mask])


# %% add frontal-ablation observation data
obs_id_l = []
missing_id_l = []

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

obs_id = pd.concat(obs_id_l, ignore_index=True)
missing_id = pd.concat(missing_id_l, ignore_index=True)

obs_id = obs_id.drop_duplicates(subset=['RGIId']).copy()
missing_id = missing_id.drop_duplicates(subset=['RGIId']).copy()

obs_id['rgi_area_km2'] = area_by_rgi.reindex(obs_id['RGIId']).values
missing_id['rgi_area_km2'] = area_by_rgi.reindex(missing_id['RGIId']).values


# %% calculate statistics
class_regions = [
    None,
    np.array([1]),
    np.array([3]),
    np.array([4]),
    np.array([5]),
    np.array([7]),
    np.array([9]),
    np.array([17]),
    np.array([19]),
]
x_labels = ['Global'] + [str(reg) for reg in regions]
x = np.arange(len(x_labels))

mt_area_percent_global = []
missing_area_percent_mt = []
mt_count = []
missing_count_percent_mt = []

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
    mt_count_n = np.sum(mt_mask)
    missing_count_n = len(missing_sub)

    mt_area_percent_global.append(mt_area_km2 / global_rgi_area_km2 * 100)
    missing_area_percent_mt.append(
        np.nansum(missing_sub['rgi_area_km2'].values) / mt_area_km2 * 100
    )
    mt_count.append(mt_count_n)
    missing_count_percent_mt.append(missing_count_n / mt_count_n * 100)

mt_area_percent_global = np.array(mt_area_percent_global)
missing_area_percent_mt = np.array(missing_area_percent_mt)
mt_count = np.array(mt_count)
missing_count_percent_mt = np.array(missing_count_percent_mt)


# %% Create plot
fig = plt.figure(figsize=(5.7, 4.5))

gs = GridSpec(
    2, 2, figure=fig,
    left=0.08, right=0.99, bottom=0.08, top=0.96,
    wspace=0.18, hspace=0.10,
    width_ratios=[1, 1], height_ratios=[1, 1]
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])


############################################################################ Figure_S12A
ax_a.bar(
    x,
    mt_area_percent_global,
    width=0.72,
    color='0.55',
    alpha=0.75,
    linewidth=0,
    zorder=10
)

ax_a.set_ylabel('Marine-terminating glacier area\n(% of RGI global glacier area)', fontsize=8)
ax_a.text(0.01, 1.00, 'A', transform=ax_a.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')
ax_a.text(
    0.98, 0.93,
    f'RGI global glacier area\n{global_rgi_area_km2:,.0f} km$^2$',
    transform=ax_a.transAxes,
    fontsize=8,
    ha='right',
    va='top'
)
for i in range(len(x)):
    ax_a.text(
        x[i],
        mt_area_percent_global[i],
        f'{mt_area_percent_global[i]:.1f}',
        fontsize=7,
        ha='center',
        va='bottom'
    )
    
ax_a.set_ylim(0, 45)


############################################################################ Figure_S12B
ax_b.bar(
    x,
    missing_area_percent_mt,
    width=0.72,
    color='0.72',
    alpha=0.75,
    linewidth=0,
    zorder=10
)

ax_b.set_ylabel('"Missing" area fraction (%)', fontsize=8)
ax_b.text(0.01, 1.00, 'B', transform=ax_b.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')
for i in range(len(x)):
    ax_b.text(
        x[i],
        missing_area_percent_mt[i],
        f'{missing_area_percent_mt[i]:.1f}',
        fontsize=7,
        ha='center',
        va='bottom'
    )
    
ax_b.set_ylim(0, 110)


############################################################################ Figure_S12C
ax_c.bar(
    x,
    mt_count,
    width=0.72,
    color='0.55',
    alpha=0.75,
    linewidth=0,
    zorder=10
)

for i in range(len(x)):
    ax_c.text(
        x[i],
        mt_count[i],
        f'{mt_count[i]:.0f}',
        fontsize=6,
        ha='center',
        va='bottom'
    )

ax_c.text(0.01, 1.00, 'C', transform=ax_c.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')
ax_c.set_ylabel('Number of marine-terminating glaciers', fontsize=8)

ax_c.set_ylim(0, 3400)

############################################################################ Figure_S12D
ax_d.bar(
    x,
    missing_count_percent_mt,
    width=0.72,
    color='0.72',
    alpha=0.75,
    linewidth=0,
    zorder=10
)

for i in range(len(x)):
    ax_d.text(
        x[i],
        missing_count_percent_mt[i],
        f'{missing_count_percent_mt[i]:.1f}',
        fontsize=6,
        ha='center',
        va='bottom'
    )

ax_d.text(0.01, 1.00, 'D', transform=ax_d.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')
ax_d.set_ylabel('"Missing" glacier number fraction (%)', fontsize=8)

ax_d.set_ylim(0, 110)


for ax in [ax_a, ax_b, ax_c, ax_d]:
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_xlabel('RGI region', fontsize=8)
    ax.set_xlim(-0.6, len(x_labels) - 0.4)
    ax.tick_params(axis='both', which='major', length=2, width=0.5)
    ax.tick_params(axis='both', which='minor', length=1, width=0.5)

for ax in [ax_a, ax_b]:
    ax.set_xlabel('')
    ax.tick_params(labelbottom=False)

out_png = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_S13.png'
plt.savefig(out_png, dpi=600)

plt.show()
plt.close(fig)
