import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

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
tag = 'median_k'
regions = np.array([1, 3, 4, 5, 7, 9, 17, 19])


# %% add data
with xr.open_dataset(path + f'PyGEM_global_glacier_stats_{tag}.nc') as ds:
    rgi_id = ds['rgi_id'].values
    gcm = ds['gcm'].values.astype(str)
    experiment = ds['experiment'].values
    AAR_steady = ds['AAR_steady'].values

rgi_index = pd.Series(np.arange(len(rgi_id)), index=rgi_id)


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


# %% calculate statistics
class_regions = [
    None,
    regions[regions != 19],
    np.array([19]),
]

class_labels = ['Global', 'outside 19', '19']
x = np.arange(len(class_labels))
x_labels = class_labels

stats_l = []

for class_label, class_region in zip(class_labels, class_regions):
    if class_region is None:
        obs_sub = obs_id.copy()
        missing_sub = missing_id.copy()
    else:
        obs_sub = obs_id[obs_id['region'].isin(class_region)].copy()
        missing_sub = missing_id[missing_id['region'].isin(class_region)].copy()

    for obs_class, sub in zip(
        ['with observations', 'missing observations'],
        [obs_sub, missing_sub]
    ):
        idx = rgi_index.reindex(sub['RGIId']).dropna().astype(int).values

        for exp_idx, exp in enumerate(experiment):
            values = AAR_steady[idx, exp_idx]
            values = values[np.isnan(values) == False]

            stats_l.append({
                'class_label': class_label,
                'obs_class': obs_class,
                'experiment': exp,
                'gcm': gcm[exp_idx],
                'AAR_steady_mean': np.nanmean(values),
                'AAR_steady_std': np.nanstd(values),
                'AAR_steady_median': np.nanmedian(values),
                'AAR_steady_median_std': np.nanstd(values),
                'n': len(values),
            })

stats = pd.DataFrame(stats_l)


# %% Create plot
fig = plt.figure(figsize=(5.7, 2.26))

gs = GridSpec(1, 2, figure=fig,left=0.07, right=0.99, bottom=0.125, top=0.96, wspace=0.03, width_ratios=[1, 1])

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)

ax_b.set_ylabel('')
ax_b.tick_params(labelleft=False)

colors = {
    'with observations': '#489FE3',
    'missing observations': '#C93735',
}


############################################################################ Figure_S14A
offsets = {
    'with observations': -0.10,
    'missing observations': 0.10,
}

for obs_class in ['with observations', 'missing observations']:
    obs_stats = stats[stats['obs_class'] == obs_class].copy()

    for exp in experiment:
        sub = (
            obs_stats[obs_stats['experiment'] == exp]
            .set_index('class_label')
            .reindex(class_labels)
        )

        if sub['gcm'].dropna().iloc[0] == 'era5':
            alpha = 0.85
            markersize = 3.0
            markeredgecolor = 'black'
            markerfacecolor = colors[obs_class]
            zorder = 30
        else:
            alpha = 0.18
            markersize = 2.4
            markeredgecolor = colors[obs_class]
            markerfacecolor = colors[obs_class]
            zorder = 10

        ax_a.errorbar(
            x + offsets[obs_class],
            sub['AAR_steady_mean'].values,
            yerr=sub['AAR_steady_std'].values,
            fmt='o',
            color=colors[obs_class],
            ecolor=colors[obs_class],
            elinewidth=0.35,
            capsize=0.8,
            capthick=0.35,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor,
            markeredgewidth=0.25,
            alpha=alpha,
            zorder=zorder
        )

ax_a.set_xlabel('RGI region', fontsize=8)
ax_a.set_ylabel(r'AAR$_0$', fontsize=8)
ax_a.text(0.01, 0.93, 'A', transform=ax_a.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')


############################################################################ Figure_S14B
for obs_class in ['with observations', 'missing observations']:
    obs_stats = stats[stats['obs_class'] == obs_class].copy()

    for exp in experiment:
        sub = (
            obs_stats[obs_stats['experiment'] == exp]
            .set_index('class_label')
            .reindex(class_labels)
        )

        if sub['gcm'].dropna().iloc[0] == 'era5':
            alpha = 0.85
            markersize = 2.8
            markeredgecolor = 'black'
            markerfacecolor = colors[obs_class]
            zorder = 30
        else:
            alpha = 0.18
            markersize = 2.2
            markeredgecolor = colors[obs_class]
            markerfacecolor = colors[obs_class]
            zorder = 10

        ax_b.errorbar(
            x + offsets[obs_class],
            sub['AAR_steady_median'].values,
            yerr=sub['AAR_steady_median_std'].values,
            fmt='s',
            color=colors[obs_class],
            ecolor=colors[obs_class],
            elinewidth=0.35,
            capsize=0.8,
            capthick=0.35,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor,
            markeredgewidth=0.25,
            alpha=alpha,
            zorder=zorder
        )

ax_b.set_xlabel('RGI region', fontsize=8)
ax_b.text(0.01, 0.93, 'B', transform=ax_b.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')


##############################################################################
### Legend
ax_a.text(1, 0.94, 'Mean ± std', fontsize=8,
          fontweight='bold', ha='center', va='top')
ax_b.text(1, 0.94, 'Median ± MAD', fontsize=8,
          fontweight='bold', ha='center', va='top')

handles = [
    Line2D([0], [0], marker='o', color=colors['with observations'],
           markeredgecolor='black', markeredgewidth=0.25, linewidth=0.8,
           markersize=3.2, label='with observations'),
    Line2D([0], [0], marker='o', color=colors['missing observations'],
           markeredgecolor='black', markeredgewidth=0.25, linewidth=0.8,
           markersize=3.2, label='missing observations'),
    Line2D([0], [0], marker='o', color='black',
           markerfacecolor='0.55', markeredgecolor='black',
           markeredgewidth=0.35, linewidth=0,
           markersize=3.2, label='ERA5'),
    Line2D([0], [0], marker='o', color='0.55',
           markerfacecolor='0.55', markeredgecolor='0.55',
           markeredgewidth=0.25, linewidth=0, alpha=0.35,
           markersize=3.2, label='GCMs'),
]

ax_b.legend(
    handles=handles,
    loc='lower right',
    ncols=2,
    frameon=False,
    fontsize=7,
    handletextpad=0.4,
    borderpad=-0.1
)


for ax in [ax_a, ax_b]:
    ax.set_xlim(-0.6, len(x_labels) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.3, 0.95)
    ax.set_yticks([0.3, 0.5, 0.7, 0.9])


out_png = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_S17.png'
plt.savefig(out_png, dpi=600)

plt.show()
