import pandas as pd
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


var = 'mass_remaining'
path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/pygem_oggm/'


# %% add global data for Figure_6A
pygem_a = pd.read_csv(
    path + 'PyGEM_global_mass_median_a.csv'
)

pygem_k = pd.read_csv(
    path + 'PyGEM_global_mass_median_k.csv'
)

pygem_a_lowess = pd.read_csv(
    path + f'PyGEM_global_mass_{var}_median_a_lowess_fit.csv'
)

pygem_k_lowess = pd.read_csv(
    path + f'PyGEM_global_mass_{var}_median_k_lowess_fit.csv'
)


# %% add MT data for Figure_6B-D
mt_a_l = []
mt_k_l = []
for cla in ['all', 'outAntarc', 'inAntarc']:
    _df = pd.read_csv(
        path + f'PyGEM_glacier_mass_MT_{cla}_median_a.csv'
    )
    _df['region_class'] = f'MT_{cla}'
    mt_a_l.append(_df)

    _df = pd.read_csv(
        path + f'PyGEM_glacier_mass_MT_{cla}_median_k.csv'
    )
    _df['region_class'] = f'MT_{cla}'
    mt_k_l.append(_df)

mt_a = pd.concat(mt_a_l, ignore_index=True)
mt_k = pd.concat(mt_k_l, ignore_index=True)

tidewater_result_l = []
for cla in ['land-terminating', 'marine-terminating']:
    _df = pd.read_csv(
        path + f'PyGEM_glacier_mass_{cla}_median_a.csv'
    )
    _df['tidewater_class'] = cla
    tidewater_result_l.append(_df)

tidewater_result = pd.concat(tidewater_result_l, ignore_index=True)

mt_a = mt_a.dropna(subset=['region_class', 'temp_ch_ipcc', var]).copy()
mt_k = mt_k.dropna(subset=['region_class', 'temp_ch_ipcc', var]).copy()


# %% add lowess fit data for MT classes
mt_a_lowess_l = []
mt_k_lowess_l = []
for cla in ['all', 'outAntarc', 'inAntarc']:
    _df = pd.read_csv(
        path + f'PyGEM_glacier_mass_MT_{cla}_{var}_median_a_lowess_fit.csv'
    )
    _df['region_class'] = f'MT_{cla}'
    mt_a_lowess_l.append(_df)

    _df = pd.read_csv(
        path + f'PyGEM_glacier_mass_MT_{cla}_{var}_median_k_lowess_fit.csv'
    )
    _df['region_class'] = f'MT_{cla}'
    mt_k_lowess_l.append(_df)

mt_a_lowess = pd.concat(mt_a_lowess_l, ignore_index=True)
mt_k_lowess = pd.concat(mt_k_lowess_l, ignore_index=True)

if 'variable' in mt_a_lowess.columns:
    mt_a_lowess = mt_a_lowess[mt_a_lowess['variable'] == var].copy()
elif 'y_col' in mt_a_lowess.columns:
    mt_a_lowess = mt_a_lowess[mt_a_lowess['y_col'] == var].copy()

if 'variable' in mt_k_lowess.columns:
    mt_k_lowess = mt_k_lowess[mt_k_lowess['variable'] == var].copy()
elif 'y_col' in mt_k_lowess.columns:
    mt_k_lowess = mt_k_lowess[mt_k_lowess['y_col'] == var].copy()

median_col = 'median' if 'median' in mt_a_lowess.columns else '0.5'
q17_col = 'q17' if 'q17' in mt_a_lowess.columns else '0.17'
q83_col = 'q83' if 'q83' in mt_a_lowess.columns else '0.83'


# %% classes
region_order = [
    'MT_all',
    'MT_outAntarc',
    'MT_inAntarc',
]

region_titles = [
    'Global marine-terminating glaciers',
    'Marine-terminating glaciers outside\nAntarctic & Subantarctic',
    'Marine-terminating glaciers in\nAntarctic & Subantarctic',
]

panel_labels = ['B', 'C', 'D']

colors_b = ['#489FE3', '#C93735']

tidewater_summary = (
    tidewater_result
    .drop_duplicates('tidewater_class')
    .set_index('tidewater_class')
)

total_global_vol_2020 = tidewater_summary['vol_2020_m3'].sum(skipna=True)

mt_class_summary = (
    mt_a
    .drop_duplicates('region_class')
    .set_index('region_class')
    .reindex(region_order)
)

mt_class_summary['vol_percent_global'] = (
    mt_class_summary['vol_2020_m3'] / total_global_vol_2020 * 100
)


# %% Create plot
fig = plt.figure(figsize=(5.7, 4.5))

gs = GridSpec(
    2, 2, figure=fig,
    left=0.07, right=0.99, bottom=0.08, top=0.96,
    wspace=0.03, hspace=0.18,
    width_ratios=[1, 1], height_ratios=[1, 1]
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
ax_c = fig.add_subplot(gs[1, 0], sharey=ax_a)
ax_d = fig.add_subplot(gs[1, 1], sharey=ax_a)

ax_b.set_ylabel('')
ax_d.set_ylabel('')
ax_b.tick_params(labelleft=False)
ax_d.tick_params(labelleft=False)

for ax in [ax_a, ax_b, ax_c, ax_d]:
    ax.set_xlim(-0.1, 6.85)
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 50, 100, 150])


############################################################################ Figure_6A
pygem_a_era5 = pygem_a[pygem_a['gcm'].astype(str).str.lower() == 'era5']
pygem_a_gcm = pygem_a[pygem_a['gcm'].astype(str).str.lower() != 'era5']

pygem_k_era5 = pygem_k[pygem_k['gcm'].astype(str).str.lower() == 'era5']
pygem_k_gcm = pygem_k[pygem_k['gcm'].astype(str).str.lower() != 'era5']

ax_a.scatter(
    pygem_a_gcm['temp_ch_ipcc'].values,
    pygem_a_gcm[var].values,
    s=8,
    color=colors_b[0],
    alpha=0.42,
    linewidths=0,
    zorder=5
)

ax_a.scatter(
    pygem_a_era5['temp_ch_ipcc'].values,
    pygem_a_era5[var].values,
    marker='*',
    s=38,
    color=colors_b[0],
    edgecolor='black',
    linewidths=0.35,
    alpha=0.95,
    zorder=30
)

ax_a.plot(
    pygem_a_lowess['temp_ch_ipcc'].values,
    pygem_a_lowess['0.5'].values,
    color=colors_b[0],
    linewidth=1,
    zorder=10
)

ax_a.fill_between(
    pygem_a_lowess['temp_ch_ipcc'].values,
    pygem_a_lowess['0.17'].values,
    pygem_a_lowess['0.83'].values,
    color=colors_b[0],
    alpha=0.15,
    linewidth=0,
    zorder=1
)

ax_a.scatter(
    pygem_k_gcm['temp_ch_ipcc'].values,
    pygem_k_gcm[var].values,
    s=8,
    color=colors_b[1],
    alpha=0.42,
    linewidths=0,
    zorder=5
)

ax_a.scatter(
    pygem_k_era5['temp_ch_ipcc'].values,
    pygem_k_era5[var].values,
    marker='*',
    s=38,
    color=colors_b[1],
    edgecolor='black',
    linewidths=0.35,
    alpha=0.95,
    zorder=30
)

ax_a.plot(
    pygem_k_lowess['temp_ch_ipcc'].values,
    pygem_k_lowess['0.5'].values,
    color=colors_b[1],
    linewidth=1,
    zorder=10
)

ax_a.fill_between(
    pygem_k_lowess['temp_ch_ipcc'].values,
    pygem_k_lowess['0.17'].values,
    pygem_k_lowess['0.83'].values,
    color=colors_b[1],
    alpha=0.15,
    linewidth=0,
    zorder=1
)

ax_a.set_title('Global glaciers', fontsize=7, fontweight='bold', loc='center', pad=3)

ax_a.set_xlabel('ΔT (°C)', fontsize=8)
ax_a.text(0.01, 1.00, 'A', transform=ax_a.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')

ax_a.text(3.5, 144, 'Method for marine-terminating glaciers missing \nfrontal-ablation observations:', fontsize=7,
          fontweight='bold', ha='center', va='top')

ax_a.text(3.5, 118, r'use median $\alpha$', color=colors_b[0],
          fontsize=7, ha='center', va='top')
ax_a.text(3.5, 126, 'use median $k$', color=colors_b[1],
          fontsize=7, ha='center', va='top')

##############################################################################
### Legend
ax_a.scatter(
    [3.9], [88],
    marker='o',
    s=8,
    color='black',
    alpha=0.42,
    linewidths=0,
    zorder=100
)
ax_a.text(4.1, 88, 'based on GCMs', fontsize=7, ha='left', va='center')

ax_a.scatter(
    [3.9], [80],
    marker='*',
    s=38,
    color='white',
    edgecolor='black',
    linewidths=0.35,
    zorder=100
)
ax_a.text(4.1, 80, 'based on ERA5', fontsize=7, ha='left', va='center')

ax_a.fill_between(
    [3.8, 4.2], [62, 62], [72, 72],
    color='black',
    alpha=0.15,
    linewidth=0
)
ax_a.plot(
    [3.8, 4.2], [67, 67],
    color='black',
    linewidth=1,
    zorder=100
)
ax_a.text(4.3, 67, 'LOWESS fit\n50% [17% to 83%]',
          fontsize=7, ha='left', va='center')


############################################################################ Figure_6B-D
for ax, cls, title, panel in zip([ax_b, ax_c, ax_d], region_order, region_titles, panel_labels):

    mt_a_sub = mt_a[mt_a['region_class'] == cls].sort_values('temp_ch_ipcc')
    mt_a_era5 = mt_a_sub[mt_a_sub['gcm'].astype(str).str.lower() == 'era5']
    mt_a_gcm = mt_a_sub[mt_a_sub['gcm'].astype(str).str.lower() != 'era5']

    mt_k_sub = mt_k[mt_k['region_class'] == cls].sort_values('temp_ch_ipcc')
    mt_k_era5 = mt_k_sub[mt_k_sub['gcm'].astype(str).str.lower() == 'era5']
    mt_k_gcm = mt_k_sub[mt_k_sub['gcm'].astype(str).str.lower() != 'era5']

    mt_a_lowess_sub = mt_a_lowess[
        mt_a_lowess['region_class'] == cls
    ].sort_values('temp_ch_ipcc')

    mt_k_lowess_sub = mt_k_lowess[
        mt_k_lowess['region_class'] == cls
    ].sort_values('temp_ch_ipcc')

    ax.scatter(
        mt_a_gcm['temp_ch_ipcc'].values,
        mt_a_gcm[var].values,
        s=8,
        color=colors_b[0],
        alpha=0.42,
        linewidths=0,
        zorder=5
    )

    ax.scatter(
        mt_a_era5['temp_ch_ipcc'].values,
        mt_a_era5[var].values,
        marker='*',
        s=38,
        color=colors_b[0],
        edgecolor='black',
        linewidths=0.35,
        alpha=0.95,
        zorder=30
    )

    ax.plot(
        mt_a_lowess_sub['temp_ch_ipcc'].values,
        mt_a_lowess_sub[median_col].values,
        color=colors_b[0],
        linewidth=1,
        zorder=10
    )

    ax.fill_between(
        mt_a_lowess_sub['temp_ch_ipcc'].values,
        mt_a_lowess_sub[q17_col].values,
        mt_a_lowess_sub[q83_col].values,
        color=colors_b[0],
        alpha=0.15,
        linewidth=0,
        zorder=1
    )

    ax.scatter(
        mt_k_gcm['temp_ch_ipcc'].values,
        mt_k_gcm[var].values,
        s=8,
        color=colors_b[1],
        alpha=0.42,
        linewidths=0,
        zorder=5
    )

    ax.scatter(
        mt_k_era5['temp_ch_ipcc'].values,
        mt_k_era5[var].values,
        marker='*',
        s=38,
        color=colors_b[1],
        edgecolor='black',
        linewidths=0.35,
        alpha=0.95,
        zorder=30
    )

    ax.plot(
        mt_k_lowess_sub['temp_ch_ipcc'].values,
        mt_k_lowess_sub[median_col].values,
        color=colors_b[1],
        linewidth=1,
        zorder=10
    )

    ax.fill_between(
        mt_k_lowess_sub['temp_ch_ipcc'].values,
        mt_k_lowess_sub[q17_col].values,
        mt_k_lowess_sub[q83_col].values,
        color=colors_b[1],
        alpha=0.15,
        linewidth=0,
        zorder=1
    )
    
    ax.set_title(title, fontsize=7, fontweight='bold', loc='center', pad=3)

    ax.set_xlabel('ΔT (°C)', fontsize=8)
    ax.text(0.01, 1.00, panel, transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='left', va='bottom')

    n = int(mt_class_summary.loc[cls, 'n_glaciers'])
    p = float(mt_class_summary.loc[cls, 'vol_percent_global'])
    ax.text(0.5, 0.90, f'n={n}; {p:.0f} % of global mass (2020)',
            transform=ax.transAxes, fontsize=6.5, ha='center', va='top')

for ax in [ax_a, ax_b]:
    ax.set_xlabel('')
    ax.tick_params(labelbottom=False)

fig.text(
    0.015, 0.5,
    'Steady-state glacier mass (% rel. to 2020)',
    rotation='vertical',
    fontsize=8,
    ha='center',
    va='center'
)

out_png = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_6.png'
plt.savefig(out_png, dpi=600)

plt.show()
