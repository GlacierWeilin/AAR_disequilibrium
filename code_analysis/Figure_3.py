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
tag = 'median_a'


# %% add data
area_result_l = []
for cla, class_name in zip(
    ['-1', '1-10', '10-100', '100-'],
    ['<1 km2', '1-10 km2', '10-100 km2', '>100 km2'],
):
    _df = pd.read_csv(
        path + f'PyGEM_glacier_mass_area_{cla}_{tag}.csv'
    )
    _df['area_class'] = class_name
    area_result_l.append(_df)

area_result = pd.concat(area_result_l, ignore_index=True)

tidewater_result_l = []
for cla in ['land-terminating', 'marine-terminating']:
    _df = pd.read_csv(
        path + f'PyGEM_glacier_mass_{cla}_{tag}.csv'
    )
    _df['tidewater_class'] = cla
    tidewater_result_l.append(_df)

tidewater_result = pd.concat(tidewater_result_l, ignore_index=True)

area_result = area_result.dropna(subset=['area_class', 'temp_ch_ipcc', var]).copy()
tidewater_result = tidewater_result.dropna(subset=['tidewater_class', 'temp_ch_ipcc', var]).copy()

global_result = pd.read_csv(
    path + f'PyGEM_global_mass_{tag}.csv'
)
global_area_km2 = global_result['area_rgi'].iloc[0]
global_mass_2020_km3 = global_result['mass_2020'].iloc[0]


# %% add lowess fit data
area_lowess_l = []
for cla, class_name in zip(
    ['-1', '1-10', '10-100', '100-'],
    ['<1 km2', '1-10 km2', '10-100 km2', '>100 km2'],
):
    _df = pd.read_csv(
        path + f'PyGEM_glacier_mass_area_{cla}_{var}_{tag}_lowess_fit.csv'
    )
    _df['area_class'] = class_name
    area_lowess_l.append(_df)

area_lowess = pd.concat(area_lowess_l, ignore_index=True)

tidewater_lowess_l = []
for cla in ['land-terminating', 'marine-terminating']:
    _df = pd.read_csv(
        path + f'PyGEM_glacier_mass_{cla}_{var}_{tag}_lowess_fit.csv'
    )
    _df['tidewater_class'] = cla
    tidewater_lowess_l.append(_df)

tidewater_lowess = pd.concat(tidewater_lowess_l, ignore_index=True)

if 'variable' in area_lowess.columns:
    area_lowess = area_lowess[area_lowess['variable'] == var].copy()
elif 'y_col' in area_lowess.columns:
    area_lowess = area_lowess[area_lowess['y_col'] == var].copy()

if 'variable' in tidewater_lowess.columns:
    tidewater_lowess = tidewater_lowess[tidewater_lowess['variable'] == var].copy()
elif 'y_col' in tidewater_lowess.columns:
    tidewater_lowess = tidewater_lowess[tidewater_lowess['y_col'] == var].copy()

median_col = 'median' if 'median' in area_lowess.columns else '0.5'
q17_col = 'q17' if 'q17' in area_lowess.columns else '0.17'
q83_col = 'q83' if 'q83' in area_lowess.columns else '0.83'


# %% classes
area_order = ['<1 km2', '1-10 km2', '10-100 km2', '>100 km2']
tidewater_order = ['land-terminating', 'marine-terminating']

area_name = {
    '<1 km2': r'<1 km$^2$',
    '1-10 km2': r'1-10 km$^2$',
    '10-100 km2': r'10-100 km$^2$',
    '>100 km2': r'>100 km$^2$',
}

tidewater_name = {
    'land-terminating': 'land-terminating',
    'marine-terminating': 'marine-terminating',
}

area_class_summary = (
    area_result
    .drop_duplicates('area_class')
    .set_index('area_class')
    .reindex(area_order)
)

total_area_vol_2020 = area_class_summary['vol_2020_m3'].sum(skipna=True)
area_class_summary['vol_percent'] = (
    area_class_summary['vol_2020_m3'] / total_area_vol_2020 * 100
)
area_class_summary['area_percent_global'] = (
    area_class_summary['rgi_area_km2'] / global_area_km2 * 100
)
area_class_summary['mass_percent_global'] = (
    area_class_summary['vol_2020_m3'] / 1e9 / global_mass_2020_km3 * 100
)

area_labels = []
for cls in area_order:
    n = int(area_class_summary.loc[cls, 'n_glaciers'])
    area_labels.append(f'{area_name[cls]} (n={n:,})')

tidewater_class_summary = (
    tidewater_result
    .drop_duplicates('tidewater_class')
    .set_index('tidewater_class')
    .reindex(tidewater_order)
)

total_tidewater_vol_2020 = tidewater_class_summary['vol_2020_m3'].sum(skipna=True)
tidewater_class_summary['vol_percent'] = (
    tidewater_class_summary['vol_2020_m3'] / total_tidewater_vol_2020 * 100
)
tidewater_class_summary['area_percent_global'] = (
    tidewater_class_summary['rgi_area_km2'] / global_area_km2 * 100
)
tidewater_class_summary['mass_percent_global'] = (
    tidewater_class_summary['vol_2020_m3'] / 1e9 / global_mass_2020_km3 * 100
)

tidewater_labels = []
for cls in tidewater_order:
    n = int(tidewater_class_summary.loc[cls, 'n_glaciers'])
    tidewater_labels.append(f'{tidewater_name[cls]}\n(n={n:,})')

area_colors = ['#C93735', '#F09137', '#5BBBD0', '#5266B0']
tidewater_colors = ['#DC6D57','#489FE3']


# %% Create plot
fig = plt.figure(figsize=(5.7, 2.26))

gs = GridSpec(1, 2, figure=fig,left=0.07, right=0.99, bottom=0.125, top=0.96, wspace=0.03, width_ratios=[1, 1])

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)

ax_b.set_ylabel('')
ax_b.tick_params(labelleft=False)

for ax in [ax_a, ax_b]:
    ax.set_xlim(-0.1, 6.85)
    ax.set_ylim(0, 190)


############################################################################ Figure_2A
for cls, color, label in zip(area_order, area_colors, area_labels):
    raw_sub = area_result[area_result['area_class'] == cls].sort_values('temp_ch_ipcc')
    era5_sub = raw_sub[raw_sub['gcm'].astype(str).str.lower() == 'era5']
    non_era5_sub = raw_sub[raw_sub['gcm'].astype(str).str.lower() != 'era5']

    sub = area_lowess[area_lowess['area_class'] == cls].sort_values('temp_ch_ipcc')

    ax_a.scatter(
        non_era5_sub['temp_ch_ipcc'].values,
        non_era5_sub[var].values,
        s=8,
        color=color,
        alpha=0.42,
        linewidths=0,
        zorder=5
    )

    ax_a.scatter(
        era5_sub['temp_ch_ipcc'].values,
        era5_sub[var].values,
        marker='*',
        s=38,
        color=color,
        edgecolor='black',
        linewidths=0.35,
        alpha=0.95,
        zorder=30
    )

    x = sub['temp_ch_ipcc'].values
    y_med = sub[median_col].values
    y_low = sub[q17_col].values
    y_high = sub[q83_col].values

    ax_a.plot(
        x, y_med,
        color=color,
        linewidth=1,
        label=label,
        zorder=10
    )

    ax_a.fill_between(
        x, y_low, y_high,
        color=color,
        alpha=0.15,
        linewidth=0,
        zorder=1
    )

ax_a.set_xlabel('ΔT (°C)', fontsize=8)
ax_a.set_ylabel('Steady-state glacier mass (% rel. to 2020)', fontsize=8)
ax_a.text(0.01, 0.93, 'A', transform=ax_a.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')

ax_a.text(3.8, 184, 'Glacier size group:', fontsize=7,
          fontweight='bold', ha='left', va='top')

for i, color in enumerate(area_colors):
    ax_a.text(3.8, 184 - 12 * (i + 1), area_labels[i],
              color=color, fontsize=7, ha='left', va='top')

ax_a_area_pie = ax_a.inset_axes([0.49, 0.22, 0.3, 0.3])
ax_a_area_pie.pie(
    area_class_summary['area_percent_global'].values,
    colors=area_colors,
    startangle=90,
    counterclock=False,
    autopct='%.0f',
    pctdistance=0.6,
    textprops={'fontsize': 6, 'color': 'w'}
)
ax_a_area_pie.set_title('% of global\narea (RGI)', fontsize=7, pad=0)

ax_a_mass_pie = ax_a.inset_axes([0.73, 0.22, 0.3, 0.3])
ax_a_mass_pie.pie(
    area_class_summary['mass_percent_global'].values,
    colors=area_colors,
    startangle=90,
    counterclock=False,
    autopct=lambda pct: '' if pct < 1 else f'{pct:.0f}',
    pctdistance=0.6,
    textprops={'fontsize': 6, 'color': 'w'}
)
ax_a_mass_pie.set_title('% of global\nmass (2020)', fontsize=7, pad=0)


############################################################################ Figure_2B
for cls, color, label in zip(tidewater_order, tidewater_colors, tidewater_labels):
    raw_sub = tidewater_result[
        tidewater_result['tidewater_class'] == cls
    ].sort_values('temp_ch_ipcc')
    era5_sub = raw_sub[raw_sub['gcm'].astype(str).str.lower() == 'era5']
    non_era5_sub = raw_sub[raw_sub['gcm'].astype(str).str.lower() != 'era5']

    sub = tidewater_lowess[
        tidewater_lowess['tidewater_class'] == cls
    ].sort_values('temp_ch_ipcc')

    ax_b.scatter(
        non_era5_sub['temp_ch_ipcc'].values,
        non_era5_sub[var].values,
        s=8,
        color=color,
        alpha=0.42,
        linewidths=0,
        zorder=5
    )

    ax_b.scatter(
        era5_sub['temp_ch_ipcc'].values,
        era5_sub[var].values,
        marker='*',
        s=38,
        color=color,
        edgecolor='black',
        linewidths=0.35,
        alpha=0.95,
        zorder=30
    )

    x = sub['temp_ch_ipcc'].values
    y_med = sub[median_col].values
    y_low = sub[q17_col].values
    y_high = sub[q83_col].values

    ax_b.plot(
        x, y_med,
        color=color,
        linewidth=1,
        label=label,
        zorder=10
    )

    ax_b.fill_between(
        x, y_low, y_high,
        color=color,
        alpha=0.15,
        linewidth=0,
        zorder=1
    )

ax_b.set_xlabel('ΔT (°C)', fontsize=8)
ax_b.text(0.01, 0.93, 'B', transform=ax_b.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')

ax_b.text(3.8, 184, 'Glacier terminus type:', fontsize=7,
          fontweight='bold', ha='left', va='top')

for i, color in enumerate(tidewater_colors):
    ax_b.text(3.8, 184 - 11 * (2*i + 1), tidewater_labels[i],
              color=color, fontsize=7, ha='left', va='top')

ax_b_area_pie = ax_b.inset_axes([0.49, 0.22, 0.3, 0.3])
ax_b_area_pie.pie(
    tidewater_class_summary['area_percent_global'].values,
    colors=tidewater_colors,
    startangle=90,
    counterclock=False,
    autopct='%.0f',
    pctdistance=0.65,
    textprops={'fontsize': 6, 'color':'w'}
)
ax_b_area_pie.set_title('% of global\narea (RGI)', fontsize=7, pad=0.2)

ax_b_mass_pie = ax_b.inset_axes([0.73, 0.22, 0.3, 0.3])
ax_b_mass_pie.pie(
    tidewater_class_summary['mass_percent_global'].values,
    colors=tidewater_colors,
    startangle=90,
    counterclock=False,
    autopct='%.0f',
    pctdistance=0.65,
    textprops={'fontsize': 6, 'color': 'w'}
)
ax_b_mass_pie.set_title('% of global\nmass (2020)', fontsize=7, pad=0.2)

##################################################################
### Legend
ax_b.scatter(
    [0.6], [180],
    marker='o',
    s=8,
    color='black',
    alpha=0.42,
    linewidths=0,
    zorder=100
)
ax_b.text(0.8, 180, 'based on GCMs', fontsize=7, ha='left', va='center')

ax_b.scatter(
    [0.6], [168],
    marker='*',
    s=38,
    color='white',
    edgecolor='black',
    linewidths=0.35,
    zorder=100
)
ax_b.text(0.8, 168, 'based on ERA5', fontsize=7, ha='left', va='center')

ax_b.fill_between(
    [0.45, 0.75], [145, 145], [155, 155],
    color='black',
    alpha=0.15,
    linewidth=0
)
ax_b.plot(
    [0.45, 0.75], [150, 150],
    color='black',
    linewidth=1,
    zorder=100
)
ax_b.text(0.9, 150, 'LOWESS fit\n50% [17% to 83%]', fontsize=7, ha='left', va='center')

out_png = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_3.png'
plt.savefig(out_png, dpi=600)

plt.show()
