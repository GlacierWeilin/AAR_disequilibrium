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
tag = 'median_a'
area_scale = 1e5
volume_scale = 1e4

glacier_stats = xr.open_dataset(
    path + f'PyGEM_global_glacier_stats_{tag}.nc'
)
global_rgi_area_km2 = glacier_stats['rgi_area_km2'].sum(skipna=True).item()
global_itmix_volume_km3 = glacier_stats['vol_itmix_m3'].sum(skipna=True).item() / 1e9


# %% add data
global_result = pd.read_csv(
    path + f'PyGEM_global_mass_{tag}.csv'
)

global_result = global_result.dropna(
    subset=['temp_ch_ipcc', 'area_steady', 'volume_steady']
).copy()


# %% add lowess fit data
area_lowess = pd.read_csv(
    path + f'PyGEM_global_mass_area_steady_{tag}_lowess_fit.csv'
)

volume_lowess = pd.read_csv(
    path + f'PyGEM_global_mass_volume_steady_{tag}_lowess_fit.csv'
)

if 'x' in area_lowess.columns:
    area_lowess = area_lowess.rename(columns={'x': 'temp_ch_ipcc'})

if 'x' in volume_lowess.columns:
    volume_lowess = volume_lowess.rename(columns={'x': 'temp_ch_ipcc'})

if 'variable' in area_lowess.columns:
    area_lowess = area_lowess[area_lowess['variable'] == 'area_steady'].copy()
elif 'y_col' in area_lowess.columns:
    area_lowess = area_lowess[area_lowess['y_col'] == 'area_steady'].copy()

if 'variable' in volume_lowess.columns:
    volume_lowess = volume_lowess[volume_lowess['variable'] == 'volume_steady'].copy()
elif 'y_col' in volume_lowess.columns:
    volume_lowess = volume_lowess[volume_lowess['y_col'] == 'volume_steady'].copy()

median_col = 'median' if 'median' in area_lowess.columns else '0.5'
q17_col = 'q17' if 'q17' in area_lowess.columns else '0.17'
q83_col = 'q83' if 'q83' in area_lowess.columns else '0.83'


# %% Create plot
fig = plt.figure(figsize=(5.7, 2.26))

gs = GridSpec(1, 2, figure=fig,left=0.07, right=0.99, bottom=0.125, top=0.96, wspace=0.2, width_ratios=[1, 1])

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

for ax in [ax_a, ax_b]:
    ax.set_xlim(-0.1, 6.85)


############################################################################ Figure_S4A
var = 'area_steady'
raw_sub = global_result.sort_values('temp_ch_ipcc')
era5_sub = raw_sub[raw_sub['gcm'].astype(str).str.lower() == 'era5']
non_era5_sub = raw_sub[raw_sub['gcm'].astype(str).str.lower() != 'era5']

ax_a.scatter(
    non_era5_sub['temp_ch_ipcc'].values,
    non_era5_sub[var].values / area_scale,
    s=8,
    color='#489FE3',
    alpha=0.42,
    linewidths=0,
    zorder=5
)

ax_a.scatter(
    era5_sub['temp_ch_ipcc'].values,
    era5_sub[var].values / area_scale,
    marker='*',
    s=38,
    color='#489FE3',
    edgecolor='black',
    linewidths=0.35,
    alpha=0.95,
    zorder=30
)

x = area_lowess['temp_ch_ipcc'].values
y_med = area_lowess[median_col].values / area_scale
y_low = area_lowess[q17_col].values / area_scale
y_high = area_lowess[q83_col].values / area_scale

ax_a.plot(
    x, y_med,
    color='black',
    linewidth=1,
    zorder=10
)

ax_a.fill_between(
    x, y_low, y_high,
    color='black',
    alpha=0.15,
    linewidth=0,
    zorder=1
)

ax_a.set_xlabel('ΔT (°C)', fontsize=8)
ax_a.set_ylabel(r'Steady-state glacier area ($\times 10^5$ km$^2$)', fontsize=8)
ax_a.text(0.01, 0.93, 'A', transform=ax_a.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')
ax_a.text(
    0.3, 0.95,
    f'RGI global glacier area$^{{22}}$\n{global_rgi_area_km2:,.0f} km$^2$',
    transform=ax_a.transAxes,
    fontsize=8,
    ha='left',
    va='top'
)


############################################################################ Figure_S4B
var = 'volume_steady'
raw_sub = global_result.sort_values('temp_ch_ipcc')
era5_sub = raw_sub[raw_sub['gcm'].astype(str).str.lower() == 'era5']
non_era5_sub = raw_sub[raw_sub['gcm'].astype(str).str.lower() != 'era5']

ax_b.scatter(
    non_era5_sub['temp_ch_ipcc'].values,
    non_era5_sub[var].values / volume_scale,
    s=8,
    color='#489FE3',
    alpha=0.42,
    linewidths=0,
    zorder=5
)

ax_b.scatter(
    era5_sub['temp_ch_ipcc'].values,
    era5_sub[var].values / volume_scale,
    marker='*',
    s=38,
    color='#489FE3',
    edgecolor='black',
    linewidths=0.35,
    alpha=0.95,
    zorder=30
)

x = volume_lowess['temp_ch_ipcc'].values
y_med = volume_lowess[median_col].values / volume_scale
y_low = volume_lowess[q17_col].values / volume_scale
y_high = volume_lowess[q83_col].values / volume_scale

ax_b.plot(
    x, y_med,
    color='black',
    linewidth=1,
    zorder=10
)

ax_b.fill_between(
    x, y_low, y_high,
    color='black',
    alpha=0.15,
    linewidth=0,
    zorder=1
)

ax_b.set_xlabel('ΔT (°C)', fontsize=8)
ax_b.set_ylabel(r'Steady-state glacier volume ($\times 10^4$ km$^3$)', fontsize=8)
ax_b.text(0.01, 0.93, 'B', transform=ax_b.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')
ax_b.text(
    0.3, 0.95,
    f'Consensus volume$^{{48}}$\n{global_itmix_volume_km3:,.0f} km$^3$',
    transform=ax_b.transAxes,
    fontsize=8,
    ha='left',
    va='top'
)


##################################################################
### Legend
ax_b.scatter(
    [3.9], [14],
    marker='o',
    s=8,
    color='black',
    alpha=0.42,
    linewidths=0,
    zorder=100
)
ax_b.text(4.1, 14, 'based on GCMs', fontsize=7, ha='left', va='center')

ax_b.scatter(
    [3.9], [13],
    marker='*',
    s=38,
    color='white',
    edgecolor='black',
    linewidths=0.35,
    zorder=100
)
ax_b.text(4.1, 13, 'based on ERA5', fontsize=7, ha='left', va='center')

ax_b.fill_between(
    [3.7, 4.1], [10.5, 10.5], [11.5, 11.5],
    color='black',
    alpha=0.15,
    linewidth=0
)
ax_b.plot(
    [3.7, 4.1], [11, 11],
    color='black',
    linewidth=1,
    zorder=100
)
ax_b.text(4.25, 11, 'LOWESS fit\n50% [17% to 83%]', fontsize=7, ha='left', va='center')

out_png = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_S6.png'
plt.savefig(out_png, dpi=600)

plt.show()
