import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
import matplotlib as mpl
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

plt.rcParams.update({'lines.linewidth': 0.5})
plt.rcParams.update({'font.size': 6})
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


var_alpha = 'disequilibrium'
stat_col = f'{var_alpha}_area_weighted_mean'
tag = 'median_a'
var_mass = 'mass_remaining'

path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/pygem_oggm/'
glaciermip3_path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/GlacierMIP3/'
regions_shp = path + '../data/00_rgi60_regions/00_rgi60_O1Regions.shp'

proj = ccrs.PlateCarree()
shape_feature = ShapelyFeature(
    Reader(regions_shp).geometries(),
    ccrs.PlateCarree(),
    edgecolor='white',
    alpha=1,
    facecolor='None',
    linewidth=0.5,
)

lonmin = -179.5
lonmax = 179.5
latmin = -90
latmax = 90
extents = [lonmin, lonmax, latmin, latmax]

text_lon = np.array([-177, -132.5, -127, -88.5, -56, -25, -8, 33.5, 37, 177, -4.5, 33.5, 107, 66.5, 103.5, 140, -60, 162, -177])
text_lat = np.array([48, 33, 83.5, 55, 57, 57, 83.5, 57, 69, 87, 38, 29, 44, 24, 24, -23, -27, -35.5, -47.5])
text_loc = np.array(['lt', 'lt', 'rt', 'lt', 'lt', 'lt', 'lt', 'rb', 'lb', 'rt', 'lt', 'lt', 'lt', 'lt', 'rt', 'rb', 'lt', 'rt', 'lt'])

point_lon = np.array([-177 + 7, -132.5 + 7, -127 - 3, -88.5 + 7, -56 + 7, -24 + 6, -8 + 7, 33.5 - 3, 37 + 7, 177 - 3, -4.5 + 10, 33.5 + 11, 107 + 11, 66.5 + 11, 103.5 - 3, 140 - 3, -60 + 11, 162 - 3, -177 + 11])
point_lat = np.array([48 - 3, 33 - 3, 83.5 - 3, 55 - 3, 57 - 3, 57 - 3, 83.5 - 3, 57 + 3, 69 + 3.5, 87 - 3, 38 - 3, 29 - 3, 44 - 3, 24 - 3, 24 - 3, -23 + 3, -27 - 3, -35.5 - 3, -47.5 - 3])


# %% Data for Figure_abstractA
grid_era5 = xr.open_dataset(path + 'PyGEM_glacier_stats_grid_2.nc')
grid_era5 = grid_era5[var_alpha]
grid_era5 = np.flip(grid_era5, axis=0)

glacier_stats_nc = xr.open_dataset(path + f'PyGEM_global_glacier_stats_{tag}.nc')
glacier_era5 = glacier_stats_nc[var_alpha].where(glacier_stats_nc['gcm'] == 'era5', drop=True)
glacier_era5_values = glacier_era5.values.flatten()
glacier_era5_values = glacier_era5_values[~np.isnan(glacier_era5_values)]


# %% Data for Figure_abstractB
pygem_a = pd.read_csv(
    path + 'PyGEM_global_mass_median_a.csv'
)

glaciermip3_median = pd.read_csv(
    glaciermip3_path + 'fig1b_scatter_median_by_temp.csv'
)

pygem_a_lowess = pd.read_csv(
    path + f'PyGEM_global_mass_{var_mass}_median_a_lowess_fit.csv'
)

glaciermip3_median_lowess = pd.read_csv(
    glaciermip3_path + 'lowess_fit_rel_2020_101yr_avg_steady_state_Feb12_2024.csv'
)

glaciermip3_median_lowess = glaciermip3_median_lowess[
    glaciermip3_median_lowess['region'].astype(str) == 'All'
].copy()


# %% Data for Figure_abstractC
pygem_k = pd.read_csv(
    path + 'PyGEM_global_mass_median_k.csv'
)

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

mt_a_lowess_l = []
mt_k_lowess_l = []
for cla in ['all', 'outAntarc', 'inAntarc']:
    _df = pd.read_csv(
        path + f'PyGEM_glacier_mass_MT_{cla}_{var_mass}_median_a_lowess_fit.csv'
    )
    _df['region_class'] = f'MT_{cla}'
    mt_a_lowess_l.append(_df)

    _df = pd.read_csv(
        path + f'PyGEM_glacier_mass_MT_{cla}_{var_mass}_median_k_lowess_fit.csv'
    )
    _df['region_class'] = f'MT_{cla}'
    mt_k_lowess_l.append(_df)

mt_a_lowess = pd.concat(mt_a_lowess_l, ignore_index=True)
mt_k_lowess = pd.concat(mt_k_lowess_l, ignore_index=True)

if 'variable' in mt_a_lowess.columns:
    mt_a_lowess = mt_a_lowess[mt_a_lowess['variable'] == var_mass].copy()
elif 'y_col' in mt_a_lowess.columns:
    mt_a_lowess = mt_a_lowess[mt_a_lowess['y_col'] == var_mass].copy()

if 'variable' in mt_k_lowess.columns:
    mt_k_lowess = mt_k_lowess[mt_k_lowess['variable'] == var_mass].copy()
elif 'y_col' in mt_k_lowess.columns:
    mt_k_lowess = mt_k_lowess[mt_k_lowess['y_col'] == var_mass].copy()

median_col = 'median' if 'median' in mt_a_lowess.columns else '0.5'
q17_col = 'q17' if 'q17' in mt_a_lowess.columns else '0.17'
q83_col = 'q83' if 'q83' in mt_a_lowess.columns else '0.83'


# %% colors
colors_global = ['#489FE3', '#F09137']
colors_mt = ['#489FE3', '#C93735']


# %% Setup figure
fig_width_inch = 4
fig = plt.figure(figsize=(fig_width_inch, 3.7), dpi=600)

############################################################################ Figure_abstractA
ax_a = fig.add_subplot([0.12, 0.40, 0.96, 0.6], projection=proj)

ax_a.set_global()
ax_a.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='gainsboro'))
ax_a.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='dimgrey', alpha=0.5))
ax_a.set_title(r'Regional area-weighted mean glacier climate-imbalance'
               '\n'
               r'index ($\boldsymbol{\alpha}$) '
               r'under current climate (2014–2023)',
               fontsize=7, fontweight='bold', loc='center', pad=3)

#ax_a.text(0, 1.01, 'A', fontsize=8, fontweight='bold', transform=ax_a.transAxes)
ax_a.spines['geo'].set_edgecolor('black')
ax_a.add_feature(shape_feature)

ax_a.set_xticks(np.arange(-180, 180 + 60, 60), crs=proj)
ax_a.set_xticks(np.arange(-180, 180 + 30, 30), minor=True, crs=proj)
ax_a.set_yticks(np.arange(-90, 90 + 30, 30), crs=proj)
ax_a.set_yticks(np.arange(-90, 90 + 15, 15), minor=True, crs=proj)
ax_a.xaxis.set_major_formatter(LongitudeFormatter())
ax_a.yaxis.set_major_formatter(LatitudeFormatter())
ax_a.tick_params(axis='both', which='major', length=2, width=0.5, color='black', labelcolor='black', pad=3)
ax_a.tick_params(axis='both', which='minor', length=1, width=0.5, color='black')

color_low = np.round(np.nanpercentile(glacier_era5_values, 2.5), 1)
color_mid = 1
color_high = np.round(np.nanpercentile(glacier_era5_values, 97.5), 1)
col_bounds = np.linspace(color_low, color_mid, 8)
col_bounds = np.append(col_bounds, np.linspace(color_mid, color_high, 8))
cb = []
cb_val = np.linspace(1, 0, len(col_bounds))
for j in range(len(cb_val)):
    cb.append(mpl.cm.RdBu_r(cb_val[j]))
cmap_cus = mpl.colors.LinearSegmentedColormap.from_list(
    'my_cb',
    list(zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)),
    N=1000,
)

norm = mpl.colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))
im1 = ax_a.imshow(
    grid_era5,
    extent=extents,
    transform=ccrs.PlateCarree(),
    alpha=0.8,
    norm=norm,
    cmap=cmap_cus,
    zorder=2,
)

cbar1 = fig.colorbar(
    im1,
    ax=ax_a,
    ticks=np.append(np.linspace(color_low, color_mid, 6), np.linspace(color_mid, color_high, 5)[1:]),
    extend='both',
    shrink=0.7,
    aspect=30,
    pad=0.02,
    orientation='vertical',
)
cbar1.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
cbar1.ax.tick_params(direction='in', size=2, width=0.5, labelsize=6, pad=2, labelcolor='black')

############################################################################ Figure_abstractB
ax_b = fig.add_subplot([0.12, 0.07, 0.40, 0.31])
#ax_b.text(0, 1.01, 'B', fontsize=8, fontweight='bold', transform=ax_b.transAxes)
ax_b.set_title('Comparison with GlacierMIP3$^{18}$',
               fontsize=7, fontweight='bold', loc='center', pad=3)

ax_b.set_xlim(-0.1, 6.85)
ax_b.set_ylim(0, 150)
ax_b.set_yticks([0, 50, 100, 150])
ax_b.tick_params(axis='y', labelleft=True)

pygem_a_era5 = pygem_a[pygem_a['gcm'].astype(str).str.lower() == 'era5']
pygem_a_gcm = pygem_a[pygem_a['gcm'].astype(str).str.lower() != 'era5']

ax_b.scatter(
    pygem_a_gcm['temp_ch_ipcc'].values,
    pygem_a_gcm[var_mass].values,
    s=7,
    color=colors_global[0],
    alpha=0.42,
    linewidths=0,
    zorder=5,
    label='This study'
)

ax_b.scatter(
    pygem_a_era5['temp_ch_ipcc'].values,
    pygem_a_era5[var_mass].values,
    s=7,
    color=colors_global[0],
    alpha=0.42,
    linewidths=0,
    zorder=5,
)

ax_b.plot(
    pygem_a_lowess['temp_ch_ipcc'].values,
    pygem_a_lowess['0.5'].values,
    color=colors_global[0],
    linewidth=1,
    zorder=10
)

ax_b.fill_between(
    pygem_a_lowess['temp_ch_ipcc'].values,
    pygem_a_lowess['0.17'].values,
    pygem_a_lowess['0.83'].values,
    color=colors_global[0],
    alpha=0.15,
    linewidth=0,
    zorder=1
)

ax_b.scatter(
    glaciermip3_median['temp_ch_ipcc'].values,
    glaciermip3_median['median_mass_percent_rel_2020'].values,
    s=7,
    color=colors_global[1],
    alpha=0.42,
    linewidths=0,
    zorder=5,
    label=r'GlacierMIP3'
            '\n'
            r'ensemble median$^{18}$'
)

ax_b.plot(
    glaciermip3_median_lowess['temp_ch'].values,
    glaciermip3_median_lowess['0.5'].values,
    color=colors_global[1],
    linewidth=1,
    zorder=10
)

ax_b.fill_between(
    glaciermip3_median_lowess['temp_ch'].values,
    glaciermip3_median_lowess['0.17'].values,
    glaciermip3_median_lowess['0.83'].values,
    color=colors_global[1],
    alpha=0.15,
    linewidth=0,
    zorder=1
)

ax_b.set_xlabel(r'$\Delta$T ($^\circ$C)', fontsize=7)
ax_b.set_ylabel('Steady-state glacier mass\n(% rel. to 2020)', fontsize=7)
ax_b.legend(
    loc='upper right',
    ncols=1,
    markerscale=1,
    frameon=False,
    borderpad=0.2,
    labelspacing=0.25,
    handletextpad=0.4,
    alignment='left',
)

ax_b.fill_between(
    [3.0, 3.6],
    [82, 82],
    [92, 92],
    color='black',
    alpha=0.15,
    linewidth=0,
)
ax_b.plot(
    [3.0, 3.6],
    [87, 87],
    color='black',
    linewidth=1,
    zorder=100,
)
ax_b.text(3.8, 87, 'LOWESS fit\n50% [17% to 83%]', fontsize=6, ha='left', va='center')

############################################################################ Figure_abstractC
ax_c = fig.add_subplot([0.54, 0.07, 0.38, 0.31], sharey=ax_b)
#ax_c.text(0, 1.01, 'C', fontsize=9, fontweight='bold', transform=ax_c.transAxes)
ax_c.set_title('Antarctic and Subantarctic',
               fontsize=7, fontweight='bold', loc='center', pad=3)

ax_c.set_xlim(-0.1, 6.85)
ax_c.set_ylim(0, 150)
ax_c.set_yticks([0, 50, 100, 150])
ax_c.tick_params(axis='y', labelleft=False)

cls = 'MT_inAntarc'
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

ax_c.scatter(
    mt_a_gcm['temp_ch_ipcc'].values,
    mt_a_gcm[var_mass].values,
    s=7,
    color=colors_mt[0],
    alpha=0.42,
    linewidths=0,
    zorder=5,
    label=r'use median $\alpha$'
)

ax_c.scatter(
    mt_a_era5['temp_ch_ipcc'].values,
    mt_a_era5[var_mass].values,
    s=7,
    color=colors_mt[0],
    alpha=0.42,
    linewidths=0,
    zorder=5,
)

ax_c.plot(
    mt_a_lowess_sub['temp_ch_ipcc'].values,
    mt_a_lowess_sub[median_col].values,
    color=colors_mt[0],
    linewidth=1,
    zorder=10
)

ax_c.fill_between(
    mt_a_lowess_sub['temp_ch_ipcc'].values,
    mt_a_lowess_sub[q17_col].values,
    mt_a_lowess_sub[q83_col].values,
    color=colors_mt[0],
    alpha=0.15,
    linewidth=0,
    zorder=1
)

ax_c.scatter(
    mt_k_gcm['temp_ch_ipcc'].values,
    mt_k_gcm[var_mass].values,
    s=7,
    color=colors_mt[1],
    alpha=0.42,
    linewidths=0,
    zorder=5,
    label=r'use median $k$'
)

ax_c.scatter(
    mt_k_era5['temp_ch_ipcc'].values,
    mt_k_era5[var_mass].values,
    s=7,
    color=colors_mt[1],
    alpha=0.42,
    linewidths=0,
    zorder=5,
)

ax_c.plot(
    mt_k_lowess_sub['temp_ch_ipcc'].values,
    mt_k_lowess_sub[median_col].values,
    color=colors_mt[1],
    linewidth=1,
    zorder=10
)

ax_c.fill_between(
    mt_k_lowess_sub['temp_ch_ipcc'].values,
    mt_k_lowess_sub[q17_col].values,
    mt_k_lowess_sub[q83_col].values,
    color=colors_mt[1],
    alpha=0.15,
    linewidth=0,
    zorder=1
)

ax_c.set_xlabel(r'$\Delta$T ($^\circ$C)', fontsize=6)
ax_c.set_ylabel('', fontsize=6)
ax_c.legend(
    loc='upper right',
    title='Different treatment of\nmarine-terminating glaciers:',
    ncols=1,
    markerscale=1,
    frameon=False,
    borderpad=0.2,
    labelspacing=0.25,
    handletextpad=0.4,
    alignment='left',
)

out_png = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_abstract.png'
plt.savefig(out_png, dpi=300)

plt.show()
