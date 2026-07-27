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
import scipy.stats

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


var = 'AAR_mean'
stat_col = f'{var}_area_weighted_mean'

tag = 'median_a'

path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/pygem_oggm/'
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

#                        1,     2,    3,     4,     5,   6,   7,   8,    9,  10,    11,    12,  13,    14,    15,  16,   17,    18,   19
text_lon = np.array([-177, -132.5, -127, -88.5, -56, -25, -8, 33.5, 37, 177, -4.5, 33.5, 107, 66.5, 103.5, 140, -60, 162, -177])
text_lat = np.array([48, 33, 83.5, 55, 57, 57, 83.5, 57, 69, 87, 38, 29, 44, 24, 24, -23, -27, -35.5, -47.5])
text_loc = np.array(['lt', 'lt', 'rt', 'lt', 'lt', 'lt', 'lt', 'rb', 'lb', 'rt', 'lt', 'lt', 'lt', 'lt', 'rt', 'rb', 'lt', 'rt', 'lt'])

point_lon = np.array([-177 + 7, -132.5 + 7, -127 - 3, -88.5 + 7, -56 + 7, -24 + 6, -8 + 7, 33.5 - 3, 37 + 7, 177 - 3, -4.5 + 10, 33.5 + 11, 107 + 11, 66.5 + 11, 103.5 - 3, 140 - 3, -60 + 11, 162 - 3, -177 + 11])
point_lat = np.array([48 - 3, 33 - 3, 83.5 - 3, 55 - 3, 57 - 3, 57 - 3, 83.5 - 3, 57 + 3, 69 + 3.5, 87 - 3, 38 - 3, 29 - 3, 44 - 3, 24 - 3, 24 - 3, -23 + 3, -27 - 3, -35.5 - 3, -47.5 - 3])


#%% Data for Figure_1A
global_stats = pd.read_csv(path + f'PyGEM_global_stats_{tag}.csv')
global_stats_era5 = global_stats[global_stats['gcm'] == 'era5']

global_mean_era5 = global_stats_era5[f'{var}_mean'].item()
global_std_era5 = global_stats_era5[f'{var}_std'].item()
global_median_era5 = global_stats_era5[f'{var}_median'].item()
global_mad_era5 = global_stats_era5[f'{var}_MAD'].item()
global_area_weighted_mean_era5 = global_stats_era5[stat_col].item()

regions = np.arange(1, 20, 1)
regional_area_weighted_mean_era5 = np.full_like(regions, np.nan, dtype=float)
for i, region in enumerate(regions):
    if region in {1, 3, 4, 5, 7, 9, 17, 19}:
        regional_stats = pd.read_csv(path + f'PyGEM_regional_stats_{region:02d}_{tag}.csv')
    else:
        regional_stats = pd.read_csv(path + f'PyGEM_regional_stats_{region:02d}_median_k.csv')
    region_stats = regional_stats[regional_stats['gcm'] == 'era5']
    regional_area_weighted_mean_era5[i] = region_stats[stat_col].item()
        
# Gridded values are already area-weighted means if generated with the updated regional_avg.
grid_era5 = xr.open_dataset(path + 'PyGEM_glacier_stats_grid_2.nc')
grid_era5 = grid_era5[var]
grid_era5 = np.flip(grid_era5, axis=0)

#%% Data for Figure_1B
glacier_stats_nc = xr.open_dataset(path + f'PyGEM_global_glacier_stats_{tag}.nc')

glacier_era5 = glacier_stats_nc[var].where(glacier_stats_nc['gcm'] == 'era5', drop=True)
glacier_era5_values = glacier_era5.values.flatten()
glacier_era5_values = glacier_era5_values[~np.isnan(glacier_era5_values)]

#%% Data for Figure_1C
global_lowess = pd.read_csv(path + f'PyGEM_global_stats_{stat_col}_{tag}_lowess_fit.csv')
global_lowess = global_lowess.rename(columns={'x': 'temp_ch_ipcc'})

global_gcm = global_stats[global_stats['gcm'] != 'era5']
global_era5 = global_stats[global_stats['gcm'] == 'era5']


#%% Setup figure
fig_width_inch = 6.3
fig = plt.figure(figsize=(fig_width_inch, 5.5), dpi=600)

############################################################################ Figure_1A
ax_a = fig.add_subplot([0.08, 0.43, 1, 0.6], projection=proj)

ax_a.set_global()
ax_a.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='gainsboro'))
ax_a.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='dimgrey', alpha=0.5))
ax_a.set_title(r'Area-weighted mean $\overline{\mathbf{AAR}}$ under current climate conditions (2014-2023)',
               fontsize=9, fontweight='bold', loc='center', pad=3)

ax_a.text(0, 1.01, 'A', fontsize=9, fontweight='bold', transform=ax_a.transAxes)
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

ax_a.text(
    -150,
    0,
    'Global\n      ',
    fontsize=8,
    alpha=1,
    color='black',
    ha='center',
    va='center',
    transform=ax_a.transData,
    bbox={'facecolor': 'white', 'pad': 1, 'linewidth': 0.2},
)

for i in range(0, len(text_lon)):
    if text_loc[i] == 'lt':
        ha = 'left'
        va = 'top'
    elif text_loc[i] == 'rt':
        ha = 'right'
        va = 'top'
    elif text_loc[i] == 'rb':
        ha = 'right'
        va = 'bottom'
    elif text_loc[i] == 'lb':
        ha = 'left'
        va = 'bottom'

    ax_a.text(
        text_lon[i],
        text_lat[i],
        str(i + 1) + '   ',
        fontsize=7,
        alpha=1,
        color='black',
        ha=ha,
        va=va,
        transform=ax_a.transData,
        bbox={'facecolor': 'white', 'pad': 1, 'linewidth': 0.2},
    )

# Use the gridded 2.5th/97.5th percentiles for the color scale and center it on alpha = 1.
color_low = np.round(np.nanpercentile(glacier_era5_values, 2.5),2)
color_mid = np.round(global_area_weighted_mean_era5, 2)
color_high = np.round(np.nanpercentile(glacier_era5_values, 97.5),2)
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

ax_a.scatter(-150, -4, c=global_area_weighted_mean_era5, s=15, norm=norm, cmap=cmap_cus, zorder=3)
ax_a.scatter(point_lon, point_lat, c=regional_area_weighted_mean_era5, s=10, norm=norm, cmap=cmap_cus, zorder=3)

cbar1 = fig.colorbar(
    im1,
    ax=ax_a,
    ticks=np.append(np.linspace(color_low, color_mid, 5), np.linspace(color_mid, color_high, 5)[1:]),
    extend='both',
    shrink=0.8,
    aspect=30,
    pad=0.02,
    orientation='vertical',
)
cbar1.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
cbar1.ax.tick_params(direction='in', size=2, width=0.5, labelsize=7, pad=2, labelcolor='black')

############################################################################ Figure_1B
ax_b = fig.add_subplot([0.08, 0.13, 0.4, 0.3])
ax_b.text(0, 1.01, 'B', fontsize=9, fontweight='bold', transform=ax_b.transAxes)
ax_b.set_title(r'Histogram of glacier-level $\overline{\mathbf{AAR}}$',
               fontsize=8, fontweight='bold', loc='center', pad=3)
ax_b.hist(
    glacier_era5,
    density=True,
    bins=np.linspace(0, 2, 40),
    alpha=1,
    color='gainsboro',
    edgecolor='white',
    linewidth=0.5,
    orientation='vertical',
    label='Histogram',
)

compile_a_normx = np.linspace(glacier_era5.min().item(), glacier_era5.max().item(), 1000)
compile_a_normy = scipy.stats.norm.pdf(compile_a_normx, global_mean_era5, global_std_era5)
ax_b.plot(compile_a_normx, compile_a_normy, color='dimgrey', linestyle='-', label='Gaussian DIST')
ax_b.set_xlim(0, 1)

median_y = 1.8
ax_b.errorbar(
    global_median_era5,
    median_y,
    fmt='s',
    xerr=global_mad_era5,
    capsize=1,
    elinewidth=0.5,
    capthick=0.5,
    label='Median ± MAD',
    c='black',
    markersize=3
)

ax_b.errorbar(
    global_mean_era5,
    max(compile_a_normy),
    fmt='D',
    xerr=global_std_era5,
    capsize=1,
    elinewidth=0.5,
    capthick=0.5,
    label='Mean ± std',
    c='black',
    markersize=3
)

awm_y = 2.0
ax_b.plot(
    global_area_weighted_mean_era5,
    awm_y,
    marker='*',
    linestyle='None',
    label='Area-weighted mean',
    c='#489FE3',
    markersize=6,
    markeredgecolor='black',
    markeredgewidth=0.3,
)

ax_b.axvline(1, color='black', linestyle=':', linewidth=0.8, zorder=1)

ax_b.set_xlabel(r'$\overline{AAR}$', fontsize=8)
ax_b.set_ylabel('Probability density', fontsize=8)

ax_b.legend(
    loc='lower center',
    ncols=1,
    markerscale=0.7,
    frameon=False,
    borderpad=0.3,
    labelspacing=0.3,
    handletextpad=0.4,
    alignment='left',
)

ax_b.text(
    global_mean_era5,
    max(compile_a_normy)-0.05,
    f'{global_mean_era5:.2f} ± {global_std_era5:.2f}',
    color='black',
    fontsize=7,
    ha='center',
    va='top',
)

ax_b.text(
    global_median_era5,
    median_y-0.05,
    f'{global_median_era5:.2f} ± {global_mad_era5:.2f}',
    color='black',
    fontsize=7,
    ha='center',
    va='top',
)

ax_b.text(
    global_area_weighted_mean_era5,
    awm_y-0.05,
    f'{global_area_weighted_mean_era5:.2f}',
    color='#489FE3',
    fontsize=7,
    ha='center',
    va='top',
)

############################################################################ Figure_1C
ax_c = fig.add_subplot([0.51, 0.13, 0.4, 0.3])
ax_c.text(0, 1.01, 'C', fontsize=9, fontweight='bold', transform=ax_c.transAxes)
ax_c.set_title(r'$\overline{\mathbf{AAR}}$ under different warming level',
               fontsize=8, fontweight='bold', loc='center', pad=3)

ax_c.yaxis.tick_right()
ax_c.yaxis.set_label_position('right')

ax_c.plot(
    global_gcm['temp_ch_ipcc'].values,
    global_gcm[stat_col].values,
    'o',
    label='based on GCMs',
    c='#489FE3',
    alpha=0.6,
    markersize=2,
)

ax_c.plot(
    global_era5['temp_ch_ipcc'].values,
    global_era5[stat_col].values,
    '*',
    label='based on ERA5',
    c='#489FE3',
    alpha=1,
    markersize=7,
    markeredgecolor='black',
    markeredgewidth=0.3,
)

ax_c.set_xlabel(r'$\Delta$T ($^\circ$C)', fontsize=8)
ax_c.set_ylabel(r'$\overline{AAR}$', fontsize=8)

x = global_lowess['temp_ch_ipcc']
y_med = global_lowess['0.5']
y_low = global_lowess['0.17']
y_high = global_lowess['0.83']

ax_c.plot(x, y_med, color='black', linewidth=1, zorder=100)
ax_c.fill_between(x, y_low, y_high, color='black', alpha=0.15, linewidth=0)
ax_c.set_ylim(0,1.0)

ax_c.axvline(1.2, color='grey', linestyle='--', linewidth=0.8, zorder=1)
ax_c.text(-0.45, 0.01, 'Current\nclimatic\nconditions\n(2014-2023)\nΔT (°C)=1.2', 
          color='grey', fontsize=7, ha='left', va='bottom')

leg = ax_c.legend(
    title=r'Global area-weighted mean:',
    loc='upper right',
    ncols=1,
    markerscale=0.7,
    frameon=False,
    borderpad=0.3,
    labelspacing=0.3,
    handletextpad=0.4,
    alignment='left'
)

leg.get_title().set_fontsize(7)

ax_c.fill_between(
    [3.0, 3.6],
    [0.47, 0.47],
    [0.53, 0.53],
    color='black',
    alpha=0.15,
    linewidth=0,
)
ax_c.plot(
    [3.0, 3.6],
    [0.50, 0.50],
    color='black',
    linewidth=1,
    zorder=100,
)
ax_c.text(3.8, 0.50, 'LOWESS fit\n50% [17% to 83%]', fontsize=7, ha='left', va='center')

############################################################################ Save figure
ax_a.text(
    0.08,
    0.055,
    '1: Alaska  2: W Canada & US  3: Arctic Canada North  4: Arctic Canada South  5: Greenland Periphery  6: Iceland  7: Svalbard  8: Scandinavia',
    fontsize=6,
    transform=fig.transFigure,
)
ax_a.text(
    0.08,
    0.035,
    '9: Russian Arctic  10: North Asia  11: Central Europe  12: Caucasus & Middle East 13: Central Asia  14: South Asia West  15: South Asia East',
    fontsize=6,
    transform=fig.transFigure,
)
ax_a.text(
    0.08,
    0.015,
    '16: Low Latitudes  17: Southern Andes  18: New Zealand  19: Antarctic & Subantarctic',
    fontsize=6,
    transform=fig.transFigure,
)

out_pdf = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_S3.png'
plt.savefig(out_pdf, dpi=600)

plt.show()
