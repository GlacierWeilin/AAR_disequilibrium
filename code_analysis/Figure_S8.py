import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/data/';
regions_shp = path + '../data/00_rgi60_regions/00_rgi60_O1Regions.shp'

obs = pd.read_csv(path + 'WGMS_disequilibrium_valid.csv');

result = pd.DataFrame({
    'a': obs['disequilibrium'].values,
    'AAR_mean': obs['AAR_mean'].values,
    'AAR': obs['AAR_steady'].values
})

titles = ['α', r'$\mathbf{\overline{AAR}}$',
          r'$\mathbf{AAR}_0$']

bounds = np.array([
    result.quantile(0.025).values,
    result.quantile(0.5).values,
    result.quantile(0.975).values,
])

labels = ['A', 'B', 'C']

#%%
plt.rcParams.update({'lines.linewidth':0.5})
plt.rcParams.update({'font.size': 7})
plt.rcParams.update({'font.sans-serif': 'Arial'})

plt.rcParams.update({'axes.linewidth':0.5})
plt.rcParams.update({'axes.titlepad':3})
plt.rcParams.update({'axes.titlesize':8})
plt.rcParams.update({'axes.labelpad':2})
plt.rcParams.update({'xtick.major.pad':2})
plt.rcParams.update({'ytick.major.pad':2})
plt.rcParams.update({'xtick.major.width':0.5})
plt.rcParams.update({'ytick.major.width':0.5})
plt.rcParams.update({'xtick.major.size':1.5})
plt.rcParams.update({'ytick.major.size':1.5})

fig = plt.figure(figsize=(4.71, 6.8), dpi=600)
gs = GridSpec(3, 1, figure=fig, hspace=0.04, height_ratios=[1,1,1])
plt.subplots_adjust(left=0.06, right=1.07, top=0.99, bottom=0.01)


for i in range(3):
    proj = ccrs.PlateCarree()
    shape_feature = ShapelyFeature(Reader(regions_shp).geometries(), ccrs.PlateCarree(), edgecolor='white', alpha=1,
                                   facecolor='None', linewidth=0.5)

    ax = fig.add_subplot(gs[i, 0], projection=proj)
    ax.set_global()
    ax.set_title(titles[i], fontweight='bold',loc='center', pad=2)

    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='gainsboro'))
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor='dimgrey'), alpha=0.5)

    ax.add_feature(shape_feature)

    ax.set_xticks(np.arange(-180, 180 + 60, 60), crs=proj)
    ax.set_xticks(np.arange(-180, 180 + 30, 30), minor=True, crs=proj)
    ax.set_yticks(np.arange(-90, 90 + 30, 30), crs=proj)
    ax.set_yticks(np.arange(-90, 90 + 15, 15), minor=True, crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis='both', which='major', length=2, width=0.5, color='black', labelcolor='black', pad=1)
    ax.tick_params(axis='both', which='minor', length=1, width=0.5, color='black')
    ax.spines['geo'].set_edgecolor('black') 

    color_low = bounds[0, i]
    if i == 0:
        color_mid = 1
    else:
        color_mid = bounds[1, i]
    color_high = bounds[2, i]

    col_bounds = np.linspace(color_low, color_mid, 7)
    col_bounds = np.append(col_bounds, np.linspace(color_mid, color_high, 7)[1:])
    cb = []
    cb_val = np.linspace(1, 0, len(col_bounds))
    for j in range(len(cb_val)):
        if i == 0:
            cb.append(mpl.cm.RdBu_r(cb_val[j])) #'RdYlBu_r'
        else:
            cb.append(mpl.cm.RdBu(cb_val[j]))
        cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), 
                                                                          cb)), N=1000)

    norm = mpl.colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))
    name = result.columns[i]
    x = result[name].values
    im=ax.scatter(obs['lon'].values, obs['lat'].values, c=x, s=4,
                   norm=norm, cmap=cmap_cus, zorder=3, transform=ccrs.PlateCarree())
    
    char = fig.colorbar(
        im,
        ax=ax,
        ticks=[color_low, color_mid, color_high],
        extend='both',
        shrink=0.8,
        aspect=20,
        pad=0.03,
        orientation='vertical'
    )

    if i == 0:
        char.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    else:
        char.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

    char.ax.tick_params(direction='in', size=2, width=0.5, labelsize=7)
    
    ax.text(0.01, 1.03, labels[i], transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='center', ha='center')


out_png = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_S8.png'
plt.savefig(out_png, dpi=600)

plt.show()