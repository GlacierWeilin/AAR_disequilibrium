import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

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
plt.rcParams['legend.fontsize'] = 7

var = 'mass_remaining'
path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/pygem_oggm/'

region_names = {
    0: 'Global',
    1: '(1) Alaska',
    2: '(2) W Canada & US',
    3: '(3) Arctic Canada North',
    4: '(4) Arctic Canada South',
    5: '(5) Greenland Periphery',
    6: '(6) Iceland',
    7: '(7) Svalbard',
    8: '(8) Scandinavia',
    9: '(9) Russian Arctic',
    10: '(10) North Asia',
    11: '(11) Central Europe',
    12: '(12) Caucasus & Middle East',
    13: '(13) Central Asia',
    14: '(14) South Asia West',
    15: '(15) South Asia East',
    16: '(16) Low Latitudes',
    17: '(17) Southern Andes',
    18: '(18) New Zealand',
    19: '(19) Antarctic & Subantarctic',
}

regional_warming = {
    0: 1.0,  # Global mass
    1: 1.5,    # Alaska
    2: 1.3,    # W Canada & US
    3: 2.4,    # Arctic Canada North
    4: 2.2,    # Arctic Canada South
    5: 1.9,    # Greenland Periphery
    6: 0.9,    # Iceland
    7: 3.0,    # Svalbard
    8: 1.3,    # Scandinavia
    9: 3.1,    # Russian Arctic
    10: 1.7,   # North Asia
    11: 1.4,   # Central Europe
    12: 1.4,   # Caucasus & Middle East
    13: 1.4,   # Central Asia
    14: 1.4,   # South Asia West
    15: 1.2,   # South Asia East
    16: 1.2,   # Low Latitudes
    17: 0.7,   # Southern Andes
    18: 0.8,   # New Zealand
    19: 1.1,   # Antarctic & Subantarctic
}

#%%
regions = np.arange(1, 20, 1)
# Mass remaining
# global
df = pd.read_csv(path + 'PyGEM_global_mass_median_a.csv')

temp_ch_ipcc = df['temp_ch_ipcc'].values
mass_remaining = np.full((len(temp_ch_ipcc), len(regions) + 1), np.nan)

mass_remaining[:, 0] = df[var].values

# regional
for i, region in enumerate(regions):

    if region in {1, 3, 4, 5, 7, 9, 17, 19}:
        df = pd.read_csv(path + f'PyGEM_regional_mass_{region:02d}_median_a.csv')
    else:
        df = pd.read_csv(path + f'PyGEM_regional_mass_{region:02d}_median_k.csv')

    mass_remaining[:, i + 1] = df[var].values

#%% Lowess fit
# global
df = pd.read_csv(path + f'PyGEM_global_mass_{var}_median_a_lowess_fit.csv')
temp = df['temp_ch_ipcc'].values
lowess = np.full((len(temp), len(regions) + 1, 3), np.nan)

lowess[:, 0, 0] = df['0.17'].values
lowess[:, 0, 1] = df['0.5'].values
lowess[:, 0, 2] = df['0.83'].values

# regional
for i, region in enumerate(regions):
    region_2d = f'{region:02d}'

    if region in {1, 3, 4, 5, 7, 9, 17, 19}:
        df = pd.read_csv(path + f'PyGEM_regional_mass_{region_2d}_{var}_median_a_lowess_fit.csv')
    else:
        df = pd.read_csv(path + f'PyGEM_regional_mass_{region_2d}_{var}_median_k_lowess_fit.csv')

    lowess[:, i + 1, 0] = df['0.17'].values
    lowess[:, i + 1, 1] = df['0.5'].values
    lowess[:, i + 1, 2] = df['0.83'].values

# Get mass_2020 from GlacierMIP3
mass = pd.read_csv(path + '../GlacierMIP3/table_S3.csv', index_col=0)
mass_2020 = mass['Glacier mass in 2020$^b$ (Gt)']

region_mass_names = {
    0: 'Global',
    1: 'Alaska (01)',
    2: 'W Canada & US (02)',
    3: 'Arctic Canada N (03)',
    4: 'Arctic Canada S (04)',
    5: 'Greenland Periphery (05)',
    6: 'Iceland (06)',
    7: 'Svalbard (07)',
    8: 'Scandinavia (08)',
    9: 'Russian Arctic (09)',
    10: 'North Asia (10)',
    11: 'Central Europe (11)',
    12: 'Caucasus & Middle East (12)',
    13: 'Central Asia (13)',
    14: 'South Asia W (14)',
    15: 'South Asia E (15)',
    16: 'Low Latitudes (16)',
    17: 'Southern Andes (17)',
    18: 'New Zealand (18)',
    19: 'Sub- & Antarctic Islands (19)',
}

global_mass_2020 = mass_2020.loc['Global']

mass_2020_ordered = np.array([
    mass_2020.loc[region_mass_names[i]]
    for i in range(20)
])

mass_2020_percent = mass_2020_ordered / global_mass_2020 * 100

#%% Create plot
fig = plt.figure(figsize=(6.3, 6.3))

gs = GridSpec(
    5, 5,
    figure=fig,
    left=0.06,
    right=0.99,
    bottom=0.05,
    top=0.975,
    wspace=0.08,
    hspace=0.18
)

ax_dict = {}
ax_pos = {}

# Global takes 2 x 2 grids
ax_dict[0] = fig.add_subplot(gs[0:2, 0:2])
ax_pos[0] = (0, 0)

# Other regions fill remaining cells
positions = []
for r in range(5):
    for c in range(5):
        if r in [0, 1] and c in [0, 1]:
            continue
        positions.append((r, c))

for region, (r, c) in zip(range(1, 20), positions):
    ax_dict[region] = fig.add_subplot(gs[r, c])
    ax_pos[region] = (r, c)

for region in range(20):
    ax = ax_dict[region]
    r, c = ax_pos[region]

    ax.scatter(
        temp_ch_ipcc,
        mass_remaining[:, region],
        s=5,
        color='#489FE3',
        alpha=0.8,
        linewidths=0,
        zorder=3
    )

    ax.scatter(
        temp_ch_ipcc[0],
        mass_remaining[0, region],
        s=30,
        marker='*',
        color='#DC6D57',
        alpha=1,
        linewidths=0.3,
        zorder=5,
        edgecolor='black',
    )

    ax.plot(
        temp,
        lowess[:, region, 1],
        color='black',
        linewidth=0.8,
        zorder=10
    )

    ax.fill_between(
        temp,
        lowess[:, region, 0],
        lowess[:, region, 2],
        color='lightgrey',
        alpha=0.7,
        linewidth=0,
        zorder=1
    )
    
    if region == 0:
        ax.text(
            0.0, 1.05,
            region_names[region],
            transform=ax.transAxes,
            fontsize=9,
            fontweight='bold',
            ha='left',
            va='top'
        )
    elif region == 13:
        ax.text(
            0.2, 1.1,
            region_names[region],
            transform=ax.transAxes,
            fontsize=7,
            fontweight='bold',
            ha='left',
            va='top'
        )
    else:
        ax.text(
            0.0, 1.1,
            region_names[region],
            transform=ax.transAxes,
            fontsize=7,
            fontweight='bold',
            ha='left',
            va='top'
        )

    if region == 0:
        txt = (f'{global_mass_2020:.0f} Gt\n'
               f'{mass_2020_percent[region]:.1f}% of global mass (2020)\n'
               f'{regional_warming[region]:.1f} x global warming'
               )
        
    else:
        p = mass_2020_percent[region]
        if round(p, 1) == 0.0:
            p_txt = '<0.1%'
        else:
            p_txt = f'{p:.1f}%'

        txt = (
            f'{p_txt}\n'
            f'{regional_warming[region]:.1f} x'
        )
    
    if region == 0:
        ax.text(
            0.96, 0.78,
            txt,
            transform=ax.transAxes,
            fontsize=9,
            ha='right',
            va='bottom'
        )
    else:
        ax.text(
            0.96, 0.75,
            txt,
            transform=ax.transAxes,
            fontsize=7,
            ha='right',
            va='bottom'
        )
    
    

    ax.set_xlim(-0.1, 6.85)
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 50, 100, 150])
    ax.tick_params(direction='out')

    # Only left column has y-axis labels / tick labels
    if c == 0:
        ax.set_ylabel('', fontsize=8)
        ax.tick_params(axis='y', labelleft=True)
    else:
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)

    # Only bottom row has x-axis labels / tick labels
    if r == 4:
        ax.set_xlabel('', fontsize=8)
        ax.tick_params(axis='x', labelbottom=True)
    else:
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelbottom=False)

# shared figure-level labels
fig.text(0.50, 0.015, 'ΔT (°C)', ha='center', va='center', fontsize=8, fontweight='bold')
fig.text(0.015, 0.50, 'Steady-state glacier mass (% rel. to 2020)', ha='center', va='center',
         rotation='vertical', fontsize=8, fontweight='bold')

# add legend
ax_legend = fig.add_subplot(gs[4, 3])
ax_legend.axis('off')

scatter_gcm = Line2D(
    [0], [0],
    marker='o',
    linestyle='none',
    markerfacecolor='#489FE3',
    markeredgecolor='none',
    markersize=3
)

scatter_era5 = Line2D(
    [0], [0],
    marker='*',
    linestyle='none',
    markerfacecolor='#DC6D57',
    markeredgecolor='black',
    markersize=6,
    markeredgewidth=0.3
)

lowess_patch = Patch(
    facecolor='lightgrey',
    edgecolor='none',
    alpha=0.55
)

lowess_line = Line2D(
    [0], [0],
    color='black',
    linewidth=0.8
)

handles = [
    scatter_gcm,
    scatter_era5,
    (lowess_patch, lowess_line),
]

labels = [
    'based on GCMs',
    'based on ERA5',
    'LOWESS fit\n50% [17% to 83%]',
]

ax_legend.legend(
    handles,
    labels,
    loc='center left',
    frameon=False,
    fontsize=8,
    handlelength=2,
    handleheight=2,
    handletextpad=0.6,
    labelspacing=0.2,
    borderpad=0,
    handler_map={
        tuple: HandlerTuple(ndivide=1)
    }
)

out_pdf = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_2.png'
plt.savefig(out_pdf, dpi=600)

plt.show()