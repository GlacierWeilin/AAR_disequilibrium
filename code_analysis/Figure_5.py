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
glaciermip3_path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/GlacierMIP3/'


# %% add data for Figure_5A
pygem_a = pd.read_csv(
    path + 'PyGEM_global_mass_median_a.csv'
)

glaciermip3_median = pd.read_csv(
    glaciermip3_path + 'fig1b_scatter_median_by_temp.csv'
)

glaciermip3_pygem_oggm = pd.read_csv(
    glaciermip3_path + 'fig1b_scatter_by_temp_model.csv'
)

glaciermip3_pygem_oggm = glaciermip3_pygem_oggm[
    glaciermip3_pygem_oggm['model_author'] == 'PyGEM-OGGM_v13'
].copy()


# %% add lowess fit data for Figure_5A
pygem_a_lowess = pd.read_csv(
    path + f'PyGEM_global_mass_{var}_median_a_lowess_fit.csv'
)

glaciermip3_median_lowess = pd.read_csv(
    glaciermip3_path + 'lowess_fit_rel_2020_101yr_avg_steady_state_Feb12_2024.csv'
)

glaciermip3_median_lowess = glaciermip3_median_lowess[
    glaciermip3_median_lowess['region'].astype(str) == 'All'
].copy()

glaciermip3_pygem_oggm_lowess = pd.read_csv(
    glaciermip3_path + 'lowess_fit_rel_2020_101yr_avg_steady_state_Feb12_2024_per_glac_model.csv'
)

glaciermip3_pygem_oggm_lowess = glaciermip3_pygem_oggm_lowess[
    (glaciermip3_pygem_oggm_lowess['region'].astype(str) == 'All') &
    (glaciermip3_pygem_oggm_lowess['model_author'] == 'PyGEM-OGGM_v13')
].copy()


# %% add data for Figure_5B
glaciermip3_add_this_study = pd.read_csv(
    glaciermip3_path + 'GlacierMIP3_add_this_study_median.csv'
)

glaciermip3_add_this_study_lowess = pd.read_csv(
    glaciermip3_path + 'GlacierMIP3_add_this_study_lowess_fit.csv'
)


# %% colors
colors_a = ['#489FE3', '#F09137', '#5266B0']
colors_b = 'grey'


# %% Create plot
fig = plt.figure(figsize=(5.7, 2.26))

gs = GridSpec(1, 2, figure=fig,left=0.07, right=0.99, bottom=0.125, top=0.96, wspace=0.03, width_ratios=[1, 1])

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)

ax_b.set_ylabel('')
ax_b.tick_params(labelleft=False)

for ax in [ax_a, ax_b]:
    ax.set_xlim(-0.1, 6.85)
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 50, 100, 150])


############################################################################ Figure_6A
pygem_a_era5 = pygem_a[pygem_a['gcm'].astype(str).str.lower() == 'era5']
pygem_a_gcm = pygem_a[pygem_a['gcm'].astype(str).str.lower() != 'era5']

ax_a.scatter(
    pygem_a_gcm['temp_ch_ipcc'].values,
    pygem_a_gcm[var].values,
    s=8,
    color=colors_a[0],
    alpha=0.42,
    linewidths=0,
    zorder=5
)

ax_a.scatter(
    pygem_a_era5['temp_ch_ipcc'].values,
    pygem_a_era5[var].values,
    marker='*',
    s=38,
    color=colors_a[0],
    edgecolor='black',
    linewidths=0.35,
    alpha=0.95,
    zorder=30
)

ax_a.plot(
    pygem_a_lowess['temp_ch_ipcc'].values,
    pygem_a_lowess['0.5'].values,
    color=colors_a[0],
    linewidth=1,
    zorder=10
)

ax_a.fill_between(
    pygem_a_lowess['temp_ch_ipcc'].values,
    pygem_a_lowess['0.17'].values,
    pygem_a_lowess['0.83'].values,
    color=colors_a[0],
    alpha=0.15,
    linewidth=0,
    zorder=1
)

ax_a.scatter(
    glaciermip3_median['temp_ch_ipcc'].values,
    glaciermip3_median['median_mass_percent_rel_2020'].values,
    s=8,
    color=colors_a[1],
    alpha=0.42,
    linewidths=0,
    zorder=5,
)

ax_a.plot(
    glaciermip3_median_lowess['temp_ch'].values,
    glaciermip3_median_lowess['0.5'].values,
    color=colors_a[1],
    linewidth=1,
    zorder=10
)

ax_a.fill_between(
    glaciermip3_median_lowess['temp_ch'].values,
    glaciermip3_median_lowess['0.17'].values,
    glaciermip3_median_lowess['0.83'].values,
    color=colors_a[1],
    alpha=0.15,
    linewidth=0,
    zorder=1
)

ax_a.scatter(
    glaciermip3_pygem_oggm['temp_ch_ipcc'].values,
    glaciermip3_pygem_oggm['mass_percent_rel_2020'].values,
    s=8,
    color=colors_a[2],
    alpha=0.42,
    linewidths=0,
    zorder=5
)

ax_a.plot(
    glaciermip3_pygem_oggm_lowess['temp_ch'].values,
    glaciermip3_pygem_oggm_lowess['0.5'].values,
    color=colors_a[2],
    linewidth=1,
    zorder=10
)

ax_a.set_xlabel('ΔT (°C)', fontsize=8)
ax_a.set_ylabel('Steady-state glacier mass (% rel. to 2020)', fontsize=8)
ax_a.text(0.01, 0.93, 'A', transform=ax_a.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')

ax_a.text(3.0, 145, 'This study', color=colors_a[0],
          fontsize=7, ha='left', va='top')
ax_a.text(3.0, 136, r'GlacierMIP3 ensemble median$^{18}$', color=colors_a[1],
          fontsize=7, ha='left', va='top')
ax_a.text(3.0, 127, r'GlacierMIP3 PyGEM-OGGM$^{18}$', color=colors_a[2],
          fontsize=7, ha='left', va='top')

for i, color in enumerate(colors_a):
    ax_a.scatter(
        2.8,
        141.5 - 9*i,
        s=8,
        color=color,
        alpha=0.42,
        linewidths=0,
        zorder=5
    )


############################################################################ Figure_6B
ax_b.scatter(
    glaciermip3_add_this_study['temp_ch_ipcc'].values,
    glaciermip3_add_this_study[var].values,
    s=8,
    color=colors_b,
    alpha=0.42,
    linewidths=0,
    zorder=5
)

ax_b.plot(
    glaciermip3_add_this_study_lowess['temp_ch_ipcc'].values,
    glaciermip3_add_this_study_lowess['0.5'].values,
    color=colors_b,
    linewidth=1,
    zorder=10
)

ax_b.fill_between(
    glaciermip3_add_this_study_lowess['temp_ch_ipcc'].values,
    glaciermip3_add_this_study_lowess['0.17'].values,
    glaciermip3_add_this_study_lowess['0.83'].values,
    color=colors_b,
    alpha=0.15,
    linewidth=0,
    zorder=1
)

ax_b.scatter(
    glaciermip3_median['temp_ch_ipcc'].values,
    glaciermip3_median['median_mass_percent_rel_2020'].values,
    s=8,
    color=colors_a[1],
    alpha=0.42,
    linewidths=0,
    zorder=5,
)

ax_b.plot(
    glaciermip3_median_lowess['temp_ch'].values,
    glaciermip3_median_lowess['0.5'].values,
    color=colors_a[1],
    linewidth=1,
    zorder=10
)

ax_b.fill_between(
    glaciermip3_median_lowess['temp_ch'].values,
    glaciermip3_median_lowess['0.17'].values,
    glaciermip3_median_lowess['0.83'].values,
    color=colors_a[1],
    alpha=0.15,
    linewidth=0,
    zorder=1
)

ax_b.set_xlabel('ΔT (°C)', fontsize=8)
ax_b.text(0.01, 0.93, 'B', transform=ax_b.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')

ax_b.scatter(
    [3.7], [137],
    marker='o',
    s=8,
    color=colors_b,
    alpha=0.42,
    linewidths=0,
    zorder=100
)

ax_b.text(3.9, 145, 'Ensemble median for\nGlacierMIP3 + this study', color=colors_b,
          fontsize=7, ha='left', va='top')

##############################################################################
### Legend

ax_b.fill_between(
    [3.8, 4.2], [80, 80], [90, 90],
    color=colors_b,
    alpha=0.15,
    linewidth=0
)
ax_b.plot(
    [3.8, 4.2], [85, 85],
    color=colors_b,
    linewidth=1,
    zorder=100
)
ax_b.text(4.3, 85, 'LOWESS fit\n50% [17% to 83%]',
          fontsize=7, ha='left', va='center')

out_png = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_5.png'
plt.savefig(out_png, dpi=600)

plt.show()
