import numpy as np
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


path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/data/'


# %% add data
wgms_test = pd.read_csv(
    path + 'WGMS_test.csv'
)

wgms_test = wgms_test.dropna(subset=['start_year']).copy()

per_glacier = pd.read_csv(
    path + 'wgms_test_per_glacier.csv'
)

year_cols = [
    c for c in per_glacier.columns
    if str(c).startswith('AAR_steady_')
]
start_years = np.array([int(str(c).replace('AAR_steady_', '')) for c in year_cols])

trend_l = []
for i in range(0, len(per_glacier)):
    y = per_glacier.loc[i, year_cols].values.astype(float)
    valid = np.isnan(y) == False
    if valid.sum() >= 2:
        coef = np.polyfit(start_years[valid], y[valid], 1)
        trend_l.append(coef[0])
    else:
        trend_l.append(np.nan)

per_glacier['AAR_steady_trend_slope'] = trend_l

per_glacier = per_glacier.dropna(
    subset=['rgi_area_km2', 'AAR_steady_trend_slope']
).copy()


# %% Create plot
fig = plt.figure(figsize=(5.7, 2.26))

gs = GridSpec(1, 2, figure=fig,left=0.07, right=0.99, bottom=0.15, top=0.98, wspace=0.22, width_ratios=[1, 1])

ax_a = fig.add_subplot(gs[0, 0])
ax_c = fig.add_subplot(gs[0, 1])

color_mean = '#489FE3'
color_median = '#DC6D57'
color_awm = '#5266B0'


############################################################################ Figure_S11A
x = wgms_test['start_year'].values

y = wgms_test['AAR_steady_median'].values
yerr = wgms_test['AAR_steady_MAD'].values

ax_a.errorbar(
    x, y,
    yerr=yerr,
    fmt='s',
    color=color_median,
    ecolor=color_median,
    elinewidth=0.5,
    capsize=1,
    capthick=0.5,
    markersize=2.5,
    markeredgecolor='black',
    markeredgewidth=0.25,
    alpha=0.85,
    label='median ± MAD',
    zorder=10
)

coef = np.polyfit(x, wgms_test['AAR_steady_median'].values, 1)
trend = np.poly1d(coef)

ax_a.plot(
    x, trend(x),
    color=color_median,
    linestyle='--',
    linewidth=0.8,
    zorder=30,
    label='median trend'
)

y = wgms_test['AAR_steady_mean'].values
yerr = wgms_test['AAR_steady_std'].values

ax_a.errorbar(
    x, y,
    yerr=yerr,
    fmt='o',
    color=color_mean,
    ecolor=color_mean,
    elinewidth=0.5,
    capsize=1,
    capthick=0.5,
    markersize=2.5,
    markeredgecolor='black',
    markeredgewidth=0.25,
    alpha=0.85,
    label='mean ± std',
    zorder=10
)

coef = np.polyfit(x, wgms_test['AAR_steady_mean'].values, 1)
trend = np.poly1d(coef)

ax_a.plot(
    x, trend(x),
    color=color_mean,
    linestyle='--',
    linewidth=0.8,
    zorder=30,
    label='mean trend'
)

ax_a.set_xlabel('Start year', fontsize=8)
ax_a.set_ylabel(r'AAR$_0$', fontsize=8)
ax_a.text(0.01, 0.93, 'A', transform=ax_a.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')

ax_a.legend(
    loc='upper right',
    frameon=False,
    fontsize=7,
    handletextpad=0.4,
    borderpad=-0.1
)
# add results
for label_year in [1980, 1985, 1990, 1995, 2000]:
    value_mean = wgms_test.loc[
        wgms_test['start_year'] == label_year,
        'AAR_steady_mean'
    ].iloc[0]

    ax_a.text(
        label_year, value_mean-0.04,
        f'{value_mean:.2f}',
        fontsize=7,
        ha='center',
        va='bottom',
        color=color_mean
    )

    value_median = wgms_test.loc[
        wgms_test['start_year'] == label_year,
        'AAR_steady_median'
    ].iloc[0]

    ax_a.text(
        label_year, value_median + 0.01,
        f'{value_median:.2f}',
        fontsize=7,
        ha='center',
        va='bottom',
        color=color_median
    )

ax_a.set_ylim(0.3, 0.8)
ax_a.set_xlim(1979,
              2001)
ax_a.text(0.98, 0.02, f"n={int(wgms_test['n'].iloc[0])}",
          transform=ax_a.transAxes, fontsize=7,
          ha='right', va='bottom')


############################################################################ Figure_S11B
ax_c.scatter(
    per_glacier['rgi_area_km2'].values,
    per_glacier['AAR_steady_trend_slope'].values,
    s=8,
    color='#7A7A7A',
    edgecolor='black',
    linewidths=0.25,
    alpha=0.65,
    zorder=10
)

ax_c.axhline(
    y=0,
    color='black',
    linestyle=':',
    linewidth=0.8,
    zorder=20
)

ax_c.set_xscale('log')
ax_c.set_xlabel(r'RGI area (km$^2$)', fontsize=8)
ax_c.set_ylabel(r'Linear trend for AAR$_0$ (glacier level)', fontsize=8)
ax_c.text(0.01, 0.93, 'B', transform=ax_c.transAxes,
          fontsize=9, fontweight='bold', ha='left', va='bottom')

ax_c.text(0.98, 0.02, f'n={len(per_glacier)}',
          transform=ax_c.transAxes, fontsize=7,
          ha='right', va='bottom')

out_pdf = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_S12.png'
plt.savefig(out_pdf, dpi=600)

plt.show()
