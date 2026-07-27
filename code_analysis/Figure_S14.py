import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import mean_squared_error

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
plt.rcParams.update({'ytick.major.size': 1.5})
plt.rcParams.update({'xtick.major.size': 1.5})


def paired_sign_test(paired_data, data_cols):
    p_values = pd.DataFrame(np.nan, index=data_cols, columns=data_cols)
    rmse_values = pd.DataFrame(np.nan, index=data_cols, columns=data_cols)
    greater_percent_values = pd.DataFrame(np.nan, index=data_cols, columns=data_cols)
    n_values = pd.DataFrame(np.nan, index=data_cols, columns=data_cols)

    for row_name in data_cols:
        for col_name in data_cols:
            sub = paired_data[[row_name, col_name]].dropna()
            diff = sub[row_name].values - sub[col_name].values
            n_pos = np.sum(diff > 0)
            n_neg = np.sum(diff < 0)
            n = n_pos + n_neg
            n_values.loc[row_name, col_name] = n

            if row_name == col_name:
                p_values.loc[row_name, col_name] = 1
                rmse_values.loc[row_name, col_name] = 0
                greater_percent_values.loc[row_name, col_name] = 50
            else:
                if n == 0:
                    p_values.loc[row_name, col_name] = 1
                    greater_percent_values.loc[row_name, col_name] = 50
                else:
                    p_values.loc[row_name, col_name] = st.binomtest(
                        n_pos,
                        n,
                        p=0.5,
                        alternative='two-sided'
                    ).pvalue
                    greater_percent_values.loc[row_name, col_name] = n_pos / n * 100
                rmse_values.loc[row_name, col_name] = mean_squared_error(
                    sub[row_name].values,
                    sub[col_name].values
                ) ** 0.5

    return p_values, rmse_values, greater_percent_values, n_values


def plot_heatmap(ax, p_values, rmse_values, greater_percent_values, data_cols, panel, title, vmax,
                 note=None, yticklabels=None, ytick_rotation=0):
    plot_values = rmse_values.astype(float).values

    cmap = plt.cm.Blues.copy()

    im = ax.imshow(
        plot_values,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(data_cols)))
    ax.set_yticks(np.arange(len(data_cols)))
    ax.set_xticklabels(data_cols, rotation=0, ha='center', fontsize=6)
    if yticklabels is None:
        ax.set_yticklabels(data_cols, fontsize=7)
    else:
        ax.set_yticklabels(yticklabels, fontsize=7)
    for label in ax.get_yticklabels():
        label.set_rotation(ytick_rotation)
        label.set_ha('right')
        label.set_va('center')
    ax.tick_params(axis='both', which='both', length=0)

    ax.set_title(title, fontsize=8, fontweight='bold', loc='center', pad=4)
    ax.text(-0.1, 1.00, panel, transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='left', va='bottom')

    for i in range(len(data_cols)):
        for j in range(len(data_cols)):
            if i == j:
                ax.text(
                    j, i, '-',
                    ha='center',
                    va='center',
                    fontsize=7,
                    color='0.45'
                )
                continue

            p_value = p_values.iloc[i, j]
            rmse = rmse_values.iloc[i, j]
            greater_percent = greater_percent_values.iloc[i, j]
            txt_color = 'white' if rmse / vmax > 0.45 else 'black'

            if np.isclose(p_value, 1):
                direction_txt = r'$\approx$'
            else:
                direction_txt = '>' if greater_percent >= 50 else '<'

            if p_value < 0.01:
                p_txt = r'p<0.01$^*$'
            else:
                p_txt = f'p={p_value:.3f}'

            ax.text(
                j, i, f'{direction_txt}\n{p_txt}\nRMSE={rmse:.1f}',
                ha='center',
                va='center',
                fontsize=6,
                color=txt_color
            )

    if note is not None:
        ax.text(
            0.5, -0.42,
            note,
            transform=ax.transAxes,
            fontsize=7,
            ha='center',
            va='top'
        )

    return im


var = 'mass_remaining'
path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/pygem_oggm/'


# %% add data for global glaciers
pygem_a = pd.read_csv(
    path + 'PyGEM_global_mass_median_a.csv'
)

pygem_k = pd.read_csv(
    path + 'PyGEM_global_mass_median_k.csv'
)


# %% add data for marine-terminating glaciers
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


# %% paired data
paired_a = (
    pygem_a[['gcm', 'period_scenario', var]]
    .rename(columns={var: r'use median $\alpha$'})
    .merge(
        pygem_k[['gcm', 'period_scenario', var]]
        .rename(columns={var: r'use median $k$'}),
        on=['gcm', 'period_scenario'],
        how='inner'
    )
)

paired_bcd_l = []
for cls in ['MT_all', 'MT_outAntarc', 'MT_inAntarc']:
    mt_a_sub = mt_a[mt_a['region_class'] == cls].copy()
    mt_k_sub = mt_k[mt_k['region_class'] == cls].copy()
    paired_bcd_l.append(
        mt_a_sub[['gcm', 'period_scenario', var]]
        .rename(columns={var: r'use median $\alpha$'})
        .merge(
            mt_k_sub[['gcm', 'period_scenario', var]]
            .rename(columns={var: r'use median $k$'}),
            on=['gcm', 'period_scenario'],
            how='inner'
        )
    )


# %% paired tests
cols = [r'use median $\alpha$', r'use median $k$']

p_a, rmse_a, greater_a, n_a = paired_sign_test(paired_a, cols)
p_b, rmse_b, greater_b, n_b = paired_sign_test(paired_bcd_l[0], cols)
p_c, rmse_c, greater_c, n_c = paired_sign_test(paired_bcd_l[1], cols)
p_d, rmse_d, greater_d, n_d = paired_sign_test(paired_bcd_l[2], cols)

rmse_vmax = np.nanmax([
    rmse_a.values[np.tril_indices_from(rmse_a.values, k=-1)].max(),
    rmse_b.values[np.tril_indices_from(rmse_b.values, k=-1)].max(),
    rmse_c.values[np.tril_indices_from(rmse_c.values, k=-1)].max(),
    rmse_d.values[np.tril_indices_from(rmse_d.values, k=-1)].max(),
])


# %% Create plot
fig_width_inch = 5.7
fig = plt.figure(figsize=(fig_width_inch, 4.5), dpi=600)

gs = GridSpec(
    2, 2, figure=fig,
    left=0.13, right=0.90, bottom=0.08, top=0.96,
    wspace=0.35, hspace=0.30,
    width_ratios=[1, 1],
    height_ratios=[1, 1]
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

im = plot_heatmap(
    ax_a,
    p_a,
    rmse_a,
    greater_a,
    cols,
    'A',
    'Global glaciers',
    rmse_vmax,
    #note=f'n={len(paired_a)}'
)

plot_heatmap(
    ax_b,
    p_b,
    rmse_b,
    greater_b,
    cols,
    'B',
    'Global marine-terminating glaciers',
    rmse_vmax,
    #note=f'n={len(paired_bcd_l[0])}',
    yticklabels=cols
)

plot_heatmap(
    ax_c,
    p_c,
    rmse_c,
    greater_c,
    cols,
    'C',
    'Marine-terminating glaciers outside\nAntarctic & Subantarctic',
    rmse_vmax,
    #note=f'n={len(paired_bcd_l[1])}',
    yticklabels=cols
)

plot_heatmap(
    ax_d,
    p_d,
    rmse_d,
    greater_d,
    cols,
    'D',
    'Marine-terminating glaciers in\nAntarctic & Subantarctic',
    rmse_vmax,
    #note=f'n={len(paired_bcd_l[2])}',
    yticklabels=cols
)

cax = fig.add_axes([0.92, 0.25, 0.015, 0.52])
cbar = fig.colorbar(
    im,
    cax=cax,
    orientation='vertical'
)
cbar.set_label('RMSE', fontsize=7)
cbar.ax.tick_params(direction='in', size=2, width=0.5, labelsize=7)

out_pdf = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_S14.png'
plt.savefig(out_pdf, dpi=600)

plt.show()
