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
    ax.set_xticklabels(data_cols, rotation=20, ha='center', fontsize=6)
    if yticklabels is None:
        ax.set_yticklabels(data_cols, fontsize=6)
    else:
        ax.set_yticklabels(yticklabels, fontsize=6)
    for label in ax.get_yticklabels():
        label.set_rotation(ytick_rotation)
        label.set_ha('right')
        label.set_va('center')
    ax.tick_params(axis='both', which='both', length=0)

    #ax.set_title(title, fontsize=7, fontweight='bold', loc='center', pad=4)
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
glaciermip3_path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/GlacierMIP3/'


# %% add data for Figure_S9A
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

pygem_a_no_era5 = pygem_a[
    pygem_a['gcm'].astype(str).str.lower() != 'era5'
].copy()


# %% add data for Figure_S9B
glaciermip3_add_this_study = pd.read_csv(
    glaciermip3_path + 'GlacierMIP3_add_this_study_median.csv'
)

glaciermip3_add_this_study = glaciermip3_add_this_study[
    ~np.isclose(glaciermip3_add_this_study['temp_ch_ipcc'], 1.2)
].copy()


# %% paired data
paired_a = (
    pygem_a_no_era5[['gcm', 'period_scenario', var]]
    .rename(columns={var: 'This study'})
    .merge(
        glaciermip3_median[
            ['gcm', 'period_scenario', 'median_mass_percent_rel_2020']
        ].rename(columns={
            'median_mass_percent_rel_2020': 'GlacierMIP3\nensemble median'
        }),
        on=['gcm', 'period_scenario'],
        how='inner'
    )
    .merge(
        glaciermip3_pygem_oggm[
            ['gcm', 'period_scenario', 'mass_percent_rel_2020']
        ].rename(columns={
            'mass_percent_rel_2020': 'GlacierMIP3\nPyGEM-OGGM'
        }),
        on=['gcm', 'period_scenario'],
        how='inner'
    )
)

paired_b = (
    glaciermip3_median[
        ['temp_ch_ipcc', 'median_mass_percent_rel_2020']
    ].rename(columns={
        'median_mass_percent_rel_2020': 'GlacierMIP3\nensemble median'
    })
    .merge(
        glaciermip3_add_this_study[
            ['temp_ch_ipcc', 'mass_remaining']
        ].rename(columns={
            'mass_remaining': 'GlacierMIP3 +\nthis study'
        }),
        on='temp_ch_ipcc',
        how='inner'
    )
)


# %% paired sign tests
cols_a = [
    'This study',
    'GlacierMIP3\nensemble median',
    'GlacierMIP3\nPyGEM-OGGM',
]

cols_b = [
    'GlacierMIP3\nensemble median',
    'GlacierMIP3 +\nthis study',
]

p_a, rmse_a, greater_a, n_a = paired_sign_test(paired_a, cols_a)
p_b, rmse_b, greater_b, n_b = paired_sign_test(paired_b, cols_b)

rmse_vmax = np.nanmax([
    rmse_a.values[np.tril_indices_from(rmse_a.values, k=-1)].max(),
    rmse_b.values[np.tril_indices_from(rmse_b.values, k=-1)].max(),
])


# %% Create plot
fig = plt.figure(figsize=(5.7, 2.26), dpi=600)

gs = GridSpec(
    1, 2, figure=fig,
    left=0.10, right=0.90, bottom=0.2, top=0.95,
    wspace=0.30,
    width_ratios=[1, 1]
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

im = plot_heatmap(
    ax_a,
    p_a,
    rmse_a,
    greater_a,
    cols_a,
    'A',
    '',
    rmse_vmax,
    #note=f'n={len(paired_a)}'
)

plot_heatmap(
    ax_b,
    p_b,
    rmse_b,
    greater_b,
    cols_b,
    'B',
    '',
    rmse_vmax,
    #note=f'n={len(paired_b.dropna())}'
)

cax = fig.add_axes([0.92, 0.27, 0.015, 0.50])
cbar = fig.colorbar(
    im,
    cax=cax,
    orientation='vertical'
)
cbar.set_label('RMSE', fontsize=7)
cbar.ax.tick_params(direction='in', size=2, width=0.5, labelsize=7)

out_png = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_S11.png'
plt.savefig(out_png, dpi=600)

plt.show()
