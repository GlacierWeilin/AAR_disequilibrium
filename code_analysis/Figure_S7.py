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
    rmse_values = pd.DataFrame(np.nan, index=data_cols, columns=data_cols)
    greater_percent_values = pd.DataFrame(np.nan, index=data_cols, columns=data_cols)
    sign_p_values = pd.DataFrame(np.nan, index=data_cols, columns=data_cols)
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
                rmse_values.loc[row_name, col_name] = 0
                greater_percent_values.loc[row_name, col_name] = 50
                sign_p_values.loc[row_name, col_name] = 1
            else:
                rmse_values.loc[row_name, col_name] = mean_squared_error(
                    sub[row_name].values,
                    sub[col_name].values
                ) ** 0.5
                greater_percent_values.loc[row_name, col_name] = n_pos / n * 100
                sign_p_values.loc[row_name, col_name] = st.binomtest(
                    n_pos,
                    n,
                    p=0.5,
                    alternative='two-sided'
                ).pvalue

    return rmse_values, greater_percent_values, sign_p_values, n_values


def plot_heatmap(ax, rmse_values, greater_percent_values, sign_p_values, data_cols, panel, title, vmax,
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

            rmse = rmse_values.iloc[i, j]
            greater_percent = greater_percent_values.iloc[i, j]
            sign_p = sign_p_values.iloc[i, j]
            txt_color = 'white' if rmse / vmax > 0.45 else 'black'
            if np.isclose(sign_p, 1):
                direction_txt = r'$\approx$'
            else:
                direction_txt = '>' if greater_percent >= 50 else '<'

            if sign_p < 0.01:
                p_txt = r'p<0.01$^*$'
            else:
                p_txt = f'p={sign_p:.3f}'

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
tag = 'median_a'


# %% add data following Figure_3
area_result_l = []
for cla, class_name in zip(
    ['-1', '1-10', '10-100', '100-'],
    [r'<1 km$^2$', r'1-10 km$^2$', r'10-100 km$^2$', r'>100 km$^2$'],
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


# %% paired data
index_cols = ['experiment', 'gcm', 'period_scenario', 'temp_ch_ipcc']

area_order = [r'<1 km$^2$', r'1-10 km$^2$', r'10-100 km$^2$', r'>100 km$^2$']
tidewater_order = ['land-terminating', 'marine-terminating']

paired_area = (
    area_result
    .pivot_table(
        index=index_cols,
        columns='area_class',
        values=var,
        aggfunc='first'
    )
    .reset_index()
)

paired_tidewater = (
    tidewater_result
    .pivot_table(
        index=index_cols,
        columns='tidewater_class',
        values=var,
        aggfunc='first'
    )
    .reset_index()
)

paired_area_test = paired_area[
    paired_area['temp_ch_ipcc'] > 2.7
].copy()


# %% paired sign tests
rmse_a, greater_a, sign_p_a, n_a = paired_sign_test(paired_area_test, area_order)
rmse_b, greater_b, sign_p_b, n_b = paired_sign_test(paired_tidewater, tidewater_order)

rmse_vmax = np.nanmax([
    rmse_a.values[np.tril_indices_from(rmse_a.values, k=-1)].max(),
    rmse_b.values[np.tril_indices_from(rmse_b.values, k=-1)].max(),
])


# %% Create plot
fig = plt.figure(figsize=(5.7, 2.26), dpi=600)

gs = GridSpec(
    1, 2, figure=fig,
    left=0.07, right=0.90, bottom=0.1, top=0.95,
    wspace=0.30,
    width_ratios=[1, 1]
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

im = plot_heatmap(
    ax_a,
    rmse_a,
    greater_a,
    sign_p_a,
    area_order,
    'A',
    '',
    rmse_vmax,
    #note=f'n={len(paired_area)}'
)

plot_heatmap(
    ax_b,
    rmse_b,
    greater_b,
    sign_p_b,
    tidewater_order,
    'B',
    '',
    rmse_vmax,
    #note=f'n={len(paired_tidewater)}'
)

cax = fig.add_axes([0.92, 0.27, 0.015, 0.50])
cbar = fig.colorbar(
    im,
    cax=cax,
    orientation='vertical'
)
cbar.set_label('RMSE', fontsize=7)
cbar.ax.tick_params(direction='in', size=2, width=0.5, labelsize=7)

out_png = '/Users/wyan0065/Desktop/OGGM/disequilibrium/figures/figure_S7.png'
plt.savefig(out_png, dpi=600)

plt.show()
