'''
GlacierMIP-style LOWESS fit for mass_remaining, area_steady, and volume_steady.

Reference: Zekollari, H., Schuster, L., Maussion, F., Hock, R., Marzeion, B., Rounce, D.R., Compagno, L.,
Fujita, K., Huss, M., James, M., et al. (2025).
Glacier preservation doubled by limiting warming to 1.5°C versus 2.7°C.
Science 388, 979-983. https://doi.org/10.1126/science.adu4675.

1. Use x = temp_ch_ipcc and y = one of mass_remaining, area_steady, volume_steady.
2. Scan frac from 0.10 to 0.99 by 0.01.
3. First fit only the 0.5 quantile to choose frac.
4. Prefer fits that are non-negative and monotonically decreasing.
5. Refit selected frac with all requested quantiles.
'''

from pathlib import Path

import numpy as np
import pandas as pd
from moepy import lowess

lowess.tqdm = lambda x, *args, **kwargs: x


def glaciermip_style_lowess_fit(
    x,
    y,
    *,
    qs=None,
    frac_grid=None,
    preliminary_num_fits=500,
    final_num_fits=2000,
    robust_iters=2,
    eval_step=0.05,
):
    if qs is None:
        qs = [0.05, 0.17, 0.25, 0.5, 0.75, 0.83, 0.95]
    if frac_grid is None:
        frac_grid = np.arange(0.1, 1.0, 0.01)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    eval_x = np.arange(np.round(x.min(), 1), x.max() * 1.001, eval_step)
    x_pred = np.concatenate([eval_x, x])

    df_quantiles_l = []
    df_quantiles_ll = []

    for frac in frac_grid:
        df_quantiles = lowess.quantile_model(
            x,
            y,
            x_pred=x_pred,
            frac=float(frac),
            num_fits=preliminary_num_fits,
            robust_iters=robust_iters,
            qs=[0.5],
        )

        q = 0.5
        lowi = df_quantiles[q].copy()
        lowi[lowi < 0] = 0

        df_quantiles['frac'] = float(frac)
        df_quantiles['it'] = robust_iters
        df_quantiles['N'] = preliminary_num_fits
        df_quantiles['fit_opt'] = 'lowess_fit'
        df_quantiles['y'] = np.concatenate([np.repeat(np.nan, len(eval_x)), y])

        df_quantiles[f'min_{q}_diff'] = (
            df_quantiles[q].iloc[: len(eval_x) - 1].values
            - df_quantiles[q].iloc[1:len(eval_x)].values
        ).min()
        df_quantiles[f'min_{q}'] = df_quantiles[q].min()
        df_quantiles[f'min_{q}_diff_above_zero'] = (
            lowi.iloc[: len(eval_x) - 1].values
            - lowi.iloc[1:len(eval_x)].values
        ).min()
        df_quantiles['median_absolute_deviation'] = np.abs(
            df_quantiles.iloc[len(eval_x):]['y']
            - df_quantiles.iloc[len(eval_x):][0.5]
        ).median()
        df_quantiles['rmse'] = np.sqrt(
            np.mean(
                (
                    df_quantiles.iloc[len(eval_x):]['y']
                    - df_quantiles.iloc[len(eval_x):][0.5]
                )
                ** 2
            )
        )

        if np.all(df_quantiles[f'min_{q}_diff_above_zero'] >= 0):
            df_quantiles['algorithm_sel'] = 'only_decreasing'
            if df_quantiles[q].min() >= 0:
                df_quantiles['algorithm_sel'] = 'non_negative_and_decreasing'
        else:
            df_quantiles['algorithm_sel'] = 'not_selected'

        df_quantiles_l.append(df_quantiles)
        df_quantiles_ll.append(df_quantiles)

    df_quantiles_ll_concat = pd.concat(df_quantiles_ll)

    if (
        df_quantiles_ll_concat['min_0.5'].max() >= 0
        and len(
            df_quantiles_ll_concat.loc[
                df_quantiles_ll_concat.algorithm_sel
                == 'non_negative_and_decreasing'
            ]
        )
        >= 1
    ):
        _sel = df_quantiles_ll_concat.loc[
            df_quantiles_ll_concat.algorithm_sel == 'non_negative_and_decreasing'
        ]
        min_rmse = _sel['rmse'].min()
        sel = _sel.loc[_sel.rmse == min_rmse]
        sel = sel.sort_values('x')

    elif len(
        df_quantiles_ll_concat.loc[
            df_quantiles_ll_concat.algorithm_sel == 'only_decreasing'
        ]
    ) >= 1:
        _sel = df_quantiles_ll_concat.loc[
            df_quantiles_ll_concat.algorithm_sel == 'only_decreasing'
        ]
        max_min = _sel['min_0.5'].max()
        sel = _sel.loc[_sel['min_0.5'] == max_min]
        sel = sel.sort_values('x')
        assert len(sel) > 0

    else:
        _max_min_diff = df_quantiles_ll_concat.groupby('frac')[
            'min_0.5_diff_above_zero'
        ].min()
        frac = _max_min_diff.idxmax().round(3)
        _sel = df_quantiles_ll_concat.loc[
            df_quantiles_ll_concat.frac.round(3) == frac
        ]
        sel = _sel.sort_values('x')
        assert len(sel) > 0

    frac = sel['frac'].unique()[0]

    sel = lowess.quantile_model(
        x,
        y,
        x_pred=x_pred,
        frac=frac,
        num_fits=final_num_fits,
        robust_iters=robust_iters,
        qs=qs,
    )

    sel = sel.iloc[:len(eval_x)].copy()

    sel['frac'] = frac
    sel['it'] = robust_iters
    sel['N'] = preliminary_num_fits
    sel['final_num_fits'] = final_num_fits
    sel['fit_opt'] = 'lowess_fit'
    sel['y'] = np.nan
    sel = sel.sort_values('x')

    for qq in qs:
        sel.loc[sel[qq] < 0, qq] = 0

    pd_quantiles_concat = pd.concat(df_quantiles_l)
    return sel, pd_quantiles_concat, frac


def run_lowess_from_csv(
    input_csv,
    output_csv,
    *,
    trials_output_csv=None,
    x_col='temp_ch_ipcc',
    y_col='mass_remaining',
    qs=None,
    preliminary_num_fits=500,
    final_num_fits=2000,
    robust_iters=2,
):
    df = pd.read_csv(input_csv)

    missing_cols = [col for col in [x_col, y_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f'Missing required columns {missing_cols!r}. '
            f'Available columns are: {df.columns.tolist()}'
        )

    df = df.dropna(subset=[x_col, y_col])
    x = df[x_col].values
    y = df[y_col].values

    sel, pd_quantiles_concat, frac = glaciermip_style_lowess_fit(
        x,
        y,
        qs=qs,
        preliminary_num_fits=preliminary_num_fits,
        final_num_fits=final_num_fits,
        robust_iters=robust_iters,
    )

    sel = sel.reset_index().rename(columns={'x': x_col})
    pd_quantiles_concat = pd_quantiles_concat.reset_index().rename(columns={'x': x_col})

    sel['source_input'] = str(input_csv)
    sel['x_col'] = x_col
    sel['y_col'] = y_col

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(output_csv, index=False)

    if trials_output_csv is not None:
        trials_output_csv = Path(trials_output_csv)
        trials_output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd_quantiles_concat.to_csv(trials_output_csv, index=False)

    return sel