#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import median_abs_deviation


outpath = Path('/g/data/rd53/wy2165/disequilibrium/pygem_oggm/')

tag = 'median_k'
data = xr.open_dataset(outpath / f'PyGEM_global_glacier_stats_{tag}.nc')

stats_vars = ['AAR_mean', 'AAR_steady', 'disequilibrium']


def add_mt_class(ds, regions, class_name):
    mask = ds['region'].isin(regions) & (ds['is_tidewater'] == 1)
    ds_sub = ds.where(mask, drop=True)

    region_class = np.repeat(class_name, ds_sub.sizes['rgi_id'])

    return ds_sub.assign_coords(
        region_class=('rgi_id', region_class)
    )


def xr_nanmad(da):
    return xr.apply_ufunc(
        median_abs_deviation,
        da,
        input_core_dims=[['rgi_id']],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        kwargs={'nan_policy': 'omit'},
        output_dtypes=[float],
    )


def area_weighted_mean(da, area):
    valid_area = area.where(np.isfinite(da))
    return (da * valid_area).sum(dim='rgi_id', skipna=True) / valid_area.sum(
        dim='rgi_id',
        skipna=True,
    )


def summarize_stats_for_class(ds, class_name):
    area = ds['rgi_area_km2']

    summary = xr.Dataset()

    for var in stats_vars:
        da = ds[var]
        summary[f'{var}_mean'] = da.mean(dim='rgi_id', skipna=True)
        summary[f'{var}_std'] = da.std(dim='rgi_id', skipna=True)
        summary[f'{var}_median'] = da.median(dim='rgi_id', skipna=True)
        summary[f'{var}_MAD'] = xr_nanmad(da)
        summary[f'{var}_area_weighted_mean'] = area_weighted_mean(da, area)

    df = summary.to_dataframe().reset_index()

    df['region_class'] = class_name
    df['n_glaciers'] = int(area.notnull().sum().item())
    df['rgi_area_km2'] = float(area.sum(skipna=True).item())

    cols = [
        'region_class',
        'n_glaciers',
        'rgi_area_km2',
        'experiment',
        'gcm',
        'period_scenario',
        'temp_ch_ipcc',
    ]

    stat_cols = []
    for var in stats_vars:
        stat_cols.extend([
            f'{var}_mean',
            f'{var}_std',
            f'{var}_median',
            f'{var}_MAD',
            f'{var}_area_weighted_mean',
        ])

    return df[cols + stat_cols]


def save_mt_stats_csv(ds, class_name, outpath, tag):
    if ds.sizes.get('rgi_id', 0) == 0:
        print('Skipped empty class:', class_name)
        return

    df = summarize_stats_for_class(ds, class_name)
    if df.empty:
        print('Skipped empty class:', class_name)
        return

    output_csv = outpath / f'PyGEM_glacier_stats_{class_name}_{tag}.csv'
    df.to_csv(output_csv, index=False)
    print('Saved:', output_csv)


mt_classes = [
    ('MT_all', np.array([1, 3, 4, 5, 7, 9, 17, 19])),
    ('MT_outAntarc', np.array([1, 3, 4, 5, 7, 9, 17])),
    ('MT_inAntarc', np.array([19])),
]


for class_name, regions in mt_classes:
    ds_mt = add_mt_class(data, regions, class_name)
    save_mt_stats_csv(ds_mt, class_name, outpath, tag)
