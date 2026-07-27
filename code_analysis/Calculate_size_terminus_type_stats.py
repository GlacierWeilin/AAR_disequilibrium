#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import median_abs_deviation


outpath = Path('/g/data/rd53/wy2165/disequilibrium/pygem_oggm/')

tag = 'median_a'
data = xr.open_dataset(outpath / f'PyGEM_global_glacier_stats_{tag}.nc')

stats_vars = ['AAR_mean', 'AAR_steady', 'disequilibrium']


def add_area_class(ds):
    area = ds['rgi_area_km2']

    area_class = xr.full_like(area, '', dtype=object)
    area_class = area_class.where(~(area < 1), '<1 km2')
    area_class = area_class.where(~((area >= 1) & (area < 10)), '1-10 km2')
    area_class = area_class.where(~((area >= 10) & (area <= 100)), '10-100 km2')
    area_class = area_class.where(~(area > 100), '>100 km2')

    return ds.assign_coords(area_class=('rgi_id', area_class.values))


def add_tidewater_class(ds):
    tw = ds['is_tidewater']

    tidewater_class = xr.full_like(tw, '', dtype=object)
    tidewater_class = tidewater_class.where(~(tw == 1), 'marine-terminating')
    tidewater_class = tidewater_class.where(~(tw != 1), 'land-terminating')

    return ds.assign_coords(tidewater_class=('rgi_id', tidewater_class.values))


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


def summarize_stats_for_one_class(ds, class_coord, class_name):
    ds_class = ds.where(ds[class_coord] == class_name, drop=True)
    area = ds_class['rgi_area_km2']

    summary = xr.Dataset()

    for var in stats_vars:
        da = ds_class[var]
        summary[f'{var}_mean'] = da.mean(dim='rgi_id', skipna=True)
        summary[f'{var}_std'] = da.std(dim='rgi_id', skipna=True)
        summary[f'{var}_median'] = da.median(dim='rgi_id', skipna=True)
        summary[f'{var}_MAD'] = xr_nanmad(da)
        summary[f'{var}_area_weighted_mean'] = area_weighted_mean(da, area)

    df = summary.to_dataframe().reset_index()

    df[class_coord] = class_name
    df['n_glaciers'] = int(area.notnull().sum().item())
    df['rgi_area_km2'] = float(area.sum(skipna=True).item())

    cols = [
        class_coord,
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


def save_one_stats_csv_per_class(ds, class_coord, filename_map, outpath, tag):
    for class_name, file_label in filename_map.items():
        class_df = summarize_stats_for_one_class(ds, class_coord, class_name)
        if class_df.empty:
            print('Skipped empty class:', class_name)
            continue

        output_csv = outpath / f'PyGEM_glacier_stats_{file_label}_{tag}.csv'
        class_df.to_csv(output_csv, index=False)
        print('Saved:', output_csv)


area_filename_map = {
    '<1 km2': 'area_-1',
    '1-10 km2': 'area_1-10',
    '10-100 km2': 'area_10-100',
    '>100 km2': 'area_100-',
}

tidewater_filename_map = {
    'land-terminating': 'land-terminating',
    'marine-terminating': 'marine-terminating',
}


ds_area = add_area_class(data)
save_one_stats_csv_per_class(
    ds_area,
    'area_class',
    area_filename_map,
    outpath,
    tag,
)

ds_tw = add_tidewater_class(data)
save_one_stats_csv_per_class(
    ds_tw,
    'tidewater_class',
    tidewater_filename_map,
    outpath,
    tag,
)
