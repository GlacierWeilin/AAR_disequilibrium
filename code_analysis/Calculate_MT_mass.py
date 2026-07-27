#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import numpy as np
import xarray as xr


outpath = Path('/g/data/rd53/wy2165/disequilibrium/pygem_oggm/')

tag = 'median_k'
data = xr.open_dataset(outpath / f'PyGEM_global_glacier_stats_{tag}.nc')


def add_mt_class(ds, regions, class_name):
    mask = ds['region'].isin(regions) & (ds['is_tidewater'] == 1)
    ds_sub = ds.where(mask, drop=True)

    region_class = np.repeat(class_name, ds_sub.sizes['rgi_id'])

    return ds_sub.assign_coords(
        region_class=('rgi_id', region_class)
    )


def summarize_by_class(ds, class_coord):
    static_by_class = (
        xr.Dataset({
            'n_glaciers': ds['rgi_area_km2'].notnull(),
            'rgi_area_km2': ds['rgi_area_km2'],
            'vol_itmix_m3': ds['vol_itmix_m3'],
            'vol_2020_m3': ds['vol_2020_m3'],
        })
        .groupby(ds[class_coord])
        .sum(dim='rgi_id', skipna=True)
        .to_dataframe()
        .reset_index()
    )

    static_by_class['n_glaciers'] = static_by_class['n_glaciers'].astype(int)

    steady_by_class = (
        xr.Dataset({
            'area_steady': ds['area_steady'],
            'volume_steady': ds['volume_steady'],
        })
        .groupby(ds[class_coord])
        .sum(dim='rgi_id', skipna=True)
        .to_dataframe()
        .reset_index()
    )

    df = steady_by_class.merge(
        static_by_class,
        on=class_coord,
        how='left',
    )

    df['mass_remaining'] = (df['volume_steady'] / df['vol_2020_m3']) * 100

    cols = [
        class_coord,
        'n_glaciers',
        'rgi_area_km2',
        'vol_2020_m3',
        'experiment',
        'gcm',
        'period_scenario',
        'temp_ch_ipcc',
        'area_steady',
        'volume_steady',
        'mass_remaining',
    ]

    return df[cols]


def save_mt_class_csv(ds, class_name, outpath, tag):
    df = summarize_by_class(ds, 'region_class')
    if df.empty:
        print('Skipped empty class:', class_name)
        return

    output_csv = outpath / f'PyGEM_glacier_mass_{class_name}_{tag}.csv'
    df.to_csv(output_csv, index=False)
    print('Saved:', output_csv)


mt_classes = [
    ('MT_all', np.array([1, 3, 4, 5, 7, 9, 17, 19])),
    ('MT_outAntarc', np.array([1, 3, 4, 5, 7, 9, 17])),
    ('MT_inAntarc', np.array([19])),
]


for class_name, regions in mt_classes:
    ds_mt = add_mt_class(data, regions, class_name)
    save_mt_class_csv(ds_mt, class_name, outpath, tag)
