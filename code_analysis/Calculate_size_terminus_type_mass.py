#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import xarray as xr

from moepy import lowess

lowess.tqdm = lambda x, *args, **kwargs: x


outpath = Path('/g/data/rd53/wy2165/disequilibrium/pygem_oggm/')

tag = 'median_a'
data = xr.open_dataset(outpath / f'PyGEM_global_glacier_stats_{tag}.nc')


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


def save_one_csv_per_class(df, class_coord, filename_map, outpath, tag):
    for class_name, file_label in filename_map.items():
        class_df = df.loc[df[class_coord] == class_name].copy()
        if class_df.empty:
            print('Skipped empty class:', class_name)
            continue

        output_csv = outpath / f'PyGEM_glacier_mass_{file_label}_{tag}.csv'
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
area_volume_summary = summarize_by_class(ds_area, 'area_class')
save_one_csv_per_class(
    area_volume_summary,
    'area_class',
    area_filename_map,
    outpath,
    tag,
)

ds_tw = add_tidewater_class(data)
tidewater_volume_summary = summarize_by_class(ds_tw, 'tidewater_class')
save_one_csv_per_class(
    tidewater_volume_summary,
    'tidewater_class',
    tidewater_filename_map,
    outpath,
    tag,
)