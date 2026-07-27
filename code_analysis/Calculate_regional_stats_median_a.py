#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import median_abs_deviation

import oggm


df_itmix = pd.read_hdf(oggm.utils.get_demo_file('rgi62_itmix_df.h5'))
regions = np.array([1, 3, 4, 5, 7, 9, 17, 19])

for region in regions:
    
    path = f'/scratch/k10/wy2165/PyGEM/oggm_gdirs/None/{region:02d}/'
    outpath = '/g/data/rd53/wy2165/disequilibrium/pygem_oggm/'
    datapath = '/g/data/rd53/wy2165/disequilibrium/data/'

    result = pd.read_csv(datapath + 'temp_ch_ipcc_ar6_isimip3b.csv', index_col=0)
    result['AAR_steady_mean']               = np.nan
    result['AAR_steady_std']                = np.nan
    result['AAR_steady_median']             = np.nan
    result['AAR_steady_MAD']                = np.nan
    result['AAR_steady_area_weighted_mean'] = np.nan
    
    result['AAR_mean_mean']               = np.nan
    result['AAR_mean_std']                = np.nan
    result['AAR_mean_median']             = np.nan
    result['AAR_mean_MAD']                = np.nan
    result['AAR_mean_area_weighted_mean'] = np.nan
    
    result['disequilibrium_mean']               = np.nan
    result['disequilibrium_std']                = np.nan
    result['disequilibrium_median']             = np.nan
    result['disequilibrium_MAD']                = np.nan
    result['disequilibrium_area_weighted_mean'] = np.nan
    
    # ### Missing frontal ablation observations
    missing = pd.read_csv(f'/scratch/k10/wy2165/PyGEM/frontalablation_data/analysis/{region}-frontalablation_cal_ind-missing.csv')
    missing = missing['RGIId'].values.tolist()

    stats = pd.read_csv(path + f'glacier_statistics_PyGEM_{region:02d}.csv',
                           dtype={
                               'dem_needed_interpolation': 'str',
                               'dem_needed_extrapolation': 'str',
                           },
                        low_memory=False)
    
    rgiids = stats['rgi_id'].values.tolist()
    area_init = stats['rgi_area_km2'].to_numpy(dtype=float)

    is_tidewater = np.where(stats['is_tidewater'].values == 1)[0]
    is_landtermi = np.where(stats['is_tidewater'].values == 0)[0]

    is_icecap = stats['glacier_type'].map({'Glacier': 0, 'Ice cap': 1,}).to_numpy(dtype=float)
    # area-volume scaling
    # glacier r = 1.375 c = 0.0340 km3-2r
    # ice cap r = 1.25 c = 0.0538 km3-2r
    r = np.where(is_icecap == 0, 1.375, 1.25)
    c = np.where(is_icecap == 0, 0.0340, 0.0538)
    
    # ### Get RGI_date from GlacierMIP3
    mass = pd.read_csv('/g/data/rd53/wy2165/disequilibrium/GlacierMIP3/table_S3.csv', index_col=0)
    mass = mass[mass['rgi_reg'] == f'{region:02d}']
    rgi_year = (mass['Year$^a$ (glacier-area weighted median)']).values[0]
    
    # consensus glacier volume and geodetic volume change
    vol_itmix_m3 = df_itmix.reindex(rgiids)['vol_itmix_m3'].to_numpy(dtype=float)
    
    # ### Get glacier mass in 2020 from GlacierMIP3
    mass = pd.read_csv('/g/data/rd53/wy2165/disequilibrium/GlacierMIP3/table_S3.csv', index_col=0)
    mass = mass[mass['rgi_reg'] == f'{region:02d}']
    # calculate the mass change ratio relative to rgi date
    ratio = (mass['Glacier mass in 2020$^b$ (Gt)']).values[0] / (mass['Glacier mass$^a$ (Gt)']).values[0]
    
    # consensus glacier volume and geodetic volume change
    vol_itmix_m3 = df_itmix.reindex(rgiids)['vol_itmix_m3'].to_numpy(dtype=float)
    
    vol_2020_m3 = vol_itmix_m3 * ratio

    # ### Create a xarray to record the information for each glacier
    df = result.copy().reset_index(drop=True)
    df['experiment'] = np.arange(len(df))

    n_glaciers = len(rgiids)
    n_exp = len(df)
    
    ds_glacier = xr.Dataset(
        data_vars={
            'AAR_steady': (
                ('rgi_id', 'experiment'),
                np.full((n_glaciers, n_exp), np.nan, dtype=float)
            ),
            'AAR_mean': (
                ('rgi_id', 'experiment'),
                np.full((n_glaciers, n_exp), np.nan, dtype=float)
            ),
            'disequilibrium': (
                ('rgi_id', 'experiment'),
                np.full((n_glaciers, n_exp), np.nan, dtype=float)
            ),
            'area_steady': (
                ('rgi_id', 'experiment'),
                np.full((n_glaciers, n_exp), np.nan, dtype=float)
            ),
            'volume_steady': (
                ('rgi_id', 'experiment'),
                np.full((n_glaciers, n_exp), np.nan, dtype=float)
            ),

            'vol_2020_m3': ('rgi_id', vol_2020_m3),

            # glacier-level variables
            'rgi_area_km2': (
                ('rgi_id',),
                stats['rgi_area_km2'].to_numpy(dtype=float)
            ),
            'cenlon': (
                ('rgi_id',),
                stats['cenlon'].to_numpy(dtype=float)
            ),
            'cenlat': (
                ('rgi_id',),
                stats['cenlat'].to_numpy(dtype=float)
            ),
            'is_tidewater': (
                ('rgi_id',),
                stats['is_tidewater'].to_numpy(dtype=float)
            ),
            'is_icecap': (
                ('rgi_id',),
                is_icecap
            ),
            'vol_itmix_m3': (
                ('rgi_id',),
                vol_itmix_m3
            ),
        },
        coords={
            'rgi_id': np.asarray(rgiids, dtype=str),
            'experiment': df['experiment'].to_numpy(),
            'gcm': ('experiment', df['gcm'].astype(str).to_numpy(dtype=object)),
            'period_scenario': ('experiment', df['period_scenario'].astype(str).to_numpy(dtype=object)),
            'temp_ch_ipcc': ('experiment', df['temp_ch_ipcc'].to_numpy(dtype=float)),
        }
    )

    # ### Glacier area and volume at steady state
    for i in range(81):
        gcm = result['gcm'].values[i]
        period_scenario = result['period_scenario'].values[i]
        ssp = period_scenario[10:]
        if ssp == 'hist':
            ssp = 'ssp126'

        # Steady-state AAR
        if i == 0:
            df = pd.read_csv('/scratch/k10/wy2165/PyGEM/oggm_gdirs/OGGM/' + f'{region:02d}/' + 
                             f'glacier_AAR_steady_PyGEM_{region:02d}_{gcm.upper()}.csv', index_col=0)
        else:
            df = pd.read_csv('/scratch/k10/wy2165/PyGEM/oggm_gdirs/OGGM/' + f'{region:02d}/' + 
                             f'glacier_AAR_steady_PyGEM_{region:02d}_{gcm.upper()}_{ssp}.csv', index_col=0)

        df = df.loc[rgiids]
        AAR_steady = df['AAR_steady']
        AAR_steady.loc[missing] = np.nan
        # fillnan using regional median
        # separated by marine-terminating and land-terminating glaciers
        flag = 0
        if is_tidewater.any():
            median_tidewater = AAR_steady.iloc[is_tidewater].median()
            if pd.notna(median_tidewater):
                AAR_steady.iloc[is_tidewater] = (AAR_steady.iloc[is_tidewater].fillna(median_tidewater))
            else:
                flag = 1

        if flag == 1:
            AAR_steady = AAR_steady.fillna(AAR_steady.median())
        else:
            if is_landtermi.any():
                median_landtermi = AAR_steady.iloc[is_landtermi].median()
                AAR_steady.iloc[is_landtermi] = (AAR_steady.iloc[is_landtermi].fillna(median_landtermi))

        # AAR_mean
        column_name = gcm + '_' + period_scenario + '_AAR_mean'
        if i == 0:
            df = pd.read_csv(path + f'glacier_disequilibrium_PyGEM_{region:02d}_{gcm.upper()}.csv', index_col=0)
        else:    
            df = pd.read_csv(path + f'glacier_disequilibrium_PyGEM_{region:02d}_{gcm.upper()}_{ssp}.csv', index_col=0)
        df = df.loc[rgiids]

        AAR_mean = df[column_name]
        AAR_mean.loc[missing] = np.nan
        # fillnan using regional median
        # separated by marine-terminating and land-terminating glaciers
        flag = 0
        if is_tidewater.any():
            median_tidewater = AAR_mean.iloc[is_tidewater].median()
            if pd.notna(median_tidewater):
                AAR_mean.iloc[is_tidewater] = (AAR_mean.iloc[is_tidewater].fillna(median_tidewater))
            else:
                flag = 1

        if flag == 1:
            AAR_mean = AAR_mean.fillna(AAR_mean.median())
        else:
            if is_landtermi.any():
                median_landtermi = AAR_mean.iloc[is_landtermi].median()
                AAR_mean.iloc[is_landtermi] = (AAR_mean.iloc[is_landtermi].fillna(median_landtermi))

        disequilibrium = AAR_mean / AAR_steady
        disequilibrium.loc[AAR_mean == 0] = 0
        disequilibrium.loc[missing] = np.nan
        # fillnan using regional median
        # separated by marine-terminating and land-terminating glaciers
        flag = 0
        if is_tidewater.any():
            median_tidewater = disequilibrium.iloc[is_tidewater].median()
            if pd.notna(median_tidewater):
                disequilibrium.iloc[is_tidewater] = (disequilibrium.iloc[is_tidewater].fillna(median_tidewater))
            else:
                flag = 1

        if flag == 1:
            disequilibrium = disequilibrium.fillna(disequilibrium.median())
        else:
            if is_landtermi.any():
                median_landtermi = disequilibrium.iloc[is_landtermi].median()
                disequilibrium.iloc[is_landtermi] = (disequilibrium.iloc[is_landtermi].fillna(median_landtermi))

        disequilibrium = pd.to_numeric(disequilibrium, errors='coerce').to_numpy()
        area_steady = area_init * disequilibrium
        volume_steady = c * area_steady ** r * 1e9 # m3

        ########################################### AAR_steady
        # mean
        result.loc[i,'AAR_steady_mean'] = np.nanmean(AAR_steady)
        # std
        result.loc[i,'AAR_steady_std'] = np.nanstd(AAR_steady)
        # median
        result.loc[i,'AAR_steady_median'] = np.nanmedian(AAR_steady)
        # MAD: median absolute deviation from median
        result.loc[i,'AAR_steady_MAD'] = median_abs_deviation(AAR_steady, nan_policy='omit')
        # area-weighted mean
        mask = np.isfinite(AAR_steady) & np.isfinite(area_init) & (area_init > 0)
        aar_valid = AAR_steady[mask]
        area_valid = area_init[mask]
        result.loc[i,'AAR_steady_area_weighted_mean'] = np.sum(aar_valid * area_valid) / np.sum(area_valid)
        
        ########################################### AAR_mean
        # mean
        result.loc[i,'AAR_mean_mean'] = np.nanmean(AAR_mean)
        # std
        result.loc[i,'AAR_mean_std'] = np.nanstd(AAR_mean)
        # median
        result.loc[i,'AAR_mean_median'] = np.nanmedian(AAR_mean)
        # MAD: median absolute deviation from median
        result.loc[i,'AAR_mean_MAD'] = median_abs_deviation(AAR_mean, nan_policy='omit')
        # area-weighted mean
        mask = np.isfinite(AAR_mean) & np.isfinite(area_init) & (area_init > 0)
        aar_valid = AAR_mean[mask]
        area_valid = area_init[mask]
        result.loc[i,'AAR_mean_area_weighted_mean'] = np.sum(aar_valid * area_valid) / np.sum(area_valid)
        
        ########################################### disequilibrium
        # mean
        result.loc[i,'disequilibrium_mean'] = np.nanmean(disequilibrium)
        # std
        result.loc[i,'disequilibrium_std'] = np.nanstd(disequilibrium)
        # median
        result.loc[i,'disequilibrium_median'] = np.nanmedian(disequilibrium)
        # MAD: median absolute deviation from median
        result.loc[i,'disequilibrium_MAD'] = median_abs_deviation(disequilibrium, nan_policy='omit')
        # area-weighted mean
        mask = np.isfinite(disequilibrium) & np.isfinite(area_init) & (area_init > 0)
        disequilibrium_valid = disequilibrium[mask]
        area_valid = area_init[mask]
        result.loc[i,'disequilibrium_area_weighted_mean'] = np.sum(disequilibrium_valid * area_valid) / np.sum(area_valid)

        # record the results for each glaciers
        ds_glacier['AAR_steady'].loc[dict(experiment=i)] = AAR_steady.to_numpy()
        ds_glacier['AAR_mean'].loc[dict(experiment=i)] = AAR_mean.to_numpy()
        ds_glacier['disequilibrium'].loc[dict(experiment=i)] = disequilibrium
        ds_glacier['area_steady'].loc[dict(experiment=i)] = area_steady
        ds_glacier['volume_steady'].loc[dict(experiment=i)] = volume_steady

    result.to_csv(outpath + f'PyGEM_regional_stats_{region:02d}_median_a.csv')

    ds_glacier.to_netcdf(
        outpath + f'PyGEM_glacier_stats_{region:02d}_median_a.nc'
    )





