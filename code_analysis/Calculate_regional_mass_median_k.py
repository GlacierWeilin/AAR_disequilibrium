#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

rho_ice = 900  # kg m-3
regions = np.arange(1,20,1)

for region in regions:
    
    path = f'/scratch/k10/wy2165/PyGEM/oggm_gdirs/None/{region:02d}/'
    outpath = '/g/data/rd53/wy2165/disequilibrium/pygem_oggm/'
    datapath = '/g/data/rd53/wy2165/disequilibrium/data/'

    result = pd.read_csv(datapath + 'temp_ch_ipcc_ar6_isimip3b.csv', index_col=0)
    result['area_rgi'] = np.nan
    result['mass_2020'] = np.nan
    result['area_steady'] = np.nan
    result['volume_steady'] = np.nan
    result['mass_steady'] = np.nan
    result['mass_remaining'] = np.nan

    # ### Get 2020 glacier mass from GlacierMIP3
    mass = pd.read_csv('/g/data/rd53/wy2165/disequilibrium/GlacierMIP3/table_S3.csv', index_col=0)
    mass = mass[mass['rgi_reg'] == f'{region:02d}']
    mass_2020 = (mass['Glacier mass in 2020$^b$ (Gt)']).values[0]

    # ### Get RGI glacier area
    stats = pd.read_csv(path + f'glacier_statistics_PyGEM_{region:02d}.csv',
                           dtype={
                               'dem_needed_interpolation': 'str',
                               'dem_needed_extrapolation': 'str',
                           },
                        low_memory=False)

    rgiids = stats['rgi_id'].values.tolist()
    area_init = stats['rgi_area_km2'].values
    result['area_rgi'] = np.nansum(area_init)
    result['mass_2020'] = mass_2020

    is_tidewater = np.where(stats['is_tidewater'].values == 1)[0]
    is_landtermi = np.where(stats['is_tidewater'].values == 0)[0]
    
    is_icecap = stats['glacier_type'].map({'Glacier': 0, 'Ice cap': 1,}).to_numpy(dtype=float)
    # area-volume scaling
    # glacier r = 1.375 c = 0.0340 km3-2r
    # ice cap r = 1.25 c = 0.0538 km3-2r
    r = np.where(is_icecap == 0, 1.375, 1.25)
    c = np.where(is_icecap == 0, 0.0340, 0.0538)

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

        result.loc[i,'area_steady'] = np.nansum(area_steady)

        volume_steady = c * area_steady ** r
        volume_steady = np.nansum(volume_steady)

        result.loc[i,'volume_steady'] = volume_steady # km3
        result.loc[i,'mass_steady'] = volume_steady * rho_ice * 1e9 / 1e12 # Gt
        result.loc[i,'mass_remaining'] = result.loc[i,'mass_steady'] / result.loc[i,'mass_2020'] * 100 #%
        

    result.to_csv(outpath + f'PyGEM_regional_mass_{region:02d}_median_k.csv')