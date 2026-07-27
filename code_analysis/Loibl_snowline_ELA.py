#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Sep 29 19:05:36 2025

@author: wyan0065
'''

import numpy as np
import pandas as pd
import xarray as xr


# Loibl, D., Richter, N. & Grünberg, I. 
# Remote sensing-derived time series of transient glacier snowline altitudes for High Mountain Asia, 1985–2021. 
# Sci Data 12, 103 (2025). https://doi.org/10.1038/s41597-024-04309-6
data = xr.open_dataset('/Users/wyan0065/Desktop/OGGM/disequilibrium/data/TSLA-HMA-2021-12-filtered.nc')

months = data['LS_DATE'].dt.month

summer_mask = (months >= 7) & (months <= 10) # July, August, September, October “we defined the ‘ablation phase’ as the time span July to October (JASO)”
summer_data = data.isel(LS_DATE=summer_mask)

df = pd.DataFrame({
    'RGI_ID': summer_data['RGI_ID'].values,
    'Year': summer_data['LS_DATE'].dt.year.values,
    'ELA': summer_data['TSLArange_max_masl'].values
})

#df = pd.DataFrame({
#    'RGI_ID': data['RGI_ID'].values,
#    'Year': data['LS_DATE'].dt.year.values,
#    'ELA': data['TSLArange_max_masl'].values
#})

df = df[(df['Year'] >= 1985) & (df['Year'] <= 2021)]

# ELA: the maximum TSLA during the ablation season
ela = df.groupby(['RGI_ID', 'Year'])['ELA'].max().reset_index()


## save file
years = np.arange(1985, 2021)

rgi_ids = ela['RGI_ID'].unique()
ela_df = pd.DataFrame(index=rgi_ids, columns=years, dtype=float)

for _, row in ela.iterrows():
    ela_df.loc[row['RGI_ID'], row['Year']] = row['ELA']
    
ela_xr = xr.DataArray(
    ela_df.values,
    coords={'RGIId': ela_df.index, 'Year': ela_df.columns},
    dims=['RGIId', 'Year']
)

ela_xr.name = 'ELA'
ela_xr.attrs['long_name'] = 'Equilibrium Line Altitude'
ela_xr.attrs['units'] = 'm'
ela_xr.attrs['description'] = 'Annual maximum transient snowline altitude during ablation season (JJA), 1995–2014'
ela_xr.attrs['source'] = 'Loibl et al., 2025, Sci Data 12, 103.'


ela_xr.to_netcdf('/Users/wyan0065/Desktop/OGGM/disequilibrium/data/ELA_1985_2021_Loibl.nc')