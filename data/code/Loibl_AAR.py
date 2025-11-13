#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Sep 29 22:01:21 2025

@author: wyan0065
'''

import pandas as pd
import numpy as np
import scipy.stats as st

# period: 1995–2014 is selected to be consistent with the AAR calculation method used in PyGEM.
# Calcluate steady-state AAR by linear regression
def cal_AAR(AAR=None, MB=None):
    
    x=[]
    y=[]
    
    for i in range(1985,2021):
        year = str(i)
        if np.isnan(AAR[year]) == False and np.isnan(MB[year]) == False:
            x = np.append(x, MB[year])
            y = np.append(y, AAR[year])
                    
    if len(x) >= 5:
        slope, intercept, r_value, p_value, mad_err = st.linregress(x, y);
        if intercept < 0.05 or intercept > 0.95:
            result = pd.Series(np.zeros(8) * np.nan,
                               index=['n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 'AAR_mean', 'a'])
        else:
            result = pd.Series([len(x), slope, intercept, r_value, p_value, mad_err, np.nanmean(y), np.nanmean(y)/intercept],
                               index=['n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 'AAR_mean', 'a'])
    else:
        result = pd.Series(np.zeros(8) * np.nan,
                           index=['n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 'AAR_mean', 'a'])
    
    return result

#%%
# Dussaillant, I., Hugonnet, R., Huss, M., Berthier, E., Bannwart, J., Paul, F., and Zemp, M.: 
# Annual mass change of the world's glaciers from 1976 to 2024 by temporal downscaling of satellite data with in situ observations, 
# Earth Syst. Sci. Data, 17, 1977–2006, https://doi.org/10.5194/essd-17-1977-2025, 2025.

# ASC_gla_MEAN-CAL-mass-change-series_obs_unobs.csv: 13
# ASW_gla_MEAN-CAL-mass-change-series_obs_unobs.csv: 14
# ASE_gla_MEAN-CAL-mass-change-series_obs_unobs.csv: 15

AARs = pd.read_csv('/Users/wyan0065/Desktop/PyGEM/calving/Output/AAR_1985_2021_Loibl.csv')
rgi_ids = AARs['RGIId'].values

path = '/Users/wyan0065/Desktop/PyGEM/calving/Output/wgms-amce-2025-02b/individual-glacier/'
asc = pd.read_csv(path + 'ASC_gla_MEAN-CAL-mass-change-series_obs_unobs.csv')
asw = pd.read_csv(path + 'ASW_gla_MEAN-CAL-mass-change-series_obs_unobs.csv')
ase = pd.read_csv(path + 'ASE_gla_MEAN-CAL-mass-change-series_obs_unobs.csv')

df_all = pd.concat([asc, asw, ase], ignore_index=True)

years = [str(y) for y in range(1985, 2021)]
columns_to_keep = ['RGIId'] + [c for c in df_all.columns if c in years]
df_all = df_all[columns_to_keep]

df_all = df_all[df_all['RGIId'].isin(rgi_ids)]

df_all.to_csv('/Users/wyan0065/Desktop/PyGEM/calving/Output/MB_1985_2021_Dussaillant.csv', index=False)

#%%
param = pd.DataFrame()
for i in range(0, len(AARs)):
    rgi_id = AARs.iloc[i]['RGIId']
    AAR = AARs.iloc[i]
    AAR = AAR.iloc[1:]
    
    MB = df_all[df_all['RGIId'] == rgi_id]
    MB = MB.iloc[0, 1:]
    
    result = cal_AAR(AAR=AAR, MB=MB)
    result = pd.DataFrame(result).T
    result.insert(0, 'RGIId', rgi_id)
    
    param = pd.concat([param, result])

param.set_index('RGIId', inplace=True)
param = param.dropna(how='all')   

param.to_csv('/Users/wyan0065/Desktop/PyGEM/calving/Output/AAR0_Loibl_Dussaillant.csv', index=True)

#%% ===== Compare simulation results with observations =====
import pandas as pd
import xarray as xr

filepath = '/Users/wyan0065/Desktop/PyGEM/calving/Output/';
loibl_AAR  = pd.read_csv(filepath + 'AAR0_Loibl_Dussaillant.csv')
RGIId = (loibl_AAR['RGIId']).tolist()
n = len(RGIId)

fn = 'ERA5_MCMC_ba1_2014_2023_corrected.nc';
output_ds_all = xr.open_dataset(filepath + fn)
find_id = np.where(np.isin(output_ds_all['RGIId'].values, RGIId)==True);
loibl_AAR['is_icecap']    = output_ds_all['is_icecap'].values[find_id]

loibl_AAR['CenLon']       = output_ds_all['CenLon'].values[find_id]
loibl_AAR['CenLat']       = output_ds_all['CenLat'].values[find_id]
loibl_AAR['is_tidewater'] = output_ds_all['is_tidewater'].values[find_id]
loibl_AAR['is_icecap']    = output_ds_all['is_icecap'].values[find_id]
loibl_AAR['is_debris']    = output_ds_all['is_debris'].values[find_id]
loibl_AAR['Area']         = output_ds_all['Area'].values[find_id]
loibl_AAR['area_2020_simu']    = (output_ds_all['area_2020'].values[find_id, 0]).reshape(n)
loibl_AAR['volume_2020']  = (output_ds_all['volume_2020'].values[find_id, 0]).reshape(n)

loibl_AAR['intercept_AAR_median']      = (output_ds_all['intercept_AAR'].values[find_id, 0]).reshape(n)
loibl_AAR['intercept_AAR_mad']       = (output_ds_all['intercept_AAR'].values[find_id, 1]).reshape(n)
loibl_AAR['intercept_AAR_mean_median'] = (output_ds_all['intercept_AAR_mean'].values[find_id, 0]).reshape(n)
loibl_AAR['intercept_AAR_mean_mad']  = (output_ds_all['intercept_AAR_mean'].values[find_id, 1]).reshape(n)
loibl_AAR['intercept_a_median']        = (output_ds_all['intercept_a'].values[find_id, 0]).reshape(n)
loibl_AAR['intercept_a_mad']         = (output_ds_all['intercept_a'].values[find_id, 1]).reshape(n)
loibl_AAR['intercept_dA_median']        = (output_ds_all['intercept_dA'].values[find_id, 0]).reshape(n)
loibl_AAR['intercept_dV_median']         = (output_ds_all['intercept_dV'].values[find_id, 0]).reshape(n)

loibl_AAR['equil_AAR_median']      = (output_ds_all['equil_AAR'].values[find_id, 0]).reshape(n)
loibl_AAR['equil_AAR_mad']       = (output_ds_all['equil_AAR'].values[find_id, 1]).reshape(n)
loibl_AAR['equil_AAR_mean_median'] = (output_ds_all['equil_AAR_mean'].values[find_id, 0]).reshape(n)
loibl_AAR['equil_AAR_mean_mad']  = (output_ds_all['equil_AAR_mean'].values[find_id, 1]).reshape(n)
loibl_AAR['equil_a_median']        = (output_ds_all['equil_a'].values[find_id, 0]).reshape(n)
loibl_AAR['equil_a_mad']         = (output_ds_all['equil_a'].values[find_id, 1]).reshape(n)
loibl_AAR['equil_dA_median']        = (output_ds_all['equil_dA'].values[find_id, 0]).reshape(n)
loibl_AAR['equil_dV_median']         = (output_ds_all['equil_dV'].values[find_id, 0]).reshape(n)

loibl_AAR['parameterization_AAR_median']      = (output_ds_all['parameterization_AAR'].values[find_id, 0]).reshape(n)
loibl_AAR['parameterization_AAR_mad']       = (output_ds_all['parameterization_AAR'].values[find_id, 1]).reshape(n)
loibl_AAR['parameterization_AAR_mean_median'] = (output_ds_all['parameterization_AAR_mean'].values[find_id, 0]).reshape(n)
loibl_AAR['parameterization_AAR_mean_mad']  = (output_ds_all['parameterization_AAR_mean'].values[find_id, 1]).reshape(n)
loibl_AAR['parameterization_a_median']        = (output_ds_all['parameterization_a'].values[find_id, 0]).reshape(n)
loibl_AAR['parameterization_a_mad']         = (output_ds_all['parameterization_a'].values[find_id, 1]).reshape(n)
loibl_AAR['parameterization_dA_median']        = (output_ds_all['parameterization_dA'].values[find_id, 0]).reshape(n)
loibl_AAR['parameterization_dV_median']         = (output_ds_all['parameterization_dV'].values[find_id, 0]).reshape(n)

output_ds_all.close()

loibl_AAR.to_csv(filepath + '/Loibl_disequilibrium_comparison.csv', index=False)





