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
    
    AAR_steady = np.nan
    # AAR_steady
    for i in range(2000,2021): # 2000-2025
        year = str(i)
        if np.isnan(AAR[year]) == False and np.isnan(MB[year]) == False:
            x = np.append(x, MB[year])
            y = np.append(y, AAR[year])
                    
    if len(x) >= 5:
        slope, intercept, r_value, p_value, mad_err = st.linregress(x, y);
        if intercept < 0.05 or intercept > 0.95:
            AAR_steady = np.nan
        else:
            AAR_steady = intercept
    
    # AAR_mean
    y = []
    AAR_mean = np.nan
    for i in range(2014,2021): # 2014-2023
        year = str(i)
        if np.isnan(AAR[year]) == False:
            y = np.append(y, AAR[year])
    
    AAR_mean = np.nanmean(y)
    disequilibrium = AAR_mean / AAR_steady
    
    result = pd.Series([AAR_mean, AAR_steady, disequilibrium],
                       index=['AAR_mean', 'AAR_steady', 'disequilibrium'])
    
    return result

#%%
# Dussaillant, I., Hugonnet, R., Huss, M., Berthier, E., Bannwart, J., Paul, F., and Zemp, M.: 
# Annual mass change of the world's glaciers from 1976 to 2024 by temporal downscaling of satellite data with in situ observations, 
# Earth Syst. Sci. Data, 17, 1977–2006, https://doi.org/10.5194/essd-17-1977-2025, 2025.

# ASC_gla_MEAN-CAL-mass-change-series_obs_unobs.csv: 13
# ASW_gla_MEAN-CAL-mass-change-series_obs_unobs.csv: 14
# ASE_gla_MEAN-CAL-mass-change-series_obs_unobs.csv: 15

AARs = pd.read_csv('/Users/wyan0065/Desktop/OGGM/disequilibrium/data/AAR_1985_2021_Loibl.csv')
rgi_ids = AARs['RGIId'].values

path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/data/wgms-amce-2025-02b/individual-glacier/'
asc = pd.read_csv(path + 'ASC_gla_MEAN-CAL-mass-change-series_obs_unobs.csv')
asw = pd.read_csv(path + 'ASW_gla_MEAN-CAL-mass-change-series_obs_unobs.csv')
ase = pd.read_csv(path + 'ASE_gla_MEAN-CAL-mass-change-series_obs_unobs.csv')

df_all = pd.concat([asc, asw, ase], ignore_index=True)

years = [str(y) for y in range(1985, 2021)]
columns_to_keep = ['RGIId'] + [c for c in df_all.columns if c in years]
df_all = df_all[columns_to_keep]

df_all = df_all[df_all['RGIId'].isin(rgi_ids)]

df_all.to_csv('/Users/wyan0065/Desktop/OGGM/disequilibrium/data/MB_1985_2021_Dussaillant.csv', index=False)

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

param.to_csv('/Users/wyan0065/Desktop/OGGM/disequilibrium/data/Loibl_Dussaillant_disequilibrium.csv', index=True)
