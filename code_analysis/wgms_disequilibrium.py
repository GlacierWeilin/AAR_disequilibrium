#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:17:57 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""

# AARs: 2014-2023
import pandas as pd
import numpy as np
import scipy.stats as st

# Calcluate steady-state AAR by linear regression
def cal_AAR(AAR=None, smb=None):
    
    smb = smb.set_index('year')
    AAR = AAR.set_index('year')
    
    # calculate AAR_steady
    selected_years = np.arange(2000, 2026, 1)
    smb = smb.loc[smb.index.isin(selected_years)]
    AAR = AAR.loc[AAR.index.isin(selected_years)]
    
    year = AAR.index
    x=[]
    y=[]
    mb_unc=[]
    for t in year:
        if t in smb.index:
            if np.isnan(smb['annual_balance'][t]) == False and np.isnan(AAR['aar'][t]) == False \
                and AAR['aar'][t] >=0 and AAR['aar'][t] <= 1:
                    x = np.append(x, smb['annual_balance'][t])
                    y = np.append(y, AAR['aar'][t])
                    mb_unc = np.append(mb_unc, smb['annual_balance_unc'][t])
    n = len(x)
    if n >= 5:
        slope, intercept, r_value, p_value, mad_err = st.linregress(x, y);
        intercept = intercept;
        if intercept < 0.05 and intercept > 0.95:
            intercept = np.nan;
        
        mb_unc = mb_unc[~np.isnan(mb_unc)]
        if len(mb_unc) != 0:
            mb_unc = np.sqrt(np.sum(mb_unc ** 2))/np.shape(mb_unc)[0]
        else:
            mb_unc = np.nan
            
        AAR_steady = intercept
    else:
        AAR_steady = np.nan
    
    # calculate AAR_mean
    AAR_mean = np.nan
    selected_years = np.arange(2014, 2024, 1)
    AAR = AAR.loc[AAR.index.isin(selected_years)]
    
    year = AAR.index
    y=[]
    for t in year:
        if np.isnan(AAR['aar'][t]) == False and AAR['aar'][t] >=0 and AAR['aar'][t] <= 1:
                    y = np.append(y, AAR['aar'][t])
    
    AAR_mean = np.nanmean(y)
    
    disequilibrium = AAR_mean / AAR_steady
    
    result = pd.Series([AAR_mean, AAR_steady, disequilibrium],
                       index=['AAR_mean', 'AAR_steady', 'disequilibrium'])
    
    return result

filepath = '/Users/wyan0065/Desktop/OGGM/disequilibrium/data/DOI-WGMS-FoG-2026-02-10/data/';
wgms_id  = pd.read_csv(filepath + 'WGMS_ID_AAR.csv')
wgms   = pd.read_csv(filepath + 'glacier.csv');
_AAR     = pd.read_csv(filepath + 'mass_balance.csv')
_smb     = pd.read_csv(filepath + 'mass_balance.csv')
_area    = pd.read_csv(filepath + 'mass_balance.csv')

wgms     = wgms.set_index('id')
_AAR     = _AAR.set_index('glacier_id')
_smb     = _smb.set_index('glacier_id')
_area    = _area.set_index('glacier_id')

param = pd.DataFrame()
for i in range(0, len(wgms_id)):
    n = wgms_id['glacier_id'][i]
    AAR = _AAR.loc[n]
    if n in _smb.index:
        smb = _smb.loc[n]
        if type(AAR['year']) is not np.int64 and type(smb['year']) is not np.int64:
            result = cal_AAR(AAR=AAR, smb=smb)
            result = pd.DataFrame(result, columns=[n]).T
            
            result.insert(0, 'lon', [wgms.loc[n]['longitude']])
            result.insert(0, 'lat', [wgms.loc[n]['latitude']])
            if n in wgms.index:
                result.insert(0, 'RGIId', [wgms.loc[n]['rgi60_ids']])
            else:
                result.insert(0, 'RGIId', ['NaN'])
            
        else:
            result = pd.Series(np.zeros(6) * np.nan,
                               index=['RGIId', 'lat', 'lon', 'AAR_mean', 'AAR_steady', 'disequilibrium'])
            result = pd.DataFrame(result, columns=[n]).T
    else:
        result = pd.Series(np.zeros(6) * np.nan,
                           index=['RGIId', 'lat', 'lon', 'AAR_mean', 'AAR_steady', 'disequilibrium'])
        result = pd.DataFrame(result, columns=[n]).T
    
    param = pd.concat([param, result])

param.to_csv('/Users/wyan0065/Desktop/OGGM/disequilibrium/data/WGMS_disequilibrium_all.csv')



