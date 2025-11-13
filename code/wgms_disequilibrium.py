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
    
    selected_years = np.arange(2014, 2024, 1)
    smb = smb.loc[smb.index.isin(selected_years)]
    AAR = AAR.loc[AAR.index.isin(selected_years)]
    
    year = AAR.index
    x=[]
    y=[]
    mb_unc=[]
    for t in year:
        if t in smb.index:
            if np.isnan(smb['annual_balance'][t]) == False and np.isnan(AAR['aar'][t]) == False \
                and AAR['aar'][t] >=0.05 and AAR['aar'][t] <= 0.95:
                    x = np.append(x, smb['annual_balance'][t])
                    y = np.append(y, AAR['aar'][t])
                    mb_unc = np.append(mb_unc, smb['annual_balance_unc'][t])
    n = len(x)
    if n >= 5:
        std_deviation = np.std(y);
        std_errors     = std_deviation/np.sqrt(n);
        slope, intercept, r_value, p_value, mad_err = st.linregress(x, y);
        intercept = intercept;
        if intercept < 0.05 and intercept > 0.95:
            intercept = np.nan;
        
        mb_unc = mb_unc[~np.isnan(mb_unc)]
        if len(mb_unc) != 0:
            mb_unc = np.sqrt(np.sum(mb_unc ** 2))/np.shape(mb_unc)[0]
        else:
            mb_unc = np.nan
        result = pd.Series([len(x), slope, intercept, r_value, p_value, mad_err, np.nanmean(y), 
                            np.nanmean(x), std_errors, mb_unc],
                           index=['n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 
                                  'wgms_AAR_mean', 'wgms_SMB_mean','standard_error','mb_unc'])
    else:
        result = pd.Series(np.zeros(10) * np.nan,
                           index=['n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 
                                  'wgms_AAR_mean', 'wgms_SMB_mean', 'standard_error', 'mb_unc'])
    
    return result

filepath = '/Users/wyan0065/Desktop/PyGEM/calving/WGMS/DOI-WGMS-FoG-2025-02b/data/';
wgms_id  = pd.read_csv(filepath + 'WGMS_ID_AAR.csv')
wgms   = pd.read_csv(filepath + 'glacier.csv');
_AAR     = pd.read_csv(filepath + 'mass_balance.csv')
_smb     = pd.read_csv(filepath + 'mass_balance.csv')
_area    = pd.read_csv(filepath + 'mass_balance.csv')

wgms     = wgms.set_index('WGMS_ID')
_AAR     = _AAR.set_index('WGMS_ID')
_smb     = _smb.set_index('WGMS_ID')
_area    = _area.set_index('WGMS_ID')

param = pd.DataFrame()
for i in range(0, len(wgms_id)):
    n = wgms_id['WGMS_ID'][i]
    AAR = _AAR.loc[n]
    if n in _smb.index:
        smb = _smb.loc[n]
        if type(AAR['year']) is not np.int64 and type(smb['year']) is not np.int64:
            result = cal_AAR(AAR=AAR, smb=smb)
            result = pd.DataFrame(result, columns=[n]).T
            
            result.insert(0, 'area_2020', _area.loc[(_area.index==n)&(_area['year']==2020),'area'])
            
            result.insert(0, 'lon', [wgms.loc[n]['longitude']])
            result.insert(0, 'lat', [wgms.loc[n]['latitude']])
            if n in wgms.index:
                result.insert(0, 'RGIId', [wgms.loc[n]['rgi60_ids']])
            else:
                result.insert(0, 'RGIId', ['NaN'])
            
        else:
            result = pd.Series(np.zeros(14) * np.nan,
                               index=['RGIId', 'lat', 'lon', 'area_2020',
                                      'n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 
                                      'wgms_AAR_mean', 'wgms_SMB_mean', 'standard_error','mb_unc'])
            result = pd.DataFrame(result, columns=[n]).T
    else:
        result = pd.Series(np.zeros(14) * np.nan,
                           index=['RGIId', 'lat', 'lon', 'area_2020',
                                  'n', 'slope', 'intercept', 'r_value', 'p_value', 'mad_err', 
                                  'wgms_AAR_mean', 'wgms_SMB_mean', 'standard_error','mb_unc'])
        result = pd.DataFrame(result, columns=[n]).T
    
    param = pd.concat([param, result])

param.to_csv('/Users/wyan0065/Desktop/PyGEM/calving/Output/WGMS_disequilibrium_all.csv')



