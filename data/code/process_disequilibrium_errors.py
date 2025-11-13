#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:47:13 2024

@author: Weilin Yang (weilinyang.yang@monash.edu)
"""
import pandas as pd
import numpy as np
import xarray as xr
import glob
import os
import collections
import rasterio

from scipy.stats import median_abs_deviation
from scipy.interpolate import griddata

def calc_stats_array(data):
    """
    Calculate stats for a given variable

    Parameters
    ----------
    vn : str
        variable name
    ds : xarray dataset
        dataset of output with all ensemble simulations

    Returns
    -------
    stats : np.array
        Statistics related to a given variable
    """
    
    stats = None
    
    if data.ndim==1:
        if stats is None:
            stats = np.nanmedian(data)
        else:
            stats = np.append(stats, np.nanmedian(data))
        stats = np.append(stats, median_abs_deviation(data, nan_policy='omit'))
    else:
        if stats is None:
            stats = np.nanmedian(data, axis=1)[:,np.newaxis]
        else:
            stats = np.append(stats, np.nanmedian(data, axis=1)[:,np.newaxis], axis=1)
        stats = np.append(stats, median_abs_deviation(data, axis=1, nan_policy='omit')[:,np.newaxis], axis=1)
    return stats

csv_files = glob.glob('/Users/wyan0065/Desktop/PyGEM/calving/RGI/rgi60/ori/*.csv')
csv_files = np.sort(csv_files)
dfs = [pd.read_csv(file) for file in csv_files]
RGI_all = pd.concat(dfs, ignore_index=True)
m = len(RGI_all);

#%% Output
glac_values = np.arange(m);

dim = np.arange(2)
climate_dim = np.arange(7)
params_dim  = np.arange(11)

# Variable coordinates dictionary
output_coords_dict = collections.OrderedDict()
output_coords_dict['RGIId']        =  collections.OrderedDict([('glac', glac_values)])
output_coords_dict['CenLon']       = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['CenLat']       = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['O1Region']     = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['O2Region']     = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['is_tidewater'] = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['is_icecap']    = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['is_debris']    = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['Area']         = collections.OrderedDict([('glac', glac_values)])

# simulated glacier results in 2020
output_coords_dict['area_2020']       = collections.OrderedDict([('glac', glac_values),
                                                                 ('dim', dim)])
output_coords_dict['volume_2020']     = collections.OrderedDict([('glac', glac_values),
                                                                 ('dim', dim)])
output_coords_dict['volume_bsl_2020'] = collections.OrderedDict([('glac', glac_values),
                                                                 ('dim', dim)])

# results of parameterization approach and equilibrium simulaiton
# intercept_results
# 0-intercept_ELA_mean, 1-intercept_AAR_mean, 2-intercept_AAR, 3-intercept_a, 4-intercept_dA, 5-intercept_dV
# 6-intercept_ELA_steady, 7-intercept_THAR, 8-intercept_At, 9-intercept_dV_bwl, 10-intercept_dV_eff
output_coords_dict['intercept_ELA_mean']     = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['intercept_AAR_mean']     = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['intercept_AAR']          = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['intercept_a']            = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['intercept_dA']           = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['intercept_dV']           = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['intercept_ELA_steady']   = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['intercept_THAR']         = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['intercept_At']           = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['intercept_dV_bwl']       = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['intercept_dV_eff']       = collections.OrderedDict([('glac', glac_values),('dim', dim)])

# equil_results
# 0-equil_ELA_mean, 1-equil_AAR_mean, 2-equil_AAR, 3-equil_a, 4-equil_dA, 5-equil_dV, 6-equil_ELA_equil
# 7-equil_THAR, 8-equil_At, 9-equil_dV_bwl, 10-equil_dV_eff
output_coords_dict['equil_ELA_mean']   = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['equil_AAR_mean']   = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['equil_AAR']        = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['equil_a']          = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['equil_dA']         = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['equil_dV']         = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['equil_ELA_steady'] = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['equil_THAR']       = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['equil_Ah']         = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['equil_At']         = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['equil_dV_bwl']     = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['equil_dV_eff']     = collections.OrderedDict([('glac', glac_values),('dim', dim)])

output_coords_dict['equil_time']    = collections.OrderedDict([('glac', glac_values)])

# parameterization_results: For tidewater glaciers lacking frontal ablation observations, set the corresponding results to NaN.
# 0-parameterization_ELA_mean, 1-parameterization_AAR_mean, 2-parameterization_AAR, 3-parameterization_a, 4-parameterization_dA, 5-parameterization_dV, 6-parameterization_ELA_steady
# 7-parameterization_THAR, 8-parameterization_At, 9-parameterization_dV_bwl, 10-parameterization_dV_eff
output_coords_dict['parameterization_ELA_mean']   = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['parameterization_AAR_mean']   = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['parameterization_AAR']        = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['parameterization_a']          = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['parameterization_dA']         = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['parameterization_dV']         = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['parameterization_ELA_steady'] = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['parameterization_THAR']       = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['parameterization_Ah']         = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['parameterization_At']         = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['parameterization_dV_bwl']     = collections.OrderedDict([('glac', glac_values),('dim', dim)])
output_coords_dict['parameterization_dV_eff']     = collections.OrderedDict([('glac', glac_values),('dim', dim)])

# output_climate, equil_climate
# 0-temp, 1-prec, 2-acc, 3-refreeze, 4-melt, 5-frontalablation, 6-massbaltotal
output_coords_dict['output_climate'] =  collections.OrderedDict([('glac', glac_values), 
                                                                 ('climate_dim', climate_dim),
                                                                 ('dim', dim)])
output_coords_dict['equil_climate'] =  collections.OrderedDict([('glac', glac_values), 
                                                                ('climate_dim', climate_dim),
                                                                ('dim', dim)])

# Parameters
# 0-debris_hd, 1-debris_ed, 2-debris_area
# 3-kp, 4-tbias, 5-ddfsnow, 6-ddfice, 7-tsnow_threshold, 8-precgrad, 9-calving_k_values, 10-water_level
output_coords_dict['params'] =  collections.OrderedDict([('glac', glac_values), 
                                                         ('params_dim', params_dim),
                                                         ('dim', dim)])

# Attributes dictionary
output_attrs_dict = {
        'glac': {
                'long_name': 'glacier index',
                 'comment': 'glacier index referring to glaciers properties and model results'},
        'sims': {
                'long_name': 'simulation number',
                'comment': 'simulation number referring to the MCMC simulation; otherwise, only 1'},
        'dim': {
                'long_name': 'stats for a given variable',
                'comment': '0-median, 1-mad'},
        'climate_dim': {
                'long_name': 'baseline climate',
                'comment': '0-temp, 1-prec, 2-acc, 3-refreeze, 4-melt, 5-frontalablation, 6-massbaltotal'},
        'params_dim': {
                'long_name': 'model parameters',
                'comment': '0-debris_hd, 1-debris_ed, 2-debris_area, 3-kp, 4-tbias, 5-ddfsnow, 6-ddfice, 7-tsnow_threshold, 8-precgrad, \
                    9-calving_k_values, 10-water_level'},
        'RGIId': {
                'long_name': 'Randolph Glacier Inventory ID',
                'comment': 'RGIv6.2'},
        'CenLon': {
                'long_name': 'center longitude',
                'units': 'degrees E',
                'comment': 'value from RGIv6.2'},
        'CenLat': {
                'long_name': 'center latitude',
                'units': 'degrees N',
                'comment': 'value from RGIv6.2'},
        'O1Region': {
                'long_name': 'RGI order 1 region',
                'comment': 'value from RGIv6.2'},
        'O2Region': {
                'long_name': 'RGI order 2 region',
                'comment': 'value from RGIv6.2'},
        'is_tidewater': {
                'long_name': 'is marine-terminating glacier',
                'comment': 'value from RGIv6.2'},
        'is_icecap': {
                'long_name': 'is ice cap',
                'comment': 'value from RGIv6.2'},
        'is_debris': {
                'long_name': 'is debris covered glacier',
                'comment': 'value from Rounce et al. (2021)'},
        'Area': {
                'long_name': 'glacier area',
                'units': 'm2',
                'comment': 'value from RGIv6.2'},
        'area_2020': {
                'long_name': 'simulated glacier area in 2020',
                'units': 'm2'},
        'volume_2020': {
                'long_name': 'simulated glacier volume in 2020',
                'units': 'm3'},
        'volume_bsl_2020': {
                'long_name': 'simulated glacier volume below sea level in 2020',
                'units': 'm3'},
        'intercept_ELA_mean': {
                'long_name': 'mean yearly ELA'},
        'intercept_AAR_mean': {
                'long_name': 'mean yearly AAR'},
        'intercept_AAR': {
                'long_name': 'glacier steady-state Accumulation Area Ratio based on linear regression'},
        'intercept_a': {
                'long_name': 'the climate disequilibrium of each glacier (linear regression)',
                'units': '%',
                'comment': 'AAR(mean)/AAR(steady-state)'},
        'intercept_dA': {
                'long_name': 'glacier area change when reaching equilibrum (linear regression)',
                'units': 'm2',
                'comment': 'dA = Area*(a-1)'},
        'intercept_dV': {
                'long_name': 'glacier volume change when reaching equilibrum (linear regression)',
                'units': 'm3',
                'comment': 'not in water equivalent; dV = Volume*(a^r-1); r=1.375 for glaciers; r=1.25 for ice caps;'},
        'intercept_ELA_steady': {
                'long_name': 'steady ELA of current glacier',
                'units': 'm'},
        'intercept_THAR': {
                'long_name': 'glacier steady-state toe-to-headwall altitude ratio',
                'comment': 'THAR=(ELA-At)/(Ah-At)'},
        'intercept_At': {
                'long_name': 'glacier terminus elevaiton when reaching equilibrum',
                'units': 'm'},
        'intercept_dV_bwl': {
                'long_name': 'glacier volume change below sea level when reaching equilibrum',
                'units': 'm3',
                'comment': 'not in water equivalent; based on steady-state THAR, mean ELA and glacier terimus elevation when reaching equilibrium'},
        'intercept_dV_eff': {
                'long_name': 'effective glacier volume change',
                'units': 'm3',
                'comment': 'not in water equivalent; dV_eff = dV_steady - dV_bwl'},
        'equil_ELA_mean': {
                'long_name': 'mean yearly ELA'},
        'equil_AAR_mean': {
                'long_name': 'mean yearly AAR'},
        'equil_AAR': {
                'long_name': 'glacier steady-state Accumulation Area Ratio (equilibrium experiment)'},
        'equil_a': {
                'long_name': 'the climate disequilibrium of each glacier (equilibrium experiment)',
                'units': '%',
                'comment': 'AAR(mean)/AAR(steady-state)'},
        'equil_dA': {
                'long_name': 'glacier area change when reaching equilibrum (equilibrium experiment)',
                'units': 'm2',
                'comment': 'dA = Area*(a-1)'},
        'equil_dV': {
                'long_name': 'glacier volume change when reaching equilibrum (equilibrium experiment)',
                'units': 'm3',
                'comment': 'not in water equivalent; dV = Volume*(a^r-1); r=1.375 for glaciers; r=1.25 for ice caps;'},
        'equil_ELA_steady': {
                'long_name': 'steady ELA of current glacier',
                'units': 'm'},
        'equil_THAR': {
                'long_name': 'glacier steady-state toe-to-headwall altitude ratio',
                'comment': 'THAR=(ELA-At)/(Ah-At)'},
        'equil_Ah': {
                'long_name': 'glacier headwall elevaiton when reaching equilibrum',
                'units': 'm'},
        'equil_At': {
                'long_name': 'glacier terminus elevaiton when reaching equilibrum',
                'units': 'm'},
        'equil_dV_bwl': {
                'long_name': 'glacier volume change below sea level when reaching equilibrum',
                'units': 'm3',
                'comment': 'not in water equivalent; based on steady-state THAR, mean ELA and glacier terimus elevation when reaching equilibrium'},
        'equil_dV_eff': {
                'long_name': 'effective glacier volume change',
                'units': 'm3',
                'comment': 'not in water equivalent; dV_eff = dV_steady - dV_bwl'},
        'equil_time': {
                'long_name': 'year when glacier reaches equilibrium state',
                'units': 'year'},
        'parameterization_ELA_mean': {
                'long_name': 'mean yearly ELA'},
        'parameterization_AAR_mean': {
                'long_name': 'mean yearly AAR'},
        'parameterization_AAR': {
                'long_name': 'glacier steady-state Accumulation Area Ratio'},
        'parameterization_a': {
                'long_name': 'the climate disequilibrium of each glacier',
                'units': '%',
                'comment': 'AAR(mean)/AAR(steady-state)'},
        'parameterization_dA': {
                'long_name': 'glacier area change when reaching equilibrium',
                'units': 'm2',
                'comment': 'dA = Area*(a-1)'},
        'parameterization_dV': {
                'long_name': 'glacier volume change when reaching equilibrium',
                'units': 'm3',
                'comment': 'not in water equivalent; dV = Volume*(a^r-1); r=1.375 for glaciers; r=1.25 for ice caps;'},
        'parameterization_ELA_steady': {
                'long_name': 'steady ELA of current glacier',
                'units': 'm'},
        'parameterization_THAR': {
                'long_name': 'glacier steady-state toe-to-headwall altitude ratio',
                'comment': 'THAR=(ELA-At)/(Ah-At)'},
        'parameterization_Ah': {
                'long_name': 'glacier headwall elevaiton when reaching equilibrium',
                'units': 'm'},
        'parameterization_At': {
                'long_name': 'glacier terminus elevaiton when reaching equilibrium',
                'units': 'm'},
        'parameterization_dV_bwl': {
                'long_name': 'glacier volume change below sea level when reaching equilibrium',
                'units': 'm3',
                'comment': 'not in water equivalent; based on steady-state THAR, mean ELA and glacier terimus elevation when reaching equilibrium'},
        'parameterization_dV_eff': {
                'long_name': 'effective glacier volume change',
                'units': 'm3',
                'comment': 'not in water equivalent; dV_eff = dV_steady - dV_bwl'},
        'output_climate': {
                'long_name': 'mean climate during 2014-2023',
                'comment': 'climate_dim, dim'},
        'equil_climate': {
                'long_name': 'mean climate over the last 10 years of the equilibrium simulation',
                'comment': 'climate_dim, dim'},
        'params': {
                'long_name': 'model parameters',
                'comment': 'params_dim, dim'},
        }

count_vn = 0
encoding = {}
for vn in output_coords_dict.keys():
    count_vn += 1
    empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
    output_ds = xr.Dataset({vn: (list(output_coords_dict[vn].keys()), empty_holder)},
                               coords=output_coords_dict[vn])
    # Merge datasets of stats into one output
    if count_vn == 1:
        output_ds_all = output_ds
    else:
        output_ds_all = xr.merge((output_ds_all, output_ds))
noencoding_vn = ['RGIId']
# Add attributes
for vn in output_ds_all.variables:
    try:
        output_ds_all[vn].attrs = output_attrs_dict[vn]
    except:
        pass
    # Encoding (specify _FillValue, offsets, etc.)
       
    if vn not in noencoding_vn:
        encoding[vn] = {'_FillValue': None,
                        'zlib':True,
                        'complevel':9
                        }

output_ds_all.attrs = {'Source' : 'PyGEMv0.2.5 developed by David Rounce (drounce@alaska.edu)',
                       'Further developed by': 'Weilin Yang (weilinyang.yang@monash.edu)',
                       'Code reviewed by': 'Wenchao Chu (peterchuwenchao@foxmail.com)'}

#%% all
output_ds_all['RGIId'].values        = RGI_all['RGIId'].values;
output_ds_all['CenLon'].values       = RGI_all['CenLon'].values;
output_ds_all['CenLat'].values       = RGI_all['CenLat'].values;
output_ds_all['O1Region'].values     = RGI_all['O1Region'].values;
output_ds_all['O2Region'].values     = RGI_all['O2Region'].values;
output_ds_all['is_tidewater'].values = RGI_all['IsTidewater'].values;
output_ds_all['is_icecap'].values    = np.where(RGI_all['GlacierType']=='Glacier', 0, 1);
output_ds_all['is_debris'].values    = np.zeros(m)*np.nan
output_ds_all['Area'].values         = RGI_all['Area'].values;

# simulated glacier results in 2020
output_coords_dict['area_2020'].values       = np.zeros([m,np.shape(dim)[0]])*np.nan
output_coords_dict['volume_2020'].values     = np.zeros([m,np.shape(dim)[0]])*np.nan
output_coords_dict['volume_bsl_2020'].values = np.zeros([m,np.shape(dim)[0]])*np.nan

output_ds_all['intercept_ELA_mean'].values     = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['intercept_AAR_mean'].values     = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['intercept_AAR'].values          = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['intercept_a'].values            = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['intercept_dA'].values           = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['intercept_dV'].values           = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['intercept_ELA_steady'].values   = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['intercept_THAR'].values         = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['intercept_dV_bwl'].values       = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['intercept_dV_eff'].values       = np.zeros([m,np.shape(dim)[0]])*np.nan

output_ds_all['equil_ELA_mean'].values   = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['equil_AAR_mean'].values   = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['equil_AAR'].values        = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['equil_a'].values          = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['equil_dA'].values         = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['equil_dV'].values         = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['equil_ELA_steady'].values = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['equil_THAR'].values       = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['equil_dV_bwl'].values     = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['equil_dV_eff'].values     = np.zeros([m,np.shape(dim)[0]])*np.nan
output_ds_all['equil_time'].values = np.zeros(m)*np.nan

output_ds_all['output_climate'].values = np.zeros([m,np.shape(climate_dim)[0],np.shape(dim)[0]])*np.nan
output_ds_all['equil_climate'].values  = np.zeros([m,np.shape(climate_dim)[0],np.shape(dim)[0]])*np.nan

output_ds_all['params'].values = np.zeros([m,np.shape(params_dim)[0],np.shape(dim)[0]])*np.nan

#%% simulated glaciers
data = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/Output/ERA5_MCMC_ba1_50sets_2014_2023_all.nc');

find = np.where(RGI_all['RGIId'].isin(data['RGIId'].values)==1)[0]

output_ds_all['is_debris'].values[find] = data['is_debris'].values
output_ds_all['is_tidewater'].values[find] = data['is_tidewater'].values

output_ds_all['area_2020'].values[find,:]       = calc_stats_array(data['area_2020'].values)
output_ds_all['volume_2020'].values[find,:]     = calc_stats_array(data['volume_2020'].values)
output_ds_all['volume_bsl_2020'].values[find,:] = calc_stats_array(data['volume_bsl_2020'].values)

# intercept_results
output_ds_all['intercept_ELA_mean'].values[find,:]     = calc_stats_array(data['intercept_ELA_mean'].values)
output_ds_all['intercept_AAR_mean'].values[find,:]     = calc_stats_array(data['intercept_AAR_mean'].values)
output_ds_all['intercept_AAR'].values[find,:]          = calc_stats_array(data['intercept_AAR'].values)
output_ds_all['intercept_a'].values[find,:]            = calc_stats_array(data['intercept_a'].values)
output_ds_all['intercept_dA'].values[find,:]           = calc_stats_array(data['intercept_dA'].values)
output_ds_all['intercept_dV'].values[find,:]           = calc_stats_array(data['intercept_dV'].values)
output_ds_all['intercept_ELA_steady'].values[find,:]   = calc_stats_array(data['intercept_ELA_steady'].values)
output_ds_all['intercept_THAR'].values[find,:]         = calc_stats_array(data['intercept_THAR'].values)
output_ds_all['intercept_dV_bwl'].values[find,:]       = calc_stats_array(data['intercept_dV_bwl'].values)
output_ds_all['intercept_dV_eff'].values[find,:]       = calc_stats_array(data['intercept_dV_eff'].values)

# equil_results
output_ds_all['equil_ELA_mean'].values[find,:]   = calc_stats_array(data['equil_ELA_mean'].values)
output_ds_all['equil_AAR_mean'].values[find,:]   = calc_stats_array(data['equil_AAR_mean'].values)
output_ds_all['equil_AAR'].values[find,:]        = calc_stats_array(data['equil_AAR'].values)
output_ds_all['equil_a'].values[find,:]          = calc_stats_array(data['equil_a'].values)
output_ds_all['equil_dA'].values[find,:]         = calc_stats_array(data['equil_dA'].values)
output_ds_all['equil_dV'].values[find,:]         = calc_stats_array(data['equil_dV'].values)
output_ds_all['equil_ELA_steady'].values[find,:] = calc_stats_array(data['equil_ELA_steady'].values)
output_ds_all['equil_THAR'].values[find,:]       = calc_stats_array(data['equil_THAR'].values)
output_ds_all['equil_dV_bwl'].values[find,:]     = calc_stats_array(data['equil_dV_bwl'].values)
output_ds_all['equil_dV_eff'].values[find,:]     = calc_stats_array(data['equil_dV_eff'].values)
output_ds_all['equil_time'].values[find] = data['equil_time'].values

# climate_data
output_ds_all['output_climate'].values[find,:,:] = data['output_climate'].values[:,:,[4,7]]
output_ds_all['equil_climate'].values[find,:,:]  = data['equil_climate'].values[:,:,[4,7]]

# Parameters
output_ds_all['params'].values[find,:3,:] = data['params'].values[:,:3,[0,1]]
output_ds_all['params'].values[find,3:,:] = data['params'].values[:,3:,[4,7]]

#%%
data100 = xr.open_dataset('/Users/wyan0065/Desktop/PyGEM/calving/Output/ERA5_MCMC_ba1_100sets_2014_2023_all.nc');
rgiids_data100 = data100['RGIId'].values
rgiids_output = output_ds_all['RGIId'].values
output_index_map = {rgiid: idx for idx, rgiid in enumerate(rgiids_output)}
matched_indices_in_output = [output_index_map.get(rid, -1) for rid in rgiids_data100]
matched_indices_in_output = np.array(matched_indices_in_output)
valid_mask = matched_indices_in_output != -1
valid_data100_indices = np.where(valid_mask)[0]
valid_output_indices = matched_indices_in_output[valid_mask]

output_ds_all['is_debris'].values[valid_output_indices] = data100['is_debris'].values
output_ds_all['is_tidewater'].values[valid_output_indices] = data100['is_tidewater'].values

output_ds_all['area_2020'].values[valid_output_indices,:]       = calc_stats_array(data100['area_2020'].values)
output_ds_all['volume_2020'].values[valid_output_indices,:]     = calc_stats_array(data100['volume_2020'].values)
output_ds_all['volume_bsl_2020'].values[valid_output_indices,:] = calc_stats_array(data100['volume_bsl_2020'].values)

# intercept_results
output_ds_all['intercept_ELA_mean'].values[valid_output_indices,:]     = calc_stats_array(data100['intercept_ELA_mean'].values)
output_ds_all['intercept_AAR_mean'].values[valid_output_indices,:]     = calc_stats_array(data100['intercept_AAR_mean'].values)
output_ds_all['intercept_AAR'].values[valid_output_indices,:]          = calc_stats_array(data100['intercept_AAR'].values)
output_ds_all['intercept_a'].values[valid_output_indices,:]            = calc_stats_array(data100['intercept_a'].values)
output_ds_all['intercept_dA'].values[valid_output_indices,:]           = calc_stats_array(data100['intercept_dA'].values)
output_ds_all['intercept_dV'].values[valid_output_indices,:]           = calc_stats_array(data100['intercept_dV'].values)
output_ds_all['intercept_ELA_steady'].values[valid_output_indices,:]   = calc_stats_array(data100['intercept_ELA_steady'].values)
output_ds_all['intercept_THAR'].values[valid_output_indices,:]         = calc_stats_array(data100['intercept_THAR'].values)
output_ds_all['intercept_dV_bwl'].values[valid_output_indices,:]       = calc_stats_array(data100['intercept_dV_bwl'].values)
output_ds_all['intercept_dV_eff'].values[valid_output_indices,:]       = calc_stats_array(data100['intercept_dV_eff'].values)

# equil_results
output_ds_all['equil_ELA_mean'].values[valid_output_indices,:]   = calc_stats_array(data100['equil_ELA_mean'].values)
output_ds_all['equil_AAR_mean'].values[valid_output_indices,:]   = calc_stats_array(data100['equil_AAR_mean'].values)
output_ds_all['equil_AAR'].values[valid_output_indices,:]        = calc_stats_array(data100['equil_AAR'].values)
output_ds_all['equil_a'].values[valid_output_indices,:]          = calc_stats_array(data100['equil_a'].values)
output_ds_all['equil_dA'].values[valid_output_indices,:]         = calc_stats_array(data100['equil_dA'].values)
output_ds_all['equil_dV'].values[valid_output_indices,:]         = calc_stats_array(data100['equil_dV'].values)
output_ds_all['equil_ELA_steady'].values[valid_output_indices,:] = calc_stats_array(data100['equil_ELA_steady'].values)
output_ds_all['equil_THAR'].values[valid_output_indices,:]       = calc_stats_array(data100['equil_THAR'].values)
output_ds_all['equil_dV_bwl'].values[valid_output_indices,:]     = calc_stats_array(data100['equil_dV_bwl'].values)
output_ds_all['equil_dV_eff'].values[valid_output_indices,:]     = calc_stats_array(data100['equil_dV_eff'].values)

output_ds_all['equil_time'].values[valid_output_indices] = data100['equil_time'].values

# climate_data100
output_ds_all['output_climate'].values[valid_output_indices,:,:] = data100['output_climate'].values[:,:,[4,7]]
output_ds_all['equil_climate'].values[valid_output_indices,:,:]  = data100['equil_climate'].values[:,:,[4,7]]

# Parameters
output_ds_all['params'].values[valid_output_indices,:3,:] = data100['params'].values[:,:3,[0,1]]
output_ds_all['params'].values[valid_output_indices,3:,:] = data100['params'].values[:,3:,[4,7]]

#%%
output_ds_all['parameterization_ELA_mean'].values   = output_ds_all['intercept_ELA_mean'].values.copy()
output_ds_all['parameterization_AAR_mean'].values   = output_ds_all['intercept_AAR_mean'].values.copy()
output_ds_all['parameterization_AAR'].values        = output_ds_all['intercept_AAR'].values.copy()
output_ds_all['parameterization_a'].values          = output_ds_all['intercept_a'].values.copy()
output_ds_all['parameterization_dA'].values         = output_ds_all['intercept_dA'].values.copy()
output_ds_all['parameterization_dV'].values         = output_ds_all['intercept_dV'].values.copy()
output_ds_all['parameterization_ELA_steady'].values = output_ds_all['intercept_ELA_steady'].values.copy()
output_ds_all['parameterization_THAR'].values       = output_ds_all['intercept_THAR'].values.copy()
output_ds_all['parameterization_dV_bwl'].values     = output_ds_all['intercept_dV_bwl'].values.copy()
output_ds_all['parameterization_dV_eff'].values     = output_ds_all['intercept_dV_eff'].values.copy()

df = pd.read_csv('/Users/wyan0065/Desktop/PyGEM/calving/Output/tidewater_has_obs.csv')
tidewater_has_obs = RGI_all['RGIId'].isin(df['RGIId'])

loc = np.where(output_ds_all['O1Region'].values==10)[0]
tidewater_has_obs[loc] = True

loc = np.where(output_ds_all['is_tidewater'].values==0)[0]
tidewater_has_obs[loc] = True

loc = np.where(tidewater_has_obs==0)[0]
output_ds_all['parameterization_ELA_mean'].values[loc,:]   = np.nan
output_ds_all['parameterization_AAR_mean'].values[loc,:]   = np.nan
output_ds_all['parameterization_AAR'].values[loc,:]        = np.nan
output_ds_all['parameterization_a'].values[loc,:]          = np.nan
output_ds_all['parameterization_dA'].values[loc,:]         = np.nan
output_ds_all['parameterization_dV'].values[loc,:]         = np.nan
output_ds_all['parameterization_ELA_steady'].values[loc,:] = np.nan
output_ds_all['parameterization_THAR'].values[loc,:]       = np.nan
output_ds_all['parameterization_dV_bwl'].values[loc,:]     = np.nan
output_ds_all['parameterization_dV_eff'].values[loc,:]     = np.nan

#%%
# is_debris
loc = np.where(np.isnan(output_ds_all['is_debris'].values))[0]
RGIId = output_ds_all['RGIId'].values[loc]
for i,rgi_id in enumerate(RGIId):
    if int(rgi_id[6:8])<10:
        debris_ed = '/Users/wyan0065/Desktop/PyGEM/calving/debris_data/ed_tifs/'+rgi_id[6:8]+'/'+rgi_id[7:14]+'_meltfactor.tif';
        debris_hd = '/Users/wyan0065/Desktop/PyGEM/calving/debris_data/hd_tifs/'+rgi_id[6:8]+'/'+rgi_id[7:14]+'_hdts_m.tif';
    else:
        debris_ed = '/Users/wyan0065/Desktop/PyGEM/calving/debris_data/ed_tifs/'+rgi_id[6:8]+'/'+rgi_id[6:14]+'_meltfactor.tif';
        debris_hd = '/Users/wyan0065/Desktop/PyGEM/calving/debris_data/hd_tifs/'+rgi_id[6:8]+'/'+rgi_id[6:14]+'_hdts_m.tif';
    
    output_ds_all['is_debris'].values[loc[i]] = os.path.exists(debris_hd);
    if os.path.exists(debris_hd):
        with rasterio.open(debris_hd) as dataset:
                tif = dataset.read()
                transform = dataset.transform
                tif[tif>1.e+19]=np.nan
                output_ds_all['params'].values[loc[i],0,0] = np.nanmean(tif)
                output_ds_all['params'].values[loc[i],0,1] = np.nanstd(tif)
                output_ds_all['params'].values[loc[i],2,0] = np.nansum(tif) * transform[0] * transform[0]
                
        with rasterio.open(debris_ed) as dataset:
                tif = dataset.read()
                tif[tif>1.e+19]=np.nan
                output_ds_all['params'].values[loc[i],1,0] = np.nanmean(tif)
                output_ds_all['params'].values[loc[i],1,1] = np.nanstd(tif)

# check the glacier area in 2020
loc = np.where(np.isnan(output_ds_all['area_2020'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['area_2020'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['area_2020'].values[loc, :] = griddata((find_lon, find_lat),
                                                     output_ds_all['area_2020'].values[find_loc,:]/np.tile((output_ds_all['Area'].values[find_loc])[:, np.newaxis], (1,2)), 
                                                     (loc_lon, loc_lat), method='nearest') * np.tile((output_ds_all['Area'].values[loc])[:, np.newaxis], (1,2))

loc = np.where(output_ds_all['area_2020'].values[:,0]==0)[0]
output_ds_all['area_2020'].values[loc,:]=0
output_ds_all['volume_2020'].values[loc,:]=0
output_ds_all['intercept_a'].values[loc,:]=0
output_ds_all['intercept_dA'].values[loc,:]=0
output_ds_all['intercept_dV'].values[loc,:]=0
output_ds_all['intercept_dV_bwl'].values[loc,:]=0
output_ds_all['intercept_dV_eff'].values[loc,:]=0
output_ds_all['equil_a'].values[loc,:]=0
output_ds_all['equil_dA'].values[loc,:]=0
output_ds_all['equil_dV'].values[loc,:]=0
output_ds_all['equil_dV_bwl'].values[loc,:]=0
output_ds_all['equil_dV_eff'].values[loc,:]=0
output_ds_all['parameterization_a'].values[loc,:]=0
output_ds_all['parameterization_dA'].values[loc,:]=0
output_ds_all['parameterization_dV'].values[loc,:]=0
output_ds_all['parameterization_dV_bwl'].values[loc,:]=0
output_ds_all['parameterization_dV_eff'].values[loc,:]=0

# check the glacier volume in 2020
loc = np.where(np.isnan(output_ds_all['volume_2020'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['volume_2020'].values[:,0]) & (output_ds_all['volume_2020'].values[:,0] != 0))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['volume_2020'].values[loc,:] = griddata((find_lon, find_lat),
                                                      output_ds_all['volume_2020'].values[find_loc, :]/output_ds_all['area_2020'].values[find_loc, :], 
                                                      (loc_lon, loc_lat), method='nearest') * output_ds_all['area_2020'].values[loc, :]

#%% linear_regression_method

# intercept_ELA_mean
loc = np.where(np.isnan(output_ds_all['intercept_ELA_mean'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['intercept_ELA_mean'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['intercept_ELA_mean'].values[loc,:] = griddata((find_lon, find_lat),
                                                             output_ds_all['intercept_ELA_mean'].values[find_loc,:], 
                                                             (loc_lon, loc_lat), method='nearest')

# intercept_AAR_mean
loc = np.where(np.isnan(output_ds_all['intercept_AAR_mean'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['intercept_AAR_mean'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['intercept_AAR_mean'].values[loc,:] = griddata((find_lon, find_lat),
                                                             output_ds_all['intercept_AAR_mean'].values[find_loc,:], 
                                                             (loc_lon, loc_lat), method='nearest')

# intercept_AAR
loc = np.where(np.isnan(output_ds_all['intercept_AAR'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['intercept_AAR'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['intercept_AAR'].values[loc,:] = griddata((find_lon, find_lat),
                                                        output_ds_all['intercept_AAR'].values[find_loc,:], 
                                                        (loc_lon, loc_lat), method='nearest')

# intercept_a
loc = np.where(np.isnan(output_ds_all['intercept_a'].values[:,0]))[0]

output_ds_all['intercept_a'].values[loc,0] = output_ds_all['intercept_AAR_mean'].values[loc,0] / output_ds_all['intercept_AAR'].values[loc,0]
output_ds_all['intercept_a'].values[loc,1] = output_ds_all['intercept_a'].values[loc,0] * np.sqrt(
    (output_ds_all['intercept_AAR_mean'].values[loc,1]/output_ds_all['intercept_AAR_mean'].values[loc,0])**2 + 
    (output_ds_all['intercept_AAR'].values[loc,1]/output_ds_all['intercept_AAR'].values[loc,0]))

# intercept_dA
loc = np.where(np.isnan(output_ds_all['intercept_dA'].values[:,0]))[0]

output_ds_all['intercept_dA'].values[loc,0] = output_ds_all['area_2020'].values[loc,0] * \
    (output_ds_all['intercept_a'].values[loc,0] - 1)

output_ds_all['intercept_dA'].values[loc,1] = np.sqrt((output_ds_all['area_2020'].values[loc,1] * (output_ds_all['intercept_a'].values[loc,0]-1))**2 + 
                                                      (output_ds_all['intercept_a'].values[loc,1] * output_ds_all['area_2020'].values[loc,0])**2)

# intercept_dV
loc = np.where(np.isnan(output_ds_all['intercept_dV'].values[:,0]))[0]
icecap = np.where(output_ds_all['is_icecap'].values==1, 1.25, 1.375)

output_ds_all['intercept_dV'].values[loc,0] = output_ds_all['volume_2020'].values[loc,0] * \
    (output_ds_all['intercept_a'].values[loc,0] ** icecap[loc]-1)

output_ds_all['intercept_dV'].values[loc,1] = np.sqrt((output_ds_all['intercept_a'].values[loc,0]**icecap[loc]-1)**2 *
                                                      (output_ds_all['volume_2020'].values[loc,1])**2 + 
                                                      (output_ds_all['volume_2020'].values[loc,0]*icecap[loc]*
                                                       (output_ds_all['intercept_a'].values[loc,0]**(icecap[loc]-1))*
                                                       output_ds_all['intercept_a'].values[loc,1])**2)

# intercept_dV_bwl
loc = np.where(np.isnan(output_ds_all['intercept_dV_bwl'].values[:,0]) & (output_ds_all['is_tidewater'].values==0))[0]
output_ds_all['intercept_dV_bwl'].values[loc,:] = 0

loc = np.where(np.isnan(output_ds_all['intercept_dV_bwl'].values[:,0]) & (output_ds_all['is_tidewater'].values==1) & 
               (output_ds_all['intercept_dV'].values[:,0]!=0))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['intercept_dV_bwl'].values[:,0]) & (output_ds_all['is_tidewater'].values==1) & 
                    (output_ds_all['intercept_dV'].values[:,0]!=0))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['intercept_dV_bwl'].values[loc,:] = griddata((find_lon, find_lat),
                                                           output_ds_all['intercept_dV_bwl'].values[find_loc, :]/output_ds_all['intercept_dV'].values[find_loc, :], 
                                                           (loc_lon, loc_lat), method='nearest') * output_ds_all['intercept_dV'].values[loc, :]

# intercept_dV_eff
output_ds_all['intercept_dV_eff'].values[:,0] = output_ds_all['intercept_dV'].values[:,0] - output_ds_all['intercept_dV_bwl'].values[:,0]
output_ds_all['intercept_dV_eff'].values[:,1] = np.sqrt(output_ds_all['intercept_dV'].values[:,1]**2 + output_ds_all['intercept_dV_bwl'].values[:,1]**2)

# intercept_ELA_steay
loc = np.where(np.isnan(output_ds_all['intercept_ELA_steady'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['intercept_ELA_steady'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['intercept_ELA_steady'].values[loc,:] = griddata((find_lon, find_lat),
                                                               output_ds_all['intercept_ELA_steady'].values[find_loc,:], 
                                                               (loc_lon, loc_lat), method='nearest')

# intercept_THAR
loc = np.where(np.isnan(output_ds_all['intercept_THAR'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['intercept_THAR'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['intercept_THAR'].values[loc,:] = griddata((find_lon, find_lat),
                                                         output_ds_all['intercept_THAR'].values[find_loc,:], 
                                                         (loc_lon, loc_lat), method='nearest')

#%% equilibrium simulation
# equil_ELA_mean
loc = np.where(np.isnan(output_ds_all['equil_ELA_mean'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['equil_ELA_mean'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['equil_ELA_mean'].values[loc,:] = griddata((find_lon, find_lat),
                                                             output_ds_all['equil_ELA_mean'].values[find_loc,:], 
                                                             (loc_lon, loc_lat), method='nearest')

# equil_AAR
loc = np.where(np.isnan(output_ds_all['equil_AAR'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['equil_AAR'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['equil_AAR'].values[loc,:] = griddata((find_lon, find_lat),
                                                        output_ds_all['equil_AAR'].values[find_loc,:], 
                                                        (loc_lon, loc_lat), method='nearest')

# equil_a
loc = np.where(np.isnan(output_ds_all['equil_a'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['equil_a'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['equil_a'].values[loc,:] = griddata((find_lon, find_lat),
                                                  output_ds_all['equil_a'].values[find_loc,:], 
                                                  (loc_lon, loc_lat), method='nearest')

# equil_AAR_mean
loc = np.where(np.isnan(output_ds_all['equil_AAR_mean'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['equil_AAR_mean'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['equil_AAR_mean'].values[loc,0] = output_ds_all['equil_a'].values[loc,0] * \
    output_ds_all['equil_AAR'].values[loc,0]
    
output_ds_all['equil_AAR_mean'].values[loc,1] = output_ds_all['equil_AAR_mean'].values[loc,0] * np.sqrt(
    (output_ds_all['equil_a'].values[loc,1]/output_ds_all['equil_a'].values[loc,0])**2 + 
    (output_ds_all['equil_AAR'].values[loc,1]/output_ds_all['equil_AAR'].values[loc,0]))

# equil_dA
loc = np.where(np.isnan(output_ds_all['equil_dA'].values[:,0]))[0]

output_ds_all['equil_dA'].values[loc,0] = output_ds_all['area_2020'].values[loc,0] * \
    (output_ds_all['equil_a'].values[loc,0] - 1)

output_ds_all['equil_dA'].values[loc,1] = np.sqrt((output_ds_all['area_2020'].values[loc,1] * (output_ds_all['equil_a'].values[loc,0]-1))**2 + 
                                                      (output_ds_all['equil_a'].values[loc,1] * output_ds_all['area_2020'].values[loc,0])**2)

# equil_dV
loc = np.where(np.isnan(output_ds_all['equil_dV'].values[:,0]) & (output_ds_all['volume_2020'].values[:,0]!=0))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['equil_dV'].values[:,0]) & (output_ds_all['volume_2020'].values[:,0]!=0))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['equil_dV'].values[loc,:] = griddata((find_lon, find_lat), 
                                                   output_ds_all['equil_dV'].values[find_loc,:]/output_ds_all['volume_2020'].values[find_loc,:], 
                                                   (loc_lon, loc_lat), method='nearest') * output_ds_all['volume_2020'].values[loc,:]

# equil_dV_bwl
loc = np.where(np.isnan(output_ds_all['equil_dV_bwl'].values[:,0]) & (output_ds_all['is_tidewater'].values==0))[0]
output_ds_all['equil_dV_bwl'].values[loc,:] = 0

loc = np.where(np.isnan(output_ds_all['equil_dV_bwl'].values[:,0]) & (output_ds_all['is_tidewater'].values==1) & 
               (output_ds_all['equil_dV'].values[:,0]!=0))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['equil_dV_bwl'].values[:,0]) & (output_ds_all['is_tidewater'].values==1) & 
                    (output_ds_all['equil_dV'].values[:,0]!=0))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['equil_dV_bwl'].values[loc,:] = griddata((find_lon, find_lat),
                                                           output_ds_all['equil_dV_bwl'].values[find_loc, :]/output_ds_all['equil_dV'].values[find_loc, :], 
                                                           (loc_lon, loc_lat), method='nearest') * output_ds_all['equil_dV'].values[loc, :]

# equil_dV_eff
output_ds_all['equil_dV_eff'].values[:,0] = output_ds_all['equil_dV'].values[:,0] - output_ds_all['equil_dV_bwl'].values[:,0]
output_ds_all['equil_dV_eff'].values[:,1] = np.sqrt(output_ds_all['equil_dV'].values[:,1]**2 + output_ds_all['equil_dV_bwl'].values[:,1]**2)

# equil_ELA_steay
loc = np.where(np.isnan(output_ds_all['equil_ELA_steady'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['equil_ELA_steady'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['equil_ELA_steady'].values[loc,:] = griddata((find_lon, find_lat),
                                                           output_ds_all['equil_ELA_steady'].values[find_loc,:], 
                                                           (loc_lon, loc_lat), method='nearest')

# equil_THAR
loc = np.where(np.isnan(output_ds_all['equil_THAR'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['equil_THAR'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['equil_THAR'].values[loc,:] = griddata((find_lon, find_lat),
                                                     output_ds_all['equil_THAR'].values[find_loc,:], 
                                                     (loc_lon, loc_lat), method='nearest')

#%% linear_regression_method

# parameterization_ELA_mean
loc = np.where(np.isnan(output_ds_all['parameterization_ELA_mean'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['parameterization_ELA_mean'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['parameterization_ELA_mean'].values[loc,:] = griddata((find_lon, find_lat),
                                                             output_ds_all['parameterization_ELA_mean'].values[find_loc,:], 
                                                             (loc_lon, loc_lat), method='nearest')

# parameterization_AAR_mean
loc = np.where(np.isnan(output_ds_all['parameterization_AAR_mean'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['parameterization_AAR_mean'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['parameterization_AAR_mean'].values[loc,:] = griddata((find_lon, find_lat),
                                                             output_ds_all['parameterization_AAR_mean'].values[find_loc,:], 
                                                             (loc_lon, loc_lat), method='nearest')

# parameterization_AAR
loc = np.where(np.isnan(output_ds_all['parameterization_AAR'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['parameterization_AAR'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['parameterization_AAR'].values[loc,:] = griddata((find_lon, find_lat),
                                                        output_ds_all['parameterization_AAR'].values[find_loc,:], 
                                                        (loc_lon, loc_lat), method='nearest')

# parameterization_a
loc = np.where(np.isnan(output_ds_all['parameterization_a'].values[:,0]))[0]

output_ds_all['parameterization_a'].values[loc,0] = output_ds_all['parameterization_AAR_mean'].values[loc,0] / output_ds_all['parameterization_AAR'].values[loc,0]
output_ds_all['parameterization_a'].values[loc,1] = output_ds_all['parameterization_a'].values[loc,0] * np.sqrt(
    (output_ds_all['parameterization_AAR_mean'].values[loc,1]/output_ds_all['parameterization_AAR_mean'].values[loc,0])**2 + 
    (output_ds_all['parameterization_AAR'].values[loc,1]/output_ds_all['parameterization_AAR'].values[loc,0]))

# parameterization_dA
loc = np.where(np.isnan(output_ds_all['parameterization_dA'].values[:,0]))[0]

output_ds_all['parameterization_dA'].values[loc,0] = output_ds_all['area_2020'].values[loc,0] * \
    (output_ds_all['parameterization_a'].values[loc,0] - 1)

output_ds_all['parameterization_dA'].values[loc,1] = np.sqrt((output_ds_all['area_2020'].values[loc,1] * (output_ds_all['parameterization_a'].values[loc,0]-1))**2 + 
                                                      (output_ds_all['parameterization_a'].values[loc,1] * output_ds_all['area_2020'].values[loc,0])**2)

# parameterization_dV
loc = np.where(np.isnan(output_ds_all['parameterization_dV'].values[:,0]))[0]
icecap = np.where(output_ds_all['is_icecap'].values==1, 1.25, 1.375)

output_ds_all['parameterization_dV'].values[loc,0] = output_ds_all['volume_2020'].values[loc,0] * \
    (output_ds_all['parameterization_a'].values[loc,0] ** icecap[loc]-1)

output_ds_all['parameterization_dV'].values[loc,1] = np.sqrt((output_ds_all['parameterization_a'].values[loc,0]**icecap[loc]-1)**2 *
                                                      (output_ds_all['volume_2020'].values[loc,1])**2 + 
                                                      (output_ds_all['volume_2020'].values[loc,0]*icecap[loc]*
                                                       (output_ds_all['parameterization_a'].values[loc,0]**(icecap[loc]-1))*
                                                       output_ds_all['parameterization_a'].values[loc,1])**2)

# parameterization_dV_bwl
loc = np.where(np.isnan(output_ds_all['parameterization_dV_bwl'].values[:,0]) & (output_ds_all['is_tidewater'].values==1) & 
               (output_ds_all['parameterization_dV'].values[:,0]!=0))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['parameterization_dV_bwl'].values[:,0]) & (output_ds_all['is_tidewater'].values==1) & 
                    (output_ds_all['parameterization_dV'].values[:,0]!=0))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['parameterization_dV_bwl'].values[loc,:] = griddata((find_lon, find_lat),
                                                           output_ds_all['parameterization_dV_bwl'].values[find_loc, :]/output_ds_all['parameterization_dV'].values[find_loc, :], 
                                                           (loc_lon, loc_lat), method='nearest') * output_ds_all['parameterization_dV'].values[loc, :]

# parameterization_dV_eff
output_ds_all['parameterization_dV_eff'].values[:,0] = output_ds_all['parameterization_dV'].values[:,0] - output_ds_all['parameterization_dV_bwl'].values[:,0]
output_ds_all['parameterization_dV_eff'].values[:,1] = np.sqrt(output_ds_all['parameterization_dV'].values[:,1]**2 + output_ds_all['parameterization_dV_bwl'].values[:,1]**2)

# parameterization_ELA_steay
loc = np.where(np.isnan(output_ds_all['parameterization_ELA_steady'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['parameterization_ELA_steady'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['parameterization_ELA_steady'].values[loc,:] = griddata((find_lon, find_lat),
                                                               output_ds_all['parameterization_ELA_steady'].values[find_loc,:], 
                                                               (loc_lon, loc_lat), method='nearest')

# parameterization_THAR
loc = np.where(np.isnan(output_ds_all['parameterization_THAR'].values[:,0]))[0]
loc_lon = output_ds_all['CenLon'].values[loc]
loc_lat = output_ds_all['CenLat'].values[loc]

find_loc = np.where(~np.isnan(output_ds_all['parameterization_THAR'].values[:,0]))[0]
find_lon = output_ds_all['CenLon'].values[find_loc]
find_lat = output_ds_all['CenLat'].values[find_loc]

output_ds_all['parameterization_THAR'].values[loc,:] = griddata((find_lon, find_lat),
                                                         output_ds_all['parameterization_THAR'].values[find_loc,:], 
                                                         (loc_lon, loc_lat), method='nearest')

loc = np.where(tidewater_has_obs==0)[0]
output_ds_all['parameterization_ELA_mean'].values[loc,:]   = np.nan
output_ds_all['parameterization_AAR_mean'].values[loc,:]   = np.nan
output_ds_all['parameterization_AAR'].values[loc,:]        = np.nan
output_ds_all['parameterization_a'].values[loc,:]          = np.nan
output_ds_all['parameterization_dA'].values[loc,:]         = np.nan
output_ds_all['parameterization_dV'].values[loc,:]         = np.nan
output_ds_all['parameterization_ELA_steady'].values[loc,:] = np.nan
output_ds_all['parameterization_THAR'].values[loc,:]       = np.nan
output_ds_all['parameterization_dV_bwl'].values[loc,:]     = np.nan
output_ds_all['parameterization_dV_eff'].values[loc,:]     = np.nan

#%%
for reg in np.array([1,3,4,5,7,9,17,19]):
    loc = np.where((output_ds_all['O1Region'].values==reg) & 
                   (output_ds_all['is_tidewater'].values==1) & 
                   np.isnan(output_ds_all['parameterization_dV'].values[:,0]))
    
    find_id = np.where((output_ds_all['O1Region'].values==reg) & 
                   (output_ds_all['is_tidewater'].values==1) & 
                   ~np.isnan(output_ds_all['parameterization_dV'].values[:,0]))
    
    output_ds_all['parameterization_ELA_mean'].values[loc,0] = np.nanmedian(output_ds_all['parameterization_ELA_mean'].values[find_id,0])
    output_ds_all['parameterization_ELA_mean'].values[loc,1] = np.nanstd(output_ds_all['parameterization_ELA_mean'].values[find_id,0])
    output_ds_all['parameterization_AAR_mean'].values[loc,0] = np.nanmedian(output_ds_all['parameterization_AAR_mean'].values[find_id,0])
    output_ds_all['parameterization_AAR_mean'].values[loc,1] = np.nanstd(output_ds_all['parameterization_AAR_mean'].values[find_id,0])
    output_ds_all['parameterization_AAR'].values[loc,0] = np.nanmedian(output_ds_all['parameterization_AAR'].values[find_id,0])
    output_ds_all['parameterization_AAR'].values[loc,1] = np.nanstd(output_ds_all['parameterization_AAR'].values[find_id,0])
    output_ds_all['parameterization_a'].values[loc,0] = output_ds_all['parameterization_AAR_mean'].values[loc,0] / output_ds_all['parameterization_AAR'].values[loc,0]
    
    output_ds_all['parameterization_a'].values[loc,1] = output_ds_all['parameterization_a'].values[loc,0] * np.sqrt(
        (output_ds_all['parameterization_AAR_mean'].values[loc,1]/output_ds_all['parameterization_AAR_mean'].values[loc,0])**2 + 
        (output_ds_all['parameterization_AAR'].values[loc,1]/output_ds_all['parameterization_AAR'].values[loc,0]))
    
    output_ds_all['parameterization_dA'].values[loc,0] = output_ds_all['area_2020'].values[loc,0] * \
        (output_ds_all['parameterization_a'].values[loc,0] - 1)
    output_ds_all['parameterization_dA'].values[loc,1] = np.sqrt((output_ds_all['area_2020'].values[loc,1] * (output_ds_all['parameterization_a'].values[loc,0]-1))**2 + 
                                                          (output_ds_all['parameterization_a'].values[loc,1] * output_ds_all['area_2020'].values[loc,0])**2)
    
    icecap = np.where(output_ds_all['is_icecap'].values==1, 1.25, 1.375)
    output_ds_all['parameterization_dV'].values[loc,0] = output_ds_all['volume_2020'].values[loc,0] * \
        (output_ds_all['parameterization_a'].values[loc,0] ** icecap[loc]-1)
    output_ds_all['parameterization_dV'].values[loc,1] = np.sqrt((output_ds_all['parameterization_a'].values[loc,0]**icecap[loc]-1)**2 *
                                                          (output_ds_all['volume_2020'].values[loc,1])**2 + 
                                                          (output_ds_all['volume_2020'].values[loc,0]*icecap[loc]*
                                                           (output_ds_all['parameterization_a'].values[loc,0]**(icecap[loc]-1))*
                                                           output_ds_all['parameterization_a'].values[loc,1])**2)
    
    output_ds_all['parameterization_dV_bwl'].values[loc,0] = np.nanmedian(output_ds_all['parameterization_dV_bwl'].values[find_id,0] / 
                                                                          output_ds_all['parameterization_dV'].values[find_id,0]) * output_ds_all['parameterization_dV'].values[loc,0]
    output_ds_all['parameterization_dV_bwl'].values[loc,1] = np.nanstd(output_ds_all['parameterization_dV_bwl'].values[find_id,0] / 
                                                                       output_ds_all['parameterization_dV'].values[find_id,0]) * output_ds_all['parameterization_dV'].values[loc,0]
    
    output_ds_all['parameterization_dV_eff'].values[loc,0] = output_ds_all['parameterization_dV'].values[loc,0] - output_ds_all['parameterization_dV_bwl'].values[loc,0]
    output_ds_all['parameterization_dV_eff'].values[loc,1] = np.sqrt(output_ds_all['parameterization_dV'].values[loc,1]**2 + output_ds_all['parameterization_dV_bwl'].values[loc,1]**2)
    
    output_ds_all['parameterization_ELA_steady'].values[loc,0] = np.nanmedian(output_ds_all['parameterization_ELA_steady'].values[find_id,:])
    output_ds_all['parameterization_ELA_steady'].values[loc,1] = np.nanstd(output_ds_all['parameterization_ELA_steady'].values[find_id,:])
    output_ds_all['parameterization_THAR'].values[loc,0] = np.nanmedian(output_ds_all['parameterization_THAR'].values[find_id,:])
    output_ds_all['parameterization_THAR'].values[loc,1] = np.nanstd(output_ds_all['parameterization_THAR'].values[find_id,:])
#%%

output_ds_all.to_netcdf('/Users/wyan0065/Desktop/PyGEM/calving/Output/ERA5_MCMC_ba1_2014_2023_corrected.nc');
# Close datasets
output_ds_all.close();

