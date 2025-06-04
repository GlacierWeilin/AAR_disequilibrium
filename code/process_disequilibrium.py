#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Source : PyGEMv0.2.5 developed by David Rounce (drounce@alaska.edu)
Further developed by: Weilin Yang (weilinyang.yang@monash.edu)
Code reviewed by: Wenchao Chu (peterchuwenchao@foxmail.com)

"""
import numpy as np
import pandas as pd
import xarray as xr
import os
import collections

import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup

#%% ====== Read RGI tables =====

regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
main_glac_rgi_all = pd.DataFrame();
for reg in regions:
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], rgi_regionsO2='all', rgi_glac_number='all', 
                                                      glac_no=None, rgi_fp=pygem_prms.rgi_fp);
    main_glac_rgi_all = pd.concat([main_glac_rgi_all, main_glac_rgi], axis=0);

noresults = pd.read_csv(pygem_prms.output_filepath+'no_nc_outputs.csv')
main_glac_rgi_all = main_glac_rgi_all[~main_glac_rgi_all['RGIId'].isin(noresults['RGIId'])]
[m,n]=np.shape(main_glac_rgi_all);

#%% ===== Create empty xarray dataset =====

glac_values = np.arange(m);

sims = np.arange(pygem_prms.sim_iters)
dim  = np.arange(8)

intercept_dim = np.arange(16)
equil_dim     = np.arange(11)
    
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

# simulated glacier results in 2020
output_coords_dict['area_2020']       = collections.OrderedDict([('glac', glac_values),
                                                                 ('sims', sims)])
output_coords_dict['volume_2020']     = collections.OrderedDict([('glac', glac_values),
                                                                 ('sims', sims)])
output_coords_dict['volume_bsl_2020'] = collections.OrderedDict([('glac', glac_values),
                                                                 ('sims', sims)])

# results of parameterization approach and equilibrium simulaiton
# intercept_results
# 0-intercept_ELA_mean, 1-intercept_AAR_mean, 2-intercept_AAR, 3-intercept_a, 4-intercept_dA, 5-intercept_dV
# 6-intercept_ELA_steady, 7-intercept_THAR, 8-intercept_At, 9-intercept_dV_bwl, 10-intercept_dV_eff
output_coords_dict['intercept_ELA_mean']     = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['intercept_AAR_mean']     = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['intercept_AAR']          = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['intercept_a']            = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['intercept_dA']           = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['intercept_dV']           = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['intercept_ELA_steady']   = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['intercept_THAR']         = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['intercept_At']           = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['intercept_dV_bwl']       = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['intercept_dV_eff']       = collections.OrderedDict([('glac', glac_values),('sims', sims)])

# equil_results
# 0-equil_ELA_mean, 1-equil_AAR_mean, 2-equil_AAR, 3-equil_a, 4-equil_dA, 5-equil_dV, 6-equil_ELA_equil
# 7-equil_THAR, 8-equil_At, 9-equil_dV_bwl, 10-equil_dV_eff
output_coords_dict['equil_ELA_mean']   = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['equil_AAR_mean']   = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['equil_AAR']        = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['equil_a']          = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['equil_dA']         = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['equil_dV']         = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['equil_ELA_steady'] = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['equil_THAR']       = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['equil_At']         = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['equil_dV_bwl']     = collections.OrderedDict([('glac', glac_values),('sims', sims)])
output_coords_dict['equil_dV_eff']     = collections.OrderedDict([('glac', glac_values),('sims', sims)])

output_coords_dict['equil_time']    = collections.OrderedDict([('glac', glac_values)])

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
                'comment': '0-mean, 1-std, 2-2.5%, 3-25%, 4-median, 5-75%, 6-97.5%, 7-mad'},
        'intercept_dim': {
                'long_name': 'AAR results by linear regression',
                'comment': '0-intercept_ELA_mean, 1-intercept_AAR_mean, 2-intercept_AAR, 3-intercept_a, 4-intercept_dA, 5-intercept_dV, \
                    6-intercept_ELA_steady, 7-intercept_THAR, 8-intercept_At, 9-intercept_dV_bwl, 10-intercept_dV_eff'},
        'equil_dim': {
                'long_name': 'AAR results through equilibrium experiment',
                'comment': '0-equil_ELA_mean, 1-equil_AAR_mean, 2-equil_AAR, 3-equil_a, 4-equil_dA, 5-equil_dV, 6-equil_ELA_equil, \
                    7-equil_THAR, 8-equil_At, 9-equil_dV_bwl, 10-equil_dV_eff'},
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

#%% ===== Compile simulation results =====
filepath = pygem_prms.output_filepath + '/simulations/';
RGIId=[];
output_ds_all['equil_time'].values[:] = np.zeros(m) * np.nan

for i in range(0,m):
    rgi_id = main_glac_rgi_all.RGIId.values[i];
    stats_fp = filepath + rgi_id[6:8] + '/ERA5/stats/';
    
    if pygem_prms.include_debris:
        debris_fp = pygem_prms.debris_fp;
    else:
        debris_fp = pygem_prms.output_filepath + '/../debris_data/';
    
    if int(rgi_id[6:8]) <10:
        stats_fn = rgi_id[7:14] + '_ERA5_MCMC_ba1_50sets_2014_2023_all.nc';
        debris_fn = debris_fp + 'ed_tifs/' + rgi_id[6:8] + '/' + rgi_id[7:14] + '_meltfactor.tif';
    else:
        stats_fn = rgi_id[6:14] + '_ERA5_MCMC_ba1_50sets_2014_2023_all.nc';
        debris_fn = debris_fp + 'ed_tifs/' + rgi_id[6:8] + '/' + rgi_id[6:14] + '_meltfactor.tif';
    
    stats = xr.open_dataset(stats_fp+stats_fn);
    RGIId = np.append(RGIId, stats['RGIId'].values);

    output_ds_all['CenLon'].values[i]       = stats['CenLon'].values;
    output_ds_all['CenLat'].values[i]       = stats['CenLat'].values;
    output_ds_all['O1Region'].values[i]     = stats['O1Region'].values;
    output_ds_all['O2Region'].values[i]     = stats['O2Region'].values;
    output_ds_all['is_tidewater'].values[i] = stats['is_tidewater'].values;
    output_ds_all['is_icecap'].values[i]    = stats['is_icecap'].values;
    output_ds_all['is_debris'].values[i]    = os.path.exists(debris_fn);

    # simulated glacier results in 2020
    output_ds_all['area_2020'].values[i,:]       = stats['area_2020'].values[0,:]
    output_ds_all['volume_2020'].values[i,:]     = stats['volume_2020'].values[0,:]
    output_ds_all['volume_bsl_2020'].values[i,:] = stats['volume_bsl_2020'].values[0,:]

    # intercept_results
    # 0-intercept_ELA_mean, 1-intercept_AAR_mean, 2-intercept_AAR, 3-intercept_a, 4-intercept_dA, 5-intercept_dV
    # 6-intercept_ELA_steady, 7-intercept_THAR, 8-intercept_At, 9-intercept_dV_bwl, 10-intercept_dV_eff
    output_ds_all['intercept_ELA_mean'].values[i,:]     = stats['intercept_results'].values[0,0,:]; 
    output_ds_all['intercept_AAR_mean'].values[i,:]     = stats['intercept_results'].values[0,1,:]; 
    output_ds_all['intercept_AAR'].values[i,:]          = stats['intercept_results'].values[0,2,:];
    output_ds_all['intercept_a'].values[i,:]            = stats['intercept_results'].values[0,3,:]; 
    output_ds_all['intercept_dA'].values[i,:]           = stats['intercept_results'].values[0,4,:];
    output_ds_all['intercept_dV'].values[i,:]           = stats['intercept_results'].values[0,5,:]; 
    output_ds_all['intercept_ELA_steady'].values[i,:]   = stats['intercept_results'].values[0,6,:];
    output_ds_all['intercept_THAR'].values[i,:]         = stats['intercept_results'].values[0,7,:]; 
    output_ds_all['intercept_At'].values[i,:]           = stats['intercept_results'].values[0,8,:];
    output_ds_all['intercept_dV_bwl'].values[i,:]       = stats['intercept_results'].values[0,9,:]; 
    output_ds_all['intercept_dV_eff'].values[i,:]       = stats['intercept_results'].values[0,10,:];
    
    # equil_results
    # 0-equil_ELA_mean, 1-equil_AAR_mean, 2-equil_AAR, 3-equil_a, 4-equil_dA, 5-equil_dV, 6-equil_ELA_equil
    # 7-equil_THAR, 8-equil_At, 9-equil_dV_bwl, 10-equil_dV_eff
    output_ds_all['equil_ELA_mean'].values[i,:]   = stats['equil_results'].values[0,0,:];
    output_ds_all['equil_AAR_mean'].values[i,:]   = stats['equil_results'].values[0,1,:];
    output_ds_all['equil_AAR'].values[i,:]        = stats['equil_results'].values[0,2,:];
    output_ds_all['equil_a'].values[i,:]          = stats['equil_results'].values[0,3,:];
    output_ds_all['equil_dA'].values[i,:]         = stats['equil_results'].values[0,4,:];
    output_ds_all['equil_dV'].values[i,:]         = stats['equil_results'].values[0,5,:];
    output_ds_all['equil_ELA_steady'].values[i,:] = stats['equil_results'].values[0,6,:];
    output_ds_all['equil_THAR'].values[i,:]       = stats['equil_results'].values[0,7,:];
    output_ds_all['equil_At'].values[i,:]         = stats['equil_results'].values[0,8,:];
    output_ds_all['equil_dV_bwl'].values[i,:]     = stats['equil_results'].values[0,9,:];
    output_ds_all['equil_dV_eff'].values[i,:]     = stats['equil_results'].values[0,10,:];
    
    # output_climate, equil_climate
    # 0-temp, 1-prec, 2-acc, 3-refreeze, 4-melt, 5-frontalablation, 6-massbaltotal
    output_ds_all['output_climate'].values[i,:,:] = stats['output_climate'].values[0,:,:]
    output_ds_all['equil_climate'].values[i,:,:]  = stats['equil_climate'].values[0,:,:]

    # Parameters
    # 0-debris_hd, 1-debris_ed, 2-debris_area
    # 3-kp, 4-tbias, 5-ddfsnow, 6-ddfice, 7-tsnow_threshold, 8-precgrad, 9-calving_k_values, 10-water_level
    output_ds_all['params'].values[i,:] = stats['params'].values[0,:,:]
    
    if ~np.isnan(stats['equil_volume'].values[0,0,4]):
        if np.shape(stats['equil_volume'].values)[1]==5001:
            volume = np.mean((stats['equil_volume'].values[0,:-1,4]).reshape(500,10),axis=1)
        else:
            volume = np.mean((stats['equil_volume'].values[0,:-1,4]).reshape(200,10),axis=1)

        if volume[-1] == 0:
            loc = np.where(volume==0)[0]
        else:
            loc = np.where((volume-volume[-1])/volume[-1]<0.01)[0]
        if len(loc) !=0:
            output_ds_all['equil_time'].values[i] = loc[0]

output_ds_all['RGIId'].values = RGIId;

#%% ===== Export Results =====

output_fp = pygem_prms.output_filepath;
output_fn = ('simulations/ERA5_MCMC_' + 'ba' + str(pygem_prms.option_bias_adjustment) + '_' + str(pygem_prms.sim_iters) + 'sets' + '_' + 
             str(pygem_prms.gcm_startyear) + '_' + str(pygem_prms.gcm_endyear) + '_all.nc');
output_ds_all.to_netcdf(output_fp+output_fn);
            
# Close datasets
output_ds_all.close();