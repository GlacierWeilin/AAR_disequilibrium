#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Source : PyGEMv0.2.5 developed by David Rounce (drounce@alaska.edu)
Further developed by: Weilin Yang (weilinyang.yang@monash.edu)
Code reviewed by: Wenchao Chu (peterchuwenchao@foxmail.com)

"""
# Default climate data is ERA-Interim; specify CMIP5 by specifying a filename to the argument:
#    (Command line) python run_simulation_list_multiprocess.py -gcm_list_fn=C:\...\gcm_rcpXX_filenames.txt
#      - Default is running ERA-Interim in parallel with five processors.
#    (Spyder) %run run_simulation_list_multiprocess.py C:\...\gcm_rcpXX_filenames.txt -option_parallels=0
#      - Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Built-in libraries
import argparse
import collections
import inspect
import logging
import multiprocessing
import os
from packaging.version import Version
import time
import calendar
import warnings

# External libraries
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_abs_deviation
from statsmodels.tsa.stattools import adfuller
import xarray as xr
import scipy.stats as st

# Local libraries
import class_climate
import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup
from pygem.massbalance import PyGEMMassBalance
from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import single_flowline_glacier_directory
from pygem.oggm_compat import single_flowline_glacier_directory_with_calving
from pygem.shop import debris
import pygemfxns_gcmbiasadj as gcmbiasadj
import spc_split_glaciers as split_glaciers

from oggm import __version__ as oggm_version
from oggm import cfg
from oggm import graphics
from oggm import tasks
from oggm import utils
from oggm.core import climate
from oggm.core.flowline import FluxBasedModel
from oggm.core.inversion import find_inversion_calving_from_any_mb

# Module logger
log = logging.getLogger(__name__)
cfg.set_logging_config(pygem_prms.logging_level)

cfg.PARAMS['hydro_month_nh']=1
cfg.PARAMS['hydro_month_sh']=1
cfg.PARAMS['trapezoid_lambdas'] = 1

# ----- FUNCTIONS -----
def getparser():
    """
    Use argparse to add arguments from the command line

    Parameters
    ----------
    gcm_list_fn (optional) : str
        text file that contains the climate data to be used in the model simulation
    gcm_name (optional) : str
        gcm name
    scenario (optional) : str
        representative concentration pathway or shared socioeconomic pathway (ex. 'rcp26', 'ssp585')
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    option_parallels (optional) : int
        switch to use parallels or not
    rgi_glac_number_fn (optional) : str
        filename of .pkl file containing a list of glacier numbers that used to run batches on the supercomputer
    batch_number (optional): int
        batch number used to differentiate output on supercomputer
    option_ordered : int
        option to keep glaciers ordered or to grab every n value for the batch
        (the latter helps make sure run times on each core are similar as it removes any timing differences caused by
         regional variations)
    debug (optional) : int
        Switch for turning debug printing on or off (default = 0 (off))
    debug_spc (optional) : int
        Switch for turning debug printing of spc on or off (default = 0 (off))

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run simulations from gcm list in parallel")
    # add arguments
    parser.add_argument('-gcm_list_fn', action='store', type=str, default=pygem_prms.ref_gcm_name,
                        help='text file full of commands to run')
    parser.add_argument('-gcm_name', action='store', type=str, default=None,
                        help='GCM name used for model run')
    parser.add_argument('-scenario', action='store', type=str, default=None,
                        help='rcp or ssp scenario used for model run (ex. rcp26 or ssp585)')
    parser.add_argument('-gcm_startyear', action='store', type=int, default=pygem_prms.gcm_startyear,
                        help='start year for the model run')
    parser.add_argument('-gcm_endyear', action='store', type=int, default=pygem_prms.gcm_endyear,
                        help='start year for the model run')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=1,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=0,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-batch_number', action='store', type=int, default=None,
                        help='Batch number used to differentiate output on supercomputer')
    parser.add_argument('-option_ordered', action='store', type=int, default=1,
                        help='switch to keep lists ordered or not')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off')
    parser.add_argument('-debug_spc', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off')
    return parser

def calc_stats_array(data, stats_cns=pygem_prms.sim_stat_cns):
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
        if 'mean' in stats_cns:
            if stats is None:
                stats = np.nanmean(data)
        if 'std' in stats_cns:
            stats = np.append(stats, np.nanstd(data))
        if '2.5%' in stats_cns:
            stats = np.append(stats, np.nanpercentile(data, 2.5))
        if '25%' in stats_cns:
            stats = np.append(stats, np.nanpercentile(data, 25))
        if 'median' in stats_cns:
            if stats is None:
                stats = np.nanmedian(data)
            else:
                stats = np.append(stats, np.nanmedian(data))
        if '75%' in stats_cns:
            stats = np.append(stats, np.nanpercentile(data, 75))
        if '97.5%' in stats_cns:
            stats = np.append(stats, np.nanpercentile(data, 97.5))
        if 'mad' in stats_cns:
            stats = np.append(stats, median_abs_deviation(data, nan_policy='omit'))
    else:
        if 'mean' in stats_cns:
            if stats is None:
                stats = np.nanmean(data,axis=1)[:,np.newaxis]
        if 'std' in stats_cns:
            stats = np.append(stats, np.nanstd(data,axis=1)[:,np.newaxis], axis=1)
        if '2.5%' in stats_cns:
            stats = np.append(stats, np.nanpercentile(data, 2.5, axis=1)[:,np.newaxis], axis=1)
        if '25%' in stats_cns:
            stats = np.append(stats, np.nanpercentile(data, 25, axis=1)[:,np.newaxis], axis=1)
        if 'median' in stats_cns:
            if stats is None:
                stats = np.nanmedian(data, axis=1)[:,np.newaxis]
            else:
                stats = np.append(stats, np.nanmedian(data, axis=1)[:,np.newaxis], axis=1)
        if '75%' in stats_cns:
            stats = np.append(stats, np.nanpercentile(data, 75, axis=1)[:,np.newaxis], axis=1)
        if '97.5%' in stats_cns:
            stats = np.append(stats, np.nanpercentile(data, 97.5, axis=1)[:,np.newaxis], axis=1)
        if 'mad' in stats_cns:
            stats = np.append(stats, median_abs_deviation(data, axis=1, nan_policy='omit')[:,np.newaxis], axis=1)
    return stats

def parameterization_approach(gdir, bed_h, massbaltotal_monthly,
                              ELA_annual, area_annual, volume_annual, volume_bsl_annual, 
                              bin_icethickness_annual, bin_area_annual, bin_massbalclim_annual,
                              water_level_all, nyears, count_exceed_boundary_errors):
    """
    Calculate glacier AAR using parameterization approach for a given glacier

    Parameters
    ----------
    gdir : OGGM GlacierDirectory
    bed_h : xarray dataset
            output_glac_bin_bed_h
    massbaltotal_monthly : xarray dataset
                           output_glac_massbaltotal_monthly
    
    ELA_annual : xarray dataset
                 output_glac_ELA_annual
    area_annual : xarray dataset
                  output_glac_area_annual
    volume_annual : xarray dataset
                    output_glac_volume_annual
    volume_bsl_annual : xarray dataset
                        output_glac_volume_bsl_annual
    
    bin_icethickness_annual : xarray dataset
                              output_glac_bin_icethickness_annual
    bin_area_annual : xarray dataset
                      output_glac_bin_area_annual
    bin_massbalclim_annual : xarray dataset
                             output_glac_bin_massbalclim_annual
    
    water_level_all : float
        cls = gdir.read_pickle('inversion_input')[-1]
        th = cls['hgt'][-1]
        vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
        water_level = utils.clip_scalar(0, th - vmax, th - vmin)
    nyears : int
            spinup years
    count_exceed_boundary_errors: int
            number of failed simulations
    
    Returns
    -------
    stats : np.array
    """
    
    # Imbalance(a) = AAR_mean/AAR;
    # Area change (dA) = A*(a-1);
    # Volume change (dV) = V*(a^r-1);
    # r=1.375 for glaciers; r=1.25 for ice caps;
    if gdir.is_icecap == False:
        r = 1.375;
    else:
        r = 1.25;
    
    n     = pygem_prms.gcm_endyear-pygem_prms.gcm_startyear+1
    end   = -1
    start = end-n
    
    sims = pygem_prms.sim_iters
    
    results_2020 = np.zeros([3,sims]) * np.nan
    intercept_results = np.zeros([16,sims]) * np.nan
    
    # ========================== intercept ==========================
    intercept_ELA_mean     = np.zeros(sims) * np.nan
    intercept_AAR_mean     = np.zeros(sims) * np.nan
    intercept_AAR          = np.zeros(sims) * np.nan
    intercept_a            = np.zeros(sims) * np.nan
    intercept_dA           = np.zeros(sims) * np.nan
    intercept_dV           = np.zeros(sims) * np.nan
    intercept_ELA_steady   = np.zeros(sims) * np.nan
    intercept_THAR         = np.zeros(sims) * np.nan
    intercept_At           = np.zeros(sims) * np.nan
    intercept_dV_bwl       = np.zeros(sims)
    intercept_dV_eff       = np.zeros(sims) * np.nan
    intercept_n_regression = np.zeros(sims) * np.nan
    intercept_slope        = np.zeros(sims) * np.nan
    intercept_r_value      = np.zeros(sims) * np.nan
    intercept_p_value      = np.zeros(sims) * np.nan
    intercept_std_err      = np.zeros(sims) * np.nan
    
    # initial inputs
    massbaltotal_monthly    = massbaltotal_monthly.reshape(nyears,12,sims);
    massbaltotal            = (massbaltotal_monthly.sum(axis=1))[start+1:, :];
    
    ELA = ELA_annual[start:end,:];
    area_2020       = area_annual[-5,:];
    volume_2020     = volume_annual[-5,:];
    volume_bsl_2020 = volume_bsl_annual[-5,:];
    
    bin_area         = bin_area_annual[:,start:end,:];
    bin_massbalclim  = bin_massbalclim_annual[:,start:end,:];
    
    if count_exceed_boundary_errors < sims:
        
        for i in range(sims):
            
            water_level = water_level_all[i]
            # use the glacier geometry from 2020
            nglacier = np.where(bin_area_annual[:,-5,i]>0)[0]
            nbin_bed     = bed_h[nglacier,i]
            nbin_thk     = bin_icethickness_annual[nglacier,-5,i]
            nbin_area    = bin_area_annual[nglacier,-5,i]
            nbin_surface = nbin_bed + nbin_thk

            if (np.isnan(ELA[:,i])).all():
                continue;

            # ========================== calculate AAR0 by linear regression ==========================
            smb = [];
            AAR = [];
            ela_yearly = [];

            for j in range(n):
                tot_area = np.nansum(bin_area[:,j,i])
                
                tot_area_above = 0;
                is_above = bin_massbalclim[:,j,i]>0;
                tot_area_above = np.nansum(bin_area[:,j,i] * is_above)
                
                if tot_area != 0 and np.isnan(tot_area)==0:
                    _aar = tot_area_above/tot_area;
                    
                    if _aar >= 0.05 and _aar <= 0.95 and np.isnan(_aar)+np.isnan(massbaltotal[j,i])+np.isnan(ELA[j,i])==0:
                        smb = np.append(smb, massbaltotal[j,i])
                        AAR = np.append(AAR, _aar)
                        ela_yearly = np.append(ela_yearly, ELA[j,i])

            intercept_n_regression[i] = len(AAR);
            if len(AAR) >= 5:
                intercept_ELA_mean[i] = ela_yearly.mean();
                intercept_AAR_mean[i] = AAR.mean();
                # calculate AAR0 by linear regression
                slope, intercept, r_value, p_value, std_err = st.linregress(smb, AAR);
                intercept_slope[i]   = slope;
                intercept_AAR[i]     = intercept;
                intercept_r_value[i] = r_value;
                intercept_p_value[i] = p_value;
                intercept_std_err[i] = std_err;

                if intercept >= 0.05 and intercept <= 0.95 and area_2020[i] != 0:
                    intercept_a[i]  = intercept_AAR_mean[i]/intercept; 
                    intercept_dA[i] = area_2020[i]*(intercept_a[i]-1);
                    intercept_dV[i] = volume_2020[i]*(pow(intercept_a[i],r)-1);
                    
                    # ELA_steady
                    acc_all = area_2020[i]*intercept;
                    acc     = []
                    for j in range(len(nbin_area)):
                        acc = np.append(acc, np.sum(nbin_area[0:j+1]))
                    acc = acc-acc_all;
                    ela_location = np.where(acc<0);
                    ela_judge = np.where(abs(ela_location[0][0:-1]-ela_location[0][1:])==1);
                    if len(ela_judge[0]) != 0:
                        ela_location = (ela_judge)[0][-1]+1;
                    elif len(ela_location[0]) != 0:
                        ela_location = ela_location[0][0];
                    else:
                        ela_location = 0
                        
                    if ela_location+1 == len(acc):
                        intercept_ELA_steady[i] = nbin_surface[ela_location]
                    else:
                        intercept_ELA_steady[i] = (nbin_surface[ela_location]+nbin_surface[ela_location+1])/2
                    
                    # THAR_steady and At_steady
                    Ah = nbin_surface[0];
                    At = nbin_surface[-1];
                    intercept_At[i] = At;
                    if Ah != At:
                        intercept_THAR[i] = (intercept_ELA_steady[i]-At)/(Ah-At);
                        
                        if intercept_THAR[i] != 1:
                            At = (intercept_THAR[i]*Ah-intercept_ELA_mean[i])/(intercept_THAR[i]-1);
                        else:
                            At = intercept_ELA_steady[i];
                    else:
                        intercept_THAR[i] = np.nan
                        
                    # volume below sea level
                    if gdir.is_tidewater:
                        bwl = (nbin_surface < At) & (nbin_bed < water_level)
                        calving_thk = nbin_thk.copy()
                        calving_thk[~bwl] = 0.0;
                        calving_thk[bwl] = utils.clip_max(nbin_surface[bwl], water_level) - nbin_bed[bwl];
                        intercept_dV_bwl[i] = -np.sum(calving_thk * nbin_area);
                        if np.isnan(intercept_dV_bwl[i]):
                            intercept_dV_bwl[i] = 0;
                    if intercept_dV[i] < 0:
                        dV_judge = -1;
                    else:
                        dV_judge = 1;
                    if abs(intercept_dV[i]) > abs(intercept_dV_bwl[i]):
                        intercept_dV_eff[i] = (abs(intercept_dV[i])-abs(intercept_dV_bwl[i]))*dV_judge;
                    else:
                        intercept_dV_eff[i] = 0;
                    
                    if intercept_THAR[i] < 0 or intercept_THAR[i] > 1:
                        intercept_THAR[i] = np.nan
                        intercept_ELA_steady[i] = np.nan
                        intercept_dV_bwl[i] = np.nan
                        intercept_dV_eff[i] = intercept_dV[i]
                    
                elif intercept == 0: # The glacier will disappear.
                    intercept_dA[i] = -abs(area_2020[i]);
                    intercept_dV[i] = -abs(volume_2020[i]);
                    if gdir.is_tidewater:
                        intercept_dV_bwl[i] = -abs(volume_bsl_2020[i]); # below sea-level, water_level=0
                    intercept_dV_eff[i] = (abs(intercept_dV[i])-abs(intercept_dV_bwl[i]))*-1;
                    intercept_AAR[i] = np.nan;
                else:
                    intercept_AAR[i] = np.nan;
            else:
                intercept_AAR[i] = np.nan;
            
    # Output
    AAR_nan = np.where(np.isnan(intercept_AAR));
    intercept_ELA_mean[AAR_nan]     = np.nan;
    intercept_AAR_mean[AAR_nan]     = np.nan;
    intercept_AAR[AAR_nan]          = np.nan;
    intercept_a[AAR_nan]            = np.nan;
    intercept_dA[AAR_nan]           = np.nan;
    intercept_dV[AAR_nan]           = np.nan;
    intercept_ELA_steady[AAR_nan]   = np.nan;
    intercept_THAR[AAR_nan]         = np.nan;
    intercept_At[AAR_nan]           = np.nan;
    intercept_dV_bwl[AAR_nan]       = np.nan;
    intercept_dV_eff[AAR_nan]       = np.nan;
    intercept_n_regression[AAR_nan] = np.nan;
    intercept_slope[AAR_nan]        = np.nan;
    intercept_r_value[AAR_nan]      = np.nan;
    intercept_p_value[AAR_nan]      = np.nan;
    intercept_std_err[AAR_nan]      = np.nan;
    
    # results_2020
    # 0-area_2020_m2, 1-volume_2020_m3, 2-volume_bsl_2020_m3
    results_2020[0,:] = area_2020;
    results_2020[1,:] = volume_2020;
    results_2020[2,:] = volume_bsl_2020;
    
    # intercept_results
    # 0-intercept_ELA_mean, 1-intercept_AAR_mean, 2-intercept_AAR, 3-intercept_a, 4-intercept_dA, 5-intercept_dV
    # 6-intercept_ELA_steady, 7-intercept_THAR, 8-intercept_At, 9-intercept_dV_bwl, 10-intercept_dV_eff
    # 11-intercept_n_regression, 12-intercept_slope, 13-intercept_r_value, 14-intercept_p_value, 15-intercept_std_err,
    intercept_results[0,:]  = intercept_ELA_mean;
    intercept_results[1,:]  = intercept_AAR_mean;
    intercept_results[2,:]  = intercept_AAR;
    intercept_results[3,:]  = intercept_a;
    intercept_results[4,:]  = intercept_dA;
    intercept_results[5,:]  = intercept_dV;
    intercept_results[6,:]  = intercept_ELA_steady;
    intercept_results[7,:]  = intercept_THAR;
    intercept_results[8,:]  = intercept_At;
    intercept_results[9,:]  = intercept_dV_bwl;
    intercept_results[10,:] = intercept_dV_eff;
    intercept_results[11,:] = intercept_n_regression;
    intercept_results[12,:] = intercept_slope;
    intercept_results[13,:] = intercept_r_value;
    intercept_results[14,:] = intercept_p_value;
    intercept_results[15,:] = intercept_std_err;
    
    return results_2020, intercept_results

def equilibrium_experiment(equil_gdir, bed_h, equil_ELA_annual, equil_area_annual, equil_volume_annual, equil_volume_bsl_annual,
                           equil_bin_icethickness_annual, equil_bin_area_annual, equil_bin_massbalclim_annual, 
                           water_level_all, equil_nyears, equil_count_exceed_boundary_errors, count_exceed_boundary_errors):
    """
    Calculate glacier AAR using equilibrium experiment for a given glacier

    Parameters
    ----------
    equil_gdir : OGGM GlacierDirectory
    bed_h : xarray dataset
            output_glac_bin_bed_h
    
    equil_ELA_annual : xarray dataset
                       equil_glac_ELA_annual
    equil_area_annual : xarray dataset
                        equil_glac_area_annual
    equil_volume_annual : xarray dataset
                          equil_glac_volume_annual
    equil_volume_bsl_annual : xarray dataset
                              equil_glac_volume_bsl_annual
    
    equil_bin_icethickness_annual : xarray dataset
                                    equil_glac_bin_icethickness_annual
    equil_bin_area_annual : xarray dataset
                            equil_glac_bin_area_annual
    equil_bin_massbalclim_annual : xarray dataset
                                   equil_glac_bin_massbalclim_annual
    
    water_level_all : float
        cls = gdir.read_pickle('inversion_input')[-1]
        th = cls['hgt'][-1]
        vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
        water_level = utils.clip_scalar(0, th - vmax, th - vmin)
    equil_nyears : int
                  length_exp
    equil_count_exceed_boundary_errors: int
            number of failed simulations
    count_exceed_boundary_errors: int
            number of failed spin-up simulations
    
    Returns
    -------
    stats : np.array
    """
    
    sims = pygem_prms.sim_iters
    
    equil_ELA_mean   = np.zeros(sims) * np.nan
    equil_AAR_mean   = np.zeros(sims) * np.nan
    equil_AAR        = np.zeros(sims) * np.nan
    equil_a          = np.zeros(sims) * np.nan
    equil_dA         = np.zeros(sims) * np.nan
    equil_dV         = np.zeros(sims) * np.nan
    equil_ELA_steady = np.zeros(sims) * np.nan
    equil_THAR       = np.zeros(sims) * np.nan
    equil_At         = np.zeros(sims) * np.nan
    equil_dV_bwl     = np.zeros(sims)
    equil_dV_eff     = np.zeros(sims) * np.nan
    equil_Ah         = np.zeros(sims) * np.nan
    
    equil_results    = np.zeros([11,sims]) * np.nan
    
    # ========================== equilibrium simulation ==========================

    if equil_count_exceed_boundary_errors+count_exceed_boundary_errors < sims:
        # steady-state AAR

        ELA        = equil_ELA_annual[-11:-1,:]
        area       = equil_area_annual[-11:-1,:]
        volume     = equil_volume_annual[-11:-1:,:]
        volume_bsl = equil_volume_bsl_annual[-11:-1,:]
        
        area_2020       = equil_area_annual[0,:]
        volume_2020     = equil_volume_annual[0,:]
        volume_bsl_2020 = equil_volume_bsl_annual[0,:]
        
        bin_area         = equil_bin_area_annual[:,-11:-1,:]
        bin_thickness    = equil_bin_icethickness_annual[:,-11:-1,:]
        bin_massbalclim  = equil_bin_massbalclim_annual[:,-11:-1,:]
        surface_h        = bin_thickness + np.tile(bed_h[:, np.newaxis, :], (1, 10, 1))
        
        for i in range(sims):
            
            # use the glacier geometry from 2020
            nglacier = np.where(equil_bin_icethickness_annual[:,0,i]>0)[0]
            nbin_surface = equil_bin_icethickness_annual[nglacier,0,i] + bed_h[nglacier,i]

            ela_yearly = [];
            AAR_yearly = [];
            Ah = []
            At = []
            _ela = []
            for j in range(10):
                tot_area = np.nansum(bin_area[:,j,i])
                tot_area_above = 0;
                is_above = bin_massbalclim[:,j,i] > 0;
                tot_area_above = np.nansum(bin_area[:,j,i] * is_above)
                
                if tot_area != 0 and np.isnan(tot_area)==0:
                    _aar_yearly = tot_area_above/tot_area;
                    
                    if _aar_yearly >= 0.05 and _aar_yearly <= 0.95 and np.isnan(_aar_yearly)+np.isnan(ELA[j,i])==0:
                        ela_yearly = np.append(ela_yearly, ELA[j,i])
                        AAR_yearly = np.append(AAR_yearly, _aar_yearly)
                
                _Ah = surface_h[0,j,i];
                _At = surface_h[bin_thickness[:,j,i] > 0,j,i]
                if len(_At) != 0:
                    _At = _At[-1]
                    
                    if _Ah-_At != 0 and np.isnan(_Ah)+np.isnan(ELA[j,i])==0:
                        Ah = np.append(Ah, _Ah);
                        At = np.append(At, _At);
                        _ela = np.append(_ela, ELA[j,i])
                        
            if len(Ah) >= 5:
                equil_Ah[i] = Ah.mean()
                equil_At[i] = At.mean()
                equil_THAR[i] = (_ela.mean()-equil_At[i])/(equil_Ah[i]-equil_At[i])
                
                if nbin_surface[0] != nbin_surface[-1]:
                    equil_ELA_steady[i] = equil_THAR[i] * (nbin_surface[0] - nbin_surface[-1]) + nbin_surface[-1]
                
                if equil_THAR[i] < 0 or equil_THAR[i] > 1:
                    equil_THAR[i] = np.nan
                    equil_ELA_steady[i] = np.nan

            if len(AAR_yearly) >=5:
                equil_ELA_mean[i] = ela_yearly.mean()
                equil_AAR[i]        = AAR_yearly.mean()
                if equil_AAR[i] < 0.05 or equil_AAR[i] > 0.95:
                    equil_AAR[i] = np.nan;
            else:
                equil_AAR[i] = np.nan;
                
            equil_dA[i]       = np.nanmean(area[:,i])-area_2020[i];
            if area_2020[i] != 0:
                equil_a[i]        = equil_dA[i]/area_2020[i]+1;
                equil_AAR_mean[i] = equil_a[i] * equil_AAR[i];
                equil_dV[i]       = np.nanmean(volume[:,i])-volume_2020[i];
                equil_dV_bwl[i]   = np.nanmean(volume_bsl[:,i])-volume_bsl_2020[i];
                equil_dV_eff[i]   = equil_dV[i] - equil_dV_bwl[i];

            if equil_AAR_mean[i] < 0.05 or equil_AAR_mean[i] > 0.95:
                equil_AAR[i]      = np.nan;
                equil_AAR_mean[i] = np.nan;
        
    # ========================== output ==========================
    AAR_nan = np.where(np.isnan(equil_AAR));
    equil_ELA_mean[AAR_nan]   = np.nan;
    equil_AAR_mean[AAR_nan]   = np.nan;
    equil_AAR[AAR_nan]        = np.nan;
    equil_a[AAR_nan]          = np.nan;
    equil_dA[AAR_nan]         = np.nan;
    equil_dV[AAR_nan]         = np.nan;
    equil_ELA_steady[AAR_nan] = np.nan;
    equil_THAR[AAR_nan]       = np.nan;
    equil_At[AAR_nan]         = np.nan;
    equil_dV_bwl[AAR_nan]     = np.nan;
    equil_dV_eff[AAR_nan]     = np.nan;
    # equil_results
    # 0-equil_ELA_mean, 1-equil_AAR_mean, 2-equil_AAR, 3-equil_a, 4-equil_dA, 5-equil_dV, 6-equil_ELA_equil
    # 7-equil_THAR, 8-equil_At, 9-equil_dV_bwl, 10-equil_dV_eff
    equil_results[0,:]  = equil_ELA_mean;
    equil_results[1,:]  = equil_AAR_mean;
    equil_results[2,:]  = equil_AAR;
    equil_results[3,:]  = equil_a;
    equil_results[4,:]  = equil_dA;
    equil_results[5,:]  = equil_dV;
    equil_results[6,:]  = equil_ELA_steady;
    equil_results[7,:]  = equil_THAR;
    equil_results[8,:]  = equil_At;
    equil_results[9,:]  = equil_dV_bwl;
    equil_results[10,:] = equil_dV_eff;
    
    return equil_results

def create_xrdataset(glacier_rgi_table, dates_table, equil_nyears, option_wateryear=pygem_prms.gcm_wateryear):
    """
    Create empty xarray dataset that will be used to record simulation runs.

    Parameters
    ----------
    main_glac_rgi : pandas dataframe
        dataframe containing relevant rgi glacier information
    dates_table : pandas dataframe
        table of the dates, months, days in month, etc.

    Returns
    -------
    output_ds_all : xarray Dataset
        empty xarray dataset that contains variables and attributes to be filled in by simulation runs
    encoding : dictionary
        encoding used with exporting xarray dataset to netcdf
    """
    # Create empty datasets for each variable and merge them
    # Coordinate values
    glac_values = np.array([glacier_rgi_table.name])

    # Time attributes and values
    if option_wateryear == 'hydro':
        year_type = 'water year'
        annual_columns = np.unique(dates_table['wateryear'].values)[0:int(dates_table.shape[0]/12)]
    elif option_wateryear == 'calendar':
        year_type = 'calendar year'
        annual_columns = np.unique(dates_table['year'].values)[0:int(dates_table.shape[0]/12)]
    elif option_wateryear == 'custom':
        year_type = 'custom year'
       
    # append additional year to year_values to account for volume and area at end of period
    year_values = annual_columns[pygem_prms.gcm_spinupyears:annual_columns.shape[0]]
    year_values = np.concatenate((year_values, np.array([annual_columns[-1] + 1])))
    
    equil_year_values = np.arange(equil_nyears+1)
    
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
    
    # simulated glacier results in 2020
    output_coords_dict['area_2020']       = collections.OrderedDict([('glac', glac_values),
                                                                     ('sims', sims)])
    output_coords_dict['volume_2020']     = collections.OrderedDict([('glac', glac_values),
                                                                     ('sims', sims)])
    output_coords_dict['volume_bsl_2020'] = collections.OrderedDict([('glac', glac_values),
                                                                     ('sims', sims)])
    
    # results of parameterization approach and equilibrium simulaiton
    output_coords_dict['intercept_results'] =  collections.OrderedDict([('glac', glac_values), 
                                                                        ('intercept_dim', intercept_dim),
                                                                        ('sims', sims)])
    output_coords_dict['equil_results'] =  collections.OrderedDict([('glac', glac_values), 
                                                                    ('equil_dim', equil_dim),
                                                                    ('sims', sims)])
    
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
    # Annual changes
    output_coords_dict['equil_massbaltotal'] = collections.OrderedDict([('glac', glac_values), 
                                                                        ('equil_year', equil_year_values),
                                                                        ('dim', dim)])
    output_coords_dict['equil_area'] = collections.OrderedDict([('glac', glac_values), 
                                                                ('equil_year', equil_year_values),
                                                                ('dim', dim)])
    output_coords_dict['equil_volume'] = collections.OrderedDict([('glac', glac_values), 
                                                                  ('equil_year', equil_year_values),
                                                                  ('dim', dim)])
    output_coords_dict['equil_volume_bsl'] = collections.OrderedDict([('glac', glac_values), 
                                                                      ('equil_year', equil_year_values),
                                                                      ('dim', dim)])
    
    # Attributes dictionary
    output_attrs_dict = {
        'glac': {
                'long_name': 'glacier index',
                 'comment': 'glacier index referring to glaciers properties and model results'},
        'year': {
                'long_name': 'years',
                 'year_type': year_type,
                 'comment': 'years referring to the start of each year'},
        'equil_year': {
                'long_name': 'length of eqiulibrium experiment',
                 'year_type': year_type,
                 'comment': 'the equilibrium experiments will be run a 5000-yr time period'},
        'sims': {
                'long_name': 'simulation number',
                'comment': 'simulation number referring to the MCMC simulation; otherwise, only 1'},
        'dim': {
                'long_name': 'stats for a given variable',
                'comment': '0-mean, 1-std, 2-2.5%, 3-25%, 4-median, 5-75%, 6-97.5%, 7-mad'},
        'intercept_dim': {
                'long_name': 'AAR results by linear regression',
                'comment': '0-intercept_ELA_mean, 1-intercept_AAR_mean, 2-intercept_AAR, 3-intercept_a, 4-intercept_dA, 5-intercept_dV, \
                    6-intercept_ELA_steady, 7-intercept_THAR, 8-intercept_At, 9-intercept_dV_bwl, 10-intercept_dV_eff, \
                        11-intercept_n_regression, 12-intercept_slope, 13-intercept_r_value, 14-intercept_p_value, 15-intercept_std_err'},
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
        'area_2020': {
                'long_name': 'simulated glacier area in 2020',
                'units': 'm2'},
        'volume_2020': {
                'long_name': 'simulated glacier volume in 2020',
                'units': 'm3'},
        'volume_bsl_2020': {
                'long_name': 'simulated glacier volume below sea level in 2020',
                'units': 'm3'},
        'intercept_results': {
                'long_name': 'Accumulation area ratio results by linear regression',
                'comment': 'intercept_dim, dim'},
        'equil_results': {
                'long_name': 'Accumulation area ratio results through equilibrium experiment',
                'comment': 'random_dim, dim'},
        'output_climate': {
                'long_name': 'mean climate during 2014-2023',
                'comment': 'climate_dim, dim'},
        'equil_climate': {
                'long_name': 'mean climate over the last 10 years of the equilibrium simulation',
                'comment': 'climate_dim, dim'},
        'params': {
                'long_name': 'model parameters',
                'comment': 'params_dim, dim'},
        'equil_massbaltotal': {
                'long_name': 'glacier-wide total mass balance for the equilibrium simulation, in water equivalent',
                'units': 'm3',
                'temporal_resolution': 'annual'},
        'equil_area': {
                'long_name': 'glacier area for the equilibrium simulation',
                'units': 'm2',
                'temporal_resolution': 'annual'},
        'equil_volume': {
                'long_name': 'glacier volume for the equilibrium simulation',
                'units': 'm3',
                'temporal_resolution': 'annual'},
        'equil_volume_bsl': {
                'long_name': 'glacier volume below sea level for the equilibrium simulation',
                'units': 'm3',
                'temporal_resolution': 'annual'},
        }
       
    # Add variables to empty dataset and merge together
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
    output_ds_all['RGIId'].values = np.array([glacier_rgi_table.loc['RGIId']])
    output_ds_all['CenLon'].values = np.array([glacier_rgi_table.CenLon])
    output_ds_all['CenLat'].values = np.array([glacier_rgi_table.CenLat])
    output_ds_all['O1Region'].values = np.array([glacier_rgi_table.O1Region])
    output_ds_all['O2Region'].values = np.array([glacier_rgi_table.O2Region])
   
    output_ds_all.attrs = {'Source' : 'PyGEMv0.2.5 developed by David Rounce (drounce@alaska.edu)',
                           'Further developed by': 'Weilin Yang (weilinyang.yang@monash.edu)',
                           'Code reviewed by': 'Wenchao Chu (peterchuwenchao@foxmail.com)'}
       
    return output_ds_all, encoding



def main(list_packed_vars):
    """
    Model simulation
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels

    Returns
    -------
    netcdf files of the simulation output (specific output is dependent on the output option)
    """

    # Unpack variables
    glac_no = list_packed_vars[1]
    gcm_name = list_packed_vars[2]
    parser = getparser()
    args = parser.parse_args()
    if (gcm_name != pygem_prms.ref_gcm_name) and (args.scenario is None):
        scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
    elif not args.scenario is None:
        scenario = args.scenario
    
    # ===== LOAD GLACIERS =====
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)
    

    for glac in range(main_glac_rgi.shape[0]):
        if glac == 0:
            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        reg_str = str(glacier_rgi_table.O1Region).zfill(2)
        rgiid = main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId']

        try:

            # ===== Load glacier data: area (km2), ice thickness (m), width (km) =====
            if not glacier_rgi_table['TermType'] in [1,5] or pygem_prms.ignore_calving:
                gdir = single_flowline_glacier_directory(glacier_str, logging_level='CRITICAL')
                gdir.is_tidewater = False
                calving_k = None
                
                equil_gdir = single_flowline_glacier_directory(glacier_str, logging_level='CRITICAL')
                equil_gdir.is_tidewater = False
                
            else:
                gdir = single_flowline_glacier_directory_with_calving(glacier_str, logging_level='CRITICAL')
                gdir.is_tidewater = True
                
                equil_gdir = single_flowline_glacier_directory_with_calving(glacier_str, logging_level='CRITICAL')
                equil_gdir.is_tidewater = True
            
            # ===== TIME PERIOD =====
            startyear = gdir.rgi_date
            dates_table = modelsetup.datesmodelrun(
                    startyear=startyear, endyear=args.gcm_endyear, spinupyears=pygem_prms.gcm_spinupyears,
                    option_wateryear=pygem_prms.gcm_wateryear)
            
            # ===== LOAD CLIMATE DATA =====
            # Climate class
            if gcm_name in ['ERA5', 'ERA-Interim', 'COAWST']:
                gcm = class_climate.GCM(name=gcm_name)
                assert args.gcm_endyear <= int(time.strftime("%Y")), ('Climate data not available to ' + 
                                              str(args.gcm_endyear) + '. Change gcm_endyear or climate data set.')
            else:
                # GCM object
                gcm = class_climate.GCM(name=gcm_name, scenario=scenario)
                # Reference GCM
                ref_gcm = class_climate.GCM(name=pygem_prms.ref_gcm_name)
                # Adjust reference dates in event that reference is longer than GCM data
                if pygem_prms.ref_startyear >= args.gcm_startyear:
                    ref_startyear = pygem_prms.ref_startyear
                else:
                    ref_startyear = args.gcm_startyear
                if pygem_prms.ref_endyear <= args.gcm_endyear:
                    ref_endyear = pygem_prms.ref_endyear
                else:
                    ref_endyear = args.gcm_endyear
                dates_table_ref = modelsetup.datesmodelrun(startyear=ref_startyear, endyear=ref_endyear,
                                                           spinupyears=pygem_prms.ref_spinupyears,
                                                           option_wateryear=pygem_prms.ref_wateryear)
            
            # Air temperature [degC]
            gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi,
                                                                          dates_table)
            if pygem_prms.option_ablation != 2:
                gcm_tempstd = np.zeros(gcm_temp.shape)
            elif pygem_prms.option_ablation == 2 and gcm_name in ['ERA5']:
                gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn,
                                                                                main_glac_rgi, dates_table)
            elif pygem_prms.option_ablation == 2 and pygem_prms.ref_gcm_name in ['ERA5']:
                # Compute temp std based on reference climate data
                ref_tempstd, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.tempstd_fn, ref_gcm.tempstd_vn,
                                                                                    main_glac_rgi, dates_table_ref)
                # Monthly average from reference climate data
                gcm_tempstd = gcmbiasadj.monthly_avg_array_rolled(ref_tempstd, dates_table_ref, dates_table)
            else:
                gcm_tempstd = np.zeros(gcm_temp.shape)

            # Precipitation [m]
            gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi,
                                                                          dates_table)
            # Elevation [m asl]
            gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
            # Lapse rate [degC m-1]
            if pygem_prms.use_constant_lapserate:
                gcm_lr = np.zeros(gcm_temp.shape) + pygem_prms.lapserate
            else:
                if gcm_name in ['ERA-Interim', 'ERA5']:
                    gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
                else:
                    # Compute lapse rates based on reference climate data
                    ref_lr, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.lr_fn, ref_gcm.lr_vn, main_glac_rgi,
                                                                                    dates_table_ref)
                    # Monthly average from reference climate data
                    gcm_lr = gcmbiasadj.monthly_avg_array_rolled(ref_lr, dates_table_ref, dates_table)


            # ===== BIAS CORRECTIONS =====
            # No adjustments
            if pygem_prms.option_bias_adjustment == 0 or gcm_name == pygem_prms.ref_gcm_name:
                gcm_temp_adj = gcm_temp
                gcm_prec_adj = gcm_prec
                gcm_elev_adj = gcm_elev
            # Bias correct based on reference climate data
            else:
                # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
                ref_temp, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.temp_fn, ref_gcm.temp_vn,
                                                                                  main_glac_rgi, dates_table_ref)
                ref_prec, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.prec_fn, ref_gcm.prec_vn,
                                                                                  main_glac_rgi, dates_table_ref)
                ref_elev = ref_gcm.importGCMfxnearestneighbor_xarray(ref_gcm.elev_fn, ref_gcm.elev_vn, main_glac_rgi)
               
                # OPTION 1: Adjust temp using Huss and Hock (2015), prec similar but addresses for variance and outliers
                if pygem_prms.option_bias_adjustment == 1:
                    # Temperature bias correction
                    gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp,
                                                                                dates_table_ref, dates_table)
                    # Precipitation bias correction
                    gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_opt1(ref_prec, ref_elev, gcm_prec,
                                                                              dates_table_ref, dates_table)
                # OPTION 2: Adjust temp and prec using Huss and Hock (2015)
                elif pygem_prms.option_bias_adjustment == 2:
                    # Temperature bias correction
                    gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp,
                                                                                dates_table_ref, dates_table)
                    # Precipitation bias correction
                    gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_HH2015(ref_prec, ref_elev, gcm_prec,
                                                                                dates_table_ref, dates_table)
                    
            # ===== RUN MASS BALANCE =====
            # Number of simulations
            if pygem_prms.option_calibration == 'MCMC':
                sim_iters = pygem_prms.sim_iters
            else:
                sim_iters = 1
           
            # Number of years (for OGGM's run_until_and_store)
            if pygem_prms.timestep == 'monthly':
                nyears = int(dates_table.shape[0]/12)
            else:
                assert True==False, 'Adjust nyears for non-monthly timestep'

            # Flowliness
            fls = gdir.read_pickle('inversion_flowlines')

            # Add climate data to glacier directory
            if pygem_prms.hindcast == True:
                gcm_temp_adj = gcm_temp_adj[::-1]
                gcm_tempstd = gcm_tempstd[::-1]
                gcm_prec_adj= gcm_prec_adj[::-1]
                gcm_lr = gcm_lr[::-1]
            
            gdir.historical_climate = {'elev': gcm_elev_adj[glac],
                                       'temp': gcm_temp_adj[glac,:],
                                       'tempstd': gcm_tempstd[glac,:],
                                       'prec': gcm_prec_adj[glac,:],
                                       'lr': gcm_lr[glac,:]}            
            
            gdir.dates_table = dates_table
           
            glacier_area_km2 = fls[0].widths_m * fls[0].dx_meter / 1e6

            
            # ==========================================================================================================
            # Add climate data for equilibrium simulation #####################
            length_exp = pygem_prms.length_exp
            equil_gdir.historical_climate = {'elev': gcm_elev_adj[glac],
                                             'temp': gcm_temp_adj[glac,:],
                                             'tempstd': gcm_tempstd[glac,:],
                                             'prec': gcm_prec_adj[glac,:],
                                             'lr': gcm_lr[glac,:]}
            
            # climate data
            equil_year = np.arange(0,length_exp,1)
            equil_month = np.arange(1,13,1)
            equil_month = np.tile(equil_month,len(equil_year))
            equil_year = equil_year.repeat(12)
            equil_daysinmonth = np.zeros(len(equil_year), dtype=int)
            equil_date = ['' for _ in range(len(equil_year))]
            for i in range(0,len(equil_year)):
                if equil_month[i] < 10:
                    equil_date[i] = str(equil_year[i]) + '-0' + str(equil_month[i])+'-01'
                else:
                    equil_date[i] = str(equil_year[i]) + '-' + str(equil_month[i])+'-01'
                equil_daysinmonth[i] = calendar.monthrange(equil_year[i],equil_month[i])[1]
            
            equil_dates_table = pd.DataFrame({'date': equil_date})
            equil_dates_table['year'] = equil_year
            equil_dates_table['month'] = equil_month
            equil_dates_table['daysinmonth'] = equil_daysinmonth
            equil_dates_table['timestep'] = np.arange(len(equil_dates_table['date']))
            # Set date as index
            equil_dates_table.set_index('timestep', inplace=True)
            
            if pygem_prms.option_leapyear == 0:
                equil_mask1 = (equil_dates_table['daysinmonth'] == 29)
                equil_dates_table.loc[equil_mask1,'daysinmonth'] = 28
            
            equil_dates_table['wateryear'] = equil_dates_table['year']
            for step in range(equil_dates_table.shape[0]):
                if equil_dates_table.loc[step, 'month'] >= 10:
                    equil_dates_table.loc[step, 'wateryear'] = equil_dates_table.loc[step, 'year'] + 1
                    # Add column for seasons
                    # create a season dictionary to assist groupby functions
            equil_seasondict = {}
            equil_month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            equil_season_list = []
            for i in range(len(equil_month_list)):
                if (equil_month_list[i] >= pygem_prms.summer_month_start and equil_month_list[i] < pygem_prms.winter_month_start):
                    equil_season_list.append('summer')
                    equil_seasondict[equil_month_list[i]] = equil_season_list[i]
                else:
                    equil_season_list.append('winter')
                    equil_seasondict[equil_month_list[i]] = equil_season_list[i]
            equil_dates_table['season'] = equil_dates_table['month'].apply(lambda x: equil_seasondict[x])
            
            equil_gdir.dates_table = equil_dates_table
            equil_nyears = int(equil_dates_table.shape[0]/12)
            
            # Random climate ##################################################
            shuffled_yr = pd.read_csv(pygem_prms.main_directory+'/shuffled_year.csv', index_col=0)
            shuffled_yr = shuffled_yr['2014-2023'].values[:length_exp]
            equil_temp = np.zeros(len(equil_year))
            equil_tempstd = np.zeros(len(equil_year))
            equil_prec = np.zeros(len(equil_year))
            equil_lr = np.zeros(len(equil_year))
            
            for i in range(0,equil_nyears):
                equil_temp[12*i:12*(i+1)] = equil_gdir.historical_climate['temp'][12*(shuffled_yr[i]-
                                                       args.gcm_startyear):12*(shuffled_yr[i]-args.gcm_startyear+1)];
                equil_tempstd[12*i:12*(i+1)] = equil_gdir.historical_climate['tempstd'][12*(shuffled_yr[i]-
                                                          args.gcm_startyear):12*(shuffled_yr[i]-args.gcm_startyear+1)];
                equil_prec[12*i:12*(i+1)] = equil_gdir.historical_climate['prec'][12*(shuffled_yr[i]-
                                                       args.gcm_startyear):12*(shuffled_yr[i]-args.gcm_startyear+1)];
                equil_lr[12*i:12*(i+1)] = equil_gdir.historical_climate['lr'][12*(shuffled_yr[i]-
                                                     args.gcm_startyear):12*(shuffled_yr[i]-args.gcm_startyear+1)];
            
            equil_gdir.historical_climate['temp'] = equil_temp
            equil_gdir.historical_climate['tempstd'] = equil_tempstd
            equil_gdir.historical_climate['prec'] = equil_prec
            equil_gdir.historical_climate['lr'] = equil_lr
            
            
            # model parameters ===================================================================================================================
            if (fls is not None) and (glacier_area_km2.sum() > 0):
                
    
                # Load model parameters
                if pygem_prms.use_calibrated_modelparams:
                    
                    modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                    modelprms_fp = (pygem_prms.output_filepath + 'calibration/' + glacier_str.split('.')[0].zfill(2) 
                                    + '/')
                    modelprms_fullfn = modelprms_fp + modelprms_fn
    
                    assert os.path.exists(modelprms_fullfn), 'Calibrated parameters do not exist.'
                    with open(modelprms_fullfn, 'rb') as f:
                        modelprms_dict = pickle.load(f)
    
                    assert pygem_prms.option_calibration in modelprms_dict, ('Error: ' + pygem_prms.option_calibration +
                                                                              ' not in modelprms_dict')
                    modelprms_all = modelprms_dict[pygem_prms.option_calibration]
                    # MCMC needs model parameters to be selected
                    if pygem_prms.option_calibration == 'MCMC':
                        sim_iters = pygem_prms.sim_iters
                        if sim_iters == 1:
                            modelprms_all = {'kp': [np.median(modelprms_all['kp']['chain_0'])],
                                              'tbias': [np.median(modelprms_all['tbias']['chain_0'])],
                                              'ddfsnow': [np.median(modelprms_all['ddfsnow']['chain_0'])],
                                              'ddfice': [np.median(modelprms_all['ddfice']['chain_0'])],
                                              'tsnow_threshold': modelprms_all['tsnow_threshold'],
                                              'precgrad': modelprms_all['precgrad']}
                        else:
                            # Select every kth iteration to use for the ensemble
                            mcmc_sample_no = len(modelprms_all['kp']['chain_0'])
                            mp_spacing = int((mcmc_sample_no - pygem_prms.sim_burn) / sim_iters)
                            mp_idx_start = np.arange(pygem_prms.sim_burn, pygem_prms.sim_burn + mp_spacing)
                            np.random.shuffle(mp_idx_start)
                            mp_idx_start = mp_idx_start[0]
                            mp_idx_all = np.arange(mp_idx_start, mcmc_sample_no, mp_spacing)
                            modelprms_all = {
                                    'kp': [modelprms_all['kp']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
                                    'tbias': [modelprms_all['tbias']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
                                    'ddfsnow': [modelprms_all['ddfsnow']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
                                    'ddfice': [modelprms_all['ddfice']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
                                    'tsnow_threshold': modelprms_all['tsnow_threshold'] * sim_iters,
                                    'precgrad': modelprms_all['precgrad'] * sim_iters}
                    else:
                        sim_iters = 1
                        
                    # Calving parameter
                    if not glacier_rgi_table['TermType'] in [1,5] or pygem_prms.ignore_calving:
                        calving_k = None
                    else:
                        # Load quality controlled frontal ablation data 
                        assert os.path.exists(pygem_prms.calving_fp + pygem_prms.calving_fn), 'Calibrated calving dataset does not exist'
                        calving_df = pd.read_csv(pygem_prms.calving_fp + pygem_prms.calving_fn)
                        calving_rgiids = list(calving_df.RGIId)
                        
                        # Use calibrated value if individual data available
                        if rgiid in calving_rgiids:
                            calving_idx = calving_rgiids.index(rgiid)
                            calving_k = calving_df.loc[calving_idx, 'calving_k']
                            calving_k_nmad = calving_df.loc[calving_idx, 'calving_k_nmad']
                        # Otherwise, use region's median value
                        else:
                            calving_df['O1Region'] = [int(x.split('-')[1].split('.')[0]) for x in calving_df.RGIId.values]
                            calving_df_reg = calving_df.loc[calving_df['O1Region'] == int(reg_str), :]
                            calving_k = np.median(calving_df_reg.calving_k)
                            calving_k_nmad = 0
                        
                        if sim_iters == 1:
                            calving_k_values = np.array([calving_k])
                        else:
                            calving_k_values = calving_k + np.random.normal(loc=0, scale=calving_k_nmad, size=sim_iters)
                            calving_k_values[calving_k_values < 0.001] = 0.001
                            calving_k_values[calving_k_values > 5] = 5
                            
                            
                            while not abs(np.median(calving_k_values) - calving_k) < 0.001:
                                calving_k_values = calving_k + np.random.normal(loc=0, scale=calving_k_nmad, size=sim_iters)
                                calving_k_values[calving_k_values < 0.001] = 0.001
                                calving_k_values[calving_k_values > 5] = 5
                                
                            
                            assert abs(np.median(calving_k_values) - calving_k) < 0.001, 'calving_k distribution too far off'

                        if debug:                        
                            print('calving_k_values:', np.mean(calving_k_values), np.std(calving_k_values), '\n', calving_k_values)

                        

                else:
                    modelprms_all = {'kp': [pygem_prms.kp],
                                     'tbias': [pygem_prms.tbias],
                                     'ddfsnow': [pygem_prms.ddfsnow],
                                     'ddfice': [pygem_prms.ddfice],
                                     'tsnow_threshold': [pygem_prms.tsnow_threshold],
                                     'precgrad': [pygem_prms.precgrad]}
                    calving_k = np.zeros(sim_iters) + pygem_prms.calving_k
                    
                if debug and gdir.is_tidewater:
                    print('calving_k:', calving_k)
                    

                # Load OGGM glacier dynamics parameters (if necessary)
                if pygem_prms.option_dynamics in ['OGGM', 'MassRedistributionCurves']:

                    # CFL number (may use different values for calving to prevent errors)
                    if not glacier_rgi_table['TermType'] in [1,5] or pygem_prms.ignore_calving:
                        cfg.PARAMS['cfl_number'] = pygem_prms.cfl_number
                    else:
                        cfg.PARAMS['cfl_number'] = pygem_prms.cfl_number_calving

                    
                    if debug:
                        print('cfl number:', cfg.PARAMS['cfl_number'])
                        
                    if pygem_prms.use_reg_glena:
                        glena_df = pd.read_csv(pygem_prms.glena_reg_fullfn)                    
                        glena_O1regions = [int(x) for x in glena_df.O1Region.values]
                        assert glacier_rgi_table.O1Region in glena_O1regions, glacier_str + ' O1 region not in glena_df'
                        glena_idx = np.where(glena_O1regions == glacier_rgi_table.O1Region)[0][0]
                        glen_a_multiplier = glena_df.loc[glena_idx,'glens_a_multiplier']
                        fs = glena_df.loc[glena_idx,'fs']
                    else:
                        fs = pygem_prms.fs
                        glen_a_multiplier = pygem_prms.glen_a_multiplier
    
    
                # Time attributes and values
                if pygem_prms.gcm_wateryear == 'hydro':
                    annual_columns = np.unique(dates_table['wateryear'].values)[0:int(dates_table.shape[0]/12)]
                    
                    equil_annual_columns = np.unique(equil_dates_table['wateryear'].values)[0:int(equil_dates_table.shape[0]/12)]
                    
                else:
                    annual_columns = np.unique(dates_table['year'].values)[0:int(dates_table.shape[0]/12)]
                    
                    equil_annual_columns = np.unique(equil_dates_table['year'].values)[0:int(equil_dates_table.shape[0]/12)]
                    
                # append additional year to year_values to account for volume and area at end of period
                year_values = annual_columns[pygem_prms.gcm_spinupyears:annual_columns.shape[0]]
                year_values = np.concatenate((year_values, np.array([annual_columns[-1] + 1])))
                output_glac_temp_monthly                 = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_prec_monthly                 = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_acc_monthly                  = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_refreeze_monthly             = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_melt_monthly                 = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_frontalablation_monthly      = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_massbaltotal_monthly         = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_area_annual                  = np.zeros((year_values.shape[0], sim_iters)) * np.nan
                output_glac_volume_annual                = np.zeros((year_values.shape[0], sim_iters)) * np.nan
                output_glac_volume_bsl_annual            = np.zeros((year_values.shape[0], sim_iters)) * np.nan
                output_glac_volume_change_ignored_annual = np.zeros((year_values.shape[0], sim_iters))
                output_glac_ELA_annual                   = np.zeros((year_values.shape[0], sim_iters)) * np.nan
                output_glac_bin_icethickness_annual      = None
                output_glac_bin_bed_h                    = None
                
                equil_year_values = equil_annual_columns[0:equil_annual_columns.shape[0]]
                equil_year_values = np.concatenate((equil_year_values, np.array([equil_annual_columns[-1] +1 ])))
                equil_glac_temp_monthly                 = np.zeros((equil_dates_table.shape[0], sim_iters)) * np.nan
                equil_glac_prec_monthly                 = np.zeros((equil_dates_table.shape[0], sim_iters)) * np.nan
                equil_glac_acc_monthly                  = np.zeros((equil_dates_table.shape[0], sim_iters)) * np.nan
                equil_glac_refreeze_monthly             = np.zeros((equil_dates_table.shape[0], sim_iters)) * np.nan
                equil_glac_melt_monthly                 = np.zeros((equil_dates_table.shape[0], sim_iters)) * np.nan
                equil_glac_frontalablation_monthly      = np.zeros((equil_dates_table.shape[0], sim_iters)) * np.nan
                equil_glac_massbaltotal_monthly         = np.zeros((equil_dates_table.shape[0], sim_iters)) * np.nan
                equil_glac_area_annual                  = np.zeros((equil_year_values.shape[0], sim_iters)) * np.nan
                equil_glac_volume_annual                = np.zeros((equil_year_values.shape[0], sim_iters)) * np.nan
                equil_glac_volume_bsl_annual            = np.zeros((equil_year_values.shape[0], sim_iters)) * np.nan
                equil_glac_volume_change_ignored_annual = np.zeros((equil_year_values.shape[0], sim_iters))
                equil_glac_ELA_annual                   = np.zeros((equil_year_values.shape[0], sim_iters)) * np.nan
                equil_glac_bin_icethickness_annual      = None
                
                water_level_all = np.zeros(sim_iters) * np.nan
               
                # Loop through model parameters
                count_exceed_boundary_errors = 0
                equil_count_exceed_boundary_errors = 0
                

                for n_iter in range(sim_iters):
                    
                    if debug:                    
                        print('n_iter:', n_iter)
                    
                    if not calving_k is None:
                        calving_k = calving_k_values[n_iter]
                        cfg.PARAMS['calving_k'] = calving_k
                        cfg.PARAMS['inversion_calving_k'] = calving_k
                    
                    # successful_run used to continue runs when catching specific errors
                    successful_run = True
                    
                    modelprms = {'kp': modelprms_all['kp'][n_iter],
                                  'tbias': modelprms_all['tbias'][n_iter],
                                  'ddfsnow': modelprms_all['ddfsnow'][n_iter],
                                  'ddfice': modelprms_all['ddfice'][n_iter],
                                  'tsnow_threshold': modelprms_all['tsnow_threshold'][n_iter],
                                  'precgrad': modelprms_all['precgrad'][n_iter]}
    
                    if debug:
                        print(glacier_str + '  kp: ' + str(np.round(modelprms['kp'],2)) +
                              ' ddfsnow: ' + str(np.round(modelprms['ddfsnow'],4)) +
                              ' tbias: ' + str(np.round(modelprms['tbias'],2)))
                    
                    # ----- ICE THICKNESS INVERSION using OGGM -----
                    if not pygem_prms.option_dynamics is None:
                        # Apply inversion_filter on mass balance with debris to avoid negative flux
                        if pygem_prms.include_debris:
                            inversion_filter = True
                        else:
                            inversion_filter = False
                              
                        # Perform inversion based on PyGEM MB
                        mbmod_inv = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                                     hindcast=pygem_prms.hindcast,
                                                     debug=pygem_prms.debug_mb,
                                                     debug_refreeze=pygem_prms.debug_refreeze,
                                                     fls=fls, option_areaconstant=True,
                                                     inversion_filter=inversion_filter)

                        # Non-tidewater glaciers
                        if not gdir.is_tidewater or pygem_prms.ignore_calving:
                            # Arbitrariliy shift the MB profile up (or down) until mass balance is zero (equilibrium for inversion)
                            climate.apparent_mb_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=np.arange(nyears))
                            tasks.prepare_for_inversion(gdir)
                            tasks.mass_conservation_inversion(gdir, glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs)
            #                tasks.filter_inversion_output(gdir)
                        
                        # Tidewater glaciers
                        else:
                            out_calving = find_inversion_calving_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=np.arange(nyears),
                                                                             glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs)
                                
                        # ----- INDENTED TO BE JUST WITH DYNAMICS -----
                        tasks.init_present_time_glacier(gdir) # adds bins below
                        debris.debris_binned(gdir, fl_str='model_flowlines')  # add debris enhancement factors to flowlines
        
                        try:
                            nfls = gdir.read_pickle('model_flowlines')
                        except FileNotFoundError as e:
                            if 'model_flowlines.pkl' in str(e):
                                tasks.compute_downstream_line(gdir)
                                tasks.compute_downstream_bedshape(gdir)
                                tasks.init_present_time_glacier(gdir) # adds bins below
                                nfls = gdir.read_pickle('model_flowlines')
                            else:
                                raise
                    
                    # No ice dynamics options
                    else:
                        nfls = fls
                    
                    # Water Level
                    # Check that water level is within given bounds
                    cls = gdir.read_pickle('inversion_input')[-1]
                    th = cls['hgt'][-1]
                    vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
                    water_level = utils.clip_scalar(0, th - vmax, th - vmin) 
                    
                    water_level_all[n_iter] = water_level
                    
                    if output_glac_bin_bed_h is None:
                        output_glac_bin_bed_h = (nfls[0].bed_h)[:,np.newaxis]
                    else:
                        output_glac_bin_bed_h = np.append(output_glac_bin_bed_h, 
                                                          (nfls[0].bed_h)[:,np.newaxis],
                                                          axis=1)
                    
                    # ------ MODEL WITH EVOLVING AREA ------
                    # Mass balance model
                    mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                              hindcast=pygem_prms.hindcast,
                                              debug=pygem_prms.debug_mb,
                                              debug_refreeze=pygem_prms.debug_refreeze,
                                              fls=nfls, option_areaconstant=True)

                    # Glacier dynamics model
                    if pygem_prms.option_dynamics == 'OGGM':
                        if debug:
                            print('OGGM GLACIER DYNAMICS!')
                            
                        ev_model = FluxBasedModel(nfls, y0=0, mb_model=mbmod, do_kcalving=True,
                                                  glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                                  is_tidewater=gdir.is_tidewater,
                                                  water_level=water_level
                                                  )
                        
                        # to spinup the equilibrium simulations
                        ev_model_spinup = FluxBasedModel(nfls, y0=0, mb_model=mbmod, do_kcalving=True,
                                                         glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                                         is_tidewater=gdir.is_tidewater,
                                                         water_level=water_level
                                                         )
                        
                        if debug:
                            graphics.plot_modeloutput_section(ev_model)
                            plt.show()

                        try:
                            if Version(oggm_version) < Version('1.5.3'):
                                _, diag = ev_model.run_until_and_store(nyears)
                                ev_model_spinup.run_until_and_store(nyears-4);
                            else:
                                diag = ev_model.run_until_and_store(nyears)
                                ev_model_spinup.run_until_and_store(nyears-4);
                            ev_model.mb_model.glac_wide_volume_annual[-1] = diag.volume_m3[-1]
                            ev_model.mb_model.glac_wide_area_annual[-1] = diag.area_m2[-1]
                            
                            # Record frontal ablation for tidewater glaciers and update total mass balance
                            if gdir.is_tidewater:
                                # Glacier-wide frontal ablation (m3 w.e.)
                                # - note: diag.calving_m3 is cumulative calving
                                if debug:
                                    print('\n\ndiag.calving_m3:', diag.calving_m3.values)
                                    print('calving_m3_since_y0:', ev_model.calving_m3_since_y0)
                                calving_m3_annual = ((diag.calving_m3.values[1:] - diag.calving_m3.values[0:-1]) * 
                                                     pygem_prms.density_ice / pygem_prms.density_water)
                                for n in np.arange(calving_m3_annual.shape[0]):
                                    ev_model.mb_model.glac_wide_frontalablation[12*n+11] = calving_m3_annual[n]

                                # Glacier-wide total mass balance (m3 w.e.)
                                ev_model.mb_model.glac_wide_massbaltotal = (
                                        ev_model.mb_model.glac_wide_massbaltotal  - ev_model.mb_model.glac_wide_frontalablation)
                                
                                if debug:
                                    print('avg calving_m3:', calving_m3_annual.sum() / nyears)
                                    print('avg frontal ablation [Gta]:', 
                                          np.round(ev_model.mb_model.glac_wide_frontalablation.sum() / 1e9 / nyears,4))
                                    print('avg frontal ablation [Gta]:', 
                                          np.round(ev_model.calving_m3_since_y0 * pygem_prms.density_ice / 1e12 / nyears,4))
                            
                        except RuntimeError as e:
                            if 'Glacier exceeds domain boundaries' in repr(e):
                                count_exceed_boundary_errors += 1
                                successful_run = False
                                
                                # LOG FAILURE
                                fail_domain_fp = (pygem_prms.output_sim_fp + 'fail-exceed_domain/' + reg_str + '/' 
                                                  + gcm_name + '/')
                                if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
                                    fail_domain_fp += scenario + '/'
                                if not os.path.exists(fail_domain_fp):
                                    os.makedirs(fail_domain_fp, exist_ok=True)
                                txt_fn_fail = glacier_str + "-sim_failed.txt"
                                with open(fail_domain_fp + txt_fn_fail, "w") as text_file:
                                    text_file.write(glacier_str + ' failed to complete ' + 
                                                    str(count_exceed_boundary_errors) + ' simulations')
                            elif gdir.is_tidewater:
                                if debug:
                                    print('OGGM dynamics failed, using mass redistribution curves')
                                # Mass redistribution curves glacier dynamics model
                                ev_model = MassRedistributionCurveModel(
                                                nfls, mb_model=mbmod, y0=0,
                                                glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                                is_tidewater=gdir.is_tidewater,
                                                water_level=water_level
                                                )
                                if Version(oggm_version) < Version('1.5.3'):
                                    _, diag = ev_model.run_until_and_store(nyears)
                                else:
                                    diag = ev_model.run_until_and_store(nyears)
                                diag = diag[-1]
                                ev_model.mb_model.glac_wide_volume_annual = diag.volume_m3.values
                                ev_model.mb_model.glac_wide_area_annual = diag.area_m2.values
                
                                # Record frontal ablation for tidewater glaciers and update total mass balance
                                # Update glacier-wide frontal ablation (m3 w.e.)
                                ev_model.mb_model.glac_wide_frontalablation = ev_model.mb_model.glac_bin_frontalablation.sum(0)
                                # Update glacier-wide total mass balance (m3 w.e.)
                                ev_model.mb_model.glac_wide_massbaltotal = (
                                        ev_model.mb_model.glac_wide_massbaltotal - ev_model.mb_model.glac_wide_frontalablation)

                                if debug:
                                    print('avg frontal ablation [Gta]:', 
                                          np.round(ev_model.mb_model.glac_wide_frontalablation.sum() / 1e9 / nyears,4))
                                    print('avg frontal ablation [Gta]:', 
                                          np.round(ev_model.calving_m3_since_y0 * pygem_prms.density_ice / 1e12 / nyears,4))

                        except:
                            if gdir.is_tidewater:
                                if debug:
                                    print('OGGM dynamics failed, using mass redistribution curves')
                                                                # Mass redistribution curves glacier dynamics model
                                ev_model = MassRedistributionCurveModel(
                                                nfls, mb_model=mbmod, y0=0,
                                                glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                                is_tidewater=gdir.is_tidewater,
                                                water_level=water_level
                                                )
                                
                                ev_model_spinup = MassRedistributionCurveModel(
                                                    nfls, mb_model=mbmod, y0=0,
                                                    glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                                    is_tidewater=gdir.is_tidewater,
                                                    water_level=water_level
                                                    )
                                
                                if Version(oggm_version) < Version('1.5.3'):
                                    _, diag = ev_model.run_until_and_store(nyears)
                                    ev_model_spinup.run_until_and_store(nyears-4);
                                else:
                                    diag = ev_model.run_until_and_store(nyears)
                                    ev_model_spinup.run_until_and_store(nyears-4);
                                diag = diag[-1]
                                ev_model.mb_model.glac_wide_volume_annual = diag.volume_m3.values
                                ev_model.mb_model.glac_wide_area_annual = diag.area_m2.values
                
                                # Record frontal ablation for tidewater glaciers and update total mass balance
                                # Update glacier-wide frontal ablation (m3 w.e.)
                                ev_model.mb_model.glac_wide_frontalablation = ev_model.mb_model.glac_bin_frontalablation.sum(0)
                                # Update glacier-wide total mass balance (m3 w.e.)
                                ev_model.mb_model.glac_wide_massbaltotal = (
                                        ev_model.mb_model.glac_wide_massbaltotal - ev_model.mb_model.glac_wide_frontalablation)

                                if debug:
                                    print('avg frontal ablation [Gta]:', 
                                          np.round(ev_model.mb_model.glac_wide_frontalablation.sum() / 1e9 / nyears,4))
                                    print('avg frontal ablation [Gta]:', 
                                          np.round(ev_model.calving_m3_since_y0 * pygem_prms.density_ice / 1e12 / nyears,4))
                
                            else:
                                raise
                        
                    # Record output for successful runs
                    if successful_run:
                        
                        if not pygem_prms.option_dynamics is None:
                            if debug:
                                graphics.plot_modeloutput_section(ev_model)
            #                    graphics.plot_modeloutput_map(gdir, model=ev_model)
                                plt.figure()
                                diag.volume_m3.plot()
                                plt.figure()
    #                                diag.area_m2.plot()
                                plt.show()
            
                            # Post-process data to ensure mass is conserved and update accordingly for ignored mass losses
                            #  ignored mass losses occur because mass balance model does not know ice thickness and flux divergence
                            area_initial = mbmod.glac_bin_area_annual[:,0].sum()
                            mb_mwea_diag = ((diag.volume_m3.values[-1] - diag.volume_m3.values[0]) 
                                            / area_initial / nyears * pygem_prms.density_ice / pygem_prms.density_water)
                            mb_mwea_mbmod = mbmod.glac_wide_massbaltotal.sum() / area_initial / nyears
                           
                            if debug:
                                vol_change_diag = diag.volume_m3.values[-1] - diag.volume_m3.values[0]
                                print('  vol init  [Gt]:', np.round(diag.volume_m3.values[0] * 0.9 / 1e9,5))
                                print('  vol final [Gt]:', np.round(diag.volume_m3.values[-1] * 0.9 / 1e9,5))
                                print('  vol change[Gt]:', np.round(vol_change_diag * 0.9 / 1e9,5))
                                print('  mb [mwea]:', np.round(mb_mwea_diag,2))
                                print('  mb_mbmod [mwea]:', np.round(mb_mwea_mbmod,2))
                            
                            
                            if np.abs(mb_mwea_diag - mb_mwea_mbmod) > 1e-6:
                                ev_model.mb_model.ensure_mass_conservation(diag)
                                 
                        if debug:
                            print('mass loss [Gt]:', mbmod.glac_wide_massbaltotal.sum() / 1e9)
        
                        # RECORD PARAMETERS TO DATASET
                        output_glac_temp_monthly[:, n_iter] = mbmod.glac_wide_temp
                        output_glac_prec_monthly[:, n_iter] = mbmod.glac_wide_prec
                        output_glac_acc_monthly[:, n_iter] = mbmod.glac_wide_acc
                        output_glac_refreeze_monthly[:, n_iter] = mbmod.glac_wide_refreeze
                        output_glac_melt_monthly[:, n_iter] = mbmod.glac_wide_melt
                        output_glac_frontalablation_monthly[:, n_iter] = mbmod.glac_wide_frontalablation
                        output_glac_massbaltotal_monthly[:, n_iter] = mbmod.glac_wide_massbaltotal
                        output_glac_area_annual[:, n_iter] = diag.area_m2.values
                        output_glac_volume_annual[:, n_iter] = diag.volume_m3.values
                        output_glac_volume_bsl_annual[:, n_iter] = diag.volume_bsl_m3.values
                        output_glac_volume_change_ignored_annual[:-1, n_iter] = mbmod.glac_wide_volume_change_ignored_annual
                        output_glac_ELA_annual[:, n_iter] = mbmod.glac_wide_ELA_annual
                        
                        if output_glac_bin_icethickness_annual is None:
                            output_glac_bin_area_annual_sim = (mbmod.glac_bin_area_annual)[:,:,np.newaxis]
                            
                            output_glac_bin_icethickness_annual_sim = (mbmod.glac_bin_icethickness_annual)[:,:,np.newaxis]
                            # Update the latest thickness and volume
                            if ev_model is not None:
                                fl_dx_meter = getattr(ev_model.fls[0], 'dx_meter', None)
                                fl_widths_m = getattr(ev_model.fls[0], 'widths_m', None)
                                fl_section = getattr(ev_model.fls[0],'section',None)
                            else:
                                fl_dx_meter = getattr(nfls[0], 'dx_meter', None)
                                fl_widths_m = getattr(nfls[0], 'widths_m', None)
                                fl_section = getattr(nfls[0],'section',None)
                            
                            if fl_section is not None and fl_widths_m is not None:
                                # thickness
                                icethickness_t0 = np.zeros(fl_section.shape)
                                icethickness_t0[fl_widths_m > 0] = fl_section[fl_widths_m > 0] / fl_widths_m[fl_widths_m > 0]
                                output_glac_bin_icethickness_annual_sim[:,-1,0] = icethickness_t0
                                # area
                                glacier_area_t0 = fl_widths_m * fl_dx_meter
                                output_glac_bin_area_annual_sim[:,-1,0] = glacier_area_t0
                                
                            output_glac_bin_area_annual = output_glac_bin_area_annual_sim
                            output_glac_bin_icethickness_annual = output_glac_bin_icethickness_annual_sim
                            output_glac_bin_massbalclim_annual_sim = np.zeros(mbmod.glac_bin_icethickness_annual.shape)
                            output_glac_bin_massbalclim_annual_sim[:,:-1] =  mbmod.glac_bin_massbalclim_annual
                            output_glac_bin_massbalclim_annual = output_glac_bin_massbalclim_annual_sim[:,:,np.newaxis]
                            
                        else:
                            # Update the latest thickness and area
                            output_glac_bin_area_annual_sim = (mbmod.glac_bin_area_annual)[:,:,np.newaxis]
                            output_glac_bin_icethickness_annual_sim = (mbmod.glac_bin_icethickness_annual)[:,:,np.newaxis]
                            if ev_model is not None:
                                fl_dx_meter = getattr(ev_model.fls[0], 'dx_meter', None)
                                fl_widths_m = getattr(ev_model.fls[0], 'widths_m', None)
                                fl_section = getattr(ev_model.fls[0],'section',None)
                            else:
                                fl_dx_meter = getattr(nfls[0], 'dx_meter', None)
                                fl_widths_m = getattr(nfls[0], 'widths_m', None)
                                fl_section = getattr(nfls[0],'section',None)
                            if fl_section is not None and fl_widths_m is not None:
                                # thickness
                                icethickness_t0 = np.zeros(fl_section.shape)
                                icethickness_t0[fl_widths_m > 0] = fl_section[fl_widths_m > 0] / fl_widths_m[fl_widths_m > 0]
                                output_glac_bin_icethickness_annual_sim[:,-1,0] = icethickness_t0
                                # area
                                glacier_area_t0 = fl_widths_m * fl_dx_meter
                                output_glac_bin_area_annual_sim[:,-1,0] = glacier_area_t0
                            output_glac_bin_area_annual = np.append(output_glac_bin_area_annual,
                                                                    output_glac_bin_area_annual_sim, axis=2)
                            output_glac_bin_icethickness_annual = np.append(output_glac_bin_icethickness_annual, 
                                                                            output_glac_bin_icethickness_annual_sim,
                                                                            axis=2)
                            output_glac_bin_massbalclim_annual_sim = np.zeros(mbmod.glac_bin_icethickness_annual.shape)
                            output_glac_bin_massbalclim_annual_sim[:,:-1] =  mbmod.glac_bin_massbalclim_annual
                            output_glac_bin_massbalclim_annual = np.append(output_glac_bin_massbalclim_annual, 
                                                                           output_glac_bin_massbalclim_annual_sim[:,:,np.newaxis],
                                                                           axis=2)
                   
                        # Equilibrium simulation ##############################################################################################
                        # successful_run used to continue runs when catching specific errors
                        equil_successful_run = True
                        equil_nfls = ev_model_spinup.fls
                        
                        # ------ MODEL WITH EVOLVING AREA ------
                        # Mass balance model
                        equil_mbmod = PyGEMMassBalance(equil_gdir, modelprms, glacier_rgi_table,
                                                       hindcast=pygem_prms.hindcast,
                                                       debug=pygem_prms.debug_mb,
                                                       debug_refreeze=pygem_prms.debug_refreeze,
                                                       fls=equil_nfls, option_areaconstant=True)
                        
                        # option_dynamics == 'OGGM'
                        equil_ev_model = FluxBasedModel(equil_nfls, y0=0, mb_model=equil_mbmod, do_kcalving=True,
                                                        glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                                        is_tidewater=equil_gdir.is_tidewater,
                                                        water_level=water_level
                                                        )
                        
                        try:
                            if Version(oggm_version) < Version('1.5.3'):
                                _, equil_diag = equil_ev_model.run_until_and_store(equil_nyears)
                            else:
                                equil_diag = equil_ev_model.run_until_and_store(equil_nyears)
                        
                            equil_ev_model.mb_model.glac_wide_volume_annual[-1] = equil_diag.volume_m3[-1]
                            equil_ev_model.mb_model.glac_wide_area_annual[-1]   = equil_diag.area_m2[-1]
                            
                            if equil_gdir.is_tidewater:
                                
                                equil_calving_m3_annual = ((equil_diag.calving_m3.values[1:] - equil_diag.calving_m3.values[0:-1]) * 
                                                           pygem_prms.density_ice / pygem_prms.density_water)
                                for n in np.arange(equil_calving_m3_annual.shape[0]):
                                    equil_ev_model.mb_model.glac_wide_frontalablation[12*n+11] = equil_calving_m3_annual[n]

                                # Glacier-wide total mass balance (m3 w.e.)
                                equil_ev_model.mb_model.glac_wide_massbaltotal = (
                                    equil_ev_model.mb_model.glac_wide_massbaltotal - equil_ev_model.mb_model.glac_wide_frontalablation)
                                
                        except RuntimeError as e:
                            if 'Glacier exceeds domain boundaries' in repr(e):
                                equil_count_exceed_boundary_errors += 1
                                equil_successful_run = False
                                
                                # LOG FAILURE
                                fail_domain_fp = (pygem_prms.output_sim_fp + 'equilibrium-fail-exceed_domain/' + reg_str + '/' 
                                                  + gcm_name + '/')
                                if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
                                    fail_domain_fp += scenario + '/'
                                if not os.path.exists(fail_domain_fp):
                                    os.makedirs(fail_domain_fp, exist_ok=True)
                                txt_fn_fail = glacier_str + "-sim_failed-equilibrium_experiment.txt"
                                with open(fail_domain_fp + txt_fn_fail, "w") as text_file:
                                    text_file.write(glacier_str + ' failed to complete ' + 
                                                    str(equil_count_exceed_boundary_errors) + ' simulations')
                            elif equil_gdir.is_tidewater:
                                if debug:
                                    print('OGGM dynamics failed, using mass redistribution curves')
                                # Mass redistribution curves glacier dynamics model
                                equil_ev_model = MassRedistributionCurveModel(
                                                    equil_nfls, mb_model=equil_mbmod, y0=0,
                                                    glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                                    is_tidewater=equil_gdir.is_tidewater,
                                                    water_level=water_level
                                                    )
                                if Version(oggm_version) < Version('1.5.3'):
                                    _, equil_diag = equil_ev_model.run_until_and_store(equil_nyears)
                                else:
                                    equil_diag = equil_ev_model.run_until_and_store(equil_nyears)
                                equil_diag = equil_diag[-1]
                                equil_ev_model.mb_model.glac_wide_volume_annual = equil_diag.volume_m3.values
                                equil_ev_model.mb_model.glac_wide_area_annual = equil_diag.area_m2.values
                                
                                # Record frontal ablation for tidewater glaciers and update total mass balance
                                # Update glacier-wide frontal ablation (m3 w.e.)
                                equil_ev_model.mb_model.glac_wide_frontalablation = equil_ev_model.mb_model.glac_bin_frontalablation.sum(0)
                                # Update glacier-wide total mass balance (m3 w.e.)
                                equil_ev_model.mb_model.glac_wide_massbaltotal = (
                                        equil_ev_model.mb_model.glac_wide_massbaltotal - equil_ev_model.mb_model.glac_wide_frontalablation)

                        except:
                            if equil_gdir.is_tidewater:

                                equil_ev_model = MassRedistributionCurveModel(
                                                    equil_nfls, mb_model=equil_mbmod, y0=0,
                                                    glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                                    is_tidewater=equil_gdir.is_tidewater,
                                                    water_level=water_level
                                                    )
                                if Version(oggm_version) < Version('1.5.3'):
                                    _, equil_diag = equil_ev_model.run_until_and_store(equil_nyears)
                                else:
                                    equil_diag = equil_ev_model.run_until_and_store(equil_nyears)
                                equil_diag = equil_diag[-1]
                                equil_ev_model.mb_model.glac_wide_volume_annual = equil_diag.volume_m3.values
                                equil_ev_model.mb_model.glac_wide_area_annual = equil_diag.area_m2.values
                
                                # Record frontal ablation for tidewater glaciers and update total mass balance
                                # Update glacier-wide frontal ablation (m3 w.e.)
                                equil_ev_model.mb_model.glac_wide_frontalablation = equil_ev_model.mb_model.glac_bin_frontalablation.sum(0)
                                # Update glacier-wide total mass balance (m3 w.e.)
                                equil_ev_model.mb_model.glac_wide_massbaltotal = (
                                    equil_ev_model.mb_model.glac_wide_massbaltotal - equil_ev_model.mb_model.glac_wide_frontalablation)
                
                            else:
                                raise
                            
                        if equil_successful_run:
                            
                            equil_area_initial = equil_mbmod.glac_bin_area_annual[:,0].sum()
                            equil_mb_mwea_diag = ((equil_diag.volume_m3.values[-1] - equil_diag.volume_m3.values[0])
                                            / equil_area_initial / equil_nyears * pygem_prms.density_ice / pygem_prms.density_water)
                            equil_mb_mwea_mbmod = equil_mbmod.glac_wide_massbaltotal.sum() / equil_area_initial / equil_nyears
                            
                            if np.abs(equil_mb_mwea_diag - equil_mb_mwea_mbmod) > 1e-6:
                                equil_ev_model.mb_model.ensure_mass_conservation(equil_diag)
                            
                            # RECORD PARAMETERS TO DATASET
                            equil_glac_temp_monthly[:, n_iter]                   = equil_mbmod.glac_wide_temp
                            equil_glac_prec_monthly[:, n_iter]                   = equil_mbmod.glac_wide_prec
                            equil_glac_acc_monthly[:, n_iter]                    = equil_mbmod.glac_wide_acc
                            equil_glac_refreeze_monthly[:, n_iter]               = equil_mbmod.glac_wide_refreeze
                            equil_glac_melt_monthly[:, n_iter]                   = equil_mbmod.glac_wide_melt
                            equil_glac_frontalablation_monthly[:, n_iter]        = equil_mbmod.glac_wide_frontalablation
                            equil_glac_massbaltotal_monthly[:, n_iter]           = equil_mbmod.glac_wide_massbaltotal
                            equil_glac_area_annual[:, n_iter]                    = equil_diag.area_m2.values
                            equil_glac_volume_annual[:, n_iter]                  = equil_diag.volume_m3.values
                            equil_glac_volume_bsl_annual[:, n_iter]              = equil_diag.volume_bsl_m3.values
                            equil_glac_volume_change_ignored_annual[:-1, n_iter] = equil_mbmod.glac_wide_volume_change_ignored_annual
                            equil_glac_ELA_annual[:, n_iter]                     = equil_mbmod.glac_wide_ELA_annual
                            
                            if equil_glac_bin_icethickness_annual is None:
                                equil_glac_bin_icethickness_annual_sim = (equil_mbmod.glac_bin_icethickness_annual)[:,:,np.newaxis]
                                equil_glac_bin_area_annual_sim = (equil_mbmod.glac_bin_area_annual)[:,:,np.newaxis]
                                # Update the latest thickness and volume
                                fl_dx_meter = getattr(equil_ev_model.fls[0], 'dx_meter', None)
                                fl_widths_m = getattr(equil_ev_model.fls[0], 'widths_m', None)
                                fl_section = getattr(equil_ev_model.fls[0],'section',None)
                                if fl_section is not None and fl_widths_m is not None:                                
                                    # thickness
                                    icethickness_t0 = np.zeros(fl_section.shape)
                                    icethickness_t0[fl_widths_m > 0] = fl_section[fl_widths_m > 0] / fl_widths_m[fl_widths_m > 0]
                                    equil_glac_bin_icethickness_annual_sim[:,-1,0] = icethickness_t0
                                    # area
                                    glacier_area_t0 = fl_widths_m * fl_dx_meter
                                    equil_glac_bin_area_annual_sim[:,-1,0] = glacier_area_t0
                                    
                                equil_glac_bin_icethickness_annual = equil_glac_bin_icethickness_annual_sim
                                equil_glac_bin_area_annual = equil_glac_bin_area_annual_sim
                                equil_glac_bin_massbalclim_annual_sim = np.zeros(equil_mbmod.glac_bin_icethickness_annual.shape)
                                equil_glac_bin_massbalclim_annual_sim[:,:-1] = equil_mbmod.glac_bin_massbalclim_annual
                                equil_glac_bin_massbalclim_annual = equil_glac_bin_massbalclim_annual_sim[:,:,np.newaxis]
                            else:
                                # Update the latest thickness and volume
                                equil_glac_bin_icethickness_annual_sim = (equil_mbmod.glac_bin_icethickness_annual)[:,:,np.newaxis]
                                equil_glac_bin_area_annual_sim = (equil_mbmod.glac_bin_area_annual)[:,:,np.newaxis]
                                fl_dx_meter = getattr(equil_ev_model.fls[0], 'dx_meter', None)
                                fl_widths_m = getattr(equil_ev_model.fls[0], 'widths_m', None)
                                fl_section = getattr(equil_ev_model.fls[0],'section',None)
                                if fl_section is not None and fl_widths_m is not None:
                                    # thickness
                                    icethickness_t0 = np.zeros(fl_section.shape)
                                    icethickness_t0[fl_widths_m > 0] = fl_section[fl_widths_m > 0] / fl_widths_m[fl_widths_m > 0]
                                    equil_glac_bin_icethickness_annual_sim[:,-1,0] = icethickness_t0
                                    # area
                                    glacier_area_t0 = fl_widths_m * fl_dx_meter
                                    equil_glac_bin_area_annual_sim[:,-1,0] = glacier_area_t0

                                equil_glac_bin_icethickness_annual = np.append(equil_glac_bin_icethickness_annual,
                                                                               equil_glac_bin_icethickness_annual_sim,
                                                                               axis=2)
                                equil_glac_bin_area_annual = np.append(equil_glac_bin_area_annual,
                                                                       equil_glac_bin_area_annual_sim,
                                                                       axis=2)
                                equil_glac_bin_massbalclim_annual_sim = np.zeros(equil_mbmod.glac_bin_icethickness_annual.shape)
                                equil_glac_bin_massbalclim_annual_sim[:,:-1] = equil_mbmod.glac_bin_massbalclim_annual
                                equil_glac_bin_massbalclim_annual = np.append(equil_glac_bin_massbalclim_annual, 
                                                                              equil_glac_bin_massbalclim_annual_sim[:,:,np.newaxis],
                                                                              axis=2)
                        else:
                            if equil_glac_bin_icethickness_annual is None:
                                equil_glac_bin_icethickness_annual = np.zeros([len(equil_nfls[0].surface_h), length_exp+1, 1]) * np.nan
                                equil_glac_bin_area_annual         = np.zeros([len(equil_nfls[0].surface_h), length_exp+1, 1]) * np.nan
                                equil_glac_bin_massbalclim_annual  = np.zeros([len(equil_nfls[0].surface_h), length_exp+1, 1]) * np.nan
                                
                            else:
                                equil_glac_bin_icethickness_annual_sim = np.zeros([len(equil_nfls[0].surface_h), length_exp+1, 1]) * np.nan
                                equil_glac_bin_area_annual_sim         = np.zeros([len(equil_nfls[0].surface_h), length_exp+1, 1]) * np.nan
                                equil_glac_bin_massbalclim_annual_sim  = np.zeros([len(equil_nfls[0].surface_h), length_exp+1, 1]) * np.nan
                                
                                
                                equil_glac_bin_icethickness_annual = np.append(equil_glac_bin_icethickness_annual,
                                                                               equil_glac_bin_icethickness_annual_sim,
                                                                               axis=2)
                                equil_glac_bin_area_annual = np.append(equil_glac_bin_area_annual,
                                                                       equil_glac_bin_area_annual_sim,
                                                                       axis=2)
                                equil_glac_bin_massbalclim_annual = np.append(equil_glac_bin_massbalclim_annual,
                                                                              equil_glac_bin_massbalclim_annual_sim,
                                                                              axis=2)
                    
                    else:
                        if output_glac_bin_icethickness_annual is None:
                            output_glac_bin_icethickness_annual = np.zeros([len(nfls[0].surface_h), nyears+1, 1]) * np.nan
                            output_glac_bin_area_annual         = np.zeros([len(nfls[0].surface_h), nyears+1, 1]) * np.nan
                            output_glac_bin_massbalclim_annual  = np.zeros([len(nfls[0].surface_h), nyears+1, 1]) * np.nan
                            
                            equil_glac_bin_icethickness_annual = np.zeros([len(nfls[0].surface_h), length_exp+1, 1]) * np.nan
                            equil_glac_bin_area_annual         = np.zeros([len(nfls[0].surface_h), length_exp+1, 1]) * np.nan
                            equil_glac_bin_massbalclim_annual  = np.zeros([len(nfls[0].surface_h), length_exp+1, 1]) * np.nan
                            
                        else:
                            output_glac_bin_icethickness_annual_sim = np.zeros([len(nfls[0].surface_h), nyears+1, 1]) * np.nan
                            output_glac_bin_area_annual_sim         = np.zeros([len(nfls[0].surface_h), nyears+1, 1]) * np.nan
                            output_glac_bin_massbalclim_annual_sim  = np.zeros([len(nfls[0].surface_h), nyears+1, 1]) * np.nan
                            
                            equil_glac_bin_icethickness_annual_sim = np.zeros([len(nfls[0].surface_h), length_exp+1, 1]) * np.nan
                            equil_glac_bin_area_annual_sim         = np.zeros([len(nfls[0].surface_h), length_exp+1, 1]) * np.nan
                            equil_glac_bin_massbalclim_annual_sim  = np.zeros([len(nfls[0].surface_h), length_exp+1, 1]) * np.nan
                            
                            output_glac_bin_icethickness_annual = np.append(output_glac_bin_icethickness_annual,
                                                                            output_glac_bin_icethickness_annual_sim,
                                                                            axis=2)
                            output_glac_bin_area_annual = np.append(output_glac_bin_area_annual,
                                                                    output_glac_bin_area_annual_sim,
                                                                    axis=2)
                            output_glac_bin_massbalclim_annual = np.append(output_glac_bin_massbalclim_annual,
                                                                           output_glac_bin_massbalclim_annual_sim,
                                                                           axis=2)
                            
                            equil_glac_bin_icethickness_annual = np.append(equil_glac_bin_icethickness_annual,
                                                                           equil_glac_bin_icethickness_annual_sim,
                                                                           axis=2)
                            equil_glac_bin_area_annual = np.append(equil_glac_bin_area_annual,
                                                                   equil_glac_bin_area_annual_sim,
                                                                   axis=2)
                            equil_glac_bin_massbalclim_annual = np.append(equil_glac_bin_massbalclim_annual,
                                                                          equil_glac_bin_massbalclim_annual_sim,
                                                                          axis=2)



                # ===== Export Results =====
                # Parameterization approach
                results_2020, intercept_results = parameterization_approach(gdir=gdir, bed_h = output_glac_bin_bed_h, 
                              massbaltotal_monthly = output_glac_massbaltotal_monthly, 
                              ELA_annual = output_glac_ELA_annual, area_annual = output_glac_area_annual, 
                              volume_annual = output_glac_volume_annual, volume_bsl_annual = output_glac_volume_bsl_annual, 
                              bin_icethickness_annual = output_glac_bin_icethickness_annual, 
                              bin_area_annual = output_glac_bin_area_annual, bin_massbalclim_annual = output_glac_bin_massbalclim_annual, 
                              water_level_all = water_level_all, nyears = nyears,
                              count_exceed_boundary_errors = count_exceed_boundary_errors)
                
                # Equilibrium simultion
                equil_results = equilibrium_experiment(equil_gdir=equil_gdir, bed_h = output_glac_bin_bed_h, 
                                    equil_ELA_annual = equil_glac_ELA_annual, equil_area_annual = equil_glac_area_annual, 
                                    equil_volume_annual = equil_glac_volume_annual, equil_volume_bsl_annual = equil_glac_volume_bsl_annual,
                                    equil_bin_icethickness_annual = equil_glac_bin_icethickness_annual, 
                                    equil_bin_area_annual = equil_glac_bin_area_annual, 
                                    equil_bin_massbalclim_annual = equil_glac_bin_massbalclim_annual, 
                                    water_level_all = water_level_all, equil_nyears = equil_nyears, 
                                    equil_count_exceed_boundary_errors = equil_count_exceed_boundary_errors,
                                    count_exceed_boundary_errors = count_exceed_boundary_errors)
                
                # ===== Export Results =====
                if count_exceed_boundary_errors <= pygem_prms.sim_iters:
                    sims = pygem_prms.sim_iters
                    # ----- STATS OF ALL VARIABLES -----
                    if pygem_prms.export_essential_data:
                        # Create empty dataset
                        output_ds_all_stats, encoding = create_xrdataset(glacier_rgi_table, dates_table, equil_nyears)
                        
                        output_ds_all_stats['is_icecap'].values    = np.array([gdir.is_icecap])
                        output_ds_all_stats['is_tidewater'].values = np.array([gdir.is_tidewater])
                        
                        # simulated glacier results in 2020
                        # 0-area_2020_m2, 1-volume_2020_m3, 2-volume_bsl_2020_m3
                        output_ds_all_stats['area_2020'].values[0,:]       = results_2020[0,:]
                        output_ds_all_stats['volume_2020'].values[0,:]     = results_2020[1,:]
                        output_ds_all_stats['volume_bsl_2020'].values[0,:] = results_2020[2,:]
                        
                        # results of parameterization approach and equilibrium simulaiton
                        output_ds_all_stats['intercept_results'].values[0,:,:] = intercept_results
                        output_ds_all_stats['equil_results'].values[0,:,:]     = equil_results
                        
                        # output_climate, equil_climate
                        # 0-temp, 1-prec, 2-acc, 3-refreeze, 4-melt, 5-frontalablation, 6-massbaltotal
                        output_glac_temp_stats = np.nanmean(np.nanmean((output_glac_temp_monthly.reshape(nyears,12,sims))[-10:,:,:], 
                                                                       axis=1), axis=0)
                        output_glac_prec_stats = np.nanmean(np.nansum((output_glac_prec_monthly.reshape(nyears,12,sims))[-10:,:,:], 
                                                                      axis=1), axis=0)
                        output_glac_acc_stats  = np.nanmean(np.nansum((output_glac_acc_monthly.reshape(nyears,12,sims))[-10:,:,:], 
                                                                      axis=1), axis=0)
                        output_glac_refreeze_stats = np.nanmean(np.nansum((output_glac_refreeze_monthly.reshape(nyears,12,sims))[-10:,:,:], 
                                                                      axis=1), axis=0)
                        output_glac_melt_stats = np.nanmean(np.nansum((output_glac_melt_monthly.reshape(nyears,12,sims))[-10:,:,:], 
                                                                      axis=1), axis=0)
                        output_glac_frontalablation_stats = np.nanmean(np.nansum((output_glac_frontalablation_monthly.reshape(nyears,12,sims))[-10:,:,:], 
                                                                      axis=1), axis=0)
                        output_glac_massbaltotal_stats = np.nanmean(np.nansum((output_glac_massbaltotal_monthly.reshape(nyears,12,sims))[-10:,:,:], 
                                                                      axis=1), axis=0)
                        
                        output_ds_all_stats['output_climate'].values[0,0,:] = calc_stats_array(output_glac_temp_stats)
                        output_ds_all_stats['output_climate'].values[0,1,:] = calc_stats_array(output_glac_prec_stats)
                        output_ds_all_stats['output_climate'].values[0,2,:] = calc_stats_array(output_glac_acc_stats)
                        output_ds_all_stats['output_climate'].values[0,3,:] = calc_stats_array(output_glac_melt_stats)
                        output_ds_all_stats['output_climate'].values[0,4,:] = calc_stats_array(output_glac_refreeze_stats)
                        output_ds_all_stats['output_climate'].values[0,5,:] = calc_stats_array(output_glac_frontalablation_stats)
                        output_ds_all_stats['output_climate'].values[0,6,:] = calc_stats_array(output_glac_massbaltotal_stats)
                        
                        equil_glac_temp_stats = np.nanmean(np.nanmean((equil_glac_temp_monthly.reshape(equil_nyears,12,sims))[-10:,:,:], 
                                                                       axis=1), axis=0)
                        equil_glac_prec_stats = np.nanmean(np.nansum((equil_glac_prec_monthly.reshape(equil_nyears,12,sims))[-10:,:,:], 
                                                                      axis=1), axis=0)
                        equil_glac_acc_stats  = np.nanmean(np.nansum((equil_glac_acc_monthly.reshape(equil_nyears,12,sims))[-10:,:,:], 
                                                                      axis=1), axis=0)
                        equil_glac_refreeze_stats = np.nanmean(np.nansum((equil_glac_refreeze_monthly.reshape(equil_nyears,12,sims))[-10:,:,:], 
                                                                      axis=1), axis=0)
                        equil_glac_melt_stats = np.nanmean(np.nansum((equil_glac_melt_monthly.reshape(equil_nyears,12,sims))[-10:,:,:], 
                                                                      axis=1), axis=0)
                        equil_glac_frontalablation_stats = np.nanmean(np.nansum((equil_glac_frontalablation_monthly.reshape(equil_nyears,12,sims))[-10:,:,:], 
                                                                      axis=1), axis=0)
                        equil_glac_massbaltotal_stats = np.nanmean(np.nansum((equil_glac_massbaltotal_monthly.reshape(equil_nyears,12,sims))[-10:,:,:], 
                                                                      axis=1), axis=0)
                        
                        output_ds_all_stats['equil_climate'].values[0,0,:] = calc_stats_array(equil_glac_temp_stats)
                        output_ds_all_stats['equil_climate'].values[0,1,:] = calc_stats_array(equil_glac_prec_stats)
                        output_ds_all_stats['equil_climate'].values[0,2,:] = calc_stats_array(equil_glac_acc_stats)
                        output_ds_all_stats['equil_climate'].values[0,3,:] = calc_stats_array(equil_glac_melt_stats)
                        output_ds_all_stats['equil_climate'].values[0,4,:] = calc_stats_array(equil_glac_refreeze_stats)
                        output_ds_all_stats['equil_climate'].values[0,5,:] = calc_stats_array(equil_glac_frontalablation_stats)
                        output_ds_all_stats['equil_climate'].values[0,6,:] = calc_stats_array(equil_glac_massbaltotal_stats)
                        
                        # Parameters
                        # 0-debris_hd, 1-debris_ed, 2-debris_area
                        # 3-kp, 4-tbias, 5-ddfsnow, 6-ddfice, 7-tsnow_threshold, 8-precgrad, 9-calving_k_values, 10-water_level
                        gridded_path = gdir.get_filepath('gridded_data')
                        gridded_data = xr.open_dataset(gridded_path)
                        try:
                            debris_hd = gridded_data['debris_hd'].values.flatten()
                            debris_hd = np.where(debris_hd==0., np.nan, debris_hd)
                            debris_hd = calc_stats_array(debris_hd)
                            
                            debris_ed = gridded_data['debris_ed'].values.flatten()
                            debris_ed = np.where(debris_ed==0., np.nan, debris_ed)
                            debris_ed = calc_stats_array(debris_ed)
                            
                            debris_area = np.where(debris_ed==0., 0, 1)
                            debris_area = np.nansum(debris_area) * nfls[0].dx_meter/2 *nfls[0].dx_meter/2
                            debris_area = calc_stats_array(np.array([debris_area]))
                        except:
                            debris_hd = np.zeros(8) * np.nan
                            debris_ed = np.zeros(8) * np.nan
                            debris_area = np.zeros(8) * np.nan
                        
                        output_ds_all_stats['params'].values[0,0,:] = debris_hd
                        output_ds_all_stats['params'].values[0,1,:] = debris_ed
                        output_ds_all_stats['params'].values[0,2,:] = debris_area
                        output_ds_all_stats['params'].values[0,3,:] = calc_stats_array(np.array(modelprms_all['kp'][:sims]))
                        output_ds_all_stats['params'].values[0,4,:] = calc_stats_array(np.array(modelprms_all['tbias'][:sims]))
                        output_ds_all_stats['params'].values[0,5,:] = calc_stats_array(np.array(modelprms_all['ddfsnow'][:sims]))
                        output_ds_all_stats['params'].values[0,6,:] = calc_stats_array(np.array(modelprms_all['ddfice'][:sims]))
                        output_ds_all_stats['params'].values[0,7,:] = calc_stats_array(np.array(modelprms_all['tsnow_threshold'][:sims]))
                        output_ds_all_stats['params'].values[0,8,:] = calc_stats_array(np.array(modelprms_all['precgrad'][:sims]))
                        
                        if calving_k is None:
                            output_ds_all_stats['params'].values[0,9,:] = np.zeros(8) * np.nan
                        else:
                            output_ds_all_stats['params'].values[0,9,:] = calc_stats_array(calving_k_values[:sims])
                        
                        output_ds_all_stats['params'].values[0,10,:] = calc_stats_array(water_level_all)
                        
                        # Annual changes
                        
                        output_ds_all_stats['equil_massbaltotal'].values[0,:-1,:] = calc_stats_array(
                            np.nanmean((equil_glac_temp_monthly.reshape(equil_nyears,12,sims)), axis=1))
                        output_ds_all_stats['equil_massbaltotal'].values[0,-1,:] = np.nan
                        output_ds_all_stats['equil_area'].values[0,:,:] = calc_stats_array(equil_glac_area_annual)
                        output_ds_all_stats['equil_volume'].values[0,:,:] = calc_stats_array(equil_glac_volume_annual)
                        output_ds_all_stats['equil_volume_bsl'].values[0,:,:] = calc_stats_array(equil_glac_volume_bsl_annual)
                        
                        # Export statistics to netcdf
                        output_sim_fp = pygem_prms.output_sim_fp + reg_str + '/' + gcm_name + '/'
                        if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
                            output_sim_fp += scenario + '/'
                        output_sim_fp += 'stats/'
                        # Create filepath if it does not exist
                        if os.path.exists(output_sim_fp) == False:
                            os.makedirs(output_sim_fp, exist_ok=True)
                        # Netcdf filename
                        if gcm_name in ['ERA-Interim', 'ERA5', 'COAWST']:
                            # Filename
                            netcdf_fn = (glacier_str + '_' + gcm_name + '_' + str(pygem_prms.option_calibration) + '_ba' +
                                          str(pygem_prms.option_bias_adjustment) + '_' +  str(sim_iters) + 'sets' + '_' +
                                          str(args.gcm_startyear) + '_' + str(args.gcm_endyear) + '_all.nc')
                        else:
                            netcdf_fn = (glacier_str + '_' + gcm_name + '_' + scenario + '_' +
                                          str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
                                          '_' + str(sim_iters) + 'sets' + '_' + str(args.gcm_startyear) + '_' + 
                                          str(args.gcm_endyear) + '_all.nc')
                        # Export netcdf
                        output_ds_all_stats.to_netcdf(output_sim_fp + netcdf_fn, encoding=encoding)
            
                        # Close datasets
                        output_ds_all_stats.close()
                    
                    
#        print('\n\nADD BACK IN EXCEPTION\n\n')
        except:
            # LOG FAILURE
            fail_fp = pygem_prms.output_sim_fp + 'failed/' + reg_str + '/' + gcm_name + '/'
            if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
                fail_fp += scenario + '/'
            if not os.path.exists(fail_fp):
                os.makedirs(fail_fp, exist_ok=True)
            txt_fn_fail = glacier_str + "-sim_failed.txt"
            with open(fail_fp + txt_fn_fail, "w") as text_file:
                text_file.write(glacier_str + ' failed to complete simulation')

    # Global variables for Spyder development
    if args.option_parallels == 0:
        global main_vars
        main_vars = inspect.currentframe().f_locals


#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    if args.debug == 1:
        debug = True
    else:
        debug = False

    if not 'pygem_modelprms' in cfg.BASENAMES:
        cfg.BASENAMES['pygem_modelprms'] = ('pygem_modelprms.pkl', 'PyGEM model parameters')

    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            glac_no = pickle.load(f)
    elif pygem_prms.glac_no is not None:
        glac_no = pygem_prms.glac_no
    else:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
                rgi_glac_number=pygem_prms.rgi_glac_number, include_landterm=pygem_prms.include_landterm,
                include_laketerm=pygem_prms.include_laketerm, include_tidewater=pygem_prms.include_tidewater)
        glac_no = list(main_glac_rgi_all['rgino_str'].values)

    # Number of cores for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([len(glac_no), args.num_simultaneous_processes]))
    else:
        num_cores = 1

    # Glacier number lists to pass for parallel processing
    glac_no_lsts = split_glaciers.split_list(glac_no, n=num_cores, option_ordered=args.option_ordered)

    # Read GCM names from argument parser
    gcm_name = args.gcm_list_fn
    if args.gcm_name is not None:
        gcm_list = [args.gcm_name]
        scenario = args.scenario
    elif args.gcm_list_fn == pygem_prms.ref_gcm_name:
        gcm_list = [pygem_prms.ref_gcm_name]
        scenario = args.scenario
    else:
        with open(args.gcm_list_fn, 'r') as gcm_fn:
            gcm_list = gcm_fn.read().splitlines()
            scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
            print('Found %d gcms to process'%(len(gcm_list)))

    # Loop through all GCMs
    for gcm_name in gcm_list:
        if args.scenario is None:
            print('Processing:', gcm_name)
        elif not args.scenario is None:
            print('Processing:', gcm_name, scenario)
        # Pack variables for multiprocessing
        list_packed_vars = []
        for count, glac_no_lst in enumerate(glac_no_lsts):
            list_packed_vars.append([count, glac_no_lst, gcm_name])
#%%
        # Parallel processing
        if args.option_parallels != 0:
            print('Processing in parallel with ' + str(args.num_simultaneous_processes) + ' cores...')
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,list_packed_vars)
        # If not in parallel, then only should be one loop
        else:
            # Loop through the chunks and export bias adjustments
            for n in range(len(list_packed_vars)):
                main(list_packed_vars[n])

    print('Total processing time:', time.time()-time_start, 's')
