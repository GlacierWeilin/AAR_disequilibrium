#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: Weilin Yang (weilinyang.yang@monash.edu; ywlcwc@gmail.com)
'''

import os
import numpy as np

# pygem imports
from pygem.setup.config import ConfigManager
# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config() 
import pygem.pygem_modelsetup as modelsetup

# oggm imports
from oggm import workflow, cfg, utils

# import calculator created by Weilin Yang
import sys
sys.path.append('/g/data/rd53/wy2165/disequilibrium/code_pygem_none/')
from CalDisequilibriumPyGEM import (cal_disequilibrium, compile_disequilibrium)

cfg.initialize(logging_level=pygem_prms['oggm']['logging_level'])
cfg.PATHS['working_dir'] = f"{pygem_prms['root']}/{pygem_prms['oggm']['oggm_gdir_relpath']}"

###################################################################################################################################################
# init the glacier directory
region = 17
# get all glaciers in region to see which fraction ran successfully
main_glac_rgi_all = modelsetup.selectglaciersrgitable(
    rgi_regionsO1=[region],
    rgi_regionsO2='all',
    rgi_glac_number='all',
    glac_no=None,
    debug=True,
)

rgiids = main_glac_rgi_all['RGIId'].tolist()

###################################################################################################################################################
gcms = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL']
ssps = ['ssp126', 'ssp370', 'ssp585']
calibration = 'MCMC'
bias_adj = 1
sets = 1
startyears = [1851, 1901, 1951, 1995, 2021, 2041, 2061, 2081]
endyears   = [1870, 1920, 1970, 2014, 2040, 2060, 2080, 2100]

# add more basenames
cfg.add_to_basenames('glacier_disequilibrium', 'glacier_disequilibrium.json', docstr='')
os.makedirs(cfg.PATHS['working_dir'] + f'None/{region:02d}/', exist_ok=True)

#%% Main simulation loop
# Divide glaciers into batches to avoid memory overload
batch_size = 500
n_glaciers = len(rgiids)
n_batches = int(np.ceil(n_glaciers / batch_size))

for i in range(n_batches):
    
    start = i * batch_size
    end = start + batch_size

    subset = rgiids[start:end]  # Select current batch of glaciers

    #gdirs = workflow.init_glacier_directories(subset, from_tar=True, delete_tar=True)
    gdirs = workflow.init_glacier_directories(subset)

    ###################################################################################################################################################

    for gcm in gcms:
        for ssp in ssps:
            
            workflow.execute_entity_task(cal_disequilibrium, gdirs,
                                         pygem_prms = pygem_prms,
                                         region = region,
                                         gcm = gcm,
                                         ssp = ssp,
                                         calibration = calibration,
                                         bias_adj = bias_adj,
                                         sets = sets,
                                         startyears = startyears,
                                         endyears = endyears
                                         )

            filesuffix=f'_{region:02d}_{gcm}_{ssp}_batch_{i+1}'
            path = os.path.join(cfg.PATHS['working_dir'] + f'None/{region:02d}/', ('glacier_disequilibrium_PyGEM' + filesuffix + '.csv'))
            compile_disequilibrium(gdirs, path=path);

    # Clean up glacier directories to save disk space
    #workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=True)