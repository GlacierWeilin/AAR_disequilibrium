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
sys.path.append('/g/data/rd53/wy2165/disequilibrium/code_pygem_oggm/')
from CalDisequilibriumPyGEM import (cal_AAR_steady, compile_AAR_steady)

cfg.initialize(logging_level=pygem_prms['oggm']['logging_level'])
cfg.PATHS['working_dir'] = f"{pygem_prms['root']}/{pygem_prms['oggm']['oggm_gdir_relpath']}"

###################################################################################################################################################
# init the glacier directory
region = 2
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
gcm = 'ERA5'
calibration = 'MCMC'
bias_adj = 0
sets = 1

# add more basenames
cfg.add_to_basenames('glacier_AAR_steady', 'glacier_AAR_steady.json', docstr='')
os.makedirs(cfg.PATHS['working_dir'] + f'OGGM/{region:02d}/', exist_ok=True)

#%% Main simulation loop
# Divide glaciers into batches to avoid memory overload
batch_size = 500
n_glaciers = len(rgiids)
n_batches = int(np.ceil(n_glaciers / batch_size))

for i in range(n_batches):
    
    start = i * batch_size
    end = start + batch_size

    subset = rgiids[start:end]  # Select current batch of glaciers

    gdirs = workflow.init_glacier_directories(subset)

    filesuffix=f'_{region:02d}_{gcm}_batch_{i+1}'
    #path = os.path.join(cfg.PATHS['working_dir'] + f'OGGM/{region:02d}/', ('glacier_statistics_PyGEM' + filesuffix + '.csv'))
    #utils.compile_glacier_statistics(gdirs, path=path);
    
  ###################################################################################################################################################
    workflow.execute_entity_task(cal_AAR_steady, gdirs,
                                 pygem_prms = pygem_prms,
                                 region = region,
                                 gcm = gcm,
                                 calibration = calibration,
                                 bias_adj = bias_adj,
                                 sets = sets,
                                 startyear = 2000,
                                 endyear = 2025,
                                 benchmark = 2020
                                )
    
    path = os.path.join(cfg.PATHS['working_dir'] + f'OGGM/{region:02d}/', ('glacier_AAR_steady_PyGEM' + filesuffix + '.csv'))
    compile_AAR_steady(gdirs, path=path);