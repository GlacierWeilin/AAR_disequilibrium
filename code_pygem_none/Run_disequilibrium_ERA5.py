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
from oggm import workflow, cfg

# import calculator created by Weilin Yang
import sys
sys.path.append('/g/data/rd53/wy2165/disequilibrium/code_pygem_none/')
from CalDisequilibriumPyGEM import (cal_disequilibrium_ERA5, compile_disequilibrium)

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
gcm = 'ERA5'
calibration = 'MCMC'
sets = 1

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

    gdirs = workflow.init_glacier_directories(subset)

    ###################################################################################################################################################

    workflow.execute_entity_task(cal_disequilibrium_ERA5, gdirs,
                                 pygem_prms = pygem_prms,
                                 region = region,
                                 gcm = gcm,
                                 calibration = calibration,
                                 bias_adj = 0,
                                 sets = sets,
                                 startyear = 2014,
                                 endyear = 2023
                                 )

    filesuffix=f'_{region:02d}_{gcm}_batch_{i+1}'
    path = os.path.join(cfg.PATHS['working_dir'] + f'None/{region:02d}/', ('glacier_disequilibrium_PyGEM' + filesuffix + '.csv'))
    compile_disequilibrium(gdirs, path=path);