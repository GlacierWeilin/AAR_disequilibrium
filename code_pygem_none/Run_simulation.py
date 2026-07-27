### imports ###
import subprocess

# pygem imports
from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()   # NOTE: ensure that your root path in ~/PyGEM/config.yaml points to
                                            # the appropriate location. If any errors occur, check this first.
rootpath=pygem_prms['root']

# update the include_frontalablation key as described above
config_manager.update_config(updates={'setup.include_frontalablation' : True})

region = 9
num_cores=48     # change depending on how many cores you have/want to utilize

gcms = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL']
ssps = ['ssp126', 'ssp370', 'ssp585']
for gcm in gcms:
    for ssp in ssps:
        subprocess.run([
            'run_simulation',
            '-rgi_region01', str(region),
            '-sim_climate_name', str(gcm),
            '-sim_climate_scenario', str(ssp),
            '-sim_startyear', '1851',
            '-sim_endyear', '2100',
            '-ncores', str(num_cores),
            '-option_calibration', 'MCMC',
            '-option_dynamics', 'None',
            '-use_regional_glen_a', 'True',
            '-option_bias_adjustment', '1',
            '-nsims', '1',
            #'-export_all_simiters',
            '-export_extra_vars',
        ], check=True)



