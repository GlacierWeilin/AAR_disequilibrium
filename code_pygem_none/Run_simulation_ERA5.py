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

region = 13
num_cores=48     # change depending on how many cores you have/want to utilize

gcm = 'ERA5'
subprocess.run([
    'run_simulation',
    '-rgi_region01', str(region),
    '-sim_climate_name', str(gcm),
    '-sim_startyear', '2014', # 1995
    '-sim_endyear', '2023', # 2014
    '-ncores', str(num_cores),
    '-option_calibration', 'MCMC',
    '-option_dynamics', 'None',
    '-use_regional_glen_a', 'True',
    '-option_bias_adjustment', '1',
    '-nsims', '1',
    #'-export_all_simiters',
    '-export_extra_vars',
], check=True)


