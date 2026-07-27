import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append('/g/data/rd53/wy2165/disequilibrium/code_analysis/')
from Lowess_fit import run_lowess_from_csv


path = '/g/data/rd53/wy2165/disequilibrium/pygem_oggm/'
lowess_y_cols = ['mass_remaining', 'area_steady', 'volume_steady']
jobs = []

# area class
for cla in ['-1', '1-10', '10-100', '100-']:
    for y_col in lowess_y_cols:
        jobs.append({
            'input_csv': path + f'PyGEM_glacier_mass_area_{cla}_median_a.csv',
            'output_csv': path + f'PyGEM_glacier_mass_area_{cla}_{y_col}_median_a_lowess_fit.csv',
            'trials_output_csv': None,
            'y_col': y_col,
            'preliminary_num_fits': 500,
            'final_num_fits': 2000,
            'robust_iters': 2,
        })

# tidewater class
for cla in ['land-terminating', 'marine-terminating']:
    for y_col in lowess_y_cols:
        jobs.append({
            'input_csv': path + f'PyGEM_glacier_mass_{cla}_median_a.csv',
            'output_csv': path + f'PyGEM_glacier_mass_{cla}_{y_col}_median_a_lowess_fit.csv',
            'trials_output_csv': None,
            'y_col': y_col,
            'preliminary_num_fits': 500,
            'final_num_fits': 2000,
            'robust_iters': 2,
        })

# MT class median_a
for cla in ['all', 'outAntarc', 'inAntarc']:
    for y_col in lowess_y_cols:
        jobs.append({
            'input_csv': path + f'PyGEM_glacier_mass_MT_{cla}_median_a.csv',
            'output_csv': path + f'PyGEM_glacier_mass_MT_{cla}_{y_col}_median_a_lowess_fit.csv',
            'trials_output_csv': None,
            'y_col': y_col,
            'preliminary_num_fits': 500,
            'final_num_fits': 2000,
            'robust_iters': 2,
        })

# MT class median_k
for cla in ['all', 'outAntarc', 'inAntarc']:
    for y_col in lowess_y_cols:
        jobs.append({
            'input_csv': path + f'PyGEM_glacier_mass_MT_{cla}_median_k.csv',
            'output_csv': path + f'PyGEM_glacier_mass_MT_{cla}_{y_col}_median_k_lowess_fit.csv',
            'trials_output_csv': None,
            'y_col': y_col,
            'preliminary_num_fits': 500,
            'final_num_fits': 2000,
            'robust_iters': 2,
        })

# Run LOWESS jobs in parallel.
max_workers = 18
ctx = mp.get_context('fork')

with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
    futures = {
        executor.submit(run_lowess_from_csv, **job): job
        for job in jobs
    }

    for future in as_completed(futures):
        job = futures[future]
        try:
            future.result()
            print('Finished:', job['output_csv'])
        except Exception as e:
            print('Failed:', job['input_csv'], 'y_col:', job['y_col'])
            print(e)
