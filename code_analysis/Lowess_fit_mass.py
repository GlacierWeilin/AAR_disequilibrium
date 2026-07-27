import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append('/g/data/rd53/wy2165/disequilibrium/code_analysis/')
from Lowess_fit import run_lowess_from_csv


path = '/g/data/rd53/wy2165/disequilibrium/pygem_oggm/'
lowess_y_cols = ['mass_remaining', 'area_steady', 'volume_steady']
jobs = []

# Global a/k
for method in ['a', 'k']:
    for y_col in lowess_y_cols:
        jobs.append({
            'input_csv': path + f'PyGEM_global_mass_median_{method}.csv',
            'output_csv': path + f'PyGEM_global_mass_{y_col}_median_{method}_lowess_fit.csv',
            'trials_output_csv': None,
            'y_col': y_col,
            'preliminary_num_fits': 500,
            'final_num_fits': 2000,
            'robust_iters': 2,
        })

# Regional a only
regions_a = [1, 3, 4, 5, 7, 9, 17, 19]

for region in regions_a:
    for y_col in lowess_y_cols:
        jobs.append({
            'input_csv': path + f'PyGEM_regional_mass_{region:02d}_median_a.csv',
            'output_csv': path + f'PyGEM_regional_mass_{region:02d}_{y_col}_median_a_lowess_fit.csv',
            'trials_output_csv': None,
            'y_col': y_col,
            'preliminary_num_fits': 500,
            'final_num_fits': 2000,
            'robust_iters': 2,
        })

# Regional k all regions
regions_k = range(1, 20)

for region in regions_k:
    for y_col in lowess_y_cols:
        jobs.append({
            'input_csv': path + f'PyGEM_regional_mass_{region:02d}_median_k.csv',
            'output_csv': path + f'PyGEM_regional_mass_{region:02d}_{y_col}_median_k_lowess_fit.csv',
            'trials_output_csv': None,
            'y_col': y_col,
            'preliminary_num_fits': 500,
            'final_num_fits': 2000,
            'robust_iters': 2,
        })


# Run LOWESS jobs in parallel.
max_workers = 10
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
