import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.append('/g/data/rd53/wy2165/disequilibrium/code_analysis/')
from Lowess_fit import run_lowess_from_csv

path = '/g/data/rd53/wy2165/disequilibrium/pygem_oggm/'
lowess_y_cols = ['AAR_steady_area_weighted_mean',
                 'AAR_mean_area_weighted_mean',
                 'disequilibrium_area_weighted_mean']
jobs = []

# Global a/k
for method in ['a', 'k']:
    for y_col in lowess_y_cols:
        jobs.append({
            'input_csv': path + f'PyGEM_global_stats_median_{method}.csv',
            'output_csv': path + f'PyGEM_global_stats_{y_col}_median_{method}_lowess_fit.csv',
            'trials_output_csv': None,
            'y_col': y_col,
            'preliminary_num_fits': 500,
            'final_num_fits': 2000,
            'robust_iters': 2,
        })


# Run LOWESS jobs in parallel.
max_workers = 6
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
