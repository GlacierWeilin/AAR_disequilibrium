import numpy as np
import pandas as pd

outpath = '/Users/wyan0065/Desktop/OGGM/disequilibrium/data/'

base_file = '/Users/wyan0065/Desktop/OGGM/disequilibrium/data/WGMS_test_per_glacier.csv'
hugonnet_file = '/Users/wyan0065/Desktop/OGGM/geodetic_ref_mb/hugonnet_2021_ds_rgi60_pergla_rates_10_20_worldwide.csv'
wgms_file = '/Users/wyan0065/Desktop/OGGM/disequilibrium/data/DOI-WGMS-FoG-2026-02-10/data/mass_balance_AAR.csv'

base = pd.read_csv(base_file, usecols=['glacier_id', 'RGIId'])
hug = pd.read_csv(hugonnet_file, low_memory=False)
wgms = pd.read_csv(
    wgms_file,
    usecols=['glacier_id', 'year', 'annual_balance'],
    low_memory=False,
)

# One row per glacier in the requested RGIId base.
base = base.dropna(subset=['glacier_id', 'RGIId']).drop_duplicates(
    subset=['glacier_id', 'RGIId']
)

# Hugonnet's exact 2000-2020 product.
hug = hug.loc[
    hug['period'].eq('2000-01-01_2020-01-01'),
    ['rgiid', 'dmdtda'],
].rename(columns={'rgiid': 'RGIId'})
hug = hug.drop_duplicates(subset='RGIId')

# Average all valid annual balances from calendar/glaciological years 2000-2020,
# inclusive, for each WGMS glacier in the requested base.
wgms['year'] = pd.to_numeric(wgms['year'], errors='coerce')
wgms['annual_balance'] = pd.to_numeric(wgms['annual_balance'], errors='coerce')
wgms_period = wgms.loc[
    wgms['year'].between(2000, 2020, inclusive='both')
    & wgms['annual_balance'].notna()
].copy()

# Audit possible repeated records within a glacier-year before aggregation.
duplicate_glacier_year_rows = int(
    wgms_period.duplicated(['glacier_id', 'year'], keep=False).sum()
)

wgms_mean = (
    wgms_period.groupby('glacier_id', as_index=False)
    .agg(
        wgms_annual_balance_mean=('annual_balance', 'mean'),
        n_wgms_years=('year', 'nunique'),
        n_wgms_records=('annual_balance', 'size'),
        first_year=('year', 'min'),
        last_year=('year', 'max'),
    )
)

comparison = base.merge(hug, on='RGIId', how='left').merge(
    wgms_mean, on='glacier_id', how='left'
)
paired = comparison.dropna(subset=['dmdtda', 'wgms_annual_balance_mean']).copy()

hugonnet_dmdtda = paired['dmdtda'].to_numpy()
wgms_annual_balance = paired['wgms_annual_balance_mean'].to_numpy()
dmdtda_difference = wgms_annual_balance - hugonnet_dmdtda

hugonnet_dmdtda_mean = np.mean(hugonnet_dmdtda)
wgms_annual_balance_mean = np.mean(wgms_annual_balance)
dmdtda_mean_difference = np.mean(dmdtda_difference)
dmdtda_rmse = np.sqrt(np.mean(dmdtda_difference**2))

paired.sort_values('RGIId').to_csv(outpath + 'WGMS_Hugonnet_2000_2020_comparison.csv', index=False)