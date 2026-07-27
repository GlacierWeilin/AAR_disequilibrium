import numpy as np
import pandas as pd
import scipy.stats as st
import xarray as xr


def cal_AAR_steady(AAR=None, smb=None, selected_years=None):

    smb = smb.set_index('year')
    AAR = AAR.set_index('year')

    smb = smb.loc[smb.index.isin(selected_years)]
    AAR = AAR.loc[AAR.index.isin(selected_years)]

    year = AAR.index
    x = []
    y = []

    for t in year:
        if t in smb.index:
            if np.isnan(smb['annual_balance'][t]) == False and np.isnan(AAR['aar'][t]) == False \
                and AAR['aar'][t] >= 0 and AAR['aar'][t] <= 1:
                    x = np.append(x, smb['annual_balance'][t])
                    y = np.append(y, AAR['aar'][t])

    n = len(x)
    if n >= 5:
        slope, intercept, r_value, p_value, mad_err = st.linregress(x, y)

        if intercept < 0.05 or intercept > 0.95:
            intercept = np.nan

        AAR_steady = intercept
    else:
        AAR_steady = np.nan

    return AAR_steady


filepath = '/Users/wyan0065/Desktop/OGGM/disequilibrium/data/DOI-WGMS-FoG-2026-02-10/data/'
outpath = '/Users/wyan0065/Desktop/OGGM/disequilibrium/data/'
pygem_path = '/Users/wyan0065/Desktop/OGGM/disequilibrium/pygem_oggm/'

wgms_id = pd.read_csv(filepath + 'WGMS_ID_AAR.csv')
wgms = pd.read_csv(filepath + 'glacier.csv')
_AAR = pd.read_csv(filepath + 'mass_balance.csv')
_smb = pd.read_csv(filepath + 'mass_balance.csv')

wgms = wgms.set_index('id')
wgms = wgms.dropna(subset=['rgi60_ids']).copy()
_AAR = _AAR.set_index('glacier_id')
_smb = _smb.set_index('glacier_id')

data = xr.open_dataset(pygem_path + 'PyGEM_global_glacier_stats_median_a.nc')
area = pd.DataFrame({
    'RGIId': data['rgi_id'].values.astype(str),
    'rgi_area_km2': data['rgi_area_km2'].values,
})


# %% calculate AAR_steady during different 25-year time period
all_AAR_steady_l = []

min_start_year = 1980
window_years = 25
max_start_year = int(min(_AAR['year'].max(), 2025 - window_years))
start_years = np.arange(min_start_year, max_start_year + 1, 1)

for start_year in start_years:
    end_year = start_year + window_years
    selected_years = np.arange(start_year, end_year + 1, 1)

    for i in range(0, len(wgms_id)):
        n = wgms_id['glacier_id'][i]

        if n in _AAR.index and n in _smb.index and n in wgms.index:
            AAR = _AAR.loc[n]
            smb = _smb.loc[n]

            if type(AAR['year']) is not np.int64 and type(smb['year']) is not np.int64:
                AAR_steady = cal_AAR_steady(
                    AAR=AAR,
                    smb=smb,
                    selected_years=selected_years
                )
            else:
                AAR_steady = np.nan
        else:
            AAR_steady = np.nan

        if np.isnan(AAR_steady) == False:
            all_AAR_steady_l.append({
                'glacier_id': n,
                'RGIId': wgms.loc[n]['rgi60_ids'],
                'start_year': start_year,
                'end_year': end_year,
                'AAR_steady': AAR_steady,
            })

all_AAR_steady = pd.DataFrame(all_AAR_steady_l)

valid_count = (
    all_AAR_steady
    .groupby('glacier_id')['start_year']
    .nunique()
)

valid_glacier_id = valid_count[
    valid_count == len(start_years)
].index.values

all_AAR_steady = all_AAR_steady[
    all_AAR_steady['glacier_id'].isin(valid_glacier_id)
].copy()

all_AAR_steady['RGIId'] = all_AAR_steady['RGIId'].astype(str).str.split(';').str[0]
all_AAR_steady = all_AAR_steady.merge(
    area,
    on='RGIId',
    how='left'
)

window_stats_l = []

for start_year in start_years:
    end_year = start_year + window_years

    sub = all_AAR_steady[
        all_AAR_steady['start_year'] == start_year
    ].copy()

    AAR_steady_l = sub['AAR_steady'].values
    sub_area = sub.dropna(subset=['AAR_steady', 'rgi_area_km2']).copy()

    window_stats_l.append({
        'start_year': start_year,
        'end_year': end_year,
        'AAR_steady_mean': np.nanmean(AAR_steady_l),
        'AAR_steady_std': np.nanstd(AAR_steady_l),
        'AAR_steady_median': np.nanmedian(AAR_steady_l),
        'AAR_steady_MAD': st.median_abs_deviation(AAR_steady_l, nan_policy='omit'),
        'AAR_steady_area_weighted_mean': np.average(
            sub_area['AAR_steady'].values,
            weights=sub_area['rgi_area_km2'].values
        ),
        'n': len(AAR_steady_l),
        'n_area_weighted': len(sub_area),
    })

window_stats = pd.DataFrame(window_stats_l)

window_stats.to_csv(outpath + 'WGMS_test.csv', index=False)


# %% output per-glacier AAR_steady for each start year
per_glacier = (
    all_AAR_steady
    .pivot_table(
        index=['glacier_id', 'RGIId', 'rgi_area_km2'],
        columns='start_year',
        values='AAR_steady'
    )
    .reset_index()
)

per_glacier.columns = [
    col if isinstance(col, str) else f'AAR_steady_{int(col)}'
    for col in per_glacier.columns
]

per_glacier.to_csv(outpath + 'WGMS_test_per_glacier.csv', index=False)
