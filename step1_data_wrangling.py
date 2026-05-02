# =============================================================================
#
#  MIS 502 — Final Project: Cross-Sectional Asset Pricing
#  CRSP/Compustat Data Pipeline
#
#  Script 1 of 4: DATA PREPARATION & WRANGLING
#
#  This script loads the raw accounting-matched panel, validates,
#  cleans, engineers 11 cross-sectional factors, and saves the
#  analysis-ready dataset.
#
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
from scipy import stats

# -------------------------------------------------------------------------
# CONFIGURATION — Change this path to match your machine
# -------------------------------------------------------------------------
BASE_DIR = r'D:\4RM TAB~WPI\WPI SEM 3\DATA MANGEMENT FOR ANALYTICS\Datasets\DATA\MAIN DATA WITH DESCRIPTION\FINAL DATASET'
DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CHART_DIR = os.path.join(BASE_DIR, 'charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, 'project_step1_monthly_master_dataset_accounting_matched (2).csv')

print('=' * 80)
print('STEP 1: DATA PREPARATION & WRANGLING')
print('=' * 80)
t0 = time.time()


# =============================================================================
# 1.1  LOAD RAW DATA
# =============================================================================
print('\n--- 1.1 Loading raw data ---')

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=['date', 'datadate', 'avail_date'],
    low_memory=False
)
print(f'Loaded {len(df):,} rows x {df.shape[1]} columns')
print(f'Date range: {df["date"].min():%Y-%m-%d} to {df["date"].max():%Y-%m-%d}')
print(f'Unique firms (permno): {df["permno"].nunique():,}')

# Sort by firm and date (required for time-series operations)
df = df.sort_values(['permno', 'date']).reset_index(drop=True)


# =============================================================================
# 1.2  DATA VALIDATION
# =============================================================================
print('\n--- 1.2 Data validation ---')

# Check duplicates
dupes = df.duplicated(subset=['permno', 'date']).sum()
print(f'Duplicate permno x date rows: {dupes}')

# Check share code and exchange code
invalid_shrcd = df[~df['shrcd'].isin([10, 11])].shape[0]
invalid_exchcd = df[~df['exchcd'].isin([1, 2, 3])].shape[0]
print(f'Invalid share codes: {invalid_shrcd}')
print(f'Invalid exchange codes: {invalid_exchcd}')

# Missing value summary for key columns
print('\nMissing values in key columns:')
key_cols = ['ret', 'ret_adj', 'prc', 'me', 'at', 'ceq', 'sale', 'ni',
            'oibdp', 'dltt', 'dlc', 'capx', 'book_equity']
for c in key_cols:
    n_miss = df[c].isnull().sum()
    pct = n_miss / len(df) * 100
    print(f'  {c:20s}: {n_miss:>10,} ({pct:.2f}%)')

# Exchange distribution
print('\nExchange distribution:')
exch_map = {1: 'NYSE', 2: 'AMEX', 3: 'NASDAQ'}
for code, name in exch_map.items():
    n = (df['exchcd'] == code).sum()
    print(f'  {name}: {n:>10,} ({n/len(df)*100:.1f}%)')

# Extreme returns
extreme = (df['ret_adj'].abs() > 3.0).sum()
print(f'\nExtreme returns (|ret_adj| > 300%): {extreme}')

# Negative book equity
neg_be = (df['book_equity'] < 0).sum()
print(f'Negative book equity observations: {neg_be:,}')


# =============================================================================
# 1.3  CLEANING AND FILTERING
# =============================================================================
print('\n--- 1.3 Cleaning and filtering ---')
rows_before = len(df)

# Operation 1: Remove duplicates
df = df.drop_duplicates(subset=['permno', 'date'], keep='first')
removed_1 = rows_before - len(df)
print(f'Op 1 - Remove duplicates: {removed_1:,} removed -> {len(df):,} remaining')

# Operation 2: Drop null returns
rows_b = len(df)
df = df.dropna(subset=['ret_adj'])
removed_2 = rows_b - len(df)
print(f'Op 2 - Drop null ret_adj: {removed_2:,} removed -> {len(df):,} remaining')

# Operation 3: Remove penny stocks (|price| < $1.00)
rows_b = len(df)
df = df[df['prc'].abs() >= 1.0]
removed_3 = rows_b - len(df)
print(f'Op 3 - Remove penny stocks: {removed_3:,} removed -> {len(df):,} remaining')

# Operation 4: Remove non-positive market equity
rows_b = len(df)
df = df[df['me'] > 0]
removed_4 = rows_b - len(df)
print(f'Op 4 - Remove non-positive ME: {removed_4:,} removed -> {len(df):,} remaining')

df = df.reset_index(drop=True)
print(f'\nCleaned panel: {len(df):,} rows x {df.shape[1]} columns')


# =============================================================================
# 1.4  RETURN WINSORIZATION
#
#  Mathematical definition:
#    For each month t, let r_{0.005,t} and r_{0.995,t} be the 0.5th
#    and 99.5th percentiles of the cross-section of returns.
#    The winsorized return is:
#
#      r_i,t^w = max( r_{0.005,t},  min( r_i,t,  r_{0.995,t} ) )
#
# =============================================================================
print('\n--- 1.4 Return winsorization ---')

# Preserve raw return
df['ret_adj_raw'] = df['ret_adj'].copy()

def winsorize_column(group, col, lower=0.005, upper=0.995):
    """Winsorize a column within a group at given percentiles."""
    lo = group[col].quantile(lower)
    hi = group[col].quantile(upper)
    group[col] = group[col].clip(lo, hi)
    return group

df = df.groupby('date', group_keys=False).apply(
    winsorize_column, col='ret_adj', lower=0.005, upper=0.995
)
clipped = (df['ret_adj'] != df['ret_adj_raw']).sum()
print(f'Winsorized {clipped:,} return observations')


# =============================================================================
# 1.5  DERIVED VARIABLES AND ACCOUNTING RATIOS
#
#  Book-to-Market:    BM_i,t = BE_i / ME_i,t-1
#  Profitability:     PROF_i,t = OIBDP_i / AT_i
#  Investment:        INV_i,t  = CAPX_i / AT_i
#  Leverage:          LEV_i,t  = (DLTT_i + DLC_i) / AT_i
#  Earnings/Price:    EP_i,t   = NI_i / ME_i,t-1
#
# =============================================================================
print('\n--- 1.5 Derived variables and accounting ratios ---')

# Lagged market equity (prior month ME for each firm)
df['me_lag1'] = df.groupby('permno')['me'].shift(1)

# Log market equity
df['log_me'] = np.log(df['me'].clip(lower=1))
df['log_me_lag1'] = np.log(df['me_lag1'].clip(lower=1))

# Book-to-Market ratio (use lagged ME to avoid look-ahead)
df['bm'] = df['book_equity'] / df['me_lag1']
# Set B/M to NaN where book equity is negative or ME is missing
df.loc[df['book_equity'] <= 0, 'bm'] = np.nan
df.loc[df['me_lag1'] <= 0, 'bm'] = np.nan

# Profitability = OIBDP / AT
df['prof'] = df['oibdp'] / df['at']
df.loc[df['at'] <= 0, 'prof'] = np.nan

# Investment = CAPX / AT
df['inv'] = df['capx'] / df['at']
df.loc[df['at'] <= 0, 'inv'] = np.nan

# Leverage = (DLTT + DLC) / AT  (fill missing debt with 0)
df['leverage'] = (df['dltt'].fillna(0) + df['dlc'].fillna(0)) / df['at']
df.loc[df['at'] <= 0, 'leverage'] = np.nan

# Earnings-to-Price = NI / ME_lag1
df['ep'] = df['ni'] / df['me_lag1']
df.loc[df['me_lag1'] <= 0, 'ep'] = np.nan

# Sales Growth = (sale_t - sale_{t-12}) / |sale_{t-12}|
df['sale_lag12'] = df.groupby('permno')['sale'].shift(12)
df['sales_growth'] = (df['sale'] - df['sale_lag12']) / df['sale_lag12'].abs()
df.loc[df['sale_lag12'].abs() < 0.01, 'sales_growth'] = np.nan  # avoid division by tiny numbers
df = df.drop(columns=['sale_lag12'])

# Winsorize all accounting ratios at 1st and 99th percentile per month
ratio_cols = ['bm', 'prof', 'inv', 'leverage', 'ep', 'sales_growth']
for col in ratio_cols:
    df[col + '_w'] = df[col].copy()
    df = df.groupby('date', group_keys=False).apply(
        winsorize_column, col=col + '_w', lower=0.01, upper=0.99
    )

print('Accounting ratios computed and winsorized:')
for col in ratio_cols:
    n_valid = df[col + '_w'].notna().sum()
    pct = n_valid / len(df) * 100
    print(f'  {col + "_w":20s}: {n_valid:>10,} valid ({pct:.1f}%)')


# =============================================================================
# 1.6  FACTOR ENGINEERING
#
#  Momentum:   MOM_i,t = Product(1 + r_i,s) - 1,  s = t-12 to t-2
#              (skip month t-1 to avoid short-term reversal)
#
#  Reversal:   REV_i,t = r_i,t-1  (prior month return)
#
#  Beta:       β_i,t = Cov(r_i, r_m) / Var(r_m)
#              estimated over rolling 60-month window, min 36 obs.
#
#  Volatility: VOL_i,t = Std(r_i,s),  s = t-12 to t-1
#              (12-month rolling standard deviation)
#
# =============================================================================
print('\n--- 1.6 Factor engineering ---')

# ---- Momentum (cumulative return from t-12 to t-2) ----
print('  Computing momentum...')
df['ret1p'] = 1 + df['ret_adj']
df['mom_12m'] = df.groupby('permno')['ret1p'].transform(
    lambda x: x.shift(2).rolling(window=11, min_periods=8).apply(np.prod, raw=True) - 1
)
df = df.drop(columns=['ret1p'])

# ---- Short-term reversal (prior month return) ----
print('  Computing short-term reversal...')
df['reversal'] = df.groupby('permno')['ret_adj'].shift(1)

# ---- Volatility (12-month rolling std) ----
print('  Computing volatility...')
df['volatility'] = df.groupby('permno')['ret_adj'].transform(
    lambda x: x.rolling(window=12, min_periods=8).std()
)

# ---- Beta (rolling 60-month CAPM beta) ----
print('  Computing beta (this takes a minute)...')
# First compute value-weighted market return each month
df['me_weight'] = df.groupby('date')['me'].transform(lambda x: x / x.sum())
mkt_ret = df.groupby('date').apply(lambda g: (g['ret_adj'] * g['me_weight']).sum()).reset_index()
mkt_ret.columns = ['date', 'mkt_ret']
df = df.merge(mkt_ret, on='date', how='left')

def rolling_beta(group, window=60, min_periods=36):
    """Compute rolling CAPM beta for a single stock."""
    r = group['ret_adj'].values
    m = group['mkt_ret'].values
    betas = np.full(len(r), np.nan)
    for i in range(min_periods, len(r)):
        start = max(0, i - window)
        ri = r[start:i]
        mi = m[start:i]
        if len(ri) >= min_periods:
            cov = np.cov(ri, mi)
            if cov[1, 1] > 0:
                betas[i] = cov[0, 1] / cov[1, 1]
    group['beta'] = betas
    return group

# Process beta in chunks for memory efficiency
print('    (processing beta for each firm...)')
df = df.groupby('permno', group_keys=False).apply(rolling_beta)

# ---- Size quintile (NYSE breakpoints) ----
print('  Computing size quintiles (NYSE breakpoints)...')
def assign_size_quintile(group):
    nyse = group[group['exchcd'] == 1]['log_me_lag1'].dropna()
    if len(nyse) < 5:
        group['size_quintile'] = np.nan
        return group
    breakpoints = nyse.quantile([0.2, 0.4, 0.6, 0.8]).values
    group['size_quintile'] = np.searchsorted(breakpoints, group['log_me_lag1'].values) + 1
    group.loc[group['log_me_lag1'].isna(), 'size_quintile'] = np.nan
    return group

df = df.groupby('date', group_keys=False).apply(assign_size_quintile)

# ---- Factor coverage summary ----
factor_cols = ['mom_12m', 'reversal', 'log_me_lag1', 'bm_w', 'prof_w',
               'inv_w', 'leverage_w', 'ep_w', 'beta', 'volatility', 'size_quintile']
factor_names = ['Momentum', 'Reversal', 'Size', 'Value (B/M)', 'Profitability',
                'Investment', 'Leverage', 'Earnings/Price', 'Beta', 'Volatility', 'Size Quintile']

print('\nFactor coverage:')
for name, col in zip(factor_names, factor_cols):
    n_valid = df[col].notna().sum()
    pct = n_valid / len(df) * 100
    print(f'  {name:25s} ({col}): {n_valid:>10,} valid ({pct:.1f}%)')


# =============================================================================
# 1.7  CROSS-SECTIONAL RANKS
#
#  For each factor f and month t, the percentile rank is:
#    rank_i,t = Rank(f_i,t) / N_t
#  where N_t is the number of non-null observations in month t.
#  Ranks range from 0 (lowest) to 1 (highest).
#
# =============================================================================
print('\n--- 1.7 Cross-sectional percentile ranks ---')

rank_factors = ['mom_12m', 'reversal', 'log_me_lag1', 'bm_w', 'prof_w',
                'inv_w', 'leverage_w', 'ep_w', 'beta', 'volatility']

for col in rank_factors:
    rank_col = col + '_rank'
    df[rank_col] = df.groupby('date')[col].rank(pct=True)
    print(f'  Created {rank_col}')


# =============================================================================
# 1.8  NEXT-MONTH RETURN (Target Variable)
#
#  For predictive modeling, the target is the next month's return:
#    y_i,t = r_i,t+1
#
# =============================================================================
print('\n--- 1.8 Creating target variable (next-month return) ---')
df['ret_next'] = df.groupby('permno')['ret_adj'].shift(-1)
n_valid_target = df['ret_next'].notna().sum()
print(f'Next-month return (ret_next): {n_valid_target:,} valid observations')


# =============================================================================
# 1.9  TRAIN / TEST SPLIT
#
#  Training:  All observations through December 2015
#  Testing:   January 2016 through November 2023
#
# =============================================================================
print('\n--- 1.9 Train/test split ---')
train_end = '2015-12-31'
df['split'] = np.where(df['date'] <= train_end, 'train', 'test')
n_train = (df['split'] == 'train').sum()
n_test = (df['split'] == 'test').sum()
print(f'Training set: {n_train:,} observations (through {train_end})')
print(f'Test set:     {n_test:,} observations (Jan 2016 - Nov 2023)')


# =============================================================================
# 1.10  SAVE ANALYSIS-READY DATASET
# =============================================================================
print('\n--- 1.10 Saving analysis-ready dataset ---')

output_path = os.path.join(OUTPUT_DIR, 'analysis_ready.csv')
df.to_csv(output_path, index=False)
print(f'Saved: {output_path}')
print(f'Final dataset: {len(df):,} rows x {df.shape[1]} columns')

# Also save a smaller summary for quick reference
summary = {
    'Total rows': len(df),
    'Total columns': df.shape[1],
    'Unique firms': df['permno'].nunique(),
    'Date range': f'{df["date"].min():%Y-%m-%d} to {df["date"].max():%Y-%m-%d}',
    'Training rows': n_train,
    'Test rows': n_test,
    'Factors engineered': len(factor_cols),
}
pd.Series(summary).to_csv(os.path.join(OUTPUT_DIR, 'pipeline_summary.csv'))

elapsed = time.time() - t0
print(f'\nStep 1 completed in {elapsed/60:.1f} minutes')
print('=' * 80)
