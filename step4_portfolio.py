# =============================================================================
#
#  MIS 502 — Final Project: Cross-Sectional Asset Pricing
#  CRSP/Compustat Data Pipeline
#
#  Script 4 of 4: PORTFOLIO OPTIMIZATION
#
#  Implements Markowitz Mean-Variance Optimization:
#
#    max_{w}   w'μ / sqrt(w'Σw)      (maximize Sharpe ratio)
#
#    subject to:
#      Σ w_i = 1                     (fully invested)
#      0 ≤ w_i ≤ 0.05               (no short selling, max 5% per stock)
#
#  Uses Ledoit-Wolf covariance shrinkage for numerical stability:
#    Σ_shrunk = δ·F + (1-δ)·S
#    where F = structured target, S = sample covariance, δ = shrinkage intensity
#
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
BASE_DIR = r'D:\4RM TAB~WPI\WPI SEM 3\DATA MANGEMENT FOR ANALYTICS\Datasets\DATA\MAIN DATA WITH DESCRIPTION\FINAL DATASET'
DATA_DIR = os.path.join(BASE_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CHART_DIR = os.path.join(BASE_DIR, 'charts')
os.makedirs(RESULTS_DIR, exist_ok=True)

print('=' * 80)
print('STEP 4: PORTFOLIO OPTIMIZATION')
print('=' * 80)
t0 = time.time()

# Load data
print('\nLoading data...')
df = pd.read_csv(os.path.join(DATA_DIR, 'analysis_ready.csv'), parse_dates=['date'], low_memory=False)

# Use training period only
train = df[df['split'] == 'train'].copy()
print(f'Training data: {len(train):,} rows')


# =============================================================================
# 4.1  SELECT OPTIMIZATION UNIVERSE
# =============================================================================
print('\n--- 4.1 Select optimization universe ---')

# Top 200 stocks by average market cap during training period
avg_me = train.groupby('permno')['me'].mean().nlargest(200)
universe = avg_me.index.tolist()

# Pivot to get monthly returns matrix
ret_matrix = train[train['permno'].isin(universe)].pivot_table(
    index='date', columns='permno', values='ret_adj'
)

# Drop columns with too many missing values
min_obs = 120  # require at least 10 years of data
ret_matrix = ret_matrix.dropna(axis=1, thresh=min_obs)
ret_matrix = ret_matrix.dropna()

n_stocks = ret_matrix.shape[1]
n_months = ret_matrix.shape[0]
print(f'Optimization universe: {n_stocks} stocks x {n_months} months')

# Get stock names for display
stock_names = {}
for permno in ret_matrix.columns:
    name_data = df[df['permno'] == permno][['tic', 'conm']].dropna().iloc[-1:] if len(df[df['permno'] == permno]) > 0 else pd.DataFrame()
    if len(name_data) > 0:
        stock_names[permno] = f"{name_data['tic'].values[0]} ({name_data['conm'].values[0][:20]})"
    else:
        stock_names[permno] = str(permno)


# =============================================================================
# 4.2  ESTIMATE EXPECTED RETURNS AND COVARIANCE
# =============================================================================
print('\n--- 4.2 Estimate expected returns and covariance ---')

mu = ret_matrix.mean().values  # Monthly expected returns
print(f'Average monthly return range: {mu.min()*100:.3f}% to {mu.max()*100:.3f}%')

# Ledoit-Wolf shrinkage covariance
lw = LedoitWolf().fit(ret_matrix.values)
Sigma = lw.covariance_
shrinkage = lw.shrinkage_
print(f'Ledoit-Wolf shrinkage intensity: {shrinkage:.4f}')


# =============================================================================
# 4.3  OPTIMIZE: MAXIMIZE SHARPE RATIO
#
#  Optimization problem:
#    max_{w}   (w'μ) / sqrt(w'Σw)
#
#  Equivalently (for numerical stability):
#    min_{w}  -w'μ / sqrt(w'Σw)
#
#  Constraints:
#    Σ w_i = 1
#    0 ≤ w_i ≤ 0.05 for all i
#
# =============================================================================
print('\n--- 4.3 Optimize portfolio (maximize Sharpe ratio) ---')

def neg_sharpe(w, mu, Sigma):
    """Negative Sharpe ratio (to minimize)."""
    port_ret = np.dot(w, mu)
    port_vol = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
    if port_vol < 1e-10:
        return 1e10
    return -port_ret / port_vol

n = len(mu)
w0 = np.ones(n) / n  # equal-weight initial guess
bounds = [(0.0, 0.05)] * n  # 0-5% per stock
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

result = minimize(
    neg_sharpe, w0, args=(mu, Sigma),
    method='SLSQP', bounds=bounds, constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-12}
)

w_opt = result.x
print(f'Optimization converged: {result.success}')


# =============================================================================
# 4.4  PORTFOLIO RESULTS
# =============================================================================
print('\n--- 4.4 Portfolio Results ---')

port_ret_monthly = np.dot(w_opt, mu)
port_vol_monthly = np.sqrt(np.dot(w_opt.T, np.dot(Sigma, w_opt)))
port_ret_annual = (1 + port_ret_monthly) ** 12 - 1
port_vol_annual = port_vol_monthly * np.sqrt(12)
sharpe_monthly = port_ret_monthly / port_vol_monthly
sharpe_annual = port_ret_annual / port_vol_annual

# Count non-zero positions
n_positions = (w_opt > 0.001).sum()

print(f'\nOptimal Portfolio Metrics:')
print(f'  Non-zero positions:     {n_positions}')
print(f'  Monthly return:         {port_ret_monthly*100:.3f}%')
print(f'  Annualized return:      {port_ret_annual*100:.2f}%')
print(f'  Monthly volatility:     {port_vol_monthly*100:.3f}%')
print(f'  Annualized volatility:  {port_vol_annual*100:.2f}%')
print(f'  Monthly Sharpe ratio:   {sharpe_monthly:.3f}')
print(f'  Annualized Sharpe ratio:{sharpe_annual:.3f}')

# Top holdings
print(f'\nTop 10 Holdings:')
top_idx = np.argsort(w_opt)[::-1][:10]
for rank, i in enumerate(top_idx, 1):
    permno = ret_matrix.columns[i]
    name = stock_names.get(permno, str(permno))
    print(f'  {rank:2d}. {name:35s}  Weight: {w_opt[i]*100:.2f}%')

# Save portfolio weights
port_weights = pd.DataFrame({
    'permno': ret_matrix.columns,
    'weight': w_opt,
    'name': [stock_names.get(p, str(p)) for p in ret_matrix.columns]
})
port_weights = port_weights[port_weights['weight'] > 0.001].sort_values('weight', ascending=False)
port_weights.to_csv(os.path.join(RESULTS_DIR, 'portfolio_weights.csv'), index=False)

# Save portfolio summary
summary = pd.DataFrame({
    'Metric': ['Stocks in universe', 'Non-zero positions', 'Monthly return',
               'Annualized return', 'Monthly volatility', 'Annualized volatility',
               'Monthly Sharpe', 'Annualized Sharpe', 'Max single-stock weight'],
    'Value': [n_stocks, n_positions, f'{port_ret_monthly*100:.3f}%',
              f'{port_ret_annual*100:.2f}%', f'{port_vol_monthly*100:.3f}%',
              f'{port_vol_annual*100:.2f}%', f'{sharpe_monthly:.3f}',
              f'{sharpe_annual:.3f}', f'{w_opt.max()*100:.2f}%']
})
summary.to_csv(os.path.join(RESULTS_DIR, 'portfolio_summary.csv'), index=False)
print('\nPortfolio results saved.')

elapsed = time.time() - t0
print(f'\nStep 4 completed in {elapsed/60:.1f} minutes')
print('=' * 80)
