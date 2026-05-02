# =============================================================================
#
#  MIS 502 — Final Project: Cross-Sectional Asset Pricing
#  CRSP/Compustat Data Pipeline
#
#  Script 2 of 4: DATA MINING OUTCOMES
#
#  Techniques implemented:
#    (1) Cluster Analysis (KMeans on factor characteristics)
#    (2) Ridge Regression (L2-penalized return prediction)
#    (3) Lasso Regression (L1-penalized, automatic variable selection)
#    (4) Random Forest (non-linear ensemble)
#    (5) Fama-MacBeth Cross-Sectional Regression
#    (6) Portfolio Quintile Sort Tests
#    (7) Correlation / Association Analysis
#    (8) PCA Dimensionality Reduction
#
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
BASE_DIR = r'D:\4RM TAB~WPI\WPI SEM 3\DATA MANGEMENT FOR ANALYTICS\Datasets\DATA\MAIN DATA WITH DESCRIPTION\FINAL DATASET'
DATA_DIR = os.path.join(BASE_DIR, 'processed')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
CHART_DIR = os.path.join(BASE_DIR, 'charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, 'analysis_ready.csv')

print('=' * 80)
print('STEP 2: DATA MINING OUTCOMES')
print('=' * 80)
t0 = time.time()


# =============================================================================
# 2.0  LOAD ANALYSIS-READY DATA
# =============================================================================
print('\n--- 2.0 Loading analysis-ready data ---')

df = pd.read_csv(INPUT_FILE, parse_dates=['date'], low_memory=False)
print(f'Loaded {len(df):,} rows x {df.shape[1]} columns')

# Define feature columns (rank-transformed factors)
FEATURES = ['mom_12m_rank', 'reversal_rank', 'log_me_lag1_rank', 'bm_w_rank',
            'prof_w_rank', 'inv_w_rank', 'leverage_w_rank', 'ep_w_rank',
            'beta_rank', 'volatility_rank']

FEATURE_NAMES = ['Momentum', 'Reversal', 'Size', 'Value (B/M)', 'Profitability',
                 'Investment', 'Leverage', 'Earnings/Price', 'Beta', 'Volatility']

TARGET = 'ret_next'

# Separate train and test
train = df[df['split'] == 'train'].copy()
test = df[df['split'] == 'test'].copy()
print(f'Training: {len(train):,} | Test: {len(test):,}')

# Drop rows with missing features or target
train_clean = train.dropna(subset=FEATURES + [TARGET])
test_clean = test.dropna(subset=FEATURES + [TARGET])
print(f'After dropping NaN — Training: {len(train_clean):,} | Test: {len(test_clean):,}')

X_train = train_clean[FEATURES].values
y_train = train_clean[TARGET].values
X_test = test_clean[FEATURES].values
y_test = test_clean[TARGET].values


# =============================================================================
# 2.1  CLUSTER ANALYSIS (KMeans)
#
#  Group stocks into K clusters based on their factor characteristics.
#  Use silhouette score to determine optimal K.
#
# =============================================================================
print('\n--- 2.1 Cluster Analysis ---')

from sklearn.metrics import silhouette_score

# Use a sample for efficiency (100k observations)
np.random.seed(42)
sample_idx = np.random.choice(len(X_train), size=min(100000, len(X_train)), replace=False)
X_cluster = X_train[sample_idx]

# Remove any remaining NaN
mask = ~np.isnan(X_cluster).any(axis=1)
X_cluster = X_cluster[mask]

print('Silhouette scores by K:')
best_k, best_score = 2, -1
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
    labels = km.fit_predict(X_cluster)
    score = silhouette_score(X_cluster, labels, sample_size=10000)
    print(f'  K = {k}: Silhouette = {score:.4f}')
    if score > best_score:
        best_k, best_score = k, score

print(f'\nBest K = {best_k} (Silhouette = {best_score:.4f})')

# Fit final model with best K
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_final.fit(X_cluster)
print(f'\nCluster centroids (each row = cluster center across {len(FEATURES)} factors):')
centroid_df = pd.DataFrame(km_final.cluster_centers_, columns=FEATURE_NAMES)
print(centroid_df.round(3))

# Cluster sizes
labels_final = km_final.predict(X_cluster)
print('\nCluster sizes:')
for i in range(best_k):
    print(f'  Cluster {i}: {(labels_final == i).sum():,} observations')


# =============================================================================
# 2.2  RIDGE REGRESSION
#
#  Mathematical model:
#    min_{β}  Σ (y_i - X_i β)² + α ||β||₂²
#
#  The L2 penalty shrinks coefficients toward zero without eliminating
#  any, making it suitable for multicollinear factor data.
#
# =============================================================================
print('\n--- 2.2 Ridge Regression ---')

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Metrics
oos_r2_ridge = 1 - np.sum((y_test - y_pred_ridge)**2) / np.sum((y_test - y_test.mean())**2)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
hit_rate_ridge = np.mean(np.sign(y_pred_ridge) == np.sign(y_test))

print(f'Ridge Regression Results:')
print(f'  Out-of-Sample R²:  {oos_r2_ridge*100:.4f}%')
print(f'  RMSE:              {rmse_ridge:.6f}')
print(f'  MAE:               {mae_ridge:.6f}')
print(f'  Hit Rate:          {hit_rate_ridge*100:.2f}%')

print('\nRidge Coefficients:')
for name, coef in zip(FEATURE_NAMES, ridge.coef_):
    print(f'  {name:20s}: {coef:.6f}')


# =============================================================================
# 2.3  LASSO REGRESSION
#
#  Mathematical model:
#    min_{β}  Σ (y_i - X_i β)² + α ||β||₁
#
#  The L1 penalty drives some coefficients to exactly zero,
#  performing automatic variable selection.
#
# =============================================================================
print('\n--- 2.3 Lasso Regression ---')

lasso = Lasso(alpha=0.0001)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

oos_r2_lasso = 1 - np.sum((y_test - y_pred_lasso)**2) / np.sum((y_test - y_test.mean())**2)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
hit_rate_lasso = np.mean(np.sign(y_pred_lasso) == np.sign(y_test))

print(f'Lasso Regression Results:')
print(f'  Out-of-Sample R²:  {oos_r2_lasso*100:.4f}%')
print(f'  RMSE:              {rmse_lasso:.6f}')
print(f'  MAE:               {mae_lasso:.6f}')
print(f'  Hit Rate:          {hit_rate_lasso*100:.2f}%')

print('\nLasso Coefficients (non-zero = selected variables):')
for name, coef in zip(FEATURE_NAMES, lasso.coef_):
    marker = '***' if abs(coef) > 0 else '   '
    print(f'  {name:20s}: {coef:.6f} {marker}')
n_selected = (np.abs(lasso.coef_) > 0).sum()
print(f'Variables selected by Lasso: {n_selected} of {len(FEATURES)}')


# =============================================================================
# 2.4  RANDOM FOREST REGRESSION
#
#  Ensemble of decision trees with bagging and random feature subsets.
#  Captures non-linear interactions between factors.
#
#  Parameters:
#    n_estimators = 200 trees
#    max_depth = 6 (shallow trees to prevent overfitting)
#    min_samples_leaf = 100
#
# =============================================================================
print('\n--- 2.4 Random Forest Regression ---')

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

oos_r2_rf = 1 - np.sum((y_test - y_pred_rf)**2) / np.sum((y_test - y_test.mean())**2)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
hit_rate_rf = np.mean(np.sign(y_pred_rf) == np.sign(y_test))

print(f'Random Forest Results:')
print(f'  Out-of-Sample R²:  {oos_r2_rf*100:.4f}%')
print(f'  RMSE:              {rmse_rf:.6f}')
print(f'  MAE:               {mae_rf:.6f}')
print(f'  Hit Rate:          {hit_rate_rf*100:.2f}%')

print('\nFeature Importance (Random Forest):')
for name, imp in sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda x: -x[1]):
    print(f'  {name:20s}: {imp:.4f}')


# =============================================================================
# 2.5  MODEL COMPARISON SUMMARY
# =============================================================================
print('\n--- 2.5 Model Comparison Summary ---')

results = pd.DataFrame({
    'Model': ['Ridge', 'Lasso', 'Random Forest'],
    'OOS_R2_pct': [oos_r2_ridge*100, oos_r2_lasso*100, oos_r2_rf*100],
    'RMSE': [rmse_ridge, rmse_lasso, rmse_rf],
    'MAE': [mae_ridge, mae_lasso, mae_rf],
    'Hit_Rate_pct': [hit_rate_ridge*100, hit_rate_lasso*100, hit_rate_rf*100],
})
print(results.to_string(index=False))
results.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)


# =============================================================================
# 2.6  PORTFOLIO QUINTILE SORT TESTS
#
#  For each model, sort stocks into quintiles by predicted return
#  each month. The long-short spread (Q5 - Q1) tests whether
#  predictions have economic content.
#
#  t-statistic:  t = mean(spread) / (std(spread) / sqrt(T))
#
# =============================================================================
print('\n--- 2.6 Portfolio Quintile Sort Tests ---')

def quintile_sort_test(predictions, returns, dates, model_name):
    """Sort stocks into quintiles by prediction, compute L/S spread."""
    temp = pd.DataFrame({
        'date': dates,
        'pred': predictions,
        'ret': returns
    })
    temp = temp.dropna()

    # Assign quintiles within each month
    temp['quintile'] = temp.groupby('date')['pred'].transform(
        lambda x: pd.qcut(x, 5, labels=[1,2,3,4,5], duplicates='drop')
    )
    temp['quintile'] = temp['quintile'].astype(float)

    # Average return by quintile per month
    q_ret = temp.groupby(['date', 'quintile'])['ret'].mean().unstack()

    if 1.0 in q_ret.columns and 5.0 in q_ret.columns:
        spread = q_ret[5.0] - q_ret[1.0]
        mean_spread = spread.mean()
        t_stat = mean_spread / (spread.std() / np.sqrt(len(spread)))

        print(f'\n  {model_name} Quintile Returns (monthly avg):')
        for q in sorted(q_ret.columns):
            print(f'    Q{int(q)}: {q_ret[q].mean()*100:.3f}%')
        print(f'    L/S Spread (Q5-Q1): {mean_spread*100:.3f}% per month')
        print(f'    t-statistic:        {t_stat:.2f}')
        return mean_spread, t_stat
    return None, None

# Run sort tests for each model
test_dates = test_clean['date'].values
spread_ridge, t_ridge = quintile_sort_test(y_pred_ridge, y_test, test_dates, 'Ridge')
spread_lasso, t_lasso = quintile_sort_test(y_pred_lasso, y_test, test_dates, 'Lasso')
spread_rf, t_rf = quintile_sort_test(y_pred_rf, y_test, test_dates, 'Random Forest')


# =============================================================================
# 2.7  CORRELATION / ASSOCIATION ANALYSIS
# =============================================================================
print('\n--- 2.7 Correlation Analysis ---')

# Factor correlation matrix
factor_raw_cols = ['mom_12m', 'reversal', 'log_me_lag1', 'bm_w', 'prof_w',
                   'inv_w', 'leverage_w', 'ep_w', 'beta', 'volatility']
corr_data = train_clean[factor_raw_cols].dropna()
corr_matrix = corr_data.corr()

print('Factor Correlation Matrix:')
# Rename for display
corr_display = corr_matrix.copy()
corr_display.index = FEATURE_NAMES
corr_display.columns = FEATURE_NAMES
print(corr_display.round(3))
corr_display.to_csv(os.path.join(OUTPUT_DIR, 'factor_correlations.csv'))


# =============================================================================
# 2.8  PCA DIMENSIONALITY REDUCTION
#
#  Compress 10 factors into fewer principal components.
#  Compare predictive R² with and without PCA.
#
#  Mathematical model:
#    X_reduced = X · W_k   where W_k are the top-k eigenvectors
#                           of the covariance matrix Σ = X'X / n
#
# =============================================================================
print('\n--- 2.8 PCA Dimensionality Reduction ---')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for n_comp in [3, 5, 7]:
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    variance_explained = pca.explained_variance_ratio_.sum()

    ridge_pca = Ridge(alpha=1.0)
    ridge_pca.fit(X_train_pca, y_train)
    y_pred_pca = ridge_pca.predict(X_test_pca)
    r2_pca = 1 - np.sum((y_test - y_pred_pca)**2) / np.sum((y_test - y_test.mean())**2)

    print(f'  PCA({n_comp}): Variance explained = {variance_explained*100:.1f}%, '
          f'OOS R² = {r2_pca*100:.4f}%')

# Compare to non-PCA
print(f'  No PCA (10 factors): OOS R² = {oos_r2_ridge*100:.4f}%')


# =============================================================================
# 2.9  FAMA-MACBETH CROSS-SECTIONAL REGRESSION
#
#  For each month t:
#    r_{i,t} = α_t + Σ_k γ_{k,t} · f_{k,i,t} + ε_{i,t}
#
#  Then average the coefficients across months:
#    γ̄_k = (1/T) Σ_t γ_{k,t}
#
#  t-statistic:  t_k = γ̄_k / (σ(γ_k) / √T)
#
# =============================================================================
print('\n--- 2.9 Fama-MacBeth Cross-Sectional Regression ---')

fmb_data = train_clean[['date', 'ret_adj'] + [f for f in FEATURES]].dropna()

# Run cross-sectional regression for each month
months = sorted(fmb_data['date'].unique())
gamma_list = []

for month in months:
    month_data = fmb_data[fmb_data['date'] == month]
    if len(month_data) < 50:
        continue
    X_m = month_data[FEATURES].values
    y_m = month_data['ret_adj'].values
    try:
        coefs = np.linalg.lstsq(
            np.column_stack([np.ones(len(X_m)), X_m]), y_m, rcond=None
        )[0]
        gamma_list.append(coefs[1:])  # exclude intercept
    except:
        continue

gammas = np.array(gamma_list)
mean_gamma = gammas.mean(axis=0)
std_gamma = gammas.std(axis=0)
t_stats = mean_gamma / (std_gamma / np.sqrt(len(gammas)))

print(f'\nFama-MacBeth Results ({len(gammas)} months):')
print(f'{"Factor":20s} {"Mean Coef":>12s} {"Std":>10s} {"t-stat":>10s} {"Signif":>8s}')
print('-' * 65)
for name, mg, sg, ts in zip(FEATURE_NAMES, mean_gamma, std_gamma, t_stats):
    sig = '***' if abs(ts) > 2.58 else '**' if abs(ts) > 1.96 else '*' if abs(ts) > 1.64 else ''
    print(f'{name:20s} {mg:12.6f} {sg:10.6f} {ts:10.2f} {sig:>8s}')

# Save results
fmb_results = pd.DataFrame({
    'Factor': FEATURE_NAMES,
    'Mean_Coefficient': mean_gamma,
    'Std_Coefficient': std_gamma,
    't_statistic': t_stats
})
fmb_results.to_csv(os.path.join(OUTPUT_DIR, 'fama_macbeth_results.csv'), index=False)


# =============================================================================
# SAVE PREDICTIONS FOR VISUALIZATION SCRIPT
# =============================================================================
print('\n--- Saving predictions for visualization ---')

pred_df = pd.DataFrame({
    'date': test_clean['date'].values,
    'permno': test_clean['permno'].values,
    'ret_actual': y_test,
    'pred_ridge': y_pred_ridge,
    'pred_lasso': y_pred_lasso,
    'pred_rf': y_pred_rf,
})
pred_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)

# Save feature importances
fi_df = pd.DataFrame({
    'Factor': FEATURE_NAMES,
    'Ridge_Coef': np.abs(ridge.coef_),
    'Lasso_Coef': np.abs(lasso.coef_),
    'RF_Importance': rf.feature_importances_,
})
fi_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importances.csv'), index=False)

elapsed = time.time() - t0
print(f'\nStep 2 completed in {elapsed/60:.1f} minutes')
print('=' * 80)
