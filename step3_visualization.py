# =============================================================================
#
#  MIS 502 — Final Project: Cross-Sectional Asset Pricing
#  CRSP/Compustat Data Pipeline
#
#  Script 3 of 4: DATA VISUALIZATION OUTCOMES
#
#  Charts produced:
#    (1) Firm Count Area Chart (by exchange, over time)
#    (2) Return Distribution Histograms (by exchange)
#    (3) Aggregate Market Capitalization (time series)
#    (4) Factor Correlation Heatmap
#    (5) Factor Quintile Bar Charts (momentum, value, size, profitability)
#    (6) Cumulative Market Return (log scale)
#    (7) Efficient Frontier Plot
#    (8) Feature Importance Comparison (Ridge, Lasso, RF)
#    (9) Predicted vs Actual Scatter Plot
#
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Set publication-quality style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox_inches': 'tight'
})

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
BASE_DIR = r'D:\4RM TAB~WPI\WPI SEM 3\DATA MANGEMENT FOR ANALYTICS\Datasets\DATA\MAIN DATA WITH DESCRIPTION\FINAL DATASET'
DATA_DIR = os.path.join(BASE_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CHART_DIR = os.path.join(BASE_DIR, 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

print('=' * 80)
print('STEP 3: DATA VISUALIZATION OUTCOMES')
print('=' * 80)

# Load data
print('\nLoading data...')
df = pd.read_csv(os.path.join(DATA_DIR, 'analysis_ready.csv'), parse_dates=['date'], low_memory=False)
print(f'Loaded {len(df):,} rows')

# Load predictions and feature importances
pred_df = pd.read_csv(os.path.join(RESULTS_DIR, 'predictions.csv'), parse_dates=['date'])
fi_df = pd.read_csv(os.path.join(RESULTS_DIR, 'feature_importances.csv'))
corr_df = pd.read_csv(os.path.join(RESULTS_DIR, 'factor_correlations.csv'), index_col=0)

exch_map = {1: 'NYSE', 2: 'AMEX', 3: 'NASDAQ'}
df['exchange'] = df['exchcd'].map(exch_map)


# =============================================================================
# CHART 1: FIRM COUNT AREA CHART
# =============================================================================
print('\n  (1) Firm Count Area Chart...')

df['year'] = df['date'].dt.year
firm_counts = df.groupby(['year', 'exchange'])['permno'].nunique().unstack(fill_value=0)
firm_counts = firm_counts[['NYSE', 'AMEX', 'NASDAQ']]

fig, ax = plt.subplots(figsize=(12, 6))
firm_counts.plot.area(ax=ax, alpha=0.7, color=['#2c7bb6', '#fdae61', '#d7191c'])
ax.set_title('Number of Unique Firms by Exchange Over Time')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Firms')
ax.legend(title='Exchange')
ax.set_xlim(firm_counts.index.min(), firm_counts.index.max())
plt.savefig(os.path.join(CHART_DIR, '01_firm_count_area.png'))
plt.close()
print('    Saved 01_firm_count_area.png')


# =============================================================================
# CHART 2: RETURN DISTRIBUTION HISTOGRAMS
# =============================================================================
print('  (2) Return Distribution Histograms...')

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for i, (exch_name, color) in enumerate([('NYSE', '#2c7bb6'), ('AMEX', '#fdae61'), ('NASDAQ', '#d7191c')]):
    data = df[df['exchange'] == exch_name]['ret_adj'].dropna()
    data = data[(data > -1) & (data < 1)]  # trim for visibility
    axes[i].hist(data, bins=100, alpha=0.7, color=color, density=True)
    axes[i].set_title(f'{exch_name}\nμ={data.mean():.4f}, σ={data.std():.4f}')
    axes[i].set_xlabel('Monthly Return')
    axes[i].axvline(data.mean(), color='black', linestyle='--', linewidth=1)
    if i == 0:
        axes[i].set_ylabel('Density')

fig.suptitle('Monthly Return Distributions by Exchange', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, '02_return_distributions.png'))
plt.close()
print('    Saved 02_return_distributions.png')


# =============================================================================
# CHART 3: AGGREGATE MARKET CAPITALIZATION
# =============================================================================
print('  (3) Aggregate Market Capitalization...')

mkt_cap = df.groupby('date')['me'].sum() / 1e6  # Convert to billions
fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(mkt_cap.index, mkt_cap.values, alpha=0.4, color='#2c7bb6')
ax.plot(mkt_cap.index, mkt_cap.values, linewidth=1, color='#2c7bb6')
ax.set_title('Aggregate Market Capitalization Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Market Cap ($ Billions)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}B'))
plt.savefig(os.path.join(CHART_DIR, '03_market_cap_timeseries.png'))
plt.close()
print('    Saved 03_market_cap_timeseries.png')


# =============================================================================
# CHART 4: FACTOR CORRELATION HEATMAP
# =============================================================================
print('  (4) Factor Correlation Heatmap...')

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_df, dtype=bool))  # upper triangle mask
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, ax=ax,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Factor Correlation Matrix (Lower Triangle)')
plt.savefig(os.path.join(CHART_DIR, '04_factor_correlation_heatmap.png'))
plt.close()
print('    Saved 04_factor_correlation_heatmap.png')


# =============================================================================
# CHART 5: FACTOR QUINTILE BAR CHARTS
# =============================================================================
print('  (5) Factor Quintile Bar Charts...')

factors_to_plot = [
    ('mom_12m', 'Momentum'),
    ('bm_w', 'Value (Book-to-Market)'),
    ('log_me_lag1', 'Size (Log Market Equity)'),
    ('prof_w', 'Profitability')
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (col, title) in enumerate(factors_to_plot):
    temp = df[['date', col, 'ret_adj']].dropna()
    temp['quintile'] = temp.groupby('date')[col].transform(
        lambda x: pd.qcut(x, 5, labels=[1,2,3,4,5], duplicates='drop')
    )
    q_ret = temp.groupby('quintile')['ret_adj'].mean() * 100  # to percentage

    colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
    axes[idx].bar(q_ret.index.astype(str), q_ret.values, color=colors, edgecolor='black', linewidth=0.5)
    axes[idx].set_title(f'{title}')
    axes[idx].set_xlabel('Quintile (1=Low, 5=High)')
    axes[idx].set_ylabel('Avg Monthly Return (%)')
    axes[idx].axhline(y=0, color='black', linewidth=0.5)

fig.suptitle('Average Monthly Returns by Factor Quintile', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, '05_quintile_bar_charts.png'))
plt.close()
print('    Saved 05_quintile_bar_charts.png')


# =============================================================================
# CHART 6: CUMULATIVE MARKET RETURN
# =============================================================================
print('  (6) Cumulative Market Return...')

# Value-weighted market return
df['me_weight'] = df.groupby('date')['me'].transform(lambda x: x / x.sum())
mkt_ret = df.groupby('date').apply(lambda g: (g['ret_adj'] * g['me_weight']).sum())
mkt_ret = mkt_ret.sort_index()
cum_ret = (1 + mkt_ret).cumprod()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(cum_ret.index, cum_ret.values, linewidth=1.5, color='#2c7bb6')
ax.set_yscale('log')
ax.set_title('Cumulative Value-Weighted Market Return (Log Scale)')
ax.set_xlabel('Date')
ax.set_ylabel('Growth of $1 (Log Scale)')
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(CHART_DIR, '06_cumulative_market_return.png'))
plt.close()
print('    Saved 06_cumulative_market_return.png')


# =============================================================================
# CHART 7: EFFICIENT FRONTIER
# =============================================================================
print('  (7) Efficient Frontier Plot...')

# Use top 50 stocks by market cap for tractability
top_stocks = df[df['date'] == df['date'].max()].nlargest(50, 'me')['permno'].values
port_data = df[df['permno'].isin(top_stocks)].pivot_table(
    index='date', columns='permno', values='ret_adj'
).dropna(axis=1)

if port_data.shape[1] >= 10:
    mean_ret = port_data.mean() * 12  # annualize
    cov_matrix = port_data.cov() * 12

    n_portfolios = 5000
    results = np.zeros((3, n_portfolios))
    np.random.seed(42)

    for i in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(len(mean_ret)))
        port_ret = np.dot(weights, mean_ret)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0, i] = port_vol
        results[1, i] = port_ret
        results[2, i] = port_ret / port_vol  # Sharpe

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(results[0], results[1], c=results[2], cmap='viridis',
                         alpha=0.5, s=5)
    plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')

    # Highlight max Sharpe
    max_sharpe_idx = results[2].argmax()
    ax.scatter(results[0, max_sharpe_idx], results[1, max_sharpe_idx],
               marker='*', s=300, color='red', edgecolors='black', linewidth=1.5,
               label=f'Max Sharpe ({results[2, max_sharpe_idx]:.2f})')

    ax.set_title('Efficient Frontier (Monte Carlo Simulation)')
    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CHART_DIR, '07_efficient_frontier.png'))
    plt.close()
    print('    Saved 07_efficient_frontier.png')
else:
    print('    Insufficient data for efficient frontier')


# =============================================================================
# CHART 8: FEATURE IMPORTANCE COMPARISON
# =============================================================================
print('  (8) Feature Importance Charts...')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Normalize each to sum to 1 for comparison
for idx, (col, title) in enumerate([
    ('Ridge_Coef', 'Ridge (|Coefficients|)'),
    ('Lasso_Coef', 'Lasso (|Coefficients|)'),
    ('RF_Importance', 'Random Forest (Importance)')
]):
    sorted_fi = fi_df.sort_values(col, ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_fi)))
    axes[idx].barh(sorted_fi['Factor'], sorted_fi[col], color=colors)
    axes[idx].set_title(title)
    axes[idx].set_xlabel('Importance')

fig.suptitle('Feature Importance Across Models', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, '08_feature_importance.png'))
plt.close()
print('    Saved 08_feature_importance.png')


# =============================================================================
# CHART 9: PREDICTED VS ACTUAL SCATTER
# =============================================================================
print('  (9) Predicted vs Actual Scatter...')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models = [('pred_ridge', 'Ridge'), ('pred_lasso', 'Lasso'), ('pred_rf', 'Random Forest')]

for idx, (pred_col, name) in enumerate(models):
    # Sample for visibility
    sample = pred_df.sample(min(10000, len(pred_df)), random_state=42)
    axes[idx].scatter(sample[pred_col], sample['ret_actual'], alpha=0.1, s=3, color='#2c7bb6')
    axes[idx].set_title(f'{name}')
    axes[idx].set_xlabel('Predicted Return')
    axes[idx].set_ylabel('Actual Return')

    # Add 45-degree line
    lim = max(abs(sample[pred_col].max()), abs(sample['ret_actual'].max()))
    lim = min(lim, 0.5)
    axes[idx].plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, alpha=0.5)
    axes[idx].set_xlim(-lim, lim)
    axes[idx].set_ylim(-lim, lim)
    axes[idx].grid(True, alpha=0.3)

fig.suptitle('Predicted vs Actual Monthly Returns', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, '09_predicted_vs_actual.png'))
plt.close()
print('    Saved 09_predicted_vs_actual.png')


# =============================================================================
# BONUS CHARTS: BOX PLOT AND PIE CHART (Course Requirements)
# =============================================================================
print('  (10) Box Plot - Returns by Exchange...')

fig, ax = plt.subplots(figsize=(10, 6))
plot_data = df[df['ret_adj'].between(-0.5, 0.5)]
sns.boxplot(data=plot_data, x='exchange', y='ret_adj', ax=ax,
            palette=['#2c7bb6', '#fdae61', '#d7191c'],
            order=['NYSE', 'AMEX', 'NASDAQ'])
ax.set_title('Monthly Return Distribution by Exchange')
ax.set_xlabel('Exchange')
ax.set_ylabel('Monthly Return')
plt.savefig(os.path.join(CHART_DIR, '10_boxplot_returns_by_exchange.png'))
plt.close()
print('    Saved 10_boxplot_returns_by_exchange.png')

print('  (11) Pie Chart - Exchange Composition...')

exch_counts = df['exchange'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#2c7bb6', '#d7191c', '#fdae61']
ax.pie(exch_counts.values, labels=exch_counts.index, autopct='%1.1f%%',
       colors=colors, startangle=90, textprops={'fontsize': 12})
ax.set_title('Observations by Exchange')
plt.savefig(os.path.join(CHART_DIR, '11_pie_exchange_composition.png'))
plt.close()
print('    Saved 11_pie_exchange_composition.png')

print('  (12) Line Chart - Average Return Over Time...')

avg_ret = df.groupby('date')['ret_adj'].mean()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(avg_ret.index, avg_ret.values * 100, linewidth=0.8, color='#2c7bb6')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title('Average Monthly Cross-Sectional Return Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Average Return (%)')
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(CHART_DIR, '12_line_avg_return.png'))
plt.close()
print('    Saved 12_line_avg_return.png')

print('  (13) Multi-line Chart - Factor Quintile Spreads Over Time...')

# Momentum quintile spread over time
temp = df[['date', 'mom_12m', 'ret_adj']].dropna()
temp['quintile'] = temp.groupby('date')['mom_12m'].transform(
    lambda x: pd.qcut(x, 5, labels=[1,2,3,4,5], duplicates='drop')
)
q_time = temp.groupby(['date', 'quintile'])['ret_adj'].mean().unstack()
if 1 in q_time.columns and 5 in q_time.columns:
    spread = (q_time[5] - q_time[1]).rolling(12).mean() * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(spread.index, spread.values, linewidth=1.5, color='#1a9641')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.fill_between(spread.index, spread.values, alpha=0.2, color='#1a9641')
    ax.set_title('Momentum Factor Spread (Q5-Q1, 12-Month Rolling Average)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Monthly Spread (%)')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CHART_DIR, '13_momentum_spread_timeseries.png'))
    plt.close()
    print('    Saved 13_momentum_spread_timeseries.png')


print('\n' + '=' * 80)
print(f'All charts saved to: {CHART_DIR}')
print('=' * 80)
