# =============================================================================
#
#  MIS 502 — Final Project: Cross-Sectional Asset Pricing
#  CRSP/Compustat Data Pipeline
#
#  MASTER RUNNER — Execute all 4 steps in sequence
#
#  Usage:
#    python run_all.py           # Run all steps
#    python run_all.py --step 2  # Run only step 2 (requires step 1 output)
#
#  Prerequisites:
#    pip install pandas numpy scipy scikit-learn matplotlib seaborn
#
#  File Structure:
#    fa_project/
#    ├── data/
#    │   ├── raw/          ← Put your CSV files here
#    │   │   └── project_step1_monthly_master_dataset_accounting_matched (2).csv
#    │   ├── processed/    ← Created by Step 1
#    │   │   └── analysis_ready.csv
#    │   ├── results/      ← Created by Step 2 & 4
#    │   │   ├── model_comparison.csv
#    │   │   ├── predictions.csv
#    │   │   ├── feature_importances.csv
#    │   │   ├── factor_correlations.csv
#    │   │   ├── fama_macbeth_results.csv
#    │   │   ├── portfolio_weights.csv
#    │   │   └── portfolio_summary.csv
#    │   └── charts/       ← Created by Step 3
#    │       ├── 01_firm_count_area.png
#    │       ├── 02_return_distributions.png
#    │       ├── ... (13 total charts)
#    │       └── 13_momentum_spread_timeseries.png
#    └── scripts/
#        ├── run_all.py
#        ├── step1_data_wrangling.py
#        ├── step2_data_mining.py
#        ├── step3_visualization.py
#        └── step4_portfolio.py
#
# =============================================================================

import sys
import time

print('=' * 80)
print('CROSS-SECTIONAL ASSET PRICING PIPELINE')
print('MIS 502 — Final Project')
print('=' * 80)

t_start = time.time()

# Parse command line argument
step_to_run = None
if len(sys.argv) > 2 and sys.argv[1] == '--step':
    step_to_run = int(sys.argv[2])
    print(f'\nRunning only Step {step_to_run}')
else:
    print('\nRunning all steps (1 → 2 → 3 → 4)')

print('')

# Step 1: Data Wrangling
if step_to_run is None or step_to_run == 1:
    print('▶ Executing Step 1: Data Wrangling...\n')
    exec(open('step1_data_wrangling.py').read())
    print('')

# Step 2: Data Mining
if step_to_run is None or step_to_run == 2:
    print('▶ Executing Step 2: Data Mining...\n')
    exec(open('step2_data_mining.py').read())
    print('')

# Step 3: Visualization
if step_to_run is None or step_to_run == 3:
    print('▶ Executing Step 3: Visualization...\n')
    exec(open('step3_visualization.py').read())
    print('')

# Step 4: Portfolio Optimization
if step_to_run is None or step_to_run == 4:
    print('▶ Executing Step 4: Portfolio Optimization...\n')
    exec(open('step4_portfolio.py').read())
    print('')

total = time.time() - t_start
print('=' * 80)
print(f'PIPELINE COMPLETE — Total time: {total/60:.1f} minutes')
print('=' * 80)
