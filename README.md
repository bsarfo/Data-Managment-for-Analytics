1. Data Set Description

1.1 Nature and Context
This project constructs a complete quantitative equity research pipeline using institutional-grade financial data sourced from WRDS (Wharton Research Data Services). The dataset merges two foundational databases in empirical finance: the CRSP Monthly Stock File, which provides monthly returns, prices, and shares outstanding for all common stocks on NYSE, AMEX, and NASDAQ; and Compustat Annual Fundamentals, which provides annual balance sheet and income statement data including total assets, equity, sales, net income, operating income, debt, and capital expenditures.
The merge uses CUSIP-8 as the linking key with a 6-month availability lag on Compustat data (avail_date = datadate + 6 months) to prevent look-ahead bias. The dataset spans January 1990 through December 2023, covering 14,666 unique securities, 1,348,447 firm-month observations, and 37 raw variables. After cleaning and feature engineering, the panel expands to 1,312,284 observations with 65 columns including 11 engineered cross-sectional factors.
1.2 Potential Outcomes
Five analytical outcomes were targeted at the outset of this project:
•	Factor Discovery: Identify which firm characteristics predict future stock returns.
•	Return Prediction: Build machine learning models that forecast next-month returns with out-of-sample validity.
•	Portfolio Construction: Construct optimized long-short portfolios exploiting predicted return spreads.
•	Risk Characterization: Understand the factor structure of equity returns through dimensionality reduction and clustering.
•	Market Microstructure: Profile differences in return behavior across exchanges and firm size segments.
1.3 Data Quality, Integrity, and Ethics
Automated validation confirms zero duplicate firm-months, zero null-adjusted returns, zero non-positive market equity observations, and that all share codes and exchange codes are within valid ranges. Accounting variables show 1.3-11.9% missing values, with capital expenditures (capx) having the highest missing rate at 8.16%. The 176 observations with returns exceeding 300% in absolute value are addressed via winsorization. The 58,538 observations with negative book equity are retained but excluded from book-to-market calculations.
Data integrity is ensured through four safeguards: share code filtering (10, 11 only), exchange code filtering (1, 2, 3 only), the 6-month Compustat availability lag to prevent look-ahead bias, and delisting return adjustment to prevent survivorship bias.
Ethical considerations include: survivorship bias mitigation through delisting return inclusion; data snooping prevention via a strict temporal train/test split; reproducibility through WRDS academic access and published methodology (Fama and French, 1993); acknowledgement that widely adopted factor strategies may erode the patterns they exploit; and confirmation that the dataset contains only publicly available financial data with no personally identifiable information.
