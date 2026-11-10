# E-Commerce Customer Analytics & Predictive Customer Lifetime Value (CLV) Engine

## Project Overview

This repository contains an end-to-end data science and machine learning pipeline designed for e-commerce businesses to analyze customer behavior, quantify historical monetary contribution, and accurately predict 365-day forward Customer Lifetime Value (CLV). 

By combining RFM (Recency, Frequency, Monetary) behavioral segmentation, heuristic valuation models, and gradient boosting regression algorithms, this system provides actionable intelligence for targeted customer retention, marketing budget allocation, and churn prevention.

---

## Main Challenge

Mid-sized e-commerce and retail operations routinely track aggregate performance indicators such as gross revenue, order volume, and average basket size. However, leadership often lacks granular visibility into customer distribution—specifically:

1. **Revenue Concentration Risk:** Identifies which customer cohorts generate the disproportionate share of business revenue.
2. **Early Signal Detection:** Detects high-potential customers early in their tenure based on transaction cadence and item diversity.
3. **Revenue Exposure to Churn:** Quantifies total revenue at risk from customers exhibiting increasing recency gaps.
4. **Predictive Valuation:** Formulates realistic, forward-looking 365-day CLV expectations at an individual customer level.
5. **Resource Prioritization:** Reallocates limited marketing capital toward accounts yielding the highest return on investment.

---

## Track Alignment

This project is built for the **Predictive Analytics & Customer Intelligence Track**. It addresses real-world transactional datasets by converting unstructured purchase logs into structured behavioral metrics, supervised training datasets, and practical prioritization strategies.

---

## Project Description

### Problem Statement
Aggregated sales dashboards obscure underlying customer behavior. Treating all customers uniformly leads to sub-optimal marketing spend—over-discounting low-intent buyers while under-investing in high-value accounts at risk of churning.

### Technical Solution
The engine processes raw transaction records (34,500+ records across 17 features) through a multi-stage analytical pipeline:
* Data sanitization, type coercion, and metric extraction.
* RFM behavioral segmentation and customer tenure analysis.
* Supervised regression target engineering using a 365-day observation and prediction window split.
* Ensemble machine learning (Random Forest, Gradient Boosting, LightGBM, XGBoost) to model future customer spend.
* Systematic residual error analysis across customer spend tiers.
* Prioritization framework implementation using a composite score of predicted CLV and retention opportunity.

### Target Users
* **Growth Marketing Teams:** To design targeted campaigns based on behavioral tiers rather than broad demographics.
* **Customer Success & Retention Leads:** To deploy intervention triggers for high-value accounts showing churn indicators.
* **E-Commerce Operations & Leadership:** To forecast future quarterly and annual revenue streams based on forward-looking CLV estimates.

### Business Impact
* **Capital Efficiency:** Reduces customer acquisition and re-engagement costs by directing budget exclusively to top-tier accounts.
* **Proactive Retention:** Mitigates revenue loss by systematically identifying high-value accounts before full dormancy occurs.
* **Data-Driven Strategy:** Replaces intuition with quantifiable predictive risk metrics.

---

## Technical Pipeline & Workflow Steps

The repository implements a sequential 7-stage analytical pipeline:

1. **Part 1: Setup, Data Loading, & Data Dictionary**
   * Initializes environment dependencies (`pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `seaborn`, `matplotlib`).
   * Loads transactional dataset and builds standard data dictionary mappings across all retained features.

2. **Part 2: Data Cleaning & Exploratory Data Analysis (EDA)**
   * Performs deduplication, data type coercion (datetime, numeric standardizations), and missing-value handling.
   * Conducts temporal sales trend analysis, category revenue decomposition, and transaction magnitude distribution checks.

3. **Part 3: Behavioral Feature Engineering & RFM Analysis**
   * Aggregates line-item transactions into customer-level metrics: Recency, Frequency, Monetary Value, Average Order Value (AOV), Purchase Interval, Product Diversity, and Tenure.
   * Constructs RFM percentile quantile scores (1–5 scale) and maps accounts to rule-based segments (*Champions*, *Loyal Customers*, *Potential Loyalists*, *At Risk*, *Hibernating*, *New Customers*).

4. **Part 4: Heuristic CLV & Temporal Data Split**
   * Calculates baseline heuristic CLV using classic unit-economic formulas incorporating historical margin and lifespan estimates.
   * Establishes a strict cutoff date to split historical transactions into an **Observation Window** (for feature extraction) and a **Prediction Window** (for target variable derivation: 365-day future revenue).

5. **Part 5: Supervised Model Training & Evaluation**
   * Trains four distinct regression algorithms: Random Forest, Gradient Boosting, LightGBM, and XGBoost.
   * Evaluates performance using standard regression evaluation metrics: Mean Absolute Error (**MAE**), Root Mean Squared Error (**RMSE**), Coefficient of Determination (**R²**), and Mean Absolute Percentage Error (**MAPE**).
   * Extracts feature importance metrics to identify drivers of long-term spend.

6. **Part 6: Value Tiering, Algorithmic Clustering, & Error Analysis**
   * Converts continuous predictions into discrete business tiers: *Platinum (Top 15%)*, *Gold (15%-50%)*, *Silver (50%-80%)*, and *Bronze (Bottom 20%)*.
   * Compares heuristic RFM segmentation against unsupervised K-Means clustering.
   * Conducts systematic residual investigation across spend cohorts to identify model failure patterns (e.g., high-spender under-prediction).

7. **Part 7: Business Intelligence Dashboard & Prioritization Framework**
   * Builds multi-panel visual reporting layouts for executive review.
   * Computes a prioritized engagement score: $\text{Priority Score} = \text{Predicted CLV} \times \text{Retention Opportunity}$.
   * Formulates actionable strategic recommendations per cohort.

---

## Key Results & Model Performance Metrics

| Model | MAE ($) | RMSE ($) | R² Score | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **Primary Baseline** | **Top Performer** | **Highest Variance Explained** | **Optimal** |
| LightGBM Regressor | Competitive | Competitive | High | Scalable |
| Gradient Boosting | Robust | Robust | Moderate-High | Standard |
| Random Forest | Baseline | Baseline | Baseline | Baseline |

### Core Findings
* **Top Predictors:** Historical monetary value, purchase frequency, and recency are the primary drivers of 365-day forward value.
* **Revenue Concentration:** The top 15% of accounts (Platinum tier) represent over 50% of projected total future revenue.
* **Residual Behavior:** Linear scaling over-predicts low-tenure accounts; tree-based models capture non-linear caps effectively but require monitoring at extreme tail boundaries.

---

## Tech Stack

* **Language:** Python 3.10+
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning & Modeling:** `scikit-learn`, `xgboost`, `lightgbm`

---

## Project Structure

```
├── data/
│   └── ecommerce_sales.csv            # Raw sales transaction dataset
├── charts/
├── dashboard/
└──  clv_pipeline_notebook.ipynb    # Sequential Kaggle/Jupyter notebook
└── README.md
```

---

## Strategic Business Action Plan

1. **Platinum Tier (Top 15% Spend Drivers):** Enroll in proactive VIP retention management, priority service support, and targeted product previews to preserve high-margin core revenue.
2. **At-Risk High-Monetary Segment:** Trigger automated re-engagement notifications when purchase interval recency exceeds twice the account's historical mean.
3. **Single-Purchase Onboarding:** Deploy 14-day post-purchase cross-sell campaigns to convert single-purchase transactions into habituated buyers.
