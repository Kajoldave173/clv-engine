# Model Card — CLV Prediction & Segmentation Engine

## Model Details

**Project:** Customer Lifetime Value Prediction & Segmentation Engine (`clv-engine`)
**Version:** 1.0
**Date:** July 2026
**Author:** Kajol Dave
**Repository:** [github.com/Kajoldave173/clv-engine](https://github.com/Kajoldave173/clv-engine)
**Live App:** [clv-engine-kajol-dave-project.streamlit.app](https://clv-engine-kajol-dave-project.streamlit.app/)
**Experiment Tracking:** [DagsHub MLflow](https://dagshub.com/kajoldave8/clv-engine.mlflow) (run: `best-models-final`)

### Architecture

The system uses a two-model approach combining probabilistic and machine learning methods:

1. **Probabilistic Baseline (BG/NBD + Gamma-Gamma):** Estimates customer transaction rates, dropout probabilities, and expected monetary value using only frequency, recency, and customer age. Serves as the default prediction for customers with thin behavioral history.

2. **ML Models (XGBoost, Optuna-tuned):**
   - **CLV Regressor:** Predicts 6-month-ahead customer revenue using 16 behavioral features plus probabilistic model outputs as inputs (stacking pattern).
   - **Churn Classifier:** Predicts binary churn (zero purchases in holdout) using the same feature set, improving on the BG/NBD P(alive) baseline.

3. **Behavioral Segmentation (K-Means, k=5):** Discovers natural customer groupings from behavioral features independent of CLV, then overlays CLV and churn predictions for actionable profiling.

### Hyperparameters (Optuna-tuned, best-models-final)

**CLV Regressor (XGBoost):**
- Best iteration: 73
- Tuned via Optuna cross-validation

**Churn Classifier (XGBoost):**
- n_estimators: 132
- max_depth: 4
- learning_rate: 0.0851
- subsample: 0.857
- colsample_bytree: 0.905
- min_child_weight: 2
- reg_alpha: 0.00287
- reg_lambda: 5.34e-08

---

## Intended Use

**Primary use cases:**
- Predict which customers will generate the most revenue in the next 6 months, enabling marketing budget allocation toward high-value retention.
- Identify at-risk customers before they churn, enabling targeted intervention campaigns.
- Segment the customer base into behaviorally distinct groups for differentiated marketing strategies.
- Estimate ROI of retention campaigns via the What-If Simulator (e.g., "if purchase frequency increases 20%, predicted CLV increases by £X").

**Intended users:** Marketing teams, customer success managers, CRM analysts, and business stakeholders evaluating customer portfolio health.

**Out-of-scope uses:**
- Individual credit or lending decisions.
- Real-time scoring (the system is designed for batch scoring, not sub-second latency).
- Markets or customer bases structurally different from UK wholesale e-commerce (see Limitations).

---

## Training Data

**Dataset:** UCI Online Retail II
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
**Domain:** UK-based online/wholesale retailer selling gifts, homeware, and seasonal items.
**Raw size:** ~1,046,000 transaction rows across two Excel sheets (Dec 2009 – Dec 2011).
**After cleaning:** ~530,000 rows, 5,861 unique customers.

**Temporal split:**
- **Calibration period:** December 2009 – June 2011 (~18 months) — 5,023 customers used for training.
- **Holdout period:** July 2011 – December 2011 (~6 months) — 3,358 customers with holdout activity, used for validation.

**Cleaning steps applied:**
- Dropped rows with missing Customer ID (~25%, guest checkouts).
- Removed cancellations (invoices starting with 'C') and negative-quantity returns.
- Filtered non-product stock codes (POST, DOT, BANK CHARGES, M, D, etc.).
- Dropped rows with Price ≤ 0.
- Removed exact duplicate rows.
- All decisions logged in `reports/cleaning_log.txt`.

**Validation:** 19 automated data quality checks run before any modeling (null checks, value range checks, date range checks, customer count sanity, revenue sanity, dtype checks, split integrity).

---

## Features

The ML models use 16 behavioral features engineered from calibration-period transactions, plus 2 probabilistic model outputs (23 total with RFM inputs):

**RFM inputs:** frequency, recency, T, monetary_value

**Probabilistic stacking inputs:** predicted_purchases (BG/NBD), p_alive, expected_avg_value (Gamma-Gamma)

**Temporal purchase patterns:** inter_purchase_time_mean, inter_purchase_time_std, inter_purchase_time_trend, days_since_last_purchase, purchase_velocity_recent_vs_early

**Basket and product patterns:** avg_basket_size, avg_basket_value, basket_size_trend, category_breadth, category_concentration

**Monetary trends:** monetary_trend, max_single_transaction, monetary_cv

**Engagement trajectory:** lifecycle_stage, weekend_purchase_ratio

---

## Evaluation Metrics

### CLV Regressor

| Metric | Probabilistic Baseline | XGBoost (Phase 3) | XGBoost (Optuna-tuned) |
|--------|----------------------|-------------------|----------------------|
| MAE (£) | 672.98 | 509.44 | **334.15** |
| RMSE (£) | — | — | 836.54 |
| R² | — | 0.836 | **0.965** |
| MAPE | — | — | 0.763 |

The Optuna-tuned XGBoost reduces MAE by **50.3%** compared to the probabilistic baseline and by **34.4%** compared to the untuned XGBoost from Phase 3.

### Churn Classifier

| Metric | P(alive) Baseline | XGBoost (Optuna-tuned) |
|--------|------------------|----------------------|
| AUC-ROC | 0.452 | **0.855** |
| F1 | 0.396 | **0.783** |
| AUC Improvement | — | **89.1%** |

The BG/NBD P(alive) performs near-random on this dataset (AUC 0.452) because it is overoptimistic — mean P(alive) of 0.904 for a population with 48.2% actual churn. The ML classifier captures behavioral signals (purchase velocity decline, return behavior, category concentration) that the probabilistic model cannot.

### Model Selection

Both XGBoost and LightGBM were evaluated via Optuna hyperparameter tuning. XGBoost was selected as the best model for both CLV and churn tasks based on cross-validated performance. Full experiment history is available on the [DagsHub MLflow UI](https://dagshub.com/kajoldave8/clv-engine.mlflow).

---

## Segmentation Summary

K-Means clustering (k=5) on standardized behavioral features discovered five natural customer groups:

| Cluster | Count | % of Customers | Mean Predicted CLV (£) | Profile |
|---------|-------|---------------|----------------------|---------|
| 0 | 1,543 | 30.7% | 490.74 | Steady Occasionals |
| 1 | 207 | 4.1% | 907.14 | Accelerating Buyers |
| 2 | 2,184 | 43.5% | 286.69 | Lapsed / One-and-Done |
| 3 | 15 | 0.3% | 21,448.52 | Whale Accounts |
| 4 | 1,074 | 21.4% | 2,008.92 | Loyal Regulars |

**Key insight:** Cluster 3 (Whale Accounts) represents 0.3% of customers but generates disproportionate predicted CLV. Cluster 4 (Loyal Regulars) at 21.4% of customers drives the bulk of reliable, recurring revenue. Cluster 2 (Lapsed) at 43.5% represents the largest retention opportunity.

CLV tiers (Platinum, Gold, Silver, Bronze, At-Risk) are assigned on top of clustering, with a churn override: customers with churn probability > 0.7 are moved to At-Risk regardless of CLV tier.

---

## Limitations

**Dataset constraints:**
- Single retailer, single market (UK wholesale e-commerce). Model behavior on subscription businesses, SaaS, or retail in other geographies is unknown and likely different.
- Transaction history covers only 24 months (Dec 2009 – Dec 2011). Long-term CLV trends beyond this window are extrapolated, not observed.
- ~25% of transactions lack Customer ID (guest checkouts) and are excluded. The model cannot score anonymous customers.

**Modeling constraints:**
- The 6-month prediction horizon is fixed. Performance at other horizons (3 months, 12 months) has not been validated.
- The system assumes non-contractual, continuous purchasing behavior. It is not appropriate for contractual settings (subscriptions, memberships) without architectural changes.
- MAPE of 0.763 indicates the model's percentage error is substantial for individual low-CLV customers (small denominators inflate MAPE). MAE and RMSE are more reliable for assessing accuracy.
- Feature engineering uses StockCode prefixes as a proxy for product category. A proper product taxonomy would improve category_breadth and category_concentration features.

**Operational constraints:**
- Designed for batch scoring (daily/weekly), not real-time inference.
- PSI-based drift monitoring flags feature distribution shifts but does not automatically trigger retraining.
- The model should be retrained if PSI > 0.2 on any key feature, or if the business context changes materially (new product lines, market expansion, pricing changes).

---

## Ethical Considerations

- **No personally identifiable information (PII):** The model uses anonymized Customer ID integers and behavioral aggregates. No names, emails, addresses, or demographic attributes are used as features.
- **No demographic features:** The model does not use age, gender, ethnicity, location, or any protected attributes. Predictions are based entirely on purchase behavior.
- **Fairness caveat:** While no protected attributes are used directly, purchase behavior can correlate with socioeconomic factors. High-CLV predictions may disproportionately favor customers with higher disposable income. Marketing teams should avoid using CLV tiers to reduce service quality for low-tier customers.
- **Churn labeling:** A customer labeled "churned" may simply be between purchases. The 6-month holdout window means seasonal buyers (e.g., holiday-only) may be incorrectly flagged as churned.

---

## Monitoring & Drift Detection

The batch scoring pipeline (`src/scoring/batch_scorer.py`) includes four automated monitoring checks:

1. **Feature drift (PSI):** Population Stability Index computed for all 23 features against training baselines. PSI > 0.1 triggers a warning; PSI > 0.2 triggers an alert.
2. **Prediction distribution:** Checks whether mean and standard deviation of predicted CLV are within 2σ of training distributions.
3. **Churn rate sanity:** Flags if overall predicted churn rate falls outside the 20–60% range.
4. **Coverage check:** Reports the fraction of input customers successfully scored.

Monitoring reports are saved to `reports/monitoring_YYYYMMDD.txt`.

---

## Reproducibility

The full pipeline can be reproduced via:

```bash
# Local
make all                          # Runs: data → validate → features → train → evaluate → segment → score

# Docker
docker compose run pipeline data
docker compose run pipeline train
docker compose run pipeline score

# DVC
dvc repro
```

All artifacts are versioned with DVC. Experiment history is logged to [DagsHub MLflow](https://dagshub.com/kajoldave8/clv-engine.mlflow). The pipeline is defined in `dvc.yaml` and can be reproduced with `dvc repro`.

**Tests:** 94 pytest tests cover data validation, feature computation, and scoring pipeline integrity.

---

## Citation

```
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository.
Online Retail II Dataset. https://archive.ics.uci.edu/dataset/502/online+retail+ii
```

## References

- Fader, P.S., Hardie, B.G.S., & Lee, K.L. (2005). "Counting Your Customers the Easy Way: An Alternative to the Pareto/NBD Model." Marketing Science, 24(2), 275-284.
- Fader, P.S., Hardie, B.G.S., & Lee, K.L. (2005). "RFM and CLV: Using Iso-Value Curves for Customer Base Analysis." Journal of Marketing Research, 42(4), 415-430.
- Mitchell, M. et al. (2019). "Model Cards for Model Reporting." Proceedings of FAT*.
