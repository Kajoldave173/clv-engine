# CLV Engine — Customer Lifetime Value Prediction & Segmentation

A production-grade ML pipeline for predicting customer lifetime value and discovering behavioral segments in non-contractual retail settings.

## Problem

Given a customer's transaction history, predict how much revenue they will generate over the next 6 months, estimate their probability of churning, and assign them to a behavioral segment — all in a reproducible, automated pipeline that mirrors how this would work in a production environment.

## Approach

1. **Probabilistic baseline** — BG/NBD + Gamma-Gamma models estimate future purchase count and average transaction value using only recency, frequency, and customer age. This is the interpretable, theory-grounded baseline.

2. **ML improvement** — XGBoost regressor trained on 25+ behavioral features (inter-purchase velocity trends, category breadth, basket size trends, return behavior) to predict holdout-period revenue. Compared against the probabilistic baseline on RMSE/MAE.

3. **Churn classifier** — XGBoost binary classifier that improves on the BG/NBD's P(alive) estimate by incorporating behavioral signals the probabilistic model can't use.

4. **Behavioral segmentation** — K-Means/GMM clustering on behavioral features discovers natural customer groupings. CLV and churn predictions are overlaid on segments for actionable insights.

5. **Batch scoring pipeline** — Production-style scorer that ingests new transactions, validates data quality, computes features, generates predictions, checks for drift, and outputs a scored customer table.

## Dataset

UCI Online Retail II — ~1M transactions, ~5,900 customers, December 2009 – December 2011.

## Quick Start

```bash
# Clone and install
git clone https://github.com/<username>/clv-engine.git
cd clv-engine
pip install -e .
pip install -r requirements.txt

# Run the full pipeline
make all

# Or run individual steps
make data        # Ingest and clean
make validate    # Data quality checks
make features    # Build feature matrices
make train       # Train all models
make evaluate    # Evaluate + SHAP analysis
make segment     # Behavioral clustering
make score       # Batch score all customers

# Launch the dashboard
make app
```

## Tech Stack

| Layer | Tool |
|---|---|
| Probabilistic CLV | lifetimes (BG/NBD, Gamma-Gamma) |
| ML modeling | XGBoost, scikit-learn |
| Explainability | SHAP |
| Segmentation | scikit-learn (KMeans, GMM) |
| Data validation | Pandera |
| Dashboard | Streamlit |
| Experiment tracking | MLflow (DagsHub) |
| Data/model versioning | DVC |
| Pipeline orchestration | Makefile + Typer CLI |

## Results

_To be filled after training._
