# CLV Prediction & Segmentation Engine

An end-to-end Customer Lifetime Value prediction and behavioral segmentation system built as a production-grade ML pipeline. Combines probabilistic models (BG/NBD + Gamma-Gamma) with Optuna-tuned XGBoost, unsupervised clustering, and SHAP explainability — wrapped in a reproducible pipeline with data validation, experiment tracking, batch scoring, and a live Streamlit dashboard.

**[Live App](https://clv-engine-kajol-dave-project.streamlit.app/)** · **[MLflow Experiments](https://dagshub.com/kajoldave8/clv-engine.mlflow)** · **[Model Card](reports/model_card.md)**

---

## Results at a Glance

| Model | Metric | Value |
|-------|--------|-------|
| CLV Regressor (XGBoost, Optuna-tuned) | MAE | **£334** (50.3% improvement over probabilistic baseline) |
| CLV Regressor | R² | **0.965** |
| Churn Classifier (XGBoost, Optuna-tuned) | AUC-ROC | **0.855** (89.1% improvement over P(alive) baseline) |
| Churn Classifier | F1 | **0.783** |
| Segmentation (K-Means, k=5) | Coverage | **5,023 customers** scored at 100% coverage |

---

## Dashboard

The Streamlit app provides three views:

**Portfolio View** — Executive dashboard with KPIs, segment distribution, revenue concentration (Lorenz curve), CLV tier breakdown, and model performance comparison charts.

**Customer Lookup** — Individual customer scorecards showing predicted CLV, churn probability, segment, tier, SHAP waterfall explaining the prediction, and peer comparison within their cluster.

**What-If Simulator** — Start from any customer, adjust features with sliders (purchase frequency, basket value, category breadth, etc.), and see the predicted CLV change in real time with side-by-side SHAP comparison. Designed for campaign ROI estimation.

---

## Architecture

```
Raw Transactions (UCI Online Retail II, ~1M rows)
    │
    ├── Data Cleaning → 19 validation checks
    ├── Calibration/Holdout temporal split (18mo / 6mo)
    │
    ├── Probabilistic Baseline
    │   ├── BG/NBD → transaction rate + P(alive)
    │   └── Gamma-Gamma → expected monetary value
    │
    ├── Feature Engineering (23 features)
    │   ├── RFM summary (4)
    │   ├── Probabilistic outputs as stacking inputs (3)
    │   └── Behavioral features (16)
    │
    ├── ML Models (Optuna-tuned)
    │   ├── XGBoost CLV Regressor → 6-month revenue prediction
    │   └── XGBoost Churn Classifier → binary churn prediction
    │
    ├── Behavioral Segmentation
    │   ├── K-Means (k=5) on behavioral features
    │   └── CLV tier assignment with churn override
    │
    ├── SHAP Explainability
    │   ├── Global: summary plots, dependence plots
    │   └── Per-customer: waterfall plots, top-3 drivers
    │
    └── Batch Scoring Pipeline
        ├── Feature recomputation
        ├── PSI-based drift monitoring
        └── Scored output with predictions + segments + SHAP drivers
```

---

## Segmentation

K-Means clustering on behavioral features discovered five natural customer groups:

| Cluster | Customers | Mean CLV | Profile |
|---------|-----------|----------|---------|
| Loyal Regulars | 1,074 (21.4%) | £2,009 | High frequency, most recent, consistent spend |
| Steady Occasionals | 1,543 (30.7%) | £491 | Moderate engagement, regular but infrequent |
| Accelerating Buyers | 207 (4.1%) | £907 | Growing purchase velocity and basket size |
| Whale Accounts | 15 (0.3%) | £21,449 | Extreme spend, large baskets, broad categories |
| Lapsed / One-and-Done | 2,184 (43.5%) | £287 | Low frequency, high recency, narrow categories |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Git, DVC

### Setup

```bash
git clone https://github.com/Kajoldave173/clv-engine.git
cd clv-engine
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
pip install -e .
```

### Run the Full Pipeline

```bash
# Download, clean, validate, build features, train, evaluate, segment, score
make all
```

Or step by step via the Typer CLI:

```bash
python -m src.cli data           # Ingest + clean + split
python -m src.cli validate       # 19 data quality checks
python -m src.cli features       # RFM + 16 behavioral features
python -m src.cli train          # BG/NBD, Gamma-Gamma, XGBoost CLV, XGBoost churn
python -m src.cli evaluate       # Holdout metrics + SHAP analysis (10 plots)
python -m src.cli segment        # K-Means clustering + CLV tier assignment
python -m src.cli score          # Batch score all customers
```

### Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Docker

```bash
docker compose run pipeline train     # Train models
docker compose run pipeline score     # Batch score
docker compose up app                 # Launch dashboard at localhost:8501
```

### Run Tests

```bash
pytest                                # 94 tests across 3 modules
```

---

## Project Structure

```
clv-engine/
├── app/                          # Streamlit dashboard
│   ├── streamlit_app.py          # Entry point (st.navigation API)
│   ├── data_loader.py            # Cached data/model loading
│   ├── pages/
│   │   ├── portfolio_view.py     # Executive dashboard
│   │   ├── customer_lookup.py    # Customer scorecard + SHAP
│   │   └── what_if_simulator.py  # Live re-scoring with sliders
│   └── components/
│       └── shap_plots.py         # Custom Plotly SHAP waterfall
├── src/
│   ├── cli.py                    # Typer CLI entrypoint
│   ├── data/                     # Ingestion, cleaning, validation
│   ├── features/                 # RFM, behavioral features, orchestration
│   ├── models/                   # Probabilistic, XGBoost, churn, training
│   ├── segmentation/             # K-Means clustering, profiling
│   ├── evaluation/               # Metrics, SHAP analysis
│   ├── scoring/                  # Batch scorer, PSI monitoring
│   ├── experiment/               # MLflow config, Optuna tuning
│   └── utils/                    # I/O helpers, path management
├── tests/
│   ├── conftest.py
│   ├── test_data_validation.py
│   ├── test_features.py
│   └── test_scoring.py
├── configs/                      # Configuration files
├── data/                         # Raw, interim, processed, predictions
├── models/                       # Serialized model artifacts
├── reports/                      # Figures, monitoring reports, model card
├── Dockerfile                    # Pipeline container
├── Dockerfile.app                # Streamlit app container
├── docker-compose.yaml
├── dvc.yaml                      # DVC pipeline definition
├── Makefile                      # Pipeline orchestration
├── requirements.txt
└── pyproject.toml
```

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Probabilistic CLV | lifetimes (BG/NBD, Gamma-Gamma) |
| ML Modeling | XGBoost, scikit-learn, LightGBM |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| Segmentation | scikit-learn (K-Means) |
| Data Validation | pandera, custom assertions |
| Dashboard | Streamlit, Plotly |
| Experiment Tracking | MLflow on DagsHub |
| Data/Model Versioning | DVC |
| Pipeline Orchestration | Makefile + DVC + Typer CLI |
| Containerization | Docker, Docker Compose |
| Testing | pytest (94 tests) |
| Deployment | Streamlit Community Cloud |

---

## Dataset

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — ~1,046,000 transactions from a UK-based online/wholesale retailer (Dec 2009 – Dec 2011). After cleaning: ~530,000 rows, 5,861 unique customers. Split temporally: 18-month calibration period for training, 6-month holdout for validation.

---

## License

This project is for educational and portfolio purposes. The UCI Online Retail II dataset is publicly available under its original terms.
