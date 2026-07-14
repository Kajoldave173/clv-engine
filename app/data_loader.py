"""Shared data loading, model caching, and prediction utilities."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
from glob import glob


# ── Constants ──────────────────────────────────────────────
CLUSTER_NAMES = {
    0: "Steady Occasionals",
    1: "Accelerating Buyers",
    2: "Lapsed / One-and-Done",
    3: "Whale Accounts",
    4: "Loyal Regulars",
}

TIER_ORDER = ["Platinum", "Gold", "Silver", "Bronze", "At-Risk"]

CLUSTER_COLORS = {
    "Steady Occasionals": "#636EFA",
    "Accelerating Buyers": "#00CC96",
    "Lapsed / One-and-Done": "#EF553B",
    "Whale Accounts": "#FFA15A",
    "Loyal Regulars": "#AB63FA",
}

# From best-models-final run on DagsHub
MODEL_METRICS = {
    "clv_mae_probabilistic": 672.98,
    "clv_mae_untuned": 509.44,
    "clv_mae_tuned": 334.15,
    "clv_rmse": 836.54,
    "clv_r2": 0.965,
    "churn_auc_baseline": 0.452,
    "churn_auc_ml": 0.855,
    "churn_f1": 0.783,
}

FEATURE_DISPLAY = {
    "frequency": "Purchase Frequency",
    "recency": "Recency (days)",
    "T": "Customer Age (days)",
    "monetary_value": "Avg Transaction Value (£)",
    "predicted_purchases": "Predicted Purchases (BG/NBD)",
    "p_alive": "P(Alive)",
    "expected_avg_value": "Expected Avg Value (£)",
    "n_orders": "Total Orders",
    "avg_basket_size": "Avg Basket Size",
    "avg_basket_value": "Avg Basket Value (£)",
    "max_single_transaction": "Max Single Transaction (£)",
    "days_since_last_purchase": "Days Since Last Purchase",
    "lifecycle_stage": "Lifecycle Stage",
    "weekend_purchase_ratio": "Weekend Purchase %",
    "category_breadth": "Category Breadth",
    "category_concentration": "Category Concentration",
    "inter_purchase_time_mean": "Avg Inter-Purchase Time (days)",
    "inter_purchase_time_std": "Inter-Purchase Time StdDev",
    "inter_purchase_time_trend": "Inter-Purchase Time Trend",
    "monetary_trend": "Monetary Trend",
    "monetary_cv": "Monetary CV",
    "basket_size_trend": "Basket Size Trend",
    "purchase_velocity_recent_vs_early": "Purchase Velocity (Recent/Early)",
}


# ── Data loading ───────────────────────────────────────────
@st.cache_data
def load_scored_data():
    """Load the most recent scored customers file."""
    files = sorted(glob("data/predictions/scored_customers_*.csv"))
    if not files:
        st.error("No scored data found. Run `python -m src.cli score` first.")
        st.stop()
    df = pd.read_csv(files[-1])
    df["cluster_name"] = df["behavioral_cluster"].map(CLUSTER_NAMES)
    return df


@st.cache_data
def load_feature_matrix():
    """Merge RFM + behavioral into the full 23-feature matrix."""
    rfm = pd.read_csv("data/processed/rfm_summary.csv")
    behavioral = pd.read_csv("data/processed/features_behavioral.csv")

    rfm_cols = [
        "customer_id", "frequency", "recency", "T", "monetary_value",
        "predicted_purchases", "p_alive", "expected_avg_value",
    ]
    features = rfm[rfm_cols].merge(behavioral, on="customer_id")
    features = features.set_index("customer_id")
    return features


@st.cache_resource
def load_models():
    """Load all serialized model artifacts."""
    scaler = joblib.load("models/scaler.pkl")
    xgb_clv = xgb.Booster()
    xgb_clv.load_model("models/xgb_clv_model.json")
    churn = xgb.Booster()
    churn.load_model("models/churn_model.json")

    return {
        "scaler": scaler,
        "xgb_clv": xgb_clv,
        "churn": churn,
    }


# ── Prediction helpers ─────────────────────────────────────
def _prepare_input(feature_row, models):
    """Scale a single feature row using the training scaler."""
    scaler = models["scaler"]
    feature_order = list(scaler.feature_names_in_)
    X = pd.DataFrame([feature_row[feature_order].values], columns=feature_order)
    X_scaled = scaler.transform(X)
    return xgb.DMatrix(X_scaled, feature_names=feature_order), feature_order


def predict_clv(feature_row, models):
    """Predict CLV for one customer."""
    dmat, _ = _prepare_input(feature_row, models)
    return float(models["xgb_clv"].predict(dmat)[0])


def predict_churn(feature_row, models):
    """Predict churn probability for one customer."""
    dmat, _ = _prepare_input(feature_row, models)
    return float(models["churn"].predict(dmat)[0])


def compute_shap(feature_row, models):
    """Compute SHAP values for one customer. Returns (values, names, base)."""
    scaler = models["scaler"]
    feature_order = list(scaler.feature_names_in_)
    X = pd.DataFrame([feature_row[feature_order].values], columns=feature_order)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_order)

    explainer = shap.TreeExplainer(models["xgb_clv"])
    sv = explainer.shap_values(X_scaled)
    base = explainer.expected_value

    if isinstance(base, np.ndarray):
        base = base[0]
    if isinstance(sv, list):
        sv = sv[0]

    return sv.flatten(), feature_order, float(base)