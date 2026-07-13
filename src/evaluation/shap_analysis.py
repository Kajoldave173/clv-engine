"""SHAP explainability analysis for the XGBoost CLV model.

Generates global and local explanations:
  - Beeswarm summary plot (global feature importance with direction)
  - Bar plot (mean |SHAP| ranking)
  - Dependence plots for top features (nonlinear relationships)
  - Waterfall plots for archetype customers (individual explanations)
  - Cluster-level SHAP (segment differentiation)

All plots are saved to reports/figures/.
"""

from typing import Any

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import joblib

from src.utils.io import PROJECT_ROOT, load_csv


def run_shap_analysis(config: dict[str, Any]) -> None:
    """Run the full SHAP explainability pipeline.

    Args:
        config: Pipeline configuration dictionary.
    """
    # -----------------------------------------------------------------
    # Load model, scaler, features, and segments
    # -----------------------------------------------------------------
    models_dir = PROJECT_ROOT / "models"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load XGBoost model
    model = xgb.XGBRegressor()
    model.load_model(str(models_dir / "xgb_clv_model.json"))

    # Load scaler
    scaler = joblib.load(models_dir / "scaler.pkl")

    # Load feature data
    rfm = load_csv(
        PROJECT_ROOT / "data" / "processed" / "rfm_summary.csv",
        index_col="customer_id",
    )
    behavioral = load_csv(
        PROJECT_ROOT / "data" / "processed" / "features_behavioral.csv",
        index_col="customer_id",
    )
    segments = load_csv(
        PROJECT_ROOT / "data" / "processed" / "customer_segments.csv",
        index_col="customer_id",
    )
    predictions = load_csv(
        PROJECT_ROOT / "data" / "processed" / "customer_predictions.csv",
        index_col="customer_id",
    )

    # Reconstruct the same feature matrix used during training
    rfm_features = rfm[[
        "frequency", "recency", "T", "monetary_value",
        "predicted_purchases", "p_alive", "expected_avg_value",
    ]].copy()

    feature_matrix = rfm_features.join(behavioral, how="inner")
    feature_names = list(feature_matrix.columns)

    # Scale features the same way training did
    X_scaled = pd.DataFrame(
        scaler.transform(feature_matrix),
        columns=feature_names,
        index=feature_matrix.index,
    )

    print(f"Loaded {len(X_scaled):,} customers, {len(feature_names)} features")

    # -----------------------------------------------------------------
    # Compute SHAP values
    # -----------------------------------------------------------------
    sample_size = config["evaluation"]["shap_sample_size"]
    if sample_size < len(X_scaled):
        sample_idx = X_scaled.sample(n=sample_size, random_state=42).index
    else:
        sample_idx = X_scaled.index
        sample_size = len(X_scaled)

    X_sample = X_scaled.loc[sample_idx]

    print(f"Computing SHAP values for {sample_size} customers...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # Also compute for ALL customers (needed for cluster-level analysis)
    print(f"Computing SHAP values for all {len(X_scaled):,} customers...")
    shap_values_all = explainer(X_scaled)

    # -----------------------------------------------------------------
    # 1. Beeswarm summary plot
    # -----------------------------------------------------------------
    print("\nGenerating beeswarm summary plot...")
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.title("SHAP Feature Importance (Beeswarm)", fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {figures_dir / 'shap_beeswarm.png'}")

    # -----------------------------------------------------------------
    # 2. Bar plot of mean |SHAP|
    # -----------------------------------------------------------------
    print("Generating bar plot...")
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.title("Mean |SHAP| Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {figures_dir / 'shap_bar.png'}")

    # -----------------------------------------------------------------
    # 3. Dependence plots for top 5 features
    # -----------------------------------------------------------------
    print("Generating dependence plots for top 5 features...")
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[::-1][:5]
    top_features = [feature_names[i] for i in top_features_idx]

    for feat in top_features:
        plt.figure(figsize=(8, 6))
        shap.plots.scatter(shap_values[:, feat], show=False)
        plt.title(f"SHAP Dependence: {feat}", fontsize=13)
        plt.tight_layout()
        safe_name = feat.replace("/", "_")
        plt.savefig(
            figures_dir / f"shap_dependence_{safe_name}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()
    print(f"  Top 5 features: {', '.join(top_features)}")
    print(f"  Saved 5 dependence plots to {figures_dir}")

    # -----------------------------------------------------------------
    # 4. Waterfall plots for 3 archetype customers
    # -----------------------------------------------------------------
    print("Generating waterfall plots for archetype customers...")

    # Select archetypes from the FULL shap values
    ml_clv = predictions.loc[X_scaled.index, "predicted_clv_ml"]

    # High CLV: customer near the 95th percentile
    p95 = ml_clv.quantile(0.95)
    high_clv_id = (ml_clv - p95).abs().idxmin()

    # Medium CLV: customer near the median
    p50 = ml_clv.quantile(0.50)
    medium_clv_id = (ml_clv - p50).abs().idxmin()

    # At-Risk: customer with lowest P(alive) who has some purchase history
    p_alive = rfm.loc[X_scaled.index, "p_alive"]
    has_history = rfm.loc[X_scaled.index, "frequency"] >= 2
    at_risk_candidates = p_alive[has_history]
    at_risk_id = at_risk_candidates.idxmin()

    archetypes = {
        "high_clv": high_clv_id,
        "medium_clv": medium_clv_id,
        "at_risk": at_risk_id,
    }

    for label, cust_id in archetypes.items():
        idx_pos = list(X_scaled.index).index(cust_id)
        cust_clv = ml_clv.loc[cust_id]
        cust_palive = p_alive.loc[cust_id]

        print(f"\n  {label}: customer {cust_id} "
              f"(CLV={cust_clv:.2f}, P(alive)={cust_palive:.3f})")

        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(shap_values_all[idx_pos], show=False, max_display=12)
        plt.title(f"SHAP Waterfall: {label} (customer {cust_id})", fontsize=13)
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"shap_waterfall_{label}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    # -----------------------------------------------------------------
    # 5. Cluster-level SHAP analysis
    # -----------------------------------------------------------------
    print("\nComputing cluster-level SHAP profiles...")

    cluster_labels = segments.loc[X_scaled.index, "cluster"]
    shap_df = pd.DataFrame(
        shap_values_all.values,
        columns=feature_names,
        index=X_scaled.index,
    )
    shap_df["cluster"] = cluster_labels

    # Mean absolute SHAP by cluster — shows what drives each segment
    print("\nMean |SHAP| by cluster (top 5 features per cluster):")
    print("-" * 70)

    for cluster_id in sorted(cluster_labels.unique()):
        cluster_shap = shap_df[shap_df["cluster"] == cluster_id].drop(columns=["cluster"])
        mean_abs = cluster_shap.abs().mean().sort_values(ascending=False)
        n = (cluster_labels == cluster_id).sum()

        print(f"\n  Cluster {cluster_id} ({n} customers):")
        for feat, val in mean_abs.head(5).items():
            # Show direction: is the mean SHAP positive or negative?
            mean_signed = cluster_shap[feat].mean()
            direction = "+" if mean_signed > 0 else "-"
            print(f"    {feat:40s} |SHAP|={val:>8.2f}  ({direction})")

    # -----------------------------------------------------------------
    # Print summary of all saved files
    # -----------------------------------------------------------------
    all_figures = sorted(figures_dir.glob("shap_*.png"))
    print(f"\n{'=' * 50}")
    print(f"SHAP analysis complete: {len(all_figures)} plots saved")
    print(f"{'=' * 50}")
    for fig_path in all_figures:
        print(f"  {fig_path.name}")