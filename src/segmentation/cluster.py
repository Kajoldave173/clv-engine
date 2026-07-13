"""Behavioral customer segmentation via clustering.

Discovers natural customer groupings from behavioral features,
profiles each cluster, overlays CLV and churn predictions, and
assigns CLV tiers as a business presentation layer.
"""

from typing import Any
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

from src.utils.io import PROJECT_ROOT, load_csv, save_csv


# Features to cluster on — capture distinct behavioral dimensions.
# Deliberately excludes CLV predictions: we want segments that
# EXPLAIN value, not segments defined by it.
CLUSTER_FEATURES = [
    "frequency",              # How often they buy
    "days_since_last_purchase",  # How recently
    "monetary_value",         # Spend per transaction
    "avg_basket_value",       # Revenue per visit
    "inter_purchase_time_mean",  # Purchase regularity
    "category_breadth",       # Product diversity
    "purchase_velocity_recent_vs_early",  # Accelerating or fading?
    "lifecycle_stage",        # Active vs dormant
]


def run_segmentation(config: dict[str, Any]) -> pd.DataFrame:
    """Run the full segmentation pipeline.

    Steps:
      1. Load feature matrix (RFM + behavioral)
      2. Select and scale clustering features
      3. Evaluate K=3..8 via silhouette score
      4. Fit final K-Means model
      5. Profile each cluster
      6. Overlay CLV and churn predictions
      7. Assign CLV tiers
      8. Save everything

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        DataFrame with cluster assignments and CLV tiers.
    """
    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    rfm_path = PROJECT_ROOT / "data" / "processed" / "rfm_summary.csv"
    behavioral_path = PROJECT_ROOT / "data" / "processed" / "features_behavioral.csv"
    predictions_path = PROJECT_ROOT / "data" / "processed" / "customer_predictions.csv"

    rfm = load_csv(rfm_path, index_col="customer_id")
    behavioral = load_csv(behavioral_path, index_col="customer_id")
    predictions = load_csv(predictions_path, index_col="customer_id")

    # Merge RFM and behavioral for clustering features
    all_features = rfm.join(behavioral, how="inner")

    print(f"Loaded {len(all_features):,} customers for segmentation")

    # -----------------------------------------------------------------
    # Select and scale features
    # -----------------------------------------------------------------
    X = all_features[CLUSTER_FEATURES].copy()

    # Fill any remaining edge cases
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Clustering on {len(CLUSTER_FEATURES)} features: "
          f"{', '.join(CLUSTER_FEATURES)}")

    # -----------------------------------------------------------------
    # Evaluate K values via silhouette score
    # -----------------------------------------------------------------
    print("\nSilhouette scores by K:")
    k_range = range(3, 9)
    scores = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"  K={k}: {score:.4f}")

    best_k = max(scores, key=scores.get)
    configured_k = config["segmentation"]["n_clusters"]

    print(f"\nBest K by silhouette: {best_k} (score={scores[best_k]:.4f})")
    print(f"Configured K: {configured_k}")

    # Use configured K — the user can adjust params.yaml after
    # reviewing silhouette scores
    n_clusters = configured_k
    print(f"Using K={n_clusters}")

    # -----------------------------------------------------------------
    # Fit final model
    # -----------------------------------------------------------------
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    all_features["cluster"] = kmeans.fit_predict(X_scaled)

    # Save the fitted model and scaler for batch scoring
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, models_dir / "kmeans_model.pkl")
    joblib.dump(scaler, models_dir / "cluster_scaler.pkl")
    print(f"\nK-Means model saved to {models_dir / 'kmeans_model.pkl'}")

    # -----------------------------------------------------------------
    # Profile each cluster
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLUSTER PROFILES")
    print("=" * 60)

    # Add predictions for overlay
    all_features = all_features.join(
        predictions[["predicted_clv_ml"]], how="left"
    )
    # holdout_revenue and predicted_clv are already in rfm
    # predicted_clv is the probabilistic estimate

    for cluster_id in range(n_clusters):
        mask = all_features["cluster"] == cluster_id
        segment = all_features.loc[mask]
        n = len(segment)
        pct = n / len(all_features) * 100

        print(f"\n--- Cluster {cluster_id} ({n:,} customers, {pct:.1f}%) ---")

        # Behavioral profile
        print("  Behavioral profile:")
        for feat in CLUSTER_FEATURES:
            cluster_mean = segment[feat].mean()
            overall_mean = all_features[feat].mean()
            direction = "+" if cluster_mean > overall_mean else "-"
            print(f"    {feat:40s} {cluster_mean:>10.2f}  "
                  f"(overall: {overall_mean:.2f}) [{direction}]")

        # CLV overlay
        mean_clv = segment["predicted_clv_ml"].mean()
        total_clv = segment["predicted_clv_ml"].sum()
        total_share = total_clv / all_features["predicted_clv_ml"].sum() * 100
        print(f"\n  CLV overlay:")
        print(f"    Mean predicted CLV (ML):  {mean_clv:,.2f}")
        print(f"    Total predicted CLV:      {total_clv:,.2f} "
              f"({total_share:.1f}% of total)")

        # Churn overlay
        mean_p_alive = segment["p_alive"].mean()
        high_churn_pct = (segment["p_alive"] < 0.5).mean() * 100
        print(f"\n  Churn overlay:")
        print(f"    Mean P(alive):            {mean_p_alive:.3f}")
        print(f"    High churn risk (p<0.5):  {high_churn_pct:.1f}%")

    # -----------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)

    summary = (
        all_features.groupby("cluster")
        .agg(
            n_customers=("cluster", "size"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary_value", "mean"),
            mean_lifecycle=("lifecycle_stage", "mean"),
            mean_clv_ml=("predicted_clv_ml", "mean"),
            mean_p_alive=("p_alive", "mean"),
        )
        .round(2)
    )
    summary["pct_of_customers"] = (
        summary["n_customers"] / summary["n_customers"].sum() * 100
    ).round(1)

    print(summary.to_string())

    # -----------------------------------------------------------------
    # CLV tier assignment
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLV TIER ASSIGNMENT")
    print("=" * 60)

    all_features["clv_tier"] = assign_clv_tiers(
        all_features["predicted_clv_ml"],
        all_features["p_alive"],
    )

    tier_summary = (
        all_features.groupby("clv_tier")
        .agg(
            n_customers=("clv_tier", "size"),
            mean_clv=("predicted_clv_ml", "mean"),
            total_clv=("predicted_clv_ml", "sum"),
            mean_p_alive=("p_alive", "mean"),
        )
        .round(2)
    )
    tier_summary["pct_of_customers"] = (
        tier_summary["n_customers"] / tier_summary["n_customers"].sum() * 100
    ).round(1)

    # Order tiers logically
    tier_order = ["Platinum", "Gold", "Silver", "Bronze", "At-Risk"]
    tier_summary = tier_summary.reindex(
        [t for t in tier_order if t in tier_summary.index]
    )

    print(tier_summary.to_string())

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------
    output = all_features[["cluster", "clv_tier"]].copy()
    output_path = PROJECT_ROOT / "data" / "processed" / "customer_segments.csv"
    save_csv(output, output_path, index=True)
    print(f"\nSegments saved to {output_path}")

    return output


def assign_clv_tiers(
    predicted_clv: pd.Series,
    p_alive: pd.Series,
    churn_threshold: float = 0.5,
) -> pd.Series:
    """Assign CLV tiers based on percentiles with churn override.

    Tiers:
      Platinum — top 5% by predicted CLV
      Gold — 5th to 20th percentile
      Silver — 20th to 50th percentile
      Bronze — 50th to 80th percentile
      At-Risk — bottom 20%, OR any customer with p_alive < threshold
                regardless of CLV tier

    The At-Risk override is the key business logic: a customer can
    be Gold by CLV but flagged At-Risk if churn probability is high.
    This is how retention teams actually prioritize.

    Args:
        predicted_clv: Predicted CLV per customer.
        p_alive: P(alive) per customer.
        churn_threshold: P(alive) below this triggers At-Risk override.

    Returns:
        Series of tier labels.
    """
    # Start with percentile-based tiers
    percentiles = predicted_clv.rank(pct=True)

    tiers = pd.Series("Bronze", index=predicted_clv.index)
    tiers[percentiles >= 0.80] = "Silver"
    tiers[percentiles >= 0.50] = "Silver"
    tiers[percentiles < 0.50] = "Bronze"
    tiers[percentiles < 0.20] = "At-Risk"
    tiers[percentiles >= 0.80] = "Gold"
    tiers[percentiles >= 0.95] = "Platinum"

    # Churn override: high churn risk regardless of CLV tier
    churn_override = p_alive < churn_threshold
    n_overridden = ((churn_override) & (tiers != "At-Risk")).sum()
    tiers[churn_override] = "At-Risk"

    print(f"\n  Churn override: {n_overridden} customers moved to At-Risk "
          f"(p_alive < {churn_threshold})")

    return tiers