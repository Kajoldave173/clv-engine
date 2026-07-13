"""Batch scoring pipeline: score customers through all trained models.

Orchestrates the full scoring flow:
  1. Load and validate input transactions
  2. Compute RFM and behavioral features
  3. Load trained models
  4. Generate probabilistic + ML predictions
  5. Run monitoring checks (feature drift, prediction health)
  6. Assign behavioral clusters and CLV tiers
  7. Compute per-customer SHAP drivers
  8. Save scored customer table

Usage:
    python -m src.cli score --input data/interim/transactions_clean.csv

Output:
    data/predictions/scored_customers_YYYYMMDD.csv
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import shap
from lifetimes.utils import summary_data_from_transaction_data

from src.utils.io import PROJECT_ROOT, PATHS, load_csv, save_csv
from src.scoring.monitoring import run_monitoring, save_monitoring_report
from src.segmentation.cluster import assign_clv_tiers


# ====================================================================
# 1. Model loading
# ====================================================================

def load_all_models() -> dict[str, Any]:
    """Load all trained model artifacts from disk.

    Loads 7 artifacts: BG/NBD, Gamma-Gamma, XGBoost CLV regressor,
    churn classifier, feature scaler, K-Means, and cluster scaler.

    Returns:
        Dict with keys: bgf, ggf, xgb_clv, churn, scaler,
        kmeans, cluster_scaler.

    Raises:
        FileNotFoundError: If any required model file is missing.
    """
    models_dir = PATHS["models"]

    required_files = {
        "bgnbd_model.pkl": "BG/NBD",
        "gg_model.pkl": "Gamma-Gamma",
        "xgb_clv_model.json": "XGBoost CLV",
        "churn_model.json": "Churn classifier",
        "scaler.pkl": "Feature scaler",
        "kmeans_model.pkl": "K-Means",
        "cluster_scaler.pkl": "Cluster scaler",
    }

    missing = [
        f"  - {name} ({desc})"
        for name, desc in required_files.items()
        if not (models_dir / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model artifacts:\n"
            + "\n".join(missing)
            + "\nRun the training pipeline first: python -m src.cli train"
        )

    # Load probabilistic models (joblib)
    bgf = joblib.load(models_dir / "bgnbd_model.pkl")
    ggf = joblib.load(models_dir / "gg_model.pkl")

    # Load XGBoost models (native JSON format)
    xgb_clv = xgb.XGBRegressor()
    xgb_clv.load_model(str(models_dir / "xgb_clv_model.json"))

    churn_model = xgb.XGBClassifier()
    churn_model.load_model(str(models_dir / "churn_model.json"))

    # Load scalers and clustering model
    scaler = joblib.load(models_dir / "scaler.pkl")
    kmeans = joblib.load(models_dir / "kmeans_model.pkl")
    cluster_scaler = joblib.load(models_dir / "cluster_scaler.pkl")

    print(f"  Loaded 7 model artifacts from {models_dir}")

    return {
        "bgf": bgf,
        "ggf": ggf,
        "xgb_clv": xgb_clv,
        "churn": churn_model,
        "scaler": scaler,
        "kmeans": kmeans,
        "cluster_scaler": cluster_scaler,
    }


# ====================================================================
# 2. Feature computation
# ====================================================================

def compute_scoring_rfm(
    transactions: pd.DataFrame,
    observation_end: pd.Timestamp,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute RFM summary table from scoring transactions.

    Uses the same lifetimes utility as the training pipeline to
    ensure consistent feature definitions (frequency, recency, T,
    monetary_value).

    Args:
        transactions: Cleaned transaction data.
        observation_end: End of observation window (max invoice date).
        config: Pipeline configuration.

    Returns:
        RFM DataFrame indexed by customer_id.
    """
    monetary_col = "total_amount"

    rfm = summary_data_from_transaction_data(
        transactions,
        customer_id_col="customer_id",
        datetime_col="invoice_date",
        monetary_value_col=monetary_col,
        observation_period_end=observation_end,
        freq="D",
    )

    n_repeat = (rfm["frequency"] >= 1).sum()
    n_onetime = (rfm["frequency"] == 0).sum()
    print(f"  RFM summary: {len(rfm):,} customers")
    print(f"    Repeat buyers: {n_repeat:,}  |  One-time: {n_onetime:,}")

    return rfm


def compute_scoring_behavioral(
    transactions: pd.DataFrame,
    observation_end: pd.Timestamp,
) -> pd.DataFrame:
    """Compute 16 behavioral features from scoring transactions.

    Mirrors the training pipeline's behavioral.py to ensure feature
    consistency. All features computed relative to observation_end.

    Features computed:
      Timing:   n_orders, tenure_days, days_since_last_purchase,
                inter_purchase_time_mean/std/trend,
                purchase_velocity_recent_vs_early
      Basket:   avg_basket_size, avg_basket_value, basket_size_trend
      Monetary: monetary_trend, max_single_transaction, monetary_cv
      Product:  category_breadth, category_concentration
      Stage:    lifecycle_stage

    Args:
        transactions: Cleaned transaction data.
        observation_end: End of observation window.

    Returns:
        Behavioral feature DataFrame indexed by customer_id.
    """
    txns = transactions.copy()
    txns["invoice_date"] = pd.to_datetime(txns["invoice_date"])

    # ── Per-invoice aggregation ────────────────────────────────────
    invoice_agg = (
        txns
        .groupby(["customer_id", "invoice"])
        .agg(
            invoice_date=("invoice_date", "first"),
            n_unique_items=("stock_code", "nunique"),
            total=("total_amount", "sum"),
        )
        .reset_index()
    )

    # ── Per-customer feature computation ───────────────────────────
    records = []

    for cust_id, group in invoice_agg.groupby("customer_id"):
        group = group.sort_values("invoice_date")
        dates = group["invoice_date"]
        totals = group["total"].values
        baskets = group["n_unique_items"].values.astype(float)
        n = len(group)

        first_date = dates.iloc[0]
        last_date = dates.iloc[-1]

        n_orders = n
        tenure_days = (last_date - first_date).days if n > 1 else 0
        days_since_last = (observation_end - last_date).days
        # Weekend purchase ratio
        weekend_mask = dates.dt.dayofweek >= 5  # Saturday=5, Sunday=6
        weekend_ratio = float(weekend_mask.sum()) / n if n > 0 else 0.0

        # ── Inter-purchase timing ──────────────────────────────────
        if n >= 2:
            gaps = dates.diff().dt.days.dropna().values.astype(float)
            ipt_mean = float(np.mean(gaps))
            ipt_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0
            ipt_trend = (
                float(np.polyfit(np.arange(len(gaps)), gaps, 1)[0])
                if len(gaps) >= 2
                else 0.0
            )

           # Velocity: orders in second half vs first half of tenure
            mid_date = first_date + (last_date - first_date) / 2
            early_count = int((dates <= mid_date).sum())
            recent_count = int((dates > mid_date).sum())
            velocity = float(recent_count / max(early_count, 1))
        else:
            ipt_mean = 0.0
            ipt_std = 0.0
            ipt_trend = 0.0
            velocity = 1.0

        # ── Basket features ────────────────────────────────────────
        avg_basket_size = float(np.mean(baskets))
        avg_basket_value = float(np.mean(totals))
        basket_trend = (
            float(np.polyfit(np.arange(n), baskets, 1)[0])
            if n >= 2
            else 0.0
        )

        # ── Monetary features ──────────────────────────────────────
        monetary_trend = (
            float(np.polyfit(np.arange(n), totals, 1)[0])
            if n >= 2
            else 0.0
        )
        max_single = float(np.max(totals))
        mean_total = float(np.mean(totals))
        std_total = float(np.std(totals))
        monetary_cv = std_total / mean_total if mean_total > 0 else 0.0

        # ── Lifecycle stage ────────────────────────────────────────
        # Training defines tenure as observation_end - first_purchase
        # (not last - first), so one-time buyers get lifecycle = 0.0
        tenure_from_start = (observation_end - first_date).days
        lifecycle = 1.0 - (days_since_last / max(tenure_from_start, 1))

        records.append({
            "customer_id": cust_id,
            "n_orders": n_orders,
            "weekend_purchase_ratio": weekend_ratio,
            "days_since_last_purchase": days_since_last,
            "inter_purchase_time_mean": ipt_mean,
            "inter_purchase_time_std": ipt_std,
            "inter_purchase_time_trend": ipt_trend,
            "purchase_velocity_recent_vs_early": velocity,
            "avg_basket_size": avg_basket_size,
            "avg_basket_value": avg_basket_value,
            "basket_size_trend": basket_trend,
            "monetary_trend": monetary_trend,
            "max_single_transaction": max_single,
            "monetary_cv": monetary_cv,
            "lifecycle_stage": lifecycle,
        })

    behavioral = pd.DataFrame(records).set_index("customer_id")

    # ── Category features (require raw transactions) ───────────────
    txns["category"] = txns["stock_code"].astype(str).str[:2]

    # Category breadth: number of distinct product categories
    cat_breadth = (
        txns.groupby("customer_id")["category"]
        .nunique()
        .rename("category_breadth")
    )

    # Category concentration: Herfindahl index of spending across categories
    cat_spend = (
        txns.groupby(["customer_id", "category"])["total_amount"]
        .sum()
        .reset_index()
    )
    cust_totals = cat_spend.groupby("customer_id")["total_amount"].transform("sum")
    cat_spend["share_sq"] = (cat_spend["total_amount"] / cust_totals) ** 2
    cat_concentration = (
        cat_spend.groupby("customer_id")["share_sq"]
        .sum()
        .rename("category_concentration")
    )

    behavioral = behavioral.join(cat_breadth).join(cat_concentration)

    print(f"  Behavioral: {len(behavioral.columns)} features "
          f"for {len(behavioral):,} customers")

    return behavioral


# ====================================================================
# 3. Scoring
# ====================================================================

def score_probabilistic(
    rfm: pd.DataFrame,
    models: dict,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Generate probabilistic predictions using BG/NBD + Gamma-Gamma.

    Adds to rfm DataFrame: predicted_purchases, p_alive,
    expected_avg_value, predicted_clv_probabilistic.

    One-time buyers (frequency=0) get a fallback monetary estimate
    using the mean expected value across repeat buyers.

    Args:
        rfm: RFM summary table.
        models: Loaded model artifacts.
        config: Pipeline configuration.

    Returns:
        Updated rfm DataFrame with prediction columns.
    """
    bgf = models["bgf"]
    ggf = models["ggf"]

    horizon_months = config["models"]["probabilistic"]["prediction_horizon_months"]
    horizon_days = horizon_months * 30

    # ── BG/NBD: predicted purchases and P(alive) ───────────────────
    rfm["predicted_purchases"] = (
        bgf.conditional_expected_number_of_purchases_up_to_time(
            t=horizon_days,
            frequency=rfm["frequency"],
            recency=rfm["recency"],
            T=rfm["T"],
        )
    )

    rfm["p_alive"] = bgf.conditional_probability_alive(
        frequency=rfm["frequency"],
        recency=rfm["recency"],
        T=rfm["T"],
    )

    # ── Gamma-Gamma: expected average monetary value ───────────────
    repeat_mask = rfm["frequency"] >= 1
    repeat = rfm.loc[repeat_mask]

    expected_avg = ggf.conditional_expected_average_profit(
        frequency=repeat["frequency"],
        monetary_value=repeat["monetary_value"],
    )

    # One-time buyer fallback: empirical mean of repeat buyers
    fallback = float(expected_avg.mean())

    rfm["expected_avg_value"] = fallback
    rfm.loc[repeat_mask, "expected_avg_value"] = expected_avg

    # ── Combined probabilistic CLV with discounting ────────────────
    discount_rate = 0.10  # annual
    monthly_rate = (1 + discount_rate) ** (1 / 12) - 1

    # Repeat buyers: full CLV formula
    rfm.loc[repeat_mask, "predicted_clv_probabilistic"] = (
        ggf.customer_lifetime_value(
            bgf,
            frequency=repeat["frequency"],
            recency=repeat["recency"],
            T=repeat["T"],
            monetary_value=repeat["monetary_value"],
            time=horizon_months,
            freq="D",
            discount_rate=monthly_rate,
        )
    )

    # One-time buyers: predicted_purchases × fallback value
    otb_mask = rfm["frequency"] == 0
    rfm.loc[otb_mask, "predicted_clv_probabilistic"] = (
        rfm.loc[otb_mask, "predicted_purchases"] * fallback
    )

    print(f"  Probabilistic CLV: "
          f"mean=£{rfm['predicted_clv_probabilistic'].mean():,.2f}, "
          f"total=£{rfm['predicted_clv_probabilistic'].sum():,.0f}")

    return rfm


def score_ml(
    feature_matrix: pd.DataFrame,
    models: dict,
) -> tuple[pd.Series, pd.Series]:
    """Generate ML predictions using XGBoost CLV + churn classifier.

    Validates that the scoring feature matrix matches the training
    feature set before predicting. Catches training-serving skew.

    Args:
        feature_matrix: Unscaled feature matrix (23 features).
        models: Loaded model artifacts.

    Returns:
        Tuple of (predicted_clv_ml, churn_probability_ml) Series.

    Raises:
        ValueError: If feature matrix is missing training features.
    """
    scaler = models["scaler"]
    xgb_clv = models["xgb_clv"]
    churn_model = models["churn"]

    # ── Validate feature alignment with training ───────────────────
    if hasattr(scaler, "feature_names_in_"):
        expected_features = list(scaler.feature_names_in_)
        missing = [f for f in expected_features if f not in feature_matrix.columns]
        extra = [f for f in feature_matrix.columns if f not in expected_features]

        if missing:
            raise ValueError(
                "Scoring feature matrix is missing features expected by "
                "the trained model:\n"
                f"  Missing: {missing}\n"
                f"  Extra:   {extra}\n"
                "This indicates a mismatch between training and scoring "
                "feature computation."
            )

        # Reorder columns to match training
        feature_matrix = feature_matrix[expected_features]
    else:
        print("  WARNING: Scaler has no feature_names_in_ attribute. "
              "Assuming feature order matches training.")

    feature_names = list(feature_matrix.columns)

    # ── Scale features ─────────────────────────────────────────────
    X_scaled = pd.DataFrame(
        scaler.transform(feature_matrix),
        columns=feature_names,
        index=feature_matrix.index,
    )

    # ── CLV prediction ─────────────────────────────────────────────
    predicted_clv = pd.Series(
        xgb_clv.predict(X_scaled),
        index=feature_matrix.index,
        name="predicted_clv_ml",
    )

    # ── Churn prediction ───────────────────────────────────────────
    churn_prob = pd.Series(
        churn_model.predict_proba(X_scaled)[:, 1],
        index=feature_matrix.index,
        name="churn_probability_ml",
    )

    print(f"  ML CLV: mean=£{predicted_clv.mean():,.2f}, "
          f"total=£{predicted_clv.sum():,.0f}")
    print(f"  ML Churn: mean prob={churn_prob.mean():.3f}, "
          f"predicted churned={int((churn_prob >= 0.5).sum()):,}")

    return predicted_clv, churn_prob


# ====================================================================
# 4. SHAP drivers
# ====================================================================

def compute_shap_drivers(
    feature_matrix: pd.DataFrame,
    models: dict,
    top_n: int = 3,
) -> pd.Series:
    """Compute top N SHAP drivers per customer for CLV prediction.

    For each customer, identifies the features with the largest
    absolute SHAP contribution and formats them as a pipe-separated
    string with direction indicators.

    Example output: "frequency(+)|basket_size_trend(+)|return_rate(-)"

    Args:
        feature_matrix: Unscaled feature matrix.
        models: Loaded model artifacts.
        top_n: Number of top drivers to extract per customer.

    Returns:
        Series of pipe-separated driver strings.
    """
    scaler = models["scaler"]
    xgb_clv = models["xgb_clv"]

    feature_names = list(feature_matrix.columns)

    # Scale features (SHAP needs the same input the model sees)
    if hasattr(scaler, "feature_names_in_"):
        feature_matrix = feature_matrix[list(scaler.feature_names_in_)]
        feature_names = list(scaler.feature_names_in_)

    X_scaled = pd.DataFrame(
        scaler.transform(feature_matrix),
        columns=feature_names,
        index=feature_matrix.index,
    )

    print(f"  Computing SHAP values for {len(X_scaled):,} customers...")
    explainer = shap.TreeExplainer(xgb_clv)
    shap_values = explainer.shap_values(X_scaled)

    # Extract top N drivers per customer
    drivers = []
    for i in range(len(shap_values)):
        sv = shap_values[i]
        top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
        parts = []
        for idx in top_idx:
            direction = "+" if sv[idx] >= 0 else "-"
            parts.append(f"{feature_names[idx]}({direction})")
        drivers.append("|".join(parts))

    return pd.Series(
        drivers, index=feature_matrix.index, name="top_3_shap_drivers"
    )


# ====================================================================
# 5. Segment assignment
# ====================================================================

def assign_scoring_segments(
    feature_matrix: pd.DataFrame,
    behavioral: pd.DataFrame,
    rfm: pd.DataFrame,
    predicted_clv_ml: pd.Series,
    churn_prob_ml: pd.Series,
    models: dict,
    config: dict[str, Any],
) -> tuple[pd.Series, pd.Series]:
    """Assign behavioral clusters and CLV tiers.

    Uses the trained K-Means model for cluster assignment and
    assign_clv_tiers() for tier assignment with churn override.

    Args:
        feature_matrix: Full ML feature matrix.
        behavioral: Behavioral features DataFrame.
        rfm: RFM summary with probabilistic predictions.
        predicted_clv_ml: ML CLV predictions.
        churn_prob_ml: ML churn probabilities.
        models: Loaded model artifacts.
        config: Pipeline configuration.

    Returns:
        Tuple of (cluster_labels, tier_labels) Series.
    """
    kmeans = models["kmeans"]
    cluster_scaler = models["cluster_scaler"]

    # ── Cluster assignment ─────────────────────────────────────────
    if hasattr(cluster_scaler, "feature_names_in_"):
        cluster_features = list(cluster_scaler.feature_names_in_)
    else:
        cluster_features = config["segmentation"]["features_used"]

    # Combine sources: RFM + behavioral for clustering features
    all_features = rfm[["frequency", "recency", "monetary_value"]].copy()
    all_features = all_features.join(behavioral, how="left")

    available = [f for f in cluster_features if f in all_features.columns]
    missing = [f for f in cluster_features if f not in all_features.columns]

    if missing:
        print(f"  WARNING: Missing clustering features: {missing}")
        print(f"  Using {len(available)}/{len(cluster_features)} features")

    if not available:
        print("  ERROR: No clustering features available. "
              "Assigning all to cluster 0.")
        clusters = pd.Series(
            0, index=all_features.index, name="behavioral_cluster"
        )
    else:
        X_cluster = all_features[available].fillna(0)
        X_scaled = cluster_scaler.transform(X_cluster)
        labels = kmeans.predict(X_scaled)
        clusters = pd.Series(
            labels, index=all_features.index, name="behavioral_cluster"
        )

    # ── CLV tier assignment ────────────────────────────────────────
    # Convert churn probability to survival probability for the
    # tier function, which checks p_alive < threshold for At-Risk
    survival_prob = 1 - churn_prob_ml
    tiers = assign_clv_tiers(predicted_clv_ml, survival_prob)

    cluster_dist = clusters.value_counts().sort_index().to_dict()
    tier_dist = tiers.value_counts().to_dict()
    print(f"  Clusters: {cluster_dist}")
    print(f"  Tiers: {tier_dist}")

    return clusters, tiers


# ====================================================================
# 6. Main orchestrator
# ====================================================================

def batch_score(
    config: dict[str, Any],
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Execute the full batch scoring pipeline.

    Takes a CSV of transactions (same schema as cleaned training data),
    scores every customer through all models, runs monitoring checks,
    and outputs a scored customer table.

    For demonstration, use the training data as input:
        python -m src.cli score --input data/interim/transactions_calibration.csv

    Args:
        config: Pipeline configuration.
        input_file: Path to input transactions CSV. Defaults to
            data/interim/transactions_calibration.csv.
        output_dir: Directory for output files. Defaults to
            data/predictions/.

    Returns:
        DataFrame of scored customers.

    Raises:
        FileNotFoundError: If input file or model artifacts are missing.
        ValueError: If feature matrix doesn't match training features.
    """
    if input_file is None:
        input_file = PROJECT_ROOT / "data" / "interim" / "transactions_calibration.csv"
    input_path = Path(input_file)
    if output_dir is None:
        output_dir = PATHS["predictions"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BATCH SCORING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load input transactions ────────────────────────────
    print("\n[1/7] Loading input transactions...")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    transactions = load_csv(input_path)
    transactions["invoice_date"] = pd.to_datetime(transactions["invoice_date"])

    total_input_customers = transactions["customer_id"].nunique()
    observation_end = transactions["invoice_date"].max()

    print(f"  {len(transactions):,} transactions, "
          f"{total_input_customers:,} customers")
    print(f"  Date range: {transactions['invoice_date'].min().date()} "
          f"to {observation_end.date()}")

    # ── Step 2: Compute features ───────────────────────────────────
    print("\n[2/7] Computing features...")

    rfm = compute_scoring_rfm(transactions, observation_end, config)
    behavioral = compute_scoring_behavioral(transactions, observation_end)

    # ── Step 3: Load models ────────────────────────────────────────
    print("\n[3/7] Loading trained models...")

    models = load_all_models()

    # ── Step 4: Generate predictions ───────────────────────────────
    print("\n[4/7] Generating predictions...")

    # Probabilistic predictions (adds columns to rfm)
    rfm = score_probabilistic(rfm, models, config)

    # Build the ML feature matrix: 7 RFM-derived + 16 behavioral = 23
    rfm_features = rfm[[
        "frequency", "recency", "T", "monetary_value",
        "predicted_purchases", "p_alive", "expected_avg_value",
    ]].copy()

    feature_matrix = rfm_features.join(behavioral, how="inner")
    scored_customers = len(feature_matrix)

    print(f"  Feature matrix: {scored_customers:,} customers x "
          f"{len(feature_matrix.columns)} features")

    # ML predictions
    predicted_clv_ml, churn_prob_ml = score_ml(feature_matrix, models)

    # ── Step 5: Monitoring ─────────────────────────────────────────
    print("\n[5/7] Running monitoring checks...")

    monitoring_results = run_monitoring(
        new_features=feature_matrix,
        predictions={"predicted_clv_ml": predicted_clv_ml},
        churn_probs=churn_prob_ml,
        total_input_customers=total_input_customers,
        scored_customers=scored_customers,
    )

    report_path = save_monitoring_report(monitoring_results)

    # ── Step 6: Assign segments ────────────────────────────────────
    print("\n[6/7] Assigning segments and tiers...")

    clusters, tiers = assign_scoring_segments(
        feature_matrix, behavioral, rfm,
        predicted_clv_ml, churn_prob_ml,
        models, config,
    )

    # ── Step 7: SHAP drivers ──────────────────────────────────────
    print("\n[7/7] Computing SHAP drivers...")

    shap_drivers = compute_shap_drivers(feature_matrix, models, top_n=3)

    # ── Assemble and save output ───────────────────────────────────
    output = pd.DataFrame(index=feature_matrix.index)
    output.index.name = "customer_id"

    output["predicted_clv_probabilistic"] = (
        rfm.loc[output.index, "predicted_clv_probabilistic"].round(2)
    )
    output["predicted_clv_ml"] = predicted_clv_ml.round(2)
    output["p_alive"] = rfm.loc[output.index, "p_alive"].round(4)
    output["churn_probability_ml"] = churn_prob_ml.round(4)
    output["behavioral_cluster"] = clusters
    output["clv_tier"] = tiers
    output["top_3_shap_drivers"] = shap_drivers

    # Save scored customers
    date_str = datetime.now().strftime("%Y%m%d")
    output_filename = f"scored_customers_{date_str}.csv"
    output_path = output_dir / output_filename

    save_csv(output, output_path, index=True)

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCORING COMPLETE")
    print("=" * 60)
    print(f"  Customers scored:  {scored_customers:,} / "
          f"{total_input_customers:,} input")
    print(f"  Mean CLV (prob):   £{output['predicted_clv_probabilistic'].mean():,.2f}")
    print(f"  Mean CLV (ML):     £{output['predicted_clv_ml'].mean():,.2f}")
    print(f"  Mean churn prob:   {output['churn_probability_ml'].mean():.3f}")
    print(f"  Monitoring status: {monitoring_results['overall_status']}")
    print(f"  Output saved:      {output_path}")
    print(f"  Monitoring report: {report_path}")

    return output