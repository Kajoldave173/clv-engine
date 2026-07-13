"""Training orchestration: loads data, fits models, saves artifacts.

Phase 2: Probabilistic models (BG/NBD + Gamma-Gamma)
Phase 3: ML models (XGBoost CLV regressor + churn classifier)
"""

from typing import Any

import pandas as pd
import numpy as np

from src.utils.io import PROJECT_ROOT, load_csv, save_csv
from src.models.probabilistic import (
    fit_bgnbd,
    predict_and_validate_bgnbd,
    save_bgnbd_model,
    fit_gamma_gamma,
    predict_clv,
    save_gg_model,
)
from src.models.clv_xgboost import build_feature_matrix, train_xgboost_clv
from src.models.churn_classifier import train_churn_classifier


def train_all_models(config: dict[str, Any]) -> dict:
    """Train all models in sequence.

    Pipeline:
      1. Load RFM summary and behavioral features
      2. Fit BG/NBD (adds predicted_purchases, p_alive to RFM)
      3. Fit Gamma-Gamma (adds expected_avg_value, predicted_clv to RFM)
      4. Merge RFM + behavioral into ML feature matrix
      5. Train XGBoost CLV regressor
      6. Train churn classifier
      7. Compare all baselines

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary of results and metrics.
    """
    results = {}

    # Load data
    rfm_path = PROJECT_ROOT / "data" / "processed" / "rfm_summary.csv"
    behavioral_path = PROJECT_ROOT / "data" / "processed" / "features_behavioral.csv"

    rfm = load_csv(rfm_path, index_col="customer_id")
    behavioral = load_csv(behavioral_path, index_col="customer_id")

    print(f"Loaded RFM summary: {len(rfm):,} customers")
    print(f"Loaded behavioral features: {len(behavioral):,} customers, "
          f"{behavioral.shape[1]} features\n")

    # =================================================================
    # Phase 2: Probabilistic models
    # =================================================================
    print("=" * 50)
    print("BG/NBD MODEL")
    print("=" * 50)

    bgf = fit_bgnbd(rfm, config)
    rfm = predict_and_validate_bgnbd(bgf, rfm, config)
    save_bgnbd_model(bgf)
    results["bgnbd"] = bgf

    print("\n" + "=" * 50)
    print("GAMMA-GAMMA MODEL")
    print("=" * 50)

    ggf = fit_gamma_gamma(rfm, config)
    rfm = predict_clv(bgf, ggf, rfm, config)
    save_gg_model(ggf)
    results["gamma_gamma"] = ggf

    # Save RFM with probabilistic predictions
    save_csv(rfm, rfm_path, index=True)

    # =================================================================
    # Phase 3: ML models
    # =================================================================
    print("\n" + "=" * 50)
    print("XGBOOST CLV REGRESSOR")
    print("=" * 50)

    # Build merged feature matrix
    feature_matrix = build_feature_matrix(rfm, behavioral)

    # Target: actual holdout revenue
    target = rfm["holdout_revenue"]

    # Train XGBoost CLV
    xgb_results = train_xgboost_clv(feature_matrix, target, config)
    results["xgboost_clv"] = xgb_results

    # -----------------------------------------------------------------
    # Churn classifier
    # -----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("CHURN CLASSIFIER")
    print("=" * 50)

    churn_results = train_churn_classifier(
        features=feature_matrix,
        holdout_frequency=rfm["holdout_frequency"],
        p_alive=rfm["p_alive"],
        scaler=xgb_results["scaler"],
        config=config,
    )
    results["churn"] = churn_results

    # =================================================================
    # Baseline comparison: probabilistic vs ML
    # =================================================================
    print("\n" + "=" * 50)
    print("CLV BASELINE COMPARISON")
    print("=" * 50)

    actual = rfm["holdout_revenue"]
    prob_pred = rfm["predicted_clv"]
    ml_pred = xgb_results["predictions"]

    common = actual.index.intersection(ml_pred.index)
    actual = actual.loc[common]
    prob_pred = prob_pred.loc[common]
    ml_pred = ml_pred.loc[common]

    prob_mae = np.abs(prob_pred - actual).mean()
    ml_mae = np.abs(ml_pred - actual).mean()
    improvement = (prob_mae - ml_mae) / prob_mae * 100

    prob_rmse = np.sqrt(((prob_pred - actual) ** 2).mean())
    ml_rmse = np.sqrt(((ml_pred - actual) ** 2).mean())
    rmse_improvement = (prob_rmse - ml_rmse) / prob_rmse * 100

    print(f"\n  {'Metric':<10} {'Probabilistic':>15} {'XGBoost':>15} {'Improvement':>15}")
    print(f"  {'-'*55}")
    print(f"  {'MAE':<10} {prob_mae:>15.2f} {ml_mae:>15.2f} {improvement:>14.1f}%")
    print(f"  {'RMSE':<10} {prob_rmse:>15.2f} {ml_rmse:>15.2f} {rmse_improvement:>14.1f}%")
    print(f"  {'Mean pred':<10} {prob_pred.mean():>15.2f} {ml_pred.mean():>15.2f}")
    print(f"  {'Mean actual':<10} {actual.mean():>15.2f}")

    # Save final customer-level predictions (including churn)
    predictions_df = rfm[["holdout_revenue", "predicted_clv", "p_alive"]].copy()
    predictions_df["predicted_clv_ml"] = ml_pred
    predictions_df["churn_prob_ml"] = churn_results["churn_prob_ml"]
    pred_path = PROJECT_ROOT / "data" / "processed" / "customer_predictions.csv"
    save_csv(predictions_df, pred_path, index=True)
    print(f"\n  Customer predictions saved to {pred_path}")

    return results