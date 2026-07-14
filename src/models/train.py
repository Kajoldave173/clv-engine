"""Training orchestration: loads data, fits models, saves artifacts.

Phase 2: Probabilistic models (BG/NBD + Gamma-Gamma)
Phase 3: ML models (XGBoost CLV regressor + churn classifier)

MLflow logging:
  - Each model gets its own MLflow run for comparison on DagsHub
  - Run "probabilistic-baseline": BG/NBD + Gamma-Gamma params and metrics
  - Run "xgboost-clv-default": XGBoost CLV hyperparams and metrics
  - Run "xgboost-churn-default": Churn classifier hyperparams and metrics
"""

from typing import Any

import pandas as pd
import numpy as np
import mlflow

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
from src.experiment.mlflow_config import (
    load_env_file,
    setup_mlflow,
    log_params,
    log_metrics,
    log_model_artifact,
)


def train_all_models(config: dict[str, Any]) -> dict:
    """Train all models in sequence, logging each to MLflow.

    Pipeline:
      1. Load RFM summary and behavioral features
      2. Fit BG/NBD (adds predicted_purchases, p_alive to RFM)
      3. Fit Gamma-Gamma (adds expected_avg_value, predicted_clv to RFM)
      4. Merge RFM + behavioral into ML feature matrix
      5. Train XGBoost CLV regressor
      6. Train churn classifier
      7. Compare all baselines

    Each model stage is wrapped in its own MLflow run so that
    DagsHub's comparison UI can line them up side by side.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary of results and metrics.
    """
    results = {}

    # ── MLflow setup ────────────────────────────────────────────────
    load_env_file()
    setup_mlflow(config)

    # ── Load data ───────────────────────────────────────────────────
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

    # ── Log probabilistic models to MLflow ──────────────────────────
    prob_config = config.get("models", {}).get("probabilistic", {})

    # Compute probabilistic metrics from the rfm DataFrame
    bgnbd_mae = np.abs(
        rfm["predicted_purchases"] - rfm["holdout_frequency"]
    ).mean()
    bgnbd_rmse = np.sqrt(
        ((rfm["predicted_purchases"] - rfm["holdout_frequency"]) ** 2).mean()
    )

    has_revenue = rfm["holdout_revenue"] > 0
    prob_clv_mae = np.abs(
        rfm["predicted_clv"] - rfm["holdout_revenue"]
    ).mean()
    prob_clv_rmse = np.sqrt(
        ((rfm["predicted_clv"] - rfm["holdout_revenue"]) ** 2).mean()
    )

    with mlflow.start_run(run_name="probabilistic-baseline"):
        log_params({
            "model_type": "BG/NBD + Gamma-Gamma",
            "penalizer_coef": prob_config.get("penalizer_coef", 0.001),
            "prediction_horizon_months": prob_config.get(
                "prediction_horizon_months", 6
            ),
            "discount_rate": prob_config.get("discount_rate", 0.1),
            "n_customers": len(rfm),
            "n_repeat_buyers": int((rfm["frequency"] >= 1).sum()),
        })
        log_metrics({
            "bgnbd_mae": bgnbd_mae,
            "bgnbd_rmse": bgnbd_rmse,
            "clv_mae": prob_clv_mae,
            "clv_rmse": prob_clv_rmse,
            "p_alive_mean": rfm["p_alive"].mean(),
            "predicted_clv_mean": rfm["predicted_clv"].mean(),
        })
        log_model_artifact(PROJECT_ROOT / "models" / "bgnbd_model.pkl")
        log_model_artifact(PROJECT_ROOT / "models" / "gg_model.pkl")

    print("\n  [MLflow] Logged probabilistic baseline")

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

    # ── Log XGBoost CLV to MLflow ───────────────────────────────────
    xgb_config = config.get("models", {}).get("xgboost_clv", {})

    # Compute comparison metrics
    ml_pred = xgb_results["predictions"]
    common = target.index.intersection(ml_pred.index)
    ml_mae = np.abs(ml_pred.loc[common] - target.loc[common]).mean()
    ml_rmse = np.sqrt(
        ((ml_pred.loc[common] - target.loc[common]) ** 2).mean()
    )
    improvement = (prob_clv_mae - ml_mae) / prob_clv_mae * 100

    with mlflow.start_run(run_name="xgboost-clv-default"):
        log_params({
            "model_type": "XGBoost",
            "task": "clv_regression",
            "n_estimators": xgb_config.get("n_estimators", 500),
            "max_depth": xgb_config.get("max_depth", 6),
            "learning_rate": xgb_config.get("learning_rate", 0.05),
            "subsample": xgb_config.get("subsample", 0.8),
            "colsample_bytree": xgb_config.get("colsample_bytree", 0.8),
            "early_stopping_rounds": xgb_config.get(
                "early_stopping_rounds", 50
            ),
            "n_features": feature_matrix.shape[1],
            "n_customers": len(feature_matrix),
        })
        log_metrics({
            "mae": ml_mae,
            "rmse": ml_rmse,
            "mae_improvement_over_prob": improvement,
        })
        # Log R² if available in results
        if "metrics" in xgb_results:
            for k, v in xgb_results["metrics"].items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

        log_model_artifact(PROJECT_ROOT / "models" / "xgb_clv_model.json")
        log_model_artifact(PROJECT_ROOT / "models" / "scaler.pkl")

    print("  [MLflow] Logged XGBoost CLV")

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

    # ── Log churn classifier to MLflow ──────────────────────────────
    churn_config = config.get("models", {}).get("churn_classifier", {})
    churn_metrics = churn_results.get("metrics", {})

    with mlflow.start_run(run_name="xgboost-churn-default"):
        log_params({
            "model_type": "XGBoost",
            "task": "churn_classification",
            "n_estimators": churn_config.get("n_estimators", 300),
            "max_depth": churn_config.get("max_depth", 4),
            "learning_rate": churn_config.get("learning_rate", 0.05),
            "scale_pos_weight": churn_config.get("scale_pos_weight", "auto"),
            "n_features": feature_matrix.shape[1],
            "churn_rate": float(
                (rfm["holdout_frequency"] == 0).mean()
            ),
        })
        log_metrics({
            "auc_roc": churn_metrics.get("ml_auc", 0),
            "f1": churn_metrics.get("ml_f1", 0),
            "baseline_auc_palive": churn_metrics.get("baseline_auc", 0),
            "auc_improvement_over_palive": churn_metrics.get(
                "auc_improvement", 0
            ),
        })
        log_model_artifact(PROJECT_ROOT / "models" / "churn_model.json")

        # Log the ROC curve plot if it exists
        roc_path = PROJECT_ROOT / "reports" / "figures" / "roc_churn_comparison.png"
        log_model_artifact(roc_path)

    print("  [MLflow] Logged churn classifier")

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
