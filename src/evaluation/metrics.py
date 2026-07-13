"""Model evaluation metrics and comparison.

Loads saved predictions and computes summary metrics.
This runs independently of training so you can evaluate
without retraining.
"""

from typing import Any

import pandas as pd
import numpy as np

from src.utils.io import PROJECT_ROOT, load_csv


def evaluate_models(config: dict[str, Any]) -> dict:
    """Load saved predictions and compute evaluation metrics.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary of metric results.
    """
    predictions_path = PROJECT_ROOT / "data" / "processed" / "customer_predictions.csv"
    predictions = load_csv(predictions_path, index_col="customer_id")

    actual = predictions["holdout_revenue"]
    prob_pred = predictions["predicted_clv"]
    ml_pred = predictions["predicted_clv_ml"]

    # -----------------------------------------------------------------
    # Probabilistic model metrics
    # -----------------------------------------------------------------
    prob_mae = np.abs(prob_pred - actual).mean()
    prob_rmse = np.sqrt(((prob_pred - actual) ** 2).mean())
    prob_r2 = 1 - ((actual - prob_pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()

    # -----------------------------------------------------------------
    # ML model metrics
    # -----------------------------------------------------------------
    ml_mae = np.abs(ml_pred - actual).mean()
    ml_rmse = np.sqrt(((ml_pred - actual) ** 2).mean())
    ml_r2 = 1 - ((actual - ml_pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()

    # -----------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------
    mae_improvement = (prob_mae - ml_mae) / prob_mae * 100
    rmse_improvement = (prob_rmse - ml_rmse) / prob_rmse * 100

    print("=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\n  {'Metric':<12} {'Probabilistic':>15} {'XGBoost ML':>15} {'Improvement':>12}")
    print(f"  {'-' * 55}")
    print(f"  {'MAE':<12} {prob_mae:>15.2f} {ml_mae:>15.2f} {mae_improvement:>11.1f}%")
    print(f"  {'RMSE':<12} {prob_rmse:>15.2f} {ml_rmse:>15.2f} {rmse_improvement:>11.1f}%")
    print(f"  {'R2':<12} {prob_r2:>15.4f} {ml_r2:>15.4f}")
    print(f"\n  {'Mean pred':<12} {prob_pred.mean():>15.2f} {ml_pred.mean():>15.2f}")
    print(f"  {'Mean actual':<12} {actual.mean():>15.2f}")
    print(f"  {'Customers':<12} {len(actual):>15,}")

    # -----------------------------------------------------------------
    # Revenue capture analysis
    # -----------------------------------------------------------------
    total_actual = actual.sum()
    prob_capture = prob_pred.sum() / total_actual * 100
    ml_capture = ml_pred.sum() / total_actual * 100

    print(f"\n  Revenue capture:")
    print(f"    Actual total holdout revenue:  {total_actual:>12,.2f}")
    print(f"    Probabilistic predicted total: {prob_pred.sum():>12,.2f} ({prob_capture:.1f}%)")
    print(f"    ML predicted total:            {ml_pred.sum():>12,.2f} ({ml_capture:.1f}%)")

    # -----------------------------------------------------------------
    # Churn prediction accuracy (using P(alive) as classifier)
    # -----------------------------------------------------------------
    p_alive = predictions["p_alive"]
    actually_churned = actual == 0
    predicted_alive = p_alive > 0.5

    true_positives = (predicted_alive & ~actually_churned).sum()
    false_positives = (predicted_alive & actually_churned).sum()
    true_negatives = (~predicted_alive & actually_churned).sum()
    false_negatives = (~predicted_alive & ~actually_churned).sum()

    accuracy = (true_positives + true_negatives) / len(actual)
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)

    print(f"\n  P(alive) as churn classifier (threshold=0.5):")
    print(f"    Accuracy:  {accuracy:.3f}")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall:    {recall:.3f}")

    return {
        "prob_mae": prob_mae,
        "prob_rmse": prob_rmse,
        "ml_mae": ml_mae,
        "ml_rmse": ml_rmse,
        "mae_improvement": mae_improvement,
    }