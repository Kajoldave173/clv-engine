"""ML churn classifier.

Predicts whether a customer will churn (zero purchases in holdout period)
using RFM + behavioral features + probabilistic model outputs.

The baseline comparison is BG/NBD P(alive) used as a standalone classifier.
The ML model should improve on this by incorporating behavioral signals
the probabilistic model cannot see.
"""

from typing import Any
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from src.utils.io import PROJECT_ROOT


def train_churn_classifier(
    features: pd.DataFrame,
    holdout_frequency: pd.Series,
    p_alive: pd.Series,
    scaler: RobustScaler,
    config: dict[str, Any],
) -> dict:
    """Train an XGBoost binary classifier for churn prediction.

    Args:
        features: Feature matrix (same as CLV model).
        holdout_frequency: Number of holdout purchases per customer.
        p_alive: BG/NBD P(alive) for baseline comparison.
        scaler: Already-fitted scaler from the CLV model.
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary with model, predictions, metrics, and comparison.
    """
    churn_config = config["models"]["churn_classifier"]
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Define target: churned = 1 if zero holdout purchases
    # -----------------------------------------------------------------
    common_idx = features.index.intersection(holdout_frequency.index)
    X = features.loc[common_idx]
    y = (holdout_frequency.loc[common_idx] == 0).astype(int)
    p_alive_aligned = p_alive.loc[common_idx]

    feature_names = list(X.columns)
    n_churned = y.sum()
    n_retained = (y == 0).sum()

    print(f"\nChurn classifier target:")
    print(f"  Churned (holdout_freq == 0): {n_churned:,} ({n_churned/len(y)*100:.1f}%)")
    print(f"  Retained (holdout_freq > 0): {n_retained:,} ({n_retained/len(y)*100:.1f}%)")

    # -----------------------------------------------------------------
    # Scale features (reuse the CLV model's fitted scaler)
    # -----------------------------------------------------------------
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=feature_names,
        index=X.index,
    )

    # -----------------------------------------------------------------
    # Train/validation split
    # -----------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y,
    )

    print(f"  Train: {len(X_train):,}  |  Validation: {len(X_val):,}")

    # -----------------------------------------------------------------
    # Compute scale_pos_weight for class imbalance
    # -----------------------------------------------------------------
    if churn_config["scale_pos_weight"] == "auto":
        spw = n_retained / max(n_churned, 1)
    else:
        spw = float(churn_config["scale_pos_weight"])
    print(f"  scale_pos_weight: {spw:.3f}")

    # -----------------------------------------------------------------
    # Train XGBoost classifier
    # -----------------------------------------------------------------
    model = xgb.XGBClassifier(
        n_estimators=churn_config["n_estimators"],
        max_depth=churn_config["max_depth"],
        scale_pos_weight=spw,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"  Best iteration: {model.best_iteration} / {churn_config['n_estimators']}")

    # -----------------------------------------------------------------
    # Predict on all customers
    # -----------------------------------------------------------------
    churn_prob_ml = pd.Series(
        model.predict_proba(X_scaled)[:, 1],
        index=X.index,
        name="churn_prob_ml",
    )

    # -----------------------------------------------------------------
    # ML classifier metrics
    # -----------------------------------------------------------------
    ml_auc = roc_auc_score(y, churn_prob_ml)

    # Find optimal threshold via Youden's J statistic
    fpr_ml, tpr_ml, thresholds_ml = roc_curve(y, churn_prob_ml)
    j_scores = tpr_ml - fpr_ml
    best_threshold_idx = np.argmax(j_scores)
    best_threshold = thresholds_ml[best_threshold_idx]

    y_pred_ml = (churn_prob_ml >= best_threshold).astype(int)
    ml_f1 = f1_score(y, y_pred_ml)

    print(f"\n  ML Churn Classifier:")
    print(f"    AUC-ROC:          {ml_auc:.4f}")
    print(f"    Optimal threshold: {best_threshold:.3f} (Youden's J)")
    print(f"    F1 score:          {ml_f1:.4f}")
    print(f"\n  Classification report (ML):")
    print(classification_report(y, y_pred_ml, target_names=["Retained", "Churned"]))

    # -----------------------------------------------------------------
    # P(alive) baseline metrics
    # P(alive) predicts "alive" — so churn_prob = 1 - P(alive)
    # -----------------------------------------------------------------
    churn_prob_baseline = 1 - p_alive_aligned
    baseline_auc = roc_auc_score(y, churn_prob_baseline)

    fpr_base, tpr_base, thresholds_base = roc_curve(y, churn_prob_baseline)
    j_base = tpr_base - fpr_base
    best_base_idx = np.argmax(j_base)
    best_base_threshold = thresholds_base[best_base_idx]

    y_pred_base = (churn_prob_baseline >= best_base_threshold).astype(int)
    base_f1 = f1_score(y, y_pred_base)

    print(f"  P(alive) Baseline:")
    print(f"    AUC-ROC:          {baseline_auc:.4f}")
    print(f"    Optimal threshold: {best_base_threshold:.3f}")
    print(f"    F1 score:          {base_f1:.4f}")
    print(f"\n  Classification report (P(alive)):")
    print(classification_report(y, y_pred_base, target_names=["Retained", "Churned"]))

    # -----------------------------------------------------------------
    # Comparison summary
    # -----------------------------------------------------------------
    auc_improvement = (ml_auc - baseline_auc) / baseline_auc * 100

    print(f"  {'Metric':<20} {'P(alive)':>12} {'ML Classifier':>15} {'Improvement':>12}")
    print(f"  {'-' * 60}")
    print(f"  {'AUC-ROC':<20} {baseline_auc:>12.4f} {ml_auc:>15.4f} {auc_improvement:>11.1f}%")
    print(f"  {'F1':<20} {base_f1:>12.4f} {ml_f1:>15.4f}")

    # -----------------------------------------------------------------
    # ROC curve comparison plot
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 7))
    plt.plot(fpr_base, tpr_base, label=f"P(alive) baseline (AUC={baseline_auc:.3f})",
             linewidth=2, linestyle="--", color="#888888")
    plt.plot(fpr_ml, tpr_ml, label=f"ML classifier (AUC={ml_auc:.3f})",
             linewidth=2, color="#2563eb")
    plt.plot([0, 1], [0, 1], linestyle=":", color="#cccccc", linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Churn Prediction: ROC Curve Comparison", fontsize=14)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_churn_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ROC curve saved to {figures_dir / 'roc_churn_comparison.png'}")

    # -----------------------------------------------------------------
    # Feature importance for churn
    # -----------------------------------------------------------------
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)

    print(f"\n  Top 10 churn features:")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:40s} {imp:.4f}")

    # -----------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------
    model_path = PROJECT_ROOT / "models" / "churn_model.json"
    model.save_model(str(model_path))
    print(f"\n  Churn model saved to {model_path}")

    return {
        "model": model,
        "churn_prob_ml": churn_prob_ml,
        "best_threshold": best_threshold,
        "metrics": {
            "ml_auc": ml_auc,
            "ml_f1": ml_f1,
            "baseline_auc": baseline_auc,
            "baseline_f1": base_f1,
            "auc_improvement": auc_improvement,
        },
    }