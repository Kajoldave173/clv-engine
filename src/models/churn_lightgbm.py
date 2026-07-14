"""LightGBM churn classifier.

Drop-in alternative to churn_classifier.py. Same interface:
  train_lightgbm_churn(features, holdout_frequency, p_alive, scaler, config)
  → dict with model, churn_prob_ml, best_threshold, metrics
"""

from typing import Any

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    roc_curve,
)

from src.utils.io import PROJECT_ROOT


def train_lightgbm_churn(
    features: pd.DataFrame,
    holdout_frequency: pd.Series,
    p_alive: pd.Series,
    scaler: StandardScaler,
    config: dict[str, Any],
    verbose: bool = True,
) -> dict:
    """Train a LightGBM classifier for churn prediction.

    Args:
        features: Feature matrix (same as CLV model).
        holdout_frequency: Number of purchases in holdout period per customer.
        p_alive: BG/NBD P(alive) predictions (used as baseline comparison).
        scaler: Pre-fitted scaler from the CLV model.
        config: Pipeline configuration dictionary.
        verbose: Whether to print progress.

    Returns:
        Dictionary with keys:
            model: Trained LGBMClassifier
            churn_prob_ml: pd.Series of churn probabilities
            best_threshold: Optimal classification threshold
            metrics: dict of evaluation metrics
    """
    lgb_config = config.get("models", {}).get("lightgbm_churn", {})
    random_state = lgb_config.get("random_state", 42)

    # ── Create binary churn target ──────────────────────────────────
    churned = (holdout_frequency == 0).astype(int)
    churned.name = "churned"

    n_retained = (churned == 0).sum()
    n_churned = (churned == 1).sum()

    if verbose:
        print(f"\nChurn classifier target (LightGBM):")
        print(f"  Churned (holdout_freq == 0): {n_churned:,} ({n_churned/len(churned)*100:.1f}%)")
        print(f"  Retained (holdout_freq > 0): {n_retained:,} ({n_retained/len(churned)*100:.1f}%)")

    # ── Scale features using CLV model's scaler ─────────────────────
    feature_order = list(scaler.feature_names_in_)
    X_scaled = pd.DataFrame(
        scaler.transform(features[feature_order]),
        columns=feature_order,
        index=features.index,
    )
    y = churned.loc[X_scaled.index]

    # ── Train/validation split ──────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state, stratify=y,
    )

    if verbose:
        print(f"  Train: {len(X_train):,}  |  Validation: {len(X_val):,}")

    # ── Class imbalance handling ────────────────────────────────────
    spw = lgb_config.get("scale_pos_weight", "auto")
    if spw == "auto":
        spw = float(n_retained / max(n_churned, 1))
    else:
        spw = float(spw)

    if verbose:
        print(f"  scale_pos_weight: {spw:.3f}")

    # ── Train LightGBM ──────────────────────────────────────────────
    model = lgb.LGBMClassifier(
        n_estimators=lgb_config.get("n_estimators", 300),
        num_leaves=lgb_config.get("num_leaves", 31),
        learning_rate=lgb_config.get("learning_rate", 0.05),
        feature_fraction=lgb_config.get("feature_fraction", 0.8),
        bagging_fraction=lgb_config.get("bagging_fraction", 0.8),
        bagging_freq=lgb_config.get("bagging_freq", 5),
        min_child_samples=lgb_config.get("min_child_samples", 20),
        scale_pos_weight=spw,
        random_state=random_state,
        verbosity=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=lgb_config.get("early_stopping_rounds", 50),
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iteration = model.best_iteration_
    if verbose:
        print(f"  Best iteration: {best_iteration} / "
              f"{lgb_config.get('n_estimators', 300)}")

    # ── Predict churn probabilities ─────────────────────────────────
    churn_prob_ml = pd.Series(
        model.predict_proba(X_scaled)[:, 1],
        index=X_scaled.index,
        name="churn_probability_ml",
    )

    # ── Evaluate ML classifier ──────────────────────────────────────
    ml_auc = roc_auc_score(y, churn_prob_ml)

    # Optimal threshold via Youden's J
    fpr_ml, tpr_ml, thresholds_ml = roc_curve(y, churn_prob_ml)
    j_scores = tpr_ml - fpr_ml
    best_idx = np.argmax(j_scores)
    best_threshold = float(thresholds_ml[best_idx])
    ml_preds = (churn_prob_ml >= best_threshold).astype(int)
    ml_f1 = f1_score(y, ml_preds)

    if verbose:
        print(f"\n  LightGBM Churn Classifier:")
        print(f"    AUC-ROC:          {ml_auc:.4f}")
        print(f"    Optimal threshold: {best_threshold:.3f} (Youden's J)")
        print(f"    F1 score:          {ml_f1:.4f}")

        report = classification_report(
            y, ml_preds,
            target_names=["Retained", "Churned"],
        )
        print(f"\n  Classification report (LightGBM):")
        print(report)

    # ── P(alive) baseline comparison ────────────────────────────────
    # P(alive) predicts "alive" → invert for churn: churn_prob = 1 - p_alive
    p_churn_baseline = 1 - p_alive.loc[y.index]
    baseline_auc = roc_auc_score(y, p_churn_baseline)

    fpr_base, tpr_base, thresholds_base = roc_curve(y, p_churn_baseline)
    j_base = tpr_base - fpr_base
    best_base_idx = np.argmax(j_base)
    base_threshold = float(thresholds_base[best_base_idx])
    base_preds = (p_churn_baseline >= base_threshold).astype(int)
    base_f1 = f1_score(y, base_preds)

    auc_improvement = (ml_auc - baseline_auc) / baseline_auc * 100

    if verbose:
        print(f"  Metric                   P(alive)   LightGBM       Improvement")
        print(f"  ------------------------------------------------------------")
        print(f"  AUC-ROC                    {baseline_auc:.4f}          {ml_auc:.4f}        {auc_improvement:.1f}%")
        print(f"  F1                         {base_f1:.4f}          {ml_f1:.4f}")

    metrics = {
        "ml_auc": ml_auc,
        "ml_f1": ml_f1,
        "baseline_auc": baseline_auc,
        "baseline_f1": base_f1,
        "auc_improvement": auc_improvement,
        "best_iteration": best_iteration,
    }

    # ── Save model ──────────────────────────────────────────────────
    model_path = PROJECT_ROOT / "models" / "lgbm_churn_model.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    if verbose:
        print(f"\n  Churn model saved to {model_path}")

    return {
        "model": model,
        "churn_prob_ml": churn_prob_ml,
        "best_threshold": best_threshold,
        "metrics": metrics,
    }
