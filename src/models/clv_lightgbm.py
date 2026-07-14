"""LightGBM CLV regressor.

Drop-in alternative to clv_xgboost.py. Same interface:
  train_lightgbm_clv(features, target, config) → dict with model, predictions, scaler, metrics

LightGBM uses leaf-wise tree growth (vs XGBoost's level-wise), which can
converge faster and handle categorical-like features more efficiently.
The primary complexity knob is num_leaves instead of max_depth.
"""

from typing import Any

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.io import PROJECT_ROOT


def train_lightgbm_clv(
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
    scaler: StandardScaler | None = None,
    verbose: bool = True,
) -> dict:
    """Train a LightGBM regressor for CLV prediction.

    Args:
        features: Feature matrix (n_customers × n_features).
        target: Holdout revenue per customer.
        config: Pipeline configuration dictionary.
        scaler: Pre-fitted scaler. If None, fits a new one.
        verbose: Whether to print progress.

    Returns:
        Dictionary with keys:
            model: Trained LGBMRegressor
            predictions: pd.Series of predictions (full dataset)
            scaler: Fitted StandardScaler
            metrics: dict of evaluation metrics
    """
    lgb_config = config.get("models", {}).get("lightgbm_clv", {})
    random_state = lgb_config.get("random_state", 42)

    if verbose:
        print(f"Feature matrix: {features.shape[0]:,} customers, "
              f"{features.shape[1]} features")
        print(f"\nTraining LightGBM CLV regressor")
        print(f"  Samples: {len(features):,}")
        print(f"  Features: {features.shape[1]}")
        print(f"  Target (holdout_revenue): mean={target.mean():.2f}, "
              f"median={target.median():.2f}, std={target.std():.2f}")

    # ── Scale features ──────────────────────────────────────────────
    feature_names = list(features.columns)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(features),
            columns=feature_names,
            index=features.index,
        )
    else:
        X_scaled = pd.DataFrame(
            scaler.transform(features[scaler.feature_names_in_]),
            columns=list(scaler.feature_names_in_),
            index=features.index,
        )

    y = target.loc[X_scaled.index]

    # ── Train/validation split ──────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state,
    )

    if verbose:
        print(f"  Train: {len(X_train):,}  |  Validation: {len(X_val):,}")

    # ── Train LightGBM ──────────────────────────────────────────────
    model = lgb.LGBMRegressor(
        n_estimators=lgb_config.get("n_estimators", 500),
        num_leaves=lgb_config.get("num_leaves", 31),
        learning_rate=lgb_config.get("learning_rate", 0.05),
        feature_fraction=lgb_config.get("feature_fraction", 0.8),
        bagging_fraction=lgb_config.get("bagging_fraction", 0.8),
        bagging_freq=lgb_config.get("bagging_freq", 5),
        min_child_samples=lgb_config.get("min_child_samples", 20),
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
              f"{lgb_config.get('n_estimators', 500)}")

    # ── Predict on full dataset ─────────────────────────────────────
    predictions = pd.Series(
        model.predict(X_scaled),
        index=X_scaled.index,
        name="predicted_clv_lgbm",
    )

    # ── Evaluate ────────────────────────────────────────────────────
    common = y.index.intersection(predictions.index)
    y_common = y.loc[common]
    p_common = predictions.loc[common]

    mae = mean_absolute_error(y_common, p_common)
    rmse = np.sqrt(mean_squared_error(y_common, p_common))
    r2 = r2_score(y_common, p_common)

    # MAPE on customers with revenue > 0
    has_revenue = y_common > 0
    if has_revenue.sum() > 0:
        mape = (np.abs(y_common[has_revenue] - p_common[has_revenue])
                / y_common[has_revenue]).mean() * 100
    else:
        mape = float("nan")

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "best_iteration": best_iteration,
    }

    if verbose:
        print(f"\n  LightGBM CLV Validation:")
        print(f"    MAE:  {mae:.2f}")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    MAPE: {mape:.2f}% (on {has_revenue.sum():,} with revenue > 0)")
        print(f"    R2:   {r2:.4f}")
        print(f"    Mean predicted: {p_common.mean():.2f}")
        print(f"    Mean actual:    {y_common.mean():.2f}")

        # Feature importance
        importance = pd.Series(
            model.feature_importances_,
            index=X_scaled.columns,
        ).sort_values(ascending=False)

        print(f"\n  Top 10 features by importance:")
        for feat, imp in importance.head(10).items():
            print(f"    {feat:40s} {imp}")

    # ── Save model ──────────────────────────────────────────────────
    model_path = PROJECT_ROOT / "models" / "lgbm_clv_model.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    if verbose:
        print(f"\n  Model saved to {model_path}")

    return {
        "model": model,
        "predictions": predictions,
        "scaler": scaler,
        "metrics": metrics,
    }
