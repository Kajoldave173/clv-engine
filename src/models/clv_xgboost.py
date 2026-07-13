"""XGBoost CLV regressor.

Predicts holdout-period revenue using RFM features, behavioral features,
and probabilistic model outputs (P(alive), predicted purchases).
This is the ML model that aims to beat the BG/NBD + Gamma-Gamma baseline.
"""

from typing import Any
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib

from src.utils.io import PROJECT_ROOT


def build_feature_matrix(
    rfm: pd.DataFrame,
    behavioral: pd.DataFrame,
) -> pd.DataFrame:
    """Merge RFM, behavioral, and probabilistic outputs into one matrix.

    The RFM table already contains BG/NBD predictions (predicted_purchases,
    p_alive) and Gamma-Gamma outputs (expected_avg_value, predicted_clv)
    from the probabilistic training step.

    We use predicted_purchases and p_alive as ML features (stacking),
    but NOT predicted_clv -- that's what we're trying to beat, not
    a feature to feed into our model.

    Args:
        rfm: RFM summary with probabilistic model predictions.
        behavioral: Behavioral features from Phase 3.1.

    Returns:
        Feature matrix ready for XGBoost training.
    """
    # RFM columns to use as features
    rfm_features = rfm[[
        "frequency", "recency", "T", "monetary_value",
        "predicted_purchases", "p_alive", "expected_avg_value",
    ]].copy()

    # Merge with behavioral features
    features = rfm_features.join(behavioral, how="inner")

    print(f"Feature matrix: {len(features):,} customers, "
          f"{features.shape[1]} features")

    # Check for nulls and infinities
    null_count = features.isna().sum().sum()
    inf_count = np.isinf(features.select_dtypes(include=np.number)).sum().sum()
    if null_count > 0:
        print(f"  WARNING: {null_count} null values found")
    if inf_count > 0:
        print(f"  WARNING: {inf_count} infinite values found")
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

    return features


def train_xgboost_clv(
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
) -> dict:
    """Train XGBoost regressor to predict holdout revenue.

    Args:
        features: Feature matrix (customer_id as index).
        target: Holdout revenue per customer (same index).
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary with model, scaler, metrics, and feature names.
    """
    xgb_config = config["models"]["xgboost_clv"]

    # Align features and target
    common_idx = features.index.intersection(target.index)
    X = features.loc[common_idx]
    y = target.loc[common_idx]
    feature_names = list(X.columns)

    print(f"\nTraining XGBoost CLV regressor")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Target (holdout_revenue): mean={y.mean():.2f}, "
          f"median={y.median():.2f}, std={y.std():.2f}")

    # -----------------------------------------------------------------
    # Scale features
    # XGBoost doesn't need scaling, but it helps with SHAP
    # interpretation and makes the pipeline consistent if we
    # swap in a linear model later.
    # -----------------------------------------------------------------
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_names,
        index=X.index,
    )

    # -----------------------------------------------------------------
    # Train/validation split for early stopping
    # This is an internal split for hyperparameter tuning only.
    # The real validation is against the holdout period.
    # -----------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42,
    )

    print(f"  Train: {len(X_train):,}  |  Validation: {len(X_val):,}")

    # -----------------------------------------------------------------
    # Train with early stopping
    # -----------------------------------------------------------------
    model = xgb.XGBRegressor(
        n_estimators=xgb_config["n_estimators"],
        max_depth=xgb_config["max_depth"],
        learning_rate=xgb_config["learning_rate"],
        subsample=xgb_config["subsample"],
        colsample_bytree=xgb_config["colsample_bytree"],
        eval_metric=xgb_config["eval_metric"],
        early_stopping_rounds=xgb_config["early_stopping_rounds"],
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iteration = model.best_iteration
    print(f"  Best iteration: {best_iteration} / {xgb_config['n_estimators']}")

    # -----------------------------------------------------------------
    # Predict on ALL customers (not just validation split)
    # This gives us predictions comparable to the probabilistic baseline.
    # -----------------------------------------------------------------
    predictions = pd.Series(
        model.predict(X_scaled),
        index=X.index,
        name="predicted_clv_ml",
    )

    # Clip negative predictions to 0 (revenue can't be negative)
    predictions = predictions.clip(lower=0)

    # -----------------------------------------------------------------
    # Metrics: compare against actual holdout revenue
    # -----------------------------------------------------------------
    mae = np.abs(predictions - y).mean()
    rmse = np.sqrt(((predictions - y) ** 2).mean())

    nonzero = y > 0
    mape = np.abs((predictions[nonzero] - y[nonzero]) / y[nonzero]).mean()

    r2 = 1 - ((y - predictions) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    print(f"\n  XGBoost CLV Validation:")
    print(f"    MAE:  {mae:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    MAPE: {mape:.2%} (on {nonzero.sum():,} with revenue > 0)")
    print(f"    R2:   {r2:.4f}")
    print(f"    Mean predicted: {predictions.mean():.2f}")
    print(f"    Mean actual:    {y.mean():.2f}")

    # -----------------------------------------------------------------
    # Feature importance (gain-based)
    # -----------------------------------------------------------------
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)

    print(f"\n  Top 10 features by importance:")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:40s} {imp:.4f}")

    # -----------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "xgb_clv_model.json"
    model.save_model(str(model_path))
    print(f"\n  Model saved to {model_path}")

    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")

    return {
        "model": model,
        "scaler": scaler,
        "predictions": predictions,
        "feature_names": feature_names,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "best_iteration": best_iteration,
        },
    }