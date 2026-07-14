"""Optuna hyperparameter tuning for CLV and churn models.

Runs Bayesian optimization over the search spaces defined in params.yaml.
Each trial uses 5-fold cross-validation and logs to MLflow as a nested run.

Usage:
    from src.experiment.tuning import run_all_tuning
    results = run_all_tuning(config)

    # Or from CLI:
    python -m src.cli tune
"""

from typing import Any

import numpy as np
import pandas as pd
import optuna
import mlflow
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.utils.io import PROJECT_ROOT, load_csv
from src.models.clv_xgboost import build_feature_matrix, train_xgboost_clv
from src.models.churn_classifier import train_churn_classifier
from src.models.clv_lightgbm import train_lightgbm_clv
from src.models.churn_lightgbm import train_lightgbm_churn
from src.experiment.mlflow_config import (
    load_env_file,
    setup_mlflow,
    log_params,
    log_metrics,
    log_model_artifact,
)

import warnings
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

# =====================================================================
# Optuna objective functions — one per model variant
# =====================================================================

def _xgboost_clv_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
) -> float:
    """Objective function for XGBoost CLV regressor tuning.

    Samples hyperparameters, evaluates with 5-fold CV, returns mean RMSE.
    """
    space = config.get("optuna", {}).get("xgboost_clv_space", {})
    cv_folds = config.get("optuna", {}).get("cv_folds", 5)

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", *space.get("n_estimators", [100, 1000])
        ),
        "max_depth": trial.suggest_int(
            "max_depth", *space.get("max_depth", [3, 10])
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", *space.get("learning_rate", [0.01, 0.3]), log=True
        ),
        "subsample": trial.suggest_float(
            "subsample", *space.get("subsample", [0.6, 1.0])
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *space.get("colsample_bytree", [0.5, 1.0])
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", *space.get("min_child_weight", [1, 10])
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", *space.get("reg_alpha", [1e-8, 10.0]), log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", *space.get("reg_lambda", [1e-8, 10.0]), log=True
        ),
    }

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(
            **params,
            early_stopping_rounds=50,
            eval_metric="rmse",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        preds = model.predict(X_va)
        rmse = np.sqrt(mean_squared_error(y_va, preds))
        rmse_scores.append(rmse)

    return float(np.mean(rmse_scores))


def _xgboost_churn_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
) -> float:
    """Objective for XGBoost churn classifier. Returns mean AUC (to maximize)."""
    space = config.get("optuna", {}).get("xgboost_churn_space", {})
    cv_folds = config.get("optuna", {}).get("cv_folds", 5)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    spw = float(n_neg / max(n_pos, 1))

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", *space.get("n_estimators", [100, 800])
        ),
        "max_depth": trial.suggest_int(
            "max_depth", *space.get("max_depth", [3, 8])
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", *space.get("learning_rate", [0.01, 0.3]), log=True
        ),
        "subsample": trial.suggest_float(
            "subsample", *space.get("subsample", [0.6, 1.0])
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *space.get("colsample_bytree", [0.5, 1.0])
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", *space.get("min_child_weight", [1, 10])
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", *space.get("reg_alpha", [1e-8, 10.0]), log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", *space.get("reg_lambda", [1e-8, 10.0]), log=True
        ),
    }

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=spw,
            early_stopping_rounds=50,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        probs = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, probs)
        auc_scores.append(auc)

    return float(np.mean(auc_scores))


def _lightgbm_clv_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
) -> float:
    """Objective for LightGBM CLV regressor. Returns mean RMSE (to minimize)."""
    space = config.get("optuna", {}).get("lightgbm_clv_space", {})
    cv_folds = config.get("optuna", {}).get("cv_folds", 5)

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", *space.get("n_estimators", [100, 1000])
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves", *space.get("num_leaves", [15, 63])
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", *space.get("learning_rate", [0.01, 0.3]), log=True
        ),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", *space.get("feature_fraction", [0.5, 1.0])
        ),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", *space.get("bagging_fraction", [0.5, 1.0])
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", *space.get("min_child_samples", [5, 50])
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", *space.get("reg_alpha", [1e-8, 10.0]), log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", *space.get("reg_lambda", [1e-8, 10.0]), log=True
        ),
    }

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(
            **params,
            bagging_freq=5,
            random_state=42,
            verbosity=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        preds = model.predict(X_va)
        rmse = np.sqrt(mean_squared_error(y_va, preds))
        rmse_scores.append(rmse)

    return float(np.mean(rmse_scores))


def _lightgbm_churn_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
) -> float:
    """Objective for LightGBM churn classifier. Returns mean AUC (to maximize)."""
    space = config.get("optuna", {}).get("lightgbm_churn_space", {})
    cv_folds = config.get("optuna", {}).get("cv_folds", 5)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    spw = float(n_neg / max(n_pos, 1))

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", *space.get("n_estimators", [100, 800])
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves", *space.get("num_leaves", [15, 63])
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", *space.get("learning_rate", [0.01, 0.3]), log=True
        ),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", *space.get("feature_fraction", [0.5, 1.0])
        ),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", *space.get("bagging_fraction", [0.5, 1.0])
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", *space.get("min_child_samples", [5, 50])
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", *space.get("reg_alpha", [1e-8, 10.0]), log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", *space.get("reg_lambda", [1e-8, 10.0]), log=True
        ),
    }

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(
            **params,
            scale_pos_weight=spw,
            bagging_freq=5,
            random_state=42,
            verbosity=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        probs = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, probs)
        auc_scores.append(auc)

    return float(np.mean(auc_scores))


# =====================================================================
# Run a single tuning experiment with MLflow logging
# =====================================================================

def _run_tuning_experiment(
    name: str,
    objective_fn,
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    direction: str,
) -> optuna.Study:
    """Run Optuna study with MLflow logging for each trial.

    Args:
        name: Experiment name (e.g. "xgboost-clv").
        objective_fn: Optuna objective function.
        X: Feature array.
        y: Target array.
        config: Pipeline configuration.
        direction: "minimize" for RMSE, "maximize" for AUC.

    Returns:
        Completed Optuna study.
    """
    optuna_config = config.get("optuna", {})
    n_trials = optuna_config.get("n_trials", 50)
    timeout = optuna_config.get("timeout", 1800)

    # Suppress Optuna's default logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction=direction,
        study_name=name,
    )

    print(f"\n  Running {n_trials} Optuna trials ({direction} {name})...")

    # Track progress
    best_so_far = float("inf") if direction == "minimize" else float("-inf")

    def objective_with_logging(trial: optuna.Trial) -> float:
        nonlocal best_so_far

        # Run the actual objective
        value = objective_fn(trial, X, y, config)

        # Log to MLflow as nested run
        with mlflow.start_run(
            run_name=f"trial-{trial.number}",
            nested=True,
        ):
            log_params(trial.params)
            metric_name = "cv_rmse" if direction == "minimize" else "cv_auc"
            log_metrics({metric_name: value, "trial_number": trial.number})

        # Print progress for significant improvements
        is_better = (
            (direction == "minimize" and value < best_so_far)
            or (direction == "maximize" and value > best_so_far)
        )
        if is_better:
            best_so_far = value
            print(f"    Trial {trial.number:3d}: {metric_name}={value:.4f} ★ new best")
        elif trial.number % 10 == 0:
            print(f"    Trial {trial.number:3d}: {metric_name}={value:.4f}")

        return value

    study.optimize(objective_with_logging, n_trials=n_trials, timeout=timeout)

    print(f"  Best trial: #{study.best_trial.number}, "
          f"value={study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study


# =====================================================================
# Main orchestrator
# =====================================================================

def run_all_tuning(config: dict[str, Any]) -> dict:
    """Run Optuna tuning for all 4 model variants and pick the best.

    Pipeline:
      1. Load data, build feature matrix, scale
      2. Tune XGBoost CLV (minimize RMSE)
      3. Tune LightGBM CLV (minimize RMSE)
      4. Compare → pick best CLV model
      5. Tune XGBoost churn (maximize AUC)
      6. Tune LightGBM churn (maximize AUC)
      7. Compare → pick best churn model
      8. Retrain winners on full data with best params
      9. Log final models to MLflow

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary with best models and their results.
    """
    # ── Setup ───────────────────────────────────────────────────────
    load_env_file()
    setup_mlflow(config)

    # ── Load data ───────────────────────────────────────────────────
    rfm_path = PROJECT_ROOT / "data" / "processed" / "rfm_summary.csv"
    behavioral_path = PROJECT_ROOT / "data" / "processed" / "features_behavioral.csv"

    rfm = load_csv(rfm_path, index_col="customer_id")
    behavioral = load_csv(behavioral_path, index_col="customer_id")

    feature_matrix = build_feature_matrix(rfm, behavioral)
    clv_target = rfm["holdout_revenue"]
    churn_target = (rfm["holdout_frequency"] == 0).astype(int)

    print(f"\nLoaded {len(feature_matrix):,} customers, "
          f"{feature_matrix.shape[1]} features")

    # Scale features once — shared across all experiments
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    y_clv = clv_target.loc[feature_matrix.index].values
    y_churn = churn_target.loc[feature_matrix.index].values

    results = {}

    # =================================================================
    # CLV tuning: XGBoost vs LightGBM
    # =================================================================
    print("\n" + "=" * 60)
    print("CLV MODEL TUNING")
    print("=" * 60)

    # ── XGBoost CLV ─────────────────────────────────────────────────
    with mlflow.start_run(run_name="optuna-xgboost-clv"):
        xgb_clv_study = _run_tuning_experiment(
            name="xgboost-clv",
            objective_fn=_xgboost_clv_objective,
            X=X_scaled, y=y_clv,
            config=config,
            direction="minimize",
        )
        log_params({"model_type": "XGBoost", "task": "clv_regression"})
        log_params(xgb_clv_study.best_params)
        log_metrics({"best_cv_rmse": xgb_clv_study.best_value})

    # ── LightGBM CLV ────────────────────────────────────────────────
    with mlflow.start_run(run_name="optuna-lightgbm-clv"):
        lgbm_clv_study = _run_tuning_experiment(
            name="lightgbm-clv",
            objective_fn=_lightgbm_clv_objective,
            X=X_scaled, y=y_clv,
            config=config,
            direction="minimize",
        )
        log_params({"model_type": "LightGBM", "task": "clv_regression"})
        log_params(lgbm_clv_study.best_params)
        log_metrics({"best_cv_rmse": lgbm_clv_study.best_value})

    # ── Compare CLV models ──────────────────────────────────────────
    print("\n" + "-" * 50)
    print("CLV MODEL COMPARISON (5-fold CV RMSE)")
    print("-" * 50)
    print(f"  XGBoost best CV RMSE:  {xgb_clv_study.best_value:.2f}")
    print(f"  LightGBM best CV RMSE: {lgbm_clv_study.best_value:.2f}")

    if xgb_clv_study.best_value <= lgbm_clv_study.best_value:
        best_clv_type = "xgboost"
        best_clv_params = xgb_clv_study.best_params
        print("  → Winner: XGBoost")
    else:
        best_clv_type = "lightgbm"
        best_clv_params = lgbm_clv_study.best_params
        print("  → Winner: LightGBM")

    results["best_clv_type"] = best_clv_type
    results["best_clv_params"] = best_clv_params

    # =================================================================
    # Churn tuning: XGBoost vs LightGBM
    # =================================================================
    print("\n" + "=" * 60)
    print("CHURN MODEL TUNING")
    print("=" * 60)

    # ── XGBoost churn ───────────────────────────────────────────────
    with mlflow.start_run(run_name="optuna-xgboost-churn"):
        xgb_churn_study = _run_tuning_experiment(
            name="xgboost-churn",
            objective_fn=_xgboost_churn_objective,
            X=X_scaled, y=y_churn,
            config=config,
            direction="maximize",
        )
        log_params({"model_type": "XGBoost", "task": "churn_classification"})
        log_params(xgb_churn_study.best_params)
        log_metrics({"best_cv_auc": xgb_churn_study.best_value})

    # ── LightGBM churn ──────────────────────────────────────────────
    with mlflow.start_run(run_name="optuna-lightgbm-churn"):
        lgbm_churn_study = _run_tuning_experiment(
            name="lightgbm-churn",
            objective_fn=_lightgbm_churn_objective,
            X=X_scaled, y=y_churn,
            config=config,
            direction="maximize",
        )
        log_params({"model_type": "LightGBM", "task": "churn_classification"})
        log_params(lgbm_churn_study.best_params)
        log_metrics({"best_cv_auc": lgbm_churn_study.best_value})

    # ── Compare churn models ────────────────────────────────────────
    print("\n" + "-" * 50)
    print("CHURN MODEL COMPARISON (5-fold CV AUC)")
    print("-" * 50)
    print(f"  XGBoost best CV AUC:  {xgb_churn_study.best_value:.4f}")
    print(f"  LightGBM best CV AUC: {lgbm_churn_study.best_value:.4f}")

    if xgb_churn_study.best_value >= lgbm_churn_study.best_value:
        best_churn_type = "xgboost"
        best_churn_params = xgb_churn_study.best_params
        print("  → Winner: XGBoost")
    else:
        best_churn_type = "lightgbm"
        best_churn_params = lgbm_churn_study.best_params
        print("  → Winner: LightGBM")

    results["best_churn_type"] = best_churn_type
    results["best_churn_params"] = best_churn_params

    # =================================================================
    # Retrain winners on full data and log final models
    # =================================================================
    print("\n" + "=" * 60)
    print("RETRAINING BEST MODELS ON FULL DATA")
    print("=" * 60)

    # ── Rebuild config with best CLV params ─────────────────────────
    tuned_config = _deep_copy_config(config)

    if best_clv_type == "xgboost":
        tuned_config["models"]["xgboost_clv"].update(best_clv_params)
        final_clv = train_xgboost_clv(feature_matrix, clv_target, tuned_config)
        final_clv_scaler = final_clv["scaler"]
    else:
        tuned_config["models"]["lightgbm_clv"].update(best_clv_params)
        final_clv = train_lightgbm_clv(
            feature_matrix, clv_target, tuned_config,
        )
        final_clv_scaler = final_clv["scaler"]

    results["final_clv"] = final_clv

    # ── Rebuild config with best churn params ───────────────────────
    if best_churn_type == "xgboost":
        tuned_config["models"]["churn_classifier"].update(best_churn_params)
        final_churn = train_churn_classifier(
            features=feature_matrix,
            holdout_frequency=rfm["holdout_frequency"],
            p_alive=rfm["p_alive"],
            scaler=final_clv_scaler,
            config=tuned_config,
        )
    else:
        tuned_config["models"]["lightgbm_churn"].update(best_churn_params)
        final_churn = train_lightgbm_churn(
            features=feature_matrix,
            holdout_frequency=rfm["holdout_frequency"],
            p_alive=rfm["p_alive"],
            scaler=final_clv_scaler,
            config=tuned_config,
        )

    results["final_churn"] = final_churn

    # ── Log final models to MLflow ──────────────────────────────────
    with mlflow.start_run(run_name="best-models-final"):
        log_params({
            "best_clv_model": best_clv_type,
            "best_churn_model": best_churn_type,
        })
        log_params({f"clv_{k}": v for k, v in best_clv_params.items()})
        log_params({f"churn_{k}": v for k, v in best_churn_params.items()})

        clv_metrics = final_clv.get("metrics", {})
        churn_metrics = final_churn.get("metrics", {})
        log_metrics({
            f"clv_{k}": v for k, v in clv_metrics.items()
            if isinstance(v, (int, float))
        })
        log_metrics({
            f"churn_{k}": v for k, v in churn_metrics.items()
            if isinstance(v, (int, float))
        })

        # Log all model artifacts from models/ directory
        models_dir = PROJECT_ROOT / "models"
        for artifact in models_dir.glob("*"):
            if artifact.is_file() and artifact.suffix in (
                ".json", ".pkl", ".txt",
            ):
                log_model_artifact(artifact)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"\n  Best CLV model:   {best_clv_type}")
    print(f"    CV RMSE:        {min(xgb_clv_study.best_value, lgbm_clv_study.best_value):.2f}")
    clv_m = final_clv.get("metrics", {})
    print(f"    Holdout MAE:    {clv_m.get('mae', 'N/A')}")
    print(f"    Holdout RMSE:   {clv_m.get('rmse', 'N/A')}")
    print(f"    Holdout R²:     {clv_m.get('r2', 'N/A')}")

    print(f"\n  Best churn model: {best_churn_type}")
    print(f"    CV AUC:         {max(xgb_churn_study.best_value, lgbm_churn_study.best_value):.4f}")
    churn_m = final_churn.get("metrics", {})
    print(f"    Holdout AUC:    {churn_m.get('ml_auc', 'N/A')}")
    print(f"    Holdout F1:     {churn_m.get('ml_f1', 'N/A')}")

    return results


def _deep_copy_config(config: dict) -> dict:
    """Create a deep copy of a nested config dict."""
    import copy
    return copy.deepcopy(config)
