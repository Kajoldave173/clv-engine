"""Model monitoring: feature drift detection and prediction health checks.

Provides PSI (Population Stability Index) computation for feature drift,
prediction distribution validation, churn rate sanity, and scoring
coverage checks. Used by the batch scorer to detect when models may
need retraining.

PSI thresholds (industry standard):
  < 0.1   — No significant shift
  0.1–0.2 — Moderate shift, investigate
  > 0.2   — Significant shift, likely model degradation
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from src.utils.io import PROJECT_ROOT, load_csv


# ── PSI thresholds ──────────────────────────────────────────────────
PSI_OK = 0.1
PSI_WARNING = 0.2


# ====================================================================
# Baseline computation and storage
# ====================================================================

def compute_baselines(
    feature_matrix: pd.DataFrame,
    predictions: dict[str, pd.Series],
) -> dict:
    """Compute baseline distributions from training data for drift detection.

    For each feature, stores decile bin boundaries and the proportion of
    training observations in each bin. For predictions, stores mean and
    standard deviation.

    Should be called once after training. The batch scorer calls
    ensure_baselines() which triggers this automatically if needed.

    Args:
        feature_matrix: Training feature matrix (all ML features).
        predictions: Dict mapping prediction name to its Series, e.g.
            {"predicted_clv_ml": series, "churn_probability_ml": series}.

    Returns:
        Baselines dictionary (also saved to disk).
    """
    baselines: dict[str, Any] = {"features": {}, "predictions": {}}

    # ── Feature baselines: decile bins + proportions ────────────────
    for col in feature_matrix.columns:
        values = feature_matrix[col].dropna().values

        if len(values) == 0:
            continue

        # Compute decile boundaries (9 cut points → 10 bins)
        percentiles = np.arange(10, 100, 10)  # [10, 20, ..., 90]
        boundaries = np.unique(np.percentile(values, percentiles))

        # Bin edges with -inf/+inf sentinels so all new values are captured
        bin_edges = np.concatenate([[-np.inf], boundaries, [np.inf]])

        # Compute training proportions per bin
        counts, _ = np.histogram(values, bins=bin_edges)
        proportions = counts / counts.sum()

        # Floor at 1e-4 to avoid log(0) in PSI computation
        proportions = np.maximum(proportions, 1e-4)
        proportions = proportions / proportions.sum()  # renormalize

        baselines["features"][col] = {
            "bin_edges": bin_edges,
            "expected_proportions": proportions,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n_samples": len(values),
        }

    # ── Prediction baselines: summary statistics ───────────────────
    for name, series in predictions.items():
        values = series.dropna().values
        if len(values) == 0:
            continue

        baselines["predictions"][name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    # ── Save ───────────────────────────────────────────────────────
    output_path = PROJECT_ROOT / "data" / "processed" / "feature_baselines.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(baselines, output_path)

    print(f"Feature baselines saved to {output_path}")
    print(f"  {len(baselines['features'])} features, "
          f"{len(baselines['predictions'])} prediction targets")

    return baselines


def ensure_baselines() -> dict:
    """Load baselines if they exist, otherwise compute from training artifacts.

    On first run after training, this reconstructs the training feature
    matrix from saved CSVs and computes baselines automatically. On
    subsequent runs, it just loads the cached .pkl file.

    Returns:
        Baselines dictionary.
    """
    path = PROJECT_ROOT / "data" / "processed" / "feature_baselines.pkl"

    if path.exists():
        return joblib.load(path)

    print("Feature baselines not found. Computing from training artifacts...")

    # Reconstruct the training feature matrix (same as training pipeline)
    rfm = load_csv(
        PROJECT_ROOT / "data" / "processed" / "rfm_summary.csv",
        index_col="customer_id",
    )
    behavioral = load_csv(
        PROJECT_ROOT / "data" / "processed" / "features_behavioral.csv",
        index_col="customer_id",
    )
    predictions_df = load_csv(
        PROJECT_ROOT / "data" / "processed" / "customer_predictions.csv",
        index_col="customer_id",
    )

    # Build the same feature matrix used during training
    rfm_features = rfm[[
        "frequency", "recency", "T", "monetary_value",
        "predicted_purchases", "p_alive", "expected_avg_value",
    ]].copy()
    feature_matrix = rfm_features.join(behavioral, how="inner")

    # Collect prediction series
    pred_dict: dict[str, pd.Series] = {}
    for col in ["predicted_clv_ml", "churn_probability_ml"]:
        if col in predictions_df.columns:
            pred_dict[col] = predictions_df[col]

    return compute_baselines(feature_matrix, pred_dict)


# ====================================================================
# PSI computation
# ====================================================================

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI = sum( (actual_i - expected_i) * ln(actual_i / expected_i) )

    Both arrays must be proportions that sum to 1, with no zeros.

    Args:
        expected: Expected proportions per bin (from training).
        actual: Actual proportions per bin (from scoring).

    Returns:
        PSI value.
    """
    # Safety: floor both to avoid division by zero and log(0)
    expected = np.maximum(expected, 1e-4)
    actual = np.maximum(actual, 1e-4)

    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compute_feature_drift(
    new_features: pd.DataFrame,
    baselines: dict,
) -> pd.DataFrame:
    """Compute PSI for each feature against training baselines.

    Args:
        new_features: Scoring feature matrix (same columns as training).
        baselines: Training baselines from compute_baselines().

    Returns:
        DataFrame with columns: feature, psi, status, train_mean,
        score_mean — sorted by PSI descending (worst drift first).
    """
    results = []

    for col in new_features.columns:
        # Feature not in baselines (new feature or renamed)
        if col not in baselines["features"]:
            results.append({
                "feature": col,
                "psi": None,
                "status": "MISSING_BASELINE",
                "train_mean": None,
                "score_mean": round(float(new_features[col].mean()), 4),
            })
            continue

        baseline = baselines["features"][col]
        bin_edges = np.array(baseline["bin_edges"])
        expected = np.array(baseline["expected_proportions"])

        values = new_features[col].dropna().values

        # No data for this feature
        if len(values) == 0:
            results.append({
                "feature": col,
                "psi": None,
                "status": "NO_DATA",
                "train_mean": round(baseline["mean"], 4),
                "score_mean": None,
            })
            continue

        # Bin new data using training bin edges
        counts, _ = np.histogram(values, bins=bin_edges)
        actual = counts / counts.sum()

        # Floor and renormalize
        actual = np.maximum(actual, 1e-4)
        actual = actual / actual.sum()

        psi = compute_psi(expected, actual)

        if psi < PSI_OK:
            status = "OK"
        elif psi < PSI_WARNING:
            status = "WARNING"
        else:
            status = "ALERT"

        results.append({
            "feature": col,
            "psi": round(psi, 4),
            "status": status,
            "train_mean": round(baseline["mean"], 4),
            "score_mean": round(float(np.mean(values)), 4),
        })

    df = pd.DataFrame(results)

    # Sort: alerts first, then warnings, then OK — within each group by PSI
    status_order = {"ALERT": 0, "WARNING": 1, "MISSING_BASELINE": 2, "NO_DATA": 3, "OK": 4}
    df["_sort"] = df["status"].map(status_order)
    df = df.sort_values(["_sort", "psi"], ascending=[True, False]).drop(columns="_sort")

    return df.reset_index(drop=True)


# ====================================================================
# Prediction and churn checks
# ====================================================================

def check_prediction_distribution(
    predictions: dict[str, pd.Series],
    baselines: dict,
    n_sigma: float = 2.0,
) -> list[dict]:
    """Check if prediction distributions are within expected range.

    Flags if the mean of new predictions is more than n_sigma standard
    deviations from the training mean.

    Args:
        predictions: Dict of prediction series.
        baselines: Training baselines.
        n_sigma: Number of standard deviations for threshold.

    Returns:
        List of check result dicts.
    """
    results = []

    for name, series in predictions.items():
        values = series.dropna().values

        if name not in baselines.get("predictions", {}):
            results.append({
                "metric": name,
                "status": "MISSING_BASELINE",
                "message": "No training baseline available",
            })
            continue

        baseline = baselines["predictions"][name]
        train_mean = baseline["mean"]
        train_std = baseline["std"]
        score_mean = float(np.mean(values))
        score_std = float(np.std(values))

        # Check if mean has shifted beyond threshold
        if train_std > 0:
            z_score = abs(score_mean - train_mean) / train_std
            mean_ok = z_score <= n_sigma
        else:
            mean_ok = abs(score_mean - train_mean) < 1e-6
            z_score = 0.0

        results.append({
            "metric": name,
            "train_mean": round(train_mean, 2),
            "score_mean": round(score_mean, 2),
            "train_std": round(train_std, 2),
            "score_std": round(score_std, 2),
            "z_score": round(z_score, 2),
            "status": "OK" if mean_ok else "ALERT",
            "message": (
                "Within expected range"
                if mean_ok
                else f"Mean shifted by {z_score:.1f} sigma from training"
            ),
        })

    return results


def check_churn_rate(
    churn_probs: pd.Series,
    threshold: float = 0.5,
    min_rate: float = 0.15,
    max_rate: float = 0.70,
) -> dict:
    """Check if the overall predicted churn rate is reasonable.

    Args:
        churn_probs: Predicted churn probabilities per customer.
        threshold: Probability cutoff for classifying as churned.
        min_rate: Minimum acceptable churn rate.
        max_rate: Maximum acceptable churn rate.

    Returns:
        Check result dict.
    """
    churn_rate = float((churn_probs >= threshold).mean())
    status = "OK" if min_rate <= churn_rate <= max_rate else "WARNING"

    return {
        "metric": "churn_rate",
        "value": round(churn_rate, 4),
        "threshold": threshold,
        "expected_range": f"[{min_rate:.0%}, {max_rate:.0%}]",
        "status": status,
        "message": (
            f"Churn rate {churn_rate:.1%} is within expected range"
            if status == "OK"
            else f"Churn rate {churn_rate:.1%} is outside expected range "
                 f"[{min_rate:.0%}, {max_rate:.0%}]"
        ),
    }


def check_coverage(
    total_input: int,
    scored: int,
) -> dict:
    """Check what fraction of input customers were successfully scored.

    Args:
        total_input: Number of unique customers in input data.
        scored: Number of customers that received predictions.

    Returns:
        Check result dict.
    """
    coverage = scored / total_input if total_input > 0 else 0.0

    if coverage >= 0.95:
        status = "OK"
    elif coverage >= 0.80:
        status = "WARNING"
    else:
        status = "ALERT"

    return {
        "metric": "coverage",
        "total_input": total_input,
        "scored": scored,
        "failed": total_input - scored,
        "coverage": round(coverage, 4),
        "status": status,
        "message": f"Scored {scored:,}/{total_input:,} customers ({coverage:.1%})",
    }


# ====================================================================
# Orchestrator: run all checks
# ====================================================================

def run_monitoring(
    new_features: pd.DataFrame,
    predictions: dict[str, pd.Series],
    churn_probs: pd.Series,
    total_input_customers: int,
    scored_customers: int,
) -> dict:
    """Run all monitoring checks and generate a summary.

    Loads training baselines (computing them on first run if needed),
    then runs feature drift, prediction distribution, churn rate, and
    coverage checks.

    Args:
        new_features: Scoring feature matrix (same columns as training).
        predictions: Dict of prediction series, e.g.
            {"predicted_clv_ml": series}.
        churn_probs: Churn probability series for all scored customers.
        total_input_customers: Unique customers in input transactions.
        scored_customers: Customers that received predictions.

    Returns:
        Monitoring results dict with all check outcomes.
    """
    # Load or compute training baselines
    baselines = ensure_baselines()

    # ── Feature drift ──────────────────────────────────────────────
    drift_df = compute_feature_drift(new_features, baselines)

    # ── Prediction distribution ────────────────────────────────────
    pred_checks = check_prediction_distribution(predictions, baselines)

    # ── Churn rate ─────────────────────────────────────────────────
    churn_check = check_churn_rate(churn_probs)

    # ── Coverage ───────────────────────────────────────────────────
    coverage_check = check_coverage(total_input_customers, scored_customers)

    # ── Overall status ─────────────────────────────────────────────
    n_drift_alerts = int((drift_df["status"] == "ALERT").sum())
    n_drift_warnings = int((drift_df["status"] == "WARNING").sum())
    n_pred_alerts = sum(1 for c in pred_checks if c["status"] == "ALERT")

    overall_status = "OK"
    if n_drift_warnings > 0 or churn_check["status"] != "OK":
        overall_status = "WARNING"
    if n_drift_alerts > 2 or n_pred_alerts > 0 or coverage_check["status"] == "ALERT":
        overall_status = "ALERT"

    return {
        "timestamp": datetime.now().isoformat(),
        "overall_status": overall_status,
        "feature_drift": drift_df.to_dict("records"),
        "prediction_checks": pred_checks,
        "churn_rate_check": churn_check,
        "coverage_check": coverage_check,
        "summary": {
            "features_ok": int((drift_df["status"] == "OK").sum()),
            "features_warning": n_drift_warnings,
            "features_alert": n_drift_alerts,
        },
    }


# ====================================================================
# Report generation
# ====================================================================

def save_monitoring_report(
    results: dict,
    output_path: Path | None = None,
) -> Path:
    """Save a human-readable monitoring report to a text file.

    Also prints the report to stdout for visibility during scoring runs.

    Args:
        results: Output from run_monitoring().
        output_path: Where to save. Defaults to
            reports/monitoring_YYYYMMDD.txt.

    Returns:
        Path to saved report.
    """
    if output_path is None:
        date_str = datetime.now().strftime("%Y%m%d")
        output_path = PROJECT_ROOT / "reports" / f"monitoring_{date_str}.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 72)
    lines.append("MODEL MONITORING REPORT")
    lines.append(f"Generated: {results['timestamp']}")
    lines.append(f"Overall Status: {results['overall_status']}")
    lines.append("=" * 72)

    # ── Coverage ───────────────────────────────────────────────────
    cov = results["coverage_check"]
    lines.append("")
    lines.append("--- Coverage ---")
    lines.append(f"  {cov['message']}")
    lines.append(f"  Status: {cov['status']}")

    # ── Feature drift ──────────────────────────────────────────────
    summary = results["summary"]
    lines.append("")
    lines.append("--- Feature Drift (PSI) ---")
    lines.append(
        f"  OK: {summary['features_ok']}  |  "
        f"WARNING: {summary['features_warning']}  |  "
        f"ALERT: {summary['features_alert']}"
    )
    lines.append("")
    lines.append(
        f"  {'Feature':<35} {'PSI':>8} {'Status':>10}"
        f"  {'Train Mean':>12} {'Score Mean':>12}"
    )
    lines.append(
        f"  {'-' * 35} {'-' * 8} {'-' * 10}"
        f"  {'-' * 12} {'-' * 12}"
    )

    for row in results["feature_drift"]:
        psi_str = f"{row['psi']:.4f}" if row["psi"] is not None else "N/A"
        train_str = (
            f"{row['train_mean']:.4f}" if row["train_mean"] is not None else "N/A"
        )
        score_str = (
            f"{row['score_mean']:.4f}" if row["score_mean"] is not None else "N/A"
        )
        lines.append(
            f"  {row['feature']:<35} {psi_str:>8} {row['status']:>10}"
            f"  {train_str:>12} {score_str:>12}"
        )

    # ── Prediction distribution ────────────────────────────────────
    lines.append("")
    lines.append("--- Prediction Distribution ---")
    for check in results["prediction_checks"]:
        lines.append(f"  {check['metric']}: {check['message']}")
        if "train_mean" in check:
            lines.append(
                f"    Train: mean={check['train_mean']}, std={check['train_std']}"
            )
            lines.append(
                f"    Score: mean={check['score_mean']}, std={check['score_std']}"
            )
        lines.append(f"    Status: {check['status']}")

    # ── Churn rate ─────────────────────────────────────────────────
    churn = results["churn_rate_check"]
    lines.append("")
    lines.append("--- Churn Rate ---")
    lines.append(f"  {churn['message']}")
    lines.append(f"  Status: {churn['status']}")

    # ── Footer ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 72)

    report_text = "\n".join(lines)
    output_path.write_text(report_text)

    # Print to stdout for visibility during scoring
    print(report_text)

    return output_path
