"""Probabilistic CLV models: BG/NBD and Gamma-Gamma.

BG/NBD estimates purchase frequency and P(alive) from RFM data.
Gamma-Gamma estimates expected monetary value per transaction.
Combined, they produce a probabilistic CLV estimate.
"""

from typing import Any
from pathlib import Path

import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter, GammaGammaFitter
import joblib

from src.utils.io import PROJECT_ROOT, load_csv, save_csv
def fit_bgnbd(rfm: pd.DataFrame, config: dict[str, Any]) -> BetaGeoFitter:
    """Fit the BG/NBD model on the calibration RFM summary.

    Args:
        rfm: RFM summary table with frequency, recency, T columns.
        config: Pipeline configuration dictionary.

    Returns:
        Fitted BetaGeoFitter instance.
    """
    penalizer = config["models"]["probabilistic"]["penalizer_coef"]

    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(
        frequency=rfm["frequency"],
        recency=rfm["recency"],
        T=rfm["T"],
    )

    print("BG/NBD model fitted")
    print(f"  Penalizer: {penalizer}")
    print(f"  Parameters:")
    for param, value in sorted(bgf.params_.items()):
        print(f"    {param}: {value:.4f}")

    return bgf
def predict_and_validate_bgnbd(
    bgf: BetaGeoFitter,
    rfm: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Generate BG/NBD predictions and validate against holdout.

    Predicts expected purchases and P(alive) for each customer.
    Compares predicted purchases against actual holdout frequency.

    Args:
        bgf: Fitted BetaGeoFitter.
        rfm: RFM summary with holdout_frequency for validation.
        config: Pipeline configuration.

    Returns:
        rfm DataFrame with added prediction columns.
    """
    # How many days is the holdout period?
    cal_end = pd.Timestamp(config["data"]["calibration_end"])
    holdout_end = pd.Timestamp(config["data"]["holdout_end"])
    holdout_days = (holdout_end - cal_end).days

    print(f"\nPredicting over {holdout_days}-day holdout window...")

    # -----------------------------------------------------------------
    # Predicted purchases in the holdout period
    # -----------------------------------------------------------------
    rfm["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t=holdout_days,
        frequency=rfm["frequency"],
        recency=rfm["recency"],
        T=rfm["T"],
    )

    # -----------------------------------------------------------------
    # P(alive) — probability the customer hasn't churned
    # -----------------------------------------------------------------
    rfm["p_alive"] = bgf.conditional_probability_alive(
        frequency=rfm["frequency"],
        recency=rfm["recency"],
        T=rfm["T"],
    )

    # -----------------------------------------------------------------
    # Validation: predicted vs actual holdout purchases
    # -----------------------------------------------------------------
    mae = np.abs(rfm["predicted_purchases"] - rfm["holdout_frequency"]).mean()
    rmse = np.sqrt(((rfm["predicted_purchases"] - rfm["holdout_frequency"]) ** 2).mean())

    print(f"\nBG/NBD Validation (holdout):")
    print(f"  MAE:  {mae:.3f} transactions")
    print(f"  RMSE: {rmse:.3f} transactions")
    print(f"  Mean predicted: {rfm['predicted_purchases'].mean():.2f}")
    print(f"  Mean actual:    {rfm['holdout_frequency'].mean():.2f}")

    # -----------------------------------------------------------------
    # Calibration check: compare by decile
    # Group customers into 10 bins by predicted purchases, then
    # compare the mean prediction vs mean actual within each bin.
    # -----------------------------------------------------------------
    rfm["pred_decile"] = pd.qcut(
        rfm["predicted_purchases"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    calibration = (
        rfm
        .groupby("pred_decile")
        .agg(
            n_customers=("predicted_purchases", "size"),
            mean_predicted=("predicted_purchases", "mean"),
            mean_actual=("holdout_frequency", "mean"),
        )
        .round(2)
    )

    print("\nCalibration by decile (predicted vs actual):")
    print(calibration.to_string())

    # Clean up the temporary column
    rfm = rfm.drop(columns=["pred_decile"])

    # -----------------------------------------------------------------
    # P(alive) summary
    # -----------------------------------------------------------------
    print(f"\nP(alive) distribution:")
    print(f"  Mean:   {rfm['p_alive'].mean():.3f}")
    print(f"  Median: {rfm['p_alive'].median():.3f}")
    print(f"  Likely alive (p > 0.5): {(rfm['p_alive'] > 0.5).sum():,}")
    print(f"  Likely dead (p <= 0.5): {(rfm['p_alive'] <= 0.5).sum():,}")

    # Quick check: do high-p_alive customers actually return more?
    alive_returned = rfm.loc[rfm["p_alive"] > 0.5, "holdout_frequency"].mean()
    dead_returned = rfm.loc[rfm["p_alive"] <= 0.5, "holdout_frequency"].mean()
    print(f"\n  Sanity check:")
    print(f"    Mean holdout purchases (p_alive > 0.5): {alive_returned:.2f}")
    print(f"    Mean holdout purchases (p_alive <= 0.5): {dead_returned:.2f}")

    return rfm
def save_bgnbd_model(bgf: BetaGeoFitter) -> Path:
    """Save the fitted BG/NBD model to disk.

    Args:
        bgf: Fitted BetaGeoFitter.

    Returns:
        Path where the model was saved.
    """
    model_path = PROJECT_ROOT / "models" / "bgnbd_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bgf, model_path)
    print(f"\nBG/NBD model saved to {model_path}")
    return model_path

def fit_gamma_gamma(rfm: pd.DataFrame, config: dict[str, Any]) -> GammaGammaFitter:
    """Fit the Gamma-Gamma model for monetary value estimation.

    Only uses repeat buyers (frequency >= 1) since the model needs
    at least one repeat transaction to estimate average spend.

    Args:
        rfm: RFM summary table with frequency and monetary_value columns.
        config: Pipeline configuration dictionary.

    Returns:
        Fitted GammaGammaFitter instance.
    """
    penalizer = config["models"]["probabilistic"]["penalizer_coef"]

    # Filter to repeat buyers only
    repeat_buyers = rfm[rfm["frequency"] >= 1].copy()
    print(f"\nGamma-Gamma: using {len(repeat_buyers):,} repeat buyers "
          f"(excluded {(rfm['frequency'] == 0).sum():,} one-time buyers)")

    # -----------------------------------------------------------------
    # Assumption check: frequency vs monetary_value independence
    # -----------------------------------------------------------------
    corr = repeat_buyers[["frequency", "monetary_value"]].corr().iloc[0, 1]
    print(f"  Frequency-monetary correlation: {corr:.4f}")
    if abs(corr) > 0.3:
        print(f"  WARNING: correlation is high ({corr:.4f}). "
              "Gamma-Gamma assumption of independence may be violated.")
    else:
        print(f"  OK: correlation is within acceptable range")

    # -----------------------------------------------------------------
    # Fit the model
    # -----------------------------------------------------------------
    ggf = GammaGammaFitter(penalizer_coef=penalizer)
    ggf.fit(
        frequency=repeat_buyers["frequency"],
        monetary_value=repeat_buyers["monetary_value"],
    )

    print(f"  Parameters:")
    for param, value in sorted(ggf.params_.items()):
        print(f"    {param}: {value:.4f}")

    return ggf


def predict_clv(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    rfm: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Combine BG/NBD and Gamma-Gamma into a CLV estimate.

    CLV = expected_purchases * expected_monetary_value, discounted
    to present value using an annual discount rate.

    Args:
        bgf: Fitted BG/NBD model.
        ggf: Fitted Gamma-Gamma model.
        rfm: RFM summary table (with predictions from BG/NBD step).
        config: Pipeline configuration.

    Returns:
        rfm DataFrame with added CLV columns.
    """
    horizon_months = config["models"]["probabilistic"]["prediction_horizon_months"]

    # -----------------------------------------------------------------
    # Expected average monetary value per transaction
    # For one-time buyers, use the population mean as fallback
    # -----------------------------------------------------------------
    repeat_mask = rfm["frequency"] >= 1
    rfm["expected_avg_value"] = 0.0

    rfm.loc[repeat_mask, "expected_avg_value"] = (
        ggf.conditional_expected_average_profit(
            frequency=rfm.loc[repeat_mask, "frequency"],
            monetary_value=rfm.loc[repeat_mask, "monetary_value"],
        )
    )

# For one-time buyers, use the mean of repeat buyers' expected values
    # (the closed-form population mean requires q > 1, which isn't guaranteed)
    fallback_value = rfm.loc[repeat_mask, "expected_avg_value"].mean()
    rfm.loc[~repeat_mask, "expected_avg_value"] = fallback_value

    print(f"\nExpected avg transaction value:")
    print(f"  Repeat buyers mean:  {rfm.loc[repeat_mask, 'expected_avg_value'].mean():.2f}")
    print(f"  One-time buyer fallback: {fallback_value:.2f}")

    # -----------------------------------------------------------------
    # Combined CLV using lifetimes' built-in method
    # Only works for repeat buyers — uses both models together
    # -----------------------------------------------------------------
    # Annual discount rate: 10% is standard for marketing CLV
    annual_discount = 0.10
    monthly_discount = (1 + annual_discount) ** (1 / 12) - 1

    rfm["predicted_clv"] = ggf.customer_lifetime_value(
        bgf,
        frequency=rfm["frequency"],
        recency=rfm["recency"],
        T=rfm["T"],
        monetary_value=rfm["monetary_value"],
        time=horizon_months,
        freq="D",
        discount_rate=monthly_discount,
    )

    print(f"\nPredicted CLV ({horizon_months}-month horizon, {annual_discount:.0%} annual discount):")
    print(f"  Mean:   {rfm['predicted_clv'].mean():.2f}")
    print(f"  Median: {rfm['predicted_clv'].median():.2f}")
    print(f"  Total:  {rfm['predicted_clv'].sum():,.2f}")

    # -----------------------------------------------------------------
    # Validate against actual holdout revenue
    # -----------------------------------------------------------------
    mae = np.abs(rfm["predicted_clv"] - rfm["holdout_revenue"]).mean()
    rmse = np.sqrt(((rfm["predicted_clv"] - rfm["holdout_revenue"]) ** 2).mean())

    # Only compute MAPE where actual > 0 to avoid division by zero
    nonzero_mask = rfm["holdout_revenue"] > 0
    mape = np.abs(
        (rfm.loc[nonzero_mask, "predicted_clv"] - rfm.loc[nonzero_mask, "holdout_revenue"])
        / rfm.loc[nonzero_mask, "holdout_revenue"]
    ).mean()

    print(f"\nCLV Validation against holdout revenue:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2%} (on {nonzero_mask.sum():,} customers with holdout revenue > 0)")
    print(f"  Mean predicted CLV: {rfm['predicted_clv'].mean():.2f}")
    print(f"  Mean actual holdout revenue: {rfm['holdout_revenue'].mean():.2f}")

    return rfm


def save_gg_model(ggf: GammaGammaFitter) -> Path:
    """Save the fitted Gamma-Gamma model to disk.

    Args:
        ggf: Fitted GammaGammaFitter.

    Returns:
        Path where the model was saved.
    """
    model_path = PROJECT_ROOT / "models" / "gg_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ggf, model_path)
    print(f"\nGamma-Gamma model saved to {model_path}")
    return model_path
