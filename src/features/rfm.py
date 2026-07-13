"""RFM summary table construction for the lifetimes library.

Builds customer-level frequency, recency, T, and monetary_value from
calibration-period transactions, plus holdout ground truth for validation.
"""

from typing import Any

import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data

from src.utils.io import PROJECT_ROOT, load_csv, save_csv


def build_rfm_summary(config: dict[str, Any]) -> pd.DataFrame:
    """Build the RFM summary table from calibration transactions.

    Uses lifetimes.utils.summary_data_from_transaction_data() to compute
    frequency, recency, T, and monetary_value per customer. Also computes
    holdout-period ground truth (frequency and revenue) for model validation.

    Args:
        config: Pipeline configuration dictionary (from params.yaml).

    Returns:
        DataFrame indexed by customer_id with columns:
            frequency, recency, T, monetary_value (calibration period)
            holdout_frequency, holdout_revenue (holdout period)
    """
    # Load calibration and holdout transaction data
    cal_path = PROJECT_ROOT / "data" / "interim" / "transactions_calibration.csv"
    holdout_path = PROJECT_ROOT / "data" / "interim" / "transactions_holdout.csv"

    df_cal = load_csv(cal_path, parse_dates=["invoice_date"])
    df_holdout = load_csv(holdout_path, parse_dates=["invoice_date"])

    print(f"Calibration transactions: {len(df_cal):,} rows, "
          f"{df_cal['customer_id'].nunique():,} customers")
    print(f"Holdout transactions:     {len(df_holdout):,} rows, "
          f"{df_holdout['customer_id'].nunique():,} customers")

    # -----------------------------------------------------------------
    # Calibration-period RFM summary
    # -----------------------------------------------------------------
    calibration_end = pd.Timestamp(config["data"]["calibration_end"])

    rfm = summary_data_from_transaction_data(
        transactions=df_cal,
        customer_id_col="customer_id",
        datetime_col="invoice_date",
        monetary_value_col="total_amount",
        observation_period_end=calibration_end,
        freq="D",  # time unit = days
    )

    print(f"\nRFM summary: {len(rfm):,} customers")
    print(f"  Repeat buyers (frequency >= 1): {(rfm['frequency'] >= 1).sum():,}")
    print(f"  One-time buyers (frequency == 0): {(rfm['frequency'] == 0).sum():,}")
    print(f"  Mean frequency: {rfm['frequency'].mean():.2f}")
    print(f"  Mean recency: {rfm['recency'].mean():.1f} days")
    print(f"  Mean T: {rfm['T'].mean():.1f} days")
    print(f"  Mean monetary_value (repeat buyers): "
          f"{rfm.loc[rfm['frequency'] >= 1, 'monetary_value'].mean():.2f}")

    # -----------------------------------------------------------------
    # Holdout-period ground truth
    # -----------------------------------------------------------------
    # Compute what each customer actually did in the holdout period.
    # Customers in calibration but NOT in holdout churned (or went quiet).
    holdout_summary = (
        df_holdout
        .groupby("customer_id")
        .agg(
            holdout_frequency=("invoice", "nunique"),
            holdout_revenue=("total_amount", "sum"),
        )
    )

    # Merge holdout onto RFM table -- fill missing customers with 0
    rfm = rfm.join(holdout_summary, how="left")
    rfm["holdout_frequency"] = rfm["holdout_frequency"].fillna(0).astype(int)
    rfm["holdout_revenue"] = rfm["holdout_revenue"].fillna(0.0)

    # How many calibration customers actually returned in holdout?
    returned = (rfm["holdout_frequency"] > 0).sum()
    churned = (rfm["holdout_frequency"] == 0).sum()
    print(f"\nHoldout behavior:")
    print(f"  Returned in holdout: {returned:,} "
          f"({returned / len(rfm) * 100:.1f}%)")
    print(f"  Did not return:      {churned:,} "
          f"({churned / len(rfm) * 100:.1f}%)")
    print(f"  Mean holdout revenue: {rfm['holdout_revenue'].mean():.2f}")
    print(f"  Total holdout revenue: {rfm['holdout_revenue'].sum():,.2f}")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    output_path = PROJECT_ROOT / "data" / "processed" / "rfm_summary.csv"
    save_csv(rfm, output_path, index=True)
    print(f"\nRFM summary saved to {output_path}")

    return rfm