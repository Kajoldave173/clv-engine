"""Data cleaning: handle cancellations, missing IDs, duplicates, and anomalies.

Each cleaning step logs what it drops so you can trace every decision.
Output: a clean transaction log saved to data/interim/transactions_clean.csv,
plus calibration/holdout splits.
"""

from typing import Any

import pandas as pd
import numpy as np

from src.utils.io import PROJECT_ROOT, save_csv


# Non-product stock codes — operational entries, not real purchases
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "BANK CHARGES", "PADS", "CRUK", "C2",
    "D", "S", "AMAZONFEE", "B", "gift_0001_10", "gift_0001_20",
    "gift_0001_30", "gift_0001_40", "gift_0001_50",
}


def clean_transactions(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Clean raw transaction data through a series of explicit steps.

    Each step logs what it drops. The order matters:
    1. Drop missing customer IDs (can't model anonymous buyers)
    2. Handle cancellations and returns
    3. Remove non-product stock codes
    4. Remove price anomalies
    5. Remove duplicates
    6. Validate date range
    7. Add computed columns
    8. Save clean data + calibration/holdout splits

    Args:
        df: Raw combined DataFrame from ingest.
        config: Pipeline configuration dictionary.

    Returns:
        Cleaned DataFrame.
    """
    initial_rows = len(df)
    print(f"\n--- Data Cleaning ---")
    print(f"Starting with {initial_rows:,} rows\n")

    cleaning_log = []

    # ---------------------------------------------------------------
    # Step 1: Drop missing CustomerID
    # These are guest checkouts — unusable for customer-level modeling
    # ---------------------------------------------------------------
    null_customer_count = df["customer_id"].isna().sum()
    df = df.dropna(subset=["customer_id"])
    cleaning_log.append(f"Dropped {null_customer_count:,} rows with missing customer_id ({null_customer_count/initial_rows:.1%})")
    print(f"[1] {cleaning_log[-1]}")

    # ---------------------------------------------------------------
    # Step 2: Handle cancellations and returns
    # Invoices starting with 'C' are cancellations.
    # Negative quantities are returns.
    # Strategy: remove all cancellation invoices and negative-quantity rows.
    # In production, you'd match cancellations to originals — here we
    # take the simpler approach and document it.
    # ---------------------------------------------------------------
    cancel_mask = df["invoice"].str.startswith("C", na=False)
    cancel_count = cancel_mask.sum()
    df = df[~cancel_mask]
    cleaning_log.append(f"Dropped {cancel_count:,} cancellation rows (invoice starts with 'C')")
    print(f"[2a] {cleaning_log[-1]}")

    neg_qty_mask = df["quantity"] <= 0
    neg_qty_count = neg_qty_mask.sum()
    df = df[~neg_qty_mask]
    cleaning_log.append(f"Dropped {neg_qty_count:,} rows with quantity <= 0")
    print(f"[2b] {cleaning_log[-1]}")

    # ---------------------------------------------------------------
    # Step 3: Remove non-product stock codes
    # These are operational entries (postage, bank charges, manual adjustments)
    # ---------------------------------------------------------------
    non_product_mask = df["stock_code"].astype(str).str.upper().isin(NON_PRODUCT_CODES)
    non_product_count = non_product_mask.sum()
    df = df[~non_product_mask]
    cleaning_log.append(f"Dropped {non_product_count:,} non-product stock code rows")
    print(f"[3] {cleaning_log[-1]}")

    # ---------------------------------------------------------------
    # Step 4: Remove price anomalies
    # Price <= 0 means free items, adjustments, or data errors
    # ---------------------------------------------------------------
    bad_price_mask = df["price"] <= 0
    bad_price_count = bad_price_mask.sum()
    df = df[~bad_price_mask]
    cleaning_log.append(f"Dropped {bad_price_count:,} rows with price <= 0")
    print(f"[4] {cleaning_log[-1]}")

    # ---------------------------------------------------------------
    # Step 5: Remove exact duplicates
    # Same invoice, product, quantity, timestamp = true duplicate
    # ---------------------------------------------------------------
    dup_cols = ["invoice", "stock_code", "quantity", "invoice_date", "customer_id"]
    dup_count = df.duplicated(subset=dup_cols, keep="first").sum()
    df = df.drop_duplicates(subset=dup_cols, keep="first")
    cleaning_log.append(f"Dropped {dup_count:,} exact duplicate rows")
    print(f"[5] {cleaning_log[-1]}")

    # ---------------------------------------------------------------
    # Step 6: Validate date range
    # Expected: Dec 2009 – Dec 2011
    # ---------------------------------------------------------------
    date_min = pd.Timestamp("2009-12-01")
    date_max = pd.Timestamp("2011-12-10")
    out_of_range_mask = (df["invoice_date"] < date_min) | (df["invoice_date"] > date_max)
    out_of_range_count = out_of_range_mask.sum()
    df = df[~out_of_range_mask]
    cleaning_log.append(f"Dropped {out_of_range_count:,} rows outside date range {date_min.date()} to {date_max.date()}")
    print(f"[6] {cleaning_log[-1]}")

    # ---------------------------------------------------------------
    # Step 7: Add computed columns
    # ---------------------------------------------------------------
    df["total_amount"] = df["quantity"] * df["price"]

    # Ensure customer_id is integer
    df["customer_id"] = df["customer_id"].astype(int)

    # Sort by customer and date for downstream processing
    df = df.sort_values(["customer_id", "invoice_date"]).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    final_rows = len(df)
    total_dropped = initial_rows - final_rows
    print(f"\n--- Cleaning Summary ---")
    print(f"Rows: {initial_rows:,} - > {final_rows:,} (dropped {total_dropped:,}, {total_dropped/initial_rows:.1%})")
    print(f"Unique customers: {df['customer_id'].nunique():,}")
    print(f"Date range: {df['invoice_date'].min().date()} to {df['invoice_date'].max().date()}")
    print(f"Total revenue: £{df['total_amount'].sum():,.2f}")

    # ---------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------
    # Clean transactions
    clean_path = PROJECT_ROOT / "data" / "interim" / "transactions_clean.csv"
    save_csv(df, clean_path, index=False)
    print(f"\nSaved clean data to {clean_path}")

    # Calibration/holdout split
    cal_end = pd.Timestamp(config["data"]["calibration_end"])
    holdout_start = pd.Timestamp(config["data"]["holdout_start"])
    holdout_end = pd.Timestamp(config["data"]["holdout_end"])

    df_cal = df[df["invoice_date"] <= cal_end]
    df_holdout = df[(df["invoice_date"] >= holdout_start) & (df["invoice_date"] <= holdout_end)]

    cal_path = PROJECT_ROOT / "data" / "interim" / "transactions_calibration.csv"
    holdout_path = PROJECT_ROOT / "data" / "interim" / "transactions_holdout.csv"
    save_csv(df_cal, cal_path, index=False)
    save_csv(df_holdout, holdout_path, index=False)

    print(f"Saved calibration data: {len(df_cal):,} rows ({df_cal['customer_id'].nunique():,} customers)")
    print(f"  Period: {df_cal['invoice_date'].min().date()} to {df_cal['invoice_date'].max().date()}")
    print(f"Saved holdout data: {len(df_holdout):,} rows ({df_holdout['customer_id'].nunique():,} customers)")
    print(f"  Period: {df_holdout['invoice_date'].min().date()} to {df_holdout['invoice_date'].max().date()}")

    # Save cleaning log
    log_path = PROJECT_ROOT / "reports" / "cleaning_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Data Cleaning Log\n")
        f.write("=" * 50 + "\n\n")
        for entry in cleaning_log:
            f.write(f"- {entry}\n")
        f.write(f"\nFinal: {initial_rows:,} - > {final_rows:,} rows\n")
    print(f"Saved cleaning log to {log_path}")

    return df
