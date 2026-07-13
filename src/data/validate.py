"""Data validation: quality checks that run after cleaning and before modeling.

These are assertions about data quality. If any critical check fails,
the pipeline halts with a clear error message. Warnings are logged but
don't stop execution.

This module also computes and saves baseline feature distributions
for drift detection during batch scoring.
"""

from typing import Any
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from src.utils.io import PROJECT_ROOT, load_csv


def run_validation(config: dict[str, Any]) -> str:
    """Run all data validation checks on cleaned data.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Validation report as a string.
    """
    clean_path = PROJECT_ROOT / "data" / "interim" / "transactions_clean.csv"
    cal_path = PROJECT_ROOT / "data" / "interim" / "transactions_calibration.csv"
    holdout_path = PROJECT_ROOT / "data" / "interim" / "transactions_holdout.csv"

    df = load_csv(clean_path, parse_dates=["invoice_date"])
    df_cal = load_csv(cal_path, parse_dates=["invoice_date"])
    df_holdout = load_csv(holdout_path, parse_dates=["invoice_date"])

    results = []
    passed = 0
    warned = 0
    failed = 0

    def check(name: str, condition: bool, critical: bool = True) -> None:
        nonlocal passed, warned, failed
        if condition:
            results.append(f"  PASS  | {name}")
            passed += 1
        elif critical:
            results.append(f"  FAIL  | {name}")
            failed += 1
        else:
            results.append(f"  WARN  | {name}")
            warned += 1

    results.append("=" * 60)
    results.append("DATA VALIDATION REPORT")
    results.append(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results.append("=" * 60)

    # -----------------------------------------------------------------
    # 1. Null checks on critical columns
    # -----------------------------------------------------------------
    results.append("\n--- Null Checks ---")

    for col in ["customer_id", "invoice_date", "quantity", "price"]:
        null_count = df[col].isna().sum()
        check(f"No nulls in '{col}' (found {null_count})", null_count == 0)

    # -----------------------------------------------------------------
    # 2. Value range checks
    # -----------------------------------------------------------------
    results.append("\n--- Value Range Checks ---")

    neg_qty = (df["quantity"] <= 0).sum()
    check(f"All quantity > 0 (found {neg_qty} violations)", neg_qty == 0)

    neg_price = (df["price"] <= 0).sum()
    check(f"All price > 0 (found {neg_price} violations)", neg_price == 0)

    neg_total = (df["total_amount"] <= 0).sum()
    check(f"All total_amount > 0 (found {neg_total} violations)", neg_total == 0)

    # -----------------------------------------------------------------
    # 3. Date range checks
    # -----------------------------------------------------------------
    results.append("\n--- Date Range Checks ---")

    date_min = df["invoice_date"].min()
    date_max = df["invoice_date"].max()
    check(
        f"Dates start after 2009-12-01 (actual: {date_min.date()})",
        date_min >= pd.Timestamp("2009-12-01"),
    )
    check(
        f"Dates end before 2011-12-10 (actual: {date_max.date()})",
        date_max <= pd.Timestamp("2011-12-10"),
    )

    # -----------------------------------------------------------------
    # 4. Customer count sanity check
    # -----------------------------------------------------------------
    results.append("\n--- Customer Count Checks ---")

    n_customers = df["customer_id"].nunique()
    check(
        f"Customer count in range 4000-6000 (actual: {n_customers})",
        4000 <= n_customers <= 6000,
    )

    # -----------------------------------------------------------------
    # 5. Revenue sanity check
    # -----------------------------------------------------------------
    results.append("\n--- Revenue Checks ---")

    total_rev = df["total_amount"].sum()
    check(
        f"Total revenue in sane range 5M-25M (actual: {total_rev:,.0f})",
        5_000_000 <= total_rev <= 25_000_000,
        critical=False,
    )

    # -----------------------------------------------------------------
    # 6. Duplicate check
    # -----------------------------------------------------------------
    results.append("\n--- Duplicate Checks ---")

    dup_cols = ["invoice", "stock_code", "quantity", "invoice_date", "customer_id"]
    dup_count = df.duplicated(subset=dup_cols, keep=False).sum()
    check(
        f"No remaining duplicates on key columns (found {dup_count})",
        dup_count == 0,
        critical=False,
    )

    # -----------------------------------------------------------------
    # 7. Dtype checks
    # -----------------------------------------------------------------
    results.append("\n--- Dtype Checks ---")

    check(
        f"invoice_date is datetime (actual: {df['invoice_date'].dtype})",
        pd.api.types.is_datetime64_any_dtype(df["invoice_date"]),
    )
    check(
        f"quantity is numeric (actual: {df['quantity'].dtype})",
        pd.api.types.is_numeric_dtype(df["quantity"]),
    )
    check(
        f"price is numeric (actual: {df['price'].dtype})",
        pd.api.types.is_numeric_dtype(df["price"]),
    )

    # -----------------------------------------------------------------
    # 8. Calibration/holdout split checks
    # -----------------------------------------------------------------
    results.append("\n--- Split Checks ---")

    cal_end = pd.Timestamp(config["data"]["calibration_end"])
    holdout_start = pd.Timestamp(config["data"]["holdout_start"])

    cal_max_date = df_cal["invoice_date"].max()
    holdout_min_date = df_holdout["invoice_date"].min()

    check(
        f"Calibration ends before {cal_end.date()} (actual: {cal_max_date.date()})",
        cal_max_date <= cal_end,
    )
    check(
        f"Holdout starts on/after {holdout_start.date()} (actual: {holdout_min_date.date()})",
        holdout_min_date >= holdout_start,
    )

    # No customer leakage check isn't needed here since customers CAN
    # appear in both periods (that's the point of calibration/holdout)

    cal_customers = df_cal["customer_id"].nunique()
    holdout_customers = df_holdout["customer_id"].nunique()
    check(
        f"Calibration has enough customers (actual: {cal_customers})",
        cal_customers >= 3000,
    )
    check(
        f"Holdout has enough customers (actual: {holdout_customers})",
        holdout_customers >= 1000,
    )

    # -----------------------------------------------------------------
    # 9. Compute and save baseline distributions for drift detection
    # -----------------------------------------------------------------
    results.append("\n--- Baseline Stats ---")

    numeric_cols = ["quantity", "price", "total_amount"]
    baseline_stats = df[numeric_cols].agg(["mean", "std", "min", "max", "median"]).T
    baseline_stats.columns = ["mean", "std", "min", "max", "median"]

    # Add percentiles
    for q in [0.05, 0.25, 0.75, 0.95]:
        baseline_stats[f"p{int(q*100)}"] = df[numeric_cols].quantile(q)

    baseline_path = PROJECT_ROOT / "data" / "processed" / "baseline_stats.csv"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_stats.to_csv(baseline_path)
    results.append(f"  Saved baseline stats to {baseline_path.name}")

    for col in numeric_cols:
        results.append(
            f"  {col}: mean={baseline_stats.loc[col, 'mean']:.2f}, "
            f"std={baseline_stats.loc[col, 'std']:.2f}, "
            f"median={baseline_stats.loc[col, 'median']:.2f}"
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    results.append("\n" + "=" * 60)
    results.append(f"RESULTS: {passed} passed, {warned} warnings, {failed} failed")

    if failed > 0:
        results.append("STATUS: FAILED - fix critical issues before proceeding")
    else:
        results.append("STATUS: PASSED")

    results.append("=" * 60)

    report = "\n".join(results)

    # Save report to file
    report_path = PROJECT_ROOT / "reports" / "data_validation.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Halt if critical checks failed
    if failed > 0:
        print(report)
        raise ValueError(
            f"Data validation failed with {failed} critical error(s). "
            f"See report above. Fix issues before proceeding."
        )

    return report
