"""Advanced behavioral feature engineering.

Computes customer-level features from calibration-period transactions
that capture patterns the probabilistic models cannot see:
purchase timing regularity, basket trends, product diversity,
and engagement trajectory.

These features feed the XGBoost CLV regressor and churn classifier.
"""

from typing import Any

import pandas as pd
import numpy as np

from src.utils.io import PROJECT_ROOT, load_csv, save_csv


def build_behavioral_features(config: dict[str, Any]) -> pd.DataFrame:
    """Build behavioral features from calibration transactions.

    Aggregates raw transactions to invoice level first, then computes
    per-customer features across four dimensions: purchase timing,
    basket behavior, product diversity, and engagement trajectory.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        DataFrame indexed by customer_id with 16 behavioral features.
    """
    cal_path = PROJECT_ROOT / "data" / "interim" / "transactions_calibration.csv"
    df = load_csv(cal_path, parse_dates=["invoice_date"])
    cal_end = pd.Timestamp(config["data"]["calibration_end"])

    print(f"Building behavioral features from {len(df):,} calibration transactions...")

    # =================================================================
    # Step 1: Aggregate transactions to invoice level
    # =================================================================
    # Each invoice can have many line items. We need per-visit summaries.
    invoices = (
        df.groupby(["customer_id", "invoice"])
        .agg(
            invoice_date=("invoice_date", "first"),
            total_value=("total_amount", "sum"),
            n_items=("stock_code", "nunique"),
        )
        .reset_index()
        .sort_values(["customer_id", "invoice_date"])
    )
    print(f"  Aggregated to {len(invoices):,} invoices")

    # =================================================================
    # Step 2: Simple aggregation features (groupby, no sequences)
    # =================================================================
    agg = (
        invoices.groupby("customer_id")
        .agg(
            n_orders=("invoice", "count"),
            first_purchase=("invoice_date", "min"),
            last_purchase=("invoice_date", "max"),
            avg_basket_size=("n_items", "mean"),
            avg_basket_value=("total_value", "mean"),
            max_single_transaction=("total_value", "max"),
        )
    )

    # Derived date features
    agg["tenure_days"] = (cal_end - agg["first_purchase"]).dt.days
    agg["days_since_last_purchase"] = (cal_end - agg["last_purchase"]).dt.days
    # lifecycle_stage: 1.0 = bought very recently, 0.0 = been silent a long time
    # clip tenure to avoid division by zero for same-day customers
    agg["lifecycle_stage"] = 1.0 - (
        agg["days_since_last_purchase"] / agg["tenure_days"].clip(lower=1)
    )
    agg = agg.drop(columns=["first_purchase", "last_purchase"])

    # =================================================================
    # Step 3: Weekend purchase ratio
    # =================================================================
    invoices["is_weekend"] = invoices["invoice_date"].dt.dayofweek >= 5
    weekend = (
        invoices.groupby("customer_id")["is_weekend"]
        .mean()
        .rename("weekend_purchase_ratio")
    )

    # =================================================================
    # Step 4: Category features (from transaction-level data)
    # =================================================================
    # Use first 2 characters of stock_code as a category proxy
    df["category"] = df["stock_code"].astype(str).str[:2]

    cat_breadth = (
        df.groupby("customer_id")["category"]
        .nunique()
        .rename("category_breadth")
    )

    # Herfindahl index: sum of squared spending shares across categories
    # 1.0 = all spending in one category, lower = more diversified
    cat_spend = (
        df.groupby(["customer_id", "category"])["total_amount"]
        .sum()
        .reset_index(name="cat_amount")
    )
    cust_total = (
        df.groupby("customer_id")["total_amount"]
        .sum()
        .reset_index(name="cust_total")
    )
    cat_spend = cat_spend.merge(cust_total, on="customer_id")
    cat_spend["share_sq"] = (cat_spend["cat_amount"] / cat_spend["cust_total"]) ** 2
    hhi = (
        cat_spend.groupby("customer_id")["share_sq"]
        .sum()
        .rename("category_concentration")
    )

    # =================================================================
    # Step 5: Sequential features (need per-customer iteration)
    # These capture trends and patterns across a customer's order history.
    # =================================================================
    def compute_sequences(group):
        """Compute sequence-based features for one customer's invoices."""
        group = group.sort_values("invoice_date")
        n = len(group)
        result = {}

        if n >= 2:
            # Inter-purchase time statistics
            gaps = group["invoice_date"].diff().dt.days.dropna().values
            result["inter_purchase_time_mean"] = float(gaps.mean())
            result["inter_purchase_time_std"] = (
                float(gaps.std()) if len(gaps) >= 2 else 0.0
            )

            # Trend of gaps: positive = slowing down, negative = speeding up
            if len(gaps) >= 2:
                result["inter_purchase_time_trend"] = float(
                    np.polyfit(np.arange(len(gaps)), gaps, 1)[0]
                )
            else:
                result["inter_purchase_time_trend"] = 0.0

            # Monetary trend: slope of per-order revenue over time
            vals = group["total_value"].values
            x = np.arange(n)
            result["monetary_trend"] = float(np.polyfit(x, vals, 1)[0])

            # Monetary CV: spending consistency (lower = more consistent)
            result["monetary_cv"] = (
                float(vals.std() / vals.mean()) if vals.mean() != 0 else 0.0
            )

            # Basket size trend: slope of items-per-order over time
            bsizes = group["n_items"].values
            result["basket_size_trend"] = float(np.polyfit(x, bsizes, 1)[0])

            # Purchase velocity: orders in second half vs first half of tenure
            # > 1 means accelerating, < 1 means decelerating
            mid_date = group["invoice_date"].iloc[0] + (
                group["invoice_date"].iloc[-1] - group["invoice_date"].iloc[0]
            ) / 2
            early = (group["invoice_date"] <= mid_date).sum()
            recent = (group["invoice_date"] > mid_date).sum()
            result["purchase_velocity_recent_vs_early"] = float(
                recent / max(early, 1)
            )
        else:
            # One-time buyers: neutral defaults
            result["inter_purchase_time_mean"] = 0.0
            result["inter_purchase_time_std"] = 0.0
            result["inter_purchase_time_trend"] = 0.0
            result["monetary_trend"] = 0.0
            result["monetary_cv"] = 0.0
            result["basket_size_trend"] = 0.0
            result["purchase_velocity_recent_vs_early"] = 1.0

        return pd.Series(result)

    print("  Computing sequential features (this may take a moment)...")
    seq = invoices.groupby("customer_id").apply(compute_sequences, include_groups=False)

    # =================================================================
    # Step 6: Combine all features
    # =================================================================
    features = (
        agg
        .join(weekend)
        .join(cat_breadth)
        .join(hhi)
        .join(seq)
    )

    # Drop tenure_days -- it duplicates T from the RFM table
    features = features.drop(columns=["tenure_days"])

    print(f"\nBehavioral features: {len(features):,} customers, "
          f"{features.shape[1]} features")
    print(f"\nFeature summary:")
    for col in features.columns:
        print(f"  {col:40s} mean={features[col].mean():>10.3f}  "
              f"std={features[col].std():>10.3f}  "
              f"nulls={features[col].isna().sum()}")

    # Save
    output_path = PROJECT_ROOT / "data" / "processed" / "features_behavioral.csv"
    save_csv(features, output_path, index=True)
    print(f"\nBehavioral features saved to {output_path}")

    return features