"""Shared pytest fixtures for the CLV engine test suite.

Provides synthetic transaction data with 5 carefully designed customers,
a sample configuration dict, and helper utilities. Each customer exercises
a specific edge case in the feature computation and scoring logic.

Customers:
    A (ID 10001): 5 orders, regular ~60-day intervals, 3 categories
    B (ID 10002): 1 order only (one-time buyer)
    C (ID 10003): 3 orders, all on Saturdays, 1 category
    D (ID 10004): 2 orders, 319-day gap between them
    E (ID 10005): 4 orders, accelerating gaps (89 -> 45 -> 16 days)
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_config():
    """Minimal params.yaml configuration for testing."""
    return {
        "data": {
            "raw_path": "data/raw/online_retail_II.xlsx",
            "calibration_end": "2011-06-30",
            "holdout_start": "2011-07-01",
            "holdout_end": "2011-12-09",
            "min_transactions": 2,
        },
        "features": {
            "rfm_monetary": "total_revenue",
        },
        "models": {
            "probabilistic": {
                "penalizer_coef": 0.001,
                "prediction_horizon_months": 6,
            },
        },
        "segmentation": {
            "method": "kmeans",
            "n_clusters": 5,
            "features_used": [
                "frequency",
                "recency",
                "monetary_value",
                "avg_basket_size",
                "category_breadth",
            ],
        },
    }


@pytest.fixture
def synthetic_transactions():
    """Create synthetic transaction data with known properties.

    Each customer has specific purchase patterns so that behavioral
    features can be verified against hand-computed expected values.

    Returns:
        DataFrame with columns matching the cleaned transaction schema:
        customer_id, invoice_date, invoice, stock_code, description,
        quantity, price, total_amount, country.
    """
    rows = [
        # ── Customer A: 5 orders, regular intervals, 3 categories ──
        # Invoice I001: 2 items, total = 25.00
        (10001, "2010-01-15", "I001", "85001", "ITEM_A1", 2, 10.00, 20.00, "United Kingdom"),
        (10001, "2010-01-15", "I001", "22001", "ITEM_A2", 1, 5.00, 5.00, "United Kingdom"),
        # Invoice I002: 1 item, total = 24.00
        (10001, "2010-03-15", "I002", "85002", "ITEM_A3", 3, 8.00, 24.00, "United Kingdom"),
        # Invoice I003: 1 item, total = 15.00
        (10001, "2010-05-15", "I003", "30001", "ITEM_A4", 1, 15.00, 15.00, "United Kingdom"),
        # Invoice I004: 1 item, total = 24.00
        (10001, "2010-07-15", "I004", "22002", "ITEM_A5", 2, 12.00, 24.00, "United Kingdom"),
        # Invoice I005: 1 item, total = 30.00
        (10001, "2010-09-15", "I005", "85003", "ITEM_A6", 1, 30.00, 30.00, "United Kingdom"),

        # ── Customer B: 1 order (one-time buyer) ──────────────────
        (10002, "2010-06-01", "I006", "85001", "ITEM_B1", 1, 20.00, 20.00, "United Kingdom"),

        # ── Customer C: 3 orders, all Saturdays, 1 category (22xx) ──
        # 2010-02-06 is a Saturday
        (10003, "2010-02-06", "I007", "22001", "ITEM_C1", 2, 10.00, 20.00, "United Kingdom"),
        # 2010-04-10 is a Saturday
        (10003, "2010-04-10", "I008", "22002", "ITEM_C2", 1, 15.00, 15.00, "United Kingdom"),
        # 2010-06-05 is a Saturday
        (10003, "2010-06-05", "I009", "22003", "ITEM_C3", 3, 5.00, 15.00, "United Kingdom"),

        # ── Customer D: 2 orders, large gap (319 days) ────────────
        (10004, "2010-01-05", "I010", "30001", "ITEM_D1", 1, 50.00, 50.00, "United Kingdom"),
        (10004, "2010-11-20", "I011", "85001", "ITEM_D2", 2, 25.00, 50.00, "United Kingdom"),

        # ── Customer E: 4 orders, accelerating gaps ───────────────
        # Gaps: 89 days, 45 days, 16 days
        (10005, "2010-02-01", "I012", "85001", "ITEM_E1", 1, 10.00, 10.00, "United Kingdom"),
        (10005, "2010-05-01", "I013", "22001", "ITEM_E2", 2, 8.00, 16.00, "United Kingdom"),
        (10005, "2010-06-15", "I014", "30001", "ITEM_E3", 1, 20.00, 20.00, "United Kingdom"),
        (10005, "2010-07-01", "I015", "85002", "ITEM_E4", 3, 5.00, 15.00, "United Kingdom"),
    ]

    df = pd.DataFrame(rows, columns=[
        "customer_id", "invoice_date", "invoice", "stock_code",
        "description", "quantity", "price", "total_amount", "country",
    ])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


@pytest.fixture
def calibration_transactions(synthetic_transactions):
    """Calibration-period subset (all synthetic data falls before 2011-06-30)."""
    cal_end = pd.Timestamp("2011-06-30")
    return synthetic_transactions[synthetic_transactions["invoice_date"] <= cal_end].copy()


@pytest.fixture
def observation_end():
    """Standard observation end date for scoring tests.

    Uses the max date in synthetic data (2010-11-20) rather than the
    calibration end to match how the batch scorer computes it.
    """
    return pd.Timestamp("2010-11-20")
