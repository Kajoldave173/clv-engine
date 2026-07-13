"""Data ingestion: load the UCI Online Retail II dataset.

The raw file is an .xlsx with two sheets:
  - 'Year 2009-2010'
  - 'Year 2010-2011'

This module loads both, concatenates them, and standardizes column names.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import PROJECT_ROOT, save_csv


def ingest_data(config: dict[str, Any]) -> pd.DataFrame:
    """Load raw transaction data from the Excel file.

    Args:
        config: Pipeline configuration dictionary (from params.yaml).

    Returns:
        Combined DataFrame with standardized column names.
    """
    raw_path = PROJECT_ROOT / config["data"]["raw_path"]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. "
            "Download from https://archive.ics.uci.edu/dataset/502/online+retail+ii "
            "and place in data/raw/"
        )

    print(f"Loading {raw_path}...")

    # Load both sheets
    sheet1 = pd.read_excel(raw_path, sheet_name="Year 2009-2010", engine="openpyxl")
    sheet2 = pd.read_excel(raw_path, sheet_name="Year 2010-2011", engine="openpyxl")

    print(f"  Sheet 'Year 2009-2010': {len(sheet1):,} rows")
    print(f"  Sheet 'Year 2010-2011': {len(sheet2):,} rows")

    # Concatenate into single DataFrame
    df = pd.concat([sheet1, sheet2], ignore_index=True)

    # Standardize column names: lowercase, underscores, no spaces
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Rename for consistency across the pipeline
    rename_map = {
        "invoice": "invoice",
        "stockcode": "stock_code",
        "description": "description",
        "quantity": "quantity",
        "invoicedate": "invoice_date",
        "price": "price",
        "customer_id": "customer_id",
        "country": "country",
    }
    df = df.rename(columns=rename_map)

    # Ensure invoice_date is datetime
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    # Cast customer_id to nullable integer (some rows will be NaN)
    df["customer_id"] = df["customer_id"].astype("Int64")

    # Cast invoice to string (some start with 'C' for cancellations)
    df["invoice"] = df["invoice"].astype(str)

    print(f"  Combined: {len(df):,} rows, {df['customer_id'].nunique()} unique customers (incl. NaN)")
    print(f"  Date range: {df['invoice_date'].min()} to {df['invoice_date'].max()}")

    return df
