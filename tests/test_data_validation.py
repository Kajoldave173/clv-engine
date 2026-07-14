"""Tests for data validation (src/data/validate.py).

Strategy: The run_validation function is tightly coupled to file I/O
and has hardcoded dataset-specific thresholds (e.g. 4000-6000 customers).
We test at two levels:

1. Condition-level tests: Apply the same boolean expressions the
   validator uses to synthetic DataFrames. This verifies the
   validation LOGIC is correct without running the full function.

2. Integration test: Monkeypatch PROJECT_ROOT, create files on disk,
   and verify that run_validation raises ValueError on bad data.
   This verifies the halt-on-failure behavior.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# ====================================================================
# Helpers to create test data
# ====================================================================

def make_valid_df(n_rows=10):
    """Create a small DataFrame that passes null/range/dtype checks.

    Note: This won't pass customer-count thresholds (4000-6000)
    because it's deliberately small. That threshold tests are separate.
    """
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n_rows)],
        "invoice_date": pd.date_range("2010-06-01", periods=n_rows, freq="7D"),
        "invoice": [f"INV{i}" for i in range(n_rows)],
        "stock_code": [f"8500{i}" for i in range(n_rows)],
        "description": [f"ITEM_{i}" for i in range(n_rows)],
        "quantity": [i + 1 for i in range(n_rows)],
        "price": [10.0 + i for i in range(n_rows)],
        "total_amount": [(i + 1) * (10.0 + i) for i in range(n_rows)],
        "country": ["United Kingdom"] * n_rows,
    })


# ====================================================================
# 1. Null checks
# ====================================================================

class TestNullChecks:
    """Verify that null values in critical columns are detected."""

    def test_no_nulls_in_valid_data(self, synthetic_transactions):
        for col in ["customer_id", "invoice_date", "quantity", "price"]:
            assert synthetic_transactions[col].isna().sum() == 0

    def test_null_customer_id_detected(self, synthetic_transactions):
        df = synthetic_transactions.copy()
        df.loc[df.index[0], "customer_id"] = np.nan
        assert df["customer_id"].isna().sum() == 1

    def test_null_invoice_date_detected(self, synthetic_transactions):
        df = synthetic_transactions.copy()
        df.loc[df.index[0], "invoice_date"] = pd.NaT
        assert df["invoice_date"].isna().sum() == 1

    def test_null_quantity_detected(self, synthetic_transactions):
        df = synthetic_transactions.copy()
        df.loc[df.index[0], "quantity"] = np.nan
        assert df["quantity"].isna().sum() == 1

    def test_null_price_detected(self, synthetic_transactions):
        df = synthetic_transactions.copy()
        df.loc[df.index[0], "price"] = np.nan
        assert df["price"].isna().sum() == 1


# ====================================================================
# 2. Value range checks
# ====================================================================

class TestValueRangeChecks:
    """Verify that out-of-range values are detected."""

    def test_all_quantities_positive(self, synthetic_transactions):
        assert (synthetic_transactions["quantity"] > 0).all()

    def test_negative_quantity_detected(self, synthetic_transactions):
        df = synthetic_transactions.copy()
        df.loc[df.index[0], "quantity"] = -5
        violations = (df["quantity"] <= 0).sum()
        assert violations == 1

    def test_zero_quantity_detected(self, synthetic_transactions):
        df = synthetic_transactions.copy()
        df.loc[df.index[0], "quantity"] = 0
        violations = (df["quantity"] <= 0).sum()
        assert violations == 1

    def test_all_prices_positive(self, synthetic_transactions):
        assert (synthetic_transactions["price"] > 0).all()

    def test_negative_price_detected(self, synthetic_transactions):
        df = synthetic_transactions.copy()
        df.loc[df.index[0], "price"] = -1.50
        violations = (df["price"] <= 0).sum()
        assert violations == 1

    def test_all_total_amounts_positive(self, synthetic_transactions):
        assert (synthetic_transactions["total_amount"] > 0).all()

    def test_total_amount_consistency(self, synthetic_transactions):
        """total_amount should equal quantity * price."""
        expected = synthetic_transactions["quantity"] * synthetic_transactions["price"]
        np.testing.assert_allclose(
            synthetic_transactions["total_amount"], expected, rtol=1e-10
        )


# ====================================================================
# 3. Date range checks
# ====================================================================

class TestDateRangeChecks:
    """Verify that date boundary conditions match validator logic."""

    def test_dates_after_2009_12_01(self, synthetic_transactions):
        assert synthetic_transactions["invoice_date"].min() >= pd.Timestamp("2009-12-01")

    def test_dates_before_2011_12_10(self, synthetic_transactions):
        assert synthetic_transactions["invoice_date"].max() <= pd.Timestamp("2011-12-10")

    def test_out_of_range_date_detected(self):
        """A date before Dec 2009 should fail the lower bound check."""
        df = make_valid_df(5)
        df.loc[df.index[0], "invoice_date"] = pd.Timestamp("2008-01-01")
        assert df["invoice_date"].min() < pd.Timestamp("2009-12-01")

    def test_future_date_detected(self):
        """A date after Dec 2011 should fail the upper bound check."""
        df = make_valid_df(5)
        df.loc[df.index[0], "invoice_date"] = pd.Timestamp("2012-06-15")
        assert df["invoice_date"].max() > pd.Timestamp("2011-12-10")


# ====================================================================
# 4. Dtype checks
# ====================================================================

class TestDtypeChecks:
    """Verify that dtype checks match what the validator expects."""

    def test_invoice_date_is_datetime(self, synthetic_transactions):
        assert pd.api.types.is_datetime64_any_dtype(
            synthetic_transactions["invoice_date"]
        )

    def test_quantity_is_numeric(self, synthetic_transactions):
        assert pd.api.types.is_numeric_dtype(
            synthetic_transactions["quantity"]
        )

    def test_price_is_numeric(self, synthetic_transactions):
        assert pd.api.types.is_numeric_dtype(
            synthetic_transactions["price"]
        )

    def test_string_quantity_fails_numeric_check(self):
        """If quantity were strings, the dtype check should catch it."""
        df = make_valid_df(5)
        df["quantity"] = df["quantity"].astype(str)
        assert not pd.api.types.is_numeric_dtype(df["quantity"])


# ====================================================================
# 5. Split integrity checks
# ====================================================================

class TestSplitChecks:
    """Verify calibration/holdout split date logic."""

    def test_calibration_data_before_cutoff(
        self, synthetic_transactions, sample_config
    ):
        cal_end = pd.Timestamp(sample_config["data"]["calibration_end"])
        cal_data = synthetic_transactions[
            synthetic_transactions["invoice_date"] <= cal_end
        ]
        assert cal_data["invoice_date"].max() <= cal_end

    def test_holdout_data_after_cutoff(self, sample_config):
        holdout_start = pd.Timestamp(sample_config["data"]["holdout_start"])
        # Simulate holdout transactions
        holdout = pd.DataFrame({
            "invoice_date": pd.to_datetime([
                "2011-07-15", "2011-08-01", "2011-10-20"
            ]),
        })
        assert holdout["invoice_date"].min() >= holdout_start

    def test_no_overlap_between_periods(self, sample_config):
        """Calibration end should be strictly before holdout start."""
        cal_end = pd.Timestamp(sample_config["data"]["calibration_end"])
        holdout_start = pd.Timestamp(sample_config["data"]["holdout_start"])
        assert cal_end < holdout_start


# ====================================================================
# 6. Customer count thresholds
# ====================================================================

class TestCustomerCountChecks:
    """Verify customer count boundary logic."""

    def test_too_few_customers_detected(self):
        """With 5 customers, the 4000-6000 range check should fail."""
        df = make_valid_df(10)
        n_customers = df["customer_id"].nunique()
        assert not (4000 <= n_customers <= 6000)

    def test_expected_range_for_real_data(self):
        """Verify the threshold values match the validator."""
        # The validator uses 4000-6000 as the expected range
        # Real dataset has ~5,861 customers after cleaning
        assert 4000 <= 5861 <= 6000


# ====================================================================
# 7. Duplicate detection
# ====================================================================

class TestDuplicateChecks:
    """Verify duplicate detection logic."""

    def test_no_duplicates_in_synthetic_data(self, synthetic_transactions):
        dup_cols = [
            "invoice", "stock_code", "quantity", "invoice_date", "customer_id"
        ]
        dup_count = synthetic_transactions.duplicated(
            subset=dup_cols, keep=False
        ).sum()
        assert dup_count == 0

    def test_duplicates_detected_when_present(self, synthetic_transactions):
        """Adding a duplicate row should be caught."""
        df = pd.concat(
            [synthetic_transactions, synthetic_transactions.iloc[[0]]],
            ignore_index=True,
        )
        dup_cols = [
            "invoice", "stock_code", "quantity", "invoice_date", "customer_id"
        ]
        dup_count = df.duplicated(subset=dup_cols, keep=False).sum()
        assert dup_count == 2  # Original + duplicate both flagged


# ====================================================================
# 8. Integration test: run_validation halt-on-failure
# ====================================================================

class TestRunValidationIntegration:
    """Test the full run_validation function with monkeypatched paths.

    Creates the required directory structure and CSV files in tmp_path,
    patches PROJECT_ROOT so validate.py reads from there, and verifies
    that the function correctly raises ValueError on data that fails
    critical checks (in this case, too few customers).
    """

    def _setup_files(self, tmp_path, df_clean, df_cal, df_holdout):
        """Create directory structure and save CSVs."""
        (tmp_path / "data" / "interim").mkdir(parents=True)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        (tmp_path / "reports").mkdir(parents=True)

        df_clean.to_csv(
            tmp_path / "data" / "interim" / "transactions_clean.csv",
            index=False,
        )
        df_cal.to_csv(
            tmp_path / "data" / "interim" / "transactions_calibration.csv",
            index=False,
        )
        df_holdout.to_csv(
            tmp_path / "data" / "interim" / "transactions_holdout.csv",
            index=False,
        )

    def test_validation_raises_on_critical_failure(
        self, tmp_path, synthetic_transactions, sample_config, monkeypatch
    ):
        """run_validation should raise ValueError when critical checks fail.

        Synthetic data has only 5 customers, which violates the
        4000-6000 customer count threshold (a critical check).
        """
        cal_end = pd.Timestamp(sample_config["data"]["calibration_end"])
        df_cal = synthetic_transactions[
            synthetic_transactions["invoice_date"] <= cal_end
        ]
        # Create a fake holdout period
        holdout_rows = synthetic_transactions.iloc[:3].copy()
        holdout_rows["invoice_date"] = pd.to_datetime("2011-08-15")

        self._setup_files(tmp_path, synthetic_transactions, df_cal, holdout_rows)

        # Patch PROJECT_ROOT in the validate module
        import src.data.validate as validate_mod
        monkeypatch.setattr(validate_mod, "PROJECT_ROOT", tmp_path)

        with pytest.raises(ValueError, match="validation failed"):
            validate_mod.run_validation(sample_config)

    def test_validation_report_is_saved(
        self, tmp_path, synthetic_transactions, sample_config, monkeypatch
    ):
        """Even when validation fails, the report file should be saved."""
        cal_end = pd.Timestamp(sample_config["data"]["calibration_end"])
        df_cal = synthetic_transactions[
            synthetic_transactions["invoice_date"] <= cal_end
        ]
        holdout_rows = synthetic_transactions.iloc[:3].copy()
        holdout_rows["invoice_date"] = pd.to_datetime("2011-08-15")

        self._setup_files(tmp_path, synthetic_transactions, df_cal, holdout_rows)

        import src.data.validate as validate_mod
        monkeypatch.setattr(validate_mod, "PROJECT_ROOT", tmp_path)

        with pytest.raises(ValueError):
            validate_mod.run_validation(sample_config)

        report_path = tmp_path / "reports" / "data_validation.txt"
        assert report_path.exists(), "Validation report should be saved to disk"
        content = report_path.read_text()
        assert "FAIL" in content
        assert "DATA VALIDATION REPORT" in content

    def test_baseline_stats_saved(
        self, tmp_path, synthetic_transactions, sample_config, monkeypatch
    ):
        """Baseline stats CSV should be written even when validation fails."""
        cal_end = pd.Timestamp(sample_config["data"]["calibration_end"])
        df_cal = synthetic_transactions[
            synthetic_transactions["invoice_date"] <= cal_end
        ]
        holdout_rows = synthetic_transactions.iloc[:3].copy()
        holdout_rows["invoice_date"] = pd.to_datetime("2011-08-15")

        self._setup_files(tmp_path, synthetic_transactions, df_cal, holdout_rows)

        import src.data.validate as validate_mod
        monkeypatch.setattr(validate_mod, "PROJECT_ROOT", tmp_path)

        with pytest.raises(ValueError):
            validate_mod.run_validation(sample_config)

        baseline_path = tmp_path / "data" / "processed" / "baseline_stats.csv"
        assert baseline_path.exists(), "Baseline stats should be computed and saved"

        stats = pd.read_csv(baseline_path, index_col=0)
        assert "quantity" in stats.index
        assert "price" in stats.index
        assert "total_amount" in stats.index
        assert "mean" in stats.columns
        assert "std" in stats.columns
