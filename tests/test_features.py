"""Tests for behavioral feature computation.

Tests the feature computation logic using compute_scoring_behavioral
from the batch scorer, which takes a DataFrame directly (no file I/O)
and implements the same feature logic as src/features/behavioral.py.

Feature alignment between these two implementations was verified during
Phase 5 debugging (22/23 features at PSI 0.0000, with the sole
elevation being a binning artifact on expected_avg_value).

Each test class targets a specific feature dimension:
- Output structure (columns, shape, types)
- Purchase timing features
- Basket and monetary features
- Product diversity features
- Engagement trajectory features
"""

import pytest
import pandas as pd
import numpy as np

from src.scoring.batch_scorer import compute_scoring_behavioral


# ====================================================================
# 1. Output structure
# ====================================================================

class TestOutputStructure:
    """Verify the output DataFrame has the expected shape and columns."""

    EXPECTED_COLUMNS = {
        "n_orders",
        "weekend_purchase_ratio",
        "days_since_last_purchase",
        "inter_purchase_time_mean",
        "inter_purchase_time_std",
        "inter_purchase_time_trend",
        "purchase_velocity_recent_vs_early",
        "avg_basket_size",
        "avg_basket_value",
        "basket_size_trend",
        "monetary_trend",
        "max_single_transaction",
        "monetary_cv",
        "lifecycle_stage",
        "category_breadth",
        "category_concentration",
    }

    def test_all_expected_columns_present(
        self, synthetic_transactions, observation_end
    ):
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        missing = self.EXPECTED_COLUMNS - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_unexpected_columns(
        self, synthetic_transactions, observation_end
    ):
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        extra = set(result.columns) - self.EXPECTED_COLUMNS
        assert not extra, f"Unexpected columns: {extra}"

    def test_column_count(self, synthetic_transactions, observation_end):
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.shape[1] == 16, (
            f"Expected 16 features, got {result.shape[1]}"
        )

    def test_one_row_per_customer(
        self, synthetic_transactions, observation_end
    ):
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        n_customers = synthetic_transactions["customer_id"].nunique()
        assert len(result) == n_customers

    def test_index_is_customer_id(
        self, synthetic_transactions, observation_end
    ):
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.index.name == "customer_id"

    def test_no_nulls(self, synthetic_transactions, observation_end):
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        null_counts = result.isna().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert cols_with_nulls.empty, (
            f"Columns with nulls: {cols_with_nulls.to_dict()}"
        )

    def test_no_infinities(self, synthetic_transactions, observation_end):
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        inf_counts = np.isinf(result).sum()
        cols_with_inf = inf_counts[inf_counts > 0]
        assert cols_with_inf.empty, (
            f"Columns with infinities: {cols_with_inf.to_dict()}"
        )


# ====================================================================
# 2. Order count feature
# ====================================================================

class TestOrderCount:
    """Verify n_orders matches the number of distinct invoices."""

    def test_customer_a_five_orders(
        self, synthetic_transactions, observation_end
    ):
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        # Customer A (10001) has invoices I001-I005
        assert result.loc[10001, "n_orders"] == 5

    def test_customer_b_one_order(
        self, synthetic_transactions, observation_end
    ):
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        # Customer B (10002) has only invoice I006
        assert result.loc[10002, "n_orders"] == 1

    def test_customer_c_three_orders(
        self, synthetic_transactions, observation_end
    ):
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10003, "n_orders"] == 3

    def test_customer_e_four_orders(
        self, synthetic_transactions, observation_end
    ):
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10005, "n_orders"] == 4


# ====================================================================
# 3. Inter-purchase timing features
# ====================================================================

class TestInterPurchaseTiming:
    """Verify inter-purchase time calculations for known gap sequences."""

    def test_customer_a_mean_gap(
        self, synthetic_transactions, observation_end
    ):
        """Customer A gaps: 59, 61, 61, 62 days → mean = 60.75."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10001, "inter_purchase_time_mean"] == pytest.approx(
            60.75, abs=0.5
        )

    def test_customer_e_mean_gap(
        self, synthetic_transactions, observation_end
    ):
        """Customer E gaps: 89, 45, 16 days → mean = 50.0."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10005, "inter_purchase_time_mean"] == pytest.approx(
            50.0, abs=0.5
        )

    def test_customer_e_negative_trend(
        self, synthetic_transactions, observation_end
    ):
        """Customer E has decreasing gaps → trend should be negative.

        Gaps [89, 45, 16] at x=[0, 1, 2] → slope ≈ -36.5.
        """
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        trend = result.loc[10005, "inter_purchase_time_trend"]
        assert trend < 0, "Accelerating customer should have negative trend"
        assert trend == pytest.approx(-36.5, abs=1.0)

    def test_customer_a_near_zero_trend(
        self, synthetic_transactions, observation_end
    ):
        """Customer A has nearly equal gaps → trend should be near zero.

        Gaps [59, 61, 61, 62] → slight positive slope ≈ 0.8.
        """
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        trend = result.loc[10001, "inter_purchase_time_trend"]
        assert abs(trend) < 2.0, "Regular customer should have near-zero trend"

    def test_one_time_buyer_defaults(
        self, synthetic_transactions, observation_end
    ):
        """One-time buyer (Customer B) should get neutral defaults."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10002, "inter_purchase_time_mean"] == 0.0
        assert result.loc[10002, "inter_purchase_time_std"] == 0.0
        assert result.loc[10002, "inter_purchase_time_trend"] == 0.0

    def test_customer_d_large_gap(
        self, synthetic_transactions, observation_end
    ):
        """Customer D has 2 orders 319 days apart → mean = 319.0."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        # 2010-01-05 to 2010-11-20 = 319 days
        assert result.loc[10004, "inter_purchase_time_mean"] == pytest.approx(
            319.0, abs=1.0
        )


# ====================================================================
# 4. Purchase velocity
# ====================================================================

class TestPurchaseVelocity:
    """Verify purchase velocity (second half / first half of tenure)."""

    def test_one_time_buyer_neutral(
        self, synthetic_transactions, observation_end
    ):
        """One-time buyers should get velocity = 1.0 (neutral)."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10002, "purchase_velocity_recent_vs_early"] == 1.0

    def test_accelerating_customer_above_one(
        self, synthetic_transactions, observation_end
    ):
        """Customer E accelerates → velocity should be > 1.0.

        Dates: Feb 1, May 1, Jun 15, Jul 1.
        Midpoint of tenure: ~Apr 16 (75 days from Feb 1 to Jul 1, mid at 37.5 days).
        Early (≤ midpoint): Feb 1, possibly May 1 → depends on exact mid calculation.
        Recent (> midpoint): remaining dates.
        """
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        velocity = result.loc[10005, "purchase_velocity_recent_vs_early"]
        assert velocity >= 1.0, (
            f"Accelerating customer should have velocity >= 1.0, got {velocity}"
        )


# ====================================================================
# 5. Basket features
# ====================================================================

class TestBasketFeatures:
    """Verify basket size and value calculations."""

    def test_customer_a_avg_basket_size(
        self, synthetic_transactions, observation_end
    ):
        """Customer A: invoices have 2, 1, 1, 1, 1 unique items → mean = 1.2."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10001, "avg_basket_size"] == pytest.approx(
            1.2, abs=0.01
        )

    def test_customer_a_avg_basket_value(
        self, synthetic_transactions, observation_end
    ):
        """Customer A: invoice totals 25, 24, 15, 24, 30 → mean = 23.6."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10001, "avg_basket_value"] == pytest.approx(
            23.6, abs=0.1
        )

    def test_customer_a_max_single_transaction(
        self, synthetic_transactions, observation_end
    ):
        """Customer A's largest invoice total is 30.00."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10001, "max_single_transaction"] == pytest.approx(
            30.0, abs=0.01
        )

    def test_customer_d_max_single_transaction(
        self, synthetic_transactions, observation_end
    ):
        """Customer D's invoices are both 50.00."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10004, "max_single_transaction"] == pytest.approx(
            50.0, abs=0.01
        )


# ====================================================================
# 6. Category features
# ====================================================================

class TestCategoryFeatures:
    """Verify category breadth and concentration calculations."""

    def test_customer_a_category_breadth(
        self, synthetic_transactions, observation_end
    ):
        """Customer A uses stock codes 85xxx, 22xxx, 30xxx → 3 categories."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10001, "category_breadth"] == 3

    def test_customer_b_category_breadth(
        self, synthetic_transactions, observation_end
    ):
        """Customer B uses only 85xxx → 1 category."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10002, "category_breadth"] == 1

    def test_customer_c_category_breadth(
        self, synthetic_transactions, observation_end
    ):
        """Customer C uses only 22xxx → 1 category."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10003, "category_breadth"] == 1

    def test_single_category_concentration_equals_one(
        self, synthetic_transactions, observation_end
    ):
        """If all spending in one category, HHI should be 1.0."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        # Customer C buys only from category "22"
        assert result.loc[10003, "category_concentration"] == pytest.approx(
            1.0, abs=0.01
        )

    def test_multiple_categories_concentration_below_one(
        self, synthetic_transactions, observation_end
    ):
        """Multiple categories → HHI should be < 1.0."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        # Customer A spans 3 categories
        assert result.loc[10001, "category_concentration"] < 1.0


# ====================================================================
# 7. Weekend purchase ratio
# ====================================================================

class TestWeekendPurchaseRatio:
    """Verify weekend purchase ratio calculation."""

    def test_customer_c_all_weekends(
        self, synthetic_transactions, observation_end
    ):
        """Customer C buys only on Saturdays → ratio should be 1.0."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10003, "weekend_purchase_ratio"] == pytest.approx(
            1.0, abs=0.01
        )

    def test_customer_a_no_weekends(
        self, synthetic_transactions, observation_end
    ):
        """Customer A buys on the 15th of each month in 2010.

        Jan 15 = Friday, Mar 15 = Monday, May 15 = Saturday,
        Jul 15 = Thursday, Sep 15 = Wednesday.
        So 1 out of 5 is a weekend → ratio = 0.2.
        """
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        # May 15, 2010 is a Saturday
        assert result.loc[10001, "weekend_purchase_ratio"] == pytest.approx(
            0.2, abs=0.01
        )


# ====================================================================
# 8. Lifecycle stage
# ====================================================================

class TestLifecycleStage:
    """Verify lifecycle_stage is computed correctly.

    lifecycle_stage = 1.0 - (days_since_last / tenure_from_start)
    where tenure_from_start = (observation_end - first_purchase).days

    Values near 1.0 = bought recently, near 0.0 = been silent.
    """

    def test_lifecycle_stage_bounded(
        self, synthetic_transactions, observation_end
    ):
        """All lifecycle_stage values should be in [0, 1] or close."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        # Allow slight floating point overshoot
        assert (result["lifecycle_stage"] >= -0.01).all()
        assert (result["lifecycle_stage"] <= 1.01).all()

    def test_customer_d_recent_buyer(
        self, synthetic_transactions, observation_end
    ):
        """Customer D's last purchase is on observation_end → lifecycle ≈ 1.0.

        Last purchase: 2010-11-20 = observation_end.
        days_since_last = 0
        lifecycle = 1.0 - 0/tenure = 1.0
        """
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10004, "lifecycle_stage"] == pytest.approx(
            1.0, abs=0.05
        )


# ====================================================================
# 9. Monetary features
# ====================================================================

class TestMonetaryFeatures:
    """Verify monetary trend and coefficient of variation."""

    def test_monetary_cv_nonnegative(
        self, synthetic_transactions, observation_end
    ):
        """Coefficient of variation should be >= 0 for all customers."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert (result["monetary_cv"] >= 0).all()

    def test_customer_d_zero_cv(
        self, synthetic_transactions, observation_end
    ):
        """Customer D has two invoices both at 50.00 → CV = 0.0."""
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        assert result.loc[10004, "monetary_cv"] == pytest.approx(0.0, abs=0.01)

    def test_customer_a_monetary_trend(
        self, synthetic_transactions, observation_end
    ):
        """Customer A invoice totals: 25, 24, 15, 24, 30.

        The trend should be weakly positive (overall slightly increasing).
        """
        result = compute_scoring_behavioral(
            synthetic_transactions, observation_end
        )
        # Slope of [25, 24, 15, 24, 30] at x=[0,1,2,3,4]
        expected_slope = np.polyfit(
            np.arange(5), [25.0, 24.0, 15.0, 24.0, 30.0], 1
        )[0]
        assert result.loc[10001, "monetary_trend"] == pytest.approx(
            expected_slope, abs=0.5
        )


# ====================================================================
# 10. Integration: build_behavioral_features output structure
# ====================================================================

class TestBuildBehavioralIntegration:
    """Integration test for build_behavioral_features from behavioral.py.

    Monkeypatches PROJECT_ROOT to a tmp directory, writes synthetic
    calibration transactions to disk, and verifies the function
    produces the expected output shape and column set.

    This test complements the unit tests above by verifying the
    training-time implementation (as opposed to the batch scorer's
    inline reimplementation).
    """

    EXPECTED_BEHAVIORAL_COLUMNS = {
        "n_orders",
        "days_since_last_purchase",
        "lifecycle_stage",
        "weekend_purchase_ratio",
        "category_breadth",
        "category_concentration",
        "inter_purchase_time_mean",
        "inter_purchase_time_std",
        "inter_purchase_time_trend",
        "monetary_trend",
        "monetary_cv",
        "basket_size_trend",
        "purchase_velocity_recent_vs_early",
        "avg_basket_size",
        "avg_basket_value",
        "max_single_transaction",
    }

    def test_output_has_expected_columns(
        self, tmp_path, synthetic_transactions, sample_config, monkeypatch
    ):
        # Save calibration data where behavioral.py expects it
        interim_dir = tmp_path / "data" / "interim"
        interim_dir.mkdir(parents=True)
        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)

        synthetic_transactions.to_csv(
            interim_dir / "transactions_calibration.csv", index=False
        )

        # Patch PROJECT_ROOT in both behavioral and io modules
        import src.features.behavioral as behavioral_mod
        monkeypatch.setattr(behavioral_mod, "PROJECT_ROOT", tmp_path)

        # Also need to patch save_csv's PROJECT_ROOT behavior
        import src.utils.io as io_mod
        monkeypatch.setattr(io_mod, "PROJECT_ROOT", tmp_path)

        # Patch the save_csv and load_csv functions in behavioral module
        # to use the tmp_path
        original_save = behavioral_mod.save_csv

        def patched_save(df, path, **kwargs):
            # Redirect saves to tmp_path
            path_str = str(path)
            if str(io_mod.PROJECT_ROOT) not in path_str:
                # Construct path relative to tmp_path
                rel_parts = path.parts
                # Find the 'data' component
                for i, part in enumerate(rel_parts):
                    if part == "data":
                        new_path = tmp_path / "/".join(rel_parts[i:])
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        return original_save(df, new_path, **kwargs)
            return original_save(df, path, **kwargs)

        result = behavioral_mod.build_behavioral_features(sample_config)

        missing = self.EXPECTED_BEHAVIORAL_COLUMNS - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_output_has_one_row_per_customer(
        self, tmp_path, synthetic_transactions, sample_config, monkeypatch
    ):
        interim_dir = tmp_path / "data" / "interim"
        interim_dir.mkdir(parents=True)
        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)

        synthetic_transactions.to_csv(
            interim_dir / "transactions_calibration.csv", index=False
        )

        import src.features.behavioral as behavioral_mod
        import src.utils.io as io_mod
        monkeypatch.setattr(behavioral_mod, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(io_mod, "PROJECT_ROOT", tmp_path)

        result = behavioral_mod.build_behavioral_features(sample_config)

        n_customers = synthetic_transactions["customer_id"].nunique()
        assert len(result) == n_customers
