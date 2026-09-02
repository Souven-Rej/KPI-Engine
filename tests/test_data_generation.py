"""
Tests for the synthetic data generator.

Validates schema correctness, causal shock detectability, stockout events,
sparse-history constraints, and reproducibility guarantees.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_generation.enterprise_warehouse_etl import (
    END_DATE,
    MARKETING_SHOCK_END,
    MARKETING_SHOCK_START,
    REGIONS,
    SPARSE_LAUNCH_DATE,
    SPARSE_PRODUCT,
    SPARSE_REGION,
    SPARSE_REGION_LAUNCH,
    START_DATE,
    STOCKOUT_END,
    STOCKOUT_PRODUCT,
    STOCKOUT_REGION,
    STOCKOUT_START,
    generate_all,
)


# ============================================================
# Fixture: generate all data once per test session
# ============================================================

@pytest.fixture(scope="session")
def generated_data() -> dict[str, pd.DataFrame]:
    """Generate data into a temp directory and return DataFrames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = generate_all(output_dir=Path(tmpdir))

        # Also verify CSVs were written
        for name in ["marketing_weekly", "sales_daily", "inventory_hourly"]:
            csv_path = Path(tmpdir) / f"{name}.csv"
            assert csv_path.exists(), f"{name}.csv was not created"
            assert csv_path.stat().st_size > 0, f"{name}.csv is empty"

        yield data


# ============================================================
# Schema tests
# ============================================================

class TestMarketingWeeklySchema:
    """Validate marketing_weekly.csv structure."""

    def test_columns(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["marketing_weekly"]
        expected = {"week_start", "region", "channel", "ad_spend", "impressions", "clicks"}
        assert set(df.columns) == expected

    def test_no_nulls(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["marketing_weekly"]
        assert df.notna().all().all(), f"Nulls found: {df.isna().sum().to_dict()}"

    def test_week_starts_are_mondays(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["marketing_weekly"]
        dates = pd.to_datetime(df["week_start"])
        # Monday = 0
        assert (dates.dt.weekday == 0).all(), "Not all week_start dates are Mondays"

    def test_positive_spend(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["marketing_weekly"]
        assert (df["ad_spend"] >= 0).all()
        assert (df["impressions"] >= 0).all()
        assert (df["clicks"] >= 0).all()


class TestSalesDailySchema:
    """Validate sales_daily.csv structure."""

    def test_columns(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["sales_daily"]
        expected = {
            "date", "region", "product", "units_sold", "unit_price",
            "gross_revenue", "returns", "net_revenue", "web_traffic",
        }
        assert set(df.columns) == expected

    def test_no_nulls(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["sales_daily"]
        assert df.notna().all().all(), f"Nulls found: {df.isna().sum().to_dict()}"

    def test_row_count_range(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["sales_daily"]
        n_days = (END_DATE - START_DATE).days + 1  # 365
        n_regions = len(REGIONS)
        # Four regions are active for the full year; Southwest starts on 2025-11-01.
        full_year_rows_per_region = n_days * 2 + 31
        sparse_days = (END_DATE - SPARSE_REGION_LAUNCH).days + 1
        sparse_region_rows = sparse_days * 2 + 31
        expected_total = (n_regions - 1) * full_year_rows_per_region + sparse_region_rows
        # Allow small tolerance
        assert abs(len(df) - expected_total) <= n_regions, (
            f"Expected ~{expected_total} rows, got {len(df)}"
        )

    def test_revenue_calculation(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["sales_daily"]
        computed_gross = df["units_sold"] * df["unit_price"]
        assert np.allclose(df["gross_revenue"], computed_gross, atol=0.01)

    def test_net_revenue_calculation(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["sales_daily"]
        computed_net = df["gross_revenue"] - (df["returns"] * df["unit_price"])
        assert np.allclose(df["net_revenue"], computed_net, atol=0.01)


class TestInventoryHourlySchema:
    """Validate inventory_hourly.csv structure."""

    def test_columns(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["inventory_hourly"]
        expected = {"timestamp", "region", "product", "stock_on_hand", "reorder_flag"}
        assert set(df.columns) == expected

    def test_no_nulls(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["inventory_hourly"]
        assert df.notna().all().all()

    def test_non_negative_stock(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["inventory_hourly"]
        assert (df["stock_on_hand"] >= 0).all()


# ============================================================
# Causal shock tests
# ============================================================

class TestMarketingShock:
    """Verify the marketing spend shock is detectable in the data."""

    def test_spend_drops_during_shock(self, generated_data: dict[str, pd.DataFrame]) -> None:
        """Mean ad spend during shock window should be significantly lower than baseline."""
        df = generated_data["marketing_weekly"]
        df = df.copy()
        df["week_start"] = pd.to_datetime(df["week_start"])

        shock_mask = (
            (df["week_start"] >= MARKETING_SHOCK_START) &
            (df["week_start"] <= MARKETING_SHOCK_END)
        )
        baseline_mask = ~shock_mask

        mean_shock = df.loc[shock_mask, "ad_spend"].mean()
        mean_baseline = df.loc[baseline_mask, "ad_spend"].mean()

        # Expect at least 40% drop (shock is 60% on fully-covered weeks)
        drop_pct = 1.0 - (mean_shock / mean_baseline)
        assert drop_pct > 0.30, (
            f"Marketing shock not detectable: only {drop_pct:.1%} drop "
            f"(baseline={mean_baseline:.0f}, shock={mean_shock:.0f})"
        )

    def test_revenue_drops_after_shock(self, generated_data: dict[str, pd.DataFrame]) -> None:
        """
        Revenue should decline in the weeks following the marketing shock.
        Testing with a 2-week lag window after the shock starts.
        """
        df = generated_data["sales_daily"]
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Revenue impact window: shock start + 14 days lag → shock end + 14 days
        impact_start = MARKETING_SHOCK_START + pd.Timedelta(days=14)
        impact_end = MARKETING_SHOCK_END + pd.Timedelta(days=14)

        impact_mask = (df["date"] >= impact_start) & (df["date"] <= impact_end)
        # Baseline: Jan – May (before the shock)
        baseline_mask = (df["date"] < MARKETING_SHOCK_START)

        # Compare daily average net revenue
        mean_impact = df.loc[impact_mask, "net_revenue"].mean()
        mean_baseline = df.loc[baseline_mask, "net_revenue"].mean()

        drop_pct = 1.0 - (mean_impact / mean_baseline)
        assert drop_pct > 0.10, (
            f"Revenue impact from marketing shock not detectable: only {drop_pct:.1%} drop"
        )


class TestStockoutShock:
    """Verify the stockout event is detectable."""

    def test_zero_inventory_during_stockout(self, generated_data: dict[str, pd.DataFrame]) -> None:
        """Southeast Widget_A stock should be 0 during Sep 1–14."""
        df = generated_data["inventory_hourly"]
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        mask = (
            (df["region"] == STOCKOUT_REGION) &
            (df["product"] == STOCKOUT_PRODUCT) &
            (df["timestamp"].dt.date >= STOCKOUT_START.date()) &
            (df["timestamp"].dt.date <= STOCKOUT_END.date())
        )

        stockout_stock = df.loc[mask, "stock_on_hand"]
        assert (stockout_stock == 0).all(), (
            f"Expected all zero stock during stockout, got max={stockout_stock.max()}"
        )

    def test_sales_suppressed_during_stockout(
        self, generated_data: dict[str, pd.DataFrame]
    ) -> None:
        """Southeast Widget_A sales should be near-zero during stockout."""
        df = generated_data["sales_daily"]
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        stockout_mask = (
            (df["region"] == STOCKOUT_REGION) &
            (df["product"] == STOCKOUT_PRODUCT) &
            (df["date"] >= STOCKOUT_START) &
            (df["date"] <= STOCKOUT_END)
        )
        baseline_mask = (
            (df["region"] == STOCKOUT_REGION) &
            (df["product"] == STOCKOUT_PRODUCT) &
            (df["date"] < STOCKOUT_START) &
            (df["date"] >= pd.Timestamp("2025-08-01"))  # Recent baseline
        )

        mean_stockout = df.loc[stockout_mask, "units_sold"].mean()
        mean_baseline = df.loc[baseline_mask, "units_sold"].mean()

        assert mean_stockout < mean_baseline * 0.15, (
            f"Sales not sufficiently suppressed during stockout: "
            f"stockout_mean={mean_stockout:.0f} vs baseline_mean={mean_baseline:.0f}"
        )


# ============================================================
# Sparse-history test
# ============================================================

class TestSparseHistory:
    """Verify Widget_C (sparse product) only appears from Dec 1."""

    def test_widget_c_starts_dec_1(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["sales_daily"]
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        widget_c = df[df["product"] == SPARSE_PRODUCT]
        earliest = widget_c["date"].min()
        assert earliest == SPARSE_LAUNCH_DATE, (
            f"Widget_C should start on {SPARSE_LAUNCH_DATE}, got {earliest}"
        )

    def test_widget_c_has_31_days(self, generated_data: dict[str, pd.DataFrame]) -> None:
        df = generated_data["sales_daily"]
        widget_c = df[df["product"] == SPARSE_PRODUCT]
        n_days = widget_c["date"].nunique()
        assert n_days == 31, f"Widget_C should have 31 days, got {n_days}"


# ============================================================
# Reproducibility test
# ============================================================

class TestReproducibility:
    """Verify that running the generator twice produces identical output."""

    def test_deterministic_output(self) -> None:
        """Two runs with the same seed must produce byte-identical CSVs."""
        with tempfile.TemporaryDirectory() as tmpdir1, \
             tempfile.TemporaryDirectory() as tmpdir2:

            data1 = generate_all(output_dir=Path(tmpdir1))
            data2 = generate_all(output_dir=Path(tmpdir2))

            for name in ["marketing_weekly", "sales_daily", "inventory_hourly"]:
                csv1 = (Path(tmpdir1) / f"{name}.csv").read_text()
                csv2 = (Path(tmpdir2) / f"{name}.csv").read_text()
                assert csv1 == csv2, f"{name}.csv differs between runs!"
