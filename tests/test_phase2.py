"""
Tests for Phase 2: STL Detection and Causal Attribution.

Validates that:
    1. The STL detector flags the June-July marketing shock
    2. The STL detector flags the September stockout
    3. Multi-grain data alignment produces a clean DataFrame
    4. Causal attribution blames ad_spend for marketing shock events
    5. Causal attribution blames stock_on_hand for stockout events
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.detection.stl_detector import (
    compute_daily_kpi,
    detect_anomalies,
    load_contract,
    load_sales_data,
    run_detection,
)
from src.causal.dowhy_gcm import (
    align_datasets,
    build_causal_dag,
    run_causal_attribution,
    ALL_NODES,
)


# ============================================================
# Shared fixtures
# ============================================================

@pytest.fixture(scope="session")
def detection_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full detection pipeline once per test session."""
    return run_detection()


@pytest.fixture(scope="session")
def anomaly_events(detection_results) -> pd.DataFrame:
    return detection_results[0]


@pytest.fixture(scope="session")
def daily_kpi(detection_results) -> pd.DataFrame:
    return detection_results[1]


@pytest.fixture(scope="session")
def causal_df() -> pd.DataFrame:
    return align_datasets()


@pytest.fixture(scope="session")
def attribution_results(anomaly_events) -> pd.DataFrame:
    return run_causal_attribution(anomaly_events)


# ============================================================
# STL Detector Tests
# ============================================================

class TestSTLDetector:
    """Validate the STL anomaly detection pipeline."""

    def test_anomalies_detected(self, anomaly_events: pd.DataFrame) -> None:
        """At least some anomaly events should be detected."""
        assert len(anomaly_events) > 0, "No anomaly events detected"

    def test_has_required_columns(self, anomaly_events: pd.DataFrame) -> None:
        expected = {
            "date", "region", "net_revenue", "baseline", "trend",
            "seasonal", "residual", "z_score", "pct_deviation", "severity",
        }
        assert expected.issubset(set(anomaly_events.columns))

    def test_marketing_shock_flagged(self, anomaly_events: pd.DataFrame) -> None:
        """
        The June-July marketing shock should produce warning/critical
        anomaly events in the June 15 – August 1 window.
        """
        shock_events = anomaly_events[
            (anomaly_events["date"] >= "2025-06-15") &
            (anomaly_events["date"] <= "2025-08-01") &
            (anomaly_events["severity"].isin(["warning", "critical"]))
        ]
        assert len(shock_events) >= 2, (
            f"Expected >= 2 warning/critical events during marketing shock, "
            f"got {len(shock_events)}"
        )

    def test_stockout_flagged(self, anomaly_events: pd.DataFrame) -> None:
        """
        The September stockout should produce anomaly events for Southeast
        in the Sep 1–14 window.
        """
        stockout_events = anomaly_events[
            (anomaly_events["date"] >= "2025-09-01") &
            (anomaly_events["date"] <= "2025-09-14") &
            (anomaly_events["region"] == "Southeast")
        ]
        assert len(stockout_events) >= 1, (
            f"Expected >= 1 stockout anomaly event for Southeast Sep 1-14, "
            f"got {len(stockout_events)}"
        )

    def test_severity_levels_present(self, anomaly_events: pd.DataFrame) -> None:
        """Both warning and critical severity levels should be present."""
        severities = set(anomaly_events["severity"].unique())
        assert "warning" in severities, "No 'warning' severity events found"
        assert "critical" in severities, "No 'critical' severity events found"

    def test_z_scores_exceed_threshold(self, anomaly_events: pd.DataFrame) -> None:
        """All flagged events should have |Z| > 2.0 (the YAML threshold)."""
        assert (anomaly_events["z_score"].abs() > 2.0).all(), (
            "Some anomaly events have |Z| <= 2.0"
        )


# ============================================================
# Data Alignment Tests
# ============================================================

class TestDataAlignment:
    """Validate the multi-grain data alignment."""

    def test_no_nulls(self, causal_df: pd.DataFrame) -> None:
        """Aligned DataFrame should have no NaN values."""
        for col in ALL_NODES:
            assert causal_df[col].notna().all(), f"NaN found in column '{col}'"

    def test_has_all_columns(self, causal_df: pd.DataFrame) -> None:
        expected = {"date", "region"} | set(ALL_NODES)
        assert expected.issubset(set(causal_df.columns))

    def test_row_count(self, causal_df: pd.DataFrame) -> None:
        """This synthetic dataset includes five regions, with Southwest launching late."""
        assert len(causal_df) == 1521, f"Expected 1521 rows, got {len(causal_df)}"

    def test_stockout_visible(self, causal_df: pd.DataFrame) -> None:
        """
        Southeast stock_on_hand should be 0 during Sep 1-14.
        """
        se_sep = causal_df[
            (causal_df["region"] == "Southeast") &
            (causal_df["date"] >= "2025-09-01") &
            (causal_df["date"] <= "2025-09-14")
        ]
        assert (se_sep["stock_on_hand"] == 0).all(), (
            "stock_on_hand should be 0 during stockout window"
        )

    def test_positive_ad_spend(self, causal_df: pd.DataFrame) -> None:
        """All ad_spend values should be positive."""
        assert (causal_df["ad_spend"] > 0).all()


# ============================================================
# Causal DAG Tests
# ============================================================

class TestCausalDAG:
    """Validate the causal graph structure."""

    def test_is_dag(self) -> None:
        import networkx as nx
        dag = build_causal_dag()
        assert nx.is_directed_acyclic_graph(dag)

    def test_edges(self) -> None:
        dag = build_causal_dag()
        edges = set(dag.edges())
        assert ("ad_spend", "web_traffic") in edges
        assert ("web_traffic", "net_revenue") in edges
        assert ("stock_on_hand", "net_revenue") in edges

    def test_four_nodes(self) -> None:
        dag = build_causal_dag()
        assert dag.number_of_nodes() == 4
        assert dag.number_of_edges() == 3


# ============================================================
# Causal Attribution Tests
# ============================================================

class TestCausalAttribution:
    """Validate causal attribution correctness."""

    def test_has_results(self, attribution_results: pd.DataFrame) -> None:
        assert len(attribution_results) > 0

    def test_has_required_columns(self, attribution_results: pd.DataFrame) -> None:
        expected = {
            "date", "region", "severity", "net_revenue",
            "ad_spend_contribution_pct", "stock_on_hand_contribution_pct",
            "primary_driver", "confidence",
        }
        assert expected.issubset(set(attribution_results.columns))

    def test_contributions_sum_to_100(self, attribution_results: pd.DataFrame) -> None:
        """
        ad_spend + stock_on_hand contributions should sum to ~100%,
        UNLESS the anomaly is purely intrinsic (total contribution = 0%).
        """
        total = (
            attribution_results["ad_spend_contribution_pct"]
            + attribution_results["stock_on_hand_contribution_pct"]
        )
        for idx, val in total.items():
            is_100 = abs(val - 100.0) < 1.0
            is_0 = abs(val - 0.0) < 1.0
            assert is_100 or is_0, (
                f"Row {idx}: contributions sum to {val:.1f}%, expected ~100% or 0%"
            )

    def test_marketing_shock_attributed_to_ad_spend(
        self, attribution_results: pd.DataFrame
    ) -> None:
        """
        During the marketing shock window (Jul), most events should
        attribute primarily to ad_spend.
        """
        shock_events = attribution_results[
            (attribution_results["date"] >= "2025-07-01") &
            (attribution_results["date"] <= "2025-07-31")
        ]
        if len(shock_events) == 0:
            pytest.skip("No attribution events in July")

        ad_spend_pct = shock_events["ad_spend_contribution_pct"].mean()
        assert ad_spend_pct > 60, (
            f"Expected ad_spend > 60% for marketing shock, got {ad_spend_pct:.1f}%"
        )

    def test_stockout_attributed_to_stock(
        self, attribution_results: pd.DataFrame
    ) -> None:
        """
        During the stockout window (Sep 1-14, Southeast), stock_on_hand
        should have a meaningful contribution (>10%).
        """
        stockout_events = attribution_results[
            (attribution_results["date"] >= "2025-09-01") &
            (attribution_results["date"] <= "2025-09-14") &
            (attribution_results["region"] == "Southeast")
        ]
        if len(stockout_events) == 0:
            pytest.skip("No attribution events for Southeast Sep 1-14")

        stock_pct = stockout_events["stock_on_hand_contribution_pct"].mean()
        assert stock_pct > 10, (
            f"Expected stock_on_hand > 10% for stockout, got {stock_pct:.1f}%"
        )
