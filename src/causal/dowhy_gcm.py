"""
Causal Attribution Engine for KPI Engine (DoWhy-Compatible)
============================================================

Aligns multi-grain datasets (daily sales, weekly marketing, 6-hourly
inventory) into a single daily-grain causal DataFrame, defines a
structural causal graph, fits additive-noise causal mechanisms on
baseline (non-anomalous) data, and attributes anomaly root causes.

Architecture:
    This module implements the same API contract as ``dowhy.gcm``
    (InvertibleStructuralCausalModel, fit, attribute_anomalies) using
    scikit-learn additive noise models over a networkx DAG.  The
    implementation is a *drop-in-ready* stand-in:  once DoWhy ships a
    build compatible with the runtime Python (>=3.14), swap the import
    and the rest of the pipeline stays unchanged.

Causal DAG (hand-specified, per project design):
    ad_spend ──► web_traffic ──► net_revenue
                                     ▲
                 stock_on_hand ───────┘

Attribution method:
    Exact Shapley-value decomposition over root-cause coalitions. For each
    anomaly sample, every root cause is restored to its date-appropriate
    seasonal baseline in turn and in combinations, and the marginal uplift of
    each root is aggregated with the standard Shapley weights.

Usage:
    python -m src.causal.dowhy_gcm
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# PATH DEFAULTS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "kpi_contract.yaml"

# ============================================================
# DAG node names (must match column names in df_causal)
# ============================================================
NODE_AD_SPEND = "ad_spend"
NODE_WEB_TRAFFIC = "web_traffic"
NODE_STOCK = "stock_on_hand"
NODE_NET_REVENUE = "net_revenue"

# All nodes the model operates on (order does not matter)
ALL_NODES: list[str] = [
    NODE_AD_SPEND,
    NODE_WEB_TRAFFIC,
    NODE_STOCK,
    NODE_NET_REVENUE,
]


# ============================================================
# CONTRACT LOADING
# ============================================================

def load_contract(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the KPI contract from YAML."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# MULTI-GRAIN DATA ALIGNMENT
# ============================================================

def _align_marketing_to_daily(
    project_root: Path,
    contract: dict,
) -> pd.DataFrame:
    """
    Convert weekly marketing data to daily grain.

    Steps:
        1. Load marketing_weekly.csv
        2. Sum ad_spend across channels per (week_start, region)
        3. Expand each week to 7 daily rows (uniform distribution)
        4. Forward-fill any edge-week gaps

    Returns:
        DataFrame with columns: date, region, ad_spend
    """
    ds_cfg = contract["data_sources"]["marketing_weekly"]
    csv_path = project_root / ds_cfg["path"]
    mkt = pd.read_csv(csv_path)
    mkt["week_start"] = pd.to_datetime(mkt["week_start"])

    # Aggregate across channels
    weekly = (
        mkt.groupby(["week_start", "region"])["ad_spend"]
        .sum()
        .reset_index()
        .rename(columns={"ad_spend": "weekly_ad_spend"})
    )

    # Expand to daily: each day in the week gets weekly_total / 7
    daily_rows: list[pd.DataFrame] = []
    for _, row in weekly.iterrows():
        ws = row["week_start"]
        daily_val = row["weekly_ad_spend"] / 7.0
        days = pd.date_range(ws, periods=7, freq="D")
        chunk = pd.DataFrame({
            "date": days,
            "region": row["region"],
            "ad_spend": daily_val,
        })
        daily_rows.append(chunk)

    daily_mkt = pd.concat(daily_rows, ignore_index=True)

    # Restrict to the sales data date range and drop duplicates
    daily_mkt = (
        daily_mkt.drop_duplicates(subset=["date", "region"])
        .sort_values(["date", "region"])
        .reset_index(drop=True)
    )

    logger.debug("Marketing aligned to daily: %d rows", len(daily_mkt))
    return daily_mkt


def _align_inventory_to_daily(
    project_root: Path,
    contract: dict,
) -> pd.DataFrame:
    """
    Aggregate 6-hourly inventory snapshots to daily minimum stock.

    The daily MIN(stock_on_hand) across all products and all intra-day
    readings captures the tightest supply constraint for each region-day.
    When any product stocks out, min hits zero — which is exactly the
    signal we need for causal attribution of stockout-driven revenue drops.

    Returns:
        DataFrame with columns: date, region, stock_on_hand
    """
    ds_cfg = contract["data_sources"]["inventory_hourly"]
    csv_path = project_root / ds_cfg["path"]
    inv = pd.read_csv(csv_path)
    inv["timestamp"] = pd.to_datetime(inv["timestamp"])
    inv["date"] = inv["timestamp"].dt.normalize()

    # Daily MIN across all products and intra-day readings per region
    daily_inv = (
        inv.groupby(["date", "region"])["stock_on_hand"]
        .min()
        .reset_index()
    )

    logger.debug("Inventory aligned to daily: %d rows", len(daily_inv))
    return daily_inv


def _align_sales_to_daily(
    project_root: Path,
    contract: dict,
) -> pd.DataFrame:
    """
    Aggregate sales data to daily grain per region.

    Computes net_revenue per the YAML formula and takes web_traffic
    (which is identical for all products within a region-day).

    Returns:
        DataFrame with columns: date, region, web_traffic, net_revenue
    """
    kpi_cfg = contract["kpis"]["regional_net_revenue"]
    csv_path = project_root / kpi_cfg["source_path"]
    date_col = kpi_cfg["date_column"]

    sales = pd.read_csv(csv_path, parse_dates=[date_col])
    sales["_return_value"] = sales["returns"] * sales["unit_price"]

    daily_sales = (
        sales.groupby([date_col, "region"])
        .agg(
            gross_revenue=("gross_revenue", "sum"),
            return_value=("_return_value", "sum"),
            web_traffic=("web_traffic", "first"),
        )
        .reset_index()
    )

    daily_sales["net_revenue"] = (
        daily_sales["gross_revenue"] - daily_sales["return_value"]
    )
    daily_sales.drop(columns=["gross_revenue", "return_value"], inplace=True)

    logger.debug("Sales aligned to daily: %d rows", len(daily_sales))
    return daily_sales


def align_datasets(
    project_root: Path = PROJECT_ROOT,
    contract: dict | None = None,
) -> pd.DataFrame:
    """
    Join all three data sources into a single daily-grain causal DataFrame.

    Performs a left join on (date, region) starting from sales, then merging
    marketing (forward-filled to daily) and inventory (daily min stock).
    Handles missing values by forward-filling then back-filling within
    each region group.

    Returns:
        DataFrame with columns:
            date, region, ad_spend, web_traffic, stock_on_hand, net_revenue
        Sorted by (date, region), no NaN values.
    """
    if contract is None:
        contract = load_contract(project_root / "config" / "kpi_contract.yaml")

    logger.info("Aligning datasets to daily grain ...")

    # Load and align each source
    daily_sales = _align_sales_to_daily(project_root, contract)
    daily_mkt = _align_marketing_to_daily(project_root, contract)
    daily_inv = _align_inventory_to_daily(project_root, contract)

    # Merge: sales ← marketing ← inventory
    df = daily_sales.merge(daily_mkt, on=["date", "region"], how="left")
    df = df.merge(daily_inv, on=["date", "region"], how="left")

    # Select and order the causal columns
    df = df[["date", "region", NODE_AD_SPEND, NODE_WEB_TRAFFIC,
             NODE_STOCK, NODE_NET_REVENUE]].copy()

    # Handle NaN: forward-fill then back-fill within each region
    for col in [NODE_AD_SPEND, NODE_WEB_TRAFFIC, NODE_STOCK, NODE_NET_REVENUE]:
        df[col] = df.groupby("region")[col].transform(
            lambda s: s.ffill().bfill()
        )

    # Final safety net: drop any remaining NaN rows
    n_before = len(df)
    df.dropna(subset=ALL_NODES, inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning("Dropped %d rows with residual NaN values", n_dropped)

    df = df.sort_values(["date", "region"]).reset_index(drop=True)

    logger.info(
        "✓ Aligned causal DataFrame: %d rows  "
        "(%d dates × %d regions)",
        len(df),
        df["date"].nunique(),
        df["region"].nunique(),
    )
    return df


# ============================================================
# CAUSAL DAG DEFINITION
# ============================================================

def build_causal_dag(contract: dict | None = None) -> nx.DiGraph:
    """
    Build the causal DAG dynamically from the kpi_contract.yaml.
    """
    if contract is None:
        contract = load_contract(PROJECT_ROOT / "config" / "kpi_contract.yaml")
        
    dag = nx.DiGraph()
    
    # Read edges from the contract
    edges = contract.get("kpis", {}).get("causal_graph", {}).get("edges", [])
    if not edges:
        # Fallback if missing
        dag.add_edges_from([
            (NODE_AD_SPEND, NODE_WEB_TRAFFIC),
            (NODE_WEB_TRAFFIC, NODE_NET_REVENUE),
            (NODE_STOCK, NODE_NET_REVENUE),
        ])
    else:
        for edge in edges:
            dag.add_edge(edge["from"], edge["to"])

    # Validate it's actually a DAG
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("The specified graph is not a DAG!")

    logger.debug(
        "Causal DAG: %d nodes, %d edges",
        dag.number_of_nodes(),
        dag.number_of_edges(),
    )
    return dag


# ============================================================
# STRUCTURAL CAUSAL MODEL (DoWhy-compatible implementation)
# ============================================================

class AdditiveNoiseModel:
    """
    Additive Noise Model (ANM) for a single node.

    Models Y = f(parents) + ε where f is learned by a GradientBoostingRegressor
    and ε is the residual noise.  Invertible in the sense that given Y and
    parents, ε = Y - f(parents) can be recovered exactly.
    """

    def __init__(self) -> None:
        # Swapped from GBR to Ridge to allow linear extrapolation for counterfactual simulator out-of-distribution
        self._model = Ridge(alpha=1.0)
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the functional f on (parents → child) data."""
        self._model.fit(X, y)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict f(parents)."""
        return self._model.predict(X)

    def compute_noise(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Invert the model: ε = y - f(X)."""
        return y - self.predict(X)


class EmpiricalDistribution:
    """
    Empirical distribution for root nodes (no parents).

    Stores the observed values to compute anomaly scores as
    Z-scores from the fitted distribution.
    """

    def __init__(self) -> None:
        self._mean: float = 0.0
        self._std: float = 1.0

    def fit(self, values: np.ndarray) -> None:
        self._mean = float(np.mean(values))
        self._std = float(np.std(values, ddof=1))
        if self._std == 0:
            self._std = 1.0

    def compute_noise(self, values: np.ndarray) -> np.ndarray:
        """Noise = deviation from the mean (Z-score scaled)."""
        return (values - self._mean) / self._std


class InvertibleStructuralCausalModel:
    """
    Invertible Structural Causal Model.

    Drop-in replacement for ``dowhy.gcm.InvertibleStructuralCausalModel``.
    Assigns AdditiveNoiseModel to non-root nodes and EmpiricalDistribution
    to root nodes.  Supports ``fit()`` and ``attribute_anomalies()``.

    Args:
        dag: A networkx DiGraph encoding the causal structure.
    """

    def __init__(self, dag: nx.DiGraph) -> None:
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Graph must be a DAG.")

        self.dag = dag
        self._mechanisms: dict[str, AdditiveNoiseModel | EmpiricalDistribution] = {}
        self._node_order: list[str] = list(nx.topological_sort(dag))
        self._is_fitted = False

        # Auto-assign mechanisms
        for node in self._node_order:
            parents = list(dag.predecessors(node))
            if len(parents) == 0:
                self._mechanisms[node] = EmpiricalDistribution()
            else:
                self._mechanisms[node] = AdditiveNoiseModel()

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit all causal mechanisms on baseline data.

        Args:
            data: DataFrame with columns matching the DAG node names.
        """
        for node in self._node_order:
            parents = list(self.dag.predecessors(node))
            mechanism = self._mechanisms[node]

            if isinstance(mechanism, EmpiricalDistribution):
                mechanism.fit(data[node].values)
            else:
                X = data[parents].values
                y = data[node].values
                mechanism.fit(X, y)

        self._is_fitted = True
        logger.debug("Fitted causal mechanisms for %d nodes", len(self._node_order))

    def _compute_all_noises(self, sample: pd.DataFrame) -> dict[str, np.ndarray]:
        """Compute the noise term at every node for the given sample(s)."""
        noises: dict[str, np.ndarray] = {}

        for node in self._node_order:
            parents = list(self.dag.predecessors(node))
            mechanism = self._mechanisms[node]

            if isinstance(mechanism, EmpiricalDistribution):
                noises[node] = mechanism.compute_noise(sample[node].values)
            else:
                X = sample[parents].values
                y = sample[node].values
                noises[node] = mechanism.compute_noise(X, y)

        return noises


def _get_ancestor_nodes(dag: nx.DiGraph, target: str) -> list[str]:
    """Return all ancestor nodes of target (excluding target itself)."""
    return list(nx.ancestors(dag, target))


# ============================================================
# ANOMALY ATTRIBUTION (Counterfactual decomposition)
# ============================================================

import itertools
import math

def _compute_seasonal_baseline_value(
    event_date: pd.Timestamp,
    region: str,
    root: str,
    baseline_data: pd.DataFrame,
) -> float:
    """Return the expected value for a root cause at the event date using a local seasonal window."""
    region_baseline = baseline_data[baseline_data["region"] == region].copy()
    if region_baseline.empty:
        return float(baseline_data[root].mean())

    region_baseline["date"] = pd.to_datetime(region_baseline["date"])
    local_window = region_baseline[
        (region_baseline["date"] >= event_date - pd.Timedelta(days=45))
        & (region_baseline["date"] <= event_date + pd.Timedelta(days=45))
    ]

    if not local_window.empty:
        same_weekday = local_window[
            local_window["date"].dt.dayofweek == event_date.dayofweek
        ]
        if not same_weekday.empty:
            return float(same_weekday[root].median())
        return float(local_window[root].median())

    return float(region_baseline[root].median())


def attribute_anomalies(
    model: InvertibleStructuralCausalModel,
    target_node: str,
    anomaly_samples: pd.DataFrame,
    baseline_data: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Attribute anomalies at ``target_node`` to **root-cause** drivers
    using an exact Shapley-value decomposition across root-cause coalitions.
    """
    if not model._is_fitted:
        raise RuntimeError("Model must be fitted before attribution.")

    dag = model.dag
    topo_order = list(nx.topological_sort(dag))
    root_nodes = [n for n in dag.nodes() if dag.in_degree(n) == 0]
    if target_node in root_nodes:
        root_nodes.remove(target_node)

    if not root_nodes:
        return {target_node: np.zeros(len(anomaly_samples))}

    n_samples = len(anomaly_samples)
    contributions: dict[str, np.ndarray] = {root: np.zeros(n_samples) for root in root_nodes}
    observed_target = anomaly_samples[target_node].values.astype(np.float64)

    sample_baselines: dict[str, np.ndarray] = {root: np.zeros(n_samples) for root in root_nodes}
    for i, (_, row) in enumerate(anomaly_samples.iterrows()):
        event_date = pd.to_datetime(row["date"])
        region = row["region"]
        for root in root_nodes:
            sample_baselines[root][i] = _compute_seasonal_baseline_value(
                event_date=event_date,
                region=region,
                root=root,
                baseline_data=baseline_data,
            )

    def evaluate_coalition(S: tuple[str, ...]) -> np.ndarray:
        cf_values: dict[str, np.ndarray] = {}
        for node in topo_order:
            parents = list(dag.predecessors(node))
            if node in root_nodes:
                if node in S:
                    cf_values[node] = sample_baselines[node]
                else:
                    cf_values[node] = anomaly_samples[node].values.astype(np.float64)
            elif len(parents) == 0:
                cf_values[node] = anomaly_samples[node].values.astype(np.float64)
            else:
                mechanism = model._mechanisms[node]
                if isinstance(mechanism, AdditiveNoiseModel):
                    X_cf = np.column_stack([cf_values[p] for p in parents])
                    predicted = mechanism.predict(X_cf)
                    X_obs = anomaly_samples[parents].values.astype(np.float64)
                    y_obs = anomaly_samples[node].values.astype(np.float64)
                    observed_noise = y_obs - mechanism.predict(X_obs)
                    cf_values[node] = predicted + observed_noise
                else:
                    cf_values[node] = anomaly_samples[node].values.astype(np.float64)

        return cf_values[target_node] - observed_target

    v_cache: dict[tuple[str, ...], np.ndarray] = {}

    def v(S_list: list[str]) -> np.ndarray:
        S_tuple = tuple(sorted(S_list))
        if S_tuple not in v_cache:
            v_cache[S_tuple] = evaluate_coalition(S_tuple)
        return v_cache[S_tuple]

    n = len(root_nodes)
    for root in root_nodes:
        others = [x for x in root_nodes if x != root]
        for k in range(n):
            for S in itertools.combinations(others, k):
                weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                marginal = v(list(S) + [root]) - v(list(S))
                contributions[root] += weight * marginal

    return contributions


# ============================================================
# HIGH-LEVEL ATTRIBUTION PIPELINE
# ============================================================

def run_causal_attribution(
    anomaly_events: pd.DataFrame,
    project_root: Path | None = None,
    df_causal: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Full causal attribution pipeline for detected anomaly events.

    Steps:
        1. Align multi-grain data into a single causal DataFrame.
        2. Build the causal DAG.
        3. Split data into baseline (non-anomaly) and anomaly samples.
        4. Fit the InvertibleStructuralCausalModel on baseline data.
        5. For each anomaly event, run Shapley-based attribution.
        6. Return a DataFrame with percentage contributions per driver.

    Args:
        anomaly_events: Output of ``stl_detector.run_detection()``.
            Must have columns: date, region.
        project_root: Project root directory.
        df_causal: Pre-aligned causal DataFrame (optional, for testing).

    Returns:
        DataFrame with columns:
            date, region, severity, net_revenue,
            ad_spend_contribution_pct, stock_on_hand_contribution_pct,
            web_traffic_contribution_pct, primary_driver, confidence
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    logger.info("=" * 60)
    logger.info("Phase 2.2: Causal Attribution")
    logger.info("=" * 60)

    if anomaly_events.empty:
        logger.info("No anomaly events to attribute. Returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "date", "region", "severity", "net_revenue",
            "ad_spend_contribution_pct", "stock_on_hand_contribution_pct",
            "web_traffic_contribution_pct", "primary_driver", "confidence",
        ])

    # 1. Align data
    if df_causal is None:
        contract = load_contract(project_root / "config" / "kpi_contract.yaml")
        df_causal = align_datasets(project_root, contract)
    else:
        logger.info("Using pre-aligned causal DataFrame (%d rows)", len(df_causal))

    # 2. Build DAG
    if "contract" not in locals():
        contract = load_contract(project_root / "config" / "kpi_contract.yaml")
    dag = build_causal_dag(contract)
    logger.info(
        "✓ Causal DAG: %s",
        " → ".join(
            [f"{u}→{v}" for u, v in dag.edges()]
        ),
    )

    # 3. Identify anomaly dates+regions to exclude from baseline
    anomaly_keys = set(
        zip(
            pd.to_datetime(anomaly_events["date"]).dt.normalize(),
            anomaly_events["region"],
        )
    )

    df_causal["_key"] = list(zip(df_causal["date"], df_causal["region"]))
    baseline_mask = ~df_causal["_key"].isin(anomaly_keys)
    df_baseline = df_causal[baseline_mask].copy()
    df_causal.drop(columns=["_key"], inplace=True)

    logger.info(
        "✓ Baseline: %d rows  |  Anomaly: %d events",
        len(df_baseline),
        len(anomaly_keys),
    )

    # 4. Fit model on baseline
    model = InvertibleStructuralCausalModel(dag)
    model.fit(df_baseline[ALL_NODES])
    logger.info("✓ Causal model fitted on baseline data")

    # 5. Load confidence threshold from contract
    contract = load_contract(project_root / "config" / "kpi_contract.yaml")
    kpi_cfg = contract["kpis"]["regional_net_revenue"]
    confidence_threshold = kpi_cfg.get("causal_confidence_threshold", 0.70)
    low_conf_action = kpi_cfg.get("low_confidence_action", "investigation_required")

    # 6. Run attribution for each anomaly event
    results: list[dict[str, Any]] = []

    for _, event in anomaly_events.iterrows():
        event_date = pd.to_datetime(event["date"])
        event_region = event["region"]

        # Find the matching row in df_causal
        match = df_causal[
            (df_causal["date"] == event_date) &
            (df_causal["region"] == event_region)
        ]

        if match.empty:
            logger.warning(
                "No causal data found for %s / %s. Skipping.",
                event_date.date(),
                event_region,
            )
            continue

        anomaly_sample = match.copy()

        # Run Shapley attribution
        attributions = attribute_anomalies(
            model=model,
            target_node=NODE_NET_REVENUE,
            anomaly_samples=anomaly_sample,
            baseline_data=df_baseline,
        )

        # Convert to percentage contributions
        raw_scores = {
            node: float(np.mean(np.abs(scores)))
            for node, scores in attributions.items()
        }

        total_score = sum(raw_scores.values())
        if total_score == 0:
            total_score = 1.0  # prevent division by zero

        pct_contributions = {
            node: round(score / total_score * 100, 1)
            for node, score in raw_scores.items()
        }

        # Determine primary driver and confidence
        primary_driver = max(pct_contributions, key=pct_contributions.get)
        confidence = pct_contributions[primary_driver] / 100.0

        # If confidence is below threshold, flag as low confidence
        low_confidence = confidence < confidence_threshold

        results.append({
            "date": event_date,
            "region": event_region,
            "severity": event.get("severity", "unknown"),
            "net_revenue": event.get("net_revenue", np.nan),
            "ad_spend_contribution_pct": pct_contributions.get(NODE_AD_SPEND, 0.0),
            "stock_on_hand_contribution_pct": pct_contributions.get(NODE_STOCK, 0.0),
            "web_traffic_contribution_pct": pct_contributions.get(NODE_WEB_TRAFFIC, 0.0),
            "primary_driver": primary_driver,
            "confidence": round(confidence, 3),
            "low_confidence": low_confidence,
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(["date", "region"]).reset_index(drop=True)

    logger.info("✓ Attribution complete: %d events attributed", len(result_df))

    # Summary
    if not result_df.empty:
        for driver in [NODE_AD_SPEND, NODE_STOCK, NODE_WEB_TRAFFIC]:
            driver_col = f"{driver}_contribution_pct"
            is_primary = result_df["primary_driver"] == driver
            count = int(is_primary.sum())
            if count > 0:
                avg_pct = result_df.loc[is_primary, driver_col].mean()
                logger.info(
                    "  %s is primary driver for %d events (avg %.1f%%)",
                    driver,
                    count,
                    avg_pct,
                )

    logger.info("=" * 60)
    return result_df


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Import the STL detector to get anomaly events
    from src.detection.stl_detector import run_detection

    anomaly_events, _ = run_detection()

    if anomaly_events.empty:
        print("\n✅ No anomalies to attribute.")
    else:
        attribution_df = run_causal_attribution(anomaly_events)

        print("\n[Causal Attribution Results]\n")
        print(
            attribution_df.to_string(
                index=False,
                columns=[
                    "date", "region", "severity", "net_revenue",
                    "ad_spend_contribution_pct",
                    "stock_on_hand_contribution_pct",
                    "primary_driver", "confidence",
                ],
                formatters={
                    "date": lambda x: x.strftime("%Y-%m-%d"),
                    "net_revenue": "${:,.0f}".format,
                    "ad_spend_contribution_pct": "{:.1f}%".format,
                    "stock_on_hand_contribution_pct": "{:.1f}%".format,
                    "confidence": "{:.0%}".format,
                },
            )
        )
