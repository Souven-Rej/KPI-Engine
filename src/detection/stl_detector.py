"""
STL-Based Anomaly Detector for KPI Engine
==========================================

Reads the KPI contract from YAML, loads sales data, decomposes daily
net revenue per region using STL, and flags anomaly events where the
Z-score of the residuals exceeds the YAML-configured threshold.

Pipeline:
    1. Load contract → resolve data path, STL config, thresholds
    2. Load sales_daily.csv → parse dates
    3. Compute KPI per YAML formula: SUM(gross_revenue) - SUM(returns * unit_price)
       aggregated by (date, region)
    4. For each region, run STL(period=7, seasonal=13, robust=True)
    5. Compute Z-score of residuals
    6. Flag dates where |Z| > z_score_threshold
    7. Classify severity using fraction-deviation thresholds

Expected detections on synthetic data:
    - June-July 2025: Marketing shock → revenue decline across all regions
    - September 1-14 2025: Stockout shock → Southeast revenue decline

Usage:
    python -m src.detection.stl_detector
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.seasonal import STL

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
# KPI key to load from the contract
# ============================================================
KPI_KEY = "regional_net_revenue"


# ============================================================
# CONTRACT LOADING
# ============================================================

def load_contract(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """
    Load and return the full KPI contract dictionary from YAML.

    Args:
        config_path: Absolute path to kpi_contract.yaml.

    Returns:
        Parsed YAML as a nested dict.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"KPI contract not found at {config_path}")

    with open(config_path, encoding="utf-8") as f:
        contract = yaml.safe_load(f)

    logger.debug("Loaded KPI contract from %s", config_path)
    return contract


def _get_kpi_config(contract: dict) -> dict:
    """Extract the regional_net_revenue KPI definition from the contract."""
    try:
        return contract["kpis"][KPI_KEY]
    except KeyError:
        raise KeyError(
            f"KPI '{KPI_KEY}' not found in contract. "
            f"Available KPIs: {list(contract.get('kpis', {}).keys())}"
        )


# ============================================================
# DATA INGESTION
# ============================================================

def load_sales_data(
    contract: dict,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """
    Load sales data from the path specified in the KPI contract.

    Reads `source_path` and `date_column` from the contract, resolves
    the CSV path relative to project_root, and parses the date column.

    Returns:
        DataFrame with parsed date column and all original columns.
    """
    kpi_cfg = _get_kpi_config(contract)
    rel_path = kpi_cfg["source_path"]
    date_col = kpi_cfg["date_column"]

    csv_path = project_root / rel_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Sales data not found at {csv_path}. "
            f"Run `python -m src.data_generation.enterprise_warehouse_etl` first."
        )

    df = pd.read_csv(csv_path, parse_dates=[date_col])
    logger.debug("Loaded %d rows from %s", len(df), csv_path)
    return df


# ============================================================
# KPI COMPUTATION
# ============================================================

def compute_daily_kpi(
    sales_df: pd.DataFrame,
    contract: dict,
) -> pd.DataFrame:
    """
    Compute the KPI per the YAML formula, aggregated by (date, region).

    YAML formula: SUM(gross_revenue) - SUM(returns * unit_price)

    This re-derives net_revenue from the raw components to honour the
    contract definition exactly, rather than trusting the pre-computed column.

    Returns:
        DataFrame with columns: date, region, net_revenue, web_traffic
        sorted by (date, region).
    """
    kpi_cfg = _get_kpi_config(contract)
    date_col = kpi_cfg["date_column"]

    df = sales_df.copy()

    # Per-row return value (the YAML formula's inner expression)
    df["_return_value"] = df["returns"] * df["unit_price"]

    # Aggregate by (date, region) — SUM across products per the YAML formula
    agg_df = (
        df.groupby([date_col, "region"])
        .agg(
            gross_revenue_sum=("gross_revenue", "sum"),
            return_value_sum=("_return_value", "sum"),
            # web_traffic is identical for all products within a region-day
            # (set at the region level in the generator), so take the first.
            web_traffic=("web_traffic", "first"),
        )
        .reset_index()
    )

    # Apply the YAML formula: SUM(gross_revenue) - SUM(returns * unit_price)
    agg_df["net_revenue"] = agg_df["gross_revenue_sum"] - agg_df["return_value_sum"]
    agg_df.drop(columns=["gross_revenue_sum", "return_value_sum"], inplace=True)

    agg_df = agg_df.sort_values([date_col, "region"]).reset_index(drop=True)
    logger.debug(
        "Computed daily KPI: %d region-days, %d unique dates, %d regions",
        len(agg_df),
        agg_df[date_col].nunique(),
        agg_df["region"].nunique(),
    )
    return agg_df


# ============================================================
# STL ANOMALY DETECTION
# ============================================================

def detect_anomalies(
    kpi_df: pd.DataFrame,
    contract: dict,
) -> pd.DataFrame:
    """
    Run STL decomposition per region and flag anomaly events via Z-score.

    For each region:
        1. Extract the daily net_revenue time series.
        2. Run STL with parameters from the YAML contract.
        3. Compute Z-score of the STL residuals.
        4. Flag dates where |Z| > z_score_threshold (from YAML).
        5. Classify severity using the fraction-deviation thresholds.

    The fraction-deviation thresholds (warning/critical) measure how far
    the actual revenue deviates from the STL baseline (trend + seasonal)
    as a fraction: (actual - baseline) / baseline.

    Args:
        kpi_df: Output of compute_daily_kpi().
        contract: Full KPI contract dict.

    Returns:
        DataFrame of anomaly events with columns:
            date, region, net_revenue, baseline, trend, seasonal,
            residual, z_score, pct_deviation, severity
        Empty DataFrame (with correct columns) if no anomalies found.
    """
    kpi_cfg = _get_kpi_config(contract)
    date_col = kpi_cfg["date_column"]
    stl_config = kpi_cfg["stl_config"]
    thresholds = kpi_cfg["thresholds"]
    z_threshold = kpi_cfg.get("z_score_threshold", 2.0)
    min_history = kpi_cfg.get("minimum_history_days", 30)

    period = stl_config["period"]
    seasonal = stl_config["seasonal"]
    robust = stl_config.get("robust", True)

    anomaly_events: list[dict] = []
    regions = sorted(kpi_df["region"].unique())
    enriched_regions: list[pd.DataFrame] = []

    for region in regions:
        region_df = (
            kpi_df[kpi_df["region"] == region]
            .sort_values(date_col)
            .reset_index(drop=True)
        )

        # ── Sparse-history guard ──────────────────────────────────
        if len(region_df) < min_history:
            logger.warning(
                "Region '%s' has only %d days (min=%d). Skipping STL.",
                region,
                len(region_df),
                min_history,
            )
            # Just append the raw df with NaNs for STL components
            for col in ["baseline", "trend", "seasonal", "residual", "z_score", "pct_deviation"]:
                region_df[col] = np.nan
            enriched_regions.append(region_df)
            continue

        # ── STL decomposition ─────────────────────────────────────
        # Kaggle datasets often have missing days, resample to guarantee daily frequency for STL
        ts = region_df.set_index(date_col)["net_revenue"].copy()
        ts = ts.asfreq("D").ffill().fillna(0)

        stl = STL(
            ts,
            period=period,
            seasonal=seasonal,
            robust=robust,
        )
        result = stl.fit()

        trend = result.trend
        seasonal_comp = result.seasonal
        residual = result.resid

        # ── Z-score computation ───────────────────────────────────
        resid_mean = residual.mean()
        resid_std = residual.std(ddof=1)

        if resid_std == 0 or np.isnan(resid_std):
            logger.warning(
                "Region '%s': residual std is zero/NaN. Skipping.", region
            )
            for col in ["baseline", "trend", "seasonal", "residual", "z_score", "pct_deviation"]:
                region_df[col] = np.nan
            enriched_regions.append(region_df)
            continue

        z_scores = (residual - resid_mean) / resid_std

        # ── Baseline & percentage deviation ───────────────────────
        baseline = trend + seasonal_comp
        # Guard against division by zero in baseline
        safe_baseline = baseline.replace(0, np.nan)
        pct_deviation = (ts - baseline) / safe_baseline
        
        # Attach back to the region dataframe
        region_df["baseline"] = baseline.values
        region_df["trend"] = trend.values
        region_df["seasonal"] = seasonal_comp.values
        region_df["residual"] = residual.values
        region_df["z_score"] = z_scores.values
        region_df["pct_deviation"] = pct_deviation.values
        
        enriched_regions.append(region_df)

        # ── Flag anomalies ────────────────────────────────────────
        anomaly_mask = z_scores.abs() > z_threshold

        for date_idx in z_scores[anomaly_mask].index:
            z = float(z_scores[date_idx])
            pct_dev = float(pct_deviation[date_idx])

            # Classify severity using the fraction thresholds
            if pct_dev <= thresholds["critical"]:
                severity = "critical"
            elif pct_dev <= thresholds["warning"]:
                severity = "warning"
            else:
                severity = "info"

            anomaly_events.append(
                {
                    "date": date_idx,
                    "region": region,
                    "net_revenue": round(float(ts[date_idx]), 2),
                    "baseline": round(float(baseline[date_idx]), 2),
                    "trend": round(float(trend[date_idx]), 2),
                    "seasonal": round(float(seasonal_comp[date_idx]), 2),
                    "residual": round(float(residual[date_idx]), 2),
                    "z_score": round(z, 3),
                    "pct_deviation": round(pct_dev, 4),
                    "severity": severity,
                }
            )

    enriched_kpi_df = pd.concat(enriched_regions, ignore_index=True)

    # Build result DataFrame
    result_columns = [
        "date", "region", "net_revenue", "baseline", "trend", "seasonal",
        "residual", "z_score", "pct_deviation", "severity",
    ]

    if not anomaly_events:
        return pd.DataFrame(columns=result_columns), enriched_kpi_df

    anomaly_df = pd.DataFrame(anomaly_events)
    anomaly_df = anomaly_df.sort_values(["date", "region"]).reset_index(drop=True)

    return anomaly_df, enriched_kpi_df


# ============================================================
# ORCHESTRATOR
# ============================================================

def run_detection(
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate the full STL anomaly detection pipeline.

    Steps:
        1. Load KPI contract from YAML.
        2. Load and aggregate sales data per the contract formula.
        3. Run STL decomposition and flag anomalies.

    Args:
        project_root: Root directory of the project.  Defaults to
            auto-detected PROJECT_ROOT.

    Returns:
        Tuple of (anomaly_events_df, kpi_df).
        - anomaly_events_df: Flagged anomaly events.
        - kpi_df: The full daily KPI DataFrame (useful for downstream).
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    config_path = project_root / "config" / "kpi_contract.yaml"

    logger.info("=" * 60)
    logger.info("Phase 2.1: STL Anomaly Detection")
    logger.info("=" * 60)

    # 1. Load contract
    contract = load_contract(config_path)
    kpi_cfg = _get_kpi_config(contract)
    logger.info(
        "✓ Loaded KPI contract — KPI: '%s', Z-threshold: %.1f",
        kpi_cfg["display_name"],
        kpi_cfg.get("z_score_threshold", 2.0),
    )

    # 2. Load and compute KPI
    sales_df = load_sales_data(contract, project_root)
    logger.info("✓ Loaded sales data: %s rows", f"{len(sales_df):,}")

    kpi_df = compute_daily_kpi(sales_df, contract)
    logger.info("✓ Computed daily KPI: %s region-days", f"{len(kpi_df):,}")

    # 3. Detect anomalies
    anomaly_df, kpi_df = detect_anomalies(kpi_df, contract)

    logger.info("✓ Detection complete: %d anomaly events flagged", len(anomaly_df))

    if not anomaly_df.empty:
        for severity in ["critical", "warning", "info"]:
            count = int((anomaly_df["severity"] == severity).sum())
            if count > 0:
                logger.info("    %s: %d events", severity.upper(), count)

        logger.info(
            "  Date range: %s — %s",
            anomaly_df["date"].min().strftime("%Y-%m-%d"),
            anomaly_df["date"].max().strftime("%Y-%m-%d"),
        )

        # Quick summary of worst anomaly per region
        worst = anomaly_df.loc[anomaly_df.groupby("region")["z_score"].idxmin()]
        for _, row in worst.iterrows():
            logger.info(
                "  Worst in %s: %s  Z=%.2f  (%.1f%% deviation)",
                row["region"],
                row["date"].strftime("%Y-%m-%d"),
                row["z_score"],
                row["pct_deviation"] * 100,
            )

    logger.info("=" * 60)
    return anomaly_df, kpi_df


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    anomalies, daily_kpi = run_detection()

    if not anomalies.empty:
        print("\n[Anomaly Events Detected]\n")
        print(
            anomalies.to_string(
                index=False,
                columns=[
                    "date", "region", "net_revenue", "baseline",
                    "z_score", "pct_deviation", "severity",
                ],
                formatters={
                    "date": lambda x: x.strftime("%Y-%m-%d"),
                    "net_revenue": "${:,.0f}".format,
                    "baseline": "${:,.0f}".format,
                    "z_score": "{:+.2f}".format,
                    "pct_deviation": "{:+.1%}".format,
                },
            )
        )
    else:
        print("\n✅ No anomaly events detected.")
