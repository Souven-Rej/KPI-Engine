"""
Synthetic Data Generator for KPI Engine
========================================

Generates three multi-grain CSV datasets with injected causal shocks
for ground-truth anomaly detection testing.

Causal Chain (ground truth):
    Marketing Spend ──(7-day lag)──► Web Traffic ──► Units Sold ──► Revenue
                                                          ▲
                                                    Inventory (stockout blocks sales)

Injected Shocks:
    1. Marketing Shock (Jun 15 – Jul 15): 60% spend cut across all regions
       → ~40% traffic decline (lagged 7 days) → ~35% revenue decline
    2. Stockout Shock (Sep 1–14, Southeast, Widget_A): Zero inventory
       → Revenue drops despite normal marketing — tests multi-factor root cause
    3. Sparse History (Widget_C, Dec 1+): Only 31 days of data
       → System must output bounded estimates, not definitive claims

Reproducibility:
    MASTER_SEED = 42 is enforced at the top of this module.
    All random operations use a seeded numpy RandomState.
    Running this script N times produces identical output every time.

Usage:
    python -m src.data_generation.synthetic_generator
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# REPRODUCIBILITY — Strict seed control
# ============================================================
MASTER_SEED: int = 42

# Create a dedicated RandomState so we never pollute the global state
# but still get 100% reproducible output.
_rng = np.random.RandomState(MASTER_SEED)

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION — Dates
# ============================================================
START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2025-12-31")

# ============================================================
# CONFIGURATION — Dimensions
# ============================================================
REGIONS: list[str] = ["Northeast", "Southeast", "Midwest", "West"]
PRODUCTS: list[str] = ["Widget_A", "Widget_B", "Widget_C"]
CHANNELS: list[str] = ["search", "social", "email"]

# Region demand/spend multipliers (Northeast is the largest market)
REGION_MULTIPLIERS: dict[str, float] = {
    "Northeast": 1.20,
    "Southeast": 1.00,
    "Midwest": 0.80,
    "West": 1.10,
}

# ============================================================
# CONFIGURATION — Causal Shocks
# ============================================================
# Shock 1: Marketing budget cut
MARKETING_SHOCK_START = pd.Timestamp("2025-06-15")
MARKETING_SHOCK_END = pd.Timestamp("2025-07-15")
MARKETING_SHOCK_MAGNITUDE: float = 0.60  # 60% reduction in spend

# Shock 2: Stockout event
STOCKOUT_START = pd.Timestamp("2025-09-01")
STOCKOUT_END = pd.Timestamp("2025-09-14")
STOCKOUT_REGION: str = "Southeast"
STOCKOUT_PRODUCT: str = "Widget_A"

# Shock 3: Sparse-history product
SPARSE_PRODUCT: str = "Widget_C"
SPARSE_LAUNCH_DATE = pd.Timestamp("2025-12-01")

# ============================================================
# CONFIGURATION — Marketing parameters
# ============================================================
# Base weekly spend per channel (USD)
BASE_WEEKLY_SPEND: dict[str, float] = {
    "search": 6_000.0,
    "social": 4_000.0,
    "email": 1_500.0,
}

# Channel-specific click-through rates
CTR_BY_CHANNEL: dict[str, float] = {
    "search": 0.035,  # 3.5%
    "social": 0.012,  # 1.2%
    "email": 0.045,   # 4.5%
}

# Impressions per dollar of ad spend (varies by channel)
IMPRESSIONS_PER_DOLLAR: dict[str, float] = {
    "search": 8.0,
    "social": 15.0,
    "email": 5.0,
}

# ============================================================
# CONFIGURATION — Sales parameters
# ============================================================
# Daily organic web traffic base per region (before marketing influence)
# Kept moderate so marketing is the dominant traffic driver (~65-70% of total).
# This ensures the causal chain Marketing → Traffic → Revenue is detectable.
ORGANIC_TRAFFIC_BASE: float = 1_800.0

# How much each dollar of daily marketing spend drives additional traffic
MARKETING_TRAFFIC_SENSITIVITY: float = 2.5

# Traffic → sales conversion rate
CONVERSION_RATE: float = 0.05

# Product share of converted traffic
PRODUCT_TRAFFIC_SHARE: dict[str, float] = {
    "Widget_A": 0.50,
    "Widget_B": 0.35,
    "Widget_C": 0.15,
}

# Base unit prices
UNIT_PRICES: dict[str, float] = {
    "Widget_A": 29.99,
    "Widget_B": 49.99,
    "Widget_C": 19.99,
}

# Day-of-week demand multiplier (Mon=0 … Sun=6)
DOW_MULTIPLIERS: dict[int, float] = {
    0: 0.95,  # Monday
    1: 1.00,  # Tuesday
    2: 1.00,  # Wednesday
    3: 1.05,  # Thursday
    4: 1.10,  # Friday
    5: 1.25,  # Saturday
    6: 0.65,  # Sunday
}

# Return rate (fraction of units sold)
RETURN_RATE: float = 0.03

# Promotion probability per product-day (discount applied)
PROMO_PROBABILITY: float = 0.10
PROMO_DISCOUNT_RANGE: tuple[float, float] = (0.05, 0.15)  # 5–15% off

# Traffic lag: how many days for marketing changes to fully affect traffic
TRAFFIC_LAG_DAYS: int = 7
TRAFFIC_EMA_ALPHA: float = 2.0 / (TRAFFIC_LAG_DAYS + 1)  # ~0.25

# ============================================================
# CONFIGURATION — Inventory parameters
# ============================================================
INVENTORY_SAMPLE_HOURS: list[int] = [0, 6, 12, 18]  # 4 readings/day

# Starting inventory per product (units)
INITIAL_STOCK: dict[str, int] = {
    "Widget_A": 2_000,
    "Widget_B": 1_500,
    "Widget_C": 800,
}

# Reorder policy
REORDER_POINT: dict[str, int] = {
    "Widget_A": 300,
    "Widget_B": 200,
    "Widget_C": 100,
}
REORDER_QUANTITY: dict[str, int] = {
    "Widget_A": 1_500,
    "Widget_B": 1_000,
    "Widget_C": 500,
}
LEAD_TIME_DAYS: int = 3  # Days after reorder point hit before stock arrives

# ============================================================
# OUTPUT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _monthly_seasonality(date: pd.Timestamp) -> float:
    """Gentle seasonal multiplier — peaks in Nov/Dec (holiday season)."""
    month = date.month
    # Sine curve: trough in summer (month 7), peak in winter (month 12)
    return 1.0 + 0.15 * np.sin(2 * np.pi * (month - 3) / 12)


def _shock_overlap_fraction(
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    shock_start: pd.Timestamp,
    shock_end: pd.Timestamp,
) -> float:
    """
    Compute what fraction of a week [week_start, week_end] overlaps
    with the shock window [shock_start, shock_end] (inclusive on both ends).

    Returns a float in [0.0, 1.0].
    """
    overlap_start = max(week_start, shock_start)
    overlap_end = min(week_end, shock_end)

    if overlap_start > overlap_end:
        return 0.0

    overlap_days = (overlap_end - overlap_start).days + 1
    week_days = (week_end - week_start).days + 1
    return overlap_days / week_days


# ============================================================
# GENERATOR 1: marketing_weekly.csv
# ============================================================

def generate_marketing_weekly() -> pd.DataFrame:
    """
    Generate weekly marketing spend data with the injected marketing shock.

    Weekly dates are aligned to Mondays. For weeks that partially overlap
    the shock window (Jun 15 – Jul 15), the spend reduction is proportional
    to the fraction of the week inside the shock window.

    Returns:
        pd.DataFrame with columns:
            week_start, region, channel, ad_spend, impressions, clicks
    """
    logger.info("Generating marketing_weekly.csv ...")

    # Generate all Monday-aligned week starts within [START_DATE, END_DATE]
    all_mondays = pd.date_range(
        start=START_DATE - pd.Timedelta(days=START_DATE.weekday()),  # Align to Monday
        end=END_DATE,
        freq="W-MON",
    )
    # Filter to weeks that have at least one day in our range
    all_mondays = all_mondays[(all_mondays >= START_DATE - pd.Timedelta(days=6)) &
                              (all_mondays <= END_DATE)]

    rows: list[dict] = []

    for week_start in all_mondays:
        week_end = week_start + pd.Timedelta(days=6)  # Sunday

        # Compute shock overlap fraction for this week
        shock_frac = _shock_overlap_fraction(
            week_start, week_end, MARKETING_SHOCK_START, MARKETING_SHOCK_END
        )
        # Spend multiplier: 1.0 normally, reduced during shock proportionally
        shock_multiplier = 1.0 - (MARKETING_SHOCK_MAGNITUDE * shock_frac)

        for region in REGIONS:
            region_mult = REGION_MULTIPLIERS[region]
            seasonal = _monthly_seasonality(week_start)

            for channel in CHANNELS:
                base = BASE_WEEKLY_SPEND[channel]

                # Apply region, seasonality, shock, and noise
                spend = (
                    base
                    * region_mult
                    * seasonal
                    * shock_multiplier
                    * (1.0 + _rng.normal(0, 0.05))  # ±5% noise
                )
                spend = max(spend, 0)

                # Impressions from spend
                imp_rate = IMPRESSIONS_PER_DOLLAR[channel]
                impressions = int(spend * imp_rate * (1.0 + _rng.normal(0, 0.08)))
                impressions = max(impressions, 0)

                # Clicks from impressions
                ctr = CTR_BY_CHANNEL[channel]
                clicks = int(impressions * ctr * (1.0 + _rng.normal(0, 0.10)))
                clicks = max(clicks, 0)

                rows.append({
                    "week_start": week_start.strftime("%Y-%m-%d"),
                    "region": region,
                    "channel": channel,
                    "ad_spend": round(spend, 2),
                    "impressions": impressions,
                    "clicks": clicks,
                })

    df = pd.DataFrame(rows)
    logger.info(
        f"  marketing_weekly: {len(df):,} rows, "
        f"{df['week_start'].nunique()} weeks × {len(REGIONS)} regions × {len(CHANNELS)} channels"
    )
    return df


# ============================================================
# HELPER: Interpolate weekly marketing → daily influence
# ============================================================

def _compute_daily_marketing_influence(marketing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribute weekly marketing spend to daily values and apply an
    exponential moving average to simulate the lagged effect of
    marketing on daily web traffic.

    This is the critical alignment step: weekly spend is distributed
    uniformly across 7 days, then EMA-smoothed with a 7-day half-life
    so that a sudden spend change takes ~7 days to fully propagate.

    Returns:
        pd.DataFrame with columns: date, region, daily_mkt_spend, lagged_mkt_influence
    """
    all_dates = pd.date_range(START_DATE, END_DATE, freq="D")

    # Aggregate total spend per week per region (sum across channels)
    mkt = marketing_df.copy()
    mkt["week_start"] = pd.to_datetime(mkt["week_start"])
    weekly_totals = mkt.groupby(["week_start", "region"])["ad_spend"].sum().reset_index()
    weekly_totals.rename(columns={"ad_spend": "weekly_total_spend"}, inplace=True)

    rows: list[dict] = []
    for region in REGIONS:
        region_weekly = weekly_totals[weekly_totals["region"] == region].sort_values("week_start")

        # Build a daily series by assigning each day the spend of its containing week / 7
        daily_spend_series = []
        for _, row in region_weekly.iterrows():
            ws = row["week_start"]
            daily_val = row["weekly_total_spend"] / 7.0
            for d in range(7):
                day = ws + pd.Timedelta(days=d)
                if START_DATE <= day <= END_DATE:
                    daily_spend_series.append((day, daily_val))

        if not daily_spend_series:
            continue

        ds_df = pd.DataFrame(daily_spend_series, columns=["date", "daily_mkt_spend"])
        ds_df = ds_df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

        # Fill any gaps (edge weeks) with the nearest value
        full_idx = pd.DataFrame({"date": all_dates})
        ds_df = full_idx.merge(ds_df, on="date", how="left")
        ds_df["daily_mkt_spend"] = ds_df["daily_mkt_spend"].ffill().bfill()

        # Apply EMA for lagged marketing influence
        ds_df["lagged_mkt_influence"] = (
            ds_df["daily_mkt_spend"]
            .ewm(alpha=TRAFFIC_EMA_ALPHA, adjust=False)
            .mean()
        )
        ds_df["region"] = region
        rows.append(ds_df)

    result = pd.concat(rows, ignore_index=True)
    return result


# ============================================================
# GENERATOR 2: sales_daily.csv
# ============================================================

def generate_sales_daily(marketing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate daily sales data driven by the causal chain:
        Lagged Marketing Influence → Web Traffic → Units Sold → Revenue

    The stockout shock (Sep 1–14, Southeast Widget_A) suppresses sales
    even when marketing is normal, creating the multi-factor test case.

    Returns:
        pd.DataFrame with columns:
            date, region, product, units_sold, unit_price, gross_revenue,
            returns, net_revenue, web_traffic
    """
    logger.info("Generating sales_daily.csv ...")

    # Get daily marketing influence per region
    daily_mkt = _compute_daily_marketing_influence(marketing_df)

    all_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    rows: list[dict] = []

    for date in all_dates:
        dow = date.weekday()
        dow_mult = DOW_MULTIPLIERS[dow]
        seasonal = _monthly_seasonality(date)

        for region in REGIONS:
            region_mult = REGION_MULTIPLIERS[region]

            # Look up lagged marketing influence for this date+region
            mkt_row = daily_mkt[
                (daily_mkt["date"] == date) & (daily_mkt["region"] == region)
            ]
            if len(mkt_row) == 0:
                lagged_mkt = 0.0
            else:
                lagged_mkt = mkt_row["lagged_mkt_influence"].iloc[0]

            # Compute web traffic: organic + marketing-driven + noise
            traffic = (
                ORGANIC_TRAFFIC_BASE * region_mult
                + MARKETING_TRAFFIC_SENSITIVITY * lagged_mkt
                + _rng.normal(0, 200)
            )
            traffic = int(max(traffic * dow_mult * seasonal, 100))

            for product in PRODUCTS:
                # Skip Widget_C before its launch date (sparse-history)
                if product == SPARSE_PRODUCT and date < SPARSE_LAUNCH_DATE:
                    continue

                product_share = PRODUCT_TRAFFIC_SHARE[product]
                base_units = int(
                    traffic * CONVERSION_RATE * product_share
                    * (1.0 + _rng.normal(0, 0.08))  # ±8% noise
                )
                base_units = max(base_units, 0)

                # Apply stockout suppression: can't sell what you don't have
                is_stockout = (
                    region == STOCKOUT_REGION
                    and product == STOCKOUT_PRODUCT
                    and STOCKOUT_START <= date <= STOCKOUT_END
                )
                if is_stockout:
                    # Stockout: only ~5% of normal sales (some stores have residual)
                    base_units = max(int(base_units * 0.05), 0)

                units_sold = base_units

                # Unit price: base price with occasional promotions
                base_price = UNIT_PRICES[product]
                if _rng.random() < PROMO_PROBABILITY:
                    discount = _rng.uniform(*PROMO_DISCOUNT_RANGE)
                    unit_price = round(base_price * (1.0 - discount), 2)
                else:
                    unit_price = base_price

                gross_revenue = round(units_sold * unit_price, 2)

                # Returns: ~3% of units sold
                returns = int(_rng.binomial(units_sold, RETURN_RATE))
                net_revenue = round(gross_revenue - (returns * unit_price), 2)

                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "region": region,
                    "product": product,
                    "units_sold": units_sold,
                    "unit_price": unit_price,
                    "gross_revenue": gross_revenue,
                    "returns": returns,
                    "net_revenue": net_revenue,
                    "web_traffic": traffic,
                })

    df = pd.DataFrame(rows)
    logger.info(
        f"  sales_daily: {len(df):,} rows, "
        f"{df['date'].nunique()} days × {len(REGIONS)} regions"
    )
    return df


# ============================================================
# GENERATOR 3: inventory_hourly.csv
# ============================================================

def generate_inventory_hourly(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 6-hourly inventory snapshots driven by daily sales depletion.

    Inventory model:
    - Start each product-region with INITIAL_STOCK units.
    - Each day, deplete by that day's units_sold (spread across 4 intra-day readings).
    - When stock drops below REORDER_POINT, a reorder is placed.
    - After LEAD_TIME_DAYS, REORDER_QUANTITY arrives and replenishes stock.
    - During the stockout event (Sep 1–14, SE Widget_A), replenishment is blocked.

    Returns:
        pd.DataFrame with columns:
            timestamp, region, product, stock_on_hand, reorder_flag
    """
    logger.info("Generating inventory_hourly.csv ...")

    all_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    rows: list[dict] = []

    # Pre-aggregate daily units sold by (date, region, product)
    sales_lookup: dict[tuple[str, str, str], int] = {}
    for _, row in sales_df.iterrows():
        key = (row["date"], row["region"], row["product"])
        sales_lookup[key] = row["units_sold"]

    for region in REGIONS:
        for product in PRODUCTS:
            # Determine the start date for this product
            if product == SPARSE_PRODUCT:
                product_start = SPARSE_LAUNCH_DATE
            else:
                product_start = START_DATE

            stock = INITIAL_STOCK[product]
            reorder_pending = False
            reorder_arrives_on: pd.Timestamp | None = None

            for date in all_dates:
                if date < product_start:
                    continue

                # Check if reorder shipment arrives today
                if reorder_pending and reorder_arrives_on is not None and date >= reorder_arrives_on:
                    # Block replenishment during stockout event
                    is_stockout_block = (
                        region == STOCKOUT_REGION
                        and product == STOCKOUT_PRODUCT
                        and STOCKOUT_START <= date <= STOCKOUT_END
                    )
                    if not is_stockout_block:
                        stock += REORDER_QUANTITY[product]
                        reorder_pending = False
                        reorder_arrives_on = None

                # Get today's depletion
                daily_sold = sales_lookup.get(
                    (date.strftime("%Y-%m-%d"), region, product), 0
                )
                depletion_per_reading = daily_sold / len(INVENTORY_SAMPLE_HOURS)

                for hour in INVENTORY_SAMPLE_HOURS:
                    ts = date + pd.Timedelta(hours=hour)

                    # Deplete stock
                    stock -= depletion_per_reading
                    stock = max(stock, 0)

                    # Force stockout for the specific event
                    is_stockout_event = (
                        region == STOCKOUT_REGION
                        and product == STOCKOUT_PRODUCT
                        and STOCKOUT_START <= date <= STOCKOUT_END
                    )
                    if is_stockout_event:
                        stock = 0

                    reorder_flag = stock < REORDER_POINT[product]

                    # Trigger reorder if below point and no order pending
                    if reorder_flag and not reorder_pending:
                        reorder_pending = True
                        reorder_arrives_on = date + pd.Timedelta(days=LEAD_TIME_DAYS)

                    rows.append({
                        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "region": region,
                        "product": product,
                        "stock_on_hand": int(stock),
                        "reorder_flag": reorder_flag,
                    })

                    # Add small noise to stock for non-zero readings (shrinkage, adjustments)
                    if stock > 0:
                        stock += _rng.normal(0, 2)
                        stock = max(stock, 0)

    df = pd.DataFrame(rows)
    logger.info(
        f"  inventory_hourly: {len(df):,} rows "
        f"({len(INVENTORY_SAMPLE_HOURS)} readings/day × "
        f"{len(REGIONS)} regions × {len(PRODUCTS)} products)"
    )
    return df


# ============================================================
# ORCHESTRATOR
# ============================================================

def generate_all(output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Generate all three synthetic datasets and write them to CSV.

    The generation order enforces the causal chain:
        1. Marketing (root cause) → 2. Sales (driven by marketing) → 3. Inventory (driven by sales)

    Args:
        output_dir: Directory to write CSVs to. Defaults to data/raw/.

    Returns:
        Dict mapping filename to DataFrame for downstream use/testing.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Reset seed for consistent output regardless of import-time operations
    global _rng
    _rng = np.random.RandomState(MASTER_SEED)

    # Step 1: Marketing (the causal root)
    marketing_df = generate_marketing_weekly()
    marketing_df.to_csv(output_dir / "marketing_weekly.csv", index=False)
    logger.info(f"  ✓ Saved marketing_weekly.csv")

    # Step 2: Sales (driven by marketing through lagged traffic)
    sales_df = generate_sales_daily(marketing_df)
    sales_df.to_csv(output_dir / "sales_daily.csv", index=False)
    logger.info(f"  ✓ Saved sales_daily.csv")

    # Step 3: Inventory (driven by sales depletion)
    inventory_df = generate_inventory_hourly(sales_df)
    inventory_df.to_csv(output_dir / "inventory_hourly.csv", index=False)
    logger.info(f"  ✓ Saved inventory_hourly.csv")

    logger.info("=" * 60)
    logger.info("All datasets generated successfully.")
    logger.info(f"  marketing_weekly.csv  : {len(marketing_df):>10,} rows")
    logger.info(f"  sales_daily.csv       : {len(sales_df):>10,} rows")
    logger.info(f"  inventory_hourly.csv  : {len(inventory_df):>10,} rows")
    logger.info("=" * 60)

    return {
        "marketing_weekly": marketing_df,
        "sales_daily": sales_df,
        "inventory_hourly": inventory_df,
    }


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    generate_all()
