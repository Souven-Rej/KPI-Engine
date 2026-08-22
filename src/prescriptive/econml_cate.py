"""
Prescriptive Analytics Engine for KPI Engine
============================================

Estimates the Conditional Average Treatment Effect (CATE) using Double Machine Learning (DML)
from econml. Validates data history rules and returns expected revenue lift
if actionable levers (like ad_spend) are restored to baseline.

Rules:
    - Sparse History Rule: If region data < 30 days, skip DML, return data_ambiguity = True.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from econml.dml import LinearDML

logger = logging.getLogger(__name__)


def estimate_revenue_lift(
    df_causal: pd.DataFrame,
    anomaly_event: dict[str, Any],
    min_history_days: int = 30
) -> dict[str, Any]:
    """
    Calculate the expected revenue lift if the ad_spend lever is restored to baseline.

    Args:
        df_causal: The aligned multi-grain DataFrame.
        anomaly_event: The anomaly event dictionary containing date and region.
        min_history_days: Minimum days required to run the DML.

    Returns:
        Dict with keys: 
            expected_lift, data_ambiguity, message, current_spend, baseline_spend, cate
    """
    region = anomaly_event.get("region")
    event_date = pd.to_datetime(anomaly_event.get("date"))
    
    # 1. Check sparse history rule
    region_df = df_causal[df_causal["region"] == region].sort_values("date").reset_index(drop=True)
    history_before_event = region_df[region_df["date"] <= event_date].copy()
    
    if len(history_before_event) < min_history_days:
        logger.warning(
            "Sparse history for region %s (< %d days). Triggering data ambiguity.",
            region, min_history_days
        )
        return {
            "expected_lift": 0.0,
            "data_ambiguity": True,
            "message": f"Data history is too sparse (< {min_history_days} days) to estimate treatment effect.",
            "current_spend": None,
            "baseline_spend": None,
            "cate": None
        }

    # 2. Setup DML variables
    # We want to train on the data up to the event to learn the CATE
    # Treatment (T) = ad_spend
    # Outcome (Y) = net_revenue
    # Features (X) = stock_on_hand (so CATE is conditioned on stock)
    
    Y = history_before_event["net_revenue"].values
    T = history_before_event["ad_spend"].values
    X = history_before_event[["stock_on_hand"]].values
    
    # 3. Fit LinearDML
    logger.debug("Fitting LinearDML for region %s...", region)
    dml = LinearDML(
        model_y=LassoCV(cv=3, random_state=42),
        model_t=LassoCV(cv=3, random_state=42),
        random_state=42,
        discrete_treatment=False
    )
    
    try:
        dml.fit(Y, T, X=X)
    except Exception as e:
        logger.error("DML fitting failed: %s", e)
        return {
            "expected_lift": 0.0,
            "data_ambiguity": True,
            "message": "DML model failed to converge or fit due to insufficient variation.",
            "current_spend": None,
            "baseline_spend": None,
            "cate": None
        }
        
    # 4. Compute Lift
    # Find the current (anomalous) ad_spend
    current_row = history_before_event[history_before_event["date"] == event_date]
    if current_row.empty:
        current_spend = 0.0
        current_stock = 0.0
    else:
        current_spend = current_row["ad_spend"].values[0]
        current_stock = current_row["stock_on_hand"].values[0]
        
    # Baseline ad_spend (we take the mean of the first half of history as 'normal' baseline)
    baseline_spend = history_before_event["ad_spend"].head(len(history_before_event) // 2).mean()
    
    # CATE for the current state
    X_pred = np.array([[current_stock]])
    cate_estimate = dml.effect(X_pred)[0]
    
    expected_lift = cate_estimate * (baseline_spend - current_spend)
    # We only care about positive lift (recovering revenue)
    expected_lift = max(expected_lift, 0.0)
    
    logger.info(
        "Region %s | Prescriptive CATE = %.2f, Lift = $%.2f (Restoring spend from $%.2f to $%.2f)",
        region, cate_estimate, expected_lift, current_spend, baseline_spend
    )
    
    return {
        "expected_lift": round(float(expected_lift), 2),
        "data_ambiguity": False,
        "message": "Successfully estimated revenue lift.",
        "current_spend": round(float(current_spend), 2),
        "baseline_spend": round(float(baseline_spend), 2),
        "cate": round(float(cate_estimate), 2)
    }
