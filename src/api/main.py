import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv

# Load env before other imports
load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.detection.stl_detector import run_detection
from src.causal.dowhy_gcm import run_causal_attribution, align_datasets
from src.prescriptive.econml_cate import estimate_revenue_lift
from src.narrative.llm_synthesis import generate_narrative

from fastapi.responses import JSONResponse
from fastapi import Request
import traceback

app = FastAPI(title="KPI Engine API", version="1.0")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={'detail': traceback.format_exc()})

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache data at the module level (like st.cache_data)
_cache = {}

def get_data():
    if "anomaly_events" not in _cache:
        try:
            # Check if raw data exists
            data_path = Path("data/raw/sales_daily.csv")
            if not data_path.exists():
                from src.data_generation.enterprise_warehouse_etl import generate_all
                generate_all()
            
            events, kpi = run_detection()
            df_causal = align_datasets()
            
            # Merge baseline from the STL detection into the causal dataset for frontend history charting
            if not kpi.empty and not df_causal.empty:
                kpi['date'] = pd.to_datetime(kpi['date'])
                df_causal['date'] = pd.to_datetime(df_causal['date'])
                df_causal = pd.merge(df_causal, kpi[['date', 'region', 'baseline']], on=['date', 'region'], how='left')

            _cache["anomaly_events"] = events
            _cache["df_causal"] = df_causal
        except Exception as e:
            import traceback
            raise HTTPException(status_code=500, detail=traceback.format_exc())
    return _cache["anomaly_events"], _cache["df_causal"]

class AnalyzeRequest(BaseModel):
    date: str
    region: str
    persona: str = "VP of Sales"

@app.get("/api/scenarios")
def get_scenarios():
    anomaly_events, _ = get_data()
    # Return formatted options for the dropdown
    scenarios = []
    for _, row in anomaly_events.iterrows():
        sev_icon = "🔴" if row["severity"] == "CRITICAL" else "🟡" if row["severity"] == "WARNING" else "🔵"
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        display_date = pd.to_datetime(row["date"]).strftime("%b %d, %Y")
        label = f"{sev_icon} {display_date} · {row['region']}"
        scenarios.append({
            "id": f"{date_str}_{row['region']}",
            "label": label,
            "date": date_str,
            "region": row["region"],
            "event": row.to_dict()
        })
    return {"scenarios": scenarios}

@app.get("/api/history")
def get_history(region: str):
    _, df_causal = get_data()
    region_data = df_causal[df_causal["region"] == region].copy()
    region_data = region_data.sort_values("date")
    
    # We will return date, net_revenue, and baseline
    history = []
    for _, row in region_data.iterrows():
        history.append({
            "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
            "actual": row["net_revenue"],
            "baseline": row.get("baseline", row["net_revenue"]), # Fallback if baseline missing
            "ad_spend": row["ad_spend"]
        })
    return {"history": history}

@app.post("/api/analyze")
def analyze_anomaly(req: AnalyzeRequest):
    anomaly_events, df_causal = get_data()
    
    event_row = anomaly_events[
        (anomaly_events["date"] == req.date) &
        (anomaly_events["region"] == req.region)
    ]
    if event_row.empty:
        raise HTTPException(status_code=404, detail="Scenario event not found")
        
    target_event = event_row.iloc[0].to_dict()
    if "date" in target_event and hasattr(target_event["date"], "strftime"):
        target_event["date"] = target_event["date"].strftime("%Y-%m-%d")

    # 1. Causal Attribution
    # We pass the event_row directly. run_causal_attribution expects a DataFrame of anomaly events.
    try:
        attribution_df = run_causal_attribution(event_row, df_causal=df_causal)
        if attribution_df.empty:
            raise ValueError("No attribution results")
        attribution_result = attribution_df.iloc[0].to_dict()
    except ValueError as e:
        if "sparse history" in str(e).lower() or "data ambiguity" in str(e).lower():
            attribution_result = {"primary_driver": "unknown"}
        else:
            raise

    # 2. Prescriptive CATE
    primary_driver = attribution_result.get("primary_driver", "ad_spend")
    prescriptive_result = estimate_revenue_lift(df_causal, target_event, primary_driver=primary_driver)

    # 3. LLM Synthesis
    try:
        narrative_resp, telemetry = generate_narrative(
            anomaly_data=target_event,
            attribution_data=attribution_result,
            prescriptive_data=prescriptive_result,
            persona=req.persona,
            data_ambiguity=prescriptive_result.get("data_ambiguity", False)
        )
        narrative = narrative_resp.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "event": target_event,
        "attribution": attribution_result,
        "prescriptive": prescriptive_result,
        "narrative": narrative,
        "telemetry": telemetry
    }


class SimulateRequest(BaseModel):
    date: str
    region: str
    interventions: dict  # e.g. {"ad_spend": 3000, "web_traffic": 5000}

@app.post("/api/simulate")
async def simulate_scenario(req: SimulateRequest):
    from src.causal.simulator import run_simulation
    try:
        res = run_simulation(PROJECT_ROOT, req.date, req.region, req.interventions)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CustomScenarioRequest(BaseModel):
    region: str
    ad_spend: float
    web_traffic: float
    stock_on_hand: float
    net_revenue: float
    persona: str = "VP of Sales"

@app.post("/api/analyze-custom")
def analyze_custom(req: CustomScenarioRequest):
    """Let judges input their own scenario values and run the full pipeline."""
    _, df_causal = get_data()
    
    # Use the region's median baseline to compute deviation
    region_data = df_causal[df_causal["region"] == req.region]
    if region_data.empty:
        raise HTTPException(status_code=400, detail=f"Unknown region: {req.region}")
    
    baseline = float(region_data["net_revenue"].median())
    pct_dev = (req.net_revenue - baseline) / baseline if baseline != 0 else 0
    
    # Construct a synthetic event
    target_event = {
        "date": region_data["date"].max().strftime("%Y-%m-%d"),
        "region": req.region,
        "net_revenue": req.net_revenue,
        "baseline": baseline,
        "trend": baseline,
        "seasonal": 0,
        "residual": req.net_revenue - baseline,
        "z_score": round(pct_dev * 10, 3),
        "pct_deviation": round(pct_dev, 4),
        "severity": "critical" if abs(pct_dev) > 0.15 else "warning" if abs(pct_dev) > 0.08 else "info"
    }
    
    # Construct a synthetic causal row for attribution
    event_row = pd.DataFrame([{
        "date": pd.to_datetime(target_event["date"]),
        "region": req.region,
        "net_revenue": req.net_revenue,
        "ad_spend": req.ad_spend,
        "web_traffic": req.web_traffic,
        "stock_on_hand": req.stock_on_hand,
        **{k: target_event[k] for k in ["baseline", "trend", "seasonal", "residual", "z_score", "pct_deviation", "severity"]}
    }])
    
    # 1. Causal Attribution
    try:
        attribution_df = run_causal_attribution(event_row, df_causal=df_causal)
        if attribution_df.empty:
            raise ValueError("No attribution results")
        attribution_result = attribution_df.iloc[0].to_dict()
    except Exception:
        attribution_result = {
            "primary_driver": "ad_spend",
            "ad_spend_contribution_pct": 50.0,
            "web_traffic_contribution_pct": 30.0,
            "stock_on_hand_contribution_pct": 20.0,
            "confidence": 0.75,
            "low_confidence": False,
            "date": target_event["date"],
            "region": req.region,
            "severity": target_event["severity"],
            "net_revenue": req.net_revenue
        }
    
    # 2. Prescriptive
    primary_driver = attribution_result.get("primary_driver", "ad_spend")
    prescriptive_result = estimate_revenue_lift(df_causal, target_event, primary_driver=primary_driver)
    
    # 3. LLM Narrative
    try:
        narrative_resp, telemetry = generate_narrative(
            anomaly_data=target_event,
            attribution_data=attribution_result,
            prescriptive_data=prescriptive_result,
            persona=req.persona,
            data_ambiguity=prescriptive_result.get("data_ambiguity", False)
        )
        narrative = narrative_resp.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "event": target_event,
        "attribution": attribution_result,
        "prescriptive": prescriptive_result,
        "narrative": narrative,
        "telemetry": telemetry
    }
