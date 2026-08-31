import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv

# Load env before other imports
load_dotenv(override=True)

from src.detection.stl_detector import run_detection
from src.causal.dowhy_gcm import run_causal_attribution, align_datasets
from src.prescriptive.econml_cate import estimate_revenue_lift
from src.narrative.llm_synthesis import generate_narrative

app = FastAPI(title="KPI Engine API", version="1.0")

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
        # Check if raw data exists
        data_path = Path("data/raw/sales_daily.csv")
        if not data_path.exists():
            from src.data_generation.synthetic_generator import generate_all
            generate_all()
        
        events, kpi = run_detection()
        df_causal = align_datasets()
        _cache["anomaly_events"] = events
        _cache["df_causal"] = df_causal
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
