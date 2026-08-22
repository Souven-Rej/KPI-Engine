"""
Streamlit Interactive Decision Canvas for KPI Engine
====================================================
Phase 4: Provides the interactive dashboard proving that the engine handles
real-world complexities (telemetry, persona switching, data ambiguity).
"""

import os
import time
import pandas as pd
import streamlit as st

from src.detection.stl_detector import run_detection
from src.causal.dowhy_gcm import run_causal_attribution, align_datasets
from src.prescriptive.econml_cate import estimate_revenue_lift
from src.narrative.llm_synthesis import generate_narrative

st.set_page_config(
    page_title="KPI Engine | Reason v2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CACHED BACKEND FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_and_detect():
    """Run detection and data alignment once and cache the results."""
    anomaly_events, daily_kpi = run_detection()
    df_causal = align_datasets()
    return anomaly_events, daily_kpi, df_causal

@st.cache_data(show_spinner=False)
def run_causal_for_event(anomaly_events: pd.DataFrame, target_date: str, target_region: str):
    """Run causal attribution for a specific event."""
    event_df = anomaly_events[
        (anomaly_events["date"] == target_date) & 
        (anomaly_events["region"] == target_region)
    ]
    if event_df.empty:
        return None
        
    attribution_df = run_causal_attribution(event_df)
    return attribution_df.iloc[0].to_dict()

# ============================================================
# UI DEFINITION
# ============================================================

def main():
    # --- Sidebar ---
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/cd/Accenture.svg", width=150)
    st.sidebar.title("Reason v2 Prototype")
    st.sidebar.markdown("*BusinessIntelligence.ai*")
    st.sidebar.markdown("---")
    
    st.sidebar.header("Control Panel")
    
    persona = st.sidebar.selectbox(
        "Persona Selection",
        options=["vp_of_sales", "regional_manager"],
        format_func=lambda x: "VP of Sales" if x == "vp_of_sales" else "Regional Manager",
        help="Adjusts the LLM narrative tone and strategic depth."
    )
    
    scenarios = {
        "Marketing Shock (Jun-Jul)": {"date": "2025-07-12", "region": "Northeast", "sparse": False},
        "Stockout Conflicting Data (Sep)": {"date": "2025-09-01", "region": "Southeast", "sparse": False},
        "Sparse History (Data Ambiguity)": {"date": "2025-07-12", "region": "Northeast", "sparse": True},
    }
    
    scenario_name = st.sidebar.selectbox(
        "Scenario Injector",
        options=list(scenarios.keys()),
        help="Select a specific anomaly event to analyze."
    )
    
    analyze_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Accenture Innovation Challenge 2026")

    # --- Load Base Data ---
    with st.spinner("Initializing Pipeline..."):
        anomaly_events, daily_kpi, df_causal = load_and_detect()

    # If the user hasn't clicked analyze yet, show a welcome screen
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
        
    if analyze_btn:
        scenario = scenarios[scenario_name]
        target_date = scenario["date"]
        target_region = scenario["region"]
        is_sparse = scenario["sparse"]
        
        with st.spinner(f"Analyzing {target_region} on {target_date}..."):
            # 1. Fetch Anomaly
            event_row = anomaly_events[
                (anomaly_events["date"] == target_date) & 
                (anomaly_events["region"] == target_region)
            ]
            if event_row.empty:
                st.error("Scenario event not found in detection results!")
                return
            target_event = event_row.iloc[0].to_dict()
            
            # 2. Causal Attribution
            attribution_result = run_causal_for_event(anomaly_events, target_date, target_region)
            
            # 3. Prescriptive Analytics
            # Simulate sparse history by artificially truncating the causal dataframe
            df_prescriptive = df_causal.copy()
            if is_sparse:
                df_prescriptive = df_prescriptive[df_prescriptive["date"] >= "2025-07-01"] # Only 11 days of history
                
            prescriptive_result = estimate_revenue_lift(df_prescriptive, target_event)
            
            # 4. LLM Synthesis
            try:
                narrative_resp, telemetry = generate_narrative(
                    anomaly_data=target_event,
                    attribution_data=attribution_result,
                    prescriptive_data=prescriptive_result,
                    persona=persona,
                    data_ambiguity=prescriptive_result["data_ambiguity"]
                )
            except Exception as e:
                st.error(f"LLM Synthesis failed: {e}")
                return
                
            st.session_state.analysis_results = {
                "event": target_event,
                "attribution": attribution_result,
                "prescriptive": prescriptive_result,
                "narrative": narrative_resp,
                "telemetry": telemetry,
                "scenario": scenario_name
            }

    # --- Main Dashboard ---
    st.title("KPI Engine: Storytelling-to-Action")
    
    if not st.session_state.analysis_results:
        st.info("👈 Select a scenario and click **Run Analysis** to begin.")
        return
        
    results = st.session_state.analysis_results
    event = results["event"]
    attr = results["attribution"]
    presc = results["prescriptive"]
    narr = results["narrative"]
    telemetry = results["telemetry"]
    
    # 1. KPI Overview
    st.subheader(f"Anomaly Detected: {event['region']} Region")
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.metric(
            label="Net Revenue (Actual)",
            value=f"${event['net_revenue']:,.2f}",
            delta=f"{event['pct_deviation']:.1f}% vs baseline",
            delta_color="inverse"
        )
    with kpi_col2:
        st.metric(
            label="Statistical Severity",
            value=str(event['severity']).upper(),
            delta=f"Z-Score: {event['z_score']:.2f}",
            delta_color="inverse"
        )
    with kpi_col3:
        st.metric(
            label="Event Date",
            value=str(event['date']).split(" ")[0],
            delta="Confirmed Anomaly",
            delta_color="off"
        )
        
    st.markdown("---")
    
    # 2. Narrative Canvas & Traceability Panel
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.subheader("AI Narrative Synthesis")
        
        # Abstention Protocol Check
        if presc.get("data_ambiguity", False) or narr.confidence_status == "Investigation Required":
            st.error("⚠️ **ABSTENTION PROTOCOL ACTIVATED: Investigation Required**")
            st.warning(
                "The engine has detected sparse or highly conflicting historical data (< 30 days). "
                "Per core philosophy ('Math does the finding, AI does the explaining'), the AI is restricted "
                "from hallucinating causal drivers."
            )
            st.markdown(f"**AI Assessment:**\n{narr.narrative_summary}")
        else:
            st.success(f"✅ **Confidence Status: {narr.confidence_status}**")
            st.markdown(f"**Executive Summary:**\n{narr.narrative_summary}")
            
        st.markdown("### Key Drivers")
        for driver in narr.key_drivers:
            st.markdown(f"- {driver}")
            
        st.markdown("### Recommended Actions")
        for action in narr.recommended_actions:
            st.markdown(f"- {action}")
            
    with col_side:
        st.subheader("Traceability & Evidence")
        st.caption("Deterministic proof from Phase 2 & 3")
        
        with st.expander("Causal Attribution (DoWhy)", expanded=True):
            ad_pct = attr.get('ad_spend_contribution_pct', 0.0)
            stock_pct = attr.get('stock_on_hand_contribution_pct', 0.0)
            
            st.markdown("**Marketing (Ad Spend)**")
            st.progress(ad_pct / 100.0, text=f"{ad_pct:.1f}%")
            
            st.markdown("**Inventory (Stock on Hand)**")
            st.progress(stock_pct / 100.0, text=f"{stock_pct:.1f}%")
            
            st.caption(f"Primary Driver: **{attr.get('primary_driver', 'N/A')}**")
            
        with st.expander("Prescriptive Analytics (EconML)", expanded=True):
            if presc.get("data_ambiguity", False):
                st.warning("Data Ambiguity Flag: TRUE")
                st.metric("Expected Revenue Lift", "$0.00", "N/A")
            else:
                st.metric(
                    "Expected Revenue Lift", 
                    f"${presc.get('expected_lift', 0.0):,.2f}",
                    "If lever restored"
                )
                st.markdown(f"**CATE Estimate:** {presc.get('cate', 0.0):.2f}")
                st.caption(f"Baseline Spend: ${presc.get('baseline_spend', 0.0):,.2f}")
                st.caption(f"Current Spend: ${presc.get('current_spend', 0.0):,.2f}")
                
    st.markdown("---")
    
    # 3. Telemetry Footer
    st.caption("LLM Telemetry & Performance")
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    t_col1.metric("Latency", f"{telemetry.get('latency_seconds', 0.0):.2f}s")
    t_col2.metric("Prompt Tokens", telemetry.get("prompt_tokens", 0))
    t_col3.metric("Completion Tokens", telemetry.get("completion_tokens", 0))
    
    # Estimated cost (assuming gpt-4o pricing approx: $5/1M input, $15/1M output)
    p_tok = telemetry.get("prompt_tokens", 0)
    c_tok = telemetry.get("completion_tokens", 0)
    est_cost = (p_tok * 5.0 / 1_000_000) + (c_tok * 15.0 / 1_000_000)
    t_col4.metric("Est. API Cost", f"${est_cost:.5f}")

if __name__ == "__main__":
    main()
