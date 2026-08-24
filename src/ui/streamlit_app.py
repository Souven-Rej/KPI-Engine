"""
Streamlit Interactive Decision Canvas for KPI Engine
====================================================
Phase 4: Provides the interactive dashboard proving that the engine handles
real-world complexities (telemetry, persona switching, data ambiguity).
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
import plotly.express as px
import yaml

# Load environment variables from .env
load_dotenv()

from src.detection.stl_detector import run_detection
from src.causal.dowhy_gcm import run_causal_attribution, align_datasets
from src.prescriptive.econml_cate import estimate_revenue_lift
from src.narrative.llm_synthesis import generate_narrative
from src.security.rbac import authorize_role
from src.telemetry import RuntimeTelemetry
from src.governance.metadata import get_governance_summary, get_kpi_lineage


def _load_roles_from_contract() -> dict[str, str]:
    """Load persona names from the KPI contract instead of hardcoded UI choices."""
    contract_path = Path(__file__).resolve().parents[2] / "config" / "kpi_contract.yaml"
    with open(contract_path, encoding="utf-8") as f:
        contract = yaml.safe_load(f) or {}
    roles = contract.get("roles", {})
    return {key: value.get("display_name", key) for key, value in roles.items()}

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
    # --- Load Base Data First ---
    with st.spinner("Initializing Pipeline..."):
        anomaly_events, daily_kpi, df_causal = load_and_detect()

    # --- Sidebar ---
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/cd/Accenture.svg", width=150)
    st.sidebar.title("Reason v2 Prototype")
    st.sidebar.markdown("*BusinessIntelligence.ai*")
    st.sidebar.markdown("---")

    st.sidebar.header("Control Panel")

    role_names = _load_roles_from_contract()
    persona = st.sidebar.selectbox(
        "Persona Selection",
        options=list(role_names.keys()),
        format_func=lambda x: role_names.get(x, x),
        help="Adjusts the LLM narrative tone and strategic depth."
    )
    if not authorize_role(persona, "show_statistics"):
        st.sidebar.warning("This persona is not allowed to view statistics in the current policy.")

    if anomaly_events.empty:
        st.sidebar.caption("No anomaly events detected for the current KPI run.")
        analyze_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True, disabled=True)
        target_event = None
    else:
        event_options = {}
        for _, event in anomaly_events.sort_values(["date", "region"]).iterrows():
            label = f"{pd.to_datetime(event['date']).strftime('%Y-%m-%d')} | {event['region']} | {event.get('severity', 'unknown')}"
            event_options[label] = event

        scenario_label = st.sidebar.selectbox(
            "Scenario Injector",
            options=list(event_options.keys()),
            help="Select a detected anomaly event to analyze.",
            index=0,
        )
        target_event = event_options[scenario_label].to_dict()
        analyze_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Accenture Innovation Challenge 2026")

    # If the user hasn't clicked analyze yet, show a welcome screen
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
        
    telemetry_logger = RuntimeTelemetry(log_dir="data")

    if analyze_btn and target_event is not None:
        target_date = pd.to_datetime(target_event["date"]).strftime("%Y-%m-%d")
        target_region = target_event["region"]

        with st.spinner(f"Analyzing {target_region} on {target_date}..."):
            start = time.time()
            # 1. Fetch Anomaly from the real detected event list
            event_row = anomaly_events[
                (anomaly_events["date"] == target_date) &
                (anomaly_events["region"] == target_region)
            ]
            if event_row.empty:
                st.error("Scenario event not found in detection results!")
                return
            target_event = event_row.iloc[0].to_dict()
            if "date" in target_event and hasattr(target_event["date"], "strftime"):
                target_event["date"] = target_event["date"].strftime("%Y-%m-%d")

            # 2. Causal Attribution
            attribution_result = run_causal_for_event(anomaly_events, target_date, target_region)

            # 3. Prescriptive Analytics
            df_prescriptive = df_causal.copy()
            primary_driver = attribution_result.get("primary_driver", "ad_spend")
            prescriptive_result = estimate_revenue_lift(df_prescriptive, target_event, primary_driver=primary_driver)

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

            telemetry_logger.record(
                "analysis_run",
                "ui",
                latency_seconds=time.time() - start,
                persona=persona,
                region=target_region,
                scenario=scenario_label,
                event_date=target_date,
                primary_driver=attribution_result.get("primary_driver"),
            )

            st.session_state.analysis_results = {
                "event": target_event,
                "attribution": attribution_result,
                "prescriptive": prescriptive_result,
                "narrative": narrative_resp,
                "telemetry": telemetry,
                "scenario": scenario_label,
                "runtime_telemetry": telemetry_logger.summary(),
            }

    # --- Main Dashboard ---
    st.title("KPI Engine: Storytelling-to-Action")
    
    if not st.session_state.analysis_results:
        if anomaly_events.empty:
            st.info("No anomaly events were detected in the current run. The dashboard is waiting for data to produce an analysis.")
        else:
            st.info("👈 Select a detected anomaly and click **Run Analysis** to begin.")
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
    
    # 1.5 Time Series Visual Proof (Plotly)
    st.subheader("KPI Trend & Anomaly Detection")
    region_kpi_df = daily_kpi[daily_kpi["region"] == event["region"]].copy()
    
    # Create the Plotly figure
    fig = px.line(
        region_kpi_df, 
        x="date", 
        y=["net_revenue", "baseline"], 
        labels={"value": "Revenue ($)", "date": "Date", "variable": "Metric"},
        color_discrete_map={"net_revenue": "#1f77b4", "baseline": "#7f7f7f"}
    )
    # Add a red marker for the exact target event
    target_dt = pd.to_datetime(event["date"])
    anomaly_pt = region_kpi_df[region_kpi_df["date"] == target_dt]
    if not anomaly_pt.empty:
        fig.add_scatter(
            x=anomaly_pt["date"], 
            y=anomaly_pt["net_revenue"], 
            mode="markers", 
            marker=dict(color="red", size=12, symbol="x"),
            name="Detected Anomaly"
        )
    fig.update_layout(height=400, hovermode="x unified", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    
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
            
        st.markdown("---")
        st.markdown("### 🔄 Analyst Feedback Loop")
        st.caption("Help retrain the causal engine by validating this insight.")
        
        with st.form("feedback_form"):
            feedback_rating = st.radio("Is this root-cause attribution accurate?", ["Yes - Spot On", "Partially - Missing Context", "No - Incorrect"])
            feedback_comments = st.text_area("Additional Context for Data Science Team (Optional)")
            submit_feedback = st.form_submit_button("Submit Feedback")
            
            if submit_feedback:
                feedback_data = pd.DataFrame([{
                    "timestamp": pd.Timestamp.now(),
                    "scenario": results["scenario"],
                    "persona": persona,
                    "rating": feedback_rating,
                    "comments": feedback_comments,
                    "role_allowed": authorize_role(persona, "show_statistics"),
                }])
                feedback_file = "data/feedback_log.csv"
                if os.path.exists(feedback_file):
                    feedback_data.to_csv(feedback_file, mode='a', header=False, index=False)
                else:
                    feedback_data.to_csv(feedback_file, index=False)

                feedback_summary = telemetry_logger.summary()
                st.success(
                    "✅ Feedback successfully captured and queued for retraining. "
                    f"Runtime telemetry records: {feedback_summary.get('total_events', 0)}"
                )
            
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

        with st.expander("Governance & Evidence", expanded=True):
            governance = get_governance_summary()
            st.caption("KPI lineage and source freshness")
            for name, lineage in governance["lineage"].items():
                st.markdown(f"**{name}**")
                st.caption(f"Source: {lineage.get('source_table')} | Grain: {lineage.get('grain')}")
                st.caption(f"Formula: {lineage.get('formula')}")
            st.markdown("---")
            st.caption("Source freshness")
            for source_name, source_info in governance["source_freshness"].items():
                st.caption(f"{source_name}: {source_info.get('grain')} / {source_info.get('path')}")

    st.markdown("---")
    
    # 3. Telemetry Footer
    st.caption("LLM Telemetry & Performance")
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    t_col1.metric("Latency", f"{telemetry.get('latency_seconds', 0.0):.2f}s")
    t_col2.metric("Prompt Tokens", telemetry.get("prompt_tokens", 0))
    t_col3.metric("Completion Tokens", telemetry.get("completion_tokens", 0))
    
    p_tok = telemetry.get("prompt_tokens", 0)
    c_tok = telemetry.get("completion_tokens", 0)
    est_cost = (p_tok * 5.0 / 1_000_000) + (c_tok * 15.0 / 1_000_000)
    t_col4.metric("Est. API Cost", f"${est_cost:.5f}")

    runtime_summary = results.get("runtime_telemetry", {})
    if runtime_summary:
        st.caption(
            f"Runtime telemetry: {runtime_summary.get('total_events', 0)} events, "
            f"avg latency {runtime_summary.get('avg_latency_seconds', 0.0):.3f}s"
        )

    lineage = get_kpi_lineage("regional_net_revenue")
    st.caption(
        f"Governance: KPI lineage loaded for {lineage.get('display_name')} | "
        f"source={lineage.get('source_table')}"
    )


if __name__ == "__main__":
    main()
