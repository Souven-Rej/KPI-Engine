"""
Streamlit Interactive Decision Canvas for KPI Engine
====================================================
Phase 4: Professional dark-themed executive analytics dashboard.
Proves that the engine handles real-world complexities
(telemetry, persona switching, data ambiguity) interactively.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path to fix Streamlit Cloud nested imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yaml

# Load environment variables from .env
load_dotenv(override=True)

from src.detection.stl_detector import run_detection
from src.causal.dowhy_gcm import run_causal_attribution, align_datasets
from src.prescriptive.econml_cate import estimate_revenue_lift
from src.narrative.llm_synthesis import generate_narrative
from src.security.rbac import authorize_role
from src.telemetry import RuntimeTelemetry
from src.governance.metadata import get_governance_summary, get_kpi_lineage

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KPI Engine · AI Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Professional Dark Theme
# ============================================================

st.markdown("""
<style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hide only the deploy button and right side menu, but keep the header for the sidebar toggle */
    .stAppDeployButton {display: none;}
    
    /* ---- Branded Header Bar ---- */
    .header-bar {
        background: linear-gradient(135deg, #1A1F2E 0%, #0E1117 50%, #1a0a2e 100%);
        border-bottom: 2px solid #A100FF;
        padding: 1.2rem 2rem;
        margin: -1rem -1rem 1.5rem -1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .header-bar h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #A100FF, #c77dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .header-bar .subtitle {
        color: #8b8fa3;
        font-size: 0.85rem;
        font-weight: 400;
        margin-top: 2px;
    }
    .header-badge {
        background: rgba(161, 0, 255, 0.15);
        border: 1px solid rgba(161, 0, 255, 0.4);
        color: #c77dff;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* ---- Glass KPI Cards ---- */
    .kpi-card {
        background: rgba(26, 31, 46, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 12px 12px 0 0;
    }
    .kpi-card.accent-red::before { background: linear-gradient(90deg, #ff4d6a, #ff6b81); }
    .kpi-card.accent-blue::before { background: linear-gradient(90deg, #4dabf7, #74c0fc); }
    .kpi-card.accent-purple::before { background: linear-gradient(90deg, #A100FF, #c77dff); }
    .kpi-card.accent-green::before { background: linear-gradient(90deg, #51cf66, #8ce99a); }
    .kpi-card.accent-amber::before { background: linear-gradient(90deg, #fcc419, #ffe066); }
    
    .kpi-card .kpi-icon {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }
    .kpi-card .kpi-label {
        font-size: 0.75rem;
        color: #8b8fa3;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
    .kpi-card .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 0.2rem;
    }
    .kpi-card .kpi-delta {
        font-size: 0.8rem;
        font-weight: 500;
    }
    .kpi-delta.negative { color: #ff6b81; }
    .kpi-delta.positive { color: #51cf66; }
    .kpi-delta.neutral  { color: #8b8fa3; }
    
    /* ---- Section Headers ---- */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .section-header .icon {
        font-size: 1.3rem;
    }
    .section-header h2 {
        margin: 0;
        font-size: 1.15rem;
        font-weight: 600;
        color: #FAFAFA;
    }
    
    /* ---- Confidence Badge ---- */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-high {
        background: rgba(81, 207, 102, 0.15);
        border: 1px solid rgba(81, 207, 102, 0.4);
        color: #51cf66;
    }
    .badge-investigation {
        background: rgba(255, 77, 106, 0.15);
        border: 1px solid rgba(255, 77, 106, 0.4);
        color: #ff6b81;
    }
    .badge-model {
        background: rgba(77, 171, 247, 0.12);
        border: 1px solid rgba(77, 171, 247, 0.3);
        color: #74c0fc;
    }
    
    /* ---- Narrative Card ---- */
    .narrative-card {
        background: rgba(26, 31, 46, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    /* ---- Driver Pills ---- */
    .driver-pill {
        background: rgba(161, 0, 255, 0.1);
        border: 1px solid rgba(161, 0, 255, 0.25);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        color: #e0d0f0;
    }
    
    /* ---- Action Items ---- */
    .action-item {
        background: rgba(81, 207, 102, 0.08);
        border-left: 3px solid #51cf66;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        color: #c8e6c9;
    }
    
    /* ---- Abstention Banner ---- */
    .abstention-banner {
        background: linear-gradient(135deg, rgba(255, 77, 106, 0.12), rgba(255, 77, 106, 0.05));
        border: 1px solid rgba(255, 77, 106, 0.35);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .abstention-banner h3 {
        color: #ff6b81;
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }
    .abstention-banner p {
        color: #c8a0a8;
        font-size: 0.88rem;
        margin: 0;
    }
    
    /* ---- Trace Panel & Native Containers ---- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(26, 31, 46, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
        padding: 0.2rem !important;
    }
    .trace-title {
        font-size: 0.8rem;
        color: #8b8fa3;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
        margin-bottom: 0.6rem;
        padding: 0.5rem 0.5rem 0 0.5rem;
    }
    
    /* ---- Telemetry Footer ---- */
    .telemetry-bar {
        background: rgba(14, 17, 23, 0.8);
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    .telemetry-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.75rem;
        color: #6c7086;
    }
    .telemetry-item .tel-label { font-weight: 500; }
    .telemetry-item .tel-value { color: #8b8fa3; font-weight: 600; }
    
    /* ---- Sidebar Styling ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0E1117 100%);
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08);
    }
    
    /* Ensure sidebar toggle button is visible over custom header */
    [data-testid="collapsedControl"] {
        color: #FAFAFA !important;
        background-color: #1A1F2E !important;
        border: 1px solid #A100FF !important;
        border-radius: 50% !important;
        z-index: 999999 !important;
        top: 1rem !important;
        left: 1rem !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: #FAFAFA !important;
    }
    
    /* ---- Streamlit widget overrides ---- */
    .stSelectbox label, .stRadio label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #8b8fa3 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER: Load roles from contract
# ============================================================

def _load_roles_from_contract() -> dict[str, str]:
    contract_path = Path(__file__).resolve().parents[2] / "config" / "kpi_contract.yaml"
    with open(contract_path, encoding="utf-8") as f:
        contract = yaml.safe_load(f) or {}
    roles = contract.get("roles", {})
    return {key: value.get("display_name", key) for key, value in roles.items()}


# ============================================================
# CACHED BACKEND FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_and_detect():
    data_path = PROJECT_ROOT / "data" / "raw" / "sales_daily.csv"
    if not data_path.exists():
        from src.data_generation.enterprise_warehouse_etl import generate_all
        # Just run generation directly if missing (e.g. fresh clone on Streamlit Cloud)
        generate_all()

    anomaly_events, daily_kpi = run_detection()
    df_causal = align_datasets()
    return anomaly_events, daily_kpi, df_causal

@st.cache_data(show_spinner=False)
def run_causal_for_event(anomaly_events: pd.DataFrame, target_date: str, target_region: str):
    event_df = anomaly_events[
        (anomaly_events["date"] == target_date) & 
        (anomaly_events["region"] == target_region)
    ]
    if event_df.empty:
        return None
    attribution_df = run_causal_attribution(event_df)
    return attribution_df.iloc[0].to_dict()


# ============================================================
# COMPONENT BUILDERS
# ============================================================

def render_header():
    st.markdown("""
    <div class="header-bar">
        <div>
            <h1>⚡ KPI Engine</h1>
            <div class="subtitle">Storytelling-to-Action Pipeline · Accenture Innovation Challenge 2026</div>
        </div>
        <div class="header-badge">BusinessIntelligence.ai</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(icon, label, value, delta, accent="blue", delta_class="neutral"):
    st.markdown(f"""
    <div class="kpi-card accent-{accent}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta {delta_class}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(icon, title):
    st.markdown(f"""
    <div class="section-header">
        <span class="icon">{icon}</span>
        <h2>{title}</h2>
    </div>
    """, unsafe_allow_html=True)


def render_donut_chart(ad_pct, stock_pct):
    """Create a professional donut chart for causal attribution."""
    labels = ["Ad Spend", "Inventory"]
    values = [ad_pct, stock_pct]
    colors = ["#A100FF", "#4dabf7"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#0E1117", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#FAFAFA"),
        hovertemplate="<b>%{label}</b><br>Contribution: %{value:.1f}%<extra></extra>"
    )])
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=20, r=20, t=30, b=30),
        annotations=[dict(
            text=f"<b>{max(ad_pct, stock_pct):.0f}%</b>",
            x=0.5, y=0.5, font_size=22, font_color="#FAFAFA",
            showarrow=False
        )]
    )
    return fig


def render_timeseries(daily_kpi, event):
    """Create a professional dark-themed time series chart."""
    region_df = daily_kpi[daily_kpi["region"] == event["region"]].copy()
    target_dt = pd.to_datetime(event["date"])

    fig = go.Figure()

    # Baseline band
    if "baseline" in region_df.columns:
        fig.add_trace(go.Scatter(
            x=region_df["date"], y=region_df["baseline"],
            mode="lines",
            line=dict(color="rgba(139,143,163,0.4)", width=1, dash="dot"),
            name="Baseline (STL Trend)",
            hovertemplate="Baseline: $%{y:,.0f}<extra></extra>"
        ))

    # Revenue line
    fig.add_trace(go.Scatter(
        x=region_df["date"], y=region_df["net_revenue"],
        mode="lines",
        line=dict(color="#4dabf7", width=2.5),
        name="Net Revenue",
        fill="tonexty",
        fillcolor="rgba(77,171,247,0.06)",
        hovertemplate="Revenue: $%{y:,.0f}<extra></extra>"
    ))

    # Anomaly marker
    anomaly_pt = region_df[region_df["date"] == target_dt]
    if not anomaly_pt.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_pt["date"],
            y=anomaly_pt["net_revenue"],
            mode="markers+text",
            marker=dict(color="#ff4d6a", size=14, symbol="diamond",
                        line=dict(color="#ff6b81", width=2)),
            text=["⚠ Anomaly"],
            textposition="top center",
            textfont=dict(color="#ff6b81", size=11, family="Inter"),
            name="Detected Anomaly",
            hovertemplate="<b>ANOMALY</b><br>$%{y:,.0f}<extra></extra>"
        ))

        # Vertical line at anomaly
        fig.add_vline(
            x=target_dt.timestamp() * 1000,
            line=dict(color="rgba(255,77,106,0.3)", width=1, dash="dash")
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=11, color="#8b8fa3"),
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            title=None,
            tickfont=dict(color="#6c7086", size=10)
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            title=dict(text="Revenue ($)", font=dict(color="#6c7086", size=11)),
            tickfont=dict(color="#6c7086", size=10),
            tickprefix="$", tickformat=","
        ),
        margin=dict(l=60, r=20, t=30, b=40)
    )
    return fig


# ============================================================
# MAIN APP
# ============================================================

def main():
    # --- Load Base Data ---
    with st.spinner("⚡ Initializing deterministic pipeline..."):
        anomaly_events, daily_kpi, df_causal = load_and_detect()

    # --- Header ---
    render_header()

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        st.markdown("---")

        role_names = _load_roles_from_contract()
        persona = st.selectbox(
            "PERSONA",
            options=list(role_names.keys()),
            format_func=lambda x: role_names.get(x, x),
            help="Adjusts the LLM narrative tone and strategic depth."
        )
        if not authorize_role(persona, "show_statistics"):
            st.warning("⚠ Persona not authorized for statistics.")

        st.markdown("")

        if anomaly_events.empty:
            st.info("No anomaly events detected.")
            analyze_btn = st.button("Run Analysis", type="primary", use_container_width=True, disabled=True)
            target_event = None
        else:
            event_options = {}
            for _, event in anomaly_events.sort_values(["date", "region"]).iterrows():
                sev = event.get("severity", "unknown")
                sev_icon = "🔴" if sev == "CRITICAL" else "🟡" if sev == "WARNING" else "🔵"
                label = f"{sev_icon} {pd.to_datetime(event['date']).strftime('%b %d, %Y')} · {event['region']}"
                event_options[label] = event

            scenario_label = st.selectbox(
                "SCENARIO",
                options=list(event_options.keys()),
                help="Select a detected anomaly event to analyze.",
                index=0,
            )
            target_event = event_options[scenario_label].to_dict()
            st.markdown("")
            analyze_btn = st.button("⚡ Run Analysis", type="primary", use_container_width=True)

        st.markdown("---")
        st.caption("Accenture Innovation Challenge 2026")

    # --- Session State ---
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    telemetry_logger = RuntimeTelemetry(log_dir="data")

    # --- Run Analysis ---
    if analyze_btn and target_event is not None:
        target_date = pd.to_datetime(target_event["date"]).strftime("%Y-%m-%d")
        target_region = target_event["region"]

        with st.spinner(f"⚡ Analyzing {target_region} · {target_date}..."):
            start = time.time()

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

            attribution_result = run_causal_for_event(anomaly_events, target_date, target_region)

            df_prescriptive = df_causal.copy()
            primary_driver = attribution_result.get("primary_driver", "ad_spend")
            prescriptive_result = estimate_revenue_lift(df_prescriptive, target_event, primary_driver=primary_driver)

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
                "analysis_run", "ui",
                latency_seconds=time.time() - start,
                persona=persona, region=target_region,
                scenario=scenario_label, event_date=target_date,
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
                "persona": persona,
            }

    # --- Re-run LLM if persona changed ---
    elif st.session_state.analysis_results and st.session_state.analysis_results.get("persona") != persona:
        res = st.session_state.analysis_results
        with st.spinner(f"🔄 Adapting narrative for {persona}..."):
            try:
                narr_resp, tel = generate_narrative(
                    anomaly_data=res["event"],
                    attribution_data=res["attribution"],
                    prescriptive_data=res["prescriptive"],
                    persona=persona,
                    data_ambiguity=res["prescriptive"].get("data_ambiguity", False)
                )
                res["narrative"] = narr_resp
                res["telemetry"] = tel
                res["persona"] = persona
                st.session_state.analysis_results = res
            except Exception as e:
                st.error(f"LLM Synthesis failed: {e}")

    # --- Welcome State ---
    if not st.session_state.analysis_results:
        st.markdown("<br>", unsafe_allow_html=True)
        col_w1, col_w2, col_w3 = st.columns([1, 2, 1])
        with col_w2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
                <h2 style="color: #FAFAFA; font-weight: 600; margin-bottom: 0.5rem;">Ready to Analyze</h2>
                <p style="color: #6c7086; font-size: 1rem;">
                    Select a detected anomaly from the sidebar and click <strong>Run Analysis</strong> 
                    to trace the full deterministic pipeline.
                </p>
                <div style="margin-top: 1.5rem;">
                    <span class="badge badge-model">STL Detection</span>&nbsp;
                    <span style="color: #6c7086;">→</span>&nbsp;
                    <span class="badge badge-model">Causal Attribution</span>&nbsp;
                    <span style="color: #6c7086;">→</span>&nbsp;
                    <span class="badge badge-model">Prescriptive CATE</span>&nbsp;
                    <span style="color: #6c7086;">→</span>&nbsp;
                    <span class="badge badge-model">AI Synthesis</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return

    # ================================================================
    # RESULTS DASHBOARD
    # ================================================================

    results = st.session_state.analysis_results
    event = results["event"]
    attr = results["attribution"]
    presc = results["prescriptive"]
    narr = results["narrative"]
    telemetry = results["telemetry"]

    # ---- KPI Summary Row ----
    render_section_header("📊", f"Anomaly Overview · {event['region']} Region")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            "💰", "Net Revenue",
            f"${event['net_revenue']:,.0f}",
            f"{event['pct_deviation']:+.1f}% vs baseline",
            accent="red",
            delta_class="negative" if event['pct_deviation'] < 0 else "positive"
        )
    with k2:
        sev = str(event['severity']).upper()
        sev_accent = "red" if sev == "CRITICAL" else "amber" if sev == "WARNING" else "blue"
        render_kpi_card(
            "🎯", "Severity",
            sev,
            f"Z-Score: {event['z_score']:.2f}",
            accent=sev_accent,
            delta_class="negative"
        )
    with k3:
        render_kpi_card(
            "📅", "Event Date",
            str(event['date']).split(" ")[0],
            "Confirmed Anomaly",
            accent="blue",
            delta_class="neutral"
        )
    with k4:
        driver = attr.get("primary_driver", "N/A")
        driver_display = "Ad Spend" if driver == "ad_spend" else "Inventory" if driver == "stock_on_hand" else driver
        contrib = attr.get(driver + '_contribution_pct', 0)
        low_conf = attr.get("low_confidence", False)
        delta_text = f"{contrib:.1f}% · Low Confidence" if low_conf else f"{contrib:.1f}% contribution"
        render_kpi_card(
            "🔍", "Primary Driver",
            driver_display,
            delta_text,
            accent="amber" if low_conf else "purple",
            delta_class="neutral"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Time Series Chart ----
    render_section_header("📈", "KPI Trend & Anomaly Detection")
    fig_ts = render_timeseries(daily_kpi, event)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Two-Column Analysis Panel ----
    col_main, col_side = st.columns([5, 3])

    with col_main:
        render_section_header("🤖", "AI Narrative Synthesis")

        # Abstention Protocol
        is_abstention = presc.get("data_ambiguity", False) or narr.confidence_status == "Investigation Required"

        if is_abstention:
            st.markdown("""
            <div class="abstention-banner">
                <h3>⛔ ABSTENTION PROTOCOL ACTIVATED</h3>
                <p>Insufficient historical data (&lt;30 days) to produce a statistically confident causal attribution. 
                The AI narrative below is flagged as low-confidence and should not be used for decision-making without further investigation.</p>
            </div>
            """, unsafe_allow_html=True)
            badge_html = '<span class="badge badge-investigation">⚠ Investigation Required</span>'
        else:
            badge_html = '<span class="badge badge-high">✓ High Confidence</span>'

        st.markdown(f"""
        <div class="narrative-card">
            <div style="margin-bottom: 0.8rem;">{badge_html}</div>
            <div style="font-size: 0.95rem; line-height: 1.7; color: #d0d0e0;">
                {narr.narrative_summary}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Key Drivers
        render_section_header("🔑", "Key Drivers")
        for driver in narr.key_drivers:
            st.markdown(f'<div class="driver-pill">● {driver}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Recommended Actions
        render_section_header("✅", "Recommended Actions")
        for action in narr.recommended_actions:
            st.markdown(f'<div class="action-item">→ {action}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feedback Form
        render_section_header("💬", "Analyst Feedback Loop")
        with st.form("feedback_form"):
            feedback_rating = st.radio(
                "Is this root-cause attribution accurate?",
                ["✓ Yes — Spot On", "◐ Partially — Missing Context", "✗ No — Incorrect"],
                horizontal=True
            )
            feedback_comments = st.text_area("Additional context (optional)", height=80)
            submit_feedback = st.form_submit_button("Submit Feedback", use_container_width=True)

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
                st.success("✅ Feedback captured and queued for retraining.")

    with col_side:
        # Causal Attribution Donut
        render_section_header("🧪", "Causal Attribution")

        ad_pct = attr.get('ad_spend_contribution_pct', 0.0)
        stock_pct = attr.get('stock_on_hand_contribution_pct', 0.0)

        with st.container(border=True):
            st.markdown('<div class="trace-title">DoWhy · Counterfactual Decomposition</div>', unsafe_allow_html=True)
            
            fig_donut = render_donut_chart(ad_pct, stock_pct)
            st.plotly_chart(fig_donut, use_container_width=True)
    
            driver_label = "Ad Spend" if attr.get("primary_driver") == "ad_spend" else "Inventory"
            st.markdown(f'<div style="text-align:center; color:#8b8fa3; font-size:0.8rem; margin-top:-0.5rem; margin-bottom:1rem;">Primary Driver: <strong style="color:#c77dff;">{driver_label}</strong></div>', unsafe_allow_html=True)

        # Prescriptive Analytics
        render_section_header("💰", "Prescriptive Analytics")

        with st.container(border=True):
            st.markdown('<div class="trace-title">EconML · Double Machine Learning</div>', unsafe_allow_html=True)
            
            if presc.get("data_ambiguity", False):
                st.markdown("""
                <div style="text-align:center; padding: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚠️</div>
                    <div style="color: #ff6b81; font-weight: 600;">Data Ambiguity</div>
                    <div style="color: #8b8fa3; font-size: 0.8rem; margin-top: 0.3rem;">Insufficient data for CATE estimation</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                lift = presc.get("expected_lift", 0.0)
                cate = presc.get("cate", 0.0)
                st.markdown(f"""
                <div style="text-align:center; padding: 0.5rem;">
                    <div style="font-size: 0.75rem; color: #8b8fa3; text-transform: uppercase; letter-spacing: 0.5px;">Expected Revenue Lift</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #51cf66; margin: 0.3rem 0;">${lift:,.0f}</div>
                    <div style="font-size: 0.8rem; color: #6c7086;">if lever restored to baseline</div>
                </div>
                <div style="display:flex; justify-content:space-around; margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.06); padding-bottom: 1rem;">
                    <div style="text-align:center;">
                        <div style="font-size: 0.7rem; color: #6c7086;">CATE</div>
                        <div style="font-size: 1rem; font-weight: 600; color: #FAFAFA;">{cate:.2f}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size: 0.7rem; color: #6c7086;">Current</div>
                        <div style="font-size: 1rem; font-weight: 600; color: #ff6b81;">${presc.get('current_spend', 0):,.0f}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size: 0.7rem; color: #6c7086;">Baseline</div>
                        <div style="font-size: 1rem; font-weight: 600; color: #51cf66;">${presc.get('baseline_spend', 0):,.0f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Governance
        render_section_header("🏛️", "Governance & Lineage")
        governance = get_governance_summary()
        
        with st.container(border=True):
            st.markdown('<div class="trace-title">KPI Lineage · Data Sources</div>', unsafe_allow_html=True)
            for name, lin in governance["lineage"].items():
                display = lin.get('display_name', name)
                source = lin.get('source_table', '')
                grain = lin.get('grain', '')
                st.markdown(
                    f'<div style="margin-bottom:0.6rem; padding: 0 0.5rem;">'
                    f'<div style="color:#c77dff;font-size:0.85rem;font-weight:500;">{display}</div>'
                    f'<div style="color:#6c7086;font-size:0.75rem;">{source} · {grain}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ---- Telemetry Footer ----
    st.markdown("<br>", unsafe_allow_html=True)
    p_tok = telemetry.get("prompt_tokens", 0)
    c_tok = telemetry.get("completion_tokens", 0)
    est_cost = (p_tok * 0.15 / 1_000_000) + (c_tok * 0.60 / 1_000_000)  # Gemini pricing
    runtime = results.get("runtime_telemetry", {})

    st.markdown(f"""
    <div class="telemetry-bar" style="justify-content: center; gap: 3rem;">
        <div class="telemetry-item">
            <span style="color: #51cf66;">●</span> 
            <span class="tel-label">System Status:</span> 
            <span class="tel-value" style="color: #51cf66;">Healthy</span>
        </div>
        <div class="telemetry-item">
            <span class="tel-label">⚡ End-to-End Latency:</span> 
            <span class="tel-value">{telemetry.get('latency_seconds', 0):.2f}s</span>
        </div>
        <div class="telemetry-item">
            <span class="tel-label">🔧 Synthesis Engine:</span> 
            <span class="tel-value">{telemetry.get('model_used', 'N/A')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Governance footer
    lineage = get_kpi_lineage("regional_net_revenue")
    st.markdown(f"""
    <div style="text-align: center; padding: 0.8rem; color: #4a4e5a; font-size: 0.7rem;">
        Governance: KPI lineage loaded for {lineage.get('display_name')} · source={lineage.get('source_table')} · 
        KPI Engine v1.0 · Accenture Innovation Challenge 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
