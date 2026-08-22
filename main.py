import json
import logging
from pprint import pprint

from src.detection.stl_detector import run_detection
from src.causal.dowhy_gcm import run_causal_attribution, align_datasets
from src.prescriptive.econml_cate import estimate_revenue_lift
from src.narrative.llm_synthesis import generate_narrative

# Set up logging for the terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def main():
    print("==========================================")
    print("PIPELINE INTEGRATION TEST")
    print("==========================================")

    # 1. Detection Phase
    print("\n[1] Running STL Detection...")
    anomaly_events, _ = run_detection()
    
    # Grab a marketing shock event (e.g. July)
    marketing_events = anomaly_events[
        (anomaly_events["date"] >= "2025-07-01") & 
        (anomaly_events["date"] <= "2025-07-31")
    ]
    if marketing_events.empty:
        print("No marketing anomalies found!")
        return
        
    target_event_row = marketing_events.iloc[0]
    target_event = target_event_row.to_dict()
    print(f"Target Anomaly Selected: {target_event['date']} | {target_event['region']} | {target_event['severity']}")

    # 2. Causal Phase
    print("\n[2] Running Causal Attribution...")
    # Just attribute the single event to save time
    attribution_df = run_causal_attribution(marketing_events.head(1).copy())
    attribution_result = attribution_df.iloc[0].to_dict()
    print(f"Attribution: ad_spend={attribution_result['ad_spend_contribution_pct']}%, stock_on_hand={attribution_result['stock_on_hand_contribution_pct']}%")

    # 3. Prescriptive Phase
    print("\n[3] Running Prescriptive CATE Estimation...")
    df_causal = align_datasets()
    prescriptive_result = estimate_revenue_lift(df_causal, target_event)
    print(f"Prescriptive Lift: ${prescriptive_result['expected_lift']} (Ambiguity: {prescriptive_result['data_ambiguity']})")

    # 4. Narrative Phase
    print("\n[4] Running LLM Narrative Synthesis...")
    try:
        response, telemetry = generate_narrative(
            anomaly_data=target_event,
            attribution_data=attribution_result,
            prescriptive_data=prescriptive_result,
            persona="vp_of_sales",
            data_ambiguity=prescriptive_result["data_ambiguity"]
        )
        print("\n--- LLM Synthesis Response ---")
        # Ensure we can print the pydantic model cleanly
        if hasattr(response, "model_dump"):
            print(json.dumps(response.model_dump(), indent=2))
        else:
            print(response)
        print("\n--- Telemetry ---")
        pprint(telemetry)
    except Exception as e:
        print(f"Narrative generation failed: {e}")

if __name__ == "__main__":
    main()
