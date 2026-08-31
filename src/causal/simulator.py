import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.causal.dowhy_gcm import build_causal_dag, align_datasets, NODE_NET_REVENUE, ALL_NODES, load_contract, InvertibleStructuralCausalModel

logger = logging.getLogger(__name__)

def run_simulation(project_root: Path, date: str, region: str, intervention_node: str, new_value: float) -> dict:
    """
    Run a counterfactual simulation using the fitted DoWhy GCM causal graph.
    Returns the expected counterfactual value of Net Revenue.
    """
    contract = load_contract(project_root / "config" / "kpi_contract.yaml")
    df_causal = align_datasets(project_root, contract)
    
    # 1. Get the factual row
    event_date = pd.to_datetime(date)
    match = df_causal[
        (df_causal["date"] == event_date) &
        (df_causal["region"] == region)
    ]
    if match.empty:
        raise ValueError(f"Factual data not found for {date} / {region}")
        
    factual_row = match.copy()
    
    # 2. Fit the causal model on recent history (e.g. 60 days) to capture the current structural equation
    recent_history = df_causal[df_causal["date"] <= event_date].tail(300) # grab enough rows across 5 regions to fit
    
    dag = build_causal_dag(contract)
    model = InvertibleStructuralCausalModel(dag)
    model.fit(recent_history[ALL_NODES])
    
    # 3. Simulate Counterfactual reality
    # Compute noises of factual reality to freeze the non-intervened mechanisms
    noises = model._compute_all_noises(factual_row)
    
    simulated_values = {}
    n_samples = len(factual_row)
    
    for node in model._node_order:
        if node == intervention_node:
            val = np.full(n_samples, new_value)
            simulated_values[node] = val
            continue
            
        mechanism = model._mechanisms[node]
        parents = list(model.dag.predecessors(node))
        
        if len(parents) == 0:
            simulated_values[node] = factual_row[node].values
        else:
            X = np.column_stack([simulated_values[p] for p in parents])
            y_pred = mechanism._model.predict(X)
            val = y_pred + noises[node]
            simulated_values[node] = val
    
    cf_revenue = float(simulated_values[NODE_NET_REVENUE][0])
    factual_revenue = float(factual_row[NODE_NET_REVENUE].values[0])
    
    return {
        "factual_revenue": factual_revenue,
        "simulated_revenue": cf_revenue,
        "lift": cf_revenue - factual_revenue,
        "intervened_node": intervention_node,
        "new_value": new_value
    }
