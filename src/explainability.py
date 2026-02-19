import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import logging
from typing import List, Any, Dict

def plot_feature_importance(model: Any, feature_names: List[str], top_n: int = 10) -> None:
    """
    Extracts built-in feature importance.
    """
    logging.info(f"Extracting top {top_n} features from model...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Feature Importances (Built-in)")
    plt.barh(range(len(indices)), importances[indices], align="center", color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/feature_importance_baseline.png', bbox_inches='tight')
    plt.close()

def run_shap_analysis(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Generates SHAP Global Summary and Local Force Plots for specific cases.
    Fulfills Task 3 requirements for both Global and Local interpretability.
    """
    logging.info("Initializing SHAP Explainer...")
    
    # Initialize the explainer
    explainer = shap.TreeExplainer(model)
    
    # Sample data for performance
    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)

    # Use the new Explanation API
    explanation = explainer(X_sample)

    # Resolve class indexing for binary classification (Fraud is class 1)
    # This check handles both 2D and 3D explanation objects
    if len(explanation.shape) == 3:
        shap_values_to_plot = explanation[:, :, 1]
    else:
        shap_values_to_plot = explanation

    # 1. Global Summary Plot [Task 3 Requirement]
    logging.info("Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values_to_plot, show=False)
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/shap_summary_plot.png', bbox_inches='tight')
    plt.close()

    # 2. Local Case Explanations [Task 3 Requirement]
    logging.info("Searching for specific prediction cases (TP, FP, FN)...")
    y_pred = model.predict(X_test)
    
    try:
        # Find indices for the three specific scenarios required
        tp_idx = X_test[(y_test == 1) & (y_pred == 1)].index[0]
        fp_idx = X_test[(y_test == 0) & (y_pred == 1)].index[0]
        fn_idx = X_test[(y_test == 1) & (y_pred == 0)].index[0]
        
        cases: Dict[str, int] = {
            'True_Positive': tp_idx,
            'False_Positive': fp_idx,
            'False_Negative': fn_idx
        }
    except IndexError:
        logging.warning("Test set sample too small to find all TP/FP/FN cases.")
        return

    for name, idx in cases.items():
        logging.info(f"Generating Force Plot for {name}...")
        
        row_data = X_test.loc[[idx]]
        # Explaining a single row
        row_explanation = explainer(row_data)

        # Handle class indexing for the single row
        if len(row_explanation.shape) == 3:
            row_explanation = row_explanation[:, :, 1]

        # Extract values explicitly to avoid "ambiguous array" warnings
        base_val = float(row_explanation.base_values[0])
        shap_vals = row_explanation.values[0]
        feature_vals = row_data.iloc[0]

        plt.figure(figsize=(12, 3))
        shap.plots.force(
            base_val, 
            shap_vals, 
            feature_vals, 
            matplotlib=True, 
            show=False
        )
        plt.savefig(f'reports/figures/shap_force_{name.lower()}.png', bbox_inches='tight')
        plt.close()

    logging.info("SHAP analysis complete. Plots saved to reports/figures/")