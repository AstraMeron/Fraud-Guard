import pandas as pd
import logging
import os
from src.preprocessing import clean_data, convert_to_datetime, map_ip_to_country, scale_and_encode
from src.feature_engineering import create_time_features, create_transaction_velocity
from src.model_training import (
    handle_imbalance, 
    prepare_train_test_split, 
    train_baseline_model,
    train_ensemble_model,
    perform_cross_validation,
    save_model
)
# Task 3 Imports [cite: 103]
from src.explainability import plot_feature_importance, run_shap_analysis

# Set up logging for the main execution 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- 1. Load & Initial Preprocessing ---
    logging.info("Loading datasets...")
    fraud_data = pd.read_csv('data/raw/Fraud_Data.csv')
    ip_data = pd.read_csv('data/raw/IpAddress_to_Country.csv')
    
    fraud_data = clean_data(fraud_data)
    fraud_data = convert_to_datetime(fraud_data, ['signup_time', 'purchase_time'])
    fraud_data = map_ip_to_country(fraud_data, ip_data)

    # --- 2. Feature Engineering & Transformation ---
    fraud_data = create_time_features(fraud_data)
    fraud_data = create_transaction_velocity(fraud_data)
    fraud_data = scale_and_encode(fraud_data)

    # --- 3. Handle Imbalance (SMOTE) ---
    X, y = handle_imbalance(fraud_data, 'class')

    print("\n" + "="*30)
    print("TASK 1 COMPLETE")
    print(f"Feature matrix shape: {X.shape}")
    print("="*30 + "\n")

    # --- 4. Task 2: Data Splitting (Stratified) ---
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)

    # --- 5. Task 2: Baseline Model Training ---
    baseline_model = train_baseline_model(X_train, y_train, X_test, y_test)

    # --- 6. Task 2: Ensemble Model Training (Random Forest) ---
    ensemble_model = train_ensemble_model(X_train, y_train, X_test, y_test)

    # --- 7. Task 2: Cross-Validation ---
    cv_scores = perform_cross_validation(ensemble_model, X, y)

    # --- 8. Save Models ---
    save_model(baseline_model, 'baseline_logistic_model.pkl')
    save_model(ensemble_model, 'random_forest_model.pkl')

    print("\n" + "="*30)
    print("ALL MODELING TASKS COMPLETE")
    print("Models saved in /models directory")
    print("="*30 + "\n")

    # --- 9. Task 3: Model Explainability ---
    logging.info("Starting Task 3: Explainability Analysis...")
    feature_names = X.columns.tolist()

    plot_feature_importance(ensemble_model, feature_names)

    # Note: We now pass y_test to help find specific TP, FP, FN cases
    run_shap_analysis(ensemble_model, X_test, y_test)

    print("\n" + "="*30)
    print("TASK 3: GLOBAL EXPLAINABILITY COMPLETE")
    print("Check reports/figures/ for visualizations")
    print("="*30)

if __name__ == "__main__":
    main()