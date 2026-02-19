import pandas as pd
import numpy as np
import logging
import joblib
import os
from typing import Tuple, Any

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve, 
    auc
)

def handle_imbalance(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Applies SMOTE to balance the dataset."""
    logging.info("Applying SMOTE to handle class imbalance...")
    
    X = df[['purchase_value', 'source_encoded', 'browser_encoded', 
            'sex_encoded', 'age', 'time_since_signup', 'user_transaction_count']]
    y = df[target_col]
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    logging.info(f"Original class distribution: {y.value_counts().to_dict()}")
    logging.info(f"Resampled class distribution: {pd.Series(y_res).value_counts().to_dict()}")
    
    return X_res, y_res

def prepare_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[Any, Any, Any, Any]:
    """Split the data into training and testing sets using stratification."""
    logging.info(f"Splitting data with test_size={test_size} and stratification...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series, 
                         X_test: pd.DataFrame, y_test: pd.Series) -> LogisticRegression:
    """Train a Logistic Regression baseline and evaluate performance."""
    logging.info("Training Logistic Regression baseline...")
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    logging.info("Baseline Evaluation Results:")
    print("\n--- Logistic Regression (Baseline) ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    auc_pr = auc(recall, precision)
    print(f"AUC-PR Score: {auc_pr:.4f}")
    
    return model

def train_ensemble_model(X_train, y_train, X_test, y_test) -> RandomForestClassifier:
    """Train a Random Forest model with basic hyperparameter tuning."""
    logging.info("Training Random Forest ensemble model...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1 
    )
    
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    y_probs = rf_model.predict_proba(X_test)[:, 1]
    
    logging.info("Ensemble Evaluation Results:")
    print("\n--- Random Forest (Ensemble) ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    auc_pr = auc(recall, precision)
    print(f"Ensemble AUC-PR Score: {auc_pr:.4f}")
    
    return rf_model
    
def perform_cross_validation(model, X, y, k=5):
    """Perform Stratified K-Fold cross-validation."""
    logging.info(f"Starting {k}-fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    
    logging.info(f"Cross-Validation F1-Scores: {scores}")
    logging.info(f"Mean F1-Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    return scores

def save_model(model: Any, filename: str):
    """Saves the trained model to the 'models/' directory."""
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', filename)
    joblib.dump(model, filepath)
    logging.info(f"Model saved successfully to {filepath}")