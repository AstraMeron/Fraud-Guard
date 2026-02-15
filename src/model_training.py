import pandas as pd
from imblearn.over_sampling import SMOTE
import logging

def handle_imbalance(df: pd.DataFrame, target_col: str):
    logging.info("Applying SMOTE to handle class imbalance...")
    
    # Define features (X) and target (y)
    # We drop non-numeric and ID columns for the sampler
    X = df[['purchase_value', 'source_encoded', 'browser_encoded', 
            'sex_encoded', 'age', 'time_since_signup', 'user_transaction_count']]
    y = df[target_col]
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    logging.info(f"Original class distribution: {y.value_counts().to_dict()}")
    logging.info(f"Resampled class distribution: {pd.Series(y_res).value_counts().to_dict()}")
    
    return X_res, y_res