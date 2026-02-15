import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ip_to_int(ip):
    """Convert IPv4 string to integer for range-based lookup."""
    try:
        if pd.isna(ip): return 0
        parts = list(map(int, str(ip).split('.')))
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except Exception:
        return 0

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and duplicates."""
    logging.info("Starting data cleaning...")
    df = df.drop_duplicates()
    # Dropping rows missing critical IDs for fraud analysis
    df = df.dropna(subset=['user_id', 'device_id', 'ip_address']) 
    return df

def convert_to_datetime(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Correct data types for time-based analysis."""
    for col in columns:
        df[col] = pd.to_datetime(df[col])
    return df

def map_ip_to_country(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """Efficiently map IP addresses to countries using merge_asof."""
    logging.info("Mapping IP addresses to countries...")
    
    # 1. Convert IPs to integers and ensure they are int64
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int).astype('int64')
    
    # 2. Ensure IP range data keys are also int64 (Fixes the MergeError)
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype('int64')
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype('int64')
    
    # 3. Sort both dataframes (REQUIRED for merge_asof)
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    fraud_df = fraud_df.sort_values('ip_int')

    # 4. Perform range-based lookup
    merged_df = pd.merge_asof(
        fraud_df, 
        ip_df, 
        left_on='ip_int', 
        right_on='lower_bound_ip_address'
    )

    # 5. Validate upper bound
    merged_df['country'] = np.where(
        merged_df['ip_int'] <= merged_df['upper_bound_ip_address'],
        merged_df['country'],
        'Unknown'
    )

    return merged_df



from sklearn.preprocessing import StandardScaler, LabelEncoder

def scale_and_encode(df: pd.DataFrame):
    logging.info("Scaling numerical features and encoding categoricals...")
    
    # Fill country 'Unknown' if missing after merge
    df['country'] = df['country'].fillna('Unknown')
    
    # Label Encoding for categorical columns
    le = LabelEncoder()
    df['source_encoded'] = le.fit_transform(df['source'])
    df['browser_encoded'] = le.fit_transform(df['browser'])
    df['sex_encoded'] = le.fit_transform(df['sex'])
    
    # Scaling numerical features
    scaler = StandardScaler()
    num_cols = ['purchase_value', 'time_since_signup', 'user_transaction_count']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df