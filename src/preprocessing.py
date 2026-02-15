import pandas as pd
import numpy as np
import logging
from typing import List, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ip_to_int(ip: Union[str, float]) -> int:
    """
    Convert IPv4 string to integer for range-based lookup.
    
    Args:
        ip: The IP address string (e.g., '192.168.1.1').
        
    Returns:
        int: The integer representation of the IP.
    """
    try:
        if pd.isna(ip): return 0
        parts = list(map(int, str(ip).split('.')))
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except Exception:
        return 0

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and duplicates for the fraud dataset.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logging.info("Starting data cleaning...")
    df = df.drop_duplicates()
    # Dropping rows missing critical IDs for fraud analysis
    df = df.dropna(subset=['user_id', 'device_id', 'ip_address']) 
    return df

def convert_to_datetime(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Correct data types for time-based analysis.
    
    Args:
        df: Input DataFrame.
        columns: List of column names to convert.
        
    Returns:
        pd.DataFrame: DataFrame with converted datetime columns.
    """
    logging.info(f"Converting {columns} to datetime...")
    for col in columns:
        df[col] = pd.to_datetime(df[col])
    return df

def map_ip_to_country(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Efficiently map IP addresses to countries using merge_asof.
    
    Args:
        fraud_df: DataFrame containing transaction data.
        ip_df: DataFrame containing IP range to country mapping.
        
    Returns:
        pd.DataFrame: Merged DataFrame with country information.
    """
    logging.info("Mapping IP addresses to countries...")
    
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int).astype('int64')
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype('int64')
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype('int64')
    
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    fraud_df = fraud_df.sort_values('ip_int')

    merged_df = pd.merge_asof(
        fraud_df, 
        ip_df, 
        left_on='ip_int', 
        right_on='lower_bound_ip_address'
    )

    merged_df['country'] = np.where(
        merged_df['ip_int'] <= merged_df['upper_bound_ip_address'],
        merged_df['country'],
        'Unknown'
    )

    return merged_df

def scale_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numerical features and encode categorical variables.
    
    Args:
        df: The DataFrame after feature engineering.
        
    Returns:
        pd.DataFrame: Processed DataFrame ready for modeling.
    """
    logging.info("Scaling numerical features and encoding categoricals...")
    
    df['country'] = df['country'].fillna('Unknown')
    
    # Categorical columns
    cat_cols = ['source', 'browser', 'sex']
    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
    
    # Numerical columns
    scaler = StandardScaler()
    num_cols = ['purchase_value', 'time_since_signup', 'user_transaction_count']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df