import pandas as pd
import logging

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from signup and purchase timestamps.
    
    Args:
        df: Input DataFrame containing signup_time and purchase_time.
        
    Returns:
        pd.DataFrame: DataFrame with new temporal features.
    """
    logging.info("Creating time-based features...")
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Duration in seconds
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    
    return df

def create_transaction_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the velocity of transactions (frequency) per user.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with transaction count features.
    """
    logging.info("Calculating transaction velocity...")
    df['user_transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
    return df