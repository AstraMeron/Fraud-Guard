import pandas as pd

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day, and time_since_signup."""
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Duration in seconds (more granular for ML)
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    
    return df

def create_transaction_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate frequency/velocity of transactions per user."""
    # Count of transactions per user_id
    df['user_transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
    return df