import pandas as pd
from src.preprocessing import clean_data, convert_to_datetime, map_ip_to_country, scale_and_encode
from src.feature_engineering import create_time_features, create_transaction_velocity
from src.model_training import handle_imbalance

# 1. Load & Map
fraud_data = pd.read_csv('data/raw/Fraud_Data.csv')
ip_data = pd.read_csv('data/raw/IpAddress_to_Country.csv')
fraud_data = clean_data(fraud_data)
fraud_data = convert_to_datetime(fraud_data, ['signup_time', 'purchase_time'])
fraud_data = map_ip_to_country(fraud_data, ip_data)

# 2. Feature Engineer & Transform
fraud_data = create_time_features(fraud_data)
fraud_data = create_transaction_velocity(fraud_data)
fraud_data = scale_and_encode(fraud_data)

# 3. Handle Imbalance
X, y = handle_imbalance(fraud_data, 'class')

print("Final Task 1 Output:")
print(f"Feature matrix shape: {X.shape}")
print("Ready for Modeling!")