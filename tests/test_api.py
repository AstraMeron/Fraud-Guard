import requests
import json

# The URL where your Flask server is running
url = "http://127.0.0.1:5000/predict"

# Sample transaction data for testing
# Note: We use the raw names like 'browser' because the API 
# now handles the renaming to 'browser_encoded' internally.
data = {
    "purchase_value": 50,
    "age": 30,
    "browser": 0,
    "sex": 0,
    "source": 0,
    "time_diff": 100,
    "user_transaction_count": 1
}

def test_prediction():
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, json=data)
        
        # Check if the request was successful
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Prediction Successful!")
            # Pretty-print the JSON response
            print(json.dumps(response.json(), indent=4))
        else:
            print("❌ Prediction Failed!")
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server. Is serve_model.py running?")

if __name__ == "__main__":
    test_prediction()