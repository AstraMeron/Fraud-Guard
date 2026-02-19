import logging
import joblib
import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Logging configuration for production-grade traceability
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MODEL_PATH = "models/random_forest_model.pkl"

def load_model():
    """Load trained model and log expected features."""
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at {MODEL_PATH}")
        return None

    try:
        model = joblib.load(MODEL_PATH)
        logging.info("✅ Model loaded successfully")

        # Extract training feature order for dynamic alignment
        if hasattr(model, "feature_names_in_"):
            logging.info(f"Model expects features: {list(model.feature_names_in_)}")
        else:
            logging.warning("Model does NOT store feature names. Verification might fail.")

        return model

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

# Load model once during server startup
model = load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for monitoring service health."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "Fraud-Guard API"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for real-time fraud prediction.
    Performs dynamic schema alignment to ensure feature parity.
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400

        logging.info(f"Incoming request: {data}")

        # 1. Convert incoming JSON to DataFrame
        input_df = pd.DataFrame([data])

        # 2. Rename incoming fields to match the training schema
        rename_map = {
            "browser": "browser_encoded",
            "sex": "sex_encoded",
            "source": "source_encoded",
            "time_diff": "time_since_signup"
        }
        input_df.rename(columns=rename_map, inplace=True)

        # 3. Align EXACTLY with training features (Schema Alignment)
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)

            # Add missing columns with default value 0
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Reorder and filter columns to match training exactly
            input_df = input_df[expected_features]
        
        logging.info(f"Aligned features for model: {list(input_df.columns)}")

        # 4. Perform Prediction
        prediction = model.predict(input_df)[0]
        # Probability of class 1 (Fraud)
        probability = model.predict_proba(input_df)[0][1]

        result = {
            "prediction": int(prediction),
            "fraud_probability": round(float(probability), 4),
            "class_label": "Fraud" if prediction == 1 else "Legitimate",
            "status": "success"
        }

        logging.info(f"Prediction result: {result['class_label']} (Prob: {result['fraud_probability']})")
        return jsonify(result)

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed", "message": str(e)}), 400

if __name__ == "__main__":
    # Running on 0.0.0.0 allows access from outside the container (for Task 4 Dockerization)
    app.run(host="0.0.0.0", port=5000, debug=True)