from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# A dictionary to store the app's state
APP_STATE = {}

# Load ML model and scaler
# Make sure the paths are correct relative to where you run the script
MODEL_PATH = "models/crop_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Ideal conditions for different crops for in-depth analysis
IDEAL_CONDITIONS = {
    'wheat': {'N': [50, 100], 'P': [40, 60], 'K': [30, 50], 'T': [10, 25], 'H': [60, 80], 'pH': [6.0, 7.5], 'R': [250, 450]},
    'rice': {'N': [70, 120], 'P': [40, 60], 'K': [30, 60], 'T': [20, 35], 'H': [70, 90], 'pH': [5.5, 6.5], 'R': [1000, 2000]},
    'maize': {'N': [80, 120], 'P': [50, 80], 'K': [40, 70], 'T': [18, 30], 'H': [60, 80], 'pH': [5.5, 7.0], 'R': [500, 800]},
    'cotton': {'N': [60, 100], 'P': [30, 50], 'K': [40, 70], 'T': [25, 35], 'H': [50, 60], 'pH': [5.5, 7.5], 'R': [500, 1000]},
    'jute': {'N': [40, 80], 'P': [20, 40], 'K': [20, 40], 'T': [24, 35], 'H': [80, 90], 'pH': [6.0, 7.0], 'R': [1500, 2500]},
    'coffee': {'N': [80, 120], 'P': [50, 80], 'K': [40, 70], 'T': [15, 25], 'H': [70, 80], 'pH': [6.0, 7.0], 'R': [1000, 2000]},
    'tea': {'N': [60, 100], 'P': [30, 50], 'K': [20, 40], 'T': [15, 25], 'H': [70, 80], 'pH': [5.0, 6.0], 'R': [1500, 2500]},
    'banana': {'N': [80, 120], 'P': [50, 80], 'K': [40, 70], 'T': [25, 30], 'H': [70, 80], 'pH': [5.5, 6.5], 'R': [1000, 2000]},
    'mango': {'N': [40, 80], 'P': [20, 40], 'K': [30, 60], 'T': [24, 30], 'H': [50, 60], 'pH': [6.0, 7.5], 'R': [500, 1000]},
    'apple': {'N': [50, 80], 'P': [30, 50], 'K': [40, 70], 'T': [10, 20], 'H': [60, 80], 'pH': [6.0, 7.0], 'R': [500, 1000]},
    'grapes': {'N': [60, 100], 'P': [40, 60], 'K': [50, 80], 'T': [20, 30], 'H': [60, 70], 'pH': [5.5, 7.0], 'R': [500, 800]},
    'pomegranate': {'N': [40, 60], 'P': [20, 40], 'K': [30, 50], 'T': [25, 35], 'H': [50, 60], 'pH': [6.0, 7.5], 'R': [300, 600]},
    'orange': {'N': [50, 80], 'P': [30, 50], 'K': [40, 70], 'T': [20, 30], 'H': [60, 70], 'pH': [6.0, 7.5], 'R': [500, 800]},
    'lemon': {'N': [40, 60], 'P': [20, 40], 'K': [30, 50], 'T': [25, 35], 'H': [60, 70], 'pH': [5.5, 6.5], 'R': [500, 800]},
    'papaya': {'N': [50, 80], 'P': [30, 50], 'K': [40, 70], 'T': [20, 30], 'H': [70, 80], 'pH': [6.0, 7.0], 'R': [1000, 2000]},
    'coconut': {'N': [50, 80], 'P': [30, 50], 'K': [40, 70], 'T': [25, 35], 'H': [70, 80], 'pH': [6.0, 7.0], 'R': [1000, 2000]},
    'blackgram': {'N': [40, 60], 'P': [20, 40], 'K': [20, 40], 'T': [20, 30], 'H': [60, 70], 'pH': [6.0, 7.5], 'R': [500, 800]},
    'lentil': {'N': [40, 60], 'P': [20, 40], 'K': [20, 40], 'T': [15, 25], 'H': [60, 70], 'pH': [6.0, 7.5], 'R': [300, 500]},
    'mungbean': {'N': [40, 60], 'P': [20, 40], 'K': [20, 40], 'T': [25, 35], 'H': [60, 70], 'pH': [6.0, 7.5], 'R': [500, 800]},
    'pigeonpeas': {'N': [40, 60], 'P': [20, 40], 'K': [20, 40], 'T': [25, 35], 'H': [60, 70], 'pH': [6.0, 7.5], 'R': [500, 800]},
    'mothbeans': {'N': [40, 60], 'P': [20, 40], 'K': [20, 40], 'T': [25, 35], 'H': [60, 70], 'pH': [6.0, 7.5], 'R': [500, 800]},
    'chickpea': {'N': [40, 60], 'P': [20, 40], 'K': [20, 40], 'T': [15, 25], 'H': [60, 70], 'pH': [6.0, 7.5], 'R': [300, 500]},
    'kidneybeans': {'N': [40, 60], 'P': [20, 40], 'K': [20, 40], 'T': [20, 30], 'H': [60, 70], 'pH': [6.0, 7.5], 'R': [500, 800]}
}

try:
    APP_STATE["model"] = joblib.load(MODEL_PATH)
    APP_STATE["scaler"] = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model files not found. Make sure '{MODEL_PATH}' and '{SCALER_PATH}' exist.")
    APP_STATE["model"] = None
    APP_STATE["scaler"] = None

app = Flask(__name__)
CORS(app)

@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    if not APP_STATE["model"] or not APP_STATE["scaler"]:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data received."}), 400

        features = [
            data.get('nitrogen'),
            data.get('phosphorus'),
            data.get('potassium'),
            data.get('temperature'),
            data.get('humidity'),
            data.get('ph'),
            data.get('rainfall')
        ]

        if any(f is None for f in features):
            return jsonify({"error": "Missing one or more required features (N, P, K, T, H, pH, R)."}), 400

        features_array = np.array([features])
        scaled_features = APP_STATE["scaler"].transform(features_array)
        prediction = APP_STATE["model"].predict(scaled_features)
        
        recommended_crop = prediction[0]

        # Get ideal conditions for the recommended crop, or default to empty
        ideal_conditions = IDEAL_CONDITIONS.get(recommended_crop.lower(), {})

        return jsonify({
            "recommended_crop": recommended_crop,
            "ideal_conditions": ideal_conditions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)