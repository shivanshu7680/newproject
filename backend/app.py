from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import ee
import sys

# ✅ backend folder ko python path me add karo
sys.path.append(os.path.dirname(__file__))

from ee_processor import get_satellite_data

app = Flask(__name__)
CORS(app)

# ✅ Initialize Google Earth Engine
ee.Initialize()

# ✅ Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Load models
print("🧠 Loading AI models...")

n_model_path = os.path.join(BASE_DIR, "models", "soil_nitrogen_model.joblib")
p_model_path = os.path.join(BASE_DIR, "models", "soil_phosphorus_model.joblib")
k_model_path = os.path.join(BASE_DIR, "models", "soil_potassium_model.joblib")

n_model = joblib.load(n_model_path)
p_model = joblib.load(p_model_path)
k_model = joblib.load(k_model_path)

print("✅ Models loaded successfully.")

def get_fertilizer_suggestion(N, P, K):
    suggestion = {
        "Nitrogen": "Sufficient Nitrogen level.",
        "Phosphorus": "Sufficient Phosphorus level.",
        "Potassium": "Sufficient Potassium level."
    }

    if N < 50:
        suggestion["Nitrogen"] = "Add Urea or Ammonium Sulphate."
    elif N > 200:
        suggestion["Nitrogen"] = "Avoid Nitrogen fertilizers – excess detected."

    if P < 30:
        suggestion["Phosphorus"] = "Add DAP or Single Super Phosphate."
    elif P > 100:
        suggestion["Phosphorus"] = "Avoid Phosphorus fertilizers."

    if K < 40:
        suggestion["Potassium"] = "Add MOP (Muriate of Potash)."
    elif K > 150:
        suggestion["Potassium"] = "Avoid Potassium fertilizers – excess detected."

    return suggestion


@app.route("/predict", methods=["POST"])
def predict_nutrients():
    try:
        data = request.get_json()

        lon = float(data["lon"])
        lat = float(data["lat"])
        print(f"📍 Predicting for Latitude={lat}, Longitude={lon}")

        df = get_satellite_data(lon, lat)
        df = df[n_model.feature_names_in_]

        n_pred = n_model.predict(df)
        p_pred = p_model.predict(df)
        k_pred = k_model.predict(df)

        N_avg, P_avg, K_avg = map(np.mean, [n_pred, p_pred, k_pred])

        return jsonify({
            "Nitrogen_avg": float(N_avg),
            "Phosphorus_avg": float(P_avg),
            "Potassium_avg": float(K_avg),
            "Suggestions": get_fertilizer_suggestion(N_avg, P_avg, K_avg)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/")
def home():
    return "✅ MINO Soil Nutrient API is running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
