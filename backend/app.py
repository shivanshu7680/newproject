from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from ee_processor import get_satellite_data

app = Flask(__name__)
CORS(app)

# Load models
print("🧠 Loading AI models...")
n_model = joblib.load('models/soil_nitrogen_model.joblib')
p_model = joblib.load('models/soil_phosphorus_model.joblib')
k_model = joblib.load('models/soil_potassium_model.joblib')
print("✅ Models loaded.")

def get_fertilizer_suggestion(N, P, K):
    suggestion = {
        "Nitrogen": "Sufficient Nitrogen level. No extra fertilizer needed.",
        "Phosphorus": "Sufficient Phosphorus level.",
        "Potassium": "Sufficient Potassium level."
    }
    if N < 50:
        suggestion["Nitrogen"] = "Add Urea or Ammonium Sulphate to increase Nitrogen."
    elif N > 200:
        suggestion["Nitrogen"] = "Avoid Nitrogen fertilizers – excess detected."

    if P < 30:
        suggestion["Phosphorus"] = "Add DAP or Single Super Phosphate to increase Phosphorus."
    elif P > 100:
        suggestion["Phosphorus"] = "Avoid Phosphorus fertilizers."

    if K < 40:
        suggestion["Potassium"] = "Add MOP (Muriate of Potash) to increase Potassium."
    elif K > 150:
        suggestion["Potassium"] = "Avoid Potassium fertilizers – excess detected."

    return suggestion

@app.route('/predict', methods=['POST'])
def predict_nutrients():
    try:
        data = request.get_json()
        lon = float(data['lon'])
        lat = float(data['lat'])
        print(f"📍 Predicting for {lat}, {lon}")

        df = get_satellite_data(lon, lat)
        df = df[n_model.feature_names_in_]

        n_pred = n_model.predict(df)
        p_pred = p_model.predict(df)
        k_pred = k_model.predict(df)

        N_avg, P_avg, K_avg = map(np.mean, [n_pred, p_pred, k_pred])

        suggestions = get_fertilizer_suggestion(N_avg, P_avg, K_avg)

        return jsonify({
            "Nitrogen_avg": float(N_avg),
            "Phosphorus_avg": float(P_avg),
            "Potassium_avg": float(K_avg),
            "Suggestions": suggestions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
