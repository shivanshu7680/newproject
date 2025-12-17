from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import sys
import json
import ee
import pandas as pd

# -------------------------------
# Setup
# -------------------------------
app = Flask(__name__)

# Allow requests only from your Vercel frontend
CORS(app, origins=["https://your-vercel-frontend.vercel.app"])

# Add backend folder to path if needed
sys.path.append(os.path.dirname(__file__))

# -------------------------------
# Earth Engine Initialization
# -------------------------------
PROJECT_ID = os.environ.get("EE_PROJECT_ID", "your-default-project-id")

def init_ee():
    try:
        if ee.data._initialized:
            return

        service_account = os.environ["EE_SERVICE_ACCOUNT"]
        key_json = json.loads(os.environ["EE_PRIVATE_KEY"])

        credentials = ee.ServiceAccountCredentials(
            service_account, key_data=key_json
        )

        ee.Initialize(credentials, project=PROJECT_ID)
        print(f"✅ Earth Engine initialized: {PROJECT_ID}")

    except Exception as e:
        print("❌ Earth Engine init failed:", e)
        raise

# -------------------------------
# Load Models Globally
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

n_model = joblib.load(os.path.join(BASE_DIR, "models", "soil_nitrogen_model.joblib"))
p_model = joblib.load(os.path.join(BASE_DIR, "models", "soil_phosphorus_model.joblib"))
k_model = joblib.load(os.path.join(BASE_DIR, "models", "soil_potassium_model.joblib"))

print("✅ Models loaded successfully.")

# -------------------------------
# Fertilizer Suggestion Function
# -------------------------------
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

# -------------------------------
# Satellite Data Function
# -------------------------------
def mask_clouds(img):
    qa = img.select("QA60")
    cloud = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus = qa.bitwiseAnd(1 << 11).eq(0)
    return img.updateMask(cloud.And(cirrus))

def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
    savi = image.expression(
        '((NIR - RED) * (1 + L)) / (NIR + RED + L)',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'L': 0.5}
    ).rename('SAVI')
    bsi = image.expression(
        '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
        {'SWIR1': image.select('B11'), 'RED': image.select('B4'),
         'NIR': image.select('B8'), 'BLUE': image.select('B2')}
    ).rename('BSI')
    return image.addBands([ndvi, ndmi, savi, bsi])

def get_satellite_data(lon, lat, box_size=0.1):
    """Return Sentinel-2 stats as pandas DataFrame"""
    init_ee()

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Invalid latitude / longitude")

    geom = ee.Geometry.Rectangle([
        lon - box_size / 2, lat - box_size / 2,
        lon + box_size / 2, lat + box_size / 2
    ])

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate("2023-03-01", "2023-05-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))
        .map(mask_clouds)
        .map(add_indices)
    )

    img = col.median()
    bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDMI', 'SAVI', 'BSI']

    stats = img.select(bands).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=200,
        maxPixels=1e8
    ).getInfo()

    df = pd.DataFrame([stats])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

# -------------------------------
# API Routes
# -------------------------------
@app.route("/")
def home():
    return "✅ MINO Soil Nutrient API is running"

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

        N_avg = float(np.mean(n_pred))
        P_avg = float(np.mean(p_pred))
        K_avg = float(np.mean(k_pred))

        return jsonify({
            "Nitrogen_avg": N_avg,
            "Phosphorus_avg": P_avg,
            "Potassium_avg": K_avg,
            "Suggestions": get_fertilizer_suggestion(N_avg, P_avg, K_avg)
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": str(e)}), 400

# -------------------------------
# Run locally
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
