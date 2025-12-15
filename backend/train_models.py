import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# 🌾 Generate dummy satellite feature data
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    'B2_Blue': np.random.uniform(100, 2000, n_samples),
    'B3_Green': np.random.uniform(100, 2000, n_samples),
    'B4_Red': np.random.uniform(100, 2000, n_samples),
    'B8_NIR': np.random.uniform(100, 3000, n_samples),
    'B11_SWIR1': np.random.uniform(100, 3000, n_samples),
    'B12_SWIR2': np.random.uniform(100, 3000, n_samples),
    'NDVI': np.random.uniform(0, 1, n_samples),
    'NDMI': np.random.uniform(-1, 1, n_samples),
    'SAVI': np.random.uniform(0, 1, n_samples),
    'BSI': np.random.uniform(-1, 1, n_samples),
})

# 🌱 Generate dummy nutrient levels
data['N'] = 50 + 10 * data['NDVI'] + np.random.normal(0, 2, n_samples)
data['P'] = 30 + 5 * data['SAVI'] + np.random.normal(0, 1, n_samples)
data['K'] = 100 + 8 * data['BSI'] + np.random.normal(0, 3, n_samples)

X = data[['B2_Blue', 'B3_Green', 'B4_Red', 'B8_NIR', 'B11_SWIR1',
           'B12_SWIR2', 'NDVI', 'NDMI', 'SAVI', 'BSI']]
y_N, y_P, y_K = data['N'], data['P'], data['K']

# 🎯 Train models
models = {
    'soil_nitrogen_model.joblib': RandomForestRegressor(n_estimators=100, random_state=42),
    'soil_phosphorus_model.joblib': RandomForestRegressor(n_estimators=100, random_state=42),
    'soil_potassium_model.joblib': RandomForestRegressor(n_estimators=100, random_state=42),
}

# Train each model
models['soil_nitrogen_model.joblib'].fit(X, y_N)
models['soil_phosphorus_model.joblib'].fit(X, y_P)
models['soil_potassium_model.joblib'].fit(X, y_K)

# 💾 Save models in models/ folder
os.makedirs("models", exist_ok=True)

for name, model in models.items():
    joblib.dump(model, f"models/{name}")
    print(f"✅ Saved {name}")

print("\n🎉 All models trained and saved successfully!")
