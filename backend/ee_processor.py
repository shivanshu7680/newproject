import ee
import os
import json
import numpy as np
import pandas as pd

# -------------------------------
# 🌍 Earth Engine Initialization
# -------------------------------

PROJECT_ID = "calm-acre-472904-c1"

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
# ☁ Cloud Mask
# -------------------------------

def mask_clouds(img):
    qa = img.select("QA60")
    cloud = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus = qa.bitwiseAnd(1 << 11).eq(0)
    return img.updateMask(cloud.And(cirrus))


# -------------------------------
# 🛰 Indices
# -------------------------------

def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')

    savi = image.expression(
        '((NIR - RED) * (1 + L)) / (NIR + RED + L)',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'L': 0.5}
    ).rename('SAVI')

    bsi = image.expression(
        '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
        {
            'SWIR1': image.select('B11'),
            'RED': image.select('B4'),
            'NIR': image.select('B8'),
            'BLUE': image.select('B2')
        }
    ).rename('BSI')

    return image.addBands([ndvi, ndmi, savi, bsi])


# -------------------------------
# 📡 Main Function
# -------------------------------

def get_satellite_data(lon, lat, box_size=0.1):
    """Return Sentinel-2 stats as pandas DataFrame"""

    # 🔐 Safe EE init
    init_ee()

    # ✅ Coordinate validation
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
