import ee
import os
import numpy as np
import pandas as pd

# -------------------------------
# 🌍 Earth Engine Initialization
# -------------------------------

PROJECT_ID = 'calm-acre-472904-c1'

# Ensure service account JSON path is set (local or server)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\shiva\Downloads\Minor Project\backend\ee-service-account.json"

# Initialize EE with project explicitly
try:
    ee.Initialize(project=PROJECT_ID)
    print(f"✅ Earth Engine initialized with project: {PROJECT_ID}")
except Exception as e:
    print("❌ Failed to initialize Earth Engine:", e)
    raise e

# -------------------------------
# 🛰 Functions to process satellite data
# -------------------------------

def add_indices(image):
    """Add vegetation and soil indices to an image."""
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
    """
    Fetch satellite data (Sentinel-2) for a given longitude and latitude.
    
    Returns a pandas DataFrame with spectral bands and indices.
    """
    # Define region
    geom = ee.Geometry.Rectangle([
        lon - box_size / 2, lat - box_size / 2,
        lon + box_size / 2, lat + box_size / 2
    ])

    # Filter Sentinel-2 image collection
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(geom)
           .filterDate('2023-03-01', '2023-05-30')
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
           .map(add_indices))

    # Take median composite
    img = col.median().reproject(crs='EPSG:4326', scale=200)

    # Bands to extract
    bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDMI', 'SAVI', 'BSI']
    data = img.sampleRectangle(region=geom, defaultValue=0)

    # Convert to numpy arrays
    arrays = [np.array(data.get(b).getInfo()) for b in bands]

    # Create DataFrame
    df = pd.DataFrame(np.transpose([a.flatten() for a in arrays]),
                      columns=['B2_Blue', 'B3_Green', 'B4_Red', 'B8_NIR',
                               'B11_SWIR1', 'B12_SWIR2', 'NDVI', 'NDMI', 'SAVI', 'BSI'])

    # Clean NaN or infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df
