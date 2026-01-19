"""
Google Earth Engine Script to Create Spatial Data for Biomass Estimation

This script extracts GEDI L4A biomass data and Sentinel-2 imagery for a Region of Interest (ROI),
merges them based on geographic proximity, calculates vegetation indices, and exports the result
as a CSV file that can be used for spatial analysis in the dashboard.

Prerequisites:
1. Install Earth Engine API: pip install earthengine-api
2. Authenticate: earthengine authenticate
3. Define your Region of Interest (ROI) coordinates below

Output:
- CSV file: merged_gedi_sentinel2_data_with_indices.csv
- Columns: Longitude_gedi, Latitude_gedi, AGB_L4A, plus Sentinel-2 bands and vegetation indices
"""

import ee
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize Earth Engine
try:
    ee.Initialize()
    print("✅ Earth Engine initialized successfully")
except Exception as e:
    print(f"❌ Earth Engine initialization failed: {e}")
    print("Please run: earthengine authenticate")
    exit(1)

# =============================================================================
# CONFIGURATION - Customize these parameters for your study area
# =============================================================================

# Define your Region of Interest (ROI)
# Example: New Forest, UK
ROI_COORDINATES = [
    [-1.7, 50.8],   # Southwest corner [longitude, latitude]
    [-1.3, 50.8],   # Southeast corner
    [-1.3, 51.0],   # Northeast corner
    [-1.7, 51.0]    # Northwest corner
]

# Time range for data collection
START_DATE = '2019-01-01'
END_DATE = '2019-12-31'

# Cloud cover threshold for Sentinel-2
MAX_CLOUD_COVER = 20  # percent

# Output file path
OUTPUT_FILE = 'merged_gedi_sentinel2_data_with_indices.csv'

# =============================================================================
# FUNCTIONS
# =============================================================================

def create_roi(coordinates):
    """Create Earth Engine geometry from coordinates"""
    return ee.Geometry.Polygon(coordinates)

def mask_s2_clouds(image):
    """Mask clouds in Sentinel-2 imagery using QA band"""
    qa = image.select('QA60')
    
    # Bits 10 and 11 are clouds and cirrus
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    
    # Both flags should be set to zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    
    return image.updateMask(mask).divide(10000)

def add_vegetation_indices(image):
    """Calculate vegetation indices for Sentinel-2 image"""
    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
    
    # NDWI = (Green - NIR) / (Green + NIR)
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    
    # EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
            'BLUE': image.select('B2')
        }).rename('EVI')
    
    # ChlRe (Chlorophyll Red Edge) = (NIR / RedEdge1) - 1
    chlre = image.expression(
        '(NIR / RE1) - 1', {
            'NIR': image.select('B8'),
            'RE1': image.select('B5')
        }).rename('ChlRe')
    
    # NDCI = (RedEdge1 - Red) / (RedEdge1 + Red)
    ndci = image.normalizedDifference(['B5', 'B4']).rename('NDCI')
    
    return image.addBands([ndvi, ndmi, ndwi, evi, chlre, ndci])

def get_sentinel2_composite(roi, start_date, end_date):
    """Get Sentinel-2 median composite for ROI and time range"""
    print(f"📡 Fetching Sentinel-2 data from {start_date} to {end_date}...")
    
    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER)) \
        .map(mask_s2_clouds) \
        .map(add_vegetation_indices) \
        .median()
    
    # Select relevant bands
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
             'NDVI', 'NDMI', 'NDWI', 'EVI', 'ChlRe', 'NDCI']
    
    return s2.select(bands)

def get_gedi_l4a(roi, start_date, end_date):
    """Get GEDI L4A biomass data for ROI"""
    print(f"📡 Fetching GEDI L4A data from {start_date} to {end_date}...")
    
    # Note: MU (Model Uncertainty) ranges 0-1, where 0 = good quality, 1 = high uncertainty
    # We select 'agbd' (Above Ground Biomass Density) and filter for low uncertainty (MU close to 0)
    gedi = ee.ImageCollection('LARSE/GEDI/GEDI04_A_002_MONTHLY') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .select(['MU', 'agbd']) \
        .map(lambda img: img.updateMask(img.select('MU').lt(0.5)))  # Filter for MU < 0.5 (good quality)
    
    # Rename agbd to AGB_L4A for consistency
    gedi = gedi.select(['agbd'], ['AGB_L4A'])
    
    return gedi

def extract_gedi_points(gedi_collection, roi):
    """Extract GEDI points as features with coordinates"""
    print("🎯 Extracting GEDI point locations...")
    
    def extract_point(image):
        """Extract pixel values at each location"""
        # Sample the image at 25m resolution (GEDI footprint size)
        points = image.sample(
            region=roi,
            scale=25,
            geometries=True
        )
        
        # Add timestamp
        return points.map(lambda f: f.set('date', image.date().format()))
    
    # Flatten collection to features
    points = gedi_collection.map(extract_point).flatten()
    
    return points

def merge_gedi_sentinel2(gedi_points, sentinel2_image, roi):
    """Merge GEDI points with Sentinel-2 data"""
    print("🔗 Merging GEDI points with Sentinel-2 data...")
    
    def add_sentinel2_values(feature):
        """Add Sentinel-2 band values to GEDI point"""
        # Get Sentinel-2 values at GEDI point location
        point = feature.geometry()
        s2_values = sentinel2_image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=10  # Sentinel-2 resolution
        )
        
        # Get coordinates (Earth Engine uses [longitude, latitude] order)
        coords = point.coordinates()
        
        return feature.set(s2_values) \
                     .set('Longitude_gedi', coords.get(0)) \
                     .set('Latitude_gedi', coords.get(1))
    
    merged = gedi_points.map(add_sentinel2_values)
    
    return merged

def export_to_dataframe(feature_collection, roi):
    """Convert Earth Engine FeatureCollection to pandas DataFrame"""
    print("📥 Downloading data from Earth Engine...")
    
    # Get the feature collection as a list
    try:
        # Try to get features directly (works for small datasets)
        features = feature_collection.getInfo()['features']
        
        # Extract properties from each feature
        data = []
        for feature in features:
            props = feature['properties']
            # Filter out None values and system properties
            filtered_props = {k: v for k, v in props.items() 
                            if v is not None and not k.startswith('system:')}
            data.append(filtered_props)
        
        df = pd.DataFrame(data)
        
        print(f"✅ Downloaded {len(df)} data points")
        return df
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        print("\n💡 If dataset is too large, consider:")
        print("   1. Reducing the time range")
        print("   2. Reducing the ROI size")
        print("   3. Using Earth Engine export to Drive instead")
        return None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("=" * 80)
    print("🌍 Biomass Estimation - Spatial Data Generator")
    print("=" * 80)
    print(f"\n📍 Region of Interest: {ROI_COORDINATES}")
    print(f"📅 Date Range: {START_DATE} to {END_DATE}")
    print(f"☁️  Max Cloud Cover: {MAX_CLOUD_COVER}%\n")
    
    # Create ROI
    roi = create_roi(ROI_COORDINATES)
    print("✅ ROI created")
    
    # Get Sentinel-2 composite
    sentinel2 = get_sentinel2_composite(roi, START_DATE, END_DATE)
    print("✅ Sentinel-2 composite created")
    
    # Get GEDI L4A data
    gedi = get_gedi_l4a(roi, START_DATE, END_DATE)
    gedi_count = gedi.size().getInfo()
    print(f"✅ Found {gedi_count} GEDI images")
    
    if gedi_count == 0:
        print("⚠️  No GEDI data found for this ROI and time range")
        print("Try adjusting the date range or ROI")
        return
    
    # Extract GEDI points
    gedi_points = extract_gedi_points(gedi, roi)
    point_count = gedi_points.size().getInfo()
    print(f"✅ Extracted {point_count} GEDI points")
    
    if point_count == 0:
        print("⚠️  No GEDI points extracted")
        return
    
    # Merge with Sentinel-2
    merged_data = merge_gedi_sentinel2(gedi_points, sentinel2, roi)
    print("✅ Data merged successfully")
    
    # Export to DataFrame
    df = export_to_dataframe(merged_data, roi)
    
    if df is not None and len(df) > 0:
        # Ensure required columns exist
        required_cols = ['Longitude_gedi', 'Latitude_gedi', 'AGB_L4A']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing required columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return
        
        # Remove rows with missing critical data
        df_clean = df.dropna(subset=required_cols)
        print(f"✅ Cleaned data: {len(df_clean)} points (removed {len(df) - len(df_clean)} points with missing values)")
        
        # Save to CSV
        df_clean.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Data exported successfully to: {OUTPUT_FILE}")
        print(f"📊 Total data points: {len(df_clean)}")
        print(f"📋 Columns: {df_clean.columns.tolist()}")
        print("\n🎉 Spatial data file is ready for use in the dashboard!")
        
        # Display summary statistics
        print("\n📈 Summary Statistics:")
        print(f"   AGB Range: {df_clean['AGB_L4A'].min():.2f} - {df_clean['AGB_L4A'].max():.2f} Mg/ha")
        print(f"   Mean AGB: {df_clean['AGB_L4A'].mean():.2f} Mg/ha")
        print(f"   Latitude Range: {df_clean['Latitude_gedi'].min():.6f} - {df_clean['Latitude_gedi'].max():.6f}")
        print(f"   Longitude Range: {df_clean['Longitude_gedi'].min():.6f} - {df_clean['Longitude_gedi'].max():.6f}")
    else:
        print("❌ Failed to create dataframe")

if __name__ == "__main__":
    main()
