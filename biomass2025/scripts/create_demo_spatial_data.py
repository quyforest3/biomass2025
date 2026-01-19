#!/usr/bin/env python3
"""
Demo Spatial Data Generator for Biomass Estimation

This script creates a sample spatial data file for testing the Spatial Analysis feature
without needing Google Earth Engine access. 

The generated data is synthetic but follows the expected structure and format.
For production use, please use the GEE script: scripts/create_spatial_data_gee.py

Usage:
    python scripts/create_demo_spatial_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime

def generate_demo_spatial_data(n_points=200, region='new_forest_uk', random_seed=42):
    """
    Generate synthetic spatial data for testing
    
    Parameters:
    -----------
    n_points : int
        Number of spatial points to generate
    region : str
        Region identifier for coordinate generation
    random_seed : int
        Random seed for reproducibility (default: 42). Set to None for random data.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with spatial data
    """
    print("=" * 80)
    print("🎲 Demo Spatial Data Generator for Biomass Estimation")
    print("=" * 80)
    print(f"\nGenerating {n_points} synthetic data points for testing...")
    print(f"Region: {region}")
    print(f"Random seed: {random_seed}")
    print("\n⚠️  Note: This is DEMO data for testing only!")
    print("For real analysis, use: scripts/create_spatial_data_gee.py\n")
    
    # Define region bounds (New Forest, UK as example)
    regions = {
        'new_forest_uk': {
            'lat_range': (50.80, 50.95),
            'lon_range': (-1.70, -1.35),
            'agb_mean': 150,
            'agb_std': 60
        },
        'amazon': {
            'lat_range': (-3.5, -3.0),
            'lon_range': (-60.5, -60.0),
            'agb_mean': 300,
            'agb_std': 100
        }
    }
    
    region_config = regions.get(region, regions['new_forest_uk'])
    
    # Generate random coordinates within the region
    if random_seed is not None:
        np.random.seed(random_seed)  # For reproducibility
    
    latitudes = np.random.uniform(
        region_config['lat_range'][0],
        region_config['lat_range'][1],
        n_points
    )
    
    longitudes = np.random.uniform(
        region_config['lon_range'][0],
        region_config['lon_range'][1],
        n_points
    )
    
    # Generate AGB values with spatial correlation
    # Create some spatial patterns by adding distance-based effects
    center_lat = np.mean(region_config['lat_range'])
    center_lon = np.mean(region_config['lon_range'])
    
    # Calculate distance from center
    distances = np.sqrt((latitudes - center_lat)**2 + (longitudes - center_lon)**2)
    
    # Generate base AGB with normal distribution
    base_agb = np.random.normal(
        region_config['agb_mean'],
        region_config['agb_std'],
        n_points
    )
    
    # Add spatial correlation (higher biomass towards center)
    spatial_effect = -50 * distances / distances.max()
    agb_values = base_agb + spatial_effect
    
    # Add some clusters of high/low biomass
    # Create 3 hotspots
    n_hotspots = 3
    for i in range(n_hotspots):
        hotspot_lat = np.random.uniform(region_config['lat_range'][0], region_config['lat_range'][1])
        hotspot_lon = np.random.uniform(region_config['lon_range'][0], region_config['lon_range'][1])
        hotspot_strength = np.random.uniform(50, 100)
        
        # Points near hotspot get higher biomass
        hotspot_distances = np.sqrt((latitudes - hotspot_lat)**2 + (longitudes - hotspot_lon)**2)
        near_hotspot = hotspot_distances < 0.05  # Within ~5km
        agb_values[near_hotspot] += hotspot_strength
    
    # Ensure AGB values are positive and reasonable
    agb_values = np.clip(agb_values, 10, 500)
    
    # Generate Sentinel-2 bands (correlated with AGB)
    # Higher biomass typically means higher NIR and lower red reflectance
    ndvi_base = 0.3 + (agb_values / region_config['agb_mean']) * 0.4
    ndvi_base = np.clip(ndvi_base, 0.1, 0.9)
    
    # Generate spectral bands based on typical vegetation reflectance
    data = {
        'Longitude_gedi': longitudes,
        'Latitude_gedi': latitudes,
        'AGB_L4A': agb_values,
        
        # Sentinel-2 bands (10000 scale typical for surface reflectance)
        'B2': np.random.uniform(0.03, 0.08, n_points),    # Blue
        'B3': np.random.uniform(0.04, 0.10, n_points),    # Green
        'B4': np.random.uniform(0.03, 0.08, n_points),    # Red
        'B5': np.random.uniform(0.10, 0.20, n_points),    # Red Edge 1
        'B6': np.random.uniform(0.15, 0.25, n_points),    # Red Edge 2
        'B7': np.random.uniform(0.20, 0.30, n_points),    # Red Edge 3
        'B8': np.random.uniform(0.30, 0.50, n_points),    # NIR
        'B8A': np.random.uniform(0.30, 0.50, n_points),   # Narrow NIR
        'B11': np.random.uniform(0.20, 0.35, n_points),   # SWIR 1
        'B12': np.random.uniform(0.15, 0.30, n_points),   # SWIR 2
        
        # Vegetation indices
        'NDVI': ndvi_base + np.random.normal(0, 0.05, n_points),
        'NDMI': np.random.uniform(0.2, 0.6, n_points),
        'NDWI': np.random.uniform(-0.3, 0.3, n_points),
        'EVI': np.random.uniform(0.2, 0.8, n_points),
        'ChlRe': np.random.uniform(1.0, 3.0, n_points),
        'NDCI': np.random.uniform(-0.2, 0.3, n_points),
    }
    
    df = pd.DataFrame(data)
    
    # Add quality indicators
    df['quality'] = 1  # All high quality for demo
    df['date'] = datetime.now().strftime('%Y-%m-%d')
    
    return df

def main():
    """Main execution function"""
    
    # Generate demo data
    df = generate_demo_spatial_data(n_points=200, region='new_forest_uk')
    
    # Save to CSV
    output_file = 'merged_gedi_sentinel2_data_with_indices.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Demo spatial data generated: {output_file}")
    print(f"📊 Total points: {len(df)}")
    print(f"📋 Columns: {', '.join(df.columns)}")
    
    # Display summary statistics
    print("\n📈 Summary Statistics:")
    print(f"   Latitude Range: {df['Latitude_gedi'].min():.6f} - {df['Latitude_gedi'].max():.6f}")
    print(f"   Longitude Range: {df['Longitude_gedi'].min():.6f} - {df['Longitude_gedi'].max():.6f}")
    print(f"   AGB Range: {df['AGB_L4A'].min():.2f} - {df['AGB_L4A'].max():.2f} Mg/ha")
    print(f"   Mean AGB: {df['AGB_L4A'].mean():.2f} Mg/ha")
    print(f"   Std AGB: {df['AGB_L4A'].std():.2f} Mg/ha")
    print(f"   Mean NDVI: {df['NDVI'].mean():.3f}")
    
    print("\n🎉 Demo file ready for testing!")
    print("\n⚠️  Remember: This is synthetic data for testing only!")
    print("For real spatial analysis, use scripts/create_spatial_data_gee.py")
    print("\n📍 You can now run the Streamlit dashboard and test the Spatial Analysis feature.")

if __name__ == "__main__":
    main()
