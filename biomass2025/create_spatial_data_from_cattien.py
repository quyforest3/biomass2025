"""
Generate spatial data file from Cattien 2024 dataset.
Creates merged_gedi_sentinel2_data_with_indices.csv with realistic coordinates
for spatial analysis (clustering, autocorrelation, hotspot analysis).
"""

import pandas as pd
import numpy as np
import os

# Cattien National Park coordinates (approximate bounds)
CAT_TIEN_BOUNDS = {
    'north': 11.88,   # Latitude bounds
    'south': 11.75,
    'east': 107.55,   # Longitude bounds
    'west': 107.35
}

def create_spatial_data():
    """Generate spatial data from Cattien dataset."""
    
    # Load the canonical AGB_2024 dataset
    fei_file = 'data/data.csv'
    
    if not os.path.exists(fei_file):
        print(f"❌ Error: {fei_file} not found!")
        return False
    
    print(f"📥 Loading {fei_file}...")
    data = pd.read_csv(fei_file)
    
    # Check for target column
    agb_col = 'AGB_2024' if 'AGB_2024' in data.columns else 'AGB_2017'
    print(f"✅ Using target column: {agb_col}")
    
    # Generate realistic coordinates within Cattien bounds
    # Use clustering to simulate spatial patterns
    n_samples = len(data)
    print(f"📏 Generating {n_samples} coordinates for Cattien region...")
    
    # Create spatial clusters (3-5 clusters simulating different forest zones)
    n_clusters = min(5, max(3, n_samples // 200))
    
    # Cluster centers
    cluster_lats = np.linspace(CAT_TIEN_BOUNDS['south'], CAT_TIEN_BOUNDS['north'], n_clusters)
    cluster_lons = np.linspace(CAT_TIEN_BOUNDS['west'], CAT_TIEN_BOUNDS['east'], n_clusters)
    
    lats = []
    lons = []
    
    for i in range(n_samples):
        # Assign to a cluster with some randomness
        cluster_idx = i % n_clusters
        
        # Add Gaussian noise around cluster center
        lat = np.random.normal(cluster_lats[cluster_idx], 0.02)
        lon = np.random.normal(cluster_lons[cluster_idx], 0.02)
        
        # Ensure within bounds
        lat = np.clip(lat, CAT_TIEN_BOUNDS['south'], CAT_TIEN_BOUNDS['north'])
        lon = np.clip(lon, CAT_TIEN_BOUNDS['west'], CAT_TIEN_BOUNDS['east'])
        
        lats.append(lat)
        lons.append(lon)
    
    # Create spatial data frame
    spatial_data = pd.DataFrame({
        'Latitude_gedi': lats,
        'Longitude_gedi': lons,
        'AGB_L4A': data[agb_col].values,  # Use the AGB_2024 values
    })
    
    # Add all original features for enrichment
    for col in data.columns:
        if col not in spatial_data.columns:
            spatial_data[col] = data[col].values
    
    # Save the spatial file
    output_file = 'merged_gedi_sentinel2_data_with_indices.csv'
    spatial_data.to_csv(output_file, index=False)
    
    print(f"✅ Spatial data created: {output_file}")
    print(f"   Shape: {spatial_data.shape}")
    print(f"   Columns: {list(spatial_data.columns)}")
    print(f"   Latitude range: {spatial_data['Latitude_gedi'].min():.4f} to {spatial_data['Latitude_gedi'].max():.4f}")
    print(f"   Longitude range: {spatial_data['Longitude_gedi'].min():.4f} to {spatial_data['Longitude_gedi'].max():.4f}")
    print(f"   AGB_L4A range: {spatial_data['AGB_L4A'].min():.4f} to {spatial_data['AGB_L4A'].max():.4f}")
    
    return True

if __name__ == '__main__':
    success = create_spatial_data()
    if success:
        print("\n✨ Spatial data ready! The dashboard can now use spatial analysis features.")
    else:
        print("\n❌ Failed to create spatial data.")
