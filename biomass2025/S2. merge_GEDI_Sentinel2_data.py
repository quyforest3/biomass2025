import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np

# Define paths to your data files
gedi_data_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\filtered_latslonsL4A_within_ROI_quality_1.csv"
sentinel2_data_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel2\sentinel2_cleaned_extracted_data.csv"

# Load the GEDI and Sentinel-2 data
gedi_data = pd.read_csv(gedi_data_path)
sentinel2_data = pd.read_csv(sentinel2_data_path)

# Debug: Check if data is loaded correctly
print("GEDI Data Sample:")
print(gedi_data.head())
print("Sentinel-2 Data Sample:")
print(sentinel2_data.head())

# Ensure both datasets have common columns for merging, e.g., Latitude and Longitude
if 'Latitude' not in gedi_data.columns or 'Longitude' not in gedi_data.columns:
    raise ValueError("GEDI data does not contain 'Latitude' and 'Longitude' columns.")

if 'Latitude' not in sentinel2_data.columns or 'Longitude' not in sentinel2_data.columns:
    raise ValueError("Sentinel-2 data does not contain 'Latitude' and 'Longitude' columns.")

# Check for missing values
print("Missing values in GEDI data:")
print(gedi_data[['Latitude', 'Longitude']].isnull().sum())

print("Missing values in Sentinel-2 data:")
print(sentinel2_data[['Latitude', 'Longitude']].isnull().sum())

# Check coordinate ranges
print("Coordinate range in GEDI data:")
print("Latitude:", gedi_data['Latitude'].min(), "-", gedi_data['Latitude'].max())
print("Longitude:", gedi_data['Longitude'].min(), "-", gedi_data['Longitude'].max())

print("Coordinate range in Sentinel-2 data:")
print("Latitude:", sentinel2_data['Latitude'].min(), "-", sentinel2_data['Latitude'].max())
print("Longitude:", sentinel2_data['Longitude'].min(), "-", sentinel2_data['Longitude'].max())

# Swap Latitude and Longitude in Sentinel-2 data
sentinel2_data = sentinel2_data.rename(columns={"Latitude": "Temp_Latitude", "Longitude": "Latitude"})
sentinel2_data = sentinel2_data.rename(columns={"Temp_Latitude": "Longitude"})

# Debug: Check swapped coordinate ranges
print("Corrected coordinate range in Sentinel-2 data:")
print("Latitude:", sentinel2_data['Latitude'].min(), "-", sentinel2_data['Latitude'].max())
print("Longitude:", sentinel2_data['Longitude'].min(), "-", sentinel2_data['Longitude'].max())

# Optionally round coordinates to the same precision
gedi_data['Latitude'] = gedi_data['Latitude'].round(6)
gedi_data['Longitude'] = gedi_data['Longitude'].round(6)
sentinel2_data['Latitude'] = sentinel2_data['Latitude'].round(6)
sentinel2_data['Longitude'] = sentinel2_data['Longitude'].round(6)

# Print coordinate samples to debug potential issues
print("Sample GEDI coordinates:")
print(gedi_data[['Latitude', 'Longitude']].head())
print("Sample Sentinel-2 coordinates:")
print(sentinel2_data[['Latitude', 'Longitude']].head())

# Convert to GeoDataFrames for spatial operations
gedi_gdf = gpd.GeoDataFrame(gedi_data, geometry=gpd.points_from_xy(gedi_data.Longitude, gedi_data.Latitude))
sentinel2_gdf = gpd.GeoDataFrame(sentinel2_data, geometry=gpd.points_from_xy(sentinel2_data.Longitude, sentinel2_data.Latitude))

# Use cKDTree for efficient nearest neighbor search
gedi_coords = np.array(list(zip(gedi_gdf.geometry.x, gedi_gdf.geometry.y)))
sentinel2_coords = np.array(list(zip(sentinel2_gdf.geometry.x, sentinel2_gdf.geometry.y)))

tree = cKDTree(sentinel2_coords)
distances, indices = tree.query(gedi_coords, k=1)

# Join the nearest neighbors
nearest_sentinel2 = sentinel2_gdf.iloc[indices].reset_index(drop=True)
merged_data = gedi_gdf.reset_index(drop=True).join(nearest_sentinel2, lsuffix='_gedi', rsuffix='_sentinel2')

# Debug: Check if data is merged correctly
print("Merged Data Sample:")
print(merged_data.head())
print("Number of samples in merged data:", merged_data.shape[0])

# Check if merged data is empty
if merged_data.empty:
    raise ValueError("The merged dataset is empty. Please check the merging criteria and data integrity.")

# Save the merged data to a CSV file
output_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\merged_gedi_sentinel2_data.csv"
merged_data.to_csv(output_path, index=False)
print(f"Merged data saved to {output_path}")

