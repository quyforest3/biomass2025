import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np

# Define paths to your data files
gedi_data_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\filtered_latslonsL4A_within_ROI.csv"
sentinel2_data_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel2\sentinel2_cleaned_extracted_data.csv"
sentinel1_data_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel1\sentinel1_cleaned_extracted_data.csv"
output_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\merged_gedi_sentinel1_sentinel2_data.csv"

# Load the GEDI, Sentinel-2, and Sentinel-1 data
gedi_data = pd.read_csv(gedi_data_path)
sentinel2_data = pd.read_csv(sentinel2_data_path)
sentinel1_data = pd.read_csv(sentinel1_data_path)

# Debug: Check if data is loaded correctly
print("GEDI Data Sample:")
print(gedi_data.head())
print("Sentinel-2 Data Sample:")
print(sentinel2_data.head())
print("Sentinel-1 Data Sample:")
print(sentinel1_data.head())

# Ensure all datasets have common columns for merging, e.g., Latitude and Longitude
for dataset, name in [(gedi_data, 'GEDI'), (sentinel2_data, 'Sentinel-2'), (sentinel1_data, 'Sentinel-1')]:
    if 'Latitude' not in dataset.columns or 'Longitude' not in dataset.columns:
        raise ValueError(f"{name} data does not contain 'Latitude' and 'Longitude' columns.")

# Check for missing values
print("Missing values in GEDI data:")
print(gedi_data[['Latitude', 'Longitude']].isnull().sum())

print("Missing values in Sentinel-2 data:")
print(sentinel2_data[['Latitude', 'Longitude']].isnull().sum())

print("Missing values in Sentinel-1 data:")
print(sentinel1_data[['Latitude', 'Longitude']].isnull().sum())

# Check coordinate ranges
print("Coordinate range in GEDI data:")
print("Latitude:", gedi_data['Latitude'].min(), "-", gedi_data['Latitude'].max())
print("Longitude:", gedi_data['Longitude'].min(), "-", gedi_data['Longitude'].max())

print("Coordinate range in Sentinel-2 data:")
print("Latitude:", sentinel2_data['Latitude'].min(), "-", sentinel2_data['Latitude'].max())
print("Longitude:", sentinel2_data['Longitude'].min(), "-", sentinel2_data['Longitude'].max())

print("Coordinate range in Sentinel-1 data:")
print("Latitude:", sentinel1_data['Latitude'].min(), "-", sentinel1_data['Latitude'].max())
print("Longitude:", sentinel1_data['Longitude'].min(), "-", sentinel1_data['Longitude'].max())

# Swap Latitude and Longitude in Sentinel-2 data if needed
# (assuming they are incorrectly labeled)
sentinel2_data = sentinel2_data.rename(columns={"Latitude": "Temp_Latitude", "Longitude": "Latitude"})
sentinel2_data = sentinel2_data.rename(columns={"Temp_Latitude": "Longitude"})

# Optionally round coordinates to the same precision
rounding_precision = 6
gedi_data['Latitude'] = gedi_data['Latitude'].round(rounding_precision)
gedi_data['Longitude'] = gedi_data['Longitude'].round(rounding_precision)
sentinel2_data['Latitude'] = sentinel2_data['Latitude'].round(rounding_precision)
sentinel2_data['Longitude'] = sentinel2_data['Longitude'].round(rounding_precision)
sentinel1_data['Latitude'] = sentinel1_data['Latitude'].round(rounding_precision)
sentinel1_data['Longitude'] = sentinel1_data['Longitude'].round(rounding_precision)

# Convert to GeoDataFrames for spatial operations
gedi_gdf = gpd.GeoDataFrame(gedi_data, geometry=gpd.points_from_xy(gedi_data.Longitude, gedi_data.Latitude))
sentinel2_gdf = gpd.GeoDataFrame(sentinel2_data, geometry=gpd.points_from_xy(sentinel2_data.Longitude, sentinel2_data.Latitude))
sentinel1_gdf = gpd.GeoDataFrame(sentinel1_data, geometry=gpd.points_from_xy(sentinel1_data.Longitude, sentinel1_data.Latitude))

# Merge Sentinel-2 data with GEDI data using nearest neighbor
gedi_coords = np.array(list(zip(gedi_gdf.geometry.x, gedi_gdf.geometry.y)))
sentinel2_coords = np.array(list(zip(sentinel2_gdf.geometry.x, sentinel2_gdf.geometry.y)))

tree_sentinel2 = cKDTree(sentinel2_coords)
distances_sentinel2, indices_sentinel2 = tree_sentinel2.query(gedi_coords, k=1)

nearest_sentinel2 = sentinel2_gdf.iloc[indices_sentinel2].reset_index(drop=True)
merged_data = gedi_gdf.reset_index(drop=True).join(nearest_sentinel2, lsuffix='_gedi', rsuffix='_sentinel2')

# Debug: Check if GEDI-Sentinel2 data is merged correctly
print("Merged GEDI-Sentinel2 Data Sample:")
print(merged_data.head())
print("Number of samples in merged GEDI-Sentinel2 data:", merged_data.shape[0])

# Set the active geometry to use for merging with Sentinel-1
merged_data = merged_data.set_geometry('geometry_gedi')

# Merge Sentinel-1 data with the already merged GEDI-Sentinel2 data using nearest neighbor
merged_coords = np.array(list(zip(merged_data.geometry.x, merged_data.geometry.y)))
sentinel1_coords = np.array(list(zip(sentinel1_gdf.geometry.x, sentinel1_gdf.geometry.y)))

tree_sentinel1 = cKDTree(sentinel1_coords)
distances_sentinel1, indices_sentinel1 = tree_sentinel1.query(merged_coords, k=1)

nearest_sentinel1 = sentinel1_gdf.iloc[indices_sentinel1].reset_index(drop=True)
final_merged_data = merged_data.reset_index(drop=True).join(nearest_sentinel1, lsuffix='_merged', rsuffix='_sentinel1')

# Debug: Check if final data is merged correctly
print("Final Merged Data Sample:")
print(final_merged_data.head())
print("Number of samples in final merged data:", final_merged_data.shape[0])

# Check if merged data is empty
if final_merged_data.empty:
    raise ValueError("The final merged dataset is empty. Please check the merging criteria and data integrity.")

# Save the final merged data to a CSV file
final_merged_data.to_csv(output_path, index=False)
print(f"Final merged data saved to {output_path}")
