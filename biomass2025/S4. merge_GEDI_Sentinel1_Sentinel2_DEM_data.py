import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np

# Define paths to your data files
gedi_data_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\filtered_latslonsL4A_within_ROI_quality_1.csv"
sentinel2_data_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel2\sentinel2_cleaned_extracted_data.csv"
sentinel1_data_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel1\sentinel1_cleaned_extracted_data.csv"
dem_data_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\DEM\dem_cleaned_extended_data.csv"
output_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\merged_gedi_sentinel1_sentinel2_dem_data.csv"

# Load the GEDI, Sentinel-2, Sentinel-1, and DEM data
gedi_data = pd.read_csv(gedi_data_path)
sentinel2_data = pd.read_csv(sentinel2_data_path)
sentinel1_data = pd.read_csv(sentinel1_data_path)
dem_data = pd.read_csv(dem_data_path)

# Debug: Check if data is loaded correctly
print("GEDI Data Sample:")
print(gedi_data.head())
print("Sentinel-2 Data Sample:")
print(sentinel2_data.head())
print("Sentinel-1 Data Sample:")
print(sentinel1_data.head())
print("DEM Data Sample:")
print(dem_data.head())

# Ensure all datasets have common columns for merging, e.g., Latitude and Longitude
for dataset, name in [(gedi_data, 'GEDI'), (sentinel2_data, 'Sentinel-2'), (sentinel1_data, 'Sentinel-1'), (dem_data, 'DEM')]:
    if 'Latitude' not in dataset.columns or 'Longitude' not in dataset.columns:
        raise ValueError(f"{name} data does not contain 'Latitude' and 'Longitude' columns.")

# Check for missing values
for dataset, name in [(gedi_data, 'GEDI'), (sentinel2_data, 'Sentinel-2'), (sentinel1_data, 'Sentinel-1'), (dem_data, 'DEM')]:
    print(f"Missing values in {name} data:")
    print(dataset[['Latitude', 'Longitude']].isnull().sum())

# Check coordinate ranges
for dataset, name in [(gedi_data, 'GEDI'), (sentinel2_data, 'Sentinel-2'), (sentinel1_data, 'Sentinel-1'), (dem_data, 'DEM')]:
    print(f"Coordinate range in {name} data:")
    print("Latitude:", dataset['Latitude'].min(), "-", dataset['Latitude'].max())
    print("Longitude:", dataset['Longitude'].min(), "-", dataset['Longitude'].max())

# Swap Latitude and Longitude in Sentinel-2 data if needed
# (assuming they are incorrectly labeled)
sentinel2_data = sentinel2_data.rename(columns={"Latitude": "Temp_Latitude", "Longitude": "Latitude"})
sentinel2_data = sentinel2_data.rename(columns={"Temp_Latitude": "Longitude"})

# Optionally round coordinates to the same precision
rounding_precision = 6
for dataset in [gedi_data, sentinel2_data, sentinel1_data, dem_data]:
    dataset['Latitude'] = dataset['Latitude'].round(rounding_precision)
    dataset['Longitude'] = dataset['Longitude'].round(rounding_precision)

# Convert to GeoDataFrames for spatial operations
gedi_gdf = gpd.GeoDataFrame(gedi_data, geometry=gpd.points_from_xy(gedi_data.Longitude, gedi_data.Latitude))
sentinel2_gdf = gpd.GeoDataFrame(sentinel2_data, geometry=gpd.points_from_xy(sentinel2_data.Longitude, sentinel2_data.Latitude))
sentinel1_gdf = gpd.GeoDataFrame(sentinel1_data, geometry=gpd.points_from_xy(sentinel1_data.Longitude, sentinel1_data.Latitude))
dem_gdf = gpd.GeoDataFrame(dem_data, geometry=gpd.points_from_xy(dem_data.Longitude, dem_data.Latitude))

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

# Debug: Check if GEDI-Sentinel1-Sentinel2 data is merged correctly
print("Merged GEDI-Sentinel1-Sentinel2 Data Sample:")
print(final_merged_data.head())
print("Number of samples in merged GEDI-Sentinel1-Sentinel2 data:", final_merged_data.shape[0])

# Set the active geometry to use for merging with DEM
final_merged_data = final_merged_data.set_geometry('geometry_gedi')

# Merge DEM data using nearest neighbor
merged_coords_dem = np.array(list(zip(final_merged_data.geometry.x, final_merged_data.geometry.y)))
dem_coords = np.array(list(zip(dem_gdf.geometry.x, dem_gdf.geometry.y)))

tree_dem = cKDTree(dem_coords)
distances_dem, indices_dem = tree_dem.query(merged_coords_dem, k=1)

nearest_dem = dem_gdf.iloc[indices_dem].reset_index(drop=True)
final_merged_with_dem = final_merged_data.reset_index(drop=True).join(nearest_dem, lsuffix='_merged', rsuffix='_dem')

# Debug: Check if final data with DEM is merged correctly
print("Final Merged Data with DEM Sample:")
print(final_merged_with_dem.head())
print("Number of samples in final merged data with DEM:", final_merged_with_dem.shape[0])

# Check if merged data is empty
if final_merged_with_dem.empty:
    raise ValueError("The final merged dataset with DEM is empty. Please check the merging criteria and data integrity.")

# Save the final merged data to a CSV file
final_merged_with_dem.to_csv(output_path, index=False)
print(f"Final merged data with DEM saved to {output_path}")
