import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Define paths to your data files
landcover_data_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\WorldCover\landcover_data_with_latlon.csv"
merged_data_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\merged_gedi_sentinel1_sentinel2_dem_data.csv"
output_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\final_merged_data_with_landcover.csv"

# Load the datasets
landcover_data = pd.read_csv(landcover_data_path, engine='python')
merged_data = pd.read_csv(merged_data_path)

# Ensure that both datasets contain 'Latitude' and 'Longitude' columns
for dataset, name in [(landcover_data, 'Land Cover'), (merged_data, 'Merged Data')]:
    if 'Latitude' not in dataset.columns or 'Longitude' not in dataset.columns:
        raise ValueError(f"{name} data does not contain 'Latitude' and 'Longitude' columns.")

# Optionally round coordinates to the same precision to avoid floating-point mismatches
rounding_precision = 6
landcover_data['Latitude'] = landcover_data['Latitude'].round(rounding_precision)
landcover_data['Longitude'] = landcover_data['Longitude'].round(rounding_precision)
merged_data['Latitude'] = merged_data['Latitude'].round(rounding_precision)
merged_data['Longitude'] = merged_data['Longitude'].round(rounding_precision)

# Convert coordinates to numpy arrays for spatial operations
landcover_coords = np.array(list(zip(landcover_data['Longitude'], landcover_data['Latitude'])))
merged_coords = np.array(list(zip(merged_data['Longitude'], merged_data['Latitude'])))

# Build a KD-Tree for the landcover coordinates
tree = cKDTree(landcover_coords)

# Find the nearest neighbor in the landcover data for each point in the merged data
distances, indices = tree.query(merged_coords, k=1)

# Select the corresponding rows from the landcover data
nearest_landcover = landcover_data.iloc[indices].reset_index(drop=True)

# Combine the merged data with the nearest landcover data
final_merged_data = pd.concat([merged_data.reset_index(drop=True), nearest_landcover], axis=1)

# Save the final merged data to a CSV file
final_merged_data.to_csv(output_path, index=False)
print(f"Final merged data with Land Cover saved to {output_path}")
