import os
import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import xy

# Define the directory containing Sentinel-1 TIF files and output directory
input_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Sen1\EarthEngineExports-20240725T125257Z-001\EarthEngineExports"
output_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel1"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all TIF files in the directory
tif_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')]

# Create an empty list to store the data
sentinel1_data_list = []

# Loop through each TIF file
for tif_file in tif_files:
    with rasterio.open(tif_file) as src:
        # Read VV and VH bands
        vv = src.read(1)  # Assuming VV is band 1
        vh = src.read(2)  # Assuming VH is band 2
        
        # Remove non-positive values by setting them to NaN
        vv_masked = np.where(vv > 0, vv, np.nan)
        vh_masked = np.where(vh > 0, vh, np.nan)
        
        # Skip this file if all values are non-positive
        if np.isnan(vv_masked).all() or np.isnan(vh_masked).all():
            print(f"Skipped file {tif_file} due to all non-positive values.")
            continue
        
        # Convert to dB scale (logarithmic scale)
        vv_db = 10 * np.log10(vv_masked)
        vh_db = 10 * np.log10(vh_masked)

        # Calculate VV/VH ratio
        vv_vh_ratio = vv_db / vh_db
        
        # Get the transform for converting pixel coordinates to lat/lon
        transform = src.transform

        # Create a mask for valid values (where neither band is NaN)
        valid_mask = ~np.isnan(vv_db) & ~np.isnan(vh_db) & ~np.isnan(vv_vh_ratio)

        # Get the indices of valid values
        valid_indices = np.where(valid_mask)

        # Convert pixel indices to lat/lon
        lats, lons = xy(transform, valid_indices[1], valid_indices[0], offset='center')

        # Extract the valid data
        valid_vv = vv_db[valid_indices]
        valid_vh = vh_db[valid_indices]
        valid_ratio = vv_vh_ratio[valid_indices]

        # Combine data into a DataFrame
        data = np.column_stack((lats, lons, valid_vv, valid_vh, valid_ratio, valid_indices[1], valid_indices[0]))
        sentinel1_data_list.append(pd.DataFrame(data, columns=['Latitude', 'Longitude', 'VV Intensity (dB)', 'VH Intensity (dB)', 'VV/VH Ratio', 'Pixel X', 'Pixel Y']))

# Concatenate all DataFrames
if sentinel1_data_list:
    sentinel1_cleaned_df = pd.concat(sentinel1_data_list, ignore_index=True)

    # Save the DataFrame to a CSV file
    output_csv = os.path.join(output_dir, 'sentinel1_cleaned_extracted_data.csv')
    sentinel1_cleaned_df.to_csv(output_csv, index=False)

    print(f"Data extracted and saved to {output_csv}")
else:
    print("No valid data was found in the provided files.")
