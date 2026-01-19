import os
import rasterio
import pandas as pd
import numpy as np

# Define input and output directories
input_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Sen2"
output_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel2"
os.makedirs(output_dir, exist_ok=True)

# List of files in the input directory
sentinel2_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')]

# Function to extract data from a single Sentinel-2 file
def extract_sentinel2_data(file_path):
    with rasterio.open(file_path) as src:
        # Initialize the DataFrame
        df_sentinel2 = pd.DataFrame()

        # Get the coordinates (row, col) and corresponding bands for each pixel
        for band_index in range(1, src.count + 1):
            band_data = src.read(band_index)
            
            # Replace no-data values with NaN
            no_data_value = src.nodatavals[band_index - 1]
            band_data = np.where(band_data == no_data_value, np.nan, band_data)
            
            # Apply the non-positive value check
            band_data = np.where(band_data > 0, band_data, np.nan)
            
            # If it's the first band, initialize the DataFrame with coordinates
            if band_index == 1:
                rows, cols = np.indices(band_data.shape)
                latitudes, longitudes = src.xy(rows, cols)
                df_sentinel2['Pixel X'] = cols.flatten()
                df_sentinel2['Pixel Y'] = rows.flatten()
                df_sentinel2['Latitude'] = np.array(latitudes).flatten()
                df_sentinel2['Longitude'] = np.array(longitudes).flatten()

            # Add band data to DataFrame
            df_sentinel2[f'Band_{band_index}'] = band_data.flatten()
        
        # Print NaN statistics for this file
        nan_counts = df_sentinel2.isna().sum()
        print(f"NaN counts for {os.path.basename(file_path)}:\n{nan_counts}")

        return df_sentinel2

# Combine data from all Sentinel-2 files
all_data_sentinel2 = pd.DataFrame()

for file_path in sentinel2_files:
    df_sentinel2 = extract_sentinel2_data(file_path)
    all_data_sentinel2 = pd.concat([all_data_sentinel2, df_sentinel2], ignore_index=True)
    print(f"Data extracted from {file_path}")

# Drop rows with any NaN values
cleaned_data_sentinel2 = all_data_sentinel2.dropna()

# Save the cleaned data to a CSV file
output_file = os.path.join(output_dir, 'sentinel2_cleaned_extracted_data.csv')
cleaned_data_sentinel2.to_csv(output_file, index=False)
print(f"Cleaned data saved to {output_file}")

print("Data extraction and cleaning completed for all Sentinel-2 files.")
