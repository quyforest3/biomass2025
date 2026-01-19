import os
import numpy as np
import rasterio

# Set input and output directories
input_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinenl2"
output_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinenl2\NDSIs"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all preprocessed Sentinel-2 files
sen2_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')]

def calculate_ndsi(file_path, output_path):
    try:
        with rasterio.open(file_path) as src:
            # Check if the file contains the necessary bands
            if src.count < 12:
                raise ValueError("File must contain at least 12 bands for the required calculations.")
            
            # Read the required bands
            red_band = src.read(4)  # Band 4 (Red)
            nir_band = src.read(8)  # Band 8 (NIR)
            green_band = src.read(3)  # Band 3 (Green)
            swir_band = src.read(11)  # Band 11 (SWIR)

            # Calculate NDVI
            ndvi = (nir_band - red_band) / (nir_band + red_band)

            # Calculate NDWI
            ndwi = (green_band - nir_band) / (green_band + nir_band)

            # Calculate NDSI
            ndsi = (green_band - swir_band) / (green_band + swir_band)
            
            # Create output file metadata
            meta = src.meta
            meta.update(dtype=rasterio.float32, count=3)

            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(ndvi.astype(rasterio.float32), 1)
                dst.write(ndwi.astype(rasterio.float32), 2)
                dst.write(ndsi.astype(rasterio.float32), 3)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Process each Sentinel-2 file
for sen2_file in sen2_files:
    file_name = os.path.basename(sen2_file)
    output_file = os.path.join(output_dir, f"ndsi_{file_name}")
    calculate_ndsi(sen2_file, output_file)
    print(f"Processed {file_name} and saved NDSIs to {output_file}")

print("Sentinel-2 NDSI calculation completed.")
