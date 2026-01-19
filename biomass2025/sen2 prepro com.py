import os
import numpy as np
import rasterio

# Set input and output directories
input_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Sen2"
output_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinenl2"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all Sentinel-2 files
sen2_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')]

def preprocess_sentinel2(file_path, output_path):
    try:
        with rasterio.open(file_path) as src:
            # Check if the file contains the necessary bands
            if src.count < 10:
                raise ValueError("File must contain at least ten bands including cloud probability.")
            
            bands = src.read()
            red_band = bands[3]   # Band 4 (Red)
            nir_band = bands[7]   # Band 8 (NIR)
            cloud_prob = bands[9]  # Assuming cloud probability is stored in band 10
            
            # Cloud masking
            cloud_mask = cloud_prob > 50  # Cloud probability threshold (50%)
            
            # Calculate NDVI
            ndvi = (nir_band - red_band) / (nir_band + red_band)
            ndvi[cloud_mask] = np.nan  # Mask clouds
            
            # Update metadata to include an extra band for NDVI
            meta = src.meta
            meta.update(dtype=rasterio.float32, count=src.count + 1)
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                # Write original bands
                for i in range(1, src.count + 1):
                    dst.write(bands[i-1], i)
                # Write NDVI as the last band
                dst.write(ndvi.astype(rasterio.float32), src.count + 1)
                
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Preprocess each Sentinel-2 file
for sen2_file in sen2_files:
    file_name = os.path.basename(sen2_file)
    output_file = os.path.join(output_dir, f"preprocessed_{file_name}")
    preprocess_sentinel2(sen2_file, output_file)
    print(f"Processed {file_name} and saved to {output_file}")

print("Sentinel-2 preprocessing completed.")


import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Set input and output directories
input_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinenl2"
output_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinenl2\Composites"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all preprocessed Sentinel-2 files
sen2_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')]

# Read the first image to determine the reference shape, transform, and CRS
with rasterio.open(sen2_files[0]) as src:
    ref_shape = src.shape
    ref_transform = src.transform
    ref_crs = src.crs
    meta = src.meta

# Function to resample an image to the reference shape
def resample_image(input_file, ref_shape, ref_transform, ref_crs):
    with rasterio.open(input_file) as src:
        # Check if the CRS matches the reference CRS
        if src.crs != ref_crs:
            raise ValueError(f"CRS of {input_file} does not match the reference CRS")

        # Resample the data to the reference shape
        data = src.read(
            out_shape=(src.count, ref_shape[0], ref_shape[1]),
            resampling=Resampling.bilinear
        )
        return data

# Read and resample all preprocessed Sentinel-2 images into a list
all_bands = []

for sen2_file in sen2_files:
    try:
        resampled_data = resample_image(sen2_file, ref_shape, ref_transform, ref_crs)
        all_bands.append(resampled_data)
    except Exception as e:
        print(f"Skipping file {sen2_file}: {e}")

# Stack arrays along the third axis (time) and calculate the median for each band
all_bands_stack = np.stack(all_bands, axis=-1)
all_bands_median = np.median(all_bands_stack, axis=-1)

# Save the median composite
meta.update(dtype=rasterio.float32, count=all_bands_median.shape[0])

output_file_sen2 = os.path.join(output_dir, 'sentinel2_annual_median_composite.tif')
with rasterio.open(output_file_sen2, 'w', **meta) as dst:
    for i in range(all_bands_median.shape[0]):
        dst.write(all_bands_median[i].astype(rasterio.float32), i + 1)

print(f"Sentinel-2 annual median composite saved to {output_file_sen2}")

import os
import numpy as np
import rasterio

# Set input and output directories
input_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinenl2\Composites"
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
