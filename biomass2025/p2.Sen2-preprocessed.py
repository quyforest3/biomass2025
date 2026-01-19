import os
import rasterio
import numpy as np

# Define directories
input_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Sen2"
output_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel2"
os.makedirs(output_dir, exist_ok=True)

# List Sentinel-2 images
image_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.tif')]

# Define the cloud probability threshold
cloud_threshold = 30  # Adjust based on your specific needs or cloud probability data

def preprocess_sentinel2(image_path, output_path):
    with rasterio.open(image_path) as src:
        # Assuming band indexes: 4 - Red, 8 - NIR, and a cloud probability band (e.g., 10)
        red = src.read(4)
        nir = src.read(8)
        cloud_mask = src.read(10)  # Cloud probability band

        # Create a mask for clouds and their shadows
        mask = cloud_mask > cloud_threshold
        
        # Calculate NDVI, masking out clouds and shadows
        ndvi = np.where(mask, np.nan, (nir - red) / (nir + red))
        
        # Prepare output metadata
        meta = src.meta
        meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)

        # Write the processed NDVI data
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(ndvi.astype(rasterio.float32), 1)

# Process each file
for file_path in image_files:
    output_file_path = os.path.join(output_dir, f"processed_{os.path.basename(file_path)}")
    preprocess_sentinel2(file_path, output_file_path)
    print(f"Processed and saved to {output_file_path}")

print("Sentinel-2 preprocessing completed.")
