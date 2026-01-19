import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Set input and output directories
input_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinenl2"
output_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinenl2\Composites"

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

# Read and resample all preprocessed Sentinel-2 images into lists
ndvi_bands = []

for sen2_file in sen2_files:
    try:
        reprojected_data = resample_image(sen2_file, ref_shape, ref_transform, ref_crs)
        ndvi_bands.append(reprojected_data[0])  # Since there is only one band
    except ValueError as e:
        print(f"Skipping file {sen2_file}: {e}")

# Stack arrays along the third axis (time) and calculate the median for NDVI
ndvi_stack = np.dstack(ndvi_bands)
ndvi_median = np.median(ndvi_stack, axis=2)

# Update metadata and save the median composite
meta.update(dtype=rasterio.float32, count=1)

output_file_ndvi = os.path.join(output_dir, 'sentinel2_annual_median_ndvi.tif')
with rasterio.open(output_file_ndvi, 'w', **meta) as dst:
    dst.write(ndvi_median.astype(rasterio.float32), 1)

print(f"Sentinel-2 annual median composite saved to {output_file_ndvi}")
