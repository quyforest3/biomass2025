import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Set input and output directories
input_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel1"
output_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel1\Composites"
os.makedirs(output_dir, exist_ok=True)

# List all preprocessed Sentinel-1 files
sen1_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')]

# Read the first image to determine the reference shape, transform, and CRS
if sen1_files:
    with rasterio.open(sen1_files[0]) as src:
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

    # Read and resample all preprocessed Sentinel-1 images into lists
    vv_bands = []
    vh_bands = []

    for sen1_file in sen1_files:
        try:
            reprojected_data = resample_image(sen1_file, ref_shape, ref_transform, ref_crs)
            vv_bands.append(reprojected_data[0])
            vh_bands.append(reprojected_data[1])
        except ValueError as e:
            print(f"Skipping file {sen1_file}: {e}")

    # Stack arrays along the third axis (time) and calculate the median
    vv_stack = np.dstack(vv_bands) if vv_bands else np.empty(0)
    vh_stack = np.dstack(vh_bands) if vh_bands else np.empty(0)

    if vv_stack.size > 0 and vh_stack.size > 0:
        vv_median = np.median(vv_stack, axis=2)
        vh_median = np.median(vh_stack, axis=2)

        # Update metadata and save the median composite
        meta.update(dtype=rasterio.float32, height=ref_shape[0], width=ref_shape[1], transform=ref_transform, crs=ref_crs)

        output_file_sen1 = os.path.join(output_dir, 'sentinel1_annual_median.tif')
        with rasterio.open(output_file_sen1, 'w', **meta) as dst:
            dst.write(vv_median.astype(rasterio.float32), 1)
            dst.write(vh_median.astype(rasterio.float32), 2)

        print(f"Sentinel-1 annual median composite saved to {output_file_sen1}")
    else:
        print("No data available to create composites.")
else:
    print("No Sentinel-1 files found in the input directory.")
