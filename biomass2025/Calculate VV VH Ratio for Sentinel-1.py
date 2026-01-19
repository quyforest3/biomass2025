import os
import numpy as np
import rasterio

# Set input and output directories
input_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel1\Composites"
output_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel1\Composites"

# Load the Sentinel-1 annual median composite
input_file = os.path.join(input_dir, 'sentinel1_annual_median.tif')

# Read VV and VH bands
with rasterio.open(input_file) as src:
    vv_band = src.read(1)
    vh_band = src.read(2)
    meta = src.meta

# Calculate VV/VH ratio
vv_vh_ratio = vv_band / vh_band
vv_vh_ratio[np.isinf(vv_vh_ratio)] = np.nan  # Handle division by zero

# Update metadata and save the VV/VH ratio composite
meta.update(dtype=rasterio.float32, count=1)

output_file_ratio = os.path.join(output_dir, 'sentinel1_vv_vh_ratio.tif')
with rasterio.open(output_file_ratio, 'w', **meta) as dst:
    dst.write(vv_vh_ratio.astype(rasterio.float32), 1)

print(f"Sentinel-1 VV/VH ratio composite saved to {output_file_ratio}")
