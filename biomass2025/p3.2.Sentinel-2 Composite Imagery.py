import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Set input and output directories
input_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel2"
output_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel2\Composites"
os.makedirs(output_dir, exist_ok=True)

# List all NDVI files explicitly
ndvi_files = [
    'processed_Sentinel2_Composite-0000000000-0000000000.tif',
    'processed_Sentinel2_Composite-0000000000-0000008704.tif',
    'processed_Sentinel2_Composite-0000000000-0000017408.tif',
    'processed_Sentinel2_Composite-0000000000-0000026112.tif'
]
ndvi_files = [os.path.join(input_dir, f) for f in ndvi_files]

# Determine the smallest dimensions to resample all images to the same size
min_width = min_height = float('inf')
for file_path in ndvi_files:
    with rasterio.open(file_path) as src:
        if src.width < min_width:
            min_width = src.width
        if src.height < min_height:
            min_height = src.height

# Load and resample images
ndvi_stack = []
for file_path in ndvi_files:
    with rasterio.open(file_path) as src:
        data = src.read(
            1,
            out_shape=(min_height, min_width),
            resampling=Resampling.bilinear
        )
        ndvi_stack.append(data)

# Convert list to a NumPy array and calculate the median
ndvi_stack = np.array(ndvi_stack)
median_ndvi = np.nanmedian(ndvi_stack, axis=0)

# Save the median NDVI composite
if ndvi_stack.size > 0:
    meta = src.meta
    meta.update({
        'dtype': 'float32',
        'height': min_height,
        'width': min_width,
        'count': 1
    })
    output_file = os.path.join(output_dir, 'annual_median_ndvi.tif')
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(median_ndvi.astype('float32'), 1)
    print(f"Annual median NDVI composite saved to {output_file}")
else:
    print("No NDVI data to process.")
