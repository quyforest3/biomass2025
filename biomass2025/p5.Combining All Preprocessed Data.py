import os
import rasterio
import numpy as np

# Set input directory where all preprocessed rasters are stored
input_dir = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed'

# List of all raster files to combine
raster_files = [
    os.path.join(input_dir, 'sentinel1', 'Composites', 'sentinel1_annual_median.tif'),
    os.path.join(input_dir, 'sentinel2', 'Composites', 'annual_median_ndvi.tif'),
    os.path.join(input_dir, 'DEM', 'slope.tif'),
    os.path.join(input_dir, 'DEM', 'aspect.tif'),
    os.path.join(input_dir, 'WorldCover', 'resampled_worldcover.tif')
]

# Open the first raster to use as a reference for metadata
with rasterio.open(raster_files[0]) as src:
    meta = src.meta.copy()
    meta.update(count=len(raster_files), dtype=rasterio.float32)

# Define output path for combined raster
output_combined_raster = os.path.join(input_dir, 'combined_rasters.tif')

# Create and write to the output combined raster
with rasterio.open(output_combined_raster, 'w', **meta) as dst:
    for idx, raster_file in enumerate(raster_files, start=1):
        with rasterio.open(raster_file) as src:
            band_data = src.read(1, out_shape=(dst.height, dst.width))  # Ensure matching size
            dst.write_band(idx, band_data.astype(rasterio.float32))

print(f"Combined raster saved to {output_combined_raster}")
