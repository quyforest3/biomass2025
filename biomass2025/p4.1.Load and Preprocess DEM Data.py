import os
import rasterio
from rasterio.enums import Resampling
import numpy as np

def calculate_slope_aspect(dem_file, output_dir):
    with rasterio.open(dem_file) as src:
        print(f"Original resolution (degrees/pixel): {src.res}, width: {src.width}, height: {src.height}")
        
        # Assuming about 111km per degree (very rough approximation, only valid near the equator)
        degrees_to_meters = 111000  
        pixel_size_meters = src.res[0] * degrees_to_meters
        
        print(f"Approximate resolution (meters/pixel): {pixel_size_meters}")
        
        desired_resolution = 20  # Desired resolution in meters
        scale_factor = pixel_size_meters / desired_resolution
        
        print(f"Scale factor: {scale_factor}")
        
        # Calculate new dimensions
        new_width = max(1, int(src.width * scale_factor))
        new_height = max(1, int(src.height * scale_factor))
        
        print(f"New dimensions: Width: {new_width}, Height: {new_height}")

        # Resample the data to the desired resolution
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear
        )
        data = data.squeeze()  # Remove unnecessary dimensions if only one band
        
        # Update transform for the new resolution
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )

        new_meta = src.meta.copy()
        new_meta.update({
            "driver": "GTiff",
            "height": new_height,
            "width": new_width,
            "transform": transform,
            "crs": src.crs,
            "dtype": 'float32'
        })

        # Save the resampled DEM
        resampled_dem_path = os.path.join(output_dir, "resampled_dem.tif")
        with rasterio.open(resampled_dem_path, "w", **new_meta) as dst:
            dst.write(data, 1)

        print("Resampled DEM saved.")

# Specify your DEM file path and output directory
dem_file = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Land Cover Data\Copernicus_DEM_GLO30.tif'
output_dir = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\DEM'

calculate_slope_aspect(dem_file, output_dir)
