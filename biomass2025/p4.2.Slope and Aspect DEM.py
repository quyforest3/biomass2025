import os
import rasterio
import numpy as np
from rasterio.enums import Resampling

def calculate_terrain_attributes(dem_path, output_dir):
    with rasterio.open(dem_path) as src:
        # Read the elevation band
        elevation = src.read(1)
        
        # Calculate the gradient in the x and y directions
        grad_x, grad_y = np.gradient(elevation, src.res[0], src.res[1])
        
        # Calculate the slope
        slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * 57.2958  # Convert to degrees
        
        # Calculate the aspect
        aspect = np.arctan2(-grad_y, grad_x) * 57.2958
        aspect = np.where(aspect < 0, 90.0 - aspect, 450.0 - aspect) % 360
        
        # Define the metadata for the output datasets
        meta = src.meta.copy()
        meta.update(dtype=rasterio.float32, count=1)
        
        # Write the slope raster
        slope_output_path = os.path.join(output_dir, 'slope.tif')
        with rasterio.open(slope_output_path, 'w', **meta) as dst:
            dst.write(slope.astype(rasterio.float32), 1)
        print(f"Slope map saved to {slope_output_path}")
        
        # Write the aspect raster
        aspect_output_path = os.path.join(output_dir, 'aspect.tif')
        with rasterio.open(aspect_output_path, 'w', **meta) as dst:
            dst.write(aspect.astype(rasterio.float32), 1)
        print(f"Aspect map saved to {aspect_output_path}")

# Specify your DEM file path and output directory
dem_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\DEM\resampled_dem.tif'
output_dir = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\DEM'

calculate_terrain_attributes(dem_path, output_dir)
