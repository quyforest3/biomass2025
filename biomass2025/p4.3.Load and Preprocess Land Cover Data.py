import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def resample_raster(input_raster, output_raster, new_resolution=20):
    with rasterio.open(input_raster) as src:
        print(f"Original resolution (degrees/pixel): {src.res}")
        print(f"Original dimensions: Width: {src.width}, Height: {src.height}")

        # Calculate new dimensions
        new_width = int(src.width * (src.res[0] / (new_resolution / 100000)))
        new_height = int(src.height * (src.res[1] / (new_resolution / 100000)))

        if new_width == 0 or new_height == 0:
            raise ValueError("Resampled dimensions result in zero width or height, check scale factor and input resolution.")

        print(f"New dimensions: Width: {new_width}, Height: {new_height}")

        # Calculate new transformation
        transform, width, height = calculate_default_transform(
            src.crs, src.crs, src.width, src.height, *src.bounds,
            dst_width=new_width, dst_height=new_height
        )

        # Define metadata for the new resampled raster
        metadata = src.meta.copy()
        metadata.update({
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'transform': transform,
            'crs': src.crs
        })

        # Perform the resampling
        with rasterio.open(output_raster, 'w', **metadata) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest)

        print(f"Resampled raster saved to {output_raster}")

# Define paths
input_raster_path = 'C:\\Users\\mn2n23\\OneDrive - University of Southampton\\Desktop\\SC solutions (summer project)\\biomass\\newforrest\\worldcover\\WorldCover_LandCover.tif'
output_raster_path = 'C:\\Users\\mn2n23\\OneDrive - University of Southampton\\Desktop\\SC solutions (summer project)\\biomass\\newforrest\\Processed\\WorldCover\\resampled_worldcover.tif'

# Resample raster
resample_raster(input_raster_path, output_raster_path, new_resolution=20)
