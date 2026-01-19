import os
import rasterio
from rasterio.enums import Resampling
import numpy as np
from pyproj import Proj, transform

def degrees_to_meters(lat):
    # WGS84 ellipsoid constants
    a = 6378137.0  # Major axis
    e2 = 0.00669437999014  # Eccentricity squared

    lat_rad = np.radians(lat)

    # Radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    # Distance per degree of latitude (approximately constant)
    m_per_deg_lat = np.pi / 180.0 * N * (1 - e2) / (1 - e2 * np.sin(lat_rad) ** 2) ** (3 / 2)

    # Distance per degree of longitude (varies with latitude)
    m_per_deg_lon = np.pi / 180.0 * N * np.cos(lat_rad)

    return m_per_deg_lat, m_per_deg_lon

def calculate_slope_aspect(dem_file, output_dir):
    with rasterio.open(dem_file) as src:
        # Assuming the data is in WGS84 (EPSG:4326)
        bounds = src.bounds
        center_lat = (bounds.top + bounds.bottom) / 2

        # Calculate meters per degree at the center latitude
        m_per_deg_lat, m_per_deg_lon = degrees_to_meters(center_lat)

        print(f"Approximate resolution (meters/pixel): {src.res[0] * m_per_deg_lon} meters (lon), {src.res[1] * m_per_deg_lat} meters (lat)")

        desired_resolution = 20  # Desired resolution in meters

        # Rescale the resolution
        scale_factor_lon = (src.res[0] * m_per_deg_lon) / desired_resolution
        scale_factor_lat = (src.res[1] * m_per_deg_lat) / desired_resolution

        new_width = max(1, int(src.width * scale_factor_lon))
        new_height = max(1, int(src.height * scale_factor_lat))

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
