import pandas as pd
import rasterio
from rasterio.features import geometry_window
from shapely.geometry import Point
from pyproj import Transformer

import numpy as np

def tif_to_csv_with_coordinates(tif_path, csv_path):
    with rasterio.open(tif_path) as src:
        band1 = src.read(1)  # Read the first band of the raster
        transform = src.transform  # Get the affine transformation
        cols, rows = np.meshgrid(np.arange(band1.shape[1]), np.arange(band1.shape[0]))
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        
        # Flatten the arrays
        xs = np.array(xs).flatten()
        ys = np.array(ys).flatten()
        band1_flat = band1.flatten()
        
        # Create a DataFrame
        df = pd.DataFrame({
            'Longitude': xs,
            'Latitude': ys,
            'Value': band1_flat
        })
        
        # Save to CSV
        df.to_csv(csv_path, index=False)

# Example usage
sentinel1_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel1\Composites\sentinel1_annual_median.tif'
sentinel2_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel2\Composites\annual_median_ndvi.tif'

# Convert Sentinel-1 TIF to CSV
tif_to_csv_with_coordinates(sentinel1_path, 'sentinel1_annual_median.csv')

# Convert Sentinel-2 TIF to CSV
tif_to_csv_with_coordinates(sentinel2_path, 'sentinel2_annual_median_ndvi.csv')

# Load the GEDI data
gedi_df = pd.read_csv('filtered_latslonsL4A_within_ROI.csv')
# Load the generated Sentinel-1 and Sentinel-2 CSVs
sentinel1_df = pd.read_csv('sentinel1_annual_median.csv')
sentinel2_df = pd.read_csv('sentinel2_annual_median_ndvi.csv')
# Perform spatial joins or lookups to match GEDI points with the closest Sentinel-1 and Sentinel-2 values
def extract_closest_value(gedi_df, sentinel_df, value_column='Value'):
    # Use a simple nearest-neighbor approach to find the closest value
    merged_df = gedi_df.merge(sentinel_df, on=['Longitude', 'Latitude'], how='left')
    return merged_df[value_column]

# Extract values
gedi_df['Sentinel1'] = extract_closest_value(gedi_df, sentinel1_df)
gedi_df['Sentinel2_NDVI'] = extract_closest_value(gedi_df, sentinel2_df)

# Save the enriched GEDI data with features
gedi_df.to_csv('gedi_with_features.csv', index=False)



# Function to extract raster value at specific latitude and longitude
def extract_raster_value(raster_path, lon, lat, transformer, buffer=0.01):
    with rasterio.open(raster_path) as src:
        # Transform coordinates to match raster CRS
        lon, lat = transformer.transform(lon, lat)
        
        # Significantly expand bounds check
        if (src.bounds.left - buffer) <= lon <= (src.bounds.right + buffer) and \
           (src.bounds.bottom - buffer) <= lat <= (src.bounds.top + buffer):
            
            # Adjust coordinates to be within raster bounds if slightly out
            lon = max(min(lon, src.bounds.right), src.bounds.left)
            lat = max(min(lat, src.bounds.top), src.bounds.bottom)
            
            point = Point(lon, lat)
            window = geometry_window(src, [point])
            if window.width > 0 and window.height > 0:  # Ensure the window is valid
                data = src.read(1, window=window, resampling=rasterio.enums.Resampling.nearest)
                return data[0, 0]
            else:
                print(f"Invalid window: {window} for point: {lon}, {lat}")
                return None
        else:
            print(f"Point {lon}, {lat} is out of bounds.")
            return None

# Define paths to your raster data
sentinel1_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel1\Composites\sentinel1_annual_median.tif'
sentinel2_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel2\Composites\annual_median_ndvi.tif'
dem_slope_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\DEM\slope.tif'
dem_aspect_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\DEM\aspect.tif'
worldcover_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\WorldCover\resampled_worldcover.tif'

# Create transformer from EPSG:4326 to the raster CRS
with rasterio.open(sentinel1_path) as src:
    raster_crs = src.crs
transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)

# Extract features for each GEDI point
features = []
for index, row in gedi_df.iterrows():
    lon, lat = row['Longitude'], row['Latitude']
    sentinel1_value = extract_raster_value(sentinel1_path, lon, lat, transformer)
    sentinel2_value = extract_raster_value(sentinel2_path, lon, lat, transformer)
    dem_slope_value = extract_raster_value(dem_slope_path, lon, lat, transformer)
    dem_aspect_value = extract_raster_value(dem_aspect_path, lon, lat, transformer)
    worldcover_value = extract_raster_value(worldcover_path, lon, lat, transformer)
    
    features.append([sentinel1_value, sentinel2_value, dem_slope_value, dem_aspect_value, worldcover_value])

# Add features to GEDI DataFrame
features_df = pd.DataFrame(features, columns=['Sentinel1', 'Sentinel2_NDVI', 'DEM_Slope', 'DEM_Aspect', 'WorldCover'])
gedi_df = pd.concat([gedi_df, features_df], axis=1)

# Save the enriched GEDI data with features
gedi_df.to_csv('gedi_with_features.csv', index=False)



import rasterio

# Function to check the CRS of a raster file
def check_raster_crs(raster_path):
    with rasterio.open(raster_path) as src:
        print(f"CRS of {raster_path}: {src.crs}")
        return src.crs

# Example: Check the CRS of the Sentinel-1 raster
sentinel1_crs = check_raster_crs(r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel1\Composites\sentinel1_annual_median.tif')

# Repeat for other rasters (Sentinel-2, DEM, WorldCover)
sentinel2_crs = check_raster_crs(r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel2\Composites\annual_median_ndvi.tif')
dem_slope_crs = check_raster_crs(r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\DEM\slope.tif')
worldcover_crs = check_raster_crs(r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\WorldCover\resampled_worldcover.tif')

