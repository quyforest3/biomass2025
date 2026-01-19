import os
import rasterio
import numpy as np
import pandas as pd
from rasterio.enums import Resampling

def calculate_slope_aspect(dem, transform):
    # Calculate the gradient in x and y directions
    grad_x, grad_y = np.gradient(dem, transform[0], transform[4])
    
    # Calculate slope
    slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * 57.2958  # Convert to degrees
    
    # Calculate aspect
    aspect = np.arctan2(-grad_y, grad_x) * 57.2958  # Convert to degrees
    aspect = np.where(aspect < 0, 90.0 - aspect, 450.0 - aspect) % 360
    
    return slope, aspect

# Set the DEM file path and output directory
dem_file = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Land Cover Data\Copernicus_DEM_GLO30.tif'
output_dir = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\DEM'
os.makedirs(output_dir, exist_ok=True)

# Load the DEM data
with rasterio.open(dem_file) as src:
    dem_data = src.read(1)
    dem_data = np.where(dem_data == src.nodata, np.nan, dem_data)  # Replace nodata values with NaN
    transform = src.transform
    
    # Generate arrays of row and column indices
    rows, cols = np.indices(dem_data.shape)
    
    # Convert row and column indices to geographic coordinates
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    
    # Calculate slope and aspect
    slope, aspect = calculate_slope_aspect(dem_data, transform)
    
    # Flatten the arrays
    xs = np.array(xs).flatten()
    ys = np.array(ys).flatten()
    elevations = dem_data.flatten()
    slope = slope.flatten()
    aspect = aspect.flatten()

# Create a DataFrame from the extracted data
df_dem_extended = pd.DataFrame({
    'Longitude': xs,
    'Latitude': ys,
    'Elevation': elevations,
    'Slope': slope,
    'Aspect': aspect
})

# Drop rows with NaN values (corresponding to nodata in the DEM)
df_dem_cleaned_extended = df_dem_extended.dropna()

# Save the DataFrame to a CSV file
output_file = os.path.join(output_dir, 'dem_cleaned_extended_data.csv')
df_dem_cleaned_extended.to_csv(output_file, index=False)

print(f"Extended DEM data extracted and saved to {output_file}")
