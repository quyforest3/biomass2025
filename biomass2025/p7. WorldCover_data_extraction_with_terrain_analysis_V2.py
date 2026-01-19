import rasterio
import numpy as np
import pandas as pd

# Load the WorldCover GeoTIFF file
worldcover_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\WorldCover\resampled_worldcover.tif"
with rasterio.open(worldcover_path) as src:
    worldcover_data = src.read(1)  # Read the first band
    transform = src.transform  # Get affine transformation

# Display basic information
print(f"WorldCover data shape: {worldcover_data.shape}")
print(f"Unique land cover classes: {np.unique(worldcover_data)}")

# Calculate latitude and longitude for each pixel
nrows, ncols = worldcover_data.shape
x_indices, y_indices = np.meshgrid(np.arange(ncols), np.arange(nrows))

# Convert pixel coordinates to geographic coordinates
# (x, y) -> (lon, lat)
longitudes = transform[2] + x_indices * transform[0] + y_indices * transform[1]
latitudes = transform[5] + x_indices * transform[3] + y_indices * transform[4]

# Flatten the arrays and create a DataFrame
data = {
    "Row": y_indices.flatten(),
    "Column": x_indices.flatten(),
    "Latitude": latitudes.flatten(),
    "Longitude": longitudes.flatten(),
    "Land_Cover_Class": worldcover_data.flatten()
}

df_landcover = pd.DataFrame(data)

# Save DataFrame to CSV file
csv_file_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\WorldCover\landcover_data_with_latlon.csv"
df_landcover.to_csv(csv_file_path, index=False)

print(f"Land cover data with coordinates saved to {csv_file_path}")

# Optionally, display the first few rows of the DataFrame
print(df_landcover.head())
