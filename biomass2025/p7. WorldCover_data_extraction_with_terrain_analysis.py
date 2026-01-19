import rasterio
import pandas as pd
import numpy as np
import csv

# Define the path to the GeoTIFF file
tif_file_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\WorldCover\resampled_worldcover.tif"
output_csv_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\worldcover\WorldCover_LandCover_data.csv"

# Open the GeoTIFF file using rasterio
with rasterio.open(tif_file_path) as dataset:
    # Read the dimensions
    width = dataset.width
    height = dataset.height

    # Open the CSV file in write mode
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Longitude', 'Latitude', 'LandCover_Value'])

        # Process the data in smaller chunks
        chunk_size = 500  # Adjust the chunk size based on available memory
        for i in range(0, height, chunk_size):
            for j in range(0, width, chunk_size):
                # Read the data in the current chunk
                window = rasterio.windows.Window(j, i, min(chunk_size, width - j), min(chunk_size, height - i))
                data = dataset.read(1, window=window)

                # Get the affine transform for the current window
                transform = dataset.window_transform(window)

                # Get the coordinates for the chunk
                rows, cols = np.indices(data.shape)
                xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
                lons_chunk = np.array(xs).flatten()
                lats_chunk = np.array(ys).flatten()
                values_chunk = data.flatten()

                # Write the chunk data directly to the CSV file
                for lon, lat, value in zip(lons_chunk, lats_chunk, values_chunk):
                    writer.writerow([lon, lat, value])

print(f"Data extracted and saved to {output_csv_path}")
