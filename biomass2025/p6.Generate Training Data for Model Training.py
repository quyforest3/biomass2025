import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split

# Set paths
combined_raster_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\combined_rasters.tif'
output_csv_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\training_data.csv'

# Open combined raster
with rasterio.open(combined_raster_path) as src:
    bands_data = src.read()
    print(f"The raster contains {bands_data.shape[0]} bands.")

# Reshape data to 2D array where rows are pixels and columns are bands
n_bands, height, width = bands_data.shape
bands_data_reshaped = bands_data.reshape((n_bands, height * width)).T  # Transpose to have bands as columns

# Create a DataFrame from the pixel values
columns = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5']  # Adjust names as necessary
df = pd.DataFrame(bands_data_reshaped, columns=columns)

# Optionally: Inspect the NaN values before filling them
nan_info = df.isna().sum()
print("NaN values per column:\n", nan_info)

# Fill NaNs with column mean (or other value like 0)
df.fillna(df.mean(), inplace=True)

# Save the training data to CSV
df.to_csv(output_csv_path, index=False)
print(f"Training data saved to {output_csv_path}")

# Optionally: Split the data into training and test sets for model validation
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.3, random_state=42)

# Save the split datasets
output_dir = os.path.dirname(output_csv_path)
X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
print("Training and test datasets saved.")
