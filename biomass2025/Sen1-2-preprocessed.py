import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Define the paths for input and output directories
input_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Sen1\EarthEngineExports-20240725T125257Z-001\EarthEngineExports"
output_dir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\Processed\sentinel1"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all Sentinel-1 files in the input directory
sen1_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')]

def process_sentinel1(file_path, output_path):
    # Open the Sentinel-1 image file
    with rasterio.open(file_path) as src:
        # Read VV and VH polarizations
        vv = src.read(1)  # Assuming VV is band 1
        vh = src.read(2)  # Assuming VH is band 2
        
        # Convert to dB scale
        vv_db = 10 * np.log10(vv)
        vh_db = 10 * np.log10(vh)
        
        # Calculate VV/VH ratio
        vv_vh_ratio = vv_db / vh_db
        
        # Prepare output metadata
        meta = src.meta.copy()
        meta.update({
            'dtype': 'float32',
            'count': 3
        })
        
        # Write processed data to output file
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(vv_db.astype('float32'), 1)
            dst.write(vh_db.astype('float32'), 2)
            dst.write(vv_vh_ratio.astype('float32'), 3)

# Process each file
for file_path in sen1_files:
    output_file_path = os.path.join(output_dir, f"processed_{os.path.basename(file_path)}")
    process_sentinel1(file_path, output_file_path)
    print(f"Processed and saved: {output_file_path}")

print("Processing completed for all Sentinel-1 files.")



