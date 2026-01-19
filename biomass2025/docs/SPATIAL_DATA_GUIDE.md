# Creating Spatial Data for Biomass Estimation

This guide explains how to generate the spatial data file required for the Spatial Analysis feature in the dashboard.

## Overview

The Spatial Analysis feature requires a CSV file with geographic coordinates (latitude/longitude) and biomass data. The repository includes a Google Earth Engine (GEE) script to generate this data.

## Required Data Structure

The spatial analysis expects a CSV file named `merged_gedi_sentinel2_data_with_indices.csv` with the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `Longitude_gedi` | Longitude coordinates from GEDI | ✅ Yes |
| `Latitude_gedi` | Latitude coordinates from GEDI | ✅ Yes |
| `AGB_L4A` | Above Ground Biomass (Mg/ha) | ✅ Yes |
| Sentinel-2 bands | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 | Optional |
| Vegetation indices | NDVI, NDMI, NDWI, EVI, ChlRe, NDCI | Optional |

## Method 1: Using Google Earth Engine (Recommended)

### Prerequisites

1. **Google Earth Engine Account**
   - Sign up at https://earthengine.google.com/
   - Wait for account approval (usually 1-2 days)

2. **Python Environment**
   ```bash
   pip install earthengine-api pandas numpy
   ```

3. **Authenticate Earth Engine**
   ```bash
   earthengine authenticate
   ```
   This will open a browser for authentication.

### Steps

1. **Configure the Script**
   
   Edit `scripts/create_spatial_data_gee.py` and customize the following parameters:

   ```python
   # Define your Region of Interest (ROI)
   ROI_COORDINATES = [
       [-1.7, 50.8],   # Southwest corner [longitude, latitude]
       [-1.3, 50.8],   # Southeast corner
       [-1.3, 51.0],   # Northeast corner
       [-1.7, 51.0]    # Northwest corner
   ]

   # Time range for data collection
   START_DATE = '2019-01-01'
   END_DATE = '2019-12-31'

   # Cloud cover threshold for Sentinel-2
   MAX_CLOUD_COVER = 20  # percent
   ```

2. **Run the Script**
   ```bash
   cd /path/to/BioVision-Analytics-Hub
   python scripts/create_spatial_data_gee.py
   ```

3. **Verify Output**
   
   The script will create `merged_gedi_sentinel2_data_with_indices.csv` in the repository root. Check that it contains:
   - Longitude_gedi, Latitude_gedi, AGB_L4A columns
   - At least 100+ data points for meaningful spatial analysis
   - Reasonable coordinate ranges for your study area

### Customization Tips

- **Adjust ROI**: Use [geojson.io](http://geojson.io) to visually define your study area
- **Time Range**: Use years when GEDI data is available (2019-present)
- **Cloud Cover**: Lower values give cleaner imagery but fewer data points
- **Resolution**: GEDI has ~25m footprints, Sentinel-2 is 10m resolution

### Troubleshooting

**Error: "No GEDI data found"**
- GEDI coverage is not global. Check availability at https://gedi.umd.edu/
- Try a different time range or location
- GEDI data starts from April 2019

**Error: "Computation timeout"**
- Reduce the ROI size
- Shorten the time range
- Process data in multiple batches

**Large Dataset Issues**
- If dataset is too large to download directly, the script will suggest alternatives
- Consider exporting to Google Drive instead using Earth Engine's export functions

## Method 2: Manual Merging (Alternative)

If you already have GEDI and Sentinel-2 data as separate CSV files, you can merge them using the provided scripts:

1. **Prepare GEDI Data**
   - CSV with columns: `Latitude`, `Longitude`, `AGB_L4A`
   - Filter for quality flag = 1 (good quality)

2. **Prepare Sentinel-2 Data**
   - CSV with columns: `Latitude`, `Longitude`, spectral bands
   - Pre-processed and cloud-masked

3. **Merge Data**
   ```bash
   python "S2. merge_GEDI_Sentinel2_data.py"
   ```
   
   Or use the nearest neighbor approach:
   ```bash
   python "p4. merge_gedi_sentinel2_nearest_neighbor.py"
   ```

4. **Rename Output**
   Rename the output file to `merged_gedi_sentinel2_data_with_indices.csv`

## Method 3: Using Existing Repository Data

The current dataset `FEI data/opt_means_cleaned.csv` **cannot be used** for spatial analysis because it lacks geographic coordinates. This file contains only:
- Sentinel-2 spectral bands (B01-B12)
- Vegetation indices (NDVI, NDMI, etc.)
- Biomass values (AGB_2017)
- **Missing: Latitude and Longitude coordinates**

To enable spatial analysis with this data, you would need the original point locations where these measurements were taken.

## Verifying Your Spatial Data

Before using the data in the dashboard, verify it meets the requirements:

```python
import pandas as pd

# Load the file
df = pd.read_csv('merged_gedi_sentinel2_data_with_indices.csv')

# Check required columns
required = ['Longitude_gedi', 'Latitude_gedi', 'AGB_L4A']
print("Required columns present:", all(col in df.columns for col in required))

# Check for missing values
print("Missing values:")
print(df[required].isnull().sum())

# Check data ranges
print("\nData ranges:")
print(f"Latitude: {df['Latitude_gedi'].min():.6f} to {df['Latitude_gedi'].max():.6f}")
print(f"Longitude: {df['Longitude_gedi'].min():.6f} to {df['Longitude_gedi'].max():.6f}")
print(f"AGB: {df['AGB_L4A'].min():.2f} to {df['AGB_L4A'].max():.2f} Mg/ha")
print(f"\nTotal points: {len(df)}")
```

## Expected Results

After successfully creating the spatial data file, the Spatial Analysis dashboard will enable:

- **Geographic Clustering**: Identify regions with similar biomass patterns
- **Spatial Autocorrelation**: Detect spatial clustering using Moran's I and Geary's C
- **Hotspot Analysis**: Find areas with unusually high or low biomass
- **Spatial Interpolation**: Create continuous surface maps of biomass

## Additional Resources

- **GEDI Documentation**: https://lpdaac.usgs.gov/products/gedi04_av002/
- **Sentinel-2 Documentation**: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR
- **Earth Engine Guides**: https://developers.google.com/earth-engine/guides
- **Python Earth Engine API**: https://developers.google.com/earth-engine/guides/python_install

## Support

If you encounter issues:

1. Check that your Earth Engine account is approved and authenticated
2. Verify your ROI coordinates are in [longitude, latitude] format
3. Ensure GEDI data exists for your study area and time range
4. Review the script output for specific error messages

For additional help, refer to the main repository documentation or open an issue on GitHub.
