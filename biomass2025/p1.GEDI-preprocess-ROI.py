import os
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import warnings
from shapely.errors import ShapelyDeprecationWarning
import rasterio
import matplotlib.pyplot as plt
from pyproj import CRS
from pyproj.exceptions import ProjError

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# Function to check PROJ database access
def check_proj_database():
    try:
        CRS.from_epsg(4326)
        print("PROJ database is accessible.")
    except ProjError as e:
        print(f"PROJ database access error: {e}")
        raise

# Ensure PROJ database is accessible
check_proj_database()

# Set input directory to the specified directory
inDir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest"
os.chdir(inDir)

all_files = os.listdir(os.path.join(inDir, 'GEDI_L4A'))
print("Files in directory:", all_files)

gediL4AFiles = [g for g in all_files if g.startswith('GEDI04_A') and g.endswith('.h5')]
print("GEDI L4A files found:", gediL4AFiles)

if not gediL4AFiles:
    raise FileNotFoundError("No GEDI04_A HDF5 files found in the specified directory.")

lonSampleL4A, latSampleL4A, shotSampleL4A, qualitySampleL4A, beamSampleL4A, agbSampleL4A, agb_seSampleL4A = [], [], [], [], [], [], []

for file in gediL4AFiles:
    file_path = os.path.join('GEDI_L4A', file)
    try:
        print(f"Opening L4A file: {file_path}")
        gediL4A = h5py.File(file_path, 'r')
    except OSError as e:
        print(f"Error opening file {file_path}: {e}")
        continue

    beamNames = [g for g in gediL4A.keys() if g.startswith('BEAM')]
    print(f"L4A Beam names found: {beamNames}")

    for beam in beamNames:
        try:
            lats = gediL4A[f'{beam}/lat_lowestmode'][()]
            lons = gediL4A[f'{beam}/lon_lowestmode'][()]
            shots = gediL4A[f'{beam}/shot_number'][()]
            quality = gediL4A[f'{beam}/l4_quality_flag'][()]
            agb = gediL4A[f'{beam}/agbd'][()]
            agb_se = gediL4A[f'{beam}/agbd_se'][()]

            valid_indices = quality == 0
            lats = lats[valid_indices]
            lons = lons[valid_indices]
            shots = shots[valid_indices]
            agb = agb[valid_indices]
            agb_se = agb_se[valid_indices]
            
            for i in range(len(shots)):
                if i % 100 == 0:
                    shotSampleL4A.append(str(shots[i]))
                    lonSampleL4A.append(lons[i])
                    latSampleL4A.append(lats[i])
                    qualitySampleL4A.append(quality[i])
                    beamSampleL4A.append(beam)
                    agbSampleL4A.append(agb[i])
                    agb_seSampleL4A.append(agb_se[i])
        except KeyError as e:
            print(f"Error accessing data for beam {beam} in file {file}: {e}")

latslonsL4A = pd.DataFrame({
    'Beam': beamSampleL4A, 
    'Shot Number': shotSampleL4A, 
    'Longitude': lonSampleL4A, 
    'Latitude': latSampleL4A,
    'Quality Flag': qualitySampleL4A, 
    'AGB_L4A': agbSampleL4A, 
    'AGB_SE': agb_seSampleL4A, 
    'Data Source': 'L4A'
})

# Debugging: Check initial data
print(f"Initial data records: {len(latslonsL4A)}")

# Step 2: Retain only rows where Quality Flag is 0
latslonsL4A = latslonsL4A[latslonsL4A['Quality Flag'] == 0]
print(f"Records after quality flag filtering: {len(latslonsL4A)}")

# Calculate the relative standard error and store it in a new column
latslonsL4A['relative_standard_error'] = (latslonsL4A['AGB_SE'] / latslonsL4A['AGB_L4A']) * 100

# Keep only the rows where relative_standard_error is less than 100
latslonsL4A = latslonsL4A[latslonsL4A['relative_standard_error'] < 100]
print(f"Records after relative standard error filtering: {len(latslonsL4A)}")

# Load ROI (National Parks of the UK) GeoJSON file
roi_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\National_Parks_England_8737503764120290014.geojson"
roi_gdf = gpd.read_file(roi_path)

# Convert the DataFrame to GeoDataFrame
latslonsL4A_gdf = gpd.GeoDataFrame(
    latslonsL4A, 
    geometry=gpd.points_from_xy(latslonsL4A.Longitude, latslonsL4A.Latitude),
    crs="EPSG:4326"
)

# Ensure ROI CRS is set and matches the data CRS
if roi_gdf.crs != "EPSG:4326":
    roi_gdf = roi_gdf.to_crs("EPSG:4326")

# Perform spatial join to filter points within the ROI
points_within_roi = gpd.sjoin(latslonsL4A_gdf, roi_gdf, how="inner", predicate='within')

# Save the filtered GeoDataFrame to a CSV file
output_file_roi = 'filtered_latslonsL4A_within_ROI.csv'
points_within_roi.to_csv(output_file_roi, index=False)

print(f"Data within ROI saved to {output_file_roi}")

# Plot the initial data distribution
plt.figure(figsize=(10, 6))
plt.scatter(latslonsL4A['Longitude'], latslonsL4A['Latitude'], alpha=0.5, s=10)
plt.title('Spatial Distribution of GEDI Data Points')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# Step 5: Aggregate Filtered GEDI Measurements
def aggregate_to_cells(df, cell_size, min_measurements):
    df['x_cell'] = (df['Longitude'] // (cell_size / 111320)).astype(int)
    df['y_cell'] = (df['Latitude'] // (cell_size / 110540)).astype(int)

    aggregated = df.groupby(['x_cell', 'y_cell']).filter(lambda x: len(x) >= min_measurements)
    aggregated = df.groupby(['x_cell', 'y_cell']).agg({
        'AGB_L4A': 'mean',
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()
    return aggregated

# Aggregate to 25m, 50m, 100m and 200m cells
gedi_100m = aggregate_to_cells(latslonsL4A, 100, 2)
gedi_200m = aggregate_to_cells(latslonsL4A, 200, 2)

print(f"Step 5: Aggregated to 100m cells with {len(gedi_100m)} records.")
print(f"Step 5: Aggregated to 200m cells with {len(gedi_200m)} records.")

# Define land cover file
land_cover_file = 'C:/Users/mn2n23/OneDrive - University of Southampton/Desktop/SC solutions (summer project)/biomass/newforrest/worldcover/WorldCover_LandCover.tif'

# Step 6: Filter by Relevant Land Covers
relevant_codes = [10, 20, 30, 40, 50, 60]

def get_dominant_land_cover(cell, land_cover, transform):
    minx, miny, maxx, maxy = cell.bounds
    ul_row, ul_col = ~transform * (minx, maxy)
    lr_row, lr_col = ~transform * (maxx, miny)
    ul_row, ul_col = int(ul_row), int(ul_col)
    lr_row, lr_col = int(lr_row), int(lr_col)
    
    if ul_row < 0 or ul_col < 0 or lr_row >= land_cover.shape[0] or lr_col >= land_cover.shape[1]:
        return np.nan

    subset = land_cover[ul_row:lr_row, ul_col:lr_col]
    values, counts = np.unique(subset, return_counts=True)
    dominant_land_cover = values[np.argmax(counts)]
    return dominant_land_cover

def filter_by_relevant_land_covers(aggregated_df, cell_size):
    gdf = gpd.GeoDataFrame(aggregated_df, geometry=gpd.points_from_xy(aggregated_df.Longitude, aggregated_df.Latitude))
    gdf['geometry'] = gdf.apply(lambda row: Point(row['Longitude'], row['Latitude']).buffer(cell_size / 111320), axis=1)

    with rasterio.open(land_cover_file) as src:
        land_cover = src.read(1)
        transform = src.transform

    gdf['dominant_land_cover'] = gdf['geometry'].apply(lambda cell: get_dominant_land_cover(cell, land_cover, transform))
    filtered_gdf = gdf[gdf['dominant_land_cover'].isin(relevant_codes)]
    
    return filtered_gdf

gedi_100m_filtered = filter_by_relevant_land_covers(gedi_100m, 100)
gedi_200m_filtered = filter_by_relevant_land_covers(gedi_200m, 200)

print(f"Step 6: Filtered 100m cells to {len(gedi_100m_filtered)} records with relevant land covers.")
print(f"Step 6: Filtered 200m cells to {len(gedi_200m_filtered)} records with relevant land covers.")

import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx

# Define CRS
crs_epsg = 'EPSG:4326'

# Convert filtered data to GeoDataFrames and set CRS
gdf_100m = gpd.GeoDataFrame(gedi_100m_filtered, geometry=gpd.points_from_xy(gedi_100m_filtered.Longitude, gedi_100m_filtered.Latitude), crs=crs_epsg)
gdf_200m = gpd.GeoDataFrame(gedi_200m_filtered, geometry=gpd.points_from_xy(gedi_200m_filtered.Longitude, gedi_200m_filtered.Latitude), crs=crs_epsg)

# Scatter Plot of Aggregated AGBD Data
plt.figure(figsize=(12, 8))
plt.scatter(gdf_100m['Longitude'], gdf_100m['Latitude'], c=gdf_100m['AGB_L4A'], cmap='viridis', s=20, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.colorbar(label='AGB (Mg/ha)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of Aggregated AGBD Data (100m)')
ctx.add_basemap(plt.gca(), crs=gdf_100m.crs, source=ctx.providers.Stamen.TerrainBackground)
plt.show()

# Heatmap of AGBD Data
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
gdf_100m.plot(column='AGB_L4A', cmap='viridis', linewidth=0.8, ax=ax, edgecolor='0.8')
plt.title('Heatmap of AGBD Data (100m)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(ax.collections[0], ax=ax, label='AGB (Mg/ha)')
ctx.add_basemap(ax, crs=gdf_100m.crs, source=ctx.providers.Stamen.TerrainBackground)
plt.show()

# Histogram of AGBD Values
plt.figure(figsize=(10, 6))
plt.hist(gdf_100m['AGB_L4A'], bins=30, color='teal', edgecolor='black', alpha=0.7)
plt.xlabel('AGB (Mg/ha)')
plt.ylabel('Frequency')
plt.title('Histogram of AGBD Values (100m)')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Boxplot of AGBD Values for Different Cell Sizes
plt.figure(figsize=(10, 6))
#plt.boxplot([gdf_100m['AGB_L4A'], gdf_200m['AGB_L4A']], labels=['100m', '200m'])
plt.xlabel('Cell Size')
plt.ylabel('AGB (Mg/ha)')
plt.title('Boxplot of AGBD Values for Different Cell Sizes')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Check the data
print("Sample data from gdf_100m:")
print(gdf_100m.head())

print("Sample data from gdf_200m:")
print(gdf_200m.head())

# Verify the coordinate range
print("Latitude range in gdf_100m:", gdf_100m['Latitude'].min(), gdf_100m['Latitude'].max())
print("Longitude range in gdf_100m:", gdf_100m['Longitude'].min(), gdf_100m['Longitude'].max())

print("Latitude range in gdf_200m:", gdf_200m['Latitude'].min(), gdf_200m['Latitude'].max())
print("Longitude range in gdf_200m:", gdf_200m['Longitude'].min(), gdf_200m['Longitude'].max())

