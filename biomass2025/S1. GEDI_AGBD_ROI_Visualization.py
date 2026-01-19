import os
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import warnings
from shapely.errors import ShapelyDeprecationWarning
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

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

            valid_indices = quality == 1  # Change to select quality flag 1
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

# Step 2: Retain only rows where Quality Flag is 1
latslonsL4A = latslonsL4A[latslonsL4A['Quality Flag'] == 1]
print(f"Records after quality flag filtering: {len(latslonsL4A)}")

# Calculate the relative standard error and store it in a new column
latslonsL4A['relative_standard_error'] = (latslonsL4A['AGB_SE'] / latslonsL4A['AGB_L4A']) * 100

# Keep only the rows where relative_standard_error is less than 100
latslonsL4A = latslonsL4A[latslonsL4A['relative_standard_error'] < 100]
print(f"Records after relative standard error filtering: {len(latslonsL4A)}")

# Load ROI (KML file)
kml_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\ROI_south.kml"

roi_gdf = gpd.read_file(kml_path, driver='KML')

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
output_file_roi = 'filtered_latslonsL4A_within_ROI_quality_1.csv'
points_within_roi.to_csv(output_file_roi, index=False)

print(f"Data within ROI saved to {output_file_roi}")

# Plot the initial data distribution
plt.figure(figsize=(10, 6))
plt.scatter(latslonsL4A['Longitude'], latslonsL4A['Latitude'], alpha=0.5, s=10)
plt.title('Spatial Distribution of GEDI Data Points with Quality Flag 1')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

import geopandas as gpd
import pandas as pd
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.tile_providers import get_provider, Vendors
from bokeh.models import LinearColorMapper, ColorBar
from pyproj import Transformer

# Function to convert lat/lon to Web Mercator for Bokeh
def lat_lon_to_mercator(lat, lon):
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
    return transformer.transform(lat, lon)

# Convert Latitude and Longitude to Web Mercator
latslonsL4A['x'], latslonsL4A['y'] = lat_lon_to_mercator(latslonsL4A['Latitude'], latslonsL4A['Longitude'])

# Load new ROI from KML file
kml_path = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\ROI_south.kml"
roi_gdf = gpd.read_file(kml_path, driver='KML')

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

# Convert the filtered points to Web Mercator
points_within_roi['x'], points_within_roi['y'] = lat_lon_to_mercator(points_within_roi['Latitude'], points_within_roi['Longitude'])

# Ensure geometries are serializable
points_within_roi = points_within_roi.drop(columns='geometry')

# Create ColumnDataSource
source = ColumnDataSource(points_within_roi)

# Initialize the Bokeh figure
tile_provider = get_provider(Vendors.ESRI_IMAGERY)
p = figure(x_axis_type="mercator", y_axis_type="mercator", width=800, height=600, title="Interactive AGBD Map",
           x_axis_label='Longitude', y_axis_label='Latitude')
p.add_tile(tile_provider)

# Create a color mapper for AGBD values
color_mapper = LinearColorMapper(palette="Viridis256", low=points_within_roi['AGB_L4A'].min(), high=points_within_roi['AGB_L4A'].max())

# Add circles for each point
points = p.circle('x', 'y', size=5, source=source, color={'field': 'AGB_L4A', 'transform': color_mapper}, fill_alpha=0.6, line_color=None, legend_label="AGBD Points")

# Add hover tool
hover = HoverTool()
hover.tooltips = [("Latitude", "@Latitude"), ("Longitude", "@Longitude"), ("AGB (Mg/ha)", "@AGB_L4A")]
p.add_tools(hover)

# Add color bar
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0,0), title='AGB (Mg/ha)')
p.add_layout(color_bar, 'right')

# Highlight top 10 AGBD points with larger circles
top_agbd = points_within_roi.nlargest(10, 'AGB_L4A')
top_source = ColumnDataSource(top_agbd)
top_points = p.circle('x', 'y', size=10, source=top_source, color="red", fill_alpha=0.9, line_color="white", legend_label="Top 10 AGBD Points")

# Adjust legend location
p.legend.location = "top_right"
p.legend.click_policy="hide"

# Output the visualization to an HTML file
output_file("interactive_agbd_map_bokeh.html")
show(p)
