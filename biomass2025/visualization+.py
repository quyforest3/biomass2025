import os
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Point
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import HoverTool, ColumnDataSource, ColorBar, LinearColorMapper
from bokeh.transform import linear_cmap
from bokeh.tile_providers import get_provider, Vendors
from bokeh.io import output_notebook, curdoc, reset_output
from bokeh.layouts import gridplot
from bokeh.palettes import Viridis256
import warnings
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
output_notebook()

# Set input directory to the specified directory
inDir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest"

os.chdir(inDir)

# List all files in the GEDI_L4A directory for debugging
all_files = os.listdir(os.path.join(inDir, 'GEDI_L4A'))
print("Files in directory:", all_files)

# Adjust the file pattern if necessary for L4A
gediL4AFiles = [g for g in all_files if g.startswith('GEDI04_A') and g.endswith('.h5')]
print("GEDI L4A files found:", gediL4AFiles)

if not gediL4AFiles:
    raise FileNotFoundError("No GEDI04_A HDF5 files found in the specified directory.")

# Initialize lists to store combined data for L4A
lonSampleL4A, latSampleL4A, shotSampleL4A, qualitySampleL4A, beamSampleL4A, agbSampleL4A = [], [], [], [], [], []

# Process L4A files
for file in gediL4AFiles:
    file_path = os.path.join('GEDI_L4A', file)
    try:
        print(f"Opening L4A file: {file_path}")
        gediL4A = h5py.File(file_path, 'r')  # Read file using h5py
    except OSError as e:
        print(f"Error opening file {file_path}: {e}")
        continue

    beamNames = [g for g in gediL4A.keys() if g.startswith('BEAM')]
    print(f"L4A Beam names found: {beamNames}")

    # Loop through each beam and extract data
    for beam in beamNames:
        try:
            lats = gediL4A[f'{beam}/lat_lowestmode'][()]
            lons = gediL4A[f'{beam}/lon_lowestmode'][()]
            shots = gediL4A[f'{beam}/shot_number'][()]
            quality = gediL4A[f'{beam}/l4_quality_flag'][()]
            agb = gediL4A[f'{beam}/agbd'][()]  # Replace with actual path to AGB data

            # Take every 100th shot and append to list
            for i in range(len(shots)):
                if i % 100 == 0:
                    shotSampleL4A.append(str(shots[i]))
                    lonSampleL4A.append(lons[i])
                    latSampleL4A.append(lats[i])
                    qualitySampleL4A.append(quality[i])
                    beamSampleL4A.append(beam)
                    agbSampleL4A.append(agb[i])
        except KeyError as e:
            print(f"Error accessing data for beam {beam} in file {file}: {e}")

# Write all of the sample shots to a dataframe including AGB for L4A
latslonsL4A = pd.DataFrame({'Beam': beamSampleL4A, 'Shot Number': shotSampleL4A, 'Longitude': lonSampleL4A, 'Latitude': latSampleL4A,
                            'Quality Flag': qualitySampleL4A, 'AGB_L4A': agbSampleL4A, 'Data Source': 'L4A'})

# Remove outliers in AGB values that are higher than 1000
latslonsL4A = latslonsL4A[latslonsL4A['AGB_L4A'] <= 1000]

# Convert to GeoDataFrame
latslonsL4A['geometry'] = latslonsL4A.apply(lambda row: Point(row.Longitude, row.Latitude), axis=1)
l4a_gdf = gpd.GeoDataFrame(latslonsL4A, crs="EPSG:4326")  # Assuming WGS84 for GEDI data
l4a_gdf = l4a_gdf.drop(columns=['Latitude', 'Longitude'])

# Read the bounding box from the GeoJSON file
geojson_path = os.path.join(inDir, 'National_Parks_England_8737503764120290014.geojson')
roi_gdf = gpd.read_file(geojson_path)

# Ensure the GeoDataFrame is in WGS84 CRS
if roi_gdf.crs != "EPSG:4326":
    roi_gdf = roi_gdf.to_crs("EPSG:4326")

# Perform spatial join to filter points within the ROI
points_within_roi = gpd.sjoin(l4a_gdf, roi_gdf, how="inner", predicate='within')

# Filter out rows where the Quality Flag column is equal to 0
points_within_roi = points_within_roi[points_within_roi['Quality Flag'] != 0]

# Extract x and y coordinates from the geometry column
points_within_roi['x'] = points_within_roi.geometry.apply(lambda geom: geom.x)
points_within_roi['y'] = points_within_roi.geometry.apply(lambda geom: geom.y)

# 1. Histogram using Seaborn
plt.figure(figsize=(10, 6))
sns.histplot(points_within_roi['AGB_L4A'], bins=50, kde=True, color='green')
plt.xlabel('AGB Values')
plt.ylabel('Frequency')
plt.title('Histogram of AGB Values')
plt.grid(True)
plt.savefig('histogram_agb_values.png')
plt.show()

# 2. Scatter Plot using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quality Flag', y='AGB_L4A', data=points_within_roi, alpha=0.5, color='red')
plt.xlabel('Quality Flag')
plt.ylabel('AGB Values')
plt.title('Scatter Plot of AGB vs Quality Flag')
plt.grid(True)
plt.savefig('scatter_agb_vs_quality_flag.png')
plt.show()

# 3. Box Plot using Bokeh
reset_output()
output_file("boxplot_agb_by_beam.html")

source_box = ColumnDataSource(points_within_roi)
p_box = figure(x_range=list(points_within_roi['Beam'].unique()), title="Box Plot of AGB Values by Beam", x_axis_label='Beam', y_axis_label='AGB Values')
p_box.vbar(x='Beam', top='AGB_L4A', source=source_box, width=0.9)
hover_box = HoverTool(tooltips=[("Beam", "@Beam"), ("AGB", "@AGB_L4A")])
p_box.add_tools(hover_box)
p_box.xaxis.major_label_orientation = "vertical"
save(p_box)
show(p_box)

# 4. Heatmap using Bokeh
reset_output()
output_file("heatmap_agb_values.html")

source_heatmap = ColumnDataSource(points_within_roi)
mapper = linear_cmap(field_name='AGB_L4A', palette=Viridis256, low=min(points_within_roi['AGB_L4A']), high=max(points_within_roi['AGB_L4A']))

p_heatmap = figure(title="Heatmap of Mean AGB Values", x_axis_label='Longitude', y_axis_label='Latitude', tools='hover', tooltips=[("AGB", "@AGB_L4A")])
tile_provider = get_provider(Vendors.ESRI_IMAGERY)
p_heatmap.add_tile(tile_provider)

p_heatmap.hexbin(x='x', y='y', source=source_heatmap, size=0.01, hover_color="pink", hover_alpha=0.8, fill_color=mapper)

color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))
p_heatmap.add_layout(color_bar, 'right')
save(p_heatmap)
show(p_heatmap)

print("All visualizations generated and saved successfully.")
