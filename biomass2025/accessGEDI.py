%matplotlib inline
import h5py
import tabulate
import contextily as ctx
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from IPython.display import HTML, display
from os import path, listdir, access, R_OK
import tarfile
import requests
from shapely.geometry import Point, Polygon, box
from shapely.ops import orient
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.lines import Line2D
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from scipy import stats
import cartopy.crs as ccrs
import rasterio
from rasterio import plot
import richdem as rd

plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('mode.chained_assignment', None)

# Define the directory where the HDF5 files are stored
data_directory = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\dataaa'

# Get a list of all HDF5 files in the directory
l4aList = [f for f in listdir(data_directory) if f.endswith('.h5')]

hfList = []

# Read the L4A files
for l4a in l4aList:
    file_path = path.join(data_directory, l4a)
    if path.exists(file_path):
        hfList.append(h5py.File(file_path, 'r'))
    else:
        print(f"File not found: {file_path}")

# Print root-level groups of each file
for i, hf in enumerate(hfList):
    print(f"File {i+1}: {l4aList[i]}")
    print(list(hf.keys()))

# Read the METADATA group from the first file
if hfList:
    metadata = hfList[0]['METADATA/DatasetIdentification']
    # Store attributes and descriptions in an array
    data = []
    for attr in metadata.attrs.keys():
        data.append([attr, metadata.attrs[attr]])
    
    # Display `data` array as a table
    tbl_n = 1 # Table number
    print(f'Table {tbl_n}. Attributes and description from `METADATA` group')
    headers = ["attribute", "description"]
    display(HTML(tabulate.tabulate(data, headers, tablefmt='html')))
else:
    print("No HDF5 files were successfully loaded.")

# Read the ANCILLARY group from each file and print its subgroups
ancillary = []

for hf in hfList:
    if 'ANCILLARY' in hf:
        ancillary.append(hf['ANCILLARY'])
    else:
        print(f"ANCILLARY group not found in file: {hf.filename}")

# Print the subgroups of the ANCILLARY group from the first file that contains it
if ancillary:
    print("Subgroups in ANCILLARY group of the first file:")
    print(list(ancillary[0].keys()))
else:
    print("No ANCILLARY groups found.")

# Read model_data subgroup
model_data = []
for data in ancillary:
    if 'model_data' in data:
        model_data.append(data['model_data'])
    else:
        print(f"model_data subgroup not found in ANCILLARY group of file: {data.file.filename}")

# Print variables, data types, and data dimensions of model_data from the first file that contains it
if model_data:
    print("Variables, data types, and data dimensions of model_data in the first file:")
    print(model_data[0].dtype)
else:
    print("No model_data subgroups found.")

# Initialize an empty DataFrame
model_data_df = pd.DataFrame()

# Populate the DataFrame with parameters from the first model_data subgroup
if model_data:
    first_model = model_data[0]
    for v in first_model.dtype.names:
        # Exclude multidimensional variables
        if (len(first_model[v].shape) == 1):
            # Copy parameters as DataFrame columns
            model_data_df[v] = first_model[v]
            # Convert object datatype to string
            if model_data_df[v].dtype.kind == 'O':
                model_data_df[v] = model_data_df[v].str.decode('utf-8')
    
    # Print the parameters
    tbl_n += 1
    print(f'Table {tbl_n}. Parameters and their values in `model_data` subgroup')
    display(model_data_df)
else:
    print("No model_data subgroups found to populate the DataFrame.")

# Read pft_lut subgroup
if ancillary:
    pft_lut = ancillary[0]['pft_lut']
    headers = pft_lut.dtype.names
    # Print pft class and names
    data = zip(pft_lut[headers[0]], pft_lut[headers[1]])
    tbl_n += 1
    print(f'Table {tbl_n}. PFT class and names in `pft_lut` subgroup')
    display(HTML(tabulate.tabulate(data, headers, tablefmt='html')))
else:
    print("No ANCILLARY groups found to read pft_lut subgroup.")

# Index of DBT_NAm predict_stratum
if 'predict_stratum' in model_data_df:
    idx = model_data_df[model_data_df['predict_stratum'] == 'DBT_NAm'].index.item()
    # Print vcov matrix
    tbl_n += 1
    print(f'Table {tbl_n}. VCOV matrix for predict_stratum "DBT_NAm"')
    vcov_matrix = model_data[0]['vcov'][idx]
    display(vcov_matrix)
    
    # Get predictor_id, rh_index and par for idx = 6
    predictor_id = model_data[0]['predictor_id'][idx]
    rh_index = model_data[0]['rh_index'][idx]
    par = model_data[0]['par'][idx]

    # Print predictor_id, rh_index, and par
    print_s = f"""predictor_id: {predictor_id}
rh_index: {rh_index}
par: {par}"""
    print(print_s)
else:
    print("predict_stratum 'DBT_NAm' not found in model_data_df.")

# Initialize arrays
stratum_arr, modelname_arr, fitstratum_arr, agbd_arr = [], [], [], []

# Loop through the model_data_df dataframe
for idx, row in model_data_df.iterrows():
    stratum_arr.append(model_data_df['predict_stratum'][idx])
    modelname_arr.append(model_data_df['model_name'][idx])
    fitstratum_arr.append(model_data_df['fit_stratum'][idx])
    i_0 = 0
    predictor_id = model_data[0]['predictor_id'][idx]
    rh_index = model_data[0]['rh_index'][idx]
    par = model_data[0]['par'][idx]
    model_str = 'AGBD = ' + str(par[0]) # intercept
    for i in predictor_id[predictor_id > 0]:
        # Use product of two RH metrics when consecutive predictor_id have same values
        if (i == i_0):
            model_str += ' x RH_' + str(rh_index[i-1])
        # Adding slope coefficients
        else:
            model_str += ' + ' + str(par[i]) + ' x RH_' + str(rh_index[i-1])
        i_0 = i
    # AGBD model
    agbd_arr.append(model_str)

# Unique AGBD models
unique_models = list(set(agbd_arr))

# Printing AGBD models by predict_stratum
data = []
for model in unique_models:
    s, m, f = [], [], []
    for i, x in enumerate(agbd_arr):
        if x == model:
            s.append(stratum_arr[i])
            m.append(modelname_arr[i])
            f.append(fitstratum_arr[i])
    data.append([", ".join(s), ", ".join(list(set(m))), ", ".join(list(set(f))), model])

tbl_n += 1
print(f'Table {tbl_n}. AGBD Linear Models by Prediction Stratum')
headers = ["predict_stratum", "model_name", "fit_stratum", "AGBD model"]
display(HTML(tabulate.tabulate(data, headers, tablefmt='html', stralign="left")))

# Collect beam data
data = []
for v in list(hfList[0].keys()):
    if v.startswith('BEAM'):
        beam = hfList[0].get(v)
        b_beam = beam.get('beam')[0]
        channel = beam.get('channel')[0]
        data.append([v, hfList[0][v].attrs['description'], b_beam, channel])

# Print as a table
tbl_n += 1
print(f'Table {tbl_n}. GEDI Beams')
headers = ["beam name", "description", "beam", "channel"]
display(HTML(tabulate.tabulate(data, headers, tablefmt='html')))

# Read variables within BEAM0110 group
beam_str = ['BEAM0101', 'BEAM0110', 'BEAM1000', 'BEAM1011']
beam0110 = hfList[0][beam_str[1]]

data = []
# Loop over all the variables within BEAM0110 group
for v in beam0110.keys():
    var = beam0110[v]
    source = ''
    # If the key is a subgroup assign GROUP tag
    if isinstance(var, h5py.Group):
        data.append([v, 'GROUP', 'GROUP', 'GROUP'])
    # Read source, description, units attributes of each variable
    else:
        if 'source' in var.attrs.keys():
            source = var.attrs['source']
        data.append([v, var.attrs['description'], var.attrs['units'], source])

# Print all variable names and attributes as a table
tbl_n += 1
print(f'Table {tbl_n}. Variables within {beam_str[1]} group')
headers = ["variable", "description", "units", "source"]
data = sorted(data, key=lambda x: x[3])
display(HTML(tabulate.tabulate(data, headers, tablefmt='html')))

# Initialize lists
elev_l = []
lat_l = []
lon_l = []
agbd_l = []
error_l = []
beam_n = []
time_l = []
quality_l = []
    
for hf in hfList:
    # Loop over all base groups
    for var in list(hf.keys()):
        if var.startswith('BEAM'):
            # Reading lat, lon, time
            beam = hf.get(var)
            if beam.get('agbd') is not None:
                agbd = beam.get('agbd')[:] # biomass estimation
                error = beam.get('agbd_se')[:] # standard error of AGBD
                elev = beam.get('elev_lowestmode')[:] # elevation
                lat = beam.get('lat_lowestmode')[:] # latitude
                lon = beam.get('lon_lowestmode')[:] # longitude
                time = beam.get('delta_time')[:] # time
                quality = beam.get('l4_quality_flag')[:] # quality
                
                # Appending each beam into the array
                agbd_l.extend(agbd.tolist())
                error_l.extend(error.tolist())
                elev_l.extend(elev.tolist())
                lat_l.extend(lat.tolist()) 
                lon_l.extend(lon.tolist()) 
                time_l.extend(time.tolist()) 
                quality_l.extend(quality.tolist())
                
                # beam_n as a new column indicating beam number
                n = lat.shape[0] # number of shots in the beam group
                beam_n.extend(np.repeat(str(var), n).tolist())

# Read the lists into a dataframe
l4adf = pd.DataFrame(list(zip(beam_n, agbd_l, error_l, elev_l, lat_l, lon_l, time_l, quality_l)), 
                     columns=["beam", "agbd", "agbd_se", "elev_lowestmode", "lat_lowestmode", "lon_lowestmode", "delta_time", "l4_quality_flag"])

# Display the dataframe
print(l4adf.tail())
print(l4adf.describe())

# First filtration of the data based on GEDI data exclusively 
min_lon = -1.75
max_lon = 1.55
min_lat = 50.70
max_lat = 52.15

# Filter out invalid values
l4adf = l4adf[(l4adf['agbd'] != -9999) & (l4adf['agbd_se'] != -9999)]

# Remove relative error condition for now
filtered = l4adf[(l4adf['lat_lowestmode'] > min_lat) & (l4adf['lat_lowestmode'] < max_lat) & (l4adf['lon_lowestmode'] > min_lon) & (l4adf['lon_lowestmode'] < max_lon) & (l4adf['l4_quality_flag'] > 0)]

# Debug: Print the number of rows in the filtered DataFrame
print(f"Number of rows after filtering: {filtered.shape[0]}")

latitudes = filtered['lat_lowestmode'].values
longitudes = filtered['lon_lowestmode'].values

plt.figure(figsize=(10, 6))

plt.scatter(longitudes, latitudes, color='blue', marker='o')

plt.gca().set_facecolor('white')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Plotting Latitudes and Longitudes')

plt.show()

# Display summary of filtered data
print(filtered.describe())

# Paths to the tar.gz files
rasters_tar_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\open\rasters_COP30.tar.gz'
viz_tar_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\open\viz.tar.gz'
extract_path = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\open'

# Extract rasters_COP30.tar.gz
if not path.exists(extract_path):
    with tarfile.open(rasters_tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

# Extract viz.tar.gz (if needed)
with tarfile.open(viz_tar_path, 'r:gz') as tar:
    tar.extractall(path=extract_path)

# Path to the extracted .tif file
dem_file = path.join(extract_path, 'output_COP30.tif')

# Check if the DEM file is accessible
if path.exists(dem_file) and access(dem_file, R_OK):
    dem_data = rasterio.open(dem_file)
    dem_array = rd.rdarray(dem_data.read(1), no_data=-9999)
    slopes = rd.TerrainAttribute(dem_array, attrib='slope_degrees')
    transform = dem_data.transform

    def coords_to_index(lat, lon, transform):
        # the inverse function to the transformation matrix to undo the transformation and convert lon/lat to rows/col on the tiff file
        col, row = ~transform * (lon, lat)
        return int(row), int(col)

    def get_slope(lat, lon, transform, slopes):
        row, col = coords_to_index(lat, lon, transform)
        if 0 <= row < slopes.shape[0] and 0 <= col < slopes.shape[1]:
            return slopes[row, col]
        else:
            return np.nan  

    # Create a copy of the filtered DataFrame for slope filtering
    filtered2 = filtered.copy()

    # Add slope values to filtered2
    filtered2['slope'] = filtered2.apply(lambda row: get_slope(row['lat_lowestmode'], row['lon_lowestmode'], transform, slopes), axis=1)
    
    # Filter based on slope
    filtered2 = filtered2[filtered2['slope'] < 30]

    # Display summary of the slope-filtered data
    print(filtered2.describe())
else:
    print(f"DEM file '{dem_file}' is not accessible.")

# Paths to the WorldCover files
worldcover_directory = r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\SOuthEast'
map_files = [
    'ESA_WorldCover_10m_2020_v100_N48E000_Map.tif',
    'ESA_WorldCover_10m_2020_v100_N48W003_Map.tif',
    'ESA_WorldCover_10m_2020_v100_N51E000_Map.tif',
    'ESA_WorldCover_10m_2020_v100_N51W003_Map.tif'
]

# Read and process the WorldCover files
for index, map_file in enumerate(map_files):
    map_path = path.join(worldcover_directory, map_file)
    if path.exists(map_path):
        map_dataset = rasterio.open(map_path)
        layer1_map = map_dataset.read(1)
        transform = map_dataset.transform

        def get_map_code(lat, lon):
            col, row = ~transform * (lon, lat)
            try:
                if 0 <= int(row) < map_dataset.height and 0 <= int(col) < map_dataset.width:
                    map_code = layer1_map[int(row), int(col)]
                    return map_code
            except IndexError:
                return np.nan

        filtered2[f'map_code_{index}'] = filtered2.apply(lambda row: get_map_code(row['lat_lowestmode'], row['lon_lowestmode']), axis=1)
    else:
        print(f"File not found: {map_path}")

# Combine the map codes into a single column
filtered2['map_code'] = filtered2[[f'map_code_{i}' for i in range(len(map_files))]].sum(axis=1, skipna=True)

# Drop the intermediate map_code columns
filtered2.drop(columns=[f'map_code_{i}' for i in range(len(map_files))], inplace=True)

# Filter out specific map codes
filtered2 = filtered2[(filtered2['map_code'] != 50) & (filtered2['map_code'] != 60) & (filtered2['map_code'] != 70) & (filtered2['map_code'] != 80) & (filtered2['map_code'] != 90) & (filtered2['map_code'] != 100)]

# Display the final summary of filtered data
print(filtered2.describe())
agbd_data = filtered2['agbd']

plt.figure(figsize=(10, 6)) 
plt.hist(agbd_data, bins=100, color='green', edgecolor='black')  
plt.xlabel('AGBD Values')
plt.ylabel('Frequency')
plt.title('AGBD')
plt.grid(True) 
plt.show()

STEP = 100

gdf = gpd.GeoDataFrame(
    filtered2, 
    geometry=gpd.points_from_xy(filtered2.lon_lowestmode, filtered2.lat_lowestmode),
    crs="EPSG:4326"
)

bbox = gpd.GeoDataFrame(
    {'geometry': [Polygon([(min_lon, min_lat), (min_lon, max_lat), 
                           (max_lon, max_lat), (max_lon, min_lat), 
                           (min_lon, min_lat)])]},
    crs="EPSG:4326"
)

utm_crs = bbox.estimate_utm_crs()

min_x, min_y, max_x, max_y = bbox.to_crs(utm_crs).total_bounds

mid_x = (min_x + max_x) / 2
mid_y = (min_y + max_y) / 2

quadrants = [
    (min_x, min_y, mid_x, mid_y), 
    (mid_x, min_y, max_x, mid_y), 
    (min_x, mid_y, mid_x, max_y), 
    (mid_x, mid_y, max_x, max_y)   
]
def create_grid(min_x, min_y, max_x, max_y, step):
    return [
        box(minx, miny, maxx, maxy)
        for minx, maxx in zip(np.arange(min_x, max_x, step), np.arange(min_x, max_x, step)[1:])
        for miny, maxy in zip(np.arange(min_y, max_y, step), np.arange(min_y, max_y, step)[1:])
    ]
    
results = []
#for bottom left quadrant, each quadrant seperated to avoid runtime error
grid = create_grid(min_x, min_y, mid_x, mid_y, STEP)
gdf_grid = gpd.GeoDataFrame(geometry=grid, crs=utm_crs).to_crs(bbox.crs)
gdf_joined = gpd.sjoin(gdf, gdf_grid, how="left", predicate="within")
quadrant_result = (
        gdf_joined.groupby("index_right")
        .agg(count=("agbd", "size"), avg_agbd=("agbd", "mean"))
        .reset_index()
)
results.append(quadrant_result)

#for bottom right
grid = create_grid(mid_x, min_y, max_x, mid_y, STEP)
gdf_grid = gpd.GeoDataFrame(geometry=grid, crs=utm_crs).to_crs(bbox.crs)
gdf_joined = gpd.sjoin(gdf, gdf_grid, how="left", predicate="within")
quadrant_result = (
        gdf_joined.groupby("index_right")
        .agg(count=("agbd", "size"), avg_agbd=("agbd", "mean"))
        .reset_index()
)
results.append(quadrant_result)
#for top left
grid = create_grid(min_x, mid_y, mid_x, max_y, STEP)
gdf_grid = gpd.GeoDataFrame(geometry=grid, crs=utm_crs).to_crs(bbox.crs)
gdf_joined = gpd.sjoin(gdf, gdf_grid, how="left", predicate="within")
quadrant_result = (
        gdf_joined.groupby("index_right")
        .agg(count=("agbd", "size"), avg_agbd=("agbd", "mean"))
        .reset_index()
)
results.append(quadrant_result)
#for top right
grid = create_grid(mid_x, mid_y, max_x, max_y, STEP)
gdf_grid = gpd.GeoDataFrame(geometry=grid, crs=utm_crs).to_crs(bbox.crs)
gdf_joined = gpd.sjoin(gdf, gdf_grid, how="left", predicate="within")
quadrant_result = (
        gdf_joined.groupby("index_right")
        .agg(count=("agbd", "size"), avg_agbd=("agbd", "mean"))
        .reset_index()
)
results.append(quadrant_result)
final_result = pd.concat(results, ignore_index=True)
result_filtered = final_result[final_result['count'] > 2]

final_result_2 = gdf_grid.merge(result_filtered, left_index=True, right_on="index_right")
final_result_2 = final_result_2.drop(columns=["index_right"])

final_result_2['bbox'] = final_result_2.geometry.apply(lambda geom: geom.bounds)

final_result_2.describe()
final_result_2.to_csv("AGBD_cells.csv", index=False)
