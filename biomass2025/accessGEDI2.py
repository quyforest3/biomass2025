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

def load_hdf5_files(directory):
    files = [f for f in listdir(directory) if f.endswith('.h5')]
    hdf5_files = []
    for f in files:
        file_path = path.join(directory, f)
        if path.exists(file_path):
            hdf5_files.append(h5py.File(file_path, 'r'))
        else:
            print(f"File not found: {file_path}")
    return hdf5_files

def display_metadata(hf_list):
    if hf_list:
        metadata = hf_list[0]['METADATA/DatasetIdentification']
        data = [[attr, metadata.attrs[attr]] for attr in metadata.attrs.keys()]
        headers = ["attribute", "description"]
        display(HTML(tabulate.tabulate(data, headers, tablefmt='html')))
    else:
        print("No HDF5 files were successfully loaded.")

def read_model_data(ancillary):
    model_data = []
    for data in ancillary:
        if 'model_data' in data:
            model_data.append(data['model_data'])
        else:
            print(f"model_data subgroup not found in ANCILLARY group of file: {data.file.filename}")
    return model_data

def populate_model_data_df(model_data):
    df = pd.DataFrame()
    if model_data:
        first_model = model_data[0]
        for v in first_model.dtype.names:
            if len(first_model[v].shape) == 1:
                df[v] = first_model[v]
                if df[v].dtype.kind == 'O':
                    df[v] = df[v].str.decode('utf-8')
    return df

def extract_beam_data(hf_list):
    elev_l, lat_l, lon_l, agbd_l, error_l, beam_n, time_l, quality_l = [], [], [], [], [], [], [], []
    for hf in hf_list:
        for var in list(hf.keys()):
            if var.startswith('BEAM'):
                beam = hf.get(var)
                if beam.get('agbd') is not None:
                    agbd_l.extend(beam.get('agbd')[:].tolist())
                    error_l.extend(beam.get('agbd_se')[:].tolist())
                    elev_l.extend(beam.get('elev_lowestmode')[:].tolist())
                    lat_l.extend(beam.get('lat_lowestmode')[:].tolist())
                    lon_l.extend(beam.get('lon_lowestmode')[:].tolist())
                    time_l.extend(beam.get('delta_time')[:].tolist())
                    quality_l.extend(beam.get('l4_quality_flag')[:].tolist())
                    n = beam.get('lat_lowestmode').shape[0]
                    beam_n.extend(np.repeat(str(var), n).tolist())
    return pd.DataFrame(list(zip(beam_n, agbd_l, error_l, elev_l, lat_l, lon_l, time_l, quality_l)), 
                        columns=["beam", "agbd", "agbd_se", "elev_lowestmode", "lat_lowestmode", "lon_lowestmode", "delta_time", "l4_quality_flag"])

def filter_data(df, min_lon, max_lon, min_lat, max_lat):
    return df[(df['lat_lowestmode'] > min_lat) & (df['lat_lowestmode'] < max_lat) & 
              (df['lon_lowestmode'] > min_lon) & (df['lon_lowestmode'] < max_lon) & 
              (df['l4_quality_flag'] > 0) & 
              (df['agbd'] != -9999) & 
              (df['agbd_se'] != -9999)]

def extract_and_filter_dem_data(rasters_tar_path, viz_tar_path, extract_path, filtered):
    if not path.exists(extract_path):
        with tarfile.open(rasters_tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        with tarfile.open(viz_tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
    
    dem_file = path.join(extract_path, 'output_COP30.tif')
    if path.exists(dem_file) and access(dem_file, R_OK):
        dem_data = rasterio.open(dem_file)
        dem_array = rd.rdarray(dem_data.read(1), no_data=-9999)
        slopes = rd.TerrainAttribute(dem_array, attrib='slope_degrees')
        transform = dem_data.transform

        def get_slope(lat, lon, transform, slopes):
            row, col = ~transform * (lon, lat)
            if 0 <= int(row) < slopes.shape[0] and 0 <= int(col) < slopes.shape[1]:
                return slopes[int(row), int(col)]
            return np.nan

        filtered['slope'] = filtered.apply(lambda row: get_slope(row['lat_lowestmode'], row['lon_lowestmode'], transform, slopes), axis=1)
        return filtered[filtered['slope'] < 30]
    else:
        print(f"DEM file '{dem_file}' is not accessible.")
        return pd.DataFrame()

def process_worldcover_data(worldcover_directory, map_files, filtered2):
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
                        return layer1_map[int(row), int(col)]
                except IndexError:
                    return np.nan

            filtered2[f'map_code_{index}'] = filtered2.apply(lambda row: get_map_code(row['lat_lowestmode'], row['lon_lowestmode']), axis=1)
        else:
            print(f"File not found: {map_path}")

    filtered2['map_code'] = filtered2[[f'map_code_{i}' for i in range(len(map_files))]].sum(axis=1, skipna=True)
    filtered2.drop(columns=[f'map_code_{i}' for i in range(len(map_files))], inplace=True)
    return filtered2[(filtered2['map_code'] != 50) & (filtered2['map_code'] != 60) & (filtered2['map_code'] != 70) & (filtered2['map_code'] != 80) & (filtered2['map_code'] != 90) & (filtered2['map_code'] != 100)]

def plot_agbd_histogram(agbd_data):
    plt.figure(figsize=(10, 6)) 
    plt.hist(agbd_data, bins=100, color='green', edgecolor='black')  
    plt.xlabel('AGBD Values')
    plt.ylabel('Frequency')
    plt.title('AGBD')
    plt.grid(True) 
    plt.show()

def create_grid(min_x, min_y, max_x, max_y, step):
    return [
        box(minx, miny, maxx, maxy)
        for minx, maxx in zip(np.arange(min_x, max_x, step), np.arange(min_x, max_x, step)[1:])
        for miny, maxy in zip(np.arange(min_y, max_y, step), np.arange(min_y, max_y, step)[1:])
    ]

def process_grid(gdf, bbox, utm_crs, quadrants, step):
    results = []
    for (min_x, min_y, max_x, max_y) in quadrants:
        grid = create_grid(min_x, min_y, max_x, max_y, step)
        gdf_grid = gpd.GeoDataFrame(geometry=grid, crs=utm_crs).to_crs(bbox.crs)
        gdf_joined = gpd.sjoin(gdf, gdf_grid, how="left", predicate="within")
        quadrant_result = (
            gdf_joined.groupby("index_right")
            .agg(count=("agbd", "size"), avg_agbd=("agbd", "mean"))
            .reset_index()
        )
        results.append(quadrant_result)
    return pd.concat(results, ignore_index=True)

# Load HDF5 files
hf_list = load_hdf5_files(data_directory)
display_metadata(hf_list)

# Read ANCILLARY group and model data
ancillary = [hf['ANCILLARY'] for hf in hf_list if 'ANCILLARY' in hf]
model_data = read_model_data(ancillary)
model_data_df = populate_model_data_df(model_data)
display(model_data_df)

# Extract beam data and filter
l4adf = extract_beam_data(hf_list)
filtered = filter_data(l4adf, -1.75, 1.55, 50.70, 52.15)

# Extract and filter DEM data
filtered2 = extract_and_filter_dem_data(
    r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\open\rasters_COP30.tar.gz',
    r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\open\viz.tar.gz',
    r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\open',
    filtered
)

# Process WorldCover data
filtered2 = process_worldcover_data(
    r'C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest\SOuthEast',
    [
        'ESA_WorldCover_10m_2020_v100_N48E000_Map.tif',
        'ESA_WorldCover_10m_2020_v100_N48W003_Map.tif',
        'ESA_WorldCover_10m_2020_v100_N51E000_Map.tif',
        'ESA_WorldCover_10m_2020_v100_N51W003_Map.tif'
    ],
    filtered2
)

# Plot AGBD histogram
plot_agbd_histogram(filtered2['agbd'])

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(filtered2, geometry=gpd.points_from_xy(filtered2.lon_lowestmode, filtered2.lat_lowestmode), crs="EPSG:4326")
bbox = gpd.GeoDataFrame({'geometry': [Polygon([(-1.75, 50.70), (-1.75, 52.15), (1.55, 52.15), (1.55, 50.70), (-1.75, 50.70)])]}, crs="EPSG:4326")
utm_crs = bbox.estimate_utm_crs()
min_x, min_y, max_x, max_y = bbox.to_crs(utm_crs).total_bounds
mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2

# Define quadrants
quadrants = [(min_x, min_y, mid_x, mid_y), (mid_x, min_y, max_x, mid_y), (min_x, mid_y, mid_x, max_y), (mid_x, mid_y, max_x, max_y)]

# Process grid and filter results
final_result = process_grid(gdf, bbox, utm_crs, quadrants, 100)
result_filtered = final_result[final_result['count'] > 2]
final_result_2 = gpd.GeoDataFrame(geometry=create_grid(min_x, min_y, max_x, max_y, 100), crs=utm_crs).merge(result_filtered, left_index=True, right_on="index_right")
final_result_2.drop(columns=["index_right"], inplace=True)
final_result_2['bbox'] = final_result_2.geometry.apply(lambda geom: geom.bounds)

# Display and save final result
print(final_result_2.describe())
final_result_2.to_csv("AGBD_cells.csv", index=False)
