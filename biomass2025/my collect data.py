import os
import requests
import datetime as dt 
import pandas as pd
import geopandas as gpd
import h5py
import numpy as np
from shapely.geometry import Point
from getpass import getpass

# Define the input directory
inDir = r"C:\Users\mn2n23\OneDrive - University of Southampton\Desktop\SC solutions (summer project)\biomass\newforrest"

# DOI and API setup
doi_l4a = '10.3334/ORNLDAAC/2056'  # GEDI L4A DOI
doi_l2b = '10.3334/ORNLDAAC/2057'  # GEDI L2B DOI

# CMR API base url
cmrurl = 'https://cmr.earthdata.nasa.gov/search/' 

def get_concept_id(doi):
    doisearch = cmrurl + 'collections.json?doi=' + doi
    response = requests.get(doisearch)
    response.raise_for_status()
    data = response.json()
    if 'feed' in data and 'entry' in data['feed'] and data['feed']['entry']:
        return data['feed']['entry'][0]['id']
    else:
        print(f"No entries found for DOI: {doi}")
        print(f"Response content: {data}")
        raise ValueError("Concept ID not found.")

concept_id_l4a = get_concept_id(doi_l4a)
concept_id_l2b = get_concept_id(doi_l2b)

# Bounding box and time bounds
bound = (-1.75, 50.70, 1.55, 52.15)
start_date = dt.datetime(2020, 7, 1)  # specify your own start date
end_date = dt.datetime(2023, 3, 15)  # specify your end date
dt_format = '%Y-%m-%dT%H:%M:%SZ'
temporal_str = start_date.strftime(dt_format) + ',' + end_date.strftime(dt_format)
bound_str = ','.join(map(str, bound))
page_size = 2000  # CMR page size limit

def get_granules(concept_id, bound_str, temporal_str, page_num):
    granulesearch = f'{cmrurl}granules.json?concept_id={concept_id}&bounding_box={bound_str}&temporal={temporal_str}&page_size={page_size}&page_num={page_num}'
    response = requests.get(granulesearch)
    response.raise_for_status()
    return response.json()['feed']['entry']

def download_granule(url, dest_folder, session):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    response = session.get(url, stream=True)
    response.raise_for_status()
    filename = os.path.join(dest_folder, url.split('/')[-1])
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192): 
            f.write(chunk)
    return filename

# Get Earthdata credentials
username = input("Enter your Earthdata username: ")
password = getpass("Enter your Earthdata password: ")

session = requests.Session()
session.auth = (username, password)

# Authenticate and store cookies in session
url = 'https://urs.earthdata.nasa.gov/oauth/authorize?client_id=eszUe9OMxRl3-290KmSzxA&response_type=code&redirect_uri=https://data.ornldaac.earthdata.nasa.gov/login'
response = session.get(url)
response.raise_for_status()

page_num = 1
granules_l4a = []
granules_l2b = []

while True:
    entries = get_granules(concept_id_l4a, bound_str, temporal_str, page_num)
    if not entries:
        break
    granules_l4a.extend(entries)
    page_num += 1

page_num = 1
while True:
    entries = get_granules(concept_id_l2b, bound_str, temporal_str, page_num)
    if not entries:
        break
    granules_l2b.extend(entries)
    page_num += 1

# Download granules
l4a_dir = os.path.join(inDir, 'GEDI_L4A')
l2b_dir = os.path.join(inDir, 'GEDI_L2B')

l4a_files = [download_granule(g['links'][0]['href'], l4a_dir, session) for g in granules_l4a]
l2b_files = [download_granule(g['links'][0]['href'], l2b_dir, session) for g in granules_l2b]

# Function to extract data from HDF5 files
def extract_data(files, product):
    data = []
    for file in files:
        with h5py.File(file, 'r') as f:
            if product == 'L2B':
                beams = [k for k in f.keys() if k.startswith('BEAM')]
                for beam in beams:
                    lat = f[beam]['geolocation']['lat_lowestmode'][()]
                    lon = f[beam]['geolocation']['lon_lowestmode'][()]
                    agbd = f[beam]['agbd'][()] if 'agbd' in f[beam] else np.nan
                    agbd_se = f[beam]['agbd_se'][()] if 'agbd_se' in f[beam] else np.nan
                    for i in range(len(lat)):
                        data.append([beam, lat[i], lon[i], agbd[i], agbd_se[i]])
            elif product == 'L4A':
                beams = [k for k in f.keys() if k.startswith('BEAM')]
                for beam in beams:
                    lat = f[beam]['lat_lowestmode'][()]
                    lon = f[beam]['lon_lowestmode'][()]
                    agbd = f[beam]['agbd'][()] if 'agbd' in f[beam] else np.nan
                    agbd_se = f[beam]['agbd_se'][()] if 'agbd_se' in f[beam] else np.nan
                    for i in range(len(lat)):
                        data.append([beam, lat[i], lon[i], agbd[i], agbd_se[i]])
    return data

# Extract data from downloaded files
l4a_data = extract_data(l4a_files, 'L4A')
l2b_data = extract_data(l2b_files, 'L2B')

# Convert to DataFrame
columns = ['Beam', 'Latitude', 'Longitude', 'AGBD', 'AGBD_SE']
l4a_df = pd.DataFrame(l4a_data, columns=columns)
l2b_df = pd.DataFrame(l2b_data, columns=columns)

# Convert to GeoDataFrame
l4a_gdf = gpd.GeoDataFrame(l4a_df, geometry=gpd.points_from_xy(l4a_df.Longitude, l4a_df.Latitude), crs='EPSG:4326')
l2b_gdf = gpd.GeoDataFrame(l2b_df, geometry=gpd.points_from_xy(l2b_df.Longitude, l2b_df.Latitude), crs='EPSG:4326')

# Save to files
l4a_gdf.to_file(os.path.join(inDir, 'GEDI_L4A_within_bbox.shp'))
l2b_gdf.to_file(os.path.join(inDir, 'GEDI_L2B_within_bbox.shp'))

print("Data extraction and saving completed successfully.")
