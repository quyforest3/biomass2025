import os
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx

# Load the data
df = pd.read_csv('gedi_agb_data_within_roi.csv')

# Convert the geometry column to Shapely Point objects
df['geometry'] = df['geometry'].apply(lambda x: Point(map(float, x.strip('POINT ()').split())))

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geometry')

# Manually set the CRS to EPSG:4326
gdf.crs = 'EPSG:4326'

# Histogram of AGB values
plt.figure(figsize=(10, 6))
plt.hist(df['AGB_L4A'], bins=50, color='#4CAF50', edgecolor='#388E3C')  # Green tones for growth and balance
plt.xlabel('AGB Values', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of AGB Values', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('histogram_agb_values.png')
plt.show()

# Scatter plot of AGB vs Quality Flag
plt.figure(figsize=(10, 6))
plt.scatter(df['Quality Flag'], df['AGB_L4A'], alpha=0.5, c='#FF5722')  # Orange tones for visibility and energy
plt.xlabel('Quality Flag', fontsize=14)
plt.ylabel('AGB Values', fontsize=14)
plt.title('Scatter Plot of AGB vs Quality Flag', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('scatter_agb_vs_quality_flag.png')
plt.show()

# Box plot of AGB values by Beam
plt.figure(figsize=(12, 6))
df.boxplot(column='AGB_L4A', by='Beam', grid=False, showfliers=False, color=dict(boxes='#2196F3', whiskers='#1E88E5', medians='#1976D2', caps='#1565C0'))  # Blue tones for trust and stability
plt.xlabel('Beam', fontsize=14)
plt.ylabel('AGB Values', fontsize=14)
plt.title('Box Plot of AGB Values by Beam', fontsize=16)
plt.suptitle('')  # Remove the default title to avoid overlap
plt.xticks(rotation=90, fontsize=12)
plt.savefig('boxplot_agb_by_beam.png')
plt.show()

# Extract x and y coordinates from the geometry column
df['x'] = df.geometry.apply(lambda geom: geom.x)
df['y'] = df.geometry.apply(lambda geom: geom.y)

# Heatmap of AGB values
plt.figure(figsize=(10, 10))
plt.hexbin(df['x'], df['y'], C=df['AGB_L4A'], gridsize=50, cmap='viridis', reduce_C_function=np.mean)  # Viridis colormap for perceptually uniform colors
plt.colorbar(label='Mean AGB')
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.title('Heatmap of Mean AGB Values', fontsize=16)
plt.savefig('heatmap_agb_values.png')
plt.show()

# Time-series plot of AGB values (assuming the data has a 'Date' column in datetime format)
if 'Date' in df.columns:
    plt.figure(figsize=(10, 6))
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date')['AGB_L4A'].resample('M').mean().plot(color='#673AB7')  # Purple tones for calm and creativity
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Mean AGB Values', fontsize=14)
    plt.title('Time-Series Plot of Mean AGB Values', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('timeseries_agb_values.png')
    plt.show()

# Save the plots
print("All visualizations generated and saved successfully.")
