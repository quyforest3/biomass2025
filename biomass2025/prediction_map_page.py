"""
Professional Prediction Map Page with Legend and Advanced Features
"""

import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import rasterio
from rasterio.plot import show
import os


def create_legend_html():
    """Create a professional legend for the prediction map"""
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 50px; 
        right: 10px; 
        width: 280px; 
        background-color: white; 
        border:2px solid grey; 
        border-radius: 10px;
        z-index:9999; 
        font-size:14px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    ">
        <div style="
            border-bottom: 2px solid #2E8B57;
            padding-bottom: 10px;
            margin-bottom: 10px;
            font-weight: bold;
            color: #2E8B57;
            font-size: 16px;
        ">
            📊 AGB Prediction Legend
        </div>
        
        <div style="margin-bottom: 15px;">
            <div style="font-weight: bold; color: #333; margin-bottom: 8px;">🎯 Biomass Categories (Mg/ha)</div>
            
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="
                    width: 20px; 
                    height: 20px; 
                    background: #ffffcc;
                    margin-right: 8px;
                    border-radius: 3px;
                    border: 1px solid #ccc;
                "></div>
                <span>Very Low (0 - 30 Mg/ha)</span>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="
                    width: 20px; 
                    height: 20px; 
                    background: #addd8e;
                    margin-right: 8px;
                    border-radius: 3px;
                    border: 1px solid #ccc;
                "></div>
                <span>Low (30 - 80 Mg/ha)</span>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="
                    width: 20px; 
                    height: 20px; 
                    background: #78c679;
                    margin-right: 8px;
                    border-radius: 3px;
                    border: 1px solid #ccc;
                "></div>
                <span>Medium (80 - 150 Mg/ha)</span>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="
                    width: 20px; 
                    height: 20px; 
                    background: #31a354;
                    margin-right: 8px;
                    border-radius: 3px;
                    border: 1px solid #ccc;
                "></div>
                <span>High (150 - 200 Mg/ha)</span>
            </div>
            
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="
                    width: 20px; 
                    height: 20px; 
                    background: #006837;
                    margin-right: 8px;
                    border-radius: 3px;
                    border: 1px solid #333;
                "></div>
                <span style="color: white;">Very High (200+ Mg/ha)</span>
            </div>
        </div>
        
        <div style="
            border-top: 2px solid #e0e0e0;
            border-bottom: 2px solid #e0e0e0;
            padding: 10px 0;
            margin-bottom: 10px;
        ">
            <div style="font-weight: bold; color: #333; margin-bottom: 8px;">🔧 Map Controls</div>
            <ul style="margin: 5px 0; padding-left: 20px; font-size: 12px;">
                <li>🖱️ Drag to move map</li>
                <li>🔍 Scroll to zoom</li>
                <li>📍 Click for details</li>
                <li>🗺️ Switch layers</li>
            </ul>
        </div>
        
        <div style="
            font-size: 12px;
            color: #666;
            text-align: center;
            padding-top: 8px;
            border-top: 1px solid #e0e0e0;
        ">
            <div style="margin-bottom: 4px;">Data Source: GEDI & Sentinel-2</div>
            <div>Model: XGBoost Ensemble</div>
        </div>
    </div>
    """
    return legend_html


def create_colorbar_html(min_val, max_val):
    """Create a professional colorbar for the legend"""
    colorbar_html = f"""
    <div style="
        position: fixed; 
        bottom: 50px; 
        left: 10px; 
        width: 300px; 
        background-color: white; 
        border:2px solid grey; 
        border-radius: 10px;
        z-index:9999; 
        font-size:12px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    ">
        <div style="
            border-bottom: 2px solid #2E8B57;
            padding-bottom: 8px;
            margin-bottom: 10px;
            font-weight: bold;
            color: #2E8B57;
        ">
            📏 AGB Value Range (Mg/ha)
        </div>
        
        <div style="
            height: 30px;
            background: linear-gradient(90deg, 
                #440154 0%, 
                #31688e 25%, 
                #35b779 50%, 
                #fde724 75%, 
                #ff0000 100%);
            border: 1px solid #ccc;
            border-radius: 5px;
            margin-bottom: 8px;
        "></div>
        
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="font-weight: bold;">{min_val:.1f}</span>
            <span style="color: #666; font-size: 10px;">Biomass Values</span>
            <span style="font-weight: bold;">{max_val:.1f}</span>
        </div>
        
        <div style="
            padding-top: 10px;
            border-top: 1px solid #e0e0e0;
            font-size: 11px;
            color: #666;
            line-height: 1.4;
        ">
            <strong>Color Interpretation:</strong>
            <ul style="margin: 5px 0; padding-left: 18px;">
                <li>🟣 Purple: Low biomass</li>
                <li>🟢 Green: Moderate biomass</li>
                <li>🟡 Yellow: High biomass</li>
                <li>🔴 Red: Very high biomass</li>
            </ul>
        </div>
    </div>
    """
    return colorbar_html


def create_statistics_popup(stats_dict):
    """Create a popup with detailed statistics"""
    popup_html = """
    <div style="
        font-family: Arial, sans-serif;
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        width: 280px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    ">
        <h4 style="
            color: #2E8B57;
            margin: 0 0 10px 0;
            border-bottom: 2px solid #2E8B57;
            padding-bottom: 8px;
        ">📊 Prediction Details</h4>
        
        <table style="
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        ">
            <tr style="background-color: #e8f5e9;">
                <td style="padding: 6px 8px; font-weight: bold; color: #2E8B57; border: 1px solid #e0e0e0;">AGB Value:</td>
                <td style="padding: 6px 8px; text-align: right; border: 1px solid #e0e0e0;">""" + str(stats_dict.get('agb', 'N/A')) + """ Mg/ha</td>
            </tr>
            <tr style="background-color: #f5f5f5;">
                <td style="padding: 6px 8px; color: #666; border: 1px solid #e0e0e0;">Latitude:</td>
                <td style="padding: 6px 8px; text-align: right; border: 1px solid #e0e0e0;">""" + str(round(stats_dict.get('lat', 0), 6)) + """</td>
            </tr>
            <tr style="background-color: #e8f5e9;">
                <td style="padding: 6px 8px; color: #666; border: 1px solid #e0e0e0;">Longitude:</td>
                <td style="padding: 6px 8px; text-align: right; border: 1px solid #e0e0e0;">""" + str(round(stats_dict.get('lon', 0), 6)) + """</td>
            </tr>
            <tr style="background-color: #f5f5f5;">
                <td style="padding: 6px 8px; color: #666; border: 1px solid #e0e0e0;">Category:</td>
                <td style="padding: 6px 8px; text-align: right; border: 1px solid #e0e0e0;">""" + str(stats_dict.get('category', 'N/A')) + """</td>
            </tr>
        </table>
        
        <div style="
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #ddd;
            font-size: 11px;
            color: #999;
            text-align: center;
        ">
            Generated using XGBoost Model
        </div>
    </div>
    """
    return popup_html


def get_agb_category(value):
    """Classify AGB value into categories"""
    if value < 50:
        return "Very Low"
    elif value < 100:
        return "Low"
    elif value < 150:
        return "Medium"
    else:
        return "High"


def create_prediction_map_page():
    """Create a professional prediction map page with all features"""
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #2E8B57, #3CB371);
        color: white;
        padding: 25px;
        border-radius: 10px;
        margin-bottom: 20px;
    ">
        <h1 style="margin: 0; font-size: 2.5rem;">🗺️ AGB Prediction Map</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.1rem; opacity: 0.95;">
            Professional-Grade Biomass Prediction Visualization with Advanced Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different map views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Interactive Map",
        "📊 Statistics",
        "🎨 Heatmap",
        "📈 Analysis"
    ])
    
    with tab1:
        st.subheader("Interactive Prediction Map with Legend")
        
        # Map configuration controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            map_style = st.selectbox(
                "🎨 Map Style",
                ["Light", "Dark", "Satellite", "Street"],
                key="map_style"
            )
        
        with col2:
            colormap = st.selectbox(
                "🌈 Color Scheme",
                ["YlGn", "RdYlGn_r", "Greens", "viridis", "plasma"],
                key="colormap"
            )
        
        with col3:
            transparency = st.slider(
                "🔍 Overlay Transparency",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="transparency"
            )
        
        with col4:
            show_legend = st.checkbox(
                "📋 Show Legend",
                value=True,
                key="show_legend"
            )
        
        # Try to load and display the TIF file
        tif_path = 'data/agb_2024.tif'
        
        if os.path.exists(tif_path):
            try:
                with rasterio.open(tif_path) as src:
                    raster_data = src.read(1)
                    bounds = src.bounds
                    crs = src.crs
                    
                    # Display raster information
                    col1, col2, col3, col4 = st.columns(4)
                    valid_data = raster_data[raster_data > 0]
                    
                    with col1:
                        st.metric("🔹 Min AGB", f"{np.nanmin(valid_data):.1f} Mg/ha")
                    with col2:
                        st.metric("🔹 Max AGB", f"{np.nanmax(valid_data):.1f} Mg/ha")
                    with col3:
                        st.metric("🔹 Mean AGB", f"{np.nanmean(valid_data):.1f} Mg/ha")
                    with col4:
                        st.metric("🔹 Std AGB", f"{np.nanstd(valid_data):.1f} Mg/ha")
                    
                    # Create interactive map
                    center_lat = (bounds.bottom + bounds.top) / 2
                    center_lon = (bounds.left + bounds.right) / 2
                    
                    # Select map tiles based on style
                    tile_dict = {
                        "Light": "CartoDB positron",
                        "Dark": "CartoDB voyager",
                        "Satellite": "OpenStreetMap.Mapnik",
                        "Street": "OpenStreetMap.Mapnik"
                    }
                    
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=12,
                        tiles=tile_dict.get(map_style, "CartoDB positron")
                    )
                    
                    # Extract raster points and values for visualization
                    valid_mask = raster_data > 0
                    valid_indices = np.where(valid_mask)
                    
                    if len(valid_indices[0]) > 0:
                        # Convert pixel coordinates to lat/lon
                        lats = bounds.bottom + (valid_indices[0] / raster_data.shape[0]) * (bounds.top - bounds.bottom)
                        lons = bounds.left + (valid_indices[1] / raster_data.shape[1]) * (bounds.right - bounds.left)
                        values = raster_data[valid_mask]
                        
                        # Use Density HeatMap for better performance
                        # Convert numpy types to Python types for JSON serialization
                        heat_data = [
                            [float(lat), float(lon), float(val/100)] 
                            for lat, lon, val in zip(lats, lons, values)
                        ]
                        
                        plugins.HeatMap(
                            heat_data,
                            name='AGB Heatmap',
                            radius=20,
                            blur=15,
                            max_zoom=1,
                            min_opacity=0.4,
                            gradient={
                                0.0: '#ffffcc',    # Very light yellow - low biomass
                                0.2: '#d9f0a3',    # Light yellow
                                0.4: '#addd8e',    # Light green
                                0.6: '#78c679',    # Medium green
                                0.8: '#31a354',    # Dark green
                                1.0: '#006837'     # Very dark green - high biomass
                            }
                        ).add_to(m)
                    
                    # Add layer control
                    folium.LayerControl().add_to(m)
                    
                    # Add fullscreen button
                    plugins.Fullscreen().add_to(m)
                    
                    # Add minimap
                    minimap = plugins.MiniMap(toggle_display=True)
                    m.add_child(minimap)
                    
                    # Add measure control
                    plugins.MeasureControl().add_to(m)
                    
                    # Add legend as custom HTML
                    if show_legend:
                        legend_html = create_legend_html()
                        m.get_root().html.add_child(folium.Element(legend_html))
                        
                        # Add colorbar
                        min_val = np.nanmin(valid_data)
                        max_val = np.nanmax(valid_data)
                        colorbar_html = create_colorbar_html(min_val, max_val)
                        m.get_root().html.add_child(folium.Element(colorbar_html))
                    
                    # Display the map
                    st_folium(m, width=1400, height=700)
                    
            except Exception as e:
                st.error(f"❌ Error loading raster file: {str(e)}")
                st.info("Make sure the file is a valid GeoTIFF.")
        else:
            st.warning(f"⚠️ File not found: {tif_path}")
            st.info("Please ensure the agb_2024.tif file is in the data folder.")
    
    with tab2:
        st.subheader("📊 Map Statistics & Summary")
        
        try:
            with rasterio.open('data/agb_2024.tif') as src:
                raster_data = src.read(1)
                valid_data = raster_data[raster_data > 0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Summary Statistics")
                    stats_df = pd.DataFrame({
                        'Statistic': [
                            'Count',
                            'Mean',
                            'Std Dev',
                            'Min',
                            '25%',
                            'Median',
                            '75%',
                            'Max'
                        ],
                        'Value (Mg/ha)': [
                            len(valid_data),
                            np.mean(valid_data),
                            np.std(valid_data),
                            np.min(valid_data),
                            np.percentile(valid_data, 25),
                            np.median(valid_data),
                            np.percentile(valid_data, 75),
                            np.max(valid_data)
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎯 Distribution Analysis")
                    fig = px.histogram(
                        x=valid_data,
                        nbins=50,
                        title="AGB Distribution",
                        labels={'x': 'AGB (Mg/ha)', 'y': 'Frequency'},
                        color_discrete_sequence=['#2E8B57']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Category distribution
                st.markdown("### 📊 Biomass Category Distribution")
                
                categories = []
                for val in valid_data:
                    categories.append(get_agb_category(val))
                
                category_counts = pd.Series(categories).value_counts()
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**Category Counts:**")
                    for cat, count in category_counts.items():
                        percentage = (count / len(categories)) * 100
                        st.write(f"{cat}: {count} ({percentage:.1f}%)")
                
                with col2:
                    fig_pie = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title="Biomass Category Distribution",
                        color_discrete_sequence=['#440154', '#31688e', '#35b779', '#fde724']
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")
    
    with tab3:
        st.subheader("🎨 Heatmap View")
        
        try:
            with rasterio.open('data/agb_2024.tif') as src:
                raster_data = src.read(1)
                bounds = src.bounds
                
                # Create heatmap
                center_lat = (bounds.bottom + bounds.top) / 2
                center_lon = (bounds.left + bounds.right) / 2
                
                # Sample data for heatmap
                valid_mask = raster_data > 0
                valid_indices = np.where(valid_mask)
                
                if len(valid_indices[0]) > 0:
                    # Create sample points
                    sample_size = min(1000, len(valid_indices[0]))
                    sample_idx = np.random.choice(len(valid_indices[0]), sample_size, replace=False)
                    
                    lats = bounds.bottom + (valid_indices[0][sample_idx] / raster_data.shape[0]) * (bounds.top - bounds.bottom)
                    lons = bounds.left + (valid_indices[1][sample_idx] / raster_data.shape[1]) * (bounds.right - bounds.left)
                    values = raster_data[valid_mask][sample_idx]
                    
                    # Create heatmap map
                    m_heat = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=12,
                        tiles='CartoDB positron'
                    )
                    
                    # Add heatmap layer
                    heat_data = [[lat, lon, val] for lat, lon, val in zip(lats, lons, values)]
                    plugins.HeatMap(
                        heat_data,
                        name='AGB Heatmap',
                        radius=25,
                        blur=15,
                        max_zoom=1,
                        min_opacity=0.3,
                        max=np.max(values)
                    ).add_to(m_heat)
                    
                    # Add legend
                    legend_html = create_legend_html()
                    m_heat.get_root().html.add_child(folium.Element(legend_html))
                    
                    st_folium(m_heat, width=1400, height=700)
                else:
                    st.warning("No valid data for heatmap")
        
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
    
    with tab4:
        st.subheader("📈 Advanced Analysis")
        
        try:
            with rasterio.open('data/agb_2024.tif') as src:
                raster_data = src.read(1)
                valid_data = raster_data[raster_data > 0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Cumulative Distribution")
                    sorted_data = np.sort(valid_data)
                    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    
                    fig_cum = px.line(
                        x=sorted_data,
                        y=cumulative,
                        title="Cumulative Distribution Function",
                        labels={'x': 'AGB (Mg/ha)', 'y': 'Cumulative Probability'}
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)
                
                with col2:
                    st.markdown("### 📊 Box Plot Analysis")
                    fig_box = go.Figure(data=[
                        go.Box(y=valid_data, name='AGB Values', marker_color='#2E8B57')
                    ])
                    fig_box.update_layout(
                        title="AGB Values Box Plot",
                        yaxis_title="AGB (Mg/ha)",
                        height=500
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Percentile analysis
                st.markdown("### 📏 Percentile Analysis")
                percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
                percentile_values = [np.percentile(valid_data, p) for p in percentiles]
                
                percentile_df = pd.DataFrame({
                    'Percentile': percentiles,
                    'AGB Value (Mg/ha)': percentile_values
                })
                
                st.dataframe(percentile_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error creating analysis: {str(e)}")
    
    # Footer
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f0f0f0, #e8e8e8);
        padding: 20px;
        border-radius: 10px;
        margin-top: 30px;
        text-align: center;
        color: #666;
    ">
        <p style="margin: 0; font-size: 0.9rem;">
            🛰️ Biomass Estimation System | Data Source: GEDI & Sentinel-2 | Model: XGBoost Ensemble
        </p>
        <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #999;">
            Interactive Map © OpenStreetMap contributors | Visualization powered by Folium & Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    create_prediction_map_page()
