"""
🌲 MASTER SCRIPT: Thu Thập Dữ Liệu Biomass từ Google Earth Engine
===================================================================

Script này thu thập dữ liệu từ:
- Sentinel-1 (SAR data)
- Sentinel-2 (Optical data + Vegetation Indices)
- DEM (Elevation, Slope, Aspect)
- Land Cover (ESA WorldCover)

Author: Biomass Estimation
Date: 2026-01-19
"""

import ee
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ==================== CẤU HÌNH ====================

# 1. THÔNG TIN TÀI KHOẢN GEE
PROJECT_ID = 'swift-stack-464000-v4'

# 2. KHU VỰC NGHIÊN CỨU (Region of Interest - ROI)
# OPTION 1: Sử dụng GEE Asset có sẵn
USE_GEE_ASSET = True
GEE_ASSET_PATH = 'projects/swift-stack-464000-v4/assets/cattien'

# OPTION 2: Hoặc sử dụng tọa độ thủ công (nếu USE_GEE_ASSET = False)
ROI = {
    'name': 'Cattien_National_Park',  # Tên khu vực
    'bounds': {
        'min_lon': 107.1,  # Kinh độ tây (West)
        'max_lon': 107.7,  # Kinh độ đông (East)
        'min_lat': 11.2,   # Vĩ độ nam (South)
        'max_lat': 11.8    # Vĩ độ bắc (North)
    }
}

# 3. THỜI GIAN PHÂN TÍCH
START_DATE = '2023-01-01'
END_DATE = '2023-12-31'

# 4. ĐƯỜNG DẪN LƯU KẾT QUẢ
OUTPUT_DIR = r'C:\Users\Dell 3530\OneDrive\Máy tính\xem biomass\biomass 2025\biomass2025\Processed'

# ==================== KHỞI TẠO GEE ====================

def initialize_gee():
    """Khởi tạo Google Earth Engine"""
    try:
        ee.Initialize(project=PROJECT_ID)
        print(f"✅ GEE initialized successfully with project: {PROJECT_ID}")
        return True
    except Exception as e:
        print(f"❌ Error initializing GEE: {e}")
        print("\n🔧 Hướng dẫn khắc phục:")
        print("1. Chạy: earthengine authenticate")
        print("2. Đăng nhập vào tài khoản Google có quyền truy cập GEE")
        print("3. Chạy lại script này")
        return False

# ==================== TẠO ROI GEOMETRY ====================

def create_roi_geometry(bounds=None, use_asset=False, asset_path=None):
    """
    Tạo geometry cho khu vực nghiên cứu
    
    Args:
        bounds: Dictionary chứa tọa độ
        use_asset: Nếu True, sử dụng GEE asset
        asset_path: Đường dẫn đến GEE asset
    """
    if use_asset and asset_path:
        print(f"📍 Đang tải ROI từ GEE asset: {asset_path}")
        try:
            roi = ee.FeatureCollection(asset_path).geometry()
            print("✅ ROI đã được tải từ asset")
            return roi
        except Exception as e:
            print(f"❌ Lỗi khi tải asset: {e}")
            print("⚠️  Chuyển sang sử dụng tọa độ thủ công...")
            use_asset = False
    
    if not use_asset and bounds:
        roi = ee.Geometry.Rectangle([
            bounds['min_lon'],
            bounds['min_lat'],
            bounds['max_lon'],
            bounds['max_lat']
        ])
        return roi
    
    return None

# ==================== THU THẬP SENTINEL-1 DATA ====================

def collect_sentinel1_data(roi, start_date, end_date):
    """
    Thu thập Sentinel-1 SAR data
    Returns: VV, VH polarization và VV/VH ratio
    """
    print("\n📡 Đang thu thập Sentinel-1 data...")
    
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select(['VV', 'VH'])
    
    # Tính median composite
    s1_composite = s1.median()
    
    # Tính VV/VH ratio
    s1_composite = s1_composite.addBands(
        s1_composite.select('VV').divide(s1_composite.select('VH')).rename('VV_VH_ratio')
    )
    
    print(f"✅ Sentinel-1 data collected: {s1.size().getInfo()} images")
    return s1_composite

# ==================== THU THẬP SENTINEL-2 DATA ====================

def collect_sentinel2_data(roi, start_date, end_date):
    """
    Thu thập Sentinel-2 optical data và tính các chỉ số thực vật
    """
    print("\n🌈 Đang thu thập Sentinel-2 data...")
    
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])
    
    # Tính median composite
    s2_composite = s2.median()
    
    # Tính các chỉ số thực vật
    s2_composite = add_vegetation_indices(s2_composite)
    
    print(f"✅ Sentinel-2 data collected: {s2.size().getInfo()} images")
    return s2_composite

def add_vegetation_indices(image):
    """Tính các chỉ số thực vật"""
    
    # NDVI - Normalized Difference Vegetation Index
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # NDMI - Normalized Difference Moisture Index
    ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
    
    # NDWI - Normalized Difference Water Index
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    
    # EVI - Enhanced Vegetation Index
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
            'BLUE': image.select('B2')
        }
    ).rename('EVI')
    
    # SAVI - Soil Adjusted Vegetation Index
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
        {
            'NIR': image.select('B8'),
            'RED': image.select('B4')
        }
    ).rename('SAVI')
    
    # NDCI - Normalized Difference Chlorophyll Index
    ndci = image.normalizedDifference(['B5', 'B4']).rename('NDCI')
    
    # ChlRe - Chlorophyll Red-edge
    chlre = image.expression(
        '(NIR / RedEdge) - 1',
        {
            'NIR': image.select('B8'),
            'RedEdge': image.select('B5')
        }
    ).rename('ChlRe')
    
    # MCARI - Modified Chlorophyll Absorption Ratio Index
    mcari = image.expression(
        '((B5 - B4) - 0.2 * (B5 - B3)) * (B5 / B4)',
        {
            'B3': image.select('B3'),
            'B4': image.select('B4'),
            'B5': image.select('B5')
        }
    ).rename('MCARI')
    
    # REPO - Red-Edge Position Index
    repo = image.expression(
        '700 + 40 * ((((B4 + B7) / 2) - B5) / (B6 - B5))',
        {
            'B4': image.select('B4'),
            'B5': image.select('B5'),
            'B6': image.select('B6'),
            'B7': image.select('B7')
        }
    ).rename('REPO')
    
    # NDRE - Normalized Difference Red-Edge
    ndre = image.normalizedDifference(['B8', 'B5']).rename('NDRE')
    
    return image.addBands([ndvi, ndmi, ndwi, evi, savi, ndci, chlre, mcari, repo, ndre])

# ==================== THU THẬP DEM DATA ====================

def collect_dem_data(roi):
    """Thu thập DEM data (Elevation, Slope, Aspect)"""
    print("\n⛰️  Đang thu thập DEM data...")
    
    # Sử dụng SRTM DEM
    dem = ee.Image('USGS/SRTMGL1_003')
    elevation = dem.select('elevation')
    
    # Tính slope và aspect
    terrain = ee.Terrain.products(elevation)
    slope = terrain.select('slope')
    aspect = terrain.select('aspect')
    
    dem_composite = elevation.addBands([slope, aspect])
    
    print("✅ DEM data collected")
    return dem_composite

# ==================== THU THẬP LAND COVER DATA ====================

def collect_landcover_data(roi):
    """Thu thập Land Cover data từ ESA WorldCover"""
    print("\n🗺️  Đang thu thập Land Cover data...")
    
    landcover = ee.ImageCollection('ESA/WorldCover/v200') \
        .first() \
        .select('Map')
    
    print("✅ Land Cover data collected")
    return landcover

# ==================== KẾT HỢP TẤT CẢ DỮ LIỆU ====================

def combine_all_data(s1, s2, dem, landcover):
    """Kết hợp tất cả các layers thành một image"""
    print("\n🔗 Đang kết hợp tất cả dữ liệu...")
    
    combined = s2.addBands(s1) \
                 .addBands(dem) \
                 .addBands(landcover)
    
    print("✅ Dữ liệu đã được kết hợp")
    return combined

# ==================== LẤY MẪU NGẪU NHIÊN ====================

def sample_random_points(image, roi, num_points=1000):
    """
    Lấy mẫu ngẫu nhiên từ image
    
    Args:
        image: Combined image
        roi: Region of interest
        num_points: Số lượng điểm mẫu
    
    Returns:
        pandas DataFrame với các features
    """
    print(f"\n📊 Đang lấy {num_points} điểm mẫu ngẫu nhiên...")
    
    # Tạo các điểm ngẫu nhiên
    points = ee.FeatureCollection.randomPoints(roi, num_points)
    
    # Lấy giá trị tại các điểm
    samples = image.sampleRegions(
        collection=points,
        scale=10,  # Resolution 10m
        geometries=True
    )
    
    # Chuyển sang pandas DataFrame
    try:
        sample_list = samples.getInfo()['features']
        
        data = []
        for feature in sample_list:
            props = feature['properties']
            coords = feature['geometry']['coordinates']
            props['Longitude'] = coords[0]
            props['Latitude'] = coords[1]
            data.append(props)
        
        df = pd.DataFrame(data)
        print(f"✅ Đã lấy {len(df)} điểm mẫu")
        return df
        
    except Exception as e:
        print(f"❌ Lỗi khi lấy mẫu: {e}")
        return None

# ==================== XỬ LÝ VÀ LƯU DỮ LIỆU ====================

def process_and_save_data(df, output_dir, roi_name):
    """Xử lý và lưu dữ liệu"""
    print("\n💾 Đang xử lý và lưu dữ liệu...")
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Xử lý missing values
    print(f"Missing values trước khi xử lý: {df.isnull().sum().sum()}")
    df = df.fillna(df.mean())
    print(f"Missing values sau khi xử lý: {df.isnull().sum().sum()}")
    
    # Lưu raw data
    raw_file = os.path.join(output_dir, f'{roi_name}_raw_data.csv')
    df.to_csv(raw_file, index=False)
    print(f"✅ Raw data saved: {raw_file}")
    
    # Lưu cleaned data (tương đương opt_means_cleaned.csv)
    cleaned_file = os.path.join(output_dir, f'{roi_name}_cleaned_data.csv')
    df.to_csv(cleaned_file, index=False)
    print(f"✅ Cleaned data saved: {cleaned_file}")
    
    return df

# ==================== HÀM CHÍNH ====================

def main():
    """Hàm chính để chạy toàn bộ quy trình"""
    
    print("=" * 70)
    print("🌲 Biomass Estimation - Data Collection")
    print("=" * 70)
    
    # 1. Khởi tạo GEE
    if not initialize_gee():
        return
    
    # 2. Tạo ROI geometry
    print(f"\n📍 Khu vực nghiên cứu: {ROI['name']}")
    
    if USE_GEE_ASSET:
        print(f"   Nguồn: GEE Asset - {GEE_ASSET_PATH}")
        roi_geometry = create_roi_geometry(use_asset=True, asset_path=GEE_ASSET_PATH)
    else:
        print(f"   Tọa độ: ({ROI['bounds']['min_lat']}, {ROI['bounds']['min_lon']}) "
              f"đến ({ROI['bounds']['max_lat']}, {ROI['bounds']['max_lon']})")
        roi_geometry = create_roi_geometry(bounds=ROI['bounds'])
    
    if roi_geometry is None:
        print("❌ Không thể tạo ROI geometry")
        return
    
    # 3. Thu thập dữ liệu
    print(f"\n📅 Thời gian: {START_DATE} đến {END_DATE}")
    
    s1_data = collect_sentinel1_data(roi_geometry, START_DATE, END_DATE)
    s2_data = collect_sentinel2_data(roi_geometry, START_DATE, END_DATE)
    dem_data = collect_dem_data(roi_geometry)
    landcover_data = collect_landcover_data(roi_geometry)
    
    # 4. Kết hợp dữ liệu
    combined_data = combine_all_data(s1_data, s2_data, dem_data, landcover_data)
    
    # 5. Lấy mẫu
    df = sample_random_points(combined_data, roi_geometry, num_points=1000)
    
    if df is not None:
        # 6. Xử lý và lưu
        df = process_and_save_data(df, OUTPUT_DIR, ROI['name'])
        
        # 7. Hiển thị thông tin
        print("\n" + "=" * 70)
        print("📊 THÔNG TIN DỮ LIỆU")
        print("=" * 70)
        print(f"Số lượng mẫu: {len(df)}")
        print(f"Số lượng features: {len(df.columns)}")
        print(f"\nCác features: {list(df.columns)}")
        print(f"\nThống kê cơ bản:\n{df.describe()}")
        
        print("\n" + "=" * 70)
        print("✅ HOÀN THÀNH! Dữ liệu đã sẵn sàng để huấn luyện mô hình.")
        print("=" * 70)
    else:
        print("\n❌ Không thể thu thập dữ liệu. Vui lòng kiểm tra lại cấu hình.")

# ==================== CHẠY SCRIPT ====================

if __name__ == '__main__':
    print("\n🔍 HƯỚNG DẪN SỬ DỤNG:")
    print("-" * 70)
    print("1. Cập nhật ROI (tọa độ khu vực nghiên cứu)")
    print("2. Cập nhật START_DATE và END_DATE")
    print("3. Cập nhật OUTPUT_DIR (thư mục lưu kết quả)")
    print("4. Chạy script: python MASTER_data_collection_GEE.py")
    print("-" * 70)
    print("\n⚠️  LƯU Ý:")
    print("- Cần có tài khoản Google Earth Engine")
    print("- Chạy 'earthengine authenticate' nếu chưa đăng nhập")
    print("- Quá trình có thể mất 5-15 phút tùy khu vực")
    print("-" * 70)
    
    response = input("\n▶️  Bạn đã sẵn sàng chạy? (y/n): ")
    if response.lower() == 'y':
        main()
    else:
        print("\n⏸️  Script đã dừng. Vui lòng cập nhật cấu hình và chạy lại.")
