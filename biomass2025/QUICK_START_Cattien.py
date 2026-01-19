"""
🚀 QUICK START - Thu Thập Dữ LiỆu Cattien
===========================================

Script đơn giản để thu thập dữ liỆu từ Vưỡn Quốc Gia Cattien
Sử dụng asset có sẵn trên GEE: projects/swift-stack-464000-v4/assets/cattien

Author: Biomass Estimation
Date: 2026-01-19
"""

import pandas as pd
import numpy as np
import os

# ==================== CẤU HÌNH ====================

PROJECT_ID = 'swift-stack-464000-v4'
ASSET_PATH = 'projects/swift-stack-464000-v4/assets/cattien'
START_DATE = '2024-01-01'
END_DATE = '2024-12-31'
NUM_SAMPLES = 500  # Số điểm mẫu

OUTPUT_DIR = 'Processed/Cattien'
OUTPUT_FILE = 'cattien_biomass_data.csv'

# ==================== BƯỚC 1: CÀI ĐẶT ====================

def check_and_install_packages():
    """Kiểm tra và cài đặt các package cần thiết"""
    print("📦 Đang kiểm tra các package...")
    
    try:
        import ee
        print("✅ earthengine-api đã được cài đặt")
        return True
    except ImportError:
        print("⚠️  earthengine-api chưa được cài đặt")
        print("\n🔧 Chạy lệnh sau để cài đặt:")
        print("   pip install earthengine-api")
        return False

# ==================== BƯỚC 2: XÁC THỰC GEE ====================

def authenticate_gee():
    """Xác thực Google Earth Engine"""
    import ee
    
    try:
        ee.Initialize(project=PROJECT_ID)
        print("✅ GEE đã được xác thực")
        return True
    except Exception as e:
        print(f"⚠️  Cần xác thực GEE: {e}")
        print("\n🔧 Chạy lệnh sau:")
        print("   earthengine authenticate")
        print("\nSau đó chạy lại script này")
        return False

# ==================== BƯỚC 3: THU THẬP DỮ LIỆU ====================

def collect_data_simple():
    """Thu thập dữ liệu đơn giản nhất"""
    import ee
    
    print("\n" + "="*70)
    print("🌲 BẮT ĐẦU THU THẬP DỮ LIỆU CÁT TIÊN")
    print("="*70)
    
    # Tải ROI từ asset
    print(f"\n📍 Đang tải ROI từ: {ASSET_PATH}")
    try:
        roi = ee.FeatureCollection(ASSET_PATH).geometry()
        print("✅ ROI đã được tải thành công")
    except Exception as e:
        print(f"❌ Lỗi khi tải asset: {e}")
        return None
    
    # Thu thập Sentinel-2
    print(f"\n🌈 Đang thu thập Sentinel-2 data ({START_DATE} đến {END_DATE})...")
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(START_DATE, END_DATE) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    
    print(f"   Số ảnh tìm thấy: {s2.size().getInfo()}")
    
    if s2.size().getInfo() == 0:
        print("❌ Không tìm thấy ảnh Sentinel-2. Vui lòng thay đổi thời gian.")
        return None
    
    # Tính median composite
    s2_median = s2.median()
    
    # Chọn các bands quan trọng
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    s2_selected = s2_median.select(bands)
    
    # Tính NDVI
    print("\n📊 Đang tính các chỉ số thực vật...")
    ndvi = s2_median.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndmi = s2_median.normalizedDifference(['B8', 'B11']).rename('NDMI')
    ndwi = s2_median.normalizedDifference(['B3', 'B8']).rename('NDWI')
    
    # Kết hợp tất cả
    combined = s2_selected.addBands([ndvi, ndmi, ndwi])
    
    # Lấy mẫu ngẫu nhiên
    print(f"\n🎲 Đang lấy {NUM_SAMPLES} điểm mẫu ngẫu nhiên...")
    points = ee.FeatureCollection.randomPoints(roi, NUM_SAMPLES, seed=42)
    
    samples = combined.sampleRegions(
        collection=points,
        scale=10,
        geometries=True
    )
    
    # Chuyển sang DataFrame
    print("\n💾 Đang chuyển dữ liệu sang DataFrame...")
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
        
        # Đổi tên cột để dễ đọc
        rename_map = {
            'B2': 'Blue',
            'B3': 'Green', 
            'B4': 'Red',
            'B5': 'RedEdge1',
            'B6': 'RedEdge2',
            'B7': 'RedEdge3',
            'B8': 'NIR',
            'B8A': 'NIR_Narrow',
            'B11': 'SWIR1',
            'B12': 'SWIR2'
        }
        df = df.rename(columns=rename_map)
        
        print(f"✅ Đã thu thập {len(df)} điểm dữ liệu")
        return df
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý dữ liệu: {e}")
        return None

# ==================== BƯỚC 4: XỬ LÝ VÀ LƯU ====================

def process_and_save(df):
    """Xử lý và lưu dữ liệu"""
    print("\n" + "="*70)
    print("💾 XỬ LÝ VÀ LƯU DỮ LIỆU")
    print("="*70)
    
    # Xử lý missing values
    print(f"\n🔍 Missing values: {df.isnull().sum().sum()}")
    if df.isnull().sum().sum() > 0:
        print("   Đang điền missing values bằng mean...")
        df = df.fillna(df.mean())
        print(f"   ✅ Missing values sau xử lý: {df.isnull().sum().sum()}")
    
    # Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Lưu file
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dữ liệu đã được lưu tại: {output_path}")
    
    # Hiển thị thống kê
    print("\n" + "="*70)
    print("📊 THỐNG KÊ DỮ LIỆU")
    print("="*70)
    print(f"\nSố lượng mẫu: {len(df)}")
    print(f"Số lượng features: {len(df.columns)}")
    print(f"\nCác features:\n{list(df.columns)}")
    
    print("\n📈 Thống kê cơ bản:")
    print(df.describe())
    
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH!")
    print("="*70)
    print(f"\n📁 File dữ liệu: {output_path}")
    print("🚀 Bạn có thể sử dụng file này để huấn luyện mô hình biomass!")
    
    return df

# ==================== HÀM CHÍNH ====================

def main():
    """Hàm chính"""
    print("\n" + "="*70)
    print("🌲 BIOMASS ESTIMATION")
    print("Thu Thập Dữ LiỆu Vưỡn Quốc Gia Cattien")
    print("="*70)
    
    # Bước 1: Kiểm tra packages
    if not check_and_install_packages():
        return
    
    # Bước 2: Xác thực GEE
    if not authenticate_gee():
        return
    
    # Bước 3: Thu thập dữ liệu
    df = collect_data_simple()
    
    if df is not None:
        # Bước 4: Xử lý và lưu
        process_and_save(df)
    else:
        print("\n❌ Không thể thu thập dữ liệu. Vui lòng kiểm tra lại.")

# ==================== CHẠY SCRIPT ====================

if __name__ == '__main__':
    main()
