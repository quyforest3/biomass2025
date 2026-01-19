# Hướng Dẫn Sửa Lỗi Phân Tích Không Gian / Spatial Analysis Fix Guide

## 🇻🇳 Tiếng Việt

### Vấn Đề Ban Đầu
Dashboard Streamlit hiển thị lỗi:
```
Error loading spatial data: [Errno 2] No such file or directory: 'merged_gedi_sentinel2_data_with_indices.csv'
```

### Nguyên Nhân
1. File `FEI data/opt_means_cleaned.csv` **không có tọa độ địa lý** (không có cột Latitude và Longitude)
2. Tính năng Phân Tích Không Gian cần file CSV có các cột:
   - `Longitude_gedi` - Kinh độ
   - `Latitude_gedi` - Vĩ độ
   - `AGB_L4A` - Giá trị sinh khối trên mặt đất

### Giải Pháp Đã Triển Khai

#### 1. Cải Thiện Thông Báo Lỗi
- Dashboard giờ hiển thị hướng dẫn chi tiết khi thiếu file
- Giải thích tại sao file hiện tại không dùng được
- Hướng dẫn cách tạo file mới

#### 2. Script Tạo Dữ Liệu Demo
**File**: `scripts/create_demo_spatial_data.py`

Tạo 200 điểm dữ liệu giả lập để test:
```bash
python scripts/create_demo_spatial_data.py
```

**Ưu điểm:**
- Không cần tài khoản Google Earth Engine
- Chạy nhanh (vài giây)
- Có đầy đủ cột cần thiết
- Tạo pattern không gian thực tế

**Lưu ý:** Đây là dữ liệu giả lập, chỉ dùng để test!

#### 3. Script Google Earth Engine (GEE)
**File**: `scripts/create_spatial_data_gee.py`

Tạo dữ liệu thật từ vệ tinh GEDI và Sentinel-2:

**Yêu cầu:**
```bash
# Cài đặt
pip install earthengine-api

# Xác thực (mở trình duyệt)
earthengine authenticate
```

**Cách dùng:**
1. Mở file `scripts/create_spatial_data_gee.py`
2. Chỉnh tọa độ vùng nghiên cứu (ROI_COORDINATES)
3. Chỉnh thời gian (START_DATE, END_DATE)
4. Chạy: `python scripts/create_spatial_data_gee.py`

**Tính năng:**
- Lấy dữ liệu GEDI L4A (sinh khối)
- Lấy ảnh Sentinel-2 (đã lọc mây)
- Tính các chỉ số thực vật (NDVI, NDMI, etc.)
- Ghép dữ liệu theo tọa độ
- Lọc chất lượng (MU < 0.5)
- Xuất ra CSV

#### 4. Tài Liệu Chi Tiết
- `docs/SPATIAL_DATA_GUIDE.md` - Hướng dẫn đầy đủ (tiếng Anh)
- `docs/SPATIAL_FIX_SUMMARY.md` - Tóm tắt kỹ thuật (tiếng Anh)
- `README.md` - Đã thêm phần Spatial Data Setup

### Cách Sử Dụng

#### Option 1: Test Nhanh (Khuyên dùng để test)
```bash
# 1. Tạo dữ liệu demo
python scripts/create_demo_spatial_data.py

# 2. Chạy dashboard
streamlit run dashboard_streamlit_app.py

# 3. Vào phần "🗺️ Spatial Analysis"
```

#### Option 2: Dữ Liệu Thật (Cần tài khoản GEE)
```bash
# 1. Cài đặt và xác thực GEE
pip install earthengine-api
earthengine authenticate

# 2. Chỉnh ROI trong scripts/create_spatial_data_gee.py

# 3. Chạy script
python scripts/create_spatial_data_gee.py

# 4. Chạy dashboard
streamlit run dashboard_streamlit_app.py
```

### Tại Sao File Hiện Tại Không Dùng Được?

File `FEI data/opt_means_cleaned.csv` có:
- ✅ Các dải quang phổ (B01-B12)
- ✅ Chỉ số thực vật (NDVI, NDMI, ...)
- ✅ Giá trị sinh khối (AGB_2017)
- ❌ **Không có Vĩ độ (Latitude)**
- ❌ **Không có Kinh độ (Longitude)**

Để làm phân tích không gian (clustering, autocorrelation, hotspot), **bắt buộc phải có tọa độ**.

### Các Tính Năng Phân Tích Không Gian

Khi có file dữ liệu đúng, dashboard sẽ có:

1. **Geographic Clustering** - Phân cụm địa lý
   - Tìm vùng có sinh khối tương tự nhau
   - Dùng thuật toán K-Means

2. **Spatial Autocorrelation** - Tự tương quan không gian
   - Tính Moran's I và Geary's C
   - Kiểm tra xem sinh khối có phân bố theo cụm không

3. **Hotspot Analysis** - Phân tích điểm nóng
   - Tìm vùng có sinh khối bất thường cao/thấp
   - Dùng Local Outlier Factor (LOF)

4. **Spatial Interpolation** - Nội suy không gian
   - Tạo bản đồ liên tục từ điểm rời rạc
   - Phương pháp IDW và Nearest Neighbor

### Kiểm Tra Dữ Liệu

Sau khi tạo file, kiểm tra xem đúng chưa:

```python
import pandas as pd

df = pd.read_csv('merged_gedi_sentinel2_data_with_indices.csv')

# Kiểm tra các cột cần thiết
print("Các cột:", df.columns.tolist())
print("Số điểm:", len(df))
print("Phạm vi Latitude:", df['Latitude_gedi'].min(), "-", df['Latitude_gedi'].max())
print("Phạm vi Longitude:", df['Longitude_gedi'].min(), "-", df['Longitude_gedi'].max())
print("Phạm vi AGB:", df['AGB_L4A'].min(), "-", df['AGB_L4A'].max())
```

### Khắc Phục Sự Cố

**Lỗi: "No GEDI data found"**
- GEDI không có dữ liệu toàn cầu
- Kiểm tra https://gedi.umd.edu/ xem vùng của bạn có dữ liệu không
- Thử đổi thời gian (GEDI bắt đầu từ 4/2019)

**Lỗi: "Computation timeout"**
- Giảm kích thước vùng nghiên cứu
- Rút ngắn khoảng thời gian
- Chia nhỏ vùng ra xử lý từng phần

**Lỗi xác thực GEE**
- Chạy lại: `earthengine authenticate`
- Mở trình duyệt và đăng nhập
- Sao chép mã xác thực vào terminal

### Liên Hệ & Hỗ Trợ

Nếu gặp vấn đề:
1. Xem file `docs/SPATIAL_DATA_GUIDE.md` (tiếng Anh)
2. Kiểm tra tọa độ ROI có đúng format không
3. Đảm bảo tài khoản GEE đã được duyệt
4. Mở issue trên GitHub

---

## 🇬🇧 English

### Quick Start

**For Testing:**
```bash
python scripts/create_demo_spatial_data.py
streamlit run dashboard_streamlit_app.py
```

**For Real Analysis:**
```bash
pip install earthengine-api
earthengine authenticate
# Edit ROI in scripts/create_spatial_data_gee.py
python scripts/create_spatial_data_gee.py
streamlit run dashboard_streamlit_app.py
```

### What Was Fixed

1. **Enhanced Error Handling** - Clear instructions when data is missing
2. **Demo Data Generator** - Quick testing without GEE (200 synthetic points)
3. **GEE Script** - Extract real data from GEDI L4A and Sentinel-2
4. **Documentation** - Comprehensive guides in English
5. **README Updates** - Spatial data setup section added

### Why Existing Data Can't Be Used

`FEI data/opt_means_cleaned.csv` has:
- ✅ Spectral bands (B01-B12)
- ✅ Vegetation indices (NDVI, NDMI, etc.)
- ✅ Biomass values (AGB_2017)
- ❌ **No Latitude coordinates**
- ❌ **No Longitude coordinates**

Spatial analysis requires geographic coordinates for clustering, autocorrelation, hotspot detection, and interpolation.

### Documentation

- `docs/SPATIAL_DATA_GUIDE.md` - Comprehensive setup guide
- `docs/SPATIAL_FIX_SUMMARY.md` - Technical implementation details
- `README.md` - Quick start section

### Files Created/Modified

**Created:**
- `scripts/create_spatial_data_gee.py` - GEE data extraction
- `scripts/create_demo_spatial_data.py` - Demo data generator
- `docs/SPATIAL_DATA_GUIDE.md` - User guide
- `docs/SPATIAL_FIX_SUMMARY.md` - Technical summary
- `docs/VIETNAMESE_GUIDE.md` - This bilingual guide

**Modified:**
- `dashboard_streamlit_app.py` - Enhanced error handling
- `README.md` - Added spatial data section

### Testing

All tests pass:
- ✅ Spatial data loading
- ✅ Dashboard error handling
- ✅ Data quality validation
- ✅ File structure verification
- ✅ Security scan (0 vulnerabilities)

### Support

For help:
1. Check `docs/SPATIAL_DATA_GUIDE.md`
2. Verify ROI coordinates format
3. Ensure GEE account is approved
4. Open GitHub issue if needed
