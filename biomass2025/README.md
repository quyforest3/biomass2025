# 🌱 Biomass 2025 Dashboard

Interactive dashboard for above-ground biomass (AGB) modeling using GEDI, Sentinel-1/2, and derived features. Includes training, diagnostics, feature analysis, and spatial views.

## Overview
- Predict AGB with ensemble models (RF, LightGBM, XGBoost, SVR)
- Visualize performance, feature importance, and diagnostics in Streamlit
- Spatial views (clusters, hotspots, interpolation) when coordinates are available

## Requirements
- Python 3.8+ (project venv: `.venv`)
- GDAL/PROJ binaries recommended for geospatial wheels (rasterio, geopandas, cartopy)

## Setup
```bash
git clone https://github.com/MichaelTheAnalyst/biomass2025.git
cd biomass2025
python -m venv .venv
.\.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Run the dashboard
```bash
.\.venv\Scripts\python.exe -m streamlit run biomass2025/dashboard_streamlit_app.py --server.port 8501 --server.headless true
```
Open http://localhost:8501.

## Model results
- Metrics depend on your data in `data/data.csv`; the app recomputes when you train inside the UI.
- To export the current metrics/plots, use the download buttons in the dashboard sections (Model Performance, Feature Importance, Diagnostics).
- If you want a static report in the repo, run the dashboard, capture the metrics table and figures, then commit them under `docs/` or `assets/` with a short summary (RMSE/R²/MAE per model).

## Data needed
- Place your main tabular data at `biomass2025/data/data.csv` (the app resolves this path automatically).
- For spatial plots, ensure columns like `Longitude_gedi`, `Latitude_gedi`, and target `AGB_2024` (or `AGB_2017`).

## Project structure (trimmed)
```
biomass2025/
├─ dashboard_streamlit_app.py   # Streamlit dashboard
├─ data/                        # data.csv lives here
├─ models/                      # saved models
├─ scripts/                     # utility scripts
├─ docs/                        # guides
├─ requirements.txt
└─ README.md
```

## Troubleshooting
- **Data file not found**: confirm `biomass2025/data/data.csv` exists and restart Streamlit (cache can keep old errors).
- **Geospatial wheels fail on Windows**: try `pip install --only-binary=:all: rasterio geopandas cartopy` or install GDAL/PROJ via conda/OSGeo4W.

## License
MIT License. See [LICENSE](LICENSE).

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

</div>
