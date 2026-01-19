# 🗂️ GitHub Repository Reorganization Plan

## 📁 Professional Folder Structure

```
BioVision-Analytics-Hub/
├── 📂 src/                          # Source code
│   ├── data_preprocessing/          # Data collection & preprocessing scripts
│   ├── models/                      # Model training & evaluation
│   ├── visualization/               # Visualization scripts
│   ├── dashboard/                   # Streamlit dashboard components
│   └── utils/                       # Utility functions
│
├── 📂 data/                         # Data directory (add to .gitignore)
│   ├── raw/                         # Raw unprocessed data
│   ├── interim/                     # Intermediate processed data
│   └── processed/                   # Final processed data ready for modeling
│
├── 📂 models/                       # Trained models (add to .gitignore)
│   ├── saved_models/                # All .pkl and .h5 model files
│   └── scalers/                     # Scaler objects
│
├── 📂 outputs/                      # Generated outputs (add to .gitignore)
│   ├── figures/                     # Generated plots and visualizations
│   ├── results/                     # Model results and metrics
│   └── reports/                     # Analysis reports
│
├── 📂 notebooks/                    # Jupyter notebooks
│
├── 📂 docs/                         # Documentation
│   ├── API.md                       # API documentation
│   ├── INSTALLATION.md              # Installation guide
│   ├── USER_GUIDE.md                # User guide
│   └── ARCHITECTURE.md              # System architecture
│
├── 📂 config/                       # Configuration files
│
├── 📂 scripts/                      # Utility scripts (launchers, setup)
│
├── 📂 tests/                        # Unit tests
│
├── 📂 assets/                       # Images, logos for README
│
├── 📄 README.md                     # Main README
├── 📄 requirements.txt              # Python dependencies
├── 📄 .gitignore                    # Git ignore rules
├── 📄 LICENSE                       # License file
├── 📄 CONTRIBUTING.md               # Contribution guidelines
└── 📄 CHANGELOG.md                  # Version history
```

## 🔄 File Movement Plan

### **src/data_preprocessing/**
- p1.GEDI-preprocess.py
- p1.GEDI-preprocess-ROI.py
- p2.Sen1-preprocessed.py
- p2.Sen2-preprocessed.py
- p2. sentinel1_data_extraction_and_cleaning.py
- p2. sentinel2_data_extraction_and_cleaning.py
- p3. dem_data_extraction_with_terrain_analysis.py
- p4. merge_gedi_sentinel2_nearest_neighbor.py
- p5. merge_gedi_sentinel_datasets.py
- p6. merge_gedi_sentinel_dem_datasets.py
- p7. WorldCover_data_extraction_with_terrain_analysis.py
- p8. final merge.py
- S1-S6 scripts
- accessGEDI.py

### **src/models/**
- L1-L14 scripts (LGBM and RF training scripts)
- M1-M24 scripts (All model training scripts)

### **src/visualization/**
- S1. GEDI_AGBD_ROI_Visualization.py
- M15. almost all VIS.py
- M24. TREE_based VIS.py
- visualization simple.py
- visualization+.py
- Spider chart.py
- Histogram of AGB Values.py

### **src/dashboard/**
- dashboard_streamlit_app.py (renamed to app.py)
- dashboard_core.py
- dashboard_feature_analysis.py
- dashboard_feature_engineering.py
- dashboard_model_diagnostics.py

### **notebooks/**
- accessGEDI.ipynb
- collectData.ipynb

### **scripts/**
- 🚀_ONE_CLICK_LAUNCH.bat
- launch_dashboard.py
- launch_dashboard.bat
- auto_launch.py
- run_dashboard.bat
- run_dashboard_auto.ps1

### **data/raw/**
- All .csv files (GEDI, Sentinel data)
- All .shp, .shx, .dbf, .prj, .cpg files (shapefiles)
- .geojson files
- .kml files
- .zip files

### **models/saved_models/**
- All .pkl model files
- All .h5 model files (DNN, CNN)

### **outputs/figures/**
- All .png files
- All .html interactive plots

### **outputs/results/**
- All results CSV files (SHAP, feature importance, predictions, etc.)

### **config/**
- Create config.yaml for hyperparameters

## 🚫 Files to Remove/Clean
- code1.py - code11.py (temporary experimental files)
- untitled0.py, untitled1.py, untitled2.py
- collectdata2.py, collectdata3.py (duplicates)
- code6_files/ (HTML resources)
- chapter-10.pdf (documentation, move to docs or remove)

## 📝 New Files to Create

### **README.md** - Professional project overview
### **.gitignore** - Comprehensive ignore rules
### **LICENSE** - MIT or appropriate license
### **CONTRIBUTING.md** - Contribution guidelines
### **docs/INSTALLATION.md** - Setup instructions
### **docs/USER_GUIDE.md** - How to use the dashboard
### **docs/ARCHITECTURE.md** - System design
### **CHANGELOG.md** - Version history
### **.github/workflows/ci.yml** - CI/CD pipeline (optional)

## 🎯 Priority Actions

1. ✅ Create folder structure
2. 📝 Create .gitignore
3. 📝 Create comprehensive README.md
4. 🔄 Move files systematically
5. 📝 Create documentation
6. 🧪 Add __init__.py to make packages
7. 🚀 Test dashboard still works
8. 📋 Create LICENSE
9. 📋 Create CONTRIBUTING.md
10. 🎨 Add screenshots to assets/

## 🌟 GitHub Repository Best Practices

- **Clear README** with badges, demo GIF, installation instructions
- **Comprehensive .gitignore** (don't commit large data/models)
- **Requirements.txt** with pinned versions
- **LICENSE** file (MIT recommended)
- **Contributing guidelines**
- **Code of Conduct** (optional but professional)
- **GitHub Actions** for CI/CD (optional)
- **Wiki** for detailed documentation
- **Releases** with semantic versioning
- **Topics/Tags** for discoverability


