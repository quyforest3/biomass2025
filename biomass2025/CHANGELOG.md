# 📝 Changelog

All notable changes to Biomass Estimation will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### 🎯 Planned Features
- [ ] API endpoints for model predictions
- [ ] Batch processing for large datasets
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure)
- [ ] Real-time GEDI data integration
- [ ] Mobile-responsive dashboard

---

## [1.0.0] - 2025-01-XX

### 🎉 Initial Release

#### ✨ Added
- **Interactive Streamlit Dashboard**
  - Real-time model training and evaluation
  - Comprehensive model performance comparison
  - Interactive visualizations with Plotly
  
- **Machine Learning Models**
  - Random Forest Regressor with hyperparameter tuning
  - LightGBM Regressor with gradient boosting
  - XGBoost Regressor with tree-based learning
  - Support Vector Regressor (SVR) with RBF kernel
  
- **Feature Engineering**
  - Automated vegetation indices calculation (NDVI, NDMI, NDWI, etc.)
  - Spectral band ratio extraction
  - Statistical feature aggregation
  - Polynomial feature generation
  
- **Feature Selection**
  - Variance Threshold filtering
  - F-test statistical selection
  - Mutual Information scoring
  - Recursive Feature Elimination (RFE)
  - L1-based Lasso selection
  
- **Model Diagnostics**
  - Learning curves (training/validation)
  - Residual analysis (normality, homoscedasticity)
  - Bias-variance tradeoff analysis
  - Cross-validation stability assessment
  
- **Spatial Analysis**
  - Geographic clustering (K-Means)
  - Spatial autocorrelation (Moran's I, Geary's C)
  - Hotspot detection (Local Outlier Factor)
  - Spatial interpolation (IDW, Nearest Neighbor)
  - Interactive Mapbox visualizations
  
- **Data Pipeline**
  - GEDI L4A data preprocessing
  - Sentinel-1 SAR data extraction
  - Sentinel-2 optical data extraction
  - DEM terrain analysis
  - Land cover classification integration
  
- **Documentation**
  - Comprehensive README with setup instructions
  - Contributing guidelines
  - MIT License
  - Code of conduct

#### 🔧 Technical Details
- Python 3.8+ compatibility
- Scikit-learn for ML models
- Streamlit for interactive dashboard
- Plotly for visualizations
- Pandas/NumPy for data processing
- RandomizedSearchCV for hyperparameter optimization

---

## [0.5.0] - 2024-12-XX (Pre-release)

### ✨ Added
- Core model training scripts (RF, LGBM, XGBoost, SVR)
- Basic data preprocessing pipeline
- Initial feature importance analysis
- Prototype visualizations

### 🔧 Changed
- Migrated from Jupyter notebooks to Python scripts
- Optimized data loading performance
- Improved memory efficiency for large datasets

### 🐛 Fixed
- GEDI quality flag filtering issues
- Sentinel data alignment problems
- Feature correlation calculation errors

---

## [0.3.0] - 2024-11-XX (Alpha)

### ✨ Added
- GEDI L4A data extraction
- Sentinel-1 preprocessing
- Sentinel-2 preprocessing
- Basic Random Forest model
- Initial data exploration notebooks

### 🔧 Changed
- Switched from GEE to local processing
- Updated data storage structure

---

## [0.1.0] - 2024-10-XX (Initial Development)

### ✨ Added
- Project initialization
- Literature review documentation
- Data acquisition strategy
- Initial requirements specification

---

## Version History Summary

| Version | Release Date | Highlights |
|---------|--------------|------------|
| 1.0.0 | 2025-01-XX | Full interactive dashboard, spatial analysis |
| 0.5.0 | 2024-12-XX | Core ML models, preprocessing pipeline |
| 0.3.0 | 2024-11-XX | Data extraction, initial RF model |
| 0.1.0 | 2024-10-XX | Project initialization |

---

## Semantic Versioning Guide

**Format**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (incompatible API changes)
- **MINOR**: New features (backward-compatible)
- **PATCH**: Bug fixes (backward-compatible)

### Change Categories

- 🎉 **Added**: New features
- 🔧 **Changed**: Changes in existing functionality
- 🗑️ **Deprecated**: Soon-to-be removed features
- 🚫 **Removed**: Removed features
- 🐛 **Fixed**: Bug fixes
- 🔒 **Security**: Security improvements

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this project.

---

## Links

- **Repository**: https://github.com/MichaelTheAnalyst/BioVision-Analytics-Hub
- **Issues**: https://github.com/MichaelTheAnalyst/BioVision-Analytics-Hub/issues
- **Releases**: https://github.com/MichaelTheAnalyst/BioVision-Analytics-Hub/releases

---

<div align="center">

**🌱 Thank you for using Biomass Estimation! 🌱**

</div>

