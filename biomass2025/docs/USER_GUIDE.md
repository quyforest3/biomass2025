# 📖 User Guide - Biomass Estimation

Complete guide to using Biomass Estimation for biomass prediction and analysis.

---

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Dashboard Overview](#dashboard-overview)
- [Model Training](#model-training)
- [Feature Analysis](#feature-analysis)
- [Model Diagnostics](#model-diagnostics)
- [Spatial Analysis](#spatial-analysis)
- [Exporting Results](#exporting-results)
- [Advanced Usage](#advanced-usage)
- [FAQ](#faq)

---

## 🚀 Getting Started

### Launching the Dashboard

```bash
# Windows
scripts\launch_dashboard.bat

# macOS/Linux
streamlit run src/dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### First-Time Setup

1. **Prepare Your Data**
   - Place your training data in `data/processed/`
   - Ensure data has features (B01-B12, NDVI, etc.) and target column (AGB_2017)

2. **Load Data**
   - The dashboard automatically loads `FEI data/opt_means_cleaned.csv`
   - Or specify custom path in the sidebar

3. **Explore**
   - Navigate through sections using the sidebar
   - Each section is independent and can be explored in any order

---

## 🎨 Dashboard Overview

### Main Sections

| Section | Purpose | Key Features |
|---------|---------|--------------|
| **📊 Model Performance** | Train & compare models | 4 ML models, metrics, predictions |
| **🎯 Feature Importance** | Understand feature impact | Traditional & permutation importance |
| **🔧 Feature Engineering** | Create & select features | Multiple selection techniques |
| **📈 Model Diagnostics** | Deep model analysis | Learning curves, residuals |
| **🗺️ Spatial Analysis** | Geographic patterns | Clustering, autocorrelation, hotspots |

---

## 🤖 Model Training

### Step 1: Navigate to Model Performance

Click **"📊 Model Performance"** in the sidebar.

### Step 2: Configure Training

Adjust hyperparameters if needed:

- **Random Forest**: n_estimators, max_depth
- **LightGBM**: learning_rate, num_leaves
- **XGBoost**: max_depth, gamma
- **SVR**: C, epsilon, kernel

### Step 3: Train Models

Click **"🚀 Train All Models"** button.

**What happens:**
1. Data is split into train/test (80/20)
2. Features are scaled using StandardScaler
3. Each model trains with RandomizedSearchCV (20 iterations)
4. Best models are selected based on cross-validation
5. Models are evaluated on test set

**Expected time:** 2-5 minutes depending on dataset size

### Step 4: Review Results

#### Performance Metrics

- **RMSE** (Root Mean Square Error): Lower is better
- **R² Score**: 0-1, higher is better (1 = perfect prediction)
- **MAE** (Mean Absolute Error): Lower is better
- **Training Time**: Model efficiency

#### Visualizations

1. **Metrics Comparison Bar Chart**
   - Compare RMSE, R², MAE across models
   - Hover for exact values

2. **Radar Chart**
   - Multi-dimensional performance view
   - See trade-offs between metrics

3. **Prediction Scatter Plot**
   - Actual vs Predicted values
   - Diagonal line = perfect prediction
   - Points above line = overestimation
   - Points below line = underestimation

### Step 5: Save Models

Click **"💾 Save Models & Results"** to export:
- Trained model files (.pkl)
- Performance metrics (.csv)
- Feature importance rankings

---

## 🎯 Feature Analysis

### Understanding Feature Importance

Navigate to **"🎯 Feature Importance"** section.

#### Traditional Importance (Tree-based)

**What it shows:** How often a feature is used for splitting in tree-based models.

**Interpretation:**
- Higher values = more important
- Based on model structure
- Fast to compute

#### Permutation Importance

**What it shows:** Impact on model performance when feature values are randomly shuffled.

**Interpretation:**
- Higher values = feature provides more predictive power
- More robust than traditional importance
- Slower to compute but more reliable

### Top Features

Common important features for biomass prediction:

1. **NDVI** (Normalized Difference Vegetation Index): Vegetation health
2. **NIR bands** (B08, B8A): Near-infrared reflection
3. **SWIR bands** (B11, B12): Shortwave infrared
4. **NDMI** (Moisture index): Water content
5. **ChlRe** (Chlorophyll red-edge): Vegetation structure

### Correlation Analysis

**Heatmap Interpretation:**

- **Red** (positive correlation): Features move together
- **Blue** (negative correlation): Features move oppositely
- **White** (no correlation): Independent features

**Use case:**
- Identify redundant features (high correlation >0.9)
- Find feature groups (e.g., all red-edge bands correlate)
- Detect multicollinearity issues

---

## 🔧 Feature Engineering

### Automated Feature Creation

Navigate to **"🔧 Feature Engineering"** section.

#### Available Feature Types

1. **Vegetation Indices Ratios**
   - NDVI/NDMI, NDVI/NDWI combinations
   - Enhanced vegetation signals

2. **Spectral Band Ratios**
   - NIR/Red, SWIR1/SWIR2, etc.
   - Normalized spectral responses

3. **Statistical Features**
   - Mean, std, max, min aggregations
   - Capture data distribution

4. **Polynomial Features**
   - Interaction terms (Feature1 × Feature2)
   - Quadratic terms (Feature²)
   - Capture non-linear relationships

### Feature Selection Techniques

#### 1. Variance Threshold

**How it works:** Removes features with low variance (nearly constant values)

**When to use:** Remove uninformative features quickly

**Parameters:** Threshold (e.g., 0.01)

#### 2. F-test (ANOVA)

**How it works:** Statistical test for feature-target relationship

**When to use:** Select features with strong linear relationships

**Parameters:** K (number of top features to keep)

#### 3. Mutual Information

**How it works:** Measures information gain from a feature

**When to use:** Capture non-linear relationships

**Parameters:** K (number of features)

#### 4. Recursive Feature Elimination (RFE)

**How it works:** Iteratively removes least important features

**When to use:** Find optimal feature subset with Random Forest

**Parameters:** n_features_to_select

#### 5. L1-based (Lasso)

**How it works:** Penalizes model complexity, forces coefficients to zero

**When to use:** Automatic feature selection with regularization

**Parameters:** Alpha (regularization strength)

### PCA (Principal Component Analysis)

**What it does:** Reduces dimensionality while preserving variance

**When to use:**
- Dataset has many correlated features
- Want to visualize high-dimensional data
- Need to reduce computation time

**How to choose components:**
- 0.95 = Retain 95% of variance
- 0.99 = Retain 99% of variance

---

## 📈 Model Diagnostics

Navigate to **"📈 Model Diagnostics"** section.

### Learning Curves

**What they show:** Model performance vs training size

#### Interpreting Learning Curves

**Good Model:**
- Training and validation scores converge
- Both scores are high
- Small gap between curves

**Overfitting:**
- Large gap between training (high) and validation (low)
- **Solution**: Regularization, more data, simpler model

**Underfitting:**
- Both scores are low
- Curves plateau early
- **Solution**: More features, complex model, less regularization

**High Variance:**
- Validation score fluctuates
- **Solution**: More training data, cross-validation

### Residual Analysis

**What it shows:** Prediction errors (actual - predicted)

#### Key Tests

1. **Normality Test (Shapiro-Wilk)**
   - **H0**: Residuals are normally distributed
   - **p > 0.05**: Good! Residuals are normal
   - **p < 0.05**: Non-normal residuals (may indicate model issues)

2. **Homoscedasticity (Constant Variance)**
   - **Look for**: Random scatter around zero
   - **Red flag**: Fan shape (variance increases with prediction)

### Bias-Variance Tradeoff

**What it shows:** Model stability across different data splits

**Metrics:**
- **Mean Score**: Average cross-validation performance
- **Std Score**: Consistency across folds
- **Lower std**: More stable model

**Ideal:**
- High mean score
- Low standard deviation

---

## 🗺️ Spatial Analysis

Navigate to **"🗺️ Spatial Analysis"** section.

### Geographic Clustering

**What it does:** Groups nearby points with similar biomass values

**Methods:**
- **K-Means**: Finds K circular clusters
- **Parameters**: n_clusters (3-10 typical)

**Use case:**
- Identify forest zones
- Management planning
- Stratified sampling

### Spatial Autocorrelation

#### Moran's I

**Range:** -1 to +1

- **> 0**: Positive autocorrelation (similar values cluster)
- **= 0**: Random spatial pattern
- **< 0**: Negative autocorrelation (dissimilar values cluster)

**Interpretation:**
- **I > 0.5**: Strong clustering
- **I = 0.3-0.5**: Moderate clustering
- **I < 0.3**: Weak/no clustering

#### Geary's C

**Range:** 0 to 2

- **< 1**: Positive autocorrelation
- **= 1**: Random
- **> 1**: Negative autocorrelation

**Comparison with Moran's I:**
- Geary's C is more sensitive to local differences
- Moran's I is more sensitive to global patterns

### Hotspot Analysis

**What it does:** Identifies anomalous biomass concentrations

**Method:** Local Outlier Factor (LOF)

**Interpretation:**
- **Hotspots**: Unusually high biomass areas
- **Coldspots**: Unusually low biomass areas
- **Use case**: Prioritize conservation, detect degradation

### Spatial Interpolation

**What it does:** Estimates biomass at unsampled locations

**Methods:**

1. **IDW (Inverse Distance Weighting)**
   - Closer points have more influence
   - Power parameter controls decay

2. **Nearest Neighbor**
   - Uses closest observation
   - Simple but effective

**Use case:**
- Create continuous biomass maps
- Fill gaps in satellite coverage

---

## 💾 Exporting Results

### From Dashboard

**Model Performance Section:**
- Click **"💾 Save Models & Results"**
- Exports: Models, metrics, predictions

**Outputs saved to:**
- `models/saved_models/`: Trained models
- `outputs/results/`: Performance metrics
- `outputs/figures/`: Visualizations (if enabled)

### Manual Export

```python
import joblib

# Load saved model
model = joblib.load('models/saved_models/AGB_RandomForest_Model.pkl')

# Make predictions
predictions = model.predict(X_new)
```

---

## 🔬 Advanced Usage

### Custom Data

```python
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Ensure correct column names
# Features: B01-B12, NDVI, NDMI, etc.
# Target: AGB_2017

# Run through dashboard or scripts
```

### Batch Processing

```bash
# Train multiple models
python src/models/random_forest/train_rf.py
python src/models/lightgbm/train_lgbm.py
python src/models/xgboost/train_xgboost.py
```

### Hyperparameter Tuning

Edit hyperparameter ranges in model scripts:

```python
param_distributions = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
}
```

---

## ❓ FAQ

### Q: Which model should I use?

**A:** It depends on your priorities:
- **Speed**: LightGBM (fastest training)
- **Accuracy**: Usually LightGBM or XGBoost
- **Interpretability**: Random Forest (easier to explain)
- **Small datasets**: SVR (better generalization)

### Q: How much data do I need?

**A:** Minimum recommendations:
- **100 samples**: Basic models
- **500 samples**: Reliable models
- **1000+ samples**: Advanced models, deep learning

### Q: My model has high RMSE. Why?

**A:** Possible causes:
1. **Insufficient features**: Add more spectral indices
2. **Outliers**: Check data quality
3. **Model too simple**: Try ensemble methods
4. **Data noise**: Apply filtering/smoothing
5. **Test on different area**: Model may not generalize

### Q: Can I use this for other vegetation parameters?

**A:** Yes! Replace AGB_2017 with:
- Canopy height
- Leaf Area Index (LAI)
- Biomass carbon
- Forest cover percentage

### Q: The dashboard is slow. How to speed up?

**A:**
1. Reduce dataset size (sample data)
2. Reduce `n_iter` in RandomizedSearchCV
3. Use fewer features
4. Disable heavy visualizations
5. Close other browser tabs

### Q: How to interpret R² = 0.85?

**A:** R² = 0.85 means:
- Model explains 85% of variance in biomass
- 15% is unexplained (noise, missing factors)
- Generally considered good performance

---

## 📚 Additional Resources

- **Dashboard Tutorial**: [Video walkthrough] (Coming soon)
- **API Reference**: [docs/API.md](API.md)
- **Architecture**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- **GitHub Issues**: Report bugs or ask questions

---

## 📧 Support

Need help? Contact:

- **Email**: support@biomass-estimation.com
- **Authors**: Nguyen Van Quy and Nguyen Hong Hai
- **Issues**: [GitHub Issues](https://github.com/MichaelTheAnalyst/biomass2025/issues)

---

<div align="center">

**Happy Analyzing! 🌱**

[Back to Main README](../README.md)

</div>

