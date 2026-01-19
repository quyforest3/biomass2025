# 🌲 AGB Model Dashboard Suite

A comprehensive machine learning dashboard suite for Above Ground Biomass (AGB) prediction analysis, featuring multiple specialized dashboards for model performance, feature analysis, diagnostics, engineering, and business intelligence.

## 🚀 Quick Start

### Option 1: Use the Master Launcher
```bash
python launch_dashboard.py
```

### Option 2: Run Individual Dashboards
```bash
# Core Performance Dashboard
python dashboard_core.py

# Feature Analysis Dashboard
python dashboard_feature_analysis.py

# Model Diagnostics Dashboard
python dashboard_model_diagnostics.py

# Feature Engineering Dashboard
python dashboard_feature_engineering.py

# Business Intelligence Dashboard
python dashboard_business_intelligence.py

# Interactive Web Dashboard
streamlit run dashboard_streamlit_app.py
```

## 📋 Requirements

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- `streamlit>=1.28.0`
- `pandas>=1.5.0`
- `numpy>=1.21.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.11.0`
- `scikit-learn>=1.1.0`
- `plotly>=5.10.0`
- `xgboost>=1.6.0`
- `lightgbm>=3.3.0`
- `joblib>=1.2.0`

## 📊 Dashboard Components

### 1. 📈 Core Model Performance Dashboard
**File:** `dashboard_core.py`

**Features:**
- Trains 4 ML models (Random Forest, LightGBM, XGBoost, SVR)
- Hyperparameter optimization with RandomizedSearchCV
- Performance comparison (RMSE, R², MAE, Training Time)
- Automated model saving
- Performance ranking and visualization

**Key Outputs:**
- Model performance summary table
- Comparative bar charts
- Radar chart for best model
- Saved model files (.pkl)

### 2. 🔍 Feature Analysis Dashboard
**File:** `dashboard_feature_analysis.py`

**Features:**
- Feature importance analysis across all models
- Permutation importance (model-agnostic)
- Correlation analysis with target variable
- Mutual information scoring
- Feature importance heatmaps

**Key Insights:**
- **Top Consensus Features:** ChlRe, MCARI (agreed by multiple models)
- **High Correlation Features:** 15 features with |correlation| > 0.3
- **Model Agreement Analysis:** Features selected by multiple models

### 3. 🔬 Model Diagnostics Dashboard
**File:** `dashboard_model_diagnostics.py`

**Features:**
- Learning curves analysis
- Residual analysis and statistical tests
- Bias-variance tradeoff analysis
- Overfitting detection
- Model stability assessment

**Critical Findings:**
- **Overfitting Risk:** High in RF, LightGBM, XGBoost
- **Most Stable Model:** XGBoost (lowest CV variance)
- **Residual Issues:** Homoscedasticity violations in all models

### 4. ⚙️ Feature Engineering Dashboard
**File:** `dashboard_feature_engineering.py`

**Features:**
- Creates 15+ engineered features (75% increase)
- Multiple feature selection methods (F-test, Mutual Info, RFE, Lasso)
- PCA dimensionality reduction
- Feature set optimization
- Performance comparison across feature sets

**Optimization Results:**
- **Best Performance:** Random Forest with Lasso features (RMSE: 1.9293, R²: 0.8226)
- **PCA Reduction:** 85% dimensionality reduction while preserving 95% variance
- **Consensus Features:** 19 features selected by multiple methods

### 5. 💼 Business Intelligence Dashboard
**File:** `dashboard_business_intelligence.py`

**Features:**
- ROI analysis across business scenarios
- Cost-benefit analysis
- Deployment readiness assessment
- Prediction confidence intervals
- Risk assessment and mitigation

**Business Insights:**
- **ROI Analysis:** Up to 3724% ROI for large-scale deployment
- **Recommended Model:** Production RF for all scenarios
- **Cost Savings:** $19.4M net benefit for large-scale implementation
- **Deployment Ready:** 2 out of 4 models ready for production

### 6. 🌐 Interactive Web Dashboard
**File:** `dashboard_streamlit_app.py`

**Features:**
- Interactive Streamlit web interface
- Real-time model training and comparison
- Interactive plots with Plotly
- Feature importance exploration
- Residual analysis tools
- Data overview and statistics

**Access:** Run `streamlit run dashboard_streamlit_app.py` and open http://localhost:8501

## 📁 Generated Files

Each dashboard creates various output files:

### Model Files
- `AGB_Random_Forest_Model.pkl`
- `AGB_LightGBM_Model.pkl`
- `AGB_XGBoost_Model.pkl`
- `AGB_SVR_Model.pkl`
- `AGB_Scaler.pkl`

### Analysis Results
- `model_performance_summary_[timestamp].csv`
- `feature_importance_[model].csv`
- `target_correlations.csv`
- `feature_correlation_matrix.csv`
- `learning_curve_[model].csv`
- `bias_variance_analysis.csv`
- `feature_optimization_results.csv`
- `business_cost_analysis.csv`
- `deployment_readiness.csv`

## 🎯 Key Findings Summary

### 🏆 Best Performing Model
**Random Forest** consistently outperforms across all metrics:
- **RMSE:** 1.99 (lowest)
- **R²:** 0.811 (highest)
- **MAE:** 1.44 (lowest)
- **Business ROI:** 3724% for large-scale deployment

### 🔑 Most Important Features
1. **ChlRe** (Chlorophyll Red-edge) - Consensus across models
2. **MCARI** (Modified Chlorophyll Absorption Ratio Index)
3. **NDMI** (Normalized Difference Moisture Index)
4. **NDCI** (Normalized Difference Chlorophyll Index)

### 💡 Optimization Opportunities
1. **Feature Selection:** Reduce from 35 to 10 features using Lasso
2. **Regularization:** Address overfitting in tree-based models
3. **Data Transformation:** Handle homoscedasticity violations
4. **PCA:** 85% dimensionality reduction possible

### 💰 Business Value
- **Small Scale (1K hectares):** $183K net benefit, 1141% ROI
- **Medium Scale (10K hectares):** $1.9M net benefit, 3115% ROI
- **Large Scale (100K hectares):** $19.5M net benefit, 3724% ROI

## 🛠️ Customization

### Adding New Models
1. Add model configuration to `models_config` in any dashboard
2. Ensure model has `.fit()` and `.predict()` methods
3. Add feature importance extraction if available

### Modifying Business Scenarios
Edit `cost_parameters` and `scenarios` in `dashboard_business_intelligence.py`:
```python
cost_parameters = {
    'data_collection_cost_per_sample': 50,  # Adjust costs
    'model_development_cost': 10000,
    # ... other parameters
}
```

### Adding New Features
In `dashboard_feature_engineering.py`, add to `create_engineered_features()`:
```python
# Example: Add new vegetation index
X_train_eng['New_Index'] = X_train_eng['B08'] / X_train_eng['B04']
X_test_eng['New_Index'] = X_test_eng['B08'] / X_test_eng['B04']
```

## 🔧 Troubleshooting

### Common Issues

1. **Data File Not Found**
   - Ensure `FEI data/opt_means_cleaned.csv` exists
   - Check file path and permissions

2. **Memory Issues**
   - Reduce `n_iter` in RandomizedSearchCV
   - Decrease `n_estimators` in models

3. **Slow Performance**
   - Reduce hyperparameter search space
   - Use fewer cross-validation folds
   - Limit feature engineering

4. **Streamlit Issues**
   - Clear cache: `streamlit cache clear`
   - Restart Streamlit server
   - Check port 8501 availability

### Performance Optimization

For faster execution:
```python
# Reduce hyperparameter search
n_iter=20  # Instead of 100

# Reduce model complexity
n_estimators=100  # Instead of 1000

# Reduce CV folds
cv=3  # Instead of 5
```

## 📊 Dashboard Architecture

```
AGB Dashboard Suite
├── Core Performance (dashboard_core.py)
├── Feature Analysis (dashboard_feature_analysis.py)
├── Model Diagnostics (dashboard_model_diagnostics.py)
├── Feature Engineering (dashboard_feature_engineering.py)
├── Business Intelligence (dashboard_business_intelligence.py)
├── Web Interface (dashboard_streamlit_app.py)
└── Master Launcher (launch_dashboard.py)
```

## 🤝 Contributing

To extend the dashboard suite:

1. **Fork the repository**
2. **Create a new dashboard file** following the naming convention
3. **Implement the dashboard class** with required methods
4. **Add to master launcher**
5. **Update documentation**

### Dashboard Template
```python
class NewDashboard:
    def __init__(self, data_path='FEI data/opt_means_cleaned.csv'):
        # Initialize
        
    def load_and_preprocess_data(self):
        # Load data
        
    def analyze_something(self):
        # Core analysis
        
    def create_visualizations(self):
        # Create plots
        
    def generate_insights(self):
        # Generate insights
        
    def save_results(self):
        # Save outputs
        
    def run_full_analysis(self):
        # Run complete pipeline
```

## 📈 Future Enhancements

### Planned Features
- [ ] Deep learning models (CNN, LSTM)
- [ ] Automated hyperparameter optimization
- [ ] Real-time data integration
- [ ] Advanced ensemble methods
- [ ] Geospatial visualization
- [ ] Model monitoring and drift detection
- [ ] A/B testing framework
- [ ] Docker containerization
- [ ] Cloud deployment options

### Enhancement Ideas
- **AutoML Integration:** Automated model selection and tuning
- **Explainable AI:** SHAP and LIME integration for all models
- **Time Series Analysis:** Temporal AGB prediction
- **Multi-target Modeling:** Predict multiple forest metrics
- **Uncertainty Quantification:** Bayesian approaches
- **Edge Deployment:** Mobile/IoT model deployment

## 📞 Support

For issues, questions, or contributions:
1. **Check troubleshooting section**
2. **Review generated log files**
3. **Check data file integrity**
4. **Verify package versions**

## 📄 License

This dashboard suite is designed for AGB modeling research and analysis. Please ensure appropriate data usage and model validation for your specific use case.

---

**Built with ❤️ for AGB modeling and forest analysis**

*Dashboard Suite Version: 1.0*  
*Last Updated: 2024*
