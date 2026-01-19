# ⚡ Quick Start Guide

Get Biomass Estimation running in 5 minutes!

---

## 🚀 Super Fast Setup

### 1. Prerequisites Check

```bash
# Check Python version (need 3.8+)
python --version

# Check Git installed
git --version
```

### 2. Clone & Install

```bash
# Clone repository
git clone https://github.com/MichaelTheAnalyst/BioVision-Analytics-Hub.git
cd BioVision-Analytics-Hub

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Dashboard

```bash
# Windows
scripts\launch_dashboard.bat

# macOS/Linux  
streamlit run dashboard_streamlit_app.py
```

### 4. Open Browser

Dashboard opens automatically at: **http://localhost:8501**

---

## 🎯 First Time Using?

### Step 1: Load Data
- Dashboard auto-loads `FEI data/opt_means_cleaned.csv`
- Data overview appears in sidebar

### Step 2: Train Models
- Click **"📊 Model Performance"** in sidebar
- Click **"🚀 Train All Models"**
- Wait 2-5 minutes

### Step 3: Explore Results
- View model comparison charts
- Check feature importance
- Explore spatial analysis
- Review diagnostics

---

## 📁 Using Your Own Data?

Your data needs:

**Features (columns):**
- B01-B12 (Sentinel-2 bands)
- NDVI, NDMI, NDWI, etc. (vegetation indices)

**Target (column):**
- AGB_2017 (or similar biomass column)

**Format:**
- CSV file
- Numeric values
- No missing data in key columns

**Location:**
Put your CSV in `data/processed/` folder

---

## ⚠️ Troubleshooting

### Dashboard Won't Start

```bash
# Check if Streamlit installed
pip list | grep streamlit

# Reinstall if needed
pip install streamlit --upgrade

# Try different port
streamlit run dashboard_streamlit_app.py --server.port 8502
```

### Import Errors

```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### Memory Errors

```python
# Reduce dataset size
# Add this to your script:
data = data.sample(frac=0.5)  # Use 50% of data
```

---

## 📚 Next Steps

Once running, check out:

1. **[User Guide](docs/USER_GUIDE.md)**: Complete feature walkthrough
2. **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup
3. **[README](README.md)**: Full project overview

---

## 🎓 Want to Customize?

### Change Hyperparameters

Edit in dashboard sidebar or in source files:
- `src/models/random_forest/train_rf.py`
- `src/models/lightgbm/train_lgbm.py`

### Add New Features

See feature engineering section:
- Dashboard: **"🔧 Feature Engineering"**
- Code: `src/dashboard/feature_engineering.py`

### Modify Visualizations

Edit Plotly charts:
- `src/visualization/` directory
- `src/dashboard/app.py`

---

## 💡 Pro Tips

1. **Save Your Models**: Click "💾 Save Models & Results" after training
2. **Export Data**: Models saved to `models/saved_models/`
3. **View Results**: Metrics saved to `outputs/results/`
4. **Check Outputs**: Figures saved to `outputs/figures/`

---

## 🆘 Need Help?

- **Issues**: [GitHub Issues](https://github.com/MichaelTheAnalyst/BioVision-Analytics-Hub/issues)
- **Docs**: Check `docs/` folder
- **Email**: michaelnazary@gmail.com

---

<div align="center">

**🎉 Happy Analyzing! 🎉**

[Full README](README.md) | [User Guide](docs/USER_GUIDE.md) | [Installation](docs/INSTALLATION.md)

</div>

