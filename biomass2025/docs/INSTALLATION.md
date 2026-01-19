# 📦 Installation Guide - Biomass Estimation

Complete guide to setting up Biomass Estimation on your local machine.

---

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Step-by-Step Installation](#step-by-step-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## 💻 System Requirements

### Minimum Requirements

- **Operating System**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 5 GB free space (for code + data)
- **Internet**: Required for initial setup and data download

### Recommended Requirements

- **Python**: 3.9 or 3.10
- **RAM**: 16 GB or higher
- **CPU**: Multi-core processor (for parallel processing)
- **GPU**: Optional (for deep learning models)

---

## 🚀 Installation Methods

### Option 1: Quick Install (Recommended)

For most users who want to get started quickly.

### Option 2: Development Install

For contributors who want to modify the code.

### Option 3: Docker Install (Coming Soon)

For containerized deployment.

---

## 📝 Step-by-Step Installation

### 1️⃣ Install Python

#### Windows

1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ **Check "Add Python to PATH"**
4. Click "Install Now"
5. Verify:

```bash
python --version
```

#### macOS

```bash
# Using Homebrew
brew install python@3.10

# Verify
python3 --version
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# Verify
python3 --version
```

---

### 2️⃣ Install Git

#### Windows

Download and install from [git-scm.com](https://git-scm.com/download/win)

#### macOS

```bash
brew install git
```

#### Linux

```bash
sudo apt install git
```

Verify:

```bash
git --version
```

---

### 3️⃣ Clone the Repository

```bash
# Using HTTPS
git clone https://github.com/MichaelTheAnalyst/BioVision-Analytics-Hub.git

# OR using SSH
git clone git@github.com:MichaelTheAnalyst/BioVision-Analytics-Hub.git

# Navigate to directory
cd BioVision-Analytics-Hub
```

---

### 4️⃣ Create Virtual Environment

#### Windows

```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# You should see (venv) in your terminal
```

#### macOS/Linux

```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# You should see (venv) in your terminal
```

---

### 5️⃣ Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# This may take 5-10 minutes
```

#### Expected Output:

```
Collecting streamlit>=1.28.0
Collecting pandas>=1.5.0
Collecting numpy>=1.23.0
...
Successfully installed streamlit-1.28.0 pandas-2.0.0 ...
```

---

### 6️⃣ Download Sample Data (Optional)

For testing without full dataset:

```bash
# Create data directories
mkdir -p data/sample

# Download sample data (if available)
# wget https://github.com/YOUR_USERNAME/BioVision-Analytics-Hub/releases/download/v1.0.0/sample_data.zip
# unzip sample_data.zip -d data/sample/
```

---

## ✅ Verification

### Test Python Installation

```python
python -c "import sys; print(f'Python {sys.version}')"
```

**Expected**: `Python 3.8.x` or higher

### Test Package Imports

```python
python -c "import streamlit, pandas, sklearn, plotly; print('✅ All packages installed!')"
```

**Expected**: `✅ All packages installed!`

### Test Dashboard Launch

```bash
# Windows
scripts\launch_dashboard.bat

# macOS/Linux
streamlit run src/dashboard/app.py
```

**Expected**: Browser opens to `http://localhost:8501` with dashboard

---

## 🔧 Troubleshooting

### Issue 1: Python Not Found

**Error**: `'python' is not recognized as an internal or external command`

**Solution**:

```bash
# Windows: Add Python to PATH
# Control Panel > System > Advanced > Environment Variables
# Add: C:\Users\YourName\AppData\Local\Programs\Python\Python310\

# macOS/Linux: Use python3
python3 --version
```

---

### Issue 2: Permission Denied

**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**:

```bash
# Windows: Run as Administrator
# Right-click PowerShell > Run as Administrator

# macOS/Linux: Use sudo (carefully!)
sudo pip install -r requirements.txt
```

---

### Issue 3: Package Installation Fails

**Error**: `ERROR: Could not find a version that satisfies the requirement`

**Solution**:

```bash
# Update pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Try installing again
pip install -r requirements.txt

# If specific package fails, try:
pip install package_name --upgrade
```

---

### Issue 4: Virtual Environment Issues

**Error**: `cannot activate virtual environment`

**Solution**:

```bash
# Windows: Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Recreate virtual environment
rm -rf venv  # or del venv on Windows
python -m venv venv
```

---

### Issue 5: Streamlit Port Already in Use

**Error**: `Address already in use`

**Solution**:

```bash
# Use different port
streamlit run src/dashboard/app.py --server.port 8502

# OR kill existing process
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:8501 | xargs kill -9
```

---

### Issue 6: Memory Error with Large Datasets

**Error**: `MemoryError: Unable to allocate array`

**Solution**:

1. **Reduce dataset size**:

```python
# In your script
data = pd.read_csv('data.csv').sample(frac=0.5)  # Use 50% of data
```

2. **Use chunking**:

```python
chunks = pd.read_csv('data.csv', chunksize=10000)
for chunk in chunks:
    process(chunk)
```

3. **Increase swap space** (Linux/macOS)
4. **Close other applications**

---

## 🔍 Advanced Installation

### Install Development Dependencies

```bash
# Create requirements-dev.txt with:
# pytest>=7.0.0
# black>=22.0.0
# flake8>=5.0.0
# mypy>=0.990

pip install -r requirements-dev.txt
```

### Install with Specific Versions

```bash
# Lock versions for reproducibility
pip install -r requirements.txt --no-deps

# Or use pip freeze
pip freeze > requirements-lock.txt
```

### Install from Source

```bash
# Clone specific branch
git clone -b develop https://github.com/YOUR_USERNAME/BioVision-Analytics-Hub.git

# Install in editable mode
pip install -e .
```

---

## 🐳 Docker Installation (Coming Soon)

```bash
# Build image
docker build -t biovision-hub .

# Run container
docker run -p 8501:8501 biovision-hub
```

---

## 🌐 Cloud Deployment (Optional)

### Streamlit Cloud

1. Fork repository to your GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect GitHub account
4. Deploy `src/dashboard/app.py`

### Heroku

```bash
# Create Procfile
echo "web: streamlit run src/dashboard/app.py" > Procfile

# Deploy
heroku create biovision-hub
git push heroku main
```

---

## ✨ Next Steps

After successful installation:

1. ✅ **Read the User Guide**: [docs/USER_GUIDE.md](USER_GUIDE.md)
2. 📊 **Prepare your data**: See [Data Pipeline](../README.md#data-pipeline)
3. 🚀 **Launch dashboard**: `streamlit run src/dashboard/app.py`
4. 🧪 **Run tests**: `pytest tests/` (if available)
5. 📖 **Explore notebooks**: `jupyter notebook notebooks/`

---

## 📧 Still Having Issues?

- **Open an issue**: [GitHub Issues](https://github.com/MichaelTheAnalyst/biomass2025/issues)
- **Email**: support@biomass-estimation.com
- **Authors**: Nguyen Van Quy and Nguyen Hong Hai

---

## 📚 Additional Resources

- **Python Virtual Environments**: [docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html)
- **Pip Documentation**: [pip.pypa.io](https://pip.pypa.io/en/stable/)
- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io/)
- **Git Documentation**: [git-scm.com/doc](https://git-scm.com/doc)

---

<div align="center">

**🎉 Congratulations! You're ready to use Biomass Estimation! 🎉**

[Back to Main README](../README.md)

</div>

