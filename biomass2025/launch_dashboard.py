"""
AGB Model Dashboard Launcher
============================

This script provides a centralized launcher for all dashboard components.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'plotly', 'scikit-learn', 'lightgbm', 'xgboost', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def run_core_dashboard():
    """Run the core model performance dashboard"""
    print("🚀 Running Core Model Performance Dashboard...")
    subprocess.run([sys.executable, "dashboard_core.py"])

def run_feature_analysis():
    """Run the feature analysis dashboard"""
    print("🔍 Running Feature Analysis Dashboard...")
    subprocess.run([sys.executable, "dashboard_feature_analysis.py"])

def run_model_diagnostics():
    """Run the model diagnostics dashboard"""
    print("🔬 Running Model Diagnostics Dashboard...")
    subprocess.run([sys.executable, "dashboard_model_diagnostics.py"])

def run_feature_engineering():
    """Run the feature engineering dashboard"""
    print("⚙️ Running Feature Engineering Dashboard...")
    subprocess.run([sys.executable, "dashboard_feature_engineering.py"])

def run_business_intelligence():
    """Run the business intelligence dashboard"""
    print("💼 Running Business Intelligence Dashboard...")
    subprocess.run([sys.executable, "dashboard_business_intelligence.py"])

def run_streamlit_app():
    """Launch the interactive Streamlit web dashboard"""
    print("🌐 Launching Interactive Web Dashboard...")
    print("🔗 Dashboard will open in your browser at: http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard_streamlit_app.py"])

def run_all_dashboards():
    """Run all dashboard components sequentially"""
    print("🚀 Running All Dashboard Components...")
    print("This will take several minutes to complete.\n")
    
    dashboards = [
        ("Core Performance", run_core_dashboard),
        ("Feature Analysis", run_feature_analysis),
        ("Model Diagnostics", run_model_diagnostics),
        ("Feature Engineering", run_feature_engineering),
        ("Business Intelligence", run_business_intelligence)
    ]
    
    for name, func in dashboards:
        print(f"\n{'='*50}")
        print(f"Running {name} Dashboard")
        print('='*50)
        func()
        print(f"✅ {name} Dashboard completed")

def main():
    """Main launcher interface"""
    print("🌲 AGB Model Dashboard Suite")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check data file
    if not os.path.exists("FEI data/opt_means_cleaned.csv"):
        print("❌ Data file not found: FEI data/opt_means_cleaned.csv")
        print("Please ensure the data file is in the correct location.")
        return
    
    print("\nAvailable Dashboard Components:")
    print("1. 📊 Core Model Performance Dashboard")
    print("2. 🔍 Feature Analysis Dashboard")
    print("3. 🔬 Model Diagnostics Dashboard")
    print("4. ⚙️ Feature Engineering Dashboard")
    print("5. 💼 Business Intelligence Dashboard")
    print("6. 🌐 Interactive Web Dashboard (Streamlit)")
    print("7. 🚀 Run All Dashboards")
    print("0. ❌ Exit")
    
    while True:
        try:
            choice = input("\nSelect dashboard to run (0-7): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                run_core_dashboard()
            elif choice == "2":
                run_feature_analysis()
            elif choice == "3":
                run_model_diagnostics()
            elif choice == "4":
                run_feature_engineering()
            elif choice == "5":
                run_business_intelligence()
            elif choice == "6":
                run_streamlit_app()
            elif choice == "7":
                run_all_dashboards()
            else:
                print("❌ Invalid choice. Please select 0-7.")
                continue
                
            print(f"\n✅ Dashboard execution completed!")
            
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard launcher interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
