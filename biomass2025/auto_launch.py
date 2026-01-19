#!/usr/bin/env python3
"""
AGB Dashboard Auto-Launcher
Automatically launches the AGB Estimation Dashboard with all necessary setup
"""

import os
import sys
import subprocess
import socket
import time
import webbrowser
from pathlib import Path

def print_header():
    """Print the header for the launcher"""
    print("🌲 AGB Estimation Dashboard Auto-Launcher")
    print("=" * 50)
    print()

def check_file_exists(filename):
    """Check if a file exists"""
    if os.path.exists(filename):
        print(f"✅ {filename} found")
        return True
    else:
        print(f"❌ {filename} not found")
        return False

def check_python_package(package):
    """Check if a Python package is installed"""
    try:
        __import__(package)
        print(f"✅ {package} is installed")
        return True
    except ImportError:
        print(f"⚠️  {package} not found")
        return False

def install_requirements():
    """Install requirements from requirements.txt"""
    if os.path.exists("requirements.txt"):
        print("📦 Installing requirements...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True)
            print("✅ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing requirements: {e}")
            return False
    else:
        print("⚠️  requirements.txt not found")
        return False

def find_available_port(start_port=8501, max_port=8510):
    """Find an available port for the dashboard"""
    print("🔍 Finding available port...")
    
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                print(f"✅ Port {port} is available")
                return port
        except OSError:
            continue
    
    print(f"⚠️  No available ports found in range {start_port}-{max_port}")
    return start_port

def launch_dashboard(port):
    """Launch the Streamlit dashboard"""
    print(f"🚀 Launching dashboard on port {port}...")
    print(f"🌐 Dashboard will open at: http://localhost:{port}")
    print()
    print("Press Ctrl+C to stop the dashboard")
    print()
    
    # Try to open browser after a short delay
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open(f"http://localhost:{port}")
            print("✅ Browser opened automatically")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"Please manually open: http://localhost:{port}")
    
    # Start browser opening in background
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "AGB_Dashboard.py",
            "--server.port", str(port),
            "--server.headless", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        print("Trying alternative method...")
        
        try:
            subprocess.run([
                "streamlit", "run", "AGB_Dashboard.py",
                "--server.port", str(port),
                "--server.headless", "false"
            ], check=True)
        except subprocess.CalledProcessError as e2:
            print(f"❌ Failed to launch dashboard: {e2}")
            print("Please check your Streamlit installation")
            return False
    
    return True

def main():
    """Main function to launch the dashboard"""
    print_header()
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    print(f"📍 Current directory: {current_dir}")
    
    # Check if dashboard file exists
    if not check_file_exists("AGB_Dashboard.py"):
        print("❌ Dashboard file not found!")
        print("Please run this script from the directory containing AGB_Dashboard.py")
        input("Press Enter to exit...")
        return
    
    # Check Python packages
    print("\n🔍 Checking Python packages...")
    required_packages = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "joblib", "plotly"]
    missing_packages = []
    
    for package in required_packages:
        if not check_python_package(package):
            missing_packages.append(package)
    
    # Install requirements if needed
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        if not install_requirements():
            print("❌ Failed to install requirements")
            input("Press Enter to exit...")
            return
    
    # Check Streamlit
    print("\n🔍 Checking Streamlit...")
    if not check_python_package("streamlit"):
        print("📦 Installing Streamlit...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], 
                         check=True, capture_output=True)
            print("✅ Streamlit installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Streamlit: {e}")
            input("Press Enter to exit...")
            return
    
    # Find available port
    port = find_available_port()
    
    # Launch dashboard
    print(f"\n🎯 Ready to launch dashboard on port {port}")
    input("Press Enter to launch the dashboard...")
    
    try:
        launch_dashboard(port)
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    print("\n👋 Dashboard launcher completed")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
