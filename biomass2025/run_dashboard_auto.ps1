# AGB Dashboard Auto-Launcher
# This script automatically navigates to the correct directory and launches the dashboard

Write-Host "🌲 AGB Estimation Dashboard Auto-Launcher" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

# Get the current script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Script location: $scriptPath" -ForegroundColor Yellow

# Navigate to the script directory
Set-Location $scriptPath
Write-Host "Changed to directory: $(Get-Location)" -ForegroundColor Yellow

# Check if the dashboard file exists
$dashboardFile = "AGB_Dashboard.py"
if (Test-Path $dashboardFile) {
    Write-Host "✅ Dashboard file found: $dashboardFile" -ForegroundColor Green
} else {
    Write-Host "❌ Dashboard file not found: $dashboardFile" -ForegroundColor Red
    Write-Host "Current directory contents:" -ForegroundColor Yellow
    Get-ChildItem | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
    exit 1
}

# Check if Streamlit is installed
try {
    $streamlitVersion = streamlit --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Streamlit is installed" -ForegroundColor Green
        Write-Host "Version: $streamlitVersion" -ForegroundColor Cyan
    } else {
        Write-Host "❌ Streamlit not found. Installing..." -ForegroundColor Red
        pip install streamlit
    }
} catch {
    Write-Host "❌ Streamlit not found. Installing..." -ForegroundColor Red
    pip install streamlit
}

# Check if required packages are installed
Write-Host "Checking required packages..." -ForegroundColor Yellow
$requiredPackages = @("pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "joblib", "plotly")
foreach ($package in $requiredPackages) {
    try {
        python -c "import $package" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $package is installed" -ForegroundColor Green
        } else {
            Write-Host "⚠️  $package not found, will install from requirements.txt" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠️  $package not found, will install from requirements.txt" -ForegroundColor Yellow
    }
}

# Install requirements if needed
if (Test-Path "requirements.txt") {
    Write-Host "Installing requirements from requirements.txt..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Requirements installed successfully" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Some requirements may not have installed properly" -ForegroundColor Yellow
    }
}

# Find an available port
$port = 8501
$maxPort = 8510
$portFound = $false

Write-Host "Finding available port..." -ForegroundColor Yellow
for ($port = 8501; $port -le $maxPort; $port++) {
    try {
        $connection = Test-NetConnection -ComputerName "localhost" -Port $port -InformationLevel Quiet 2>$null
        if (-not $connection.TcpTestSucceeded) {
            $portFound = $true
            break
        }
    } catch {
        $portFound = $true
        break
    }
}

if (-not $portFound) {
    $port = 8501
    Write-Host "⚠️  Could not find available port, using default: $port" -ForegroundColor Yellow
}

Write-Host "🚀 Launching dashboard on port $port..." -ForegroundColor Green
Write-Host "Dashboard will open at: http://localhost:$port" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Red
Write-Host ""

# Launch the dashboard
try {
    streamlit run $dashboardFile --server.port $port --server.headless false
} catch {
    Write-Host "❌ Error launching dashboard: $_" -ForegroundColor Red
    Write-Host "Trying alternative method..." -ForegroundColor Yellow
    
    # Alternative launch method
    try {
        python -m streamlit run $dashboardFile --server.port $port --server.headless false
    } catch {
        Write-Host "❌ Failed to launch dashboard with alternative method" -ForegroundColor Red
        Write-Host "Please check your Python and Streamlit installation" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Dashboard stopped." -ForegroundColor Yellow
Write-Host "Press any key to exit..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
