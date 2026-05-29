# Script to run Streamlit app for Champions League Predictor
param(
    [string]$Environment = "local"
)

Write-Host "🏆 Champions League Predictor - Streamlit App" -ForegroundColor Cyan
Write-Host "==========================================`n" -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "⚠️  Virtual environment not found. Creating one..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "✅ Activating virtual environment..." -ForegroundColor Green
& .venv\Scripts\Activate.ps1

# Install/update dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Green
pip install -r requirements.txt -q

# Run Streamlit app
Write-Host "`n🚀 Starting Streamlit app..." -ForegroundColor Cyan
Write-Host "📱 App will open at: http://localhost:8501" -ForegroundColor Blue
Write-Host "`nPress Ctrl+C to stop the app`n" -ForegroundColor Yellow

streamlit run app.py
