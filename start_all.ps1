# Start Medical RAG API (8000) + Streamlit UI (8501) in two windows
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Virtual env missing. Run: python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}

$env:PYTHONPATH = $root
$env:API_URL = "http://127.0.0.1:8000"

Write-Host "Starting API on http://127.0.0.1:8000 ..."
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$root'; `$env:PYTHONPATH='$root'; .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
)

Start-Sleep -Seconds 2

Write-Host "Starting UI on http://127.0.0.1:8501 ..."
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "cd '$root'; `$env:PYTHONPATH='$root'; `$env:API_URL='http://127.0.0.1:8000'; .\.venv\Scripts\streamlit.exe run frontend/app.py --server.port 8501 --server.address 127.0.0.1"
)

Write-Host ""
Write-Host "Open in browser:"
Write-Host "  API docs:  http://127.0.0.1:8000/docs"
Write-Host "  Dashboard: http://127.0.0.1:8501"
Write-Host ""
Write-Host "Keep both PowerShell windows open while using the app."
