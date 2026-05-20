$env:PYTHONPATH = (Get-Location).Path
if (-not $env:API_URL) { $env:API_URL = "http://localhost:8000" }
streamlit run frontend/app.py
