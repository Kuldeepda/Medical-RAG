$env:PYTHONPATH = (Get-Location).Path
if (Test-Path .env) { Get-Content .env | ForEach-Object { if ($_ -match '^\s*([^#][^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1].Trim())" -Value $matches[2].Trim() } } }
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
