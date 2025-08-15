param(
  [string]$Port = "8000"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating Python venv for streaming service..."
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r processor\streaming\requirements.txt

Write-Host "Starting streaming service on port $Port..."
uvicorn processor.streaming.app:app --host 0.0.0.0 --port $Port --reload
