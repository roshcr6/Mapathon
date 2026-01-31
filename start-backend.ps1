Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MAPATHON BACKEND SERVER" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "c:\Users\lenovo\OneDrive\Desktop\mapathon\backend"

Write-Host "[1/2] Checking Python dependencies..." -ForegroundColor Yellow
python --version

Write-Host ""
Write-Host "[2/2] Starting FastAPI backend server..." -ForegroundColor Yellow
Write-Host "Backend will run on: http://localhost:8000" -ForegroundColor Green
Write-Host "API docs available at: http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
