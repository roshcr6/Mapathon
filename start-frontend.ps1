Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MAPATHON FRONTEND SERVER" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "c:\Users\lenovo\OneDrive\Desktop\mapathon\frontend"

Write-Host "[1/2] Checking Node.js..." -ForegroundColor Yellow
node --version
npm --version

Write-Host ""
Write-Host "[2/2] Starting React development server..." -ForegroundColor Yellow
Write-Host "Frontend will run on: http://localhost:5175 (or next available port)" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

npm run dev
