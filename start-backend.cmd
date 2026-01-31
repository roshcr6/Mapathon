@echo off
echo ========================================
echo MAPATHON BACKEND SERVER
echo ========================================
echo.

cd /d "c:\Users\lenovo\OneDrive\Desktop\mapathon\backend"

echo [1/2] Checking Python...
python --version
echo.

echo [2/2] Starting FastAPI backend server...
echo Backend: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
