@echo off
echo ========================================
echo MAPATHON BACKEND STARTUP
echo ========================================
echo.

echo [1/3] Creating Python virtual environment...
if not exist "backend\venv\" (
    python -m venv backend\venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)
echo.

echo [2/3] Installing dependencies...
call backend\venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r backend\requirements.txt
echo.

echo [3/3] Starting FastAPI server...
echo Backend will run on http://localhost:8000
echo API docs available at http://localhost:8000/docs
echo.
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
