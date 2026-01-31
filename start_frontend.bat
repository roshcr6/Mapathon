@echo off
echo ========================================
echo MAPATHON FRONTEND STARTUP
echo ========================================
echo.

echo [1/2] Installing Node.js dependencies...
cd frontend
call npm install
echo.

echo [2/2] Starting React development server...
echo Frontend will run on http://localhost:3000
echo.
call npm run dev
