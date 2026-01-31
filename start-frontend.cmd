@echo off
echo ========================================
echo MAPATHON FRONTEND SERVER
echo ========================================
echo.

cd /d "c:\Users\lenovo\OneDrive\Desktop\mapathon\frontend"

echo [1/2] Checking Node.js...
node --version
npm --version
echo.

echo [2/2] Starting React development server...
echo Frontend: http://localhost:5175
echo.
echo Press Ctrl+C to stop
echo.

npm run dev
pause
