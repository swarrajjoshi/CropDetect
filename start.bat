@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found. Follow the setup steps in README.md.
  pause
  exit /b 1
)
echo Starting CropDetect at http://127.0.0.1:5000
echo Press Ctrl+C to stop the server.
".venv\Scripts\python.exe" app.py
