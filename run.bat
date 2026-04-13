@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo Python virtual environment was not found: "%PYTHON%"
  exit /b 1
)

"%PYTHON%" "%ROOT%scripts\analyze_videos.py" --correct "%ROOT%correct.mp4" --wrong "%ROOT%wrong.mp4" --output-dir "%ROOT%app"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" "%ROOT%scripts\serve.py" --root "%ROOT%app" --port 8000 --open

