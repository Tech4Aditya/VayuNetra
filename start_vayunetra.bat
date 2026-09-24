@echo off
cd /d "%~dp0"
cd ..
echo Starting VayuNetra backend from project root:
cd
python -m uvicorn backend.app:app --reload --port 8000
pause
