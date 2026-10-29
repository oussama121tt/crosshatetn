@echo off
REM Double-clic : publie sur GitHub (prive) et deploie la demo web sur Vercel
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy.ps1" %*
pause
