@echo off
cd /d "%~dp0"
set PORT=8123
echo Starting NeerWatch at http://localhost:8123
npm start
pause
