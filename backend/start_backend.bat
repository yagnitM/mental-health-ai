@echo off
echo Starting Mental Health AI Backend...
echo.

cd /d "%~dp0"

if not exist "venv\" (
    echo Virtual environment not found. Running setup...
    call setup.bat
)

call venv\Scripts\activate
python main.py

pause