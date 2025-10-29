@echo off
echo Creating virtual environment for backend...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✅ Backend setup complete!
echo.
echo To run the backend:
echo   1. cd backend
echo   2. venv\Scripts\activate
echo   3. python main.py
echo.
pause