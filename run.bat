@echo off
chcp 65001 >nul 2>&1
REM ALBuMS Streamlit - Quick Start Script

echo.
echo ===============================================
echo  ALBuMS Streamlit Application - Quick Start
echo ===============================================
echo.

REM Get the script directory
set "SCRIPT_DIR=%~dp0"

REM Check if .venv exists
if not exist "%SCRIPT_DIR%.venv" (
    echo ERROR: Virtual environment not found
    echo.
    echo Please run install script first:
    echo   install.bat
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call "%SCRIPT_DIR%.venv\Scripts\activate.bat"

if errorlevel 1 (
    echo ERROR: Cannot activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated OK
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Streamlit is not installed
    echo.
    echo Please run install script first:
    echo   install.bat
    echo.
    pause
    exit /b 1
)

echo Starting Streamlit application...
echo.
echo Application will open in browser: http://localhost:8501
echo.
echo (Press Ctrl+C to stop server)
echo.

REM Run streamlit
streamlit run streamlit_app.py
