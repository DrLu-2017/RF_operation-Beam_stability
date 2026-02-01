@echo off
chcp 65001 >nul 2>&1
REM ALBuMS Streamlit Installation Script for Windows

echo.
echo ===============================================
echo  ALBuMS Streamlit Application - Setup
echo ===============================================
echo.

REM Get the script directory
set "SCRIPT_DIR=%~dp0"

REM Check if .venv exists
if not exist "%SCRIPT_DIR%.venv" (
    echo ERROR: Virtual environment not found: %SCRIPT_DIR%.venv
    echo.
    echo Please create virtual environment first:
    echo   python -m venv .venv
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call "%SCRIPT_DIR%.venv\Scripts\activate.bat"

if errorlevel 1 (
    echo ERROR: Cannot activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated OK
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: pip upgrade failed, continuing...
)

echo.

REM Install requirements
echo Installing project dependencies...
echo This may take a few minutes...
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed
    echo.
    echo If network timeout, try using domestic mirror:
    echo   pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    echo or
    echo   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo.
    pause
    exit /b 1
)

echo.
echo ===============================================
echo Installation completed successfully!
echo ===============================================
echo.
echo Now you can run Streamlit application:
echo   streamlit run streamlit_app.py
echo.
echo Application will open in browser: http://localhost:8501
echo.
pause
