@echo off
REM One-click setup and run for Windows. Double-click this file.
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

python --version >nul 2>&1
if errorlevel 1 goto nopython

if not exist .venv (
  echo [1/4] Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate.bat

echo [2/4] Installing packages (first time takes a few minutes)...
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
if errorlevel 1 goto failed

echo [3/4] Running offline tests...
python -m pytest -q
if errorlevel 1 goto failed

if exist .env goto run
echo.
echo Get a free key at https://dashboard.cohere.com/api-keys
set /p COHERE_KEY=Paste your Cohere API key here and press Enter: 
> .env echo COHERE_API_KEY=%COHERE_KEY%
echo Key saved to .env (this file is never uploaded to GitHub).

:run
echo [4/4] Running the experiment with Cohere...
python run.py
if errorlevel 1 goto failed
echo.
echo DONE. Results are in the "results" folder. Tell Claude "done".
pause
exit /b 0

:nopython
echo Python is not installed or not on PATH.
echo Install Python 3.11+ from https://www.python.org/downloads/
echo IMPORTANT: tick "Add python.exe to PATH" during installation, then run this file again.
pause
exit /b 1

:failed
echo.
echo Something failed. Take a screenshot of this window and send it to Claude.
pause
exit /b 1
