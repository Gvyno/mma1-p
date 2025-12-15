@echo off
echo Starting Flask Application...
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Try different Python commands
where python >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo Using: python
    python app.py
    goto :end
)

where py >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo Using: py
    py app.py
    goto :end
)

where python3 >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo Using: python3
    python3 app.py
    goto :end
)

echo ERROR: Python not found!
echo Please install Python or activate your virtual environment.
pause

:end

