@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title VieNeu-TTS - Install requirements

echo.
echo ============================================================
echo  VieNeu-TTS - One-click installer
echo ============================================================
echo.

set "INSTALL_MODE=auto"
if not "%~1"=="" set "INSTALL_MODE=%~1"

call :find_python
if errorlevel 1 goto :fail

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating virtual environment...
    "%PYTHON_EXE%" -m venv .venv
    if errorlevel 1 goto :fail
) else (
    echo [1/5] Virtual environment already exists.
)

set "VENV_PY=.venv\Scripts\python.exe"
set "VENV_UV=.venv\Scripts\uv.exe"

echo [2/5] Upgrading installer tools...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel uv
if errorlevel 1 goto :fail

echo [3/5] Detecting install profile...
set "UV_GROUPS="
if /I "%INSTALL_MODE%"=="cpu" (
    set "PROFILE=cpu"
) else if /I "%INSTALL_MODE%"=="gpu" (
    set "PROFILE=gpu"
    set "UV_GROUPS=--group gpu"
) else if /I "%INSTALL_MODE%"=="omnivoice" (
    set "PROFILE=omnivoice"
    set "UV_GROUPS=--group omnivoice"
) else if /I "%INSTALL_MODE%"=="full" (
    set "PROFILE=full"
    set "UV_GROUPS=--group gpu --group omnivoice"
) else (
    where nvidia-smi >nul 2>&1
    if errorlevel 1 (
        set "PROFILE=cpu"
    ) else (
        set "PROFILE=gpu"
        set "UV_GROUPS=--group gpu"
    )
)

echo       Profile: !PROFILE!
if "!PROFILE!"=="cpu" (
    echo       CPU mode installs the Turbo backend and Gradio UI.
) else (
    echo       GPU mode installs CUDA/GPU dependencies from pyproject.toml.
)

echo [4/5] Installing project dependencies...
if exist "%VENV_UV%" (
    "%VENV_UV%" sync !UV_GROUPS!
) else (
    "%VENV_PY%" -m uv sync !UV_GROUPS!
)
if errorlevel 1 goto :fail

echo [5/5] Verifying installation...
"%VENV_PY%" -c "import importlib.util, sys; checks=['gradio','numpy','scipy','vieneu']; missing=[name for name in checks if importlib.util.find_spec(name) is None]; print('Missing: ' + ', '.join(missing)) if missing else print('Core packages OK'); sys.exit(1 if missing else 0)"
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  Installation completed.
echo  Run START_VIENEU_TTS.bat to launch the app.
echo ============================================================
echo.
pause
exit /b 0

:find_python
echo [0/5] Looking for Python 3.10+...
set "PYTHON_EXE="
for %%C in ("py -3.12" "py -3.11" "py -3.10" "python") do (
    if not defined PYTHON_EXE (
        for /f "tokens=*" %%P in ('%%~C -c "import sys; sys.version_info >= (3, 10) and print(sys.executable)" 2^>nul') do (
            if not "%%P"=="" set "PYTHON_EXE=%%P"
        )
    )
)
if not defined PYTHON_EXE (
    echo [ERROR] Python 3.10+ was not found.
    echo Install Python from https://www.python.org/downloads/
    exit /b 1
)
echo       Python: %PYTHON_EXE%
exit /b 0

:fail
echo.
echo ============================================================
echo  Installation failed.
echo  Try one of these commands:
echo    INSTALL_REQUIREMENTS.bat cpu
echo    INSTALL_REQUIREMENTS.bat gpu
echo    INSTALL_REQUIREMENTS.bat full
echo ============================================================
echo.
pause
exit /b 1
