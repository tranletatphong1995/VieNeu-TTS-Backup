@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title VieNeu-TTS - Launcher

echo.
echo ============================================================
echo  VieNeu-TTS - One-click launcher
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Virtual environment was not found.
    echo         Starting installer first...
    call "%~dp0INSTALL_REQUIREMENTS.bat"
    if errorlevel 1 exit /b 1
)

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
set "PORT=7860"

call :choose_port

echo [INFO] Python : %PYTHON_EXE%
echo [INFO] App    : gradio_app.py
echo [INFO] URL    : http://127.0.0.1:%PORT%
echo.
echo The browser will open automatically. Keep this window open.
echo Press Ctrl+C here to stop the app.
echo.

set "VIENEU_PORT=%PORT%"
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process 'http://127.0.0.1:%PORT%'"
"%PYTHON_EXE%" gradio_app.py

echo.
echo VieNeu-TTS has stopped.
pause
exit /b 0

:choose_port
for %%P in (7860 7861 7862 7863 7864 7865) do (
    netstat -ano | findstr /R /C:":%%P .*LISTENING" >nul 2>&1
    if errorlevel 1 (
        set "PORT=%%P"
        exit /b 0
    )
)
echo [WARN] Ports 7860-7865 look busy. Falling back to 7860.
set "PORT=7860"
exit /b 0
