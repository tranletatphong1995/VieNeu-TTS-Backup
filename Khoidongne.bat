@echo off
title Khoi chay VieNeu-TTS (Gradio)
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

echo ===================================================
echo      Dang khoi dong VieNeu-TTS...
echo ===================================================

:: Kiem tra xem moi truong ao .venv co ton tai khong
if exist ".venv\Scripts\python.exe" (
    echo [INFO] Tim thay moi truong ao .venv. Dang khoi chay...
    .venv\Scripts\python.exe gradio_app.py
) else (
    echo [INFO] Khong tim thay .venv, dang thu chay bang lenh uv...
    uv run gradio_app.py
)

echo.
echo ===================================================
echo      Ung dung da dong hoac xay ra loi.
echo ===================================================
pause
