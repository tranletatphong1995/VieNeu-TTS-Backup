@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

:: ═══════════════════════════════════════════════════════════════
::   VieNeu-TTS v2.1 — One-Click Launcher
::   Chuyen van ban tieng Viet thanh giong noi
:: ═══════════════════════════════════════════════════════════════

title VieNeu-TTS - Dang khoi dong...

:: ── Mau sac & Banner ────────────────────────────────────────
echo.
echo  ╔═══════════════════════════════════════════════════════╗
echo  ║                                                       ║
echo  ║        VieNeu-TTS  -  Text to Speech                  ║
echo  ║        Chuyen van ban thanh giong noi                  ║
echo  ║        Chay hoan toan offline                          ║
echo  ║                                                       ║
echo  ╚═══════════════════════════════════════════════════════╝
echo.

:: ── Kiem tra moi truong ─────────────────────────────────────
echo [1/3] Dang kiem tra moi truong...

:: Uu tien 1: Virtual environment (.venv)
if exist ".venv\Scripts\python.exe" (
    echo       [OK] Tim thay moi truong ao .venv
    set "PYTHON_CMD=.venv\Scripts\python.exe"
    set "ENV_TYPE=.venv"
    goto :check_gpu
)

:: Uu tien 2: uv (package manager)
where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] Tim thay uv package manager
    set "PYTHON_CMD=uv run"
    set "ENV_TYPE=uv"
    goto :check_gpu
)

:: Uu tien 3: Python he thong
where python >nul 2>&1
if %errorlevel% equ 0 (
    echo       [OK] Tim thay Python he thong
    set "PYTHON_CMD=python"
    set "ENV_TYPE=system python"
    goto :check_gpu
)

:: Khong tim thay Python
echo.
echo  ╔═══════════════════════════════════════════════════════╗
echo  ║  [LOI] Khong tim thay Python hoac uv!                ║
echo  ║                                                       ║
echo  ║  Cach khac phuc:                                      ║
echo  ║  1. Cai dat Python 3.10+: https://python.org          ║
echo  ║  2. Hoac cai uv: pip install uv                       ║
echo  ║  3. Hoac tao .venv: python -m venv .venv              ║
echo  ╚═══════════════════════════════════════════════════════╝
echo.
pause
exit /b 1

:: ── Kiem tra GPU ────────────────────────────────────────────
:check_gpu
echo [2/3] Dang kiem tra phan cung...

:: Kiem tra NVIDIA GPU
set "GPU_INFO=Khong tim thay GPU (se chay Turbo CPU mode)"
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%a in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
        set "GPU_INFO=%%a"
    )
    echo       [OK] GPU: !GPU_INFO!
) else (
    echo       [--] !GPU_INFO!
)

:: ── Khoi chay ung dung ──────────────────────────────────────
:launch
echo [3/3] Dang khoi dong VieNeu-TTS...
echo.
echo  ┌───────────────────────────────────────────────────────┐
echo  │  Moi truong : %ENV_TYPE%
echo  │  Entry point: gradio_app.py
echo  │  Dia chi    : http://localhost:7860
echo  │  GPU        : !GPU_INFO!
echo  └───────────────────────────────────────────────────────┘
echo.
echo  Dang tai model... (lan dau co the mat vai phut)
echo  Nhan Ctrl+C de dung ung dung.
echo.

title VieNeu-TTS - Dang chay tai http://localhost:7860

:: ── Chay ung dung ───────────────────────────────────────────
if "%ENV_TYPE%"=="uv" (
    uv run gradio_app.py
) else (
    %PYTHON_CMD% gradio_app.py
)

:: ── Ket thuc ────────────────────────────────────────────────
echo.
echo  ╔═══════════════════════════════════════════════════════╗
echo  ║  VieNeu-TTS da dung.                                  ║
echo  ║                                                       ║
echo  ║  Neu xay ra loi, kiem tra:                             ║
echo  ║  1. Da cai day du dependencies chua?                   ║
echo  ║     uv sync --group gpu                                ║
echo  ║  2. Port 7860 co bi chiem khong?                       ║
echo  ║     netstat -ano | findstr 7860                        ║
echo  ╚═══════════════════════════════════════════════════════╝
echo.
pause
