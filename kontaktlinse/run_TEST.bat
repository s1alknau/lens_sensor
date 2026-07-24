@echo off
REM ============================================================================
REM Sliding-FDTD QUICK TEST (nur Gesund, niedrige Aufloesung)
REM ============================================================================
REM Resolution 400 nm, Window 600 x 2500 um, Slide 500 um
REM Laufzeit: ~5-10 Minuten auf RTX A500
REM ZWECK: Geometrie-Validierung (D1/D2/D3-Position, VCSEL, Stirnflaeche) im GIF
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Conda-Env aktivieren
call C:\Users\AdminAlex\miniconda3\Scripts\activate.bat lens_sensor
if errorlevel 1 (
    echo FEHLER: Conda-Env 'lens_sensor' konnte nicht aktiviert werden.
    pause
    exit /b 1
)

echo ============================================
echo Sliding-FDTD QUICK TEST
echo Szenario:   Gesund
echo Resolution: 400 nm
echo Window:     600 x 2500 um
echo Slide:      500 um
echo Geschaetzt: ~5-10 min
echo ============================================
echo.

echo [Pre-Check] CuPy + GPU ...
python -c "import cupy as cp; print('CuPy', cp.__version__, '- GPU:', cp.cuda.runtime.getDeviceProperties(0)['name'].decode())"
if errorlevel 1 (
    echo FEHLER: CuPy/GPU nicht bereit.
    pause
    exit /b 1
)
echo.

if not exist results mkdir results

echo === START === %date% %time%
echo.
python -u sliding_window_fdtd.py ^
    --scenario Gesund ^
    --gpu ^
    --resolution 400 ^
    --window-w 600 ^
    --window-h 2500 ^
    --slide 500

if errorlevel 1 (
    echo.
    echo FEHLER beim Test-Lauf.
    pause
    exit /b 1
)

echo.
echo ============================================
echo TEST FERTIG !
echo Pruefe:  results\sliding_Gesund.gif
echo Browse:  python view_gif.py results\sliding_Gesund.gif
echo ============================================
echo.
pause
