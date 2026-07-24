@echo off
REM ============================================================================
REM KALIBRIERUNG der 3D-FDTD auf DIESER GPU (RTX A500).
REM Misst die echte Zell-Update-Rate an der gewaehlten Domaene und rechnet die
REM volle Laufzeit hoch. STARTET DEN VOLLEN LAUF NICHT - nur Messung.
REM Die Bead-Domaene deckt auch den leeren Waveguide ab (gleiches Solver-Grid;
REM ein Bead aendert die Update-Rate praktisch nicht).
REM ============================================================================
REM === KONFIGURATION (an run_beads_3D.bat Defaults halten!) ===================
set CAL_STEPS=30
REM Anzahl gemessener Steps (nach 5 Warmup-Steps). 30 reicht meist.

set RES=20
set LAMBDA=850

REM --- Domaene (wie run_beads_3D.bat) ---
set WG_MAT=pmma
set BEAD_MAT=polystyrol
set DIAM=0.5
set LX=6
set LZ=5
set AIR=2
set WG_THICK=5
set TEAR=4
set WG_TIEFE=3
REM === ENDE KONFIGURATION =====================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"
call C:\Users\AdminAlex\miniconda3\Scripts\activate.bat lens_sensor
if errorlevel 1 (echo FEHLER: Conda-Env 'lens_sensor' nicht aktivierbar. & pause & exit /b 1)

echo [Pre-Check] CuPy + GPU ...
python -c "import cupy as cp; print('CuPy', cp.__version__, '-', cp.cuda.runtime.getDeviceProperties(0)['name'].decode())"
if errorlevel 1 (echo FEHLER: CuPy/GPU nicht bereit. & pause & exit /b 1)
echo.

echo ======= KALIBRIERUNG BEAD-/WAVEGUIDE-DOMAENE =======
python -u demo_beads_3d.py --wg-material !WG_MAT! --bead-material !BEAD_MAT! ^
    --bead-diameter !DIAM! --gpu --resolution !RES! --lambda-nm !LAMBDA! ^
    --lx !LX! --lz !LZ! --air !AIR! --tear !TEAR! --wg-thickness !WG_THICK! ^
    --wg-width !WG_TIEFE! --calibrate !CAL_STEPS!

echo.
echo === KALIBRIERUNG FERTIG ===
echo Die Hochrechnung oben gilt fuer EINEN Lauf mit diesen Domaene-Werten.
echo Fuer den echten Lauf: run_beads_3D.bat starten (mit/ohne Bead via Referenz).
pause
