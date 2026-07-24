@echo off
REM HIRES Linsen-Sweep - 8 Szenarien bei 150nm Aufloesung (Sliding)
REM Editiere RES, WW, WH, SLIDE, METHOD direkt hier
setlocal enabledelayedexpansion
cd /d "%~dp0"
call C:\Users\AdminAlex\miniconda3\Scripts\activate.bat lens_sensor

REM === KONFIGURATION (editiere hier) ===
set METHOD=sliding
set RES=150
set WW=400
set WH=2000
set SLIDE=200
set LAMBDA=850
set T_LENS=250
set SCENARIO_LIST=Gesund DED Frisch Hyperosmolar MGD Mucin-Mangel Mucin-reich Lipid-reich
REM Fuer Full-Domain: METHOD=full, RES=300, WH=600 (WW+SLIDE werden ignoriert)
REM Fuer Single-Szenario: SCENARIO_LIST=Gesund
REM === ENDE ===

if not exist results mkdir results
for /f %%a in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set DT=%%a
set LOG=results\HIRES_!METHOD!_!DT!.log
echo === HIRES START %date% %time% === > "!LOG!"
echo Methode=!METHOD! Res=!RES!nm WW=!WW! WH=!WH! Slide=!SLIDE! >> "!LOG!"

set COUNT=0
for %%s in (!SCENARIO_LIST!) do set /a COUNT+=1
echo Anzahl Szenarien: !COUNT!  Methode=!METHOD!  Aufloesung=!RES!nm
set IDX=0
for %%s in (!SCENARIO_LIST!) do (
    set /a IDX+=1
    echo.
    echo ==== [%time%] Szenario !IDX!/!COUNT!: %%s ====
    echo ==== %%s ==== >> "!LOG!"
    if "!METHOD!"=="full" (
        python -u full_domain_fdtd.py --scenario %%s --gpu --resolution !RES! --window-h !WH! --lambda-nm !LAMBDA! --t-lens !T_LENS! >> "!LOG!" 2>&1
    ) else (
        python -u sliding_window_fdtd.py --scenario %%s --gpu --resolution !RES! --window-w !WW! --window-h !WH! --slide !SLIDE! --lambda-nm !LAMBDA! --t-lens !T_LENS! >> "!LOG!" 2>&1
    )
    if errorlevel 1 (echo FEHLER %%s) else (echo OK %%s)
)
echo.
echo === FERTIG === Log: !LOG!
echo Anschauen: python ..\common\fdtd_analyzer.py results\sliding_*_frames.npz
pause
