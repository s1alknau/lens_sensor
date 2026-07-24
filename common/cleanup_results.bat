@echo off
REM ============================================================================
REM Cleanup des results\-Ordners: alte Dateien loeschen, Demo-Ergebnisse behalten
REM ============================================================================

setlocal
cd /d "%~dp0"

if not exist results (
    echo Kein results\-Ordner gefunden in %CD%
    pause
    exit /b 1
)

cd results

echo ============================================
echo Cleanup im Ordner: %CD%
echo ============================================
echo.
echo Folgende ALTE Dateien werden geloescht:
echo.
echo   01_geometry.png
echo   02_sensor_results.png
echo   03_detector_impact.png
echo   data.pkl
echo   detector_impact.txt
echo   sensor_summary.txt
echo   overnight_HIRES_*.log     (alte HIRES-Test-Logs)
echo   overnight_HIRES_~0,8dt    (Datei mit fehlerhaftem Pfad)
echo   Neuer Ordner\             (leer)
echo.
echo Folgendes wird BEHALTEN:
echo.
echo   demo_*.gif, demo_*.pkl, demo_*_frames.npz   (Demo-Ergebnisse)
echo   demo_*.log                                   (Demo-Log von heute)
echo.
set /p CONFIRM="Wirklich loeschen? (y/N): "
if /i not "%CONFIRM%"=="y" (
    echo Abgebrochen.
    cd ..
    pause
    exit /b 0
)

echo.
echo Loesche ...
del /q "01_geometry.png"          2>nul && echo   [OK] 01_geometry.png
del /q "02_sensor_results.png"    2>nul && echo   [OK] 02_sensor_results.png
del /q "03_detector_impact.png"   2>nul && echo   [OK] 03_detector_impact.png
del /q "data.pkl"                 2>nul && echo   [OK] data.pkl
del /q "detector_impact.txt"      2>nul && echo   [OK] detector_impact.txt
del /q "sensor_summary.txt"       2>nul && echo   [OK] sensor_summary.txt
del /q overnight_HIRES_*.log      2>nul && echo   [OK] overnight_HIRES_*.log
del /q "overnight_HIRES_~0,8dt"   2>nul && echo   [OK] overnight_HIRES_~0,8dt
rmdir /q /s "Neuer Ordner"        2>nul && echo   [OK] Neuer Ordner\

echo.
echo ============================================
echo Verbleibende Dateien:
echo ============================================
dir /b | sort

cd ..
echo.
pause
