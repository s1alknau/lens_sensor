@echo off
REM ============================================================================
REM VEREINHEITLICHTER FDTD-LAUNCHER
REM   Aktiviert die conda-Umgebung und reicht ALLE Argumente an run_simulation.py
REM   durch (Kontaktlinse | planar, in 2D | 3D, Methode full/sliding/stitch).
REM   Ohne Argumente wird die Hilfe angezeigt.
REM ============================================================================
setlocal
cd /d "%~dp0"
call C:\Users\AdminAlex\miniconda3\Scripts\activate.bat lens_sensor
if errorlevel 1 (echo FEHLER conda-Env & pause & exit /b 1)

if "%~1"=="" (
  echo(
  echo ================= FDTD LAUNCHER =================
  echo   Ein Einstieg fuer alle Faelle ^(run_simulation.py^):
  echo(
  echo   Beispiele:
  echo     run_FDTD.bat --geometry planar --dim 3 --method full
  echo     run_FDTD.bat --geometry planar --dim 2 --method stitch
  echo     run_FDTD.bat --geometry lens   --dim 2 --scenario DED
  echo     run_FDTD.bat --geometry lens   --dim 3 --scenario Gesund --length-um 60
  echo     run_FDTD.bat --dim 3 --calibrate 30      ^(nur Laufzeit-Schaetzung^)
  echo =================================================
  echo(
  echo Vollstaendige Optionsliste:
  python run_simulation.py --help
  goto :eof
)

python -u run_simulation.py %*
endlocal
