@echo off
REM ============================================================================
REM GRAFISCHER EINSTIEG - startet run_gui.py (tkinter-Oberflaeche fuer alle
REM Optionen von run_simulation.py).
REM ============================================================================
setlocal
cd /d "%~dp0"
call C:\Users\AdminAlex\miniconda3\Scripts\activate.bat lens_sensor
if errorlevel 1 (echo FEHLER conda-Env & pause & exit /b 1)
python run_gui.py
endlocal
