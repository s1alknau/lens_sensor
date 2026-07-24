@echo off
REM ============================================================================
REM GEMEINSAMER FDTD-LAUNCHER
REM   Fragt zuerst: 3D oder 2D-planar?  Bei 2D: Methode full oder sliding?
REM   Leitet dann in den passenden (interaktiven) Lauf weiter.
REM   ENTER = Vorgabewert [in Klammern].
REM ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo(
echo ================= FDTD LAUNCHER =================
echo   3d = echte 3D-FDTD (Bead / planarer Waveguide, kurze Strecke,
echo        volle Yee-Zelle, GPU-speicherbegrenzt)
echo   2d = 2D planarer Waveguide + Bead (lange Strecke, full/sliding/stitch,
echo        gut fuer Moden / evaneszentes Feld / Sensor-Auswertung)
echo =================================================
set "DIM=3d"
set /p "DIM=Rechnungstyp 3d oder 2d [!DIM!]: "

if /i "!DIM!"=="3d" goto do3d
goto do2d

:do3d
echo(
echo -> starte interaktiven 3D-Lauf (planar_3d\run_beads_3D.bat) ...
call "%~dp0planar_3d\run_beads_3D.bat"
goto :eof

:do2d
echo(
echo -> starte interaktiven 2D-Bead/Waveguide-Lauf (planar_beads\run_BEADS.bat) ...
call "%~dp0planar_beads\run_BEADS.bat"
goto :eof
