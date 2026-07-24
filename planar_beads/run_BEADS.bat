@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo === Bead-Demo Setup ===
call C:\Users\AdminAlex\miniconda3\Scripts\activate.bat lens_sensor
if errorlevel 1 (echo FEHLER conda-Env & pause & exit /b 1)

echo.
echo --- Methode ---  1=Full-Domain  2=Sliding-Window(Puls)  3=Stitch(voller WG, CW-Handoff)
set METHOD=1
set /p METHOD="Methode [1=Full]: "
if "!METHOD!"=="2" (set METHOD_ARG=sliding) else if "!METHOD!"=="3" (set METHOD_ARG=stitch) else (set METHOD_ARG=full)
set WINW=350
set SLIDE=150
if not "!METHOD_ARG!"=="full" set /p WINW="  Fensterbreite um [350]: "
if "!WINW!"=="" set WINW=350
if not "!METHOD_ARG!"=="full" set /p SLIDE="  Schrittweite um (< Fenster!) [150]: "
if "!SLIDE!"=="" set SLIDE=150

echo.
echo --- WG-Material ---  1=Polystyrol  2=PMMA  3=beide
set WGSEL=2
set /p WGSEL="Auswahl [2=PMMA]: "
if "!WGSEL!"=="1" (set WG_LIST=polystyrol) else if "!WGSEL!"=="3" (set WG_LIST=polystyrol pmma) else (set WG_LIST=pmma)
echo --- Bead-Material ---  1=Polystyrol  2=PMMA  3=beide
set BSEL=1
set /p BSEL="Auswahl [1=Polystyrol]: "
if "!BSEL!"=="2" (set BEAD_LIST=pmma) else if "!BSEL!"=="3" (set BEAD_LIST=polystyrol pmma) else (set BEAD_LIST=polystyrol)
echo.
set REF=N
set /p REF="Auch Referenz OHNE Bead rechnen (fuer Differenz/Normierung)? (j/N): "

echo.
echo --- Bead-Durchmesser (ein Lauf pro Durchmesser) ---
set DIAMS=0.2 0.5 1 2 4
set /p DIAMS="Durchmesser um, Leerzeichen-getrennt [0.2 0.5 1 2 4]: "
if "!DIAMS!"=="" set DIAMS=0.2 0.5 1 2 4

echo.
echo --- FDTD ---
set /p RES="Resolution in nm [20]: "
if "!RES!"=="" set RES=20
set /p WG_T="Waveguide-Dicke in um [5.0]: "
if "!WG_T!"=="" set WG_T=5.0
set LAMBDAS=532 850
set /p LAMBDAS="Wellenlaengen nm, Leerzeichen-getrennt [532 850]: "
if "!LAMBDAS!"=="" set LAMBDAS=532 850
echo.
echo --- Laser (VCSEL) ---
set /p WAIST="Waist um [2.0]: "
if "!WAIST!"=="" set WAIST=2.0
set /p TILT="Tilt deg [0.0]: "
if "!TILT!"=="" set TILT=0.0
set /p OFFSET="y-Offset um [0.0]: "
if "!OFFSET!"=="" set OFFSET=0.0
set /p SRCT="Typ cw/pulse [cw]: "
if "!SRCT!"=="" set SRCT=cw
set /p POL="Polarisation s(TE,Ez)/p(TM,Ey) [s]: "
if "!POL!"=="" set POL=s
set /p SNAP="Anzahl Snapshots [12]: "
if "!SNAP!"=="" set SNAP=12

echo.
echo --- Geometrie (ENTER = Default) ---
set /p LENGTH="Propagationslaenge um [1000]: "
set /p INGAP="Einkoppelabstand Laser-zu-WG um [0]: "
set /p BEADX="Bead-x-Position um [Mitte]: "
echo.
echo --- Schichten (von WG nach unten, -y-Richtung) ---
echo     Reihenfolge: Air / WG / Lipid / Aqueous / Mucin / Cornea
set /p TLIP="Lipid-Dicke um (direkt unter WG, 0=keine) [0]: "
set /p NLIP="  n Lipid [1.480]: "
set /p TAQ="Aqueous-Dicke um (Traenenfilm, Bead darin) [6]: "
if "!TAQ!"=="" set TAQ=6
set /p NAQ="  n Aqueous [1.336]: "
set /p TMUC="Mucin-Dicke um [2]: "
set /p NMUC="  n Mucin [1.342]: "
set /p NCOR="Cornea (Halbraum darunter) - n Cornea [1.376]: "
set OPT=
if not "!LENGTH!"=="" set OPT=!OPT! --length !LENGTH!
if not "!INGAP!"=="" set OPT=!OPT! --input-gap !INGAP!
if not "!BEADX!"=="" set OPT=!OPT! --bead-x !BEADX!
if not "!TLIP!"=="" set OPT=!OPT! --t-lipid !TLIP!
if not "!NLIP!"=="" set OPT=!OPT! --n-lipid !NLIP!
if not "!NAQ!"=="" set OPT=!OPT! --n-aqueous !NAQ!
if not "!TMUC!"=="" set OPT=!OPT! --t-mucin !TMUC!
if not "!NMUC!"=="" set OPT=!OPT! --n-mucin !NMUC!
if not "!NCOR!"=="" set OPT=!OPT! --n-cornea !NCOR!

echo.
echo === Konfig ===
echo Methode=!METHOD_ARG!  Window=!WINW!um Slide=!SLIDE!um
echo WG=!WG_LIST!   Bead=!BEAD_LIST!   Durchmesser=!DIAMS!
echo Res=!RES!nm  WG-Dicke=!WG_T!um  Lambdas=!LAMBDAS!nm  waist=!WAIST! tilt=!TILT! typ=!SRCT!
echo Ein Bead pro Lauf, in der FESTEN Aqueous (!TAQ!um, an WG angrenzend) an der Grenzflaeche; Facet am WG-Ende.
set GO=Y
set /p GO="Start? (Y/n): "
if /i "!GO!"=="n" (echo Abgebrochen & pause & exit /b 0)

if not exist results mkdir results
for /f %%a in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set DT=%%a
set LOG=results\beads_RUN_!DT!.log
echo === BEADS START %date% %time% (method=!METHOD_ARG! RES=!RES!nm) === > "!LOG!"

for %%W in (!WG_LIST!) do (
  for %%B in (!BEAD_LIST!) do (
    for %%D in (!DIAMS!) do (
      for %%L in (!LAMBDAS!) do (
        echo.
        echo ==== [%time%] WG=%%W Bead=%%B d=%%D L=%%L  !METHOD_ARG! ====
        echo ==== WG=%%W Bead=%%B d=%%D L=%%L method=!METHOD_ARG! ==== >> "!LOG!"
        python -u demo_beads.py --wg-material %%W --bead-material %%B --bead-diameter %%D --gpu --method !METHOD_ARG! --window-w !WINW! --slide !SLIDE! --resolution !RES! --wg-thickness !WG_T! --t-aqueous !TAQ! --lambda-nm %%L --vcsel-waist !WAIST! --vcsel-tilt !TILT! --vcsel-offset !OFFSET! --source-type !SRCT! --polarization !POL! --snapshots !SNAP! !OPT! >> "!LOG!" 2>&1
        if errorlevel 1 (echo   FEHLER %%W/%%B/d%%D/L%%L) else (echo   OK %%W/%%B/d%%D/L%%L)
      )
    )
  )
)
if /i "!REF!"=="j" (
  for %%W in (!WG_LIST!) do (
    for %%D in (!DIAMS!) do (
      for %%L in (!LAMBDAS!) do (
        echo.
        echo ==== [%time%] REFERENZ WG=%%W d=%%D L=%%L ohne Bead ====
        echo ==== REFERENZ WG=%%W d=%%D L=%%L ==== >> "!LOG!"
        python -u demo_beads.py --wg-material %%W --bead-material polystyrol --bead-diameter %%D --no-bead --gpu --method !METHOD_ARG! --window-w !WINW! --slide !SLIDE! --resolution !RES! --wg-thickness !WG_T! --t-aqueous !TAQ! --lambda-nm %%L --vcsel-waist !WAIST! --vcsel-tilt !TILT! --vcsel-offset !OFFSET! --source-type !SRCT! --polarization !POL! --snapshots !SNAP! !OPT! >> "!LOG!" 2>&1
        if errorlevel 1 (echo   FEHLER REF %%W/d%%D/L%%L) else (echo   OK REF %%W/d%%D/L%%L)
      )
    )
  )
)
echo.
echo === FERTIG === Log: !LOG!
echo Auswertung: python ..\common\fdtd_analyzer.py results\beads_*_frames.npz  (Modi: Sensor / Diff (A-B) / Moden)
pause
