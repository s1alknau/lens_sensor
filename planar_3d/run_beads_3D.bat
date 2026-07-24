@echo off
REM ============================================================================
REM Echte 3D-FDTD (volle Yee-Zelle) - EIN Bead (echte Kugel).
REM INTERAKTIV: alle Parameter werden abgefragt. ENTER = Vorgabewert [in Klammern].
REM Kleine Domaene, ausgelegt fuer 4 GB GPU (RTX A500).
REM ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo(
echo ================= 3D-FDTD BEAD - EINGABE =================
echo   ENTER druecken = Vorgabewert [in eckigen Klammern] uebernehmen.
echo   Listen (Wellenlaenge/Durchmesser): Werte mit LEERZEICHEN trennen.
echo ==========================================================

echo(
echo --- AUFLOESUNG / ZEIT ---
set "RES=20"
set /p "RES=Yee-Gitter-Aufloesung dx in nm [!RES!]: "
set "LAMBDA_LIST=850"
set /p "LAMBDA_LIST=Wellenlaenge(n) in nm, Leerzeichen-getrennt [!LAMBDA_LIST!]: "
set "SNAPSHOTS=4"
set /p "SNAPSHOTS=Anzahl Frames (Zeit-Schnappschuesse) [!SNAPSHOTS!]: "

echo(
echo --- GEOMETRIE / DOMAENE (um)   x=Laenge  y=Querschnitt  z=Tiefe ---
set "LAENGE=6"
set /p "LAENGE=Laenge x - Propagationsstrecke [!LAENGE!]: "
echo --- Methode ---  1=Full-Domain  2=Stitch (voller WG, speicherschonend, CW-Handoff)
set METHOD=1
set /p METHOD="Methode [1=Full]: "
set METHOD_FLAGS=
if "!METHOD!"=="2" (
  set /p WINW="  Fensterbreite um [20]: "
  if "!WINW!"=="" set WINW=20
  set /p SLIDE="  Schrittweite um (< Fenster) [12]: "
  if "!SLIDE!"=="" set SLIDE=12
  set METHOD_FLAGS=--method stitch --window-w !WINW! --slide !SLIDE!
)
set "TIEFE=5"
set /p "TIEFE=Tiefe z - 3. Dimension [!TIEFE!]: "
set "LUFT=2"
set /p "LUFT=Luft ueber WG (y) [!LUFT!]: "
set "WG_DICKE=5"
set /p "WG_DICKE=WG-Dicke (Querschnitt y) [!WG_DICKE!]: "
set "TEAR=4"
set /p "TEAR=Tear+Cornea unter WG (y) [!TEAR!]: "
set "WG_TIEFE=3"
set /p "WG_TIEFE=WG-Kernbreite in Tiefe z (0=Slab) [!WG_TIEFE!]: "
set "WG_CLAD_N=1.0"
set /p "WG_CLAD_N=Seitliches Cladding n (Luft=1.0) [!WG_CLAD_N!]: "

echo(
echo --- MATERIALIEN / BRECHZAHLEN (freie n leer=Materialtabelle) ---
set "WG_MAT=pmma"
set /p "WG_MAT=WG-Material (pmma/polystyrol) [!WG_MAT!]: "
set "BEAD_MAT=polystyrol"
set /p "BEAD_MAT=Bead-Material (pmma/polystyrol) [!BEAD_MAT!]: "
set "WG_N="
set /p "WG_N=Freie WG-Brechzahl (leer=Material) []: "
set "BEAD_N="
set /p "BEAD_N=Freie Bead-Brechzahl (leer=Material) []: "
set "N_AQUEOUS="
set /p "N_AQUEOUS=Freie n Aqueous (leer=~1.336) []: "
set "N_MUCIN="
set /p "N_MUCIN=Freie n Mucin (leer=~1.342) []: "
set "N_CORNEA="
set /p "N_CORNEA=Freie n Cornea (leer=~1.376) []: "
set "N_LIPID="
set /p "N_LIPID=Freie n Lipid (nur bei T_LIPID>0) []: "

echo(
echo --- SCHICHTDICKEN (um) ---
set "T_AQUEOUS="
set /p "T_AQUEOUS=Aqueous-Dicke (leer=auto 4 bzw 1) []: "
set "T_MUCIN=2.0"
set /p "T_MUCIN=Mucin-Dicke [!T_MUCIN!]: "
set "T_LIPID=0.0"
set /p "T_LIPID=Lipid-Dicke unter WG (0=keine) [!T_LIPID!]: "

echo(
echo --- BEAD ---
set "DIAMETER_LIST=0.5"
set /p "DIAMETER_LIST=Bead-Durchmesser in um, Leerzeichen-getrennt [!DIAMETER_LIST!]: "
set "BEAD_X="
set /p "BEAD_X=Bead-Position x (leer=Mitte lx/2) []: "
set "BEAD_Y="
set /p "BEAD_Y=Bead-Position y (leer=aufliegend -d/2) []: "
set "BEAD_Z="
set /p "BEAD_Z=Bead-Position z (leer=Mitte 0) []: "
set "RUN_REFERENCE=1"
set /p "RUN_REFERENCE=Zusaetzlich Referenzlauf OHNE Bead? (1/0) [!RUN_REFERENCE!]: "

echo(
echo --- LASER / EINKOPPLUNG ---
set "WAIST_Y=2.0"
set /p "WAIST_Y=Strahltaille y in um [!WAIST_Y!]: "
set "WAIST_Z=1.0"
set /p "WAIST_Z=Strahltaille z in um [!WAIST_Z!]: "
set "TILT=0.0"
set /p "TILT=Strahlneigung in Grad (x-y-Ebene) [!TILT!]: "
set "OFFSET_Y=0.0"
set /p "OFFSET_Y=Spot-Versatz y (rel. WG-Mitte) [!OFFSET_Y!]: "
set "OFFSET_Z=0.0"
set /p "OFFSET_Z=Spot-Versatz z (rel. Mitte) [!OFFSET_Z!]: "
set "INPUT_GAP=0.0"
set /p "INPUT_GAP=Einkoppelabstand Laser-zu-WG in um [!INPUT_GAP!]: "
set "SOURCE_TYPE=cw"
set /p "SOURCE_TYPE=Quelle cw oder pulse [!SOURCE_TYPE!]: "
set "POL=s"
set /p "POL=Polarisation s(TE,Ez) oder p(TM,Ey) [!POL!]: "

echo(
echo --- RAENDER (optional) ---
set "PEC_FACES="
set /p "PEC_FACES=PEC-Spiegel-Flaechen z.B. xmax (leer=alle absorbierend) []: "
set "END_FACET=0"
set /p "END_FACET=Luftzone am WG-Ende in um (0=aus) [!END_FACET!]: "

echo(
echo --- AUSGABE / SPEICHER ---
set "VECTOR=1"
set /p "VECTOR=Vollen Vektor Ex,Ey,Ez speichern? (1/0) [!VECTOR!]: "
set "VOL_DTYPE=float16"
set /p "VOL_DTYPE=Volumen-Speicherformat (float16/float32) [!VOL_DTYPE!]: "
set "CHECK=1"
set /p "CHECK=VRAM/RAM vor Lauf pruefen? (1/0) [!CHECK!]: "

echo(
echo ================= ZUSAMMENFASSUNG =================
echo   Res=!RES!nm  Lambda=!LAMBDA_LIST!  Frames=!SNAPSHOTS!
echo   Domain LxTxH(y): !LAENGE! x !TIEFE! x (!LUFT!+!WG_DICKE!+!TEAR!) um  WG-Tiefe=!WG_TIEFE!
echo   WG=!WG_MAT!(n=!WG_N!)  Bead=!BEAD_MAT!(n=!BEAD_N!) d=!DIAMETER_LIST!  Ref=!RUN_REFERENCE!
echo   Laser: Taille y/z=!WAIST_Y!/!WAIST_Z! Tilt=!TILT! Offset y/z=!OFFSET_Y!/!OFFSET_Z! Gap=!INPUT_GAP!
echo   Vector=!VECTOR!  dtype=!VOL_DTYPE!  Check=!CHECK!
echo ===================================================
set "GO=j"
set /p "GO=Berechnung starten? (j/n) [!GO!]: "
if /i not "!GO!"=="j" (echo Abgebrochen. & pause & exit /b 0)

call C:\Users\AdminAlex\miniconda3\Scripts\activate.bat lens_sensor
if errorlevel 1 (echo FEHLER: Conda-Env 'lens_sensor' nicht aktivierbar. & pause & exit /b 1)

REM --- Flags zusammensetzen ---------------------------------------------------
set VEC_FLAG=
if "!VECTOR!"=="1" set VEC_FLAG=--save-vector
set CHK_FLAG=
if "!CHECK!"=="1" set CHK_FLAG=--check-resources
set BND_FLAGS=--end-facet !END_FACET!
if not "!PEC_FACES!"=="" set BND_FLAGS=!BND_FLAGS! --pec-faces !PEC_FACES!

REM Gemeinsame Optionen (beide Laeufe): freie n, Schichtdicken, Laser
set OPT=
if not "!WG_N!"=="" set OPT=!OPT! --wg-n !WG_N!
if not "!BEAD_N!"=="" set OPT=!OPT! --bead-n !BEAD_N!
if not "!N_AQUEOUS!"=="" set OPT=!OPT! --n-aqueous !N_AQUEOUS!
if not "!N_MUCIN!"=="" set OPT=!OPT! --n-mucin !N_MUCIN!
if not "!N_CORNEA!"=="" set OPT=!OPT! --n-cornea !N_CORNEA!
if not "!N_LIPID!"=="" set OPT=!OPT! --n-lipid !N_LIPID!
if not "!T_AQUEOUS!"=="" set OPT=!OPT! --t-aqueous !T_AQUEOUS!
set OPT=!OPT! --t-mucin !T_MUCIN! --t-lipid !T_LIPID!
set OPT=!OPT! --source-type !SOURCE_TYPE! --polarization !POL!
set OPT=!OPT! --vcsel-tilt !TILT! --vcsel-offset-y !OFFSET_Y! --vcsel-offset-z !OFFSET_Z!
set OPT=!OPT! --input-gap !INPUT_GAP!

REM Bead-Position (nur fuer den Bead-Lauf, nicht die Referenz)
set BEADPOS=
if not "!BEAD_X!"=="" set BEADPOS=!BEADPOS! --bead-x !BEAD_X!
if not "!BEAD_Y!"=="" set BEADPOS=!BEADPOS! --bead-y !BEAD_Y!
if not "!BEAD_Z!"=="" set BEADPOS=!BEADPOS! --bead-z !BEAD_Z!

if not exist results mkdir results
for /f %%a in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set DT=%%a
set LOG=results\BEADS3D_!DT!.log
echo === BEADS-3D START %date% %time% === > "!LOG!"
echo Res=!RES!nm WG=!WG_MAT!/!WG_N! Bead=!BEAD_MAT!/!BEAD_N! Lambda=!LAMBDA_LIST! Diam=!DIAMETER_LIST! Ref=!RUN_REFERENCE! >> "!LOG!"
echo OPT=!OPT! >> "!LOG!"

for %%w in (!LAMBDA_LIST!) do (
  for %%d in (!DIAMETER_LIST!) do (
    echo.
    echo ==== [%time%] Lambda=%%w nm  Bead d=%%d um ====
    echo ==== L=%%w d=%%d ==== >> "!LOG!"
    python -u demo_beads_3d.py --wg-material !WG_MAT! --bead-material !BEAD_MAT! ^
        --bead-diameter %%d --gpu --resolution !RES! --lambda-nm %%w ^
        --lx !LAENGE! --lz !TIEFE! --air !LUFT! --tear !TEAR! --wg-thickness !WG_DICKE! ^
        --wg-width !WG_TIEFE! --wg-clad-n !WG_CLAD_N! ^
        --vcsel-waist !WAIST_Y! --vcsel-waist-z !WAIST_Z! !OPT! !BEADPOS! ^
        !VEC_FLAG! !CHK_FLAG! !BND_FLAGS! !METHOD_FLAGS! ^
        --snapshots !SNAPSHOTS! --vol-dtype !VOL_DTYPE! >> "!LOG!" 2>&1
    if errorlevel 1 (echo FEHLER L=%%w d=%%d) else (echo OK L=%%w d=%%d)
    if "!RUN_REFERENCE!"=="1" (
        echo ---- Referenz OHNE Bead L=%%w d=%%d ----
        echo ---- ref L=%%w d=%%d ---- >> "!LOG!"
        python -u demo_beads_3d.py --wg-material !WG_MAT! --bead-material !BEAD_MAT! ^
            --bead-diameter %%d --no-bead --gpu --resolution !RES! --lambda-nm %%w ^
            --lx !LAENGE! --lz !TIEFE! --air !LUFT! --tear !TEAR! --wg-thickness !WG_DICKE! ^
            --wg-width !WG_TIEFE! --wg-clad-n !WG_CLAD_N! ^
            --vcsel-waist !WAIST_Y! --vcsel-waist-z !WAIST_Z! !OPT! ^
            !VEC_FLAG! !CHK_FLAG! !BND_FLAGS! !METHOD_FLAGS! ^
            --snapshots !SNAPSHOTS! --vol-dtype !VOL_DTYPE! >> "!LOG!" 2>&1
        if errorlevel 1 (echo FEHLER ref L=%%w d=%%d) else (echo OK ref L=%%w d=%%d)
    )
  )
)
echo.
echo === FERTIG === Log: !LOG!
echo Anschauen: python ..\common\fdtd_analyzer.py results\beads3d_*_frames.npz
pause
