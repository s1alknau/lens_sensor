# lens_sensor — FDTD-Simulation eines Kontaktlinsen-/Wellenleiter-Tränenfilm-Sensors

Selbst geschriebene 2D/3D-FDTD-Suite (NumPy/CuPy, **kein** externes FDTD-Paket) zur
Untersuchung eines planaren PMMA-Wellenleiters auf einem Tränenfilm-Schichtsystem
(Aqueous / Mucin / Cornea, optional Lipid) mit Polystyrol-Beads als Streuern.
Fokus: geführte Moden, evaneszentes Feld ins Tränenfilm (Sensorik), Bead-Streuung.

## Struktur

- `planar_beads/` — **2D**-Wellenleiter + Bead (`demo_beads.py`), Launcher `run_BEADS.bat`
- `planar_3d/`   — **3D**-Kern (`fdtd3d_core.py`), CLI `demo_beads_3d.py`, Launcher `run_beads_3D.bat`, GPU-Laufzeitschätzer `run_CALIBRATE.bat`
- `common/`      — gemeinsamer Analyzer `fdtd_analyzer.py` (Feld, Moden, Evaneszenz, Sensor, Diff, …)
- `kontaktlinse/`— ältere/Legacy-Kontaktlinsen-FDTD (full-domain + sliding-window)
- `run_FDTD.bat` — Einstiegs-Launcher (fragt 3D/2D und leitet weiter)

## Methoden

- **full**   — volle Domäne, zeitschreitend ab t=0 (zeigt Einlaufen/Transient)
- **sliding**— mitlaufendes Fenster (2D, lange Strecken)
- **stitch** — Gebiets-Zerlegung: voller gefüllter CW-Wellenleiter, speicherschonend
  (jedes Fenster bis zum Steady-State, CW-Phasor-Handoff, Sponge-Ränder werden beim
  Assemblieren getrimmt, um Naht-Artefakte zu vermeiden)

Jede Rechnung liefert zwei Darstellungen: **einlaufend** (transient) und
**eingeschwungen** (CW-Phasen über eine Periode) — im Analyzer umschaltbar.
Polarisation **s (TE, Ez)** / **p (TM, Ey)** wählbar.

## Installation

```
git clone <REPO-URL> lens_sensor
cd lens_sensor
pip install -r requirements.txt        # CPU-Abhängigkeiten
#   oder:  pip install .                # gleiche Abhängigkeiten via pyproject
```

Optional mit GPU (passend zur CUDA-Version):

```
pip install cupy-cuda12x               # bzw. cupy-cuda11x
#   oder:  pip install ".[gpu-cuda12]"
```

Ohne CuPy läuft alles automatisch auf der CPU. `tkinter` (Datei-Dialog des
Analyzers) gehört zu Python; unter Linux ggf. `sudo apt install python3-tk`.

## Nutzung (Windows, z.B. conda-Env `lens_sensor`)

```
run_FDTD.bat                      # geführter Einstieg (3D/2D)
python common\fdtd_analyzer.py results\<datei>_frames.npz   # Auswertung
```

## Hinweise

- **Ergebnis-Dateien** (`results/`, `*.npz`, `*.npy`, `*.gif`, …) sind per `.gitignore`
  ausgeschlossen (zu groß für Git/GitHub). Nur Quellcode wird versioniert.
- `planar_3d/repair_stitch_seams.py` repariert nachträglich Naht-Artefakte
  (weiße Streifen) in älteren Stitch-Ergebnissen.
- GPU via CuPy optional; ohne CuPy automatischer CPU-Fallback (NumPy).
