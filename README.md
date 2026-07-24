# lens_sensor — FDTD simulation of a contact-lens/waveguide tear-film sensor

Self-written 2D/3D FDTD suite (NumPy/CuPy, **no** external FDTD package) for
studying a planar PMMA waveguide on a tear-film layer system
(aqueous / mucin / cornea, optionally lipid) with polystyrene beads as scatterers.
Focus: guided modes, evanescent field into the tear film (sensing), bead scattering.

## Structure

- `planar_beads/` — **2D** waveguide + bead (`demo_beads.py`), launcher `run_BEADS.bat`
- `planar_3d/`   — **3D** core (`fdtd3d_core.py`), CLI `demo_beads_3d.py`, launcher `run_beads_3D.bat`, GPU runtime estimator `run_CALIBRATE.bat`
- `common/`      — shared analyzer `fdtd_analyzer.py` (field, modes, evanescence, sensor, diff, …)
- `kontaktlinse/`— older/legacy contact-lens FDTD (full-domain + sliding-window)
- `run_FDTD.bat` — entry-point launcher (asks 3D/2D and dispatches)

## Methods

- **full**   — full domain, time-stepped from t=0 (shows the transient onset)
- **sliding**— co-moving window (2D, long propagation distances)
- **stitch** — domain decomposition: fully filled CW waveguide, memory-efficient
  (each window runs to steady state, CW phasor handoff, sponge borders are
  trimmed during assembly to avoid seam artifacts)

Every run produces two views: **transient** (onset) and
**steady-state** (CW phases over one period) — switchable in the analyzer.
Polarization **s (TE, Ez)** / **p (TM, Ey)** selectable.

## Installation

Requires Python ≥ 3.9 (≥ 3.10 recommended).

```
git clone https://github.com/s1alknau/lens_sensor.git
cd lens_sensor
pip install -r requirements.txt        # CPU dependencies
#   or:  pip install .                 # same dependencies via pyproject
```

Optional with GPU (matching your CUDA version):

```
pip install cupy-cuda12x               # or cupy-cuda11x
#   or:  pip install ".[gpu-cuda12]"
```

Without CuPy, everything automatically falls back to the CPU. `tkinter`
(file dialog of the analyzer) is part of Python; on Linux install if needed
via `sudo apt install python3-tk`.

Check your setup (packages, GPU detection, RAM):

```
python common/check_environment.py
```

## Usage (Windows, e.g. conda env `lens_sensor`)

```
run_FDTD.bat                      # guided entry point (3D/2D)
python common\fdtd_analyzer.py results\<file>_frames.npz   # evaluation
```

## Notes

- **Result files** (`results/`, `*.npz`, `*.npy`, `*.gif`, …) are excluded via
  `.gitignore` (too large for Git/GitHub). Only source code is versioned.
- `planar_3d/repair_stitch_seams.py` retroactively repairs seam artifacts
  (white stripes) in older stitch results.
- GPU via CuPy is optional; without CuPy, automatic CPU fallback (NumPy).
