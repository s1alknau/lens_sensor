# lens_sensor — FDTD simulation of a contact-lens/waveguide tear-film sensor

Self-written 2D/3D FDTD suite (NumPy/CuPy, **no** external FDTD package) for
studying a planar PMMA waveguide on a tear-film layer system
(aqueous / mucin / cornea, optionally lipid) with polystyrene beads as scatterers.
Focus: guided modes, evanescent field into the tear film (sensing), bead scattering.

## Structure

- `run_simulation.py` — **unified entry point**: contact-lens | planar geometry, in 2D | 3D, method full/sliding/stitch. Drives the solvers below.
- `run_gui.py`     — **graphical front-end** (tkinter): auto-generates a field for every `run_simulation.py` option, assembles the command, runs it and streams the log
- `run_FDTD.bat`  — thin Windows launcher (activates the conda env, forwards all args to `run_simulation.py`); `run_GUI.bat` launches the GUI
- `planar_beads/` — **2D** FDTD solver core (`fdtd2d_core.py`), waveguide + optional bead
- `planar_3d/`    — **3D** FDTD core (`fdtd3d_core.py`)
- `common/`       — shared building blocks: `physics.py` (constants, dispersion, `n_at`), `backend.py` (CuPy/NumPy), analyzer `fdtd_analyzer.py` (field, modes, evanescence, sensor, diff, …)
- `kontaktlinse/` — older/legacy standalone contact-lens FDTD (full-domain + sliding-window, 2D)
- `tests/`        — characterization test (golden master) guarding the solvers against refactors

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

## Usage

Prefer a window? Launch the GUI (a form with every option, a live log, and
dry-run / start / stop buttons):

```
python run_gui.py        # or run_GUI.bat on Windows
```

Otherwise everything runs through the single entry point `run_simulation.py`
(`run_FDTD.bat` is just a Windows wrapper that activates the conda env and
forwards its arguments):

```
python run_simulation.py --geometry planar --dim 3 --method full
python run_simulation.py --geometry planar --dim 2 --method stitch
python run_simulation.py --geometry lens   --dim 2 --scenario DED
python run_simulation.py --geometry lens   --dim 3 --scenario Gesund --length-um 60
python run_simulation.py --dry-run ...        # show resolved config only
python run_simulation.py --help               # full option list
```

`--geometry lens` uses the flattened contact-lens layer stack (thick PMMA lens
+ dry-eye scenario); the full ~14 mm lens is memory-bound and runs only in
stitch mode (the script warns and switches automatically). `--geometry planar`
is the thin waveguide with an optional scattering bead.

**Two engines** — `--engine native` (default, own NumPy/CuPy solver) or
`--engine meep` (the [Meep](https://meep.readthedocs.io) FDTD toolbox as an
independent cross-check). Meep is Linux-only; on Windows `--engine meep` is
transparently offloaded to the WSL2 `lens_sensor` env (see
`docs/meep_setup_wsl.md`) and writes analyzer-compatible NPZ into the same
`results/`. In the GUI the engine is just a dropdown. Example:

```
python run_simulation.py --engine meep --geometry planar --dim 2 --length-um 8
```

For the Meep engine, `--method stitch` (planar 2D) runs a windowed
eigenmode-cascade stitch — only one window in memory at a time, validated
against a Meep full-domain run. `--meep-modes M` carries M guided modes through
the handoff (captures mode conversion; radiation is not carried and shows up as
guided-power loss). For large/long domains the native stitch solver
(`--engine native --method stitch`) remains the right tool.

Evaluate results with the analyzer:

```
python common/fdtd_analyzer.py results/<file>_frames.npz
```

Run the characterization test (fast, CPU) after code changes:

```
python tests/test_characterization.py
```

## Notes

- **Result files** (`results/`, `*.npz`, `*.npy`, `*.gif`, …) are excluded via
  `.gitignore` (too large for Git/GitHub). Only source code is versioned.
- `planar_3d/repair_stitch_seams.py` retroactively repairs seam artifacts
  (white stripes) in older stitch results.
- GPU via CuPy is optional; without CuPy, automatic CPU fallback (NumPy).
