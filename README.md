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

## Engine comparison & evaluation

Both engines solve the same Maxwell FDTD problem independently: `native` is the
own NumPy/CuPy solver (runs on the GPU via CuPy), `meep` is the MIT
[Meep](https://meep.readthedocs.io) reference toolbox (CPU only, parallel via
MPI). The numbers below come from `tests/benchmark_engines.py` on a single
reference scenario — an asymmetric PMMA slab waveguide (core d = 5 µm, air top,
aqueous bottom n = 1.336, λ = 850 nm, TE). The **guided index n_eff** is compared
against the analytic slab dispersion relation, which equals Meep's frequency-domain
eigenmode solver (MPB) exactly and is therefore the dispersion-free ground truth
(**n_eff = 1.4889**). Reproduce with the commands at the end of this section.

Hardware here: native on one NVIDIA RTX A500 (4 GB) GPU; Meep on 22 CPU cores
(MPI, `mpi_mpich` build). Meep has **no GPU support** — it scales only over CPU
cores.

### Accuracy — the two engines agree and both converge to the exact mode

| Scenario | Resolution | native n_eff (err) | Meep-FDTD n_eff (err) | Exact (MPB = analytic) |
|---|---|---|---|---|
| slab 2D | dx = 20 nm | 1.4919 (+0.0030) | 1.4915 (+0.0027) | 1.4889 |
| slab 2D | dx = 12 nm | 1.4900 (+0.0011) | 1.4898 (+0.0010) | 1.4889 |
| slab 3D | dx = 50 nm | 1.5082 (+0.0194) | 1.5075 (+0.0186) | 1.4889 |

At every matched resolution the two **independent** FDTD engines agree to within
≤ 0.0004 in n_eff, and both show the *same* small positive bias that shrinks as
the grid is refined (2D: +0.0030 → +0.0011 from dx = 20 → 12 nm). That bias is
generic FDTD **numerical dispersion** — not a bug in either solver — and it
vanishes toward the exact MPB/analytic value with finer resolution. Meep's own
time-domain FDTD carries an *even slightly larger* bias than the native solver at
the same coarse 3D resolution, which is the strongest possible confirmation that
the native physics is correct. The frequency-domain MPB eigensolver reproduces
the analytic n_eff to 4 decimals.

### Speed — native GPU vs. Meep CPU/MPI

| Scenario | native (1× GPU) | Meep (22× CPU, MPI) | native MCUPS | Meep MCUPS |
|---|---|---|---|---|
| slab 2D, dx = 20 nm | **128 s** | 598 s | 592 | 108 |
| slab 2D, dx = 12 nm | **585 s** | 1339 s | 601 | 221 |
| slab 3D, dx = 50 nm | **106 s** | 611 s | 215 | 49 |

MCUPS = million cell-updates per second, the resolution-independent FDTD
throughput metric. On this hardware the native GPU solver delivers **~2.7–5.5×
the raw throughput** and is **~2.3–5.8× faster in wall-clock time**. Meep’s
per-core throughput is modest but scales with cores and grows more efficient on
larger domains (108 → 221 MCUPS from dx = 20 → 12 nm, as the fixed MPI overhead
is amortized over more cells).

### Pros & cons

| Aspect | native (NumPy/CuPy) | Meep |
|---|---|---|
| Hardware | GPU (CuPy) **or** CPU (NumPy fallback) | CPU only, parallel via MPI; no GPU |
| Speed (this bench) | fastest (single GPU) | slower per core; needs many cores |
| Accuracy | correct physics + numerical dispersion | same, **plus** exact MPB eigensolver |
| Boundaries | Mur 1st/2nd-order ABC **or** CPML (`--boundary`, 2D full) | PML (stronger absorption) |
| Curved/oblique interfaces | staircased on the Yee grid | subpixel smoothing (less staircasing) |
| Mode handling | source-driven | `EigenModeSource` + MPB eigenmodes |
| Large / long domains | GPU **stitch** (windowed, low RAM) | full domain is RAM-heavy; mode-cascade stitch |
| Setup | `pip` + optional CuPy | Linux/WSL, conda, MPI build, MKL pin |
| Dependencies | none (own code) | external package |
| Role | primary workhorse, full control | independent authoritative cross-check |

### Limitations

**native**
- Absorbing boundary selectable via `--boundary` for **all 2D methods**: `mur1`
  (default, 1st-order Mur), `mur2` (2nd-order Mur, better at oblique incidence),
  or `cpml` (convolutional PML, best absorption). Measured residual reflection of
  a broadband point source (`tests/boundary_reflection.py`): mur1 ≈ −32 dB,
  mur2 ≈ −45 dB, cpml ≈ −57 dB. Coverage per method:
  - `full` — all four edges (`mur1`/`mur2`/`cpml`).
  - `sliding` — all four edges; the CPML ψ-fields co-move with the window and
    the Mur-2 history resets on each slide.
  - `stitch` — absorber on the (static) transverse **y-edges** only; the x-edges
    stay Mur/handoff (source facet + hard-overlap CW handoff). `mur2` is
    numerically incompatible with the hard handoff drive (late-time instability,
    esp. TM), so on `stitch` it auto-upgrades to `cpml` (stable, stronger).
- No subpixel averaging → grid-staircasing of curved/oblique material interfaces.
- Coarse-resolution n_eff bias (numerical dispersion); needs finer dx for < 0.001 accuracy.
- 4 GB GPU caps full-domain 3D and the large contact lens → use `--method stitch`.
- Evanescent-decay extraction in `crossval` is experimental (sub-µm decay, unreliable).

**Meep**
- No GPU/CUDA — CPU/MPI only; matching a single GPU needs many cores.
- Linux/WSL only (on Windows, `--engine meep` is auto-offloaded to WSL).
- Full-domain runs are RAM-heavy; the *full-field* stitch handoff breaks under strong
  scattering (documented) — use the eigenmode-cascade stitch (`--meep-handoff mode`).
- Heavier install (conda, `mpi_mpich` pymeep build, MKL 2024 pin for `libmkl_rt.so.2`).

### When to use which

- **Default: native (GPU).** Fastest here, self-contained, and the only option
  that fits the full contact lens (via GPU stitch).
- **Use Meep** to cross-check the physics with a fully independent engine, for
  exact guided modes (MPB), or for setups where PML / subpixel smoothing matter.

### Reproduce

```
# native (GPU) — Windows/Linux:
python tests/benchmark_engines.py native2d 20 60
python tests/benchmark_engines.py native2d 12 60
python tests/benchmark_engines.py native3d 50 12
# Meep (in WSL, across N cores) + exact MPB reference:
mpirun -np 22 python tests/benchmark_engines.py meep2d 50 60
mpirun -np 22 python tests/benchmark_engines.py meep2d 83 60
mpirun -np 22 python tests/benchmark_engines.py meep3d 20 12
python tests/benchmark_engines.py mpb
# aggregate all runs into results/bench/ENGINE_COMPARISON.md:
python tests/benchmark_engines.py report
```

## Documentation

- [`docs/sensor_model.md`](docs/sensor_model.md) — contact-lens sensor: principle
  (TIR guiding, evanescent/ATR sensing, D1/D2/D3 detectors), physics, the two
  compute paths (brute-force FDTD vs native mode propagation
  `kontaktlinse/mode_propagation.py`), coordinates/coupling, feasibility limits,
  and CLI usage.
- [`docs/meep_setup_wsl.md`](docs/meep_setup_wsl.md) — Meep in WSL (MPI build, run).

## Notes

- **Result files** (`results/`, `*.npz`, `*.npy`, `*.gif`, …) are excluded via
  `.gitignore` (too large for Git/GitHub). Only source code is versioned.
- `planar_3d/repair_stitch_seams.py` retroactively repairs seam artifacts
  (white stripes) in older stitch results.
- GPU via CuPy is optional; without CuPy, automatic CPU fallback (NumPy).
