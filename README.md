# lens_sensor — FDTD simulation of a contact-lens/waveguide tear-film sensor

Self-written 2D/3D FDTD suite (NumPy/CuPy, **no** external FDTD package) for
studying a planar PMMA waveguide on a tear-film layer system
(aqueous / mucin / cornea, optionally lipid) with polystyrene beads as scatterers.
Focus: guided modes, evanescent field into the tear film (sensing), bead scattering.

## Structure

- `run_simulation.py` — **unified entry point**: contact-lens | planar geometry, in 2D | 3D, method full/sliding/stitch. Drives the solvers below.
- `run_gui.py`     — **graphical front-end** (tkinter): a tool selector switches
  between **FDTD-Simulation** (`run_simulation.py`) and **Moden-Propagation**
  (`kontaktlinse/mode_propagation.py`); for the chosen tool it auto-generates a
  field for every CLI option, assembles the command, runs it and streams the log
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

**Absorption (complex refractive index).** Every layer and the bead accept a
complex index `n = n' + i·kappa`, either directly (`--n-aqueous 1.331+0.001i`) or
as an extinction coefficient on top of the dispersion model
(`--kappa-aqueous 0.001`; also `--kappa-wg/-lipid/-mucin/-cornea/-bead`). CAD
bodies take `n=1.50+0.002i` or `aqueous+0.001i` in the material dialog. The solver
uses `eps' = n'^2 - kappa^2` plus a conductivity `sigma = omega0 eps0 · 2 n' kappa`.
That is exact at the operating wavelength and approximate for broadband pulses.
It works in native 2D TE/TM and 3D, and in Meep (`D_conductivity`). Metals
(kappa ≥ n') would need a Drude model and are rejected. Validation:
`python tests/absorption_validation.py`. It compares the decay rate of the guided
mode with the exact complex slab eigenvalue, agreeing to < 1 % for TE and TM.

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

> **Note (material indices):** this benchmark was measured with the former
> visible-range indices (PMMA 1.491, aqueous 1.336). The solvers now take n(λ)
> from sourced Sellmeier fits in `common/physics.py` (PMMA 1.4835, tears 1.3313,
> cornea 1.3704, mucin 1.3331, lipid 1.4729 at 850 nm; see the source list
> there), for which the exact slab value is n_eff = 1.4813. The *offsets* below
> (numerical dispersion) are a property of the grid and carry over unchanged.

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
the same coarse 3D resolution. Two Yee solvers sharing the same bias is expected
(it is the standard Yee dispersion, predicted analytically: +0.0029 / +0.0010 /
+0.0187 for 2D 20 nm / 2D 12 nm / 3D 50 nm), so the agreement alone does not prove
the physics; the real evidence is the second-order convergence of both toward the
analytic value as dx is refined. At 3D 50 nm n_eff exceeds the core index, so
absolute 3D quantities at that resolution are not meaningful. The frequency-domain
MPB eigensolver reproduces the analytic n_eff to 4 decimals. Sensing physics is
checked separately in `tests/sensing_validation.py` (thin core, dn_eff/dn_aq) and
`tests/bead_subpixel_series.py` (bead vs gap).

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
- Absorbing boundary via `--boundary`: **default `cpml` in 2D and 3D** (convolutional
  PML, 10 cells, κ = 1 with a CFS term a = 2π·ε₀·f₀/10; `tests/cpml_compare.py` measured
  κ = 7 or 3 to reflect up to 25× more with only 10 cells, and the CFS term lowers
  grazing-incidence reflection by ~20 %). 3D alternative: `sponge` (graded absorber,
  fixed 1.2 µm thick so its reflection does not grow when dx is refined). `mur1` remains a
  CLI-only expert option (hidden in the GUI): 1st-order Mur with the *local* speed
  (x-ends c0/n_eff of the fundamental guided mode, transverse edges c0/n of the edge
  cells). Measured backward-wave amplitude of the guided wave (5 µm PMMA slab):
  2D mur1 0.006 (was 0.20 with the old vacuum-speed coefficient), cpml 0.002;
  3D mur1 0.012 (was 0.14), sponge 0.0015. Mur is exact only for the fundamental
  mode at normal incidence — use cpml for quantitative results. `mur2` was removed
  (no gain over local-speed mur1, unstable with the stitch handoff).
  - `full` / `sliding` — all edges (the CPML ψ-fields co-move with the sliding window).
  - `stitch` — the chosen boundary applies to the **outer** edges (y-edges, entry
    of window 0, exit of the last window). The *internal* window seams are always
    CPML in a non-assembled extension of each window.
- `transmission` is one definition for all CW methods: net CW Poynting flux
  through the planes at 10 % / 90 % of the length, integrated over the whole
  cross-section (TE exact, TM ε-weighted). The old intensity-box value is kept as
  `transmission_intensity`.
- Run length: ramp + transits at a group-index margin (1.1 × n_core); full runs
  then check CW convergence (phasor change between the last two 8-period DFT
  windows < 1 %) and extend themselves up to 2× if needed (`cw_rel_change`).
- Beads use subpixel averaging (`common/subpixel.py`): cells cut by the sphere get
  the volume-fraction mean (TE: arithmetic; TM and 3D: anisotropic per E-component,
  harmonic mean normal / arithmetic mean tangential to the surface, Farjadpour et
  al. 2006), so bead position and bead–core gap are continuous instead of snapping
  to the grid (`BEAD_SUBPIXEL = False` restores the hard mask for comparison;
  `tests/bead_subpixel_series.py` checks it). Planar layer interfaces lie on the
  grid; CAD bodies are still voxelized without smoothing.
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
