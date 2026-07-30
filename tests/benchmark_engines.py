"""Engine-Benchmark: eigener NumPy/CuPy-Solver ("native") vs. Meep.

Misst pro Lauf GLEICHZEITIG
  * Geschwindigkeit : Wall-Zeit, Gitterzellen, Zeitschritte -> MCUPS
                      (Mega-Cell-Updates pro Sekunde, das uebliche FDTD-Mass)
  * Genauigkeit     : gefuehrter Brechungsindex n_eff des Referenz-Slabs
                      gegen die analytische Loesung (dispersionsfreie Wahrheit)

Referenz-Szenario (identisch fuer beide Engines): asymmetrischer PMMA-Slab,
Kern d=5 um, Luft oben (n=1.0), Aqueous unten (n=1.336), lambda=850 nm, TE.
Die analytische Slab-Mode (= Meep-MPB exakt) ist die Grundwahrheit.

Jeder Lauf schreibt einen JSON-Datensatz nach results/bench/. 'report' fasst
alle Datensaetze zu Markdown-Tabellen fuer die README zusammen.

Wer laeuft wo:
  native*  -> Windows/GPU-Python (eigener Solver, CuPy falls verfuegbar)
  meep*/mpb-> WSL, CPU/MPI  (Meep hat KEINEN GPU-Support)

Beispiele:
  # nativ (GPU):
  python tests/benchmark_engines.py native2d 20 60
  python tests/benchmark_engines.py native3d 50 12
  # Meep (in WSL, optional unter mpirun -np N):
  mpirun -np 8 python tests/benchmark_engines.py meep2d 50 60
  mpirun -np 8 python tests/benchmark_engines.py meep3d 20 12
  python tests/benchmark_engines.py mpb            # exakte Eigenmode-Referenz
  # Auswertung:
  python tests/benchmark_engines.py report
"""
import os
import re
import sys
import json
import time
import glob
import io
import contextlib

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS = os.path.join(_ROOT, 'tests')
for _p in (_ROOT, _TESTS, os.path.join(_ROOT, 'planar_beads'), os.path.join(_ROOT, 'planar_3d')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import crossval as cv                       # analytische Referenz + Meep-Bausteine
from common.physics import n_at

_BENCH_DIR = os.path.join(_ROOT, 'results', 'bench')

# --- Referenz-Szenario (ein Ort der Wahrheit) -------------------------------
N_CORE = n_at('pmma', 850.0)
N_TOP = 1.0
N_BOT = n_at('aqueous', 850.0)
D_UM = 5.0
LAM_NM = 850.0
POL = 's'                                   # TE


def _truth_neff():
    return cv.slab_neff_analytic(N_CORE, N_TOP, N_BOT, D_UM, LAM_NM, POL, 0)


def _device_name():
    """GPU-Name, falls CuPy einsatzbereit ist; sonst 'CPU (NumPy)'."""
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props['name']
        return 'GPU ' + (name.decode() if isinstance(name, (bytes, bytearray)) else str(name))
    except Exception:
        return 'CPU (NumPy)'


def _neff_from_line(line, dx_um):
    """n_eff aus der Steigung der entwrappten Phase des komplexen CW-Feldes
    entlang der Propagationsachse (eingeschwungener Mittelteil)."""
    i0, i1 = int(0.35*len(line)), int(0.85*len(line))
    ph = np.unwrap(np.angle(line[i0:i1]))
    x = np.arange(len(ph))*dx_um
    beta = abs(np.polyfit(x, ph, 1)[0])
    return beta/(2*np.pi/(LAM_NM*1e-3))


def _mpi_rank_size():
    try:
        from mpi4py import MPI
        c = MPI.COMM_WORLD
        return c.Get_rank(), c.Get_size()
    except Exception:
        return 0, 1


def _save(rec):
    """Nur Rang 0 schreibt (bei MPI). Datensatz -> results/bench/<key>.json."""
    rank, size = _mpi_rank_size()
    rec['np_ranks'] = size
    if rank != 0:
        return
    os.makedirs(_BENCH_DIR, exist_ok=True)
    key = '%s_%s_dx%g_L%g.json' % (rec['engine'], rec['mode'],
                                   rec.get('dx_nm', 0), rec.get('length_um', 0))
    with open(os.path.join(_BENCH_DIR, key), 'w') as fh:
        json.dump(rec, fh, indent=2)
    print('[bench] %-14s %-4s dx=%-4gnm  cells=%.3gM steps=%-6s wall=%7.2fs  '
          'MCUPS=%7.1f  n_eff=%.4f (err %+.4f)  ranks=%d'
          % (rec['engine'], rec['mode'], rec.get('dx_nm', 0),
             rec['cells']/1e6, rec['steps'], rec['wall_s'], rec['mcups'],
             rec['neff'], rec['neff_err'], size))


# ---------------------------------------------------------------------------
# NATIV (Windows/GPU)
# ---------------------------------------------------------------------------
def bench_native2d(dx_nm, length_um):
    import tempfile
    import fdtd2d_core as f2d
    truth = _truth_neff()
    prev = os.getcwd(); tmp = tempfile.mkdtemp()
    buf = io.StringIO()
    try:
        os.chdir(tmp)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            r = f2d.run_beads(
                wg_mat='pmma', bead_mat='polystyrol', bead_d_um=0.0, dx_nm=dx_nm,
                save_frames=True, n_snapshots=8, wg_thickness_um=D_UM, lambda_nm=LAM_NM,
                method='full', place_bead=False, length_um=length_um, polarization=POL,
                t_aqueous_um=8.0, t_mucin_um=2.0)
        wall = time.perf_counter() - t0
        cwm = r.get('cw_frames_memmap'); cwf = r.get('cw_frames') or []
        arr = np.asarray(np.load(cwm, mmap_mode='r')[:len(cwf)])
    finally:
        os.chdir(prev)
    nfr, Nx, Ny = arr.shape
    ez = arr[0].astype(np.float64) - 1j*arr[nfr//4 or 1].astype(np.float64)
    dx_um = dx_nm/1000.0
    iy = min(max(int(round((D_UM/2 + cv_tear2d())/dx_um)), 0), Ny-1)
    neff = _neff_from_line(ez[:, iy], dx_um)
    steps = _parse_steps(buf.getvalue())
    cells = Nx*Ny
    _save(dict(engine='native', mode='2d', device=_device_name(), dx_nm=dx_nm,
               length_um=length_um, cells=cells, steps=steps, wall_s=wall,
               mcups=(cells*steps/wall/1e6 if steps else float('nan')),
               neff=neff, neff_err=neff-truth, ref='analytic'))


def bench_native3d(dx_nm, length_um, lz_um=6.0):
    import tempfile
    import fdtd3d_core as f3d
    truth = _truth_neff()
    prev = os.getcwd(); tmp = tempfile.mkdtemp()
    buf = io.StringIO()
    try:
        os.chdir(tmp)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            r = f3d.run_3d(
                label='bench3d', wg_n=n_at('pmma', LAM_NM), t_wg_um=D_UM, air_um=3.0,
                tear_um=8.0, lz_um=lz_um, t_aq_um=8.0, t_mu_um=2.0,
                n_aq=n_at('aqueous', LAM_NM), n_mu=n_at('mucin', LAM_NM),
                n_co=n_at('cornea', LAM_NM), lam_nm=LAM_NM, lx_um=length_um,
                dx_nm=dx_nm, n_snapshots=8, source_type='cw', polarization=POL,
                wg_width_um=None, bead=None)
        wall = time.perf_counter() - t0
        cw = np.asarray(r.get('cw_vol'))
    finally:
        os.chdir(prev)
    nfr = cw.shape[0]
    ez = cw[0].astype(np.float64) - 1j*cw[nfr//4 or 1].astype(np.float64)
    dx_um = dx_nm/1000.0
    iy = min(max(int(round((D_UM/2 + 8.0)/dx_um)), 0), ez.shape[1]-1)
    iz = ez.shape[2]//2
    neff = _neff_from_line(ez[:, iy, iz], dx_um)
    steps = _parse_steps(buf.getvalue())
    cells = int(np.prod(ez.shape))
    _save(dict(engine='native', mode='3d', device=_device_name(), dx_nm=dx_nm,
               length_um=length_um, cells=cells, steps=steps, wall_s=wall,
               mcups=(cells*steps/wall/1e6 if steps else float('nan')),
               neff=neff, neff_err=neff-truth, ref='analytic'))


def cv_tear2d():
    """TEAR_BUFFER (um) des 2D-Solvers fuer den Kern-y-Index."""
    import fdtd2d_core as f2d
    return f2d.TEAR_BUFFER*1e6


def _parse_steps(text):
    m = re.search(r'Steps:\s*(\d+)', text)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# MEEP (WSL, CPU/MPI)
# ---------------------------------------------------------------------------
def bench_meep2d(res, length_um):
    """Meep 2D-Zeitbereichs-FDTD desselben Slabs (z-invariant), n_eff aus der
    Phase des CW-DFT-Feldes. Wall-Zeit = reine sim.run()-Zeit."""
    import meep as mp
    truth = _truth_neff()
    air, tear, dpml = 3.0, 6.0, 1.0
    ly = D_UM + air + tear
    cy = 0.5*((D_UM + air) + (-tear))
    cell = mp.Vector3(length_um, ly, 0)
    geom = [
        mp.Block(mp.Vector3(mp.inf, tear, mp.inf), center=mp.Vector3(0, -0.5*tear - cy),
                 material=mp.Medium(index=N_BOT)),
        mp.Block(mp.Vector3(mp.inf, D_UM, mp.inf), center=mp.Vector3(0, 0.5*D_UM - cy),
                 material=mp.Medium(index=N_CORE)),
    ]
    f0 = 1000.0/LAM_NM
    src = mp.EigenModeSource(
        src=mp.ContinuousSource(frequency=f0),
        center=mp.Vector3(-length_um/2 + dpml + 0.3, 0.5*D_UM - cy),
        size=mp.Vector3(0, ly), eig_band=1, direction=mp.X,
        eig_match_freq=True, eig_parity=mp.ODD_Z)
    sim = mp.Simulation(cell_size=cell, resolution=res, geometry=geom, sources=[src],
                        default_material=mp.Medium(index=N_TOP),
                        boundary_layers=[mp.PML(dpml)], force_complex_fields=True)
    dft = sim.add_dft_fields([mp.Ez], f0, 0, 1, center=mp.Vector3(0, 0.5*D_UM - cy),
                             size=mp.Vector3(length_um, 0), yee_grid=False)
    until = 3.0*length_um*N_CORE + 40
    t0 = time.perf_counter()
    sim.run(until=until)
    wall = time.perf_counter() - t0
    line = np.asarray(sim.get_dft_array(dft, mp.Ez, 0))
    neff = _neff_from_line(line, 1.0/res)
    cells = int(round(res*length_um) * round(res*ly))
    steps = int(round(until*res/0.5))       # dt = Courant/res, Courant=0.5
    _save(dict(engine='meep', mode='2d', device='CPU (Meep/MPI)', dx_nm=1000.0/res,
               length_um=length_um, cells=cells, steps=steps, wall_s=wall,
               mcups=cells*steps/wall/1e6, neff=neff, neff_err=neff-truth, ref='analytic'))


def bench_meep3d(res, length_um, lz_um=6.0):
    """Meep 3D-Zeitbereichs-FDTD (die Engine hinter run_meep --dim 3)."""
    import meep as mp
    truth = _truth_neff()
    air, tear, dpml = 3.0, 6.0, 1.0
    ly = D_UM + air + tear
    cy = 0.5*((D_UM + air) + (-tear))
    cell = mp.Vector3(length_um, ly, lz_um)
    geom = [
        mp.Block(mp.Vector3(mp.inf, tear, mp.inf), center=mp.Vector3(0, -0.5*tear - cy, 0),
                 material=mp.Medium(index=N_BOT)),
        mp.Block(mp.Vector3(mp.inf, D_UM, mp.inf), center=mp.Vector3(0, 0.5*D_UM - cy, 0),
                 material=mp.Medium(index=N_CORE)),
    ]
    f0 = 1000.0/LAM_NM
    src = mp.EigenModeSource(
        src=mp.ContinuousSource(frequency=f0),
        center=mp.Vector3(-length_um/2 + dpml + 0.3, 0.5*D_UM - cy, 0),
        size=mp.Vector3(0, ly, lz_um), eig_band=1, direction=mp.X,
        eig_match_freq=True, eig_parity=mp.ODD_Z)
    sim = mp.Simulation(cell_size=cell, resolution=res, geometry=geom, sources=[src],
                        default_material=mp.Medium(index=N_TOP),
                        boundary_layers=[mp.PML(dpml)], force_complex_fields=True)
    dft = sim.add_dft_fields([mp.Ez], f0, 0, 1, center=mp.Vector3(0, 0.5*D_UM - cy, 0),
                             size=mp.Vector3(length_um, 0, 0), yee_grid=False)
    until = 3.0*length_um*N_CORE + 40
    t0 = time.perf_counter()
    sim.run(until=until)
    wall = time.perf_counter() - t0
    line = np.asarray(sim.get_dft_array(dft, mp.Ez, 0))
    neff = _neff_from_line(line, 1.0/res)
    cells = int(round(res*length_um) * round(res*ly) * round(res*lz_um))
    steps = int(round(until*res/0.5))
    _save(dict(engine='meep', mode='3d', device='CPU (Meep/MPI)', dx_nm=1000.0/res,
               length_um=length_um, cells=cells, steps=steps, wall_s=wall,
               mcups=cells*steps/wall/1e6, neff=neff, neff_err=neff-truth, ref='analytic'))


def bench_mpb(res=40):
    """Meep-MPB Eigenmode-Solver: exakte (frequenzbereichs-)Referenz fuer n_eff.
    Kein FDTD -> keine Numerik-Dispersion; Wall-Zeit dient nur als Orientierung."""
    truth = _truth_neff()
    t0 = time.perf_counter()
    neff = cv.slab_neff_meep(N_CORE, N_TOP, N_BOT, D_UM, LAM_NM, POL, res=res)
    wall = time.perf_counter() - t0
    _save(dict(engine='meep-mpb', mode='2d', device='CPU (Meep/MPB)', dx_nm=1000.0/res,
               length_um=0.0, cells=0, steps=0, wall_s=wall, mcups=float('nan'),
               neff=neff, neff_err=neff-truth, ref='analytic'))


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
def report():
    files = sorted(glob.glob(os.path.join(_BENCH_DIR, '*.json')))
    recs = [json.load(open(f)) for f in files]
    truth = _truth_neff()
    lines = []
    lines.append('## Engine comparison (auto-generated)\n')
    lines.append('Reference scenario: asymmetric PMMA slab, core d=5 um, air top, '
                 'aqueous bottom (n=1.336), lambda=850 nm, TE.  '
                 'Analytic n_eff (= Meep-MPB, dispersion-free ground truth) = **%.4f**.\n'
                 % truth)
    # Accuracy
    lines.append('### Accuracy (guided index n_eff vs. analytic)\n')
    lines.append('| Engine | Mode | Resolution (dx) | n_eff | error |')
    lines.append('|---|---|---|---|---|')
    for r in sorted(recs, key=lambda r: (r['mode'], r['engine'], -r['dx_nm'])):
        lines.append('| %s | %s | %.0f nm | %.4f | %+.4f |'
                     % (r['engine'], r['mode'].upper(), r['dx_nm'], r['neff'], r['neff_err']))
    # Speed
    lines.append('\n### Speed (same scenario, wall clock)\n')
    lines.append('| Engine | Mode | dx | Grid cells | Steps | Ranks | Wall time | MCUPS | Hardware |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for r in sorted(recs, key=lambda r: (r['mode'], r['engine'], -r['dx_nm'])):
        if r['engine'] == 'meep-mpb':
            continue
        lines.append('| %s | %s | %.0f nm | %.2f M | %d | %d | %.1f s | %.1f | %s |'
                     % (r['engine'], r['mode'].upper(), r['dx_nm'], r['cells']/1e6,
                        r['steps'], r.get('np_ranks', 1), r['wall_s'], r['mcups'],
                        r['device']))
    out = '\n'.join(lines) + '\n'
    os.makedirs(_BENCH_DIR, exist_ok=True)
    with open(os.path.join(_BENCH_DIR, 'ENGINE_COMPARISON.md'), 'w') as fh:
        fh.write(out)
    print(out)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    a = sys.argv
    cmd = a[1] if len(a) > 1 else 'report'
    if cmd == 'native2d':
        bench_native2d(float(a[2]) if len(a) > 2 else 20.0, float(a[3]) if len(a) > 3 else 60.0)
    elif cmd == 'native3d':
        bench_native3d(float(a[2]) if len(a) > 2 else 50.0, float(a[3]) if len(a) > 3 else 12.0)
    elif cmd == 'meep2d':
        bench_meep2d(int(a[2]) if len(a) > 2 else 50, float(a[3]) if len(a) > 3 else 60.0)
    elif cmd == 'meep3d':
        bench_meep3d(int(a[2]) if len(a) > 2 else 20, float(a[3]) if len(a) > 3 else 12.0)
    elif cmd == 'mpb':
        bench_mpb(int(a[2]) if len(a) > 2 else 40)
    elif cmd == 'report':
        report()
    else:
        print(__doc__)
        sys.exit(2)
