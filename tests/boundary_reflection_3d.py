"""Reflexions-Benchmark der 3D-Absorber (sponge / mur1 / mur2 / cpml).

Ricker-Punktquelle (treibt Ez) in der Mitte eines NxNxN-Freiraumgitters. Die an
den 6 Aussenflaechen zuruecklaufende Welle ist der Fehler. Referenz = viel
groesseres Gitter (gleiche Quelle/Sonde), dessen eigene Randreflexion die Sonde
in der Simulationsdauer noch nicht erreicht -> reines Einfallsfeld.

Reflexion R = max_t |E_test(t) - E_ref(t)| / max_t |E_ref(t)|  (an der Sonde).

Aufruf:  python tests/boundary_reflection_3d.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.backend import xp                                          # noqa: E402
from common.physics import C0, EPS0, MU0                               # noqa: E402
from planar_3d.fdtd3d_core import (_yee_step, _apply_sponge,           # noqa: E402
                                   _sponge_profile, ALL_FACES)
from planar_3d.boundaries3d import make_boundary_3d                    # noqa: E402


def _solve3d(N, dx, nsteps, boundary, src, probe, fp, t0, n_sp=12, alpha=0.3):
    dt = 0.5*dx/(C0*np.sqrt(3.0))
    f32 = xp.float32
    Ex = xp.zeros((N-1, N, N), dtype=f32); Ey = xp.zeros((N, N-1, N), dtype=f32)
    Ez = xp.zeros((N, N, N-1), dtype=f32)
    Hx = xp.zeros((N, N-1, N-1), dtype=f32); Hy = xp.zeros((N-1, N, N-1), dtype=f32)
    Hz = xp.zeros((N-1, N-1, N), dtype=f32)
    cev = f32(dt/(EPS0*dx))
    ce_x = xp.full(Ex.shape, cev, dtype=f32)
    ce_y = xp.full(Ey.shape, cev, dtype=f32)
    ce_z = xp.full(Ez.shape, cev, dtype=f32)
    Ch = f32(dt/(MU0*dx))
    g_sp = _sponge_profile(n_sp, alpha)
    bnd = make_boundary_3d(boundary, dx, dt, ALL_FACES)
    sx, sy, sz = src; px, py, pz = probe
    trace = np.empty(nsteps, dtype=np.float64)
    for n in range(nsteps):
        if bnd is not None:
            bnd.capture(Ex, Ey, Ez)
        _yee_step(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, Ch)
        arg = (np.pi*fp*(n*dt - t0))**2
        Ez[sx, sy, sz] += xp.float32((1.0 - 2.0*arg)*np.exp(-arg))     # Ricker (soft)
        if bnd is not None:
            bnd.apply(Ex, Ey, Ez)
        else:
            for F in (Ex, Ey, Ez, Hx, Hy, Hz):
                _apply_sponge(F, g_sp, n_sp, ALL_FACES)
        trace[n] = float(Ez[px, py, pz])
    return trace


def main(boundaries=('sponge', 'mur1')):
    dx = 20e-9
    N = 70
    nsteps = 320
    fp = C0/(20*dx)
    t0 = 1.0/fp
    c = N//2
    probe_off = 24
    src = (c, c, c); probe = (c, c, c + probe_off)
    PAD = 55
    Nb = N + 2*PAD
    cb = Nb//2
    src_b = (cb, cb, cb); probe_b = (cb, cb, cb + probe_off)

    print(f'Reflexions-Benchmark 3D  dx={dx*1e9:g}nm  N={N} (Ref {Nb})  steps={nsteps}  '
          f'(Backend: {"GPU" if xp.__name__=="cupy" else "CPU"})')
    ref = _solve3d(Nb, dx, nsteps, 'sponge', src_b, probe_b, fp, t0)   # Rand egal (zu weit)
    peak = np.max(np.abs(ref))
    print(f'{"Rand":<8} {"Reflexion":>12} {"dB":>9}')
    print('-'*32)
    res = {}
    for bnd in boundaries:
        tr = _solve3d(N, dx, nsteps, bnd, src, probe, fp, t0)
        R = np.max(np.abs(tr - ref))/peak
        res[bnd] = R
        print(f'{bnd:<8} {R:>12.3e} {20*np.log10(max(R,1e-12)):>8.1f}')
    return res


if __name__ == '__main__':
    main()
