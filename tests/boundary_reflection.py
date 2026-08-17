"""Reflexions-Benchmark der absorbierenden Raender (mur1 / mur2 / cpml).

Eigenstaendiger 2D-TE-Loesungskern (Ez, Hx, Hy) im Freiraum: eine Ricker-
Punktquelle in der Mitte eines NxN-Gitters. Die an den Raendern zurueck-
laufende Welle ist der Fehler. Als Referenz dient ein VIEL groesseres Gitter
(gleiche Quelle/Sonde), dessen eigene Randreflexion die Sonde innerhalb der
Simulationsdauer noch nicht erreicht -> reines Einfallsfeld.

Reflexion R = max_t |E_test(t) - E_ref(t)| / max_t |E_ref(t)|  (an der Sonde).

Aufruf:  python tests/boundary_reflection.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.backend import xp, to_np                                   # noqa: E402
from common.physics import C0, EPS0, MU0                               # noqa: E402
from planar_beads.boundaries import make_boundary, mur1_coeff          # noqa: E402


def _solve_te(N, dx, nsteps, boundary, src_ij, probe_ij, fp, t0):
    dt = 0.99*dx/(C0*np.sqrt(2.0))
    Ez = xp.zeros((N, N), dtype=xp.float32)
    Hx = xp.zeros((N, N - 1), dtype=xp.float32)
    Hy = xp.zeros((N - 1, N), dtype=xp.float32)
    Ce_H_s = xp.float32(dt/(EPS0*dx))                # Freiraum eps_r=1 (Skalar)
    Ce_H = xp.full((N, N), dt/(EPS0*dx), dtype=xp.float32)   # Array-Form (CPML)
    Ch = xp.float32(dt/(MU0*dx))
    Ce_E = xp.ones((N, N), dtype=xp.float32)         # fuer die CPML-Signatur
    mur = mur1_coeff(dx, dt)
    _cpml = boundary.lower() == 'cpml'
    bnd = make_boundary(boundary, 'te', N, N, dx, dt, npml=10)
    sx, sy = src_ij; px, py = probe_ij
    trace = np.empty(nsteps, dtype=np.float64)
    for n in range(nsteps):
        if _cpml:
            bnd.update_H(Ez, Hx, Hy, Ch)
        else:
            Hx -= Ch*(Ez[:, 1:] - Ez[:, :-1])
            Hy += Ch*(Ez[1:, :] - Ez[:-1, :])
        if bnd is not None and not _cpml:
            bnd.capture(Ez)
        ex1, ex2 = Ez[1, :].copy(), Ez[-2, :].copy()
        ey1, ey2 = Ez[:, 1].copy(), Ez[:, -2].copy()
        if _cpml:
            bnd.update_E(Ez, Hx, Hy, Ce_E, Ce_H)
        else:
            Ez[1:-1, 1:-1] += Ce_H_s*((Hy[1:, 1:-1] - Hy[:-1, 1:-1])
                                      - (Hx[1:-1, 1:] - Hx[1:-1, :-1]))
        # Ricker-Punktquelle (soft)
        arg = (np.pi*fp*(n*dt - t0))**2
        Ez[sx, sy] += xp.float32((1.0 - 2.0*arg)*np.exp(-arg))
        if _cpml:
            bnd.terminate(Ez)
        elif bnd is not None:
            bnd.apply(Ez)
        else:
            Ez[0, :] = ex1 + mur*(Ez[1, :] - Ez[0, :]); Ez[-1, :] = ex2 + mur*(Ez[-2, :] - Ez[-1, :])
            Ez[:, 0] = ey1 + mur*(Ez[:, 1] - Ez[:, 0]); Ez[:, -1] = ey2 + mur*(Ez[:, -2] - Ez[:, -1])
        trace[n] = float(Ez[px, py])
    return trace


def main():
    dx = 20e-9
    N = 200
    nsteps = 420
    fp = C0/(20*dx)                                  # ~20 Zellen/Wellenlaenge
    dt = 0.99*dx/(C0*np.sqrt(2.0))
    t0 = 1.0/fp
    # Quelle Mitte, Sonde nahe am oberen Rand (misst dessen Reflexion)
    src = (N//2, N//2)
    probe_off = 70
    probe = (N//2, N//2 + probe_off)
    # Referenz: grosses Gitter, gleiche Geometrie relativ zur Quelle
    PAD = 260
    Nb = N + 2*PAD
    src_b = (Nb//2, Nb//2)
    probe_b = (Nb//2, Nb//2 + probe_off)

    print(f'Reflexions-Benchmark  dx={dx*1e9:g}nm  N={N}  steps={nsteps}  '
          f'PML=10 Zellen  (Backend: {"GPU" if xp.__name__=="cupy" else "CPU"})')
    ref = _solve_te(Nb, dx, nsteps, 'mur1', src_b, probe_b, fp, t0)   # Rand egal (zu weit)
    peak = np.max(np.abs(ref))
    print(f'{"Rand":<8} {"Reflexion":>12} {"dB":>9}')
    print('-'*32)
    results = {}
    for bnd in ('mur1', 'mur2', 'cpml'):
        tr = _solve_te(N, dx, nsteps, bnd, src, probe, fp, t0)
        R = np.max(np.abs(tr - ref))/peak
        results[bnd] = R
        print(f'{bnd:<8} {R:>12.3e} {20*np.log10(max(R,1e-12)):>8.1f}')
    ok = results['cpml'] < results['mur2'] < results['mur1']
    print('\nOrdnung cpml < mur2 < mur1: ' + ('OK' if ok else 'NICHT erfuellt'))
    return results


if __name__ == '__main__':
    main()
