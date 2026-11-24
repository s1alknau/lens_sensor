"""CPML alt gegen neu (2D und 3D): Reflexion bei senkrechtem, schraegem und streifendem
Einfall. Alt: kappa_max = 7 (2D) bzw. 1 (3D), a_max = 0.05 S/m (praktisch kein CFS).
Neu: kappa_max = 7, a_max = 2*pi*eps0*f0/10 (CFS wirksam).

Methode wie tests/boundary_reflection*.py: Ricker-Punktquelle, Sonde; Referenz = gleiche
Anordnung in einem viel groesseren Gitter (Rand dort zu weit weg). R = max|E - E_ref| /
max|E_ref| an der Sonde.

Aufruf:  python tests/cpml_compare.py [2d|3d|all]
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tests'))
import planar_beads.boundaries as b2                        # noqa: E402
import planar_3d.boundaries3d as b3                         # noqa: E402
import boundary_reflection as br2                           # noqa: E402
import boundary_reflection_3d as br3                        # noqa: E402
from common.physics import C0                               # noqa: E402

_orig2 = b2.cfs_a_max
_orig3 = b3._cpml_bc_1d


def _set(old):
    if old:
        b2.cfs_a_max = lambda lam_m=None: 0.05
        b3._cpml_bc_1d = lambda *a, **k: _orig3(*a, **{**k, 'kappa_max': 1.0, 'a_max': 0.05})
    else:
        b2.cfs_a_max = _orig2
        b3._cpml_bc_1d = _orig3


def test_2d():
    dx = 20e-9; N = 200; nsteps = 520
    fp = C0/(20*dx); t0 = 1.0/fp
    PAD = 300; Nb = N + 2*PAD
    cases = {
        'senkrecht (Sonde ueber der Quelle)': ((N//2, N//2), (N//2, N//2 + 70)),
        'schraeg 45 Grad':                    ((N//2, N//2), (N//2 + 50, N//2 + 50)),
        'streifend (Quelle nahe Rand)':       ((N//2 - 60, N - 22), (N//2 + 60, N - 22)),
    }
    print('\n2D TE, dx 20 nm, PML 10 Zellen')
    print(f'{"Fall":<38}{"alt":>12}{"neu":>12}')
    for name, (src, probe) in cases.items():
        off = PAD
        ref = br2._solve_te(Nb, dx, nsteps, 'mur1', (src[0] + off, src[1] + off),
                            (probe[0] + off, probe[1] + off), fp, t0)
        pk = np.max(np.abs(ref)); r = {}
        for old in (True, False):
            _set(old)
            tr = br2._solve_te(N, dx, nsteps, 'cpml', src, probe, fp, t0)
            r[old] = np.max(np.abs(tr - ref))/pk
        _set(False)
        print(f'{name:<38}{r[True]:>12.3e}{r[False]:>12.3e}')


def test_3d():
    dx = 20e-9; N = 70; nsteps = 320
    fp = C0/(20*dx); t0 = 1.0/fp
    PAD = 55; Nb = N + 2*PAD
    c = N//2
    cases = {
        'senkrecht': ((c, c, c), (c, c, c + 24)),
        'schraeg':   ((c, c, c), (c + 17, c + 17, c + 17)),
        'streifend': ((c - 18, c, N - 14), (c + 18, c, N - 14)),
    }
    print('\n3D, dx 20 nm, PML 10 Zellen')
    print(f'{"Fall":<38}{"alt":>12}{"neu":>12}')
    for name, (src, probe) in cases.items():
        o = PAD
        ref = br3._solve3d(Nb, dx, nsteps, 'sponge', tuple(v + o for v in src),
                           tuple(v + o for v in probe), fp, t0, n_sp=10)
        pk = np.max(np.abs(ref)); r = {}
        for old in (True, False):
            _set(old)
            tr = br3._solve3d(N, dx, nsteps, 'cpml', src, probe, fp, t0)
            r[old] = np.max(np.abs(tr - ref))/pk
        _set(False)
        print(f'{name:<38}{r[True]:>12.3e}{r[False]:>12.3e}')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('2d', 'all'):
        test_2d()
    if what in ('3d', 'all'):
        test_3d()
