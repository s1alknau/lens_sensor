"""Fast checks of the BPM <-> FDTD coupling, ~2 min.

1. BPM -> FDTD hand-off (run_hybrid.py, fdtd2d_core.HANDOFF):

Small planar waveguide, tilted beam: BPM to x = 20 um, then 25 um FDTD (25 nm grid).
The FDTD field must continue the BPM field: same core profile (corr > 0.99, incl. the beam tilt,
i.e. the phase convention) at the start and at the end of the FDTD domain, and the
one-sided source must not radiate backwards.

2. FDTD -> BPM start (run_bpm.py --bpm-start-fdtd-um): the BPM started from the field of a
short FDTD coupling run continues like the BPM started from the analytic facet model
(same profile downstream, same coupled power within 0.5 %).

    python -m pytest tests/test_hybrid.py -q      (or: python tests/test_hybrid.py)
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _profiles(d, x):
    xf, yf, If = d['fdtd_x_um'], d['fdtd_y_um'], d['fdtd_I'].astype(float)
    xb, yb, Ib = d['bpm_x_um'], d['bpm_y_um'], d['bpm_I'].astype(float)
    i, k = np.argmin(abs(xf - x)), np.argmin(abs(xb - x))
    m = (yf > 0) & (yf < float(d['t_wg_um']))
    return If[i, m], np.interp(yf, yb, Ib[k])[m]


def test_handoff_continues_bpm_field():
    import run_hybrid
    out = os.path.join(tempfile.mkdtemp(), 'h.npz')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_hybrid.main(['--geometry', 'planar', '--no-bead', '--vcsel-tilt', '10',
                         '--resolution-nm', '25', '--handoff-x-um', '20',
                         '--fdtd-length-um', '25', '--no-save', '--hybrid-out', out])
    log = buf.getvalue()
    d = np.load(out)
    for x in (21.0, 43.0):
        f, b = _profiles(d, x)
        assert np.corrcoef(f, b)[0, 1] > 0.99, (x, np.corrcoef(f, b)[0, 1])
    # power in the core stays constant after the source (no transient / wrong kx)
    pc = d['fdtd_P_core'][d['fdtd_x_full_um'] > 21.0]
    assert pc.std()/pc.mean() < 0.02, pc.std()/pc.mean()
    # backward emission of the Huygens source ("facet reflection" in the log) < 0.5 %
    line = [s for s in log.splitlines() if 'facet reflection=' in s][0]
    refl = float(line.split('facet reflection=')[1].split('%')[0])
    assert abs(refl) < 0.5, line


def test_bpm_start_from_fdtd():
    import run_bpm
    d = tempfile.mkdtemp()
    base = ['--geometry', 'planar', '--wg-thickness-um', '30', '--no-bead', '--vcsel-tilt', '10',
            '--resolution-nm', '25',
            '--length-um', '60', '--bpm-rec-um', '1']
    with contextlib.redirect_stdout(io.StringIO()):
        run_bpm.main(base + ['--bpm-start-fdtd-um', '5', '--bpm-out', os.path.join(d, 'f.npz')])
        run_bpm.main(base + ['--bpm-out', os.path.join(d, 'a.npz')])
    f, a = np.load(os.path.join(d, 'f.npz')), np.load(os.path.join(d, 'a.npz'))
    assert abs(float(f['coupling_eta'])/float(a['coupling_eta']) - 1) < 0.005
    y = f['y_um']; m = (y > 0) & (y < float(f['t_wg_um']))
    for x in (10.0, 60.0):
        i, k = np.argmin(abs(f['x_um'] - x)), np.argmin(abs(a['x_um'] - x))
        cc = np.corrcoef(f['I'][i][m], a['I'][k][m])[0, 1]
        assert cc > 0.99, (x, cc)


if __name__ == '__main__':
    test_handoff_continues_bpm_field(); print('OK test_handoff_continues_bpm_field')
    test_bpm_start_from_fdtd(); print('OK test_bpm_start_from_fdtd')
