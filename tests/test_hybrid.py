"""Checks of the BPM <-> FDTD coupling, ~7 min.

1. BPM -> FDTD hand-off (run_hybrid.py, fdtd2d_core.HANDOFF):

Small planar waveguide, tilted beam: BPM to x = 20 um, then 25 um FDTD (25 nm grid).
The FDTD field must continue the BPM field: same core profile (corr > 0.99, incl. the beam tilt,
i.e. the phase convention) at the start and at the end of the FDTD domain, and the
one-sided source must not radiate backwards.

2. FDTD -> BPM start (run_bpm.py --bpm-start-fdtd-um): the BPM started from the field of a
short FDTD coupling run continues like the BPM started from the analytic facet model
(same profile downstream, same coupled power within 0.5 %).
3. FDTD -> BPM return (run_hybrid.py --hybrid-end-um): without a sensor the restarted
   BPM continues like the BPM alone (power within 1 %, same core profile).
4. Sensor in the FDTD part: the bead signal (scattered field E_bead - E_ref and the core
   power change behind the bead) of the hybrid equals that of a pure FDTD run from the
   facet (20 um planar core, beam hits the tear film at the bead; ~4 min).

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


def test_return_to_bpm():
    import run_hybrid
    out = os.path.join(tempfile.mkdtemp(), 'r.npz')
    with contextlib.redirect_stdout(io.StringIO()):
        run_hybrid.main(['--geometry', 'planar', '--wg-thickness-um', '30', '--no-bead',
                         '--vcsel-tilt', '20', '--resolution-nm', '25', '--handoff-x-um', '40',
                         '--fdtd-length-um', '20', '--hybrid-end-um', '200', '--no-save',
                         '--hybrid-out', out])
    d = np.load(out)
    assert abs(d['ret_P_total'][-1]/d['bpm_P_total'][-1] - 1) < 0.01
    y = d['bpm_y_um']; m = (y > 0) & (y < float(d['t_wg_um']))
    i = np.argmin(abs(d['bpm_x_um'] - 200)); k = np.argmin(abs(d['ret_x_um'] - 200))
    cc = np.corrcoef(d['bpm_I'][i][m], d['ret_I'][k][m])[0, 1]
    assert cc > 0.99, cc


def _pure_fdtd_cw(argv, bead_x_lens):
    import run_simulation as rs
    import fdtd2d_core as d2
    a = rs.build_parser().parse_args(argv)
    with contextlib.redirect_stdout(io.StringIO()):
        layers = rs.resolve_args(a)
    dx = float(a.resolution_nm)*1e-3
    a.bead_x = None if bead_x_lens is None else bead_x_lens + 12*dx + 1.0   # facet position
    d2.RETURN_CW = True
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = rs._run_2d(a, layers)
    finally:
        d2.RETURN_CW = False
    E = r['E_cw']
    return np.arange(E.shape[0])*dx - r['x_facet_global_um'], E


def test_bead_signal_matches_pure_fdtd():
    import run_hybrid
    XB, XH, LF, LP = 40.0, 25.0, 25.0, 55.0
    base = ['--geometry', 'planar', '--wg-thickness-um', '20', '--vcsel-tilt', '20',
            '--resolution-nm', '40', '--no-save']
    d = tempfile.mkdtemp(); F = {}
    for bead in (True, False):
        bb = ['--bead-x', str(XB)] if bead else ['--no-bead']
        F['p', bead] = _pure_fdtd_cw(base + ['--length-um', str(LP), '--method', 'full']
                                     + ([] if bead else ['--no-bead']), XB if bead else None)
        out = os.path.join(d, f'h{bead}.npz')
        with contextlib.redirect_stdout(io.StringIO()):
            run_hybrid.main(base + ['--handoff-x-um', str(XH), '--fdtd-length-um', str(LF),
                                    '--fdtd-y-top-um', 'none', '--hybrid-out', out] + bb)
        h = np.load(out)
        F['h', bead] = (h['fdtd_x_um'], h['fdtd_E'].astype(complex))
    xs = np.arange(28.0, 48.0, 0.2)
    G = {k: v[1][[np.argmin(abs(v[0] - q)) for q in xs]] for k, v in F.items()}
    ny = min(G['p', True].shape[1], G['h', True].shape[1])        # same y origin and step
    G = {k: v[:, :ny] for k, v in G.items()}
    y = -10.0 + np.arange(ny)*0.04
    core = (y > 0) & (y < 20.0)
    i0 = np.argmin(abs(xs - 29.0))                                  # scale/phase before the bead
    sc = np.vdot(G['h', False][i0, core], G['p', False][i0, core])/np.vdot(G['h', False][i0, core],
                                                                         G['h', False][i0, core])
    dP = np.abs(G['p', True] - G['p', False])**2
    dH = np.abs(sc*(G['h', True] - G['h', False]))**2
    w = (xs >= 36) & (xs <= 47)
    assert abs(dH[w].sum()/dP[w].sum() - 1) < 0.1, dH[w].sum()/dP[w].sum()
    assert np.corrcoef(dP[w].ravel(), dH[w].ravel())[0, 1] > 0.97
    i = np.argmin(abs(xs - 46.0))
    cP = (np.abs(G['p', True][i, core])**2).sum()/(np.abs(G['p', False][i, core])**2).sum() - 1
    cH = (np.abs(G['h', True][i, core])**2).sum()/(np.abs(G['h', False][i, core])**2).sum() - 1
    assert abs(cH/cP - 1) < 0.1, (cP, cH)


if __name__ == '__main__':
    test_handoff_continues_bpm_field(); print('OK test_handoff_continues_bpm_field')
    test_bpm_start_from_fdtd(); print('OK test_bpm_start_from_fdtd')
    test_return_to_bpm(); print('OK test_return_to_bpm')
    test_bead_signal_matches_pure_fdtd(); print('OK test_bead_signal_matches_pure_fdtd')
