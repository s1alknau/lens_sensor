"""Meep-Vektor-Stitch gegen Meep-full (gleicher 2D-Fall, mit Bead).

Erwartet (A = --engine meep --geometry planar --dim 2 --length-um 50 --resolution-nm 25
          --bead-diameter 0.5):
  results/meep_planar2d_meep_polystyrol_d0.5_frames.npz         (A)
  results/meep_planar2d_meep_polystyrol_d0.5_stitch_frames.npz  (A --method stitch --meep-handoff vector)
Vergleicht Transmission und die komplexe CW-Amplitude (Betrag und Phase) im Kern
und im Tear-Film sowie die Naht-Sprünge.

Aufruf:  python tests/meep_stitch_compare.py
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tests'))
from engine_compare import load, resample, R    # noqa: E402


def main():
    full = load(os.path.join(R, 'meep_planar2d_meep_polystyrol_d0.5_frames.npz'))
    st = load(os.path.join(R, 'meep_planar2d_meep_polystyrol_d0.5_stitch_frames.npz'))
    print(f'Transmission: full {full["T"]:.5f}   stitch {st["T"]:.5f}   '
          f'Abweichung {100*(st["T"] - full["T"])/full["T"]:+.2f} %')
    x0 = max(full['xs'][0], st['xs'][0]) + 2.0; x1 = min(full['xs'][-1], st['xs'][-1]) - 1.5
    xs = np.linspace(x0, x1, 1200); ys = np.linspace(-3.0, 6.0, 361)
    F = resample(full, xs, ys); S = resample(st, xs, ys)
    M = np.nanmax(np.abs(F))
    # globale Phase (Normierung beliebig) per Ueberlapp angleichen
    ok = np.isfinite(F) & np.isfinite(S)
    c = np.vdot(S[ok], F[ok])/np.vdot(S[ok], S[ok]).real
    S = S*c
    print(f'Kalibrierfaktor stitch->full: |c| = {abs(c):.4f}, Phase {np.angle(c):+.3f} rad')
    D = np.abs(S - F)/M
    for lab, (ya, yb) in (('Kern (0..5 um)', (0, 5)), ('Tear (-3..0 um)', (-3, 0)), ('Luft (5..6 um)', (5, 6))):
        m = (ys >= ya) & (ys <= yb)
        print(f'  {lab:<16} |E_stitch - E_full| rms {np.sqrt(np.nanmean(D[:, m]**2)):.4f}  '
              f'max {np.nanmax(D[:, m]):.4f}  (relativ zum Maximum)')
    # Verlauf entlang x (Kernmittel) -> Naht-Spruenge sichtbar
    jc = (ys >= 0) & (ys <= 5)
    pf = np.nanmean(np.abs(F[:, jc]), axis=1); ps = np.nanmean(np.abs(S[:, jc]), axis=1)
    rel = (ps - pf)/np.nanmax(pf)
    k = np.argsort(-np.abs(rel))[:3]
    print('  groesste Betragsabweichung im Kern entlang x: ' +
          ', '.join(f'x={xs[i]:.1f} um {100*rel[i]:+.2f} %' for i in sorted(k)))


if __name__ == '__main__':
    main()
