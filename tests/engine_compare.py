"""Vergleich native gegen Meep am selben 2D-Fall (Bead + Referenz).

Erwartet die Ergebnisdateien (A = --geometry planar --dim 2 --length-um 15 --resolution-nm 25):
  results/meep_planar2d_meep_polystyrol_d0.5_frames.npz      (--engine meep A --bead-diameter 0.5)
  results/meep_planar2d_meep_polystyrol_ohneBead_frames.npz  (--engine meep A --no-bead)
  results/beads_WG-pmma_Bead-polystyrol_d0.5_L850_full_frames.npz (+ _ref_full)  (native)
Vergleicht: Transmission (Fluss ueber den ganzen Querschnitt), Bead-Effekt dT,
n_eff aus der Phase im Kern, Streufeld |E_bead - E_ref| (Maximum, dB) und die
Feldprofile auf einem gemeinsamen Gitter.

Aufruf:  python tests/engine_compare.py
"""
import os
import pickle
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)          # Pickles enthalten common.physics.CorneaProfile
R = os.path.join(_ROOT, 'results')


def load(path):
    d = np.load(path)
    a = np.asarray(d['Ez_cw'], dtype=np.float64)
    n = a.shape[0]
    E = (2.0/n)*np.tensordot(np.exp(-2j*np.pi*np.arange(n)/n), a, axes=(0, 0))
    m = np.asarray(d['meta_cw'])[0]
    x0, x1, y0, y1 = (float(v) for v in m[:4])
    xs = np.linspace(x0, x1, E.shape[0]); ys = np.linspace(y0, y1, E.shape[1])
    T = float(np.asarray(d['transmission']).flat[0]) if 'transmission' in d.files else None
    return dict(E=E, xs=xs, ys=ys, T=T)


def native_T(stem):
    p = os.path.join(R, stem + '.pkl')
    try:
        return float(pickle.load(open(p, 'rb'))['transmission'])
    except Exception:
        return None


def resample(f, xs, ys):
    """Komplexes Feld bilinear auf (xs, ys)."""
    from scipy.interpolate import RegularGridInterpolator
    out = []
    for part in (f['E'].real, f['E'].imag):
        it = RegularGridInterpolator((f['xs'], f['ys']), part, bounds_error=False, fill_value=np.nan)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        out.append(it((X, Y)))
    return out[0] + 1j*out[1]


def neff(f, lam_um=0.85, t_wg=5.0):
    x = f['xs']; L = x[-1] - x[0]
    sel = (x > x[0] + 0.45*L) & (x < x[0] + 0.85*L)
    jc = (f['ys'] >= 0) & (f['ys'] <= t_wg)
    line = f['E'][sel][:, jc].sum(axis=1)
    ph = np.unwrap(np.angle(line))
    beta = abs(np.polyfit(x[sel], ph, 1)[0])
    return beta/(2*np.pi/lam_um)


def shifted(f, dx_um):
    """Feld mit um dx_um verschobener x-Achse (Meep-Quelle liegt hinter dem PML weiter
    rechts als die native hinter ihrer CPML -> gleiche Abstaende zur Quelle vergleichen)."""
    return dict(f, xs=f['xs'] - dx_um)


# Meep (full) nutzt die native x-Aufteilung (run_meep._native_x_layout): gleiche
# Quelle, gleiche Koordinaten -> keine Verschiebung noetig.
SHIFT = 0.0


def main():
    mb = shifted(load(os.path.join(R, 'meep_planar2d_meep_polystyrol_d0.5_frames.npz')), SHIFT)
    mr = shifted(load(os.path.join(R, 'meep_planar2d_meep_polystyrol_ohneBead_frames.npz')), SHIFT)
    stem_b = 'beads_WG-pmma_Bead-polystyrol_d0.5_L850_full'
    stem_r = 'beads_WG-pmma_Bead-polystyrol_d0_L850_ref_full'
    nb = load(os.path.join(R, stem_b + '_frames.npz'))
    nr = load(os.path.join(R, stem_r + '_frames.npz'))
    Tn_b, Tn_r = native_T(stem_b), native_T(stem_r)
    print('Transmission (Fluss, ganzer Querschnitt)')
    print(f'  native: Bead {Tn_b:.5f}  Ref {Tn_r:.5f}  dT = {Tn_r - Tn_b:+.2e}')
    print(f'  Meep  : Bead {mb["T"]:.5f}  Ref {mr["T"]:.5f}  dT = {mr["T"] - mb["T"]:+.2e}')
    print('n_eff aus der Phase im Kern (Referenzlauf)')
    print(f'  native {neff(nr):.5f}   Meep {neff(mr):.5f}')
    # gemeinsames Gitter (Ueberlappung, ohne Raender/Quelle)
    x0 = max(nb['xs'][0], mb['xs'][0]) + 2.0; x1 = min(nb['xs'][-1], mb['xs'][-1]) - 1.5
    y0 = max(nb['ys'][0], mb['ys'][0]) + 1.0; y1 = min(nb['ys'][-1], mb['ys'][-1]) - 1.0
    xs = np.linspace(x0, x1, 400); ys = np.linspace(y0, y1, 300)
    Snat = np.abs(resample(nb, xs, ys) - resample(nr, xs, ys))
    Smeep = np.abs(resample(mb, xs, ys) - resample(mr, xs, ys))
    Rn = np.nanmax(np.abs(resample(nr, xs, ys))); Rm = np.nanmax(np.abs(resample(mr, xs, ys)))
    print('Streufeld |E_bead - E_ref| (Maximum, relativ zum Referenzmaximum)')
    print(f'  native {20*np.log10(np.nanmax(Snat)/Rn):+.1f} dB   Meep {20*np.log10(np.nanmax(Smeep)/Rm):+.1f} dB')
    # Form des Streufelds: Korrelation der normierten Betraege
    a = (Snat/Rn).ravel(); b = (Smeep/Rm).ravel(); ok = np.isfinite(a) & np.isfinite(b)
    print(f'  Korrelation der Streufeld-Betraege: {np.corrcoef(a[ok], b[ok])[0, 1]:.3f}')
    # Referenzfeld: Betragsprofil im Querschnitt bei 0.7 L
    ix = np.argmin(abs(xs - (x0 + 0.7*(x1 - x0))))
    pn = np.abs(resample(nr, xs, ys))[ix]/Rn; pm = np.abs(resample(mr, xs, ys))[ix]/Rm
    print(f'Referenzfeld |E(y)| bei x={xs[ix]:.1f} um: mittlere Abweichung {np.nanmean(np.abs(pn - pm)):.3f} '
          f'(Maximum {np.nanmax(np.abs(pn - pm)):.3f}) relativ zum Maximum')


if __name__ == '__main__':
    main()
