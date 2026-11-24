"""Review F12: Bead-Signal gegen den Abstand Bead-Kern, in Schritten von 1/4 Zelle.

Mit harter Maske rastet der Abstand auf das Gitter ein -> das Signal springt in
Stufen (Plateaus ueber eine ganze Zelle). Mit der Subpixel-Mittelung (common/subpixel,
TE skalar / TM anisotrop) muss es STETIG und glatt (~exp(-2g/delta)) vom Abstand
abhaengen und gegen das feinere Gitter konvergieren.

Signal: Delta T = 1 - T_bead/T_ref (T = Poynting-Transmission ueber den ganzen
Querschnitt). Duenner Kern (0.5 um) -> starkes evaneszentes Feld, Polystyrol-Bead
d = 0.5 um in Traenenfluessigkeit (wie der im Review vorgeschlagene Benchmark).

Aufruf:  python tests/bead_subpixel_series.py [s|p] [dx_nm]
"""
import os
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import planar_beads.fdtd2d_core as f2d                       # noqa: E402

D_CORE, D_BEAD, L = 0.5, 0.5, 15.0


def run(pol, dx_nm, gap_um, subpixel, place=True):
    f2d.BEAD_SUBPIXEL = subpixel
    prev = os.getcwd(); tmp = tempfile.mkdtemp()
    try:
        os.chdir(tmp)
        r = f2d.run_beads('pmma', 'polystyrol', D_BEAD if place else 0.0, dx_nm=dx_nm,
                          save_frames=False, n_snapshots=2, wg_thickness_um=D_CORE,
                          vcsel_waist=0.3, method='full', place_bead=place, length_um=L,
                          polarization=pol, t_aqueous_um=4.0, t_mucin_um=0.5, t_lipid_um=0.0,
                          bead_y_um=-(D_BEAD/2 + gap_um), boundary='cpml')
    finally:
        os.chdir(prev)
    return float(r['transmission'])


def main(pol='s', dx_nm=20.0):
    gaps = np.arange(0, 13)*dx_nm/4000.0                # 0 .. 3 Zellen in 1/4-Zellen-Schritten
    T_ref = run(pol, dx_nm, 0.0, True, place=False)
    out = {}
    for mode in (False, True):
        out[mode] = np.array([1 - run(pol, dx_nm, g, mode)/T_ref for g in gaps])
    # Referenz: halbes dx, Subpixel (wenige Abstaende)
    T_ref_f = run(pol, dx_nm/2, 0.0, True, place=False)
    gf = gaps[::4]
    fine = np.array([1 - run(pol, dx_nm/2, g, True)/T_ref_f for g in gf])
    print(f'\nPol {pol}  dx {dx_nm:g} nm  (Referenz dx {dx_nm/2:g} nm)  Bead PS d={D_BEAD} um, '
          f'Kern {D_CORE} um')
    print(f'{"gap nm":>7}{"hart":>12}{"subpixel":>12}{"fein":>12}')
    for i, g in enumerate(gaps):
        fv = fine[list(gf).index(g)] if g in gf else None
        print(f'{g*1000:7.1f}{out[False][i]:12.4e}{out[True][i]:12.4e}'
              + (f'{fv:12.4e}' if fv is not None else ''))

    def roughness(y):
        """Glattheit: Abweichung von einer glatten Kurve (Fit ln y quadratisch in g)."""
        y = np.maximum(y, 1e-12)
        c = np.polyfit(gaps, np.log(y), 2)
        return float(np.max(np.abs(y/np.exp(np.polyval(c, gaps)) - 1)))
    rh, rs = roughness(out[False]), roughness(out[True])
    err_h = np.max(np.abs(out[False][::4]/fine - 1)); err_s = np.max(np.abs(out[True][::4]/fine - 1))
    print(f'\nStufigkeit (max. Abweichung von glatter Kurve): hart {100*rh:.1f} %   '
          f'subpixel {100*rs:.1f} %')
    print(f'Abweichung zum feinen Gitter (an ganzen Zellen): hart {100*err_h:.1f} %   '
          f'subpixel {100*err_s:.1f} %')
    f2d.BEAD_SUBPIXEL = True
    return rh, rs, err_h, err_s


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0] if a else 's', float(a[1]) if len(a) > 1 else 20.0)
