"""Native Moden-Propagation fuer den Kontaktlinsen-Sensor (aufloesungsrobust).

Analog zum Meep Mode-Handoff, aber nativ und OHNE FDTD ueber die Laenge:
  1) Transversale Eigenmoden des Schicht-Slabs (Luft | PMMA-Kern | Lipid |
     Aqueous | Mucin | Cornea) via tridiagonalem 1D-Eigenloeser -> exakte beta_m
     (n_eff) + Profile phi_m(y). Fein in y (loest das evaneszente Feld auf),
     ohne Gitter entlang der Propagation -> KEINE akkumulierende Dispersion.
  2) VCSEL-Einkopplung per Overlap-Integral -> Moden-Amplituden a_m.
  3) Analytische Propagation ueber den Bogen: a_m -> a_m * exp(i beta_m L).
  4) Sensor: Feld an den Detektor-y-Positionen (D1 Tear-seitig, D2 Luft-seitig)
     -> R_tear = |E(D1)|^2 / |E(D2)|^2.

Das brute-force-Stitching (sliding_window_fdtd.py / full_domain_fdtd.py) bleibt
unveraendert erhalten - dies ist eine ZUSAETZLICHE, schnelle & genaue Methode.

Ausfuehren:
    python mode_propagation.py                 # Testrechnung: R_tear je Szenario
    python mode_propagation.py --scenario DED --L-mm 12 --offset-um 0
"""
import argparse
import numpy as np

try:
    from scipy.linalg import eigh_tridiagonal
except Exception as e:                       # pragma: no cover
    raise SystemExit('scipy noetig (scipy.linalg.eigh_tridiagonal): ' + str(e))

from config import (N_PMMA, N_AIR, N_LIPID, N_MUCIN, N_CORNEA, T_LENS, LAM,
                    D_LENS, D1_S_CENTER, SCENARIOS)


def _scenarios_dict():
    return {s[0]: s[1:] for s in SCENARIOS}   # name -> (t_lip, t_aq, t_mu, n_aq) in um,um,um,-


def build_index_profile(t_lip_um, t_aq_um, t_mu_um, n_aq, dy_um,
                        core_um=None, air_um=8.0, cornea_um=15.0):
    """n(y) des Schichtstapels. y=0 = Kern-UNTERKANTE; Kern nach oben [0, core].
    Reihenfolge (wie im Solver): Cornea | Mucin | Aqueous | Lipid | KERN | Luft."""
    core_um = core_um if core_um is not None else T_LENS*1e6
    below = t_lip_um + t_aq_um + t_mu_um + cornea_um   # unter dem Kern
    y_lo = -below
    y_hi = core_um + air_um
    n_pts = int(round((y_hi - y_lo)/dy_um)) + 1
    y = y_lo + np.arange(n_pts)*dy_um
    n = np.full(n_pts, N_AIR)                          # Default Luft (oben)
    n[y < core_um] = N_PMMA                            # Kern (bis Oberkante)
    n[y < 0] = N_LIPID                                 # Lipid direkt unter dem Kern
    n[y < -t_lip_um] = n_aq                            # Aqueous
    n[y < -(t_lip_um + t_aq_um)] = N_MUCIN            # Mucin
    n[y < -(t_lip_um + t_aq_um + t_mu_um)] = N_CORNEA  # Cornea
    n[y >= core_um] = N_AIR                            # Luft ueber dem Kern
    return y, n


def solve_te_modes(y, n, lam_um):
    """Gefuehrte TE-Moden: (d2/dy2 + k0^2 n^2) Ez = beta^2 Ez, tridiagonal.
    Rueckgabe (n_eff[absteigend], profile[:, m] auf |phi|^2-dy=1 normiert)."""
    dy = y[1] - y[0]
    k0 = 2*np.pi/lam_um
    inv = 1.0/dy**2
    diag = -2.0*inv + (k0*n)**2
    off = np.full(len(y) - 1, inv)
    n_core = n.max()
    # gefuehrt: beta^2 zwischen (k0 * groesster AUSSEN-Cladding-Index) und (k0 n_core)
    n_clad_out = max(n[0], n[-1])                      # Cornea (unten) / Luft (oben)
    lo = (k0*n_clad_out)**2 + 1e-6
    hi = (k0*n_core)**2 - 1e-6
    w, v = eigh_tridiagonal(diag, off, select='v', select_range=(lo, hi))
    order = np.argsort(w)[::-1]                        # hoechstes beta^2 = Grundmode zuerst
    w = w[order]; v = v[:, order]
    neff = np.sqrt(np.clip(w, 0, None))/k0
    v = v/np.sqrt(np.sum(v**2, axis=0)*dy)            # Norm: int |phi|^2 dy = 1
    return neff, v


def couple(y, profiles, y0_um, waist_um):
    """VCSEL-Gauss (Taille waist, Zentrum y0) -> Moden-Amplituden a_m = <phi_m|g>."""
    dy = y[1] - y[0]
    g = np.exp(-((y - y0_um)/waist_um)**2)
    g = g/np.sqrt(np.sum(g**2)*dy)
    a = (profiles*g[:, None]).sum(axis=0)*dy
    return a


def sensor_R(y, profiles, a, betas_um, L_um, d1_y_um, d2_y_um):
    """Feld nach Propagation um L, R_tear = |E(D1)|^2/|E(D2)|^2 (D1 Tear-, D2 Luft-seitig)."""
    E = (profiles*(a*np.exp(1j*betas_um*L_um))[None, :]).sum(axis=1)
    i1 = int(np.argmin(np.abs(y - d1_y_um)))
    i2 = int(np.argmin(np.abs(y - d2_y_um)))
    I1 = abs(E[i1])**2; I2 = abs(E[i2])**2
    return I1/max(I2, 1e-30), I1, I2, E


def run_scenario(name, t_lip, t_aq, t_mu, n_aq, lam_um, dy_um, L_um,
                 offset_um, waist_um, verbose=False):
    core = T_LENS*1e6
    y, n = build_index_profile(t_lip, t_aq, t_mu, n_aq, dy_um)
    neff, prof = solve_te_modes(y, n, lam_um)
    k0 = 2*np.pi/lam_um
    betas = neff*k0
    # Einkopplung: VCSEL in Kernmitte + Offset
    a = couple(y, prof, core/2 + offset_um, waist_um)
    # Detektoren: D1 Tear-seitig (25 um ueber Unterkante), D2 Luft-seitig (25 um unter Oberkante)
    d1_y = 25.0
    d2_y = core - 25.0
    R, I1, I2, _ = sensor_R(y, prof, a, betas, L_um, d1_y, d2_y)
    eta = float(np.sum(np.abs(a)**2))                 # in gefuehrte Moden gekoppelte Leistung
    if verbose:
        print(f'  Moden: {len(neff)}  n_eff [{neff.min():.4f}..{neff.max():.4f}]  '
              f'Kopplung in gefuehrt eta={eta:.3f}')
    return dict(name=name, n_modes=len(neff), neff_max=float(neff.max()),
                neff_min=float(neff.min()), eta=eta, R_tear=float(R),
                I1=float(I1), I2=float(I2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', default=None, help='nur dieses Szenario (sonst alle)')
    ap.add_argument('--lambda-nm', type=float, default=LAM*1e9)
    ap.add_argument('--dy-nm', type=float, default=10.0, help='transversale Aufloesung (nm)')
    ap.add_argument('--L-mm', type=float, default=None,
                    help='Propagationslaenge Rand->Detektor in mm (Default aus Geometrie)')
    ap.add_argument('--offset-um', type=float, default=0.0, help='VCSEL-y-Offset (0=Kernmitte)')
    ap.add_argument('--waist-um', type=float, default=2.0, help='VCSEL-Taille (um)')
    args = ap.parse_args()

    lam_um = args.lambda_nm/1000.0
    dy_um = args.dy_nm/1000.0
    # Bogenlaenge Rand (s=0) -> Detektor D1 (s=D1_S_CENTER): L = D_LENS/2 - x_d1, x_d1<0
    x_d1_mm = (D1_S_CENTER - D_LENS/2)*1e3
    L_um = (args.L_mm*1000.0) if args.L_mm is not None else (D_LENS/2*1e6 - x_d1_mm*1e3)
    print(f'Moden-Propagation  lam={args.lambda_nm:g}nm  dy={args.dy_nm:g}nm  '
          f'L(Rand->D1)={L_um/1000:.2f}mm  Kern={T_LENS*1e6:g}um  waist={args.waist_um:g}um '
          f'offset={args.offset_um:g}um')
    print('  (brute-force-Stitching bleibt erhalten: sliding_window_fdtd.py / full_domain_fdtd.py)')

    scen = _scenarios_dict()
    names = [args.scenario] if args.scenario else list(scen.keys())
    print(f'\n{"Szenario":<15} {"Moden":>6} {"n_eff_max":>10} {"eta_kopp":>9} {"R_tear":>10}')
    print('-'*54)
    base = None
    for nm in names:
        t_lip, t_aq, t_mu, n_aq = scen[nm]
        r = run_scenario(nm, t_lip, t_aq, t_mu, n_aq, lam_um, dy_um, L_um,
                         args.offset_um, args.waist_um)
        if base is None:
            base = r['R_tear']
        dR = 100*(r['R_tear'] - base)/max(base, 1e-30)
        print(f'{nm:<15} {r["n_modes"]:>6} {r["neff_max"]:>10.4f} {r["eta"]:>9.3f} '
              f'{r["R_tear"]:>10.4e}  ({dR:+.1f}% vs {names[0]})')


if __name__ == '__main__':
    main()
