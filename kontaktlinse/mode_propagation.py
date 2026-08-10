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
                        core_um=None, air_um=8.0, cornea_um=15.0, r_bend_um=None):
    """n(y) des Schichtstapels. y=0 = Kern-UNTERKANTE; Kern nach oben [0, core].
    Reihenfolge (wie im Solver): Cornea | Mucin | Aqueous | Lipid | KERN | Luft.

    r_bend_um: Biegeradius (um). None/inf = gerader Guide. Sonst ECHTE Kruemmung
    via konformer Abbildung des gebogenen Wellenleiters -> aequivalenter Index
    n_eq(u) = n(u)*(1 + u/R), u = y - Kernmitte (u>0 = aussen/Luftseite, groesserer
    Radius). Rigoros fuer einen Kreisbogen; erfasst Modenversatz nach aussen +
    Biege-/Strahlungsverlust (via Kaustik). -> Bent-Mode-Solver."""
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
    if r_bend_um is not None and np.isfinite(r_bend_um) and r_bend_um > 0:
        u = y - core_um/2.0                            # radialer Versatz von der Kernmitte
        n = n*(1.0 + u/r_bend_um)                      # konforme Abbildung (echte Kruemmung)
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


def project_field(y, profiles, y_src, E_src):
    """Projiziert ein (FDTD-)Feld E_src(y_src) auf die Moden -> Amplituden a_m
    (a_m = <phi_m|E>, phi reell). y_src/E_src werden auf das Modengitter interpoliert."""
    E = (np.interp(y, y_src, np.real(E_src), left=0.0, right=0.0)
         + 1j*np.interp(y, y_src, np.imag(E_src), left=0.0, right=0.0))
    dy = y[1] - y[0]
    return (profiles*E[:, None]).sum(axis=0)*dy


def fdtd_coupling_field(lam_nm, offset_um, waist_um, dx_nm=50.0, length_um=30.0):
    """Lokales FDTD am Rand: VCSEL koppelt in den 250-um-Kern; Rueckgabe des
    komplexen CW-Feldes an einem Querschnitt NACH der Einkoppelzone (y so, dass
    Kern=[0,core]). Das ersetzt den analytischen Gauss-Overlap durch die reale
    Einkopplungs-Physik (Fresnel, Nahfeld) -> realistische Moden-Verteilung."""
    import os
    import sys
    import tempfile
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(root, 'planar_beads')
    if p not in sys.path:
        sys.path.insert(0, p)
    import fdtd2d_core as f2d
    core = T_LENS*1e6
    prev = os.getcwd(); tmp = tempfile.mkdtemp()
    try:
        os.chdir(tmp)
        r = f2d.run_beads(wg_mat='pmma', bead_mat='polystyrol', bead_d_um=0.0,
                          dx_nm=dx_nm, save_frames=True, n_snapshots=8,
                          wg_thickness_um=core, lambda_nm=lam_nm, vcsel_waist=waist_um,
                          vcsel_offset=offset_um, method='full', place_bead=False,
                          length_um=length_um, polarization='s',
                          t_aqueous_um=8.0, t_mucin_um=2.0)
        cwm = r['cw_frames_memmap']; cwf = r['cw_frames'] or []
        arr = np.asarray(np.load(cwm, mmap_mode='r')[:len(cwf)])
    finally:
        os.chdir(prev)
    nfr, Nx, Ny = arr.shape
    E = arr[0].astype(np.float64) - 1j*arr[nfr//4 or 1].astype(np.float64)
    tear = f2d.TEAR_BUFFER*1e6
    y_src = -tear + np.arange(Ny)*(dx_nm/1000.0)
    return y_src, E[int(0.6*Nx), :]        # Querschnitt in der eingeschwungenen Zone


def validate_solver(lam_um, dy_um=0.005):
    """Eigenloeser gegen analytische asymm. Slab-Formel (crossval) an einem DUENNEN
    (few-mode) Slab, wo n_eff eindeutig unter n_core liegt."""
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for pp in (os.path.join(root, 'tests'), root):
        if pp not in sys.path:
            sys.path.insert(0, pp)
    import crossval as cv
    d = 5.0
    y, n = build_index_profile(0.0, 8.0, 2.0, 1.336, dy_um, core_um=d,
                               air_um=6.0, cornea_um=6.0)
    neff, _ = solve_te_modes(y, n, lam_um)
    n_num = float(neff.max())
    n_ana = cv.slab_neff_analytic(N_PMMA, N_AIR, 1.336, d, lam_um*1000, 's', 0)
    dd = abs(n_num - n_ana)
    print(f'[Validierung] duenner Slab d={d:g}um: Eigenloeser n_eff={n_num:.5f}  '
          f'analytisch={n_ana:.5f}  Delta={dd:.2e}  -> {"OK" if dd < 2e-3 else "PRUEFEN"}')
    return dd


def sensor_R(y, profiles, a, betas_um, L_um, d1_y_um, d2_y_um):
    """Feld nach Propagation um L, R_tear = |E(D1)|^2/|E(D2)|^2 (D1 Tear-, D2 Luft-seitig)."""
    E = (profiles*(a*np.exp(1j*betas_um*L_um))[None, :]).sum(axis=1)
    i1 = int(np.argmin(np.abs(y - d1_y_um)))
    i2 = int(np.argmin(np.abs(y - d2_y_um)))
    I1 = abs(E[i1])**2; I2 = abs(E[i2])**2
    return I1/max(I2, 1e-30), I1, I2, E


def run_scenario(name, t_lip, t_aq, t_mu, n_aq, lam_um, dy_um, L_um,
                 offset_um, waist_um, alpha_tear_permm=0.0, coupling=None,
                 r_bend_um=None, verbose=False):
    """ATR-Sensor + (optional) echte Kruemmung. Traenenfilm-Absorption daempft das
    totalreflektierte Licht (alpha_m = alpha_tear*Gamma_m -> ueberlebendes TIR-Licht).
    r_bend_um: Biegeradius -> Bent-Mode via konformer Abbildung; zusaetzlich wird der
    Biege-/Strahlungsverlust ueber die Strahlungs-Kaustik abgeschaetzt (-> Durchsatz)."""
    core = T_LENS*1e6
    y, n = build_index_profile(t_lip, t_aq, t_mu, n_aq, dy_um, r_bend_um=r_bend_um)
    dy = y[1] - y[0]
    neff, prof = solve_te_modes(y, n, lam_um)
    k0 = 2*np.pi/lam_um
    betas = neff*k0
    if coupling is not None:                              # reale FDTD-Einkopplung -> Projektion
        a = project_field(y, prof, coupling[0], coupling[1])
        a = a/np.sqrt(max(np.sum(np.abs(a)**2), 1e-30))  # auf Einheitsleistung normiert
    else:                                                 # analytischer Gauss-Overlap
        a = couple(y, prof, core/2 + offset_um, waist_um)
    P = np.abs(a)**2
    eta = float(P.sum())
    # --- ATR: modaler Verlust durch Absorption im Aqueous (Traenenfilm-Analyt) ---
    aq_mask = (y <= -t_lip) & (y > -(t_lip + t_aq))    # Aqueous-Schicht unter dem Kern
    Gamma = (prof[aq_mask, :]**2).sum(axis=0)*dy        # Leistungs-Confinement je Mode
    alpha_um = (alpha_tear_permm/1000.0)*Gamma          # modaler Verlust (1/um)
    P_surv = float(np.sum(P*np.exp(-alpha_um*L_um)))     # ueberlebendes TIR-Licht -> D3
    ATR_dB = -10*np.log10(max(P_surv/max(eta, 1e-30), 1e-30))
    Gamma_eff = float(np.sum(P*Gamma)/max(eta, 1e-30))   # gewichtetes Aqueous-Confinement
    # --- DIREKTE Bestimmung des evaneszenten Feldes im Traenenfilm ---
    R, _, _, E = sensor_R(y, prof, a, betas, L_um, 25.0, core - 25.0)
    Ie = np.abs(E)**2
    i_surf = int(np.argmin(np.abs(y)))                   # Kern-Unterkante y=0
    I_surf = float(Ie[i_surf])                            # evan. Intensitaet an der Oberflaeche
    I_evan_aq = float(Ie[aq_mask].sum()*dy)              # integriertes evan. Feld im Aqueous
    I_core = float(Ie[(y >= 0) & (y <= core)].sum()*dy)   # Kern-Leistung (Referenz)
    evan_ratio = I_evan_aq/max(I_core, 1e-30)            # direkter evan. Anteil (Tear/Kern)
    # Eindringtiefe delta aus |E| direkt unter der Oberflaeche (im Aqueous)
    below = (y < -t_lip) & (y > -(t_lip + min(t_aq, 1.0)))
    if below.sum() >= 4:
        g = np.polyfit(-y[below], np.log(np.maximum(np.abs(E[below]), 1e-30)), 1)[0]
        delta_um = 1.0/abs(g) if g else float('nan')
    else:
        delta_um = float('nan')
    # --- Biege-/Strahlungsverlust (nur gekruemmt): ueber die Strahlungs-Kaustik ---
    neff_max = float(neff.max())
    bend_loss_perum = 0.0; u_caustic_um = float('inf')
    if r_bend_um is not None and np.isfinite(r_bend_um) and r_bend_um > 0:
        n_out = N_AIR                                    # Strahlung nach aussen (Luftseite)
        u_caustic_um = r_bend_um*(neff_max/n_out - 1.0)  # n_eq(u)=n_eff -> Kaustik
        gam = k0*np.sqrt(max(neff_max**2 - n_out**2, 1e-9))   # evan. Abfall in Luft (1/um)
        log_bend = -2.0*gam*max(u_caustic_um, 0.0)       # ln(alpha_bend/gam), Underflow-sicher
        bend_loss_perum = float(gam*np.exp(log_bend)) if log_bend > -700 else 0.0
    throughput_bend = float(np.exp(-bend_loss_perum*L_um))   # ueberlebt Biegung ueber L
    if verbose:
        print(f'  Moden {len(neff)}  eta={eta:.3f}  Gamma_aq={Gamma_eff:.3e}  '
              f'evan_ratio={evan_ratio:.3e}  delta={delta_um:.3f}um  '
              f'bend_loss={bend_loss_perum:.2e}/um  T_bend={throughput_bend:.3f}')
    return dict(name=name, n_modes=len(neff), neff_max=neff_max,
                eta=eta, R_tear=float(R), Gamma_aq=Gamma_eff, ATR_dB=float(ATR_dB),
                P_surv=P_surv, evan_ratio=evan_ratio, I_surf=I_surf,
                I_evan_aq=I_evan_aq, delta_um=float(delta_um),
                bend_loss_perum=bend_loss_perum, u_caustic_um=float(u_caustic_um),
                throughput_bend=throughput_bend)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', default=None, help='nur dieses Szenario (sonst alle)')
    ap.add_argument('--r-bend-mm', type=float, default=0.0,
                    help='Biegeradius in mm fuer ECHTE Kruemmung (Bent-Mode, konforme '
                         'Abbildung). 0 = gerader Guide. Kontaktlinse: 8.3')
    ap.add_argument('--lambda-nm', type=float, default=LAM*1e9)
    ap.add_argument('--dy-nm', type=float, default=10.0, help='transversale Aufloesung (nm)')
    ap.add_argument('--L-mm', type=float, default=None,
                    help='Propagationslaenge Rand->Detektor in mm (Default aus Geometrie)')
    ap.add_argument('--offset-um', type=float, default=0.0, help='VCSEL-y-Offset (0=Kernmitte)')
    ap.add_argument('--waist-um', type=float, default=2.0, help='VCSEL-Taille (um)')
    ap.add_argument('--alpha-tear-permm', type=float, default=10.0,
                    help='Absorptions-/Streukoeff. des Traenenfilm-Analyten (1/mm) fuer die '
                         'ATR-Daempfung. Relative Szenario-Unterschiede sind ~alpha-unabhaengig.')
    ap.add_argument('--couple', choices=('analytic', 'fdtd'), default='analytic',
                    help='analytic = Gauss-Overlap; fdtd = lokales FDTD am Rand -> Moden-Projektion')
    ap.add_argument('--validate', action='store_true',
                    help='Eigenloeser gegen analytische Slab-Formel pruefen und beenden')
    ap.add_argument('--offset-scan', default='',
                    help='Komma-Liste VCSEL-Offsets (um), z.B. "-120,-60,0" -> evan/ATR vs Offset (Gesund)')
    args = ap.parse_args()

    lam_um = args.lambda_nm/1000.0
    dy_um = args.dy_nm/1000.0
    if args.validate:
        validate_solver(lam_um, dy_um)
        return
    # Bogenlaenge Rand (s=0) -> Detektor D1 (s=D1_S_CENTER): L = D_LENS/2 - x_d1, x_d1<0
    x_d1_mm = (D1_S_CENTER - D_LENS/2)*1e3
    L_um = (args.L_mm*1000.0) if args.L_mm is not None else (D_LENS/2*1e6 - x_d1_mm*1e3)
    print(f'Moden-Propagation  lam={args.lambda_nm:g}nm  dy={args.dy_nm:g}nm  '
          f'L(Rand->D1)={L_um/1000:.2f}mm  Kern={T_LENS*1e6:g}um  waist={args.waist_um:g}um '
          f'offset={args.offset_um:g}um')
    print('  (brute-force-Stitching bleibt erhalten: sliding_window_fdtd.py / full_domain_fdtd.py)')

    r_bend_um = args.r_bend_mm*1000.0 if args.r_bend_mm > 0 else None
    print(f'  ATR: Traenenfilm-Absorption alpha={args.alpha_tear_permm:g}/mm  Kopplung={args.couple}'
          f'  Kruemmung={"R=%.1fmm (Bent-Mode)" % args.r_bend_mm if r_bend_um else "gerade"}')
    scen = _scenarios_dict()

    # (C) Offset-Studie: Kopplungs-Abhaengigkeit des Signals (Gesund)
    if args.offset_scan.strip():
        offs = [float(s) for s in args.offset_scan.split(',')]
        t_lip, t_aq, t_mu, n_aq = scen.get('Gesund', list(scen.values())[0])
        print(f'\nOffset-Studie (Gesund):')
        print(f'{"offset[um]":>11} {"evan_ratio":>11} {"delta[um]":>10} {"ATR[dB]":>9}')
        print('-'*44)
        for off in offs:
            cf = fdtd_coupling_field(args.lambda_nm, off, args.waist_um) if args.couple == 'fdtd' else None
            r = run_scenario('Gesund', t_lip, t_aq, t_mu, n_aq, lam_um, dy_um, L_um,
                             off, args.waist_um, args.alpha_tear_permm, coupling=cf)
            print(f'{off:>11.0f} {r["evan_ratio"]:>11.4e} {r["delta_um"]:>10.3f} {r["ATR_dB"]:>9.4f}')
        return

    # (B) FDTD-Kopplung: Einkoppelfeld EINMAL am Rand rechnen, auf alle Szenarien projizieren
    coupling = None
    if args.couple == 'fdtd':
        print('  [FDTD-Kopplung] lokaler Rand-Lauf (VCSEL Butt-Coupling) ...', flush=True)
        coupling = fdtd_coupling_field(args.lambda_nm, args.offset_um, args.waist_um)

    names = [args.scenario] if args.scenario else list(scen.keys())
    print(f'\n{"Szenario":<14} {"n_eff":>8} {"evan_ratio":>11} {"delta[um]":>10} '
          f'{"T_bend":>8} {"ATR[dB]":>9}')
    print('  (evan_ratio = DIREKT bestimmtes evan. Feld Tear/Kern; T_bend = Biege-Durchsatz; '
          'ATR = messbare Daempfung)')
    print('-'*64)
    for nm in names:
        t_lip, t_aq, t_mu, n_aq = scen[nm]
        r = run_scenario(nm, t_lip, t_aq, t_mu, n_aq, lam_um, dy_um, L_um,
                         args.offset_um, args.waist_um, args.alpha_tear_permm,
                         coupling=coupling, r_bend_um=r_bend_um)
        print(f'{nm:<14} {r["neff_max"]:>8.4f} {r["evan_ratio"]:>11.4e} {r["delta_um"]:>10.3f} '
              f'{r["throughput_bend"]:>8.4f} {r["ATR_dB"]:>9.4f}')
    if r_bend_um:
        print(f'\nBiege-Analyse (R={args.r_bend_mm}mm): Strahlungs-Kaustik ~{r["u_caustic_um"]/1000:.2f}mm '
              f'oberhalb des Kerns -> Biegeverlust {r["bend_loss_perum"]:.2e}/um.')
        print(f'  Durchsatz durch die Linse (Biege-limitiert, L={L_um/1000:.1f}mm): '
              f'T_bend={r["throughput_bend"]:.4f}.')


if __name__ == '__main__':
    main()
