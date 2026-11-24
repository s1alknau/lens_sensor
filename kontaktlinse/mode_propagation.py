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


def materials_at(lam_nm):
    """Brechzahlen bei lam_nm aus der gemeinsamen Dispersionstabelle (common/physics,
    dieselbe wie im FDTD, Sellmeier-Fits mit Quellen). Aqueous als VERSCHIEBUNG: die
    Szenario-n_aq (1.336 normal, 1.340 hyperosmolar) sind Werte im Sichtbaren; genutzt
    wird n_aq_szen + d_aq = n_aqueous(lam) + (n_aq_szen - 1.336) (= physics.tear_index_at)."""
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from common.physics import n_at, N_TEAR_SCENARIO_REF
    return dict(pmma=n_at('pmma', lam_nm), lipid=n_at('lipid', lam_nm),
                mucin=n_at('mucin', lam_nm), cornea=n_at('cornea', lam_nm),
                d_aq=n_at('aqueous', lam_nm) - N_TEAR_SCENARIO_REF)


def build_index_profile(t_lip_um, t_aq_um, t_mu_um, n_aq, dy_um,
                        core_um=None, air_um=8.0, cornea_um=15.0, r_bend_um=None,
                        mats=None):
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
    # mats (materials_at): wellenlaengenabhaengige Brechzahlen; None = Modell bei 850 nm
    m = mats or materials_at(850.0)                # Default: Dispersionsmodell bei 850 nm
    n = np.full(n_pts, N_AIR)                          # Default Luft (oben)
    n[y < core_um] = m['pmma']                         # Kern (bis Oberkante)
    n[y < 0] = m['lipid']                              # Lipid direkt unter dem Kern
    n[y < -t_lip_um] = n_aq + m['d_aq']                # Aqueous
    n[y < -(t_lip_um + t_aq_um)] = m['mucin']          # Mucin
    n[y < -(t_lip_um + t_aq_um + t_mu_um)] = m['cornea']  # Cornea
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


def solve_tm_modes(y, n, lam_um):
    """Gefuehrte TM-Moden: eps*d/dy(1/eps*dHz/dy) + k0^2*eps*Hz = beta^2*Hz.
    Der (1/eps)-gewichtete Operator wird symmetrisch transformiert (A = D_sqrt(eps)
    * (L_w + k0^2) * D_sqrt(eps), Hz = sqrt(eps)*psi) -> gleiche tridiagonale
    Eigenwertloesung wie TE. Rueckgabe (n_eff[absteigend], Hz-Profile normiert)."""
    dy = y[1] - y[0]
    k0 = 2*np.pi/lam_um
    eps = n**2
    eps_h = 0.5*(eps[:-1] + eps[1:])                   # eps an Halbknoten j+1/2
    P = 1.0/eps_h                                      # (1/eps)-Gewicht, Laenge N-1
    inv = 1.0/dy**2
    # symmetrischer, (1/eps)-gewichteter Laplace L_w (Rand: gespiegelt P_{-1/2}=P_{1/2})
    Ld = np.empty(len(y))
    Ld[1:-1] = -(P[1:] + P[:-1])*inv
    Ld[0] = -2.0*P[0]*inv; Ld[-1] = -2.0*P[-1]*inv
    Lo = P*inv                                         # Nebendiagonale von L_w (Laenge N-1)
    sq = np.sqrt(eps)
    A_diag = eps*(Ld + k0*k0)                          # eps*(L_w + k0^2)
    A_off = sq[:-1]*sq[1:]*Lo                          # sqrt(eps_j eps_{j+1}) * P/dy^2
    n_core = n.max(); n_clad_out = max(n[0], n[-1])
    lo = (k0*n_clad_out)**2 + 1e-6
    hi = (k0*n_core)**2 - 1e-6
    w, psi = eigh_tridiagonal(A_diag, A_off, select='v', select_range=(lo, hi))
    order = np.argsort(w)[::-1]
    w = w[order]; psi = psi[:, order]
    neff = np.sqrt(np.clip(w, 0, None))/k0
    Hz = sq[:, None]*psi                               # Ruecktransformation Hz = sqrt(eps)*psi
    Hz = Hz/np.sqrt(np.sum(Hz**2, axis=0)*dy)          # Norm: int |Hz|^2 dy = 1
    return neff, Hz


def solve_modes(y, n, lam_um, pol='s'):
    """TE ('s', Ez) oder TM ('p', Hz) je nach Polarisation."""
    return solve_tm_modes(y, n, lam_um) if str(pol).lower() in ('p', 'tm') \
        else solve_te_modes(y, n, lam_um)


def couple(y, profiles, y0_um, waist_um):
    """VCSEL-Gauss (Taille waist, Zentrum y0) -> Moden-Amplituden a_m = <phi_m|g>."""
    dy = y[1] - y[0]
    g = np.exp(-((y - y0_um)/waist_um)**2)
    g = g/np.sqrt(np.sum(g**2)*dy)
    a = (profiles*g[:, None]).sum(axis=0)*dy
    return a


def project_field(y, profiles, y_src, E_src, pol='s', n=None, betas=None, k0=None):
    """Projiziert ein (FDTD-)Feld E_src(y_src) auf die Moden -> Amplituden a_m.
    y_src/E_src werden auf das Modengitter interpoliert.
      s/TE: E_src = Ez, Moden = Ez-Profile -> a_m = <phi_m|Ez>.
      p/TM: E_src = Ey, Moden = Hz-Profile. Mit Ey = beta_m Hz_m/(k0 eps) und der
            TM-Orthogonalitaet int Hz_m Hz_n/eps dy = N_m delta_mn gilt EXAKT
            a_m = k0 int Hz_m Ey dy / (beta_m N_m)  (Entwicklung Hz = sum a_m Hz_m).
            Braucht n(y), beta_m, k0."""
    E = (np.interp(y, y_src, np.real(E_src), left=0.0, right=0.0)
         + 1j*np.interp(y, y_src, np.imag(E_src), left=0.0, right=0.0))
    dy = y[1] - y[0]
    ov = (profiles*E[:, None]).sum(axis=0)*dy
    if str(pol).lower() in ('p', 'tm'):
        N = (profiles**2/(n**2)[:, None]).sum(axis=0)*dy
        return k0*ov/(betas*N)
    return ov


def fdtd_coupling_field(lam_nm, offset_um, waist_um, dx_nm=50.0, length_um=30.0,
                        pol='s'):
    """Lokales FDTD am Rand: VCSEL koppelt in den 250-um-Kern; Rueckgabe des
    komplexen CW-Feldes an einem Querschnitt NACH der Einkoppelzone (y so, dass
    Kern=[0,core]). Das ersetzt den analytischen Gauss-Overlap durch die reale
    Einkopplungs-Physik (Fresnel, Nahfeld) -> realistische Moden-Verteilung.
    pol: 's' -> Ez (TE), 'p' -> Ey (TM); passend zur Moden-Polarisation."""
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
                          length_um=length_um, polarization=pol,
                          t_aqueous_um=8.0, t_mucin_um=2.0)
        cwm = r['cw_frames_memmap']; cwf = r['cw_frames'] or []
        arr = np.asarray(np.load(cwm, mmap_mode='r')[:len(cwf)])
    finally:
        os.chdir(prev)
    nfr, Nx, Ny = arr.shape
    # Komplexe Amplitude per DFT ueber alle Phasen (Frame k = Re(Ê e^{2πik/N}))
    w = np.exp(-2j*np.pi*np.arange(nfr)/nfr)
    E = (2.0/nfr)*np.tensordot(w, arr.astype(np.float64), axes=(0, 0))
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
    _m = materials_at(lam_um*1000.0)               # dieselben n wie build_index_profile
    n_ana = cv.slab_neff_analytic(_m['pmma'], N_AIR, 1.336 + _m['d_aq'], d, lam_um*1000, 's', 0)
    dd = abs(n_num - n_ana)
    print(f'[Validation] thin slab d={d:g}um: eigensolver n_eff={n_num:.5f}  '
          f'analytic={n_ana:.5f}  Delta={dd:.2e}  -> {"OK" if dd < 2e-3 else "CHECK"}')
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
                 r_bend_um=None, pol='s', verbose=False):
    """ATR-Sensor + (optional) echte Kruemmung. Traenenfilm-Absorption daempft das
    totalreflektierte Licht (alpha_m = alpha_tear*Gamma_m -> ueberlebendes TIR-Licht).
    r_bend_um: Biegeradius -> Bent-Mode via konformer Abbildung; zusaetzlich wird der
    Biege-/Strahlungsverlust ueber die Strahlungs-Kaustik abgeschaetzt (-> Durchsatz)."""
    core = T_LENS*1e6
    y, n = build_index_profile(t_lip, t_aq, t_mu, n_aq, dy_um, r_bend_um=r_bend_um,
                               mats=materials_at(lam_um*1000.0))   # n(lambda) wie im FDTD
    dy = y[1] - y[0]
    neff, prof = solve_modes(y, n, lam_um, pol)
    k0 = 2*np.pi/lam_um
    betas = neff*k0
    if str(pol).lower() in ('p', 'tm'):
        # TM: Einfallsfeld ist Ey (VCSEL-Gauss bzw. FDTD-Ey), Moden sind Hz-Profile.
        # Exakte Entwicklung (project_field) + LEISTUNGS-Gewichte aus dem Poynting-
        # Fluss: P_m = beta_m N_m |a_m|^2/(2k0), P_in = int n|Ey|^2 dy/2.
        if coupling is not None:
            y_s, E_s = coupling
        else:
            y_s, E_s = y, np.exp(-((y - (core/2 + offset_um))/waist_um)**2)
        Ey = (np.interp(y, y_s, np.real(E_s), left=0.0, right=0.0)
              + 1j*np.interp(y, y_s, np.imag(E_s), left=0.0, right=0.0))
        a = project_field(y, prof, y, Ey, pol='p', n=n, betas=betas, k0=k0)
        N_m = (prof**2/(n**2)[:, None]).sum(axis=0)*dy
        P_in = max(float(np.sum(n*np.abs(Ey)**2)*dy)/2.0, 1e-30)
        P = betas*N_m*np.abs(a)**2/(2.0*k0)/P_in
        a = a/np.sqrt(P_in)
        if coupling is not None:                          # FDTD: auf Einheitsleistung
            s = max(float(P.sum()), 1e-30); P = P/s; a = a/np.sqrt(s)
    else:
        if coupling is not None:                          # reale FDTD-Einkopplung -> Projektion
            a = project_field(y, prof, coupling[0], coupling[1])
            a = a/np.sqrt(max(np.sum(np.abs(a)**2), 1e-30))  # auf Einheitsleistung normiert
        else:                                             # analytischer Gauss-Overlap
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
        print(f'  modes {len(neff)}  eta={eta:.3f}  Gamma_aq={Gamma_eff:.3e}  '
              f'evan_ratio={evan_ratio:.3e}  delta={delta_um:.3f}um  '
              f'bend_loss={bend_loss_perum:.2e}/um  T_bend={throughput_bend:.3f}')
    return dict(name=name, n_modes=len(neff), neff_max=neff_max,
                eta=eta, R_tear=float(R), Gamma_aq=Gamma_eff, ATR_dB=float(ATR_dB),
                P_surv=P_surv, evan_ratio=evan_ratio, I_surf=I_surf,
                I_evan_aq=I_evan_aq, delta_um=float(delta_um),
                bend_loss_perum=bend_loss_perum, u_caustic_um=float(u_caustic_um),
                throughput_bend=throughput_bend)


_CLI_DESCRIPTION = '''Native mode propagation for the contact-lens sensor (resolution-robust).

Analogous to the Meep mode handoff, but native and WITHOUT FDTD along the length:
  1) Transverse eigenmodes of the layered slab (air | PMMA core | lipid |
     aqueous | mucin | cornea) via a tridiagonal 1D eigensolver -> exact beta_m
     (n_eff) + profiles phi_m(y). Fine in y (resolves the evanescent field),
     no grid along the propagation -> NO accumulating dispersion.
  2) VCSEL coupling via overlap integral -> mode amplitudes a_m.
  3) Analytic propagation along the arc: a_m -> a_m * exp(i beta_m L).
  4) Sensor: field at the detector y positions (D1 tear side, D2 air side)
     -> R_tear = |E(D1)|^2 / |E(D2)|^2.

Brute-force stitching (sliding_window_fdtd.py / full_domain_fdtd.py) is kept
unchanged - this is an ADDITIONAL fast and accurate method.

Usage:
    python mode_propagation.py                 # test run: R_tear per scenario
    python mode_propagation.py --scenario DED --L-mm 12 --offset-um 0
'''


def build_parser():
    """Argparse-Parser (auch vom GUI genutzt, um die Felder zu erzeugen)."""
    ap = argparse.ArgumentParser(description=_CLI_DESCRIPTION,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g_case = ap.add_argument_group('Scenario & physics')
    g_case.add_argument('--scenario', default=None, help='Only this scenario (otherwise all)')
    g_case.add_argument('--polarization', choices=('s', 'p'), default='s',
                        help='s = TE (Ez modes) | p = TM (Hz modes)')
    g_case.add_argument('--r-bend-mm', type=float, default=0.0,
                        help='Bend radius in mm for TRUE curvature (bent mode, conformal '
                             'mapping). 0 = straight guide. Contact lens: 8.3')
    g_case.add_argument('--lambda-nm', type=float, default=LAM*1e9, help='Wavelength (nm)')

    g_cpl = ap.add_argument_group('Coupling (VCSEL)')
    g_cpl.add_argument('--offset-um', type=float, default=0.0, help='VCSEL y offset (0 = core center)')
    g_cpl.add_argument('--waist-um', type=float, default=2.0, help='VCSEL waist (um)')
    g_cpl.add_argument('--couple', choices=('analytic', 'fdtd'), default='analytic',
                       help='analytic = Gaussian overlap; fdtd = local FDTD at the edge -> mode projection')
    g_cpl.add_argument('--offset-scan', default='',
                       help='Comma list of VCSEL offsets (um), e.g. "-120,-60,0" -> evan/ATR vs offset (Gesund)')

    g_tear = ap.add_argument_group('Tear film & ATR')
    g_tear.add_argument('--t-lipid-nm', type=float, default=None,
                        help='Override lipid thickness (nm) (otherwise scenario value; Gesund=30)')
    g_tear.add_argument('--alpha-tear-permm', type=float, default=10.0,
                        help='Absorption/scattering coefficient of the tear-film analyte (1/mm) for the '
                             'ATR attenuation. Relative scenario differences are ~independent of alpha.')

    g_num = ap.add_argument_group('Numerics')
    g_num.add_argument('--dy-nm', type=float, default=10.0, help='Transverse resolution (nm)')
    g_num.add_argument('--L-mm', type=float, default=None,
                       help='Propagation length edge->detector in mm (default from geometry)')
    g_num.add_argument('--validate', action='store_true',
                       help='Check the eigensolver against the analytic slab formula and exit')
    return ap


def main():
    args = build_parser().parse_args()
    _pol = args.polarization
    _tl = (lambda tl: args.t_lipid_nm/1000.0 if args.t_lipid_nm is not None else tl)
    lam_um = args.lambda_nm/1000.0
    dy_um = args.dy_nm/1000.0
    if args.validate:
        validate_solver(lam_um, dy_um)
        return
    # Bogenlaenge Rand (s=0) -> Detektor D1 (s=D1_S_CENTER): L = D_LENS/2 - x_d1, x_d1<0
    x_d1_mm = (D1_S_CENTER - D_LENS/2)*1e3
    L_um = (args.L_mm*1000.0) if args.L_mm is not None else (D_LENS/2*1e6 - x_d1_mm*1e3)
    print(f'Mode propagation  {"TM (Hz)" if _pol=="p" else "TE (Ez)"}  '
          f'lam={args.lambda_nm:g}nm  dy={args.dy_nm:g}nm  '
          f'L(edge->D1)={L_um/1000:.2f}mm  core={T_LENS*1e6:g}um  waist={args.waist_um:g}um '
          f'offset={args.offset_um:g}um'
          + (f'  Lipid={args.t_lipid_nm:g}nm(override)' if args.t_lipid_nm is not None else ''))
    print('  (brute-force stitching is still available: sliding_window_fdtd.py / full_domain_fdtd.py)')

    r_bend_um = args.r_bend_mm*1000.0 if args.r_bend_mm > 0 else None
    print(f'  ATR: tear-film absorption alpha={args.alpha_tear_permm:g}/mm  coupling={args.couple}'
          f'  curvature={"R=%.1fmm (bent mode)" % args.r_bend_mm if r_bend_um else "straight"}')
    scen = _scenarios_dict()

    # (C) Offset-Studie: Kopplungs-Abhaengigkeit des Signals (Gesund)
    if args.offset_scan.strip():
        offs = [float(s) for s in args.offset_scan.split(',')]
        t_lip, t_aq, t_mu, n_aq = scen.get('Gesund', list(scen.values())[0])
        print(f'\nOffset study (Gesund):')
        print(f'{"offset[um]":>11} {"evan_ratio":>11} {"delta[um]":>10} {"ATR[dB]":>9}')
        print('-'*44)
        for off in offs:
            cf = (fdtd_coupling_field(args.lambda_nm, off, args.waist_um, pol=_pol)
                  if args.couple == 'fdtd' else None)
            r = run_scenario('Gesund', _tl(t_lip), t_aq, t_mu, n_aq, lam_um, dy_um, L_um,
                             off, args.waist_um, args.alpha_tear_permm, coupling=cf, pol=_pol)
            print(f'{off:>11.0f} {r["evan_ratio"]:>11.4e} {r["delta_um"]:>10.3f} {r["ATR_dB"]:>9.4f}')
        return

    # (B) FDTD-Kopplung: Einkoppelfeld EINMAL am Rand rechnen, auf alle Szenarien projizieren
    coupling = None
    if args.couple == 'fdtd':
        print('  [FDTD coupling] local edge run (VCSEL butt coupling) ...', flush=True)
        coupling = fdtd_coupling_field(args.lambda_nm, args.offset_um, args.waist_um, pol=_pol)

    names = [args.scenario] if args.scenario else list(scen.keys())
    print(f'\n{"Scenario":<14} {"n_eff":>8} {"evan_ratio":>11} {"delta[um]":>10} '
          f'{"T_bend":>8} {"ATR[dB]":>9}')
    print('  (evan_ratio = DIRECTLY determined evan. field tear/core; T_bend = bend throughput; '
          'ATR = measurable attenuation)')
    print('-'*64)
    for nm in names:
        t_lip, t_aq, t_mu, n_aq = scen[nm]
        r = run_scenario(nm, _tl(t_lip), t_aq, t_mu, n_aq, lam_um, dy_um, L_um,
                         args.offset_um, args.waist_um, args.alpha_tear_permm,
                         coupling=coupling, r_bend_um=r_bend_um, pol=_pol)
        print(f'{nm:<14} {r["neff_max"]:>8.4f} {r["evan_ratio"]:>11.4e} {r["delta_um"]:>10.3f} '
              f'{r["throughput_bend"]:>8.4f} {r["ATR_dB"]:>9.4f}')
    if r_bend_um:
        print(f'\nBend analysis (R={args.r_bend_mm}mm): radiation caustic ~{r["u_caustic_um"]/1000:.2f}mm '
              f'above the core -> bend loss {r["bend_loss_perum"]:.2e}/um.')
        print(f'  Throughput through the lens (bend-limited, L={L_um/1000:.1f}mm): '
              f'T_bend={r["throughput_bend"]:.4f}.')


if __name__ == '__main__':
    main()
