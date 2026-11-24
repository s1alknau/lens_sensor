"""PROTOTYP: Fenster-Stitch fuer Meep (planar 2D, gefuehrte Mode).

Idee (S-Matrix-Kaskade der gefuehrten Mode): pro Fenster wird die Fundamental-
mode per EigenModeSource eingespeist; am rechten Handoff-Plane wird der VORWAERTS
laufende Mode-Koeffizient gemessen und als Eingangsamplitude fuer das naechste
Fenster benutzt. Es ist immer nur EIN Fenster im Speicher -> Speicher-Vorteil
wie beim nativen Stitch, aber mit Meep-Physik.

Grenzen: gut fuer waveguide-dominierte Felder. Starke Streuung/Radiation (z.B.
grosser Bead) wird von der reinen Fundamentalmoden-Kaskade nur genaehert
(hoehere Moden/Strahlung gehen im Handoff verloren). Genau deshalb wird gegen
einen Meep-Full-Domain-Lauf gleicher (kurzer) Laenge validiert.
"""
import numpy as np
import meep as mp
import meep_backend as mb


def _window_cfg(cfg, gx0, Wx):
    """cfg fuer ein Fenster [gx0, gx0+Wx]: Zentrum verschoben, Bead nur wenn drin."""
    wc = dict(cfg)
    wc['lx'] = Wx
    wc['cx'] = gx0 + Wx/2.0
    b = cfg.get('bead')
    wc['bead'] = b if (b and gx0 <= b['x'] <= gx0 + Wx) else None
    return wc


def _parity(cfg):
    """Eigenmode-Paritaet fuer 2D: NO_PARITY liefert als Band 1 die Mode mit dem
    hoechsten n_eff UEBER BEIDE Polarisationen (beim Slab die Ez/TE-Mode) -> bei p
    wuerde die falsche Mode gelauncht/projiziert. ODD_Z = (Ez,Hx,Hy) = s,
    EVEN_Z = (Hz,Ex,Ey) = p."""
    return mp.ODD_Z if cfg['pol'] == 's' else mp.EVEN_Z


def _fwd(sim, mon, parity):
    """Vorwaerts (+x) Mode-Koeffizient der Fundamentalmode am Monitor."""
    c = sim.get_eigenmode_coefficients(mon, [1], eig_parity=parity)
    return complex(c.alpha[0, 0, 0])


def run_stitch(cfg, window_w_um, slide_um, modes=1):
    """Fenster-Stitch als Mode-Kaskade. modes=1 -> Fundamentalmode; modes=M>1 ->
    M gefuehrte Moden im Handoff (Transfer-MATRIX pro Fenster, M Laeufe/Fenster),
    erfasst Mode-Konversion. Strahlung (Radiation) wird von keiner Mode getragen
    und erscheint als Verlust der gefuehrten Gesamtleistung."""
    comp = mp.Ez if cfg['pol'] == 's' else mp.Ey
    parity = _parity(cfg)          # 2D: Ez-Moden ODD_Z (s), Ey-Moden EVEN_Z (p)
    f0 = cfg['f0']
    lx, ly, res = cfg['lx'], cfg['ly'], cfg['resolution']
    M = max(1, int(modes))
    bands = list(range(1, M + 1))
    Wx = min(window_w_um, lx)
    dpml = max(0.1, min(cfg['lam_nm']/1000.0, 0.4*min(Wx, ly)))
    y_off = cfg['t_wg']/2 - cfg['cy']
    mode_h = min(ly - 2*dpml, cfg['t_wg'] + 4.0)
    xL = -Wx/2 + dpml + 0.5
    xR = min(xL + slide_um, Wx/2 - dpml - 0.4)
    Sx = xR - xL
    if Sx <= 0.2:
        raise SystemExit('[meep-stitch] window too small for the slide: '
                         'choose a larger --window-um or a smaller --slide-um.')
    n_win = (int(np.ceil((lx - Wx)/Sx)) + 1) if lx > Wx else 1

    Nx_full = int(round(lx*res))
    Ny = int(round(ly*res))
    Efull = np.zeros((Nx_full, Ny), dtype=np.complex64)

    print(f'[meep-stitch] {n_win} windows of {Wx:g}um, transfer slide {Sx:.2f}um, '
          f'{M} mode(s), grid/window ~{int(Wx*res)}x{Ny}')
    A = np.zeros(M, dtype=complex)
    A[0] = 1.0             # Eingang: Fundamentalmode, Amplitude 1
    Ttot = np.eye(M, dtype=complex)
    for w in range(n_win):
        gx0 = w*Sx       # NICHT am Ende klemmen: die Kaskade braucht Fensterstart = w*Sx
        wc = _window_cfg(cfg, gx0, Wx)
        Tcol = np.zeros((M, M), dtype=complex)     # Transfer-Matrix des Fensters
        fields = []                                 # Feld je EINHEITS-Eingangsmode
        for jj, j in enumerate(bands):
            src = mp.EigenModeSource(
                src=mp.GaussianSource(f0, fwidth=0.15*f0),
                center=mp.Vector3(-Wx/2 + dpml + 0.2, y_off, 0),
                size=mp.Vector3(0, mode_h, 0), eig_band=j, direction=mp.X,
                eig_match_freq=True, eig_parity=parity, amplitude=1.0)
            sim = mp.Simulation(cell_size=mp.Vector3(Wx, ly, 0), resolution=res,
                                geometry=mb.build_geometry(wc), sources=[src],
                                boundary_layers=[mp.PML(dpml)],
                                default_material=mp.Medium(index=1.0),
                                force_complex_fields=True)
            monL = sim.add_mode_monitor(f0, 0, 1, mp.FluxRegion(
                center=mp.Vector3(xL, y_off, 0), size=mp.Vector3(0, mode_h, 0)))
            monR = sim.add_mode_monitor(f0, 0, 1, mp.FluxRegion(
                center=mp.Vector3(xR, y_off, 0), size=mp.Vector3(0, mode_h, 0)))
            dft = sim.add_dft_fields([comp], f0, 0, 1,
                                     center=mp.Vector3(), size=mp.Vector3(Wx, ly, 0))
            sim.run(until_after_sources=mp.stop_when_fields_decayed(
                15, comp, mp.Vector3(xR, y_off, 0), 1e-4))
            cL = sim.get_eigenmode_coefficients(monL, bands, eig_parity=parity)
            cR = sim.get_eigenmode_coefficients(monR, bands, eig_parity=parity)
            aLj = complex(cL.alpha[jj, 0, 0])       # gelaunchte Amplitude von Mode j
            if abs(aLj) < 1e-12:
                aLj = 1e-12
            Tcol[:, jj] = np.array([complex(cR.alpha[i, 0, 0])
                                    for i in range(M)])/aLj
            fields.append(np.asarray(sim.get_dft_array(dft, comp, 0))/aLj)
            del sim
        # physikalisches Fensterfeld = Summe A[j] * Feld_j
        fld = sum(A[jj]*fields[jj] for jj in range(M))
        gi0 = int(round(gx0*res))
        Ox = Wx - Sx
        trim = int(round(0.5*Ox*res)) if Ox > 0 else 0
        l0 = trim if w > 0 else 0
        l1 = fld.shape[0] - (trim if w < n_win - 1 else 0)
        gi1 = min(gi0 + l1, Nx_full)
        ny = min(Ny, fld.shape[1])           # Meep liefert oft Ny+1/Ny+2 Punkte
        Efull[gi0 + l0:gi1, :ny] = fld[l0:l0 + (gi1 - gi0 - l0), :ny]

        A = Tcol @ A
        Ttot = Tcol @ Ttot
        print(f'  [meep-stitch] window {w+1}/{n_win} x0={gx0:.1f}um  '
              f'guided power |A|^2={float(np.sum(np.abs(A)**2)):.3f}', flush=True)

    Tr = float(np.sum(np.abs(A)**2))    # gefuehrte Gesamtleistung am Ende (Eingang=1)
    print(f'[meep-stitch Result] {cfg["label"]}: Transmission(guided, {M} mode(s))'
          f'~{Tr:.4f} ({n_win} windows, only ONE window in RAM)')
    return dict(Efull=Efull, transmission=Tr, n_win=n_win, shape=(Nx_full, Ny),
                modes=M)


def _profile_interp(prof_y, ys, y):
    """Lineare Interpolation eines 1D-Feldprofils prof_y(ys) an Stelle y."""
    return complex(np.interp(y, ys, prof_y.real) + 1j*np.interp(y, ys, prof_y.imag))


def _yee_angular_filter(prof, res, kmed):
    """Yee-Offset-Korrektur WINKELAUFGELOEST: statt einer globalen (Grundmoden-)
    Phase bekommt jede transversale ky-Komponente ihre eigene kx-Phase
    exp(-i*kx*dx/2) mit kx=sqrt(kmed^2-ky^2). Damit wird der halbe-Zellen-Versatz
    zwischen E- und H-Blatt fuer ALLE Winkel korrekt korrigiert (auch Strahlung),
    nicht nur fuer die gefuehrte Mode. Reduziert sich fuer ky~0 auf die
    Grundmoden-Korrektur. dx = 1/res; Einheiten wie Meeps kdom (cycles/um)."""
    prof = np.asarray(prof)
    N = len(prof)
    ky = np.fft.fftfreq(N, d=1.0/res)
    kx = np.sqrt((kmed*kmed - ky*ky).astype(complex))
    ph = np.exp(-1j*kx*(0.5/res))
    return np.fft.ifft(np.fft.fft(prof)*ph)


def _equiv_signs(cfg):
    """Vorzeichen (J, M) der Aequivalenz-Stromblaetter fuer Vorwaerts-Einstrahlung (+x):
    J = n x H, M = -n x E mit n = +x.  s (Ez,Hy): Jz = +Hy, My = +Ez;
    p (Ey,Hz): Jy = -Hz, Mz = -Ey.  In Meep gemessen (results/meep_sign_check.py):
    so strahlt das Blatt zu ~99 % vorwaerts; mit umgedrehtem M (frueher: My = -Ez)
    lief es fast nur RUECKWAERTS und C musste einen kleinen Rest hochskalieren."""
    return (1.0, 1.0) if cfg['pol'] == 's' else (-1.0, -1.0)


def _calibrate_C(cfg, Wx):
    """Bestimmt den Aequivalenz-Skalenfaktor C selbst: extrahiert in einem
    uniformen Fenster das Ez/Hy-Profil + Mode-Amplitude a_ref an einer Ebene,
    praegt es dann per Zwei-Strom-Quelle (C=1) ein und misst a_test.
    C = a_ref/a_test macht die Einpraegung amplituden- UND phasenerhaltend.
    Robust gegen Aufloesung (C haengt von der Diskretisierung ab)."""
    comp = mp.Ez if cfg['pol'] == 's' else mp.Ey
    hcomp = mp.Hy if cfg['pol'] == 's' else mp.Hz
    parity = _parity(cfg)
    f0 = cfg['f0']
    ly, res = cfg['ly'], cfg['resolution']
    dpml = max(0.1, min(cfg['lam_nm']/1000.0, 0.4*min(Wx, ly)))
    y_off = cfg['t_wg']/2 - cfg['cy']
    mode_h = min(ly - 2*dpml, cfg['t_wg'] + 6.0)
    x_inj = -Wx/2 + dpml + 0.3
    xP = x_inj + 1.0
    ys = (np.arange(int(round(mode_h*res))) - int(round(mode_h*res))//2)/res
    uni = dict(cfg); uni['bead'] = None; uni['cx'] = Wx/2.0

    A_PROBE = 1e3      # kraeftige Probe -> Einpraegung sicher UEBER dem Rauschen
                       # (bei jeder Aufloesung), Meep ist linear -> keine Saettigung.

    def _win(sources, mon_x):
        sim = mp.Simulation(cell_size=mp.Vector3(Wx, ly, 0), resolution=res,
                            geometry=mb.build_geometry(uni), sources=sources,
                            boundary_layers=[mp.PML(dpml)],
                            default_material=mp.Medium(index=1.0),
                            force_complex_fields=True)
        mon = sim.add_mode_monitor(f0, 0, 1, mp.FluxRegion(
            center=mp.Vector3(mon_x, y_off, 0), size=mp.Vector3(0, mode_h, 0)))
        fl = sim.add_flux(f0, 0, 1, mp.FluxRegion(
            center=mp.Vector3(mon_x, y_off, 0), size=mp.Vector3(0, mode_h, 0)))
        dft = sim.add_dft_fields([comp, hcomp], f0, 0, 1,
                                 center=mp.Vector3(), size=mp.Vector3(Wx, ly, 0))
        sim.run(until_after_sources=mp.stop_when_fields_decayed(
            15, comp, mp.Vector3(mon_x, y_off, 0), 1e-4))
        rc = sim.get_eigenmode_coefficients(mon, [1], eig_parity=parity)
        a = complex(rc.alpha[0, 0, 0])
        beta = float(abs(rc.kdom[0].x))     # Mode-Wellenzahl (fuer Yee-Offset-Phase)
        flux = float(mp.get_fluxes(fl)[0])
        ez = np.asarray(sim.get_dft_array(dft, comp, 0))
        hy = np.asarray(sim.get_dft_array(dft, hcomp, 0))
        del sim
        return a, ez, hy, flux, beta

    # Run A: Referenz (Eigenmode) -> Profil, a_ref, Ziel-Fluss P_target UND beta.
    aref, ezA, hyA, P_target, beta = _win([mp.EigenModeSource(
        src=mp.GaussianSource(f0, fwidth=0.15*f0), center=mp.Vector3(x_inj, y_off, 0),
        size=mp.Vector3(0, mode_h, 0), eig_band=1, direction=mp.X,
        eig_match_freq=True, eig_parity=parity, amplitude=1.0)], xP)
    ix = min(max(int(round((xP + Wx/2)*res)), 0), ezA.shape[0]-1)
    jc = int(round((y_off + ly/2)*res)); j0 = jc - len(ys)//2
    sl = slice(max(0, j0), max(0, j0)+len(ys))
    ez_p, hy_p = ezA[ix, sl].copy(), hyA[ix, sl].copy()
    # Yee-Offset (Hy in x um +dx/2 versetzt zu Ez) WINKELAUFGELOEST korrigieren:
    # das ez_p-Profil (treibt die magnetische Stromquelle) durch den Winkelspektrum-
    # Filter (kmed = f0*n_wg) -> jede ky-Komponente ihre eigene kx-Phase.
    kmed = f0*cfg['wg_n']
    ez_pf = _yee_angular_filter(ez_p, res, kmed)
    js, ms = _equiv_signs(cfg)
    src_t = mp.GaussianSource(f0, fwidth=0.15*f0)
    aprobe, _, _, P_probe, _ = _win([
        mp.Source(src=src_t, component=comp, center=mp.Vector3(x_inj, y_off, 0),
                  size=mp.Vector3(0, mode_h, 0),
                  amp_func=lambda p: js*A_PROBE*_profile_interp(hy_p, ys, p.y)),
        mp.Source(src=src_t, component=hcomp, center=mp.Vector3(x_inj, y_off, 0),
                  size=mp.Vector3(0, mode_h, 0),
                  amp_func=lambda p: ms*A_PROBE*_profile_interp(ez_pf, ys, p.y))], xP)
    Cmag = A_PROBE*np.sqrt(abs(P_target)/max(abs(P_probe), 1e-30))
    phase = np.angle(aref) - np.angle(aprobe) if abs(aprobe) > 1e-30 else 0.0
    C = Cmag*np.exp(1j*phase)
    print(f'[meep-stitch/fullfield] auto-calibration |C|={Cmag:.3g} '
          f'kmed={kmed:.2f} (P_target={P_target:.2e}, P_probe={P_probe:.2e})')
    return C


def run_stitch_fullfield(cfg, window_w_um, slide_um, C=None):
    """VOLL-FELDBASIERTER Handoff (Aequivalenzprinzip): statt nur gefuehrter Moden
    wird das GESAMTE komplexe Querschnittsfeld (Ez UND Hy) am Handoff-Plane
    extrahiert und im naechsten Fenster ueber ZWEI Stromblaetter (elektrisch aus
    Hy, magnetisch aus Ez) VORWAERTS eingepraegt -> traegt auch Strahlung mit.
    C wird selbst kalibriert: Betrag ueber Fluss (kraeftige Probe, rauschfrei),
    Phase ueber den Mode-Koeffizienten, plus Yee-Offset-Korrektur exp(-i*beta*dx/2)
    zwischen E- und H-Blatt. Damit AUFLOESUNGSROBUST (nm=15..60) fuer gefuehrt-
    dominierte Felder (validiert vs Full-Domain: ~0.997 vs ~1.0).

    GRENZE: Die skalare Kalibrierung C ist auf die Grundmode getunt. Ist das
    Handoff-Feld radiation-dominiert (STARKE Streuung, z.B. Bead im WG-Kern),
    wird die Strahlung fehl-skaliert -> unphysikalische Ergebnisse (Transmission
    kann >1 werden, Amplitude explodiert). In dem Regime ist die Mode-Kaskade
    (--meep-handoff mode) oder der native Stitch vorzuziehen."""
    print('  [full-field handoff] equivalence principle (Ez+Hy), '
          'flux self-calibrated + Yee-offset phase-corrected -> '
          'resolution-robust (nm=15..60) for GUIDED-DOMINATED fields.')
    print('  !!! IMPORTANT: the scalar calibration is tuned to the fundamental mode. '
          'With STRONG scattering (radiation-heavy handoff field, e.g. bead in the '
          'WG core) it scales the radiation wrongly -> UNPHYSICAL (amplitude '
          'can blow up). In that case use --meep-handoff mode (multi-mode) or the native '
          'stitch. !!!')
    comp = mp.Ez if cfg['pol'] == 's' else mp.Ey
    hcomp = mp.Hy if cfg['pol'] == 's' else mp.Hz
    parity = _parity(cfg)
    f0 = cfg['f0']
    lx, ly, res = cfg['lx'], cfg['ly'], cfg['resolution']
    Wx = min(window_w_um, lx)
    if C is None:
        C = _calibrate_C(cfg, Wx)
    kmed = cfg['f0']*cfg['wg_n']            # fuer die winkelaufgeloeste Yee-Korrektur
    dpml = max(0.1, min(cfg['lam_nm']/1000.0, 0.4*min(Wx, ly)))
    y_off = cfg['t_wg']/2 - cfg['cy']
    mode_h = min(ly - 2*dpml, cfg['t_wg'] + 6.0)
    x_inj = -Wx/2 + dpml + 0.3           # Einpraege-Plane (lokal)
    xR = min(x_inj + slide_um, Wx/2 - dpml - 0.4)
    Sx = xR - x_inj
    n_win = (int(np.ceil((lx - Wx)/Sx)) + 1) if lx > Wx else 1
    Nx_full = int(round(lx*res)); Ny = int(round(ly*res))
    Efull = np.zeros((Nx_full, Ny), dtype=np.complex64)
    # y-Koordinaten der Handoff-Profile (Meep-lokal, relativ zum Quell-Zentrum)
    ys = (np.arange(int(round(mode_h*res))) - int(round(mode_h*res))//2)/res

    print(f'[meep-stitch/fullfield] {n_win} windows of {Wx:g}um, slide {Sx:.2f}um, C={C}')
    handoff = None      # (Ez_profile, Hy_profile) am Handoff-Plane des Vorgaengers
    afwd0 = None; afwd_last = None
    for w in range(n_win):
        gx0 = w*Sx       # NICHT am Ende klemmen: die Kaskade braucht Fensterstart = w*Sx
        wc = _window_cfg(cfg, gx0, Wx)
        if w == 0:
            sources = [mp.EigenModeSource(
                src=mp.GaussianSource(f0, fwidth=0.15*f0),
                center=mp.Vector3(x_inj, y_off, 0),
                size=mp.Vector3(0, mode_h, 0), eig_band=1, direction=mp.X,
                eig_match_freq=True, eig_parity=parity, amplitude=1.0)]
        else:
            ez_p, hy_p = handoff
            ez_pf = _yee_angular_filter(ez_p, res, kmed)   # winkelaufgeloeste Yee-Korr.
            js, ms = _equiv_signs(cfg)

            def amp_e(p, hy_p=hy_p):     # J = n x H:  s: Jz=+Hy,  p: Jy=-Hz
                return js*C*_profile_interp(hy_p, ys, p.y)

            def amp_h(p, ez_pf=ez_pf):   # M = -n x E: s: My=+Ez,  p: Mz=-Ey
                return ms*C*_profile_interp(ez_pf, ys, p.y)
            src_t = mp.GaussianSource(f0, fwidth=0.15*f0)
            sources = [
                mp.Source(src=src_t, component=comp,
                          center=mp.Vector3(x_inj, y_off, 0),
                          size=mp.Vector3(0, mode_h, 0), amp_func=amp_e),
                mp.Source(src=src_t, component=hcomp,
                          center=mp.Vector3(x_inj, y_off, 0),
                          size=mp.Vector3(0, mode_h, 0), amp_func=amp_h)]
        sim = mp.Simulation(cell_size=mp.Vector3(Wx, ly, 0), resolution=res,
                            geometry=mb.build_geometry(wc), sources=sources,
                            boundary_layers=[mp.PML(dpml)],
                            default_material=mp.Medium(index=1.0),
                            force_complex_fields=True)
        # Mode-Monitor zur Richtungs-Diagnose (vorwaerts/rueckwaerts), eingeschwungen
        mon = sim.add_mode_monitor(f0, 0, 1, mp.FluxRegion(
            center=mp.Vector3(x_inj + 1.0, y_off, 0), size=mp.Vector3(0, mode_h, 0)))
        # DFT von Ez UND Hy ueber das ganze Fenster
        dft = sim.add_dft_fields([comp, hcomp], f0, 0, 1,
                                 center=mp.Vector3(), size=mp.Vector3(Wx, ly, 0))
        sim.run(until_after_sources=mp.stop_when_fields_decayed(
            15, comp, mp.Vector3(xR, y_off, 0), 1e-4))

        cc = sim.get_eigenmode_coefficients(mon, [1], eig_parity=parity)
        a_fwd, a_bwd = complex(cc.alpha[0, 0, 0]), complex(cc.alpha[0, 0, 1])
        ez_full = np.asarray(sim.get_dft_array(dft, comp, 0))    # (nx, ny)
        hy_full = np.asarray(sim.get_dft_array(dft, hcomp, 0))

        # Handoff-Profile am rechten Plane (Spalte bei x=xR) fuer naechstes Fenster
        ix = int(round((xR + Wx/2)*res))
        ix = min(max(ix, 0), ez_full.shape[0]-1)
        # y-Ausschnitt auf mode_h um WG-Mitte zentrieren
        jc = int(round((y_off + ly/2)*res))
        j0 = jc - len(ys)//2
        sl = slice(max(0, j0), max(0, j0)+len(ys))
        handoff = (ez_full[ix, sl].copy(), hy_full[ix, sl].copy())

        gi0 = int(round(gx0*res))
        Ox = Wx - Sx
        trim = int(round(0.5*Ox*res)) if Ox > 0 else 0
        l0 = trim if w > 0 else 0
        l1 = ez_full.shape[0] - (trim if w < n_win - 1 else 0)
        gi1 = min(gi0 + l1, Nx_full)
        ny = min(Ny, ez_full.shape[1])       # Meep liefert oft Ny+1/Ny+2 Punkte
        Efull[gi0 + l0:gi1, :ny] = ez_full[l0:l0 + (gi1 - gi0 - l0), :ny]

        if w == 0:
            afwd0 = a_fwd
        afwd_last = a_fwd
        ratio = abs(a_bwd)/max(abs(a_fwd), 1e-30)
        print(f'  [fullfield] window {w+1}/{n_win} x0={gx0:.1f}um  '
              f'|fwd|={abs(a_fwd):.3f} |bwd|={abs(a_bwd):.3f} '
              f'(bwd/fwd={ratio:.2f})', flush=True)
        del sim

    Tr = abs(afwd_last/afwd0)**2 if (afwd0 and abs(afwd0) > 1e-30) else 0.0
    print(f'[meep-stitch/fullfield Result] {cfg["label"]}: '
          f'Transmission(guided)~{Tr:.4f} ({n_win} windows, full-field based)')
    if Tr > 1.05:
        print('  !!! WARNING: Transmission > 1 is UNPHYSICAL -> the handoff field '
              'is radiation-dominated (strong scattering), the scalar calibration '
              'over-scales the radiation. Result NOT reliable. Use '
              '--meep-handoff mode or --engine native --method stitch. !!!')
    return dict(Efull=Efull, transmission=Tr, n_win=n_win, shape=(Nx_full, Ny))


def mode_transfer_full(cfg):
    """Referenz: EIN Full-Domain-Lauf, Mode-Transfer |a_R/a_L|^2 zwischen zwei
    Planes nahe den Enden (fairer Vergleich zum Stitch, gleiche Groesse)."""
    comp = mp.Ez if cfg['pol'] == 's' else mp.Ey
    parity = _parity(cfg)
    f0 = cfg['f0']
    lx, ly, res = cfg['lx'], cfg['ly'], cfg['resolution']
    dpml = max(0.1, min(cfg['lam_nm']/1000.0, 0.4*min(lx, ly)))
    y_off = cfg['t_wg']/2 - cfg['cy']
    mode_h = min(ly - 2*dpml, cfg['t_wg'] + 4.0)
    xL = -lx/2 + dpml + 0.5
    xR = lx/2 - dpml - 0.5
    src = mp.EigenModeSource(
        src=mp.GaussianSource(f0, fwidth=0.15*f0),
        center=mp.Vector3(-lx/2 + dpml + 0.2, y_off, 0),
        size=mp.Vector3(0, mode_h, 0), eig_band=1, direction=mp.X,
        eig_match_freq=True, eig_parity=parity, amplitude=1.0)
    sim = mp.Simulation(cell_size=mp.Vector3(lx, ly, 0), resolution=res,
                        geometry=mb.build_geometry(cfg), sources=[src],
                        boundary_layers=[mp.PML(dpml)],
                        default_material=mp.Medium(index=1.0),
                        force_complex_fields=True)
    monL = sim.add_mode_monitor(f0, 0, 1, mp.FluxRegion(
        center=mp.Vector3(xL, y_off, 0), size=mp.Vector3(0, mode_h, 0)))
    monR = sim.add_mode_monitor(f0, 0, 1, mp.FluxRegion(
        center=mp.Vector3(xR, y_off, 0), size=mp.Vector3(0, mode_h, 0)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        15, comp, mp.Vector3(xR, y_off, 0), 1e-4))
    aL = _fwd(sim, monL, parity)
    aR = _fwd(sim, monR, parity)
    return abs(aR/aL)**2, xR - xL


# ---- Standalone-Test: Stitch vs Full-Domain Mode-Transfer ----
if __name__ == '__main__':
    import os
    import sys
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _p in (_ROOT, os.path.dirname(os.path.abspath(__file__))):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import run_simulation as rs
    from run_meep import _build_cfg

    args = rs.build_parser().parse_args()
    args.dim = 2
    layers = rs.resolve_args(args)
    cfg = _build_cfg(args, layers)
    print('=== STITCH ===')
    st = run_stitch(cfg, args.window_um, args.slide_um, modes=args.meep_modes)
    print('=== FULL-DOMAIN mode transfer (reference) ===')
    Tf, span = mode_transfer_full(cfg)
    print(f'\n[COMPARISON] stitch T(mode)={st["transmission"]:.4f}  '
          f'vs full-domain T(mode over {span:.1f}um)={Tf:.4f}')


# =============================================================================
# Vektor-Stitch (2D/3D, planar, Linse, CAD): Aequivalenzprinzip mit ALLEN
# tangentialen Komponenten (Ey, Ez, Hy, Hz) - unabhaengig von Moden und Geometrie.
# =============================================================================
_TANG = None


def _tang():
    global _TANG
    if _TANG is None:
        _TANG = (mp.Ey, mp.Ez, mp.Hy, mp.Hz)
    return _TANG


def _window_cfg_v(cfg, gx0, Wx):
    """Fenster-cfg: Zelle [gx0, gx0+Wx]; Bead nur wenn (teilweise) drin; CAD-Gitter
    auf das Fenster zugeschnitten; gekruemmte Linse ueber gx0 (Gitter global)."""
    wc = dict(cfg)
    wc['lx'] = Wx
    wc['cx'] = gx0 + Wx/2.0
    wc['gx0'] = gx0
    wc['lx_total'] = cfg.get('lx_total', cfg['lx'])
    b = cfg.get('bead')
    wc['bead'] = b if (b and gx0 - b['r'] <= b['x'] <= gx0 + Wx + b['r']) else None
    if cfg.get('cad'):
        c = dict(cfg['cad'])
        dx = c['dx']
        i0 = int(round(gx0/dx)); n = max(2, int(round(Wx/dx)))
        e = c['eps'][i0:i0 + n]
        if e.shape[0] < n:                       # am Ende auffuellen (letzte Spalte)
            pad = np.repeat(e[-1:], n - e.shape[0], axis=0)
            e = np.concatenate([e, pad], axis=0)
        c['eps'] = e
        wc['cad'] = c
    return wc


def _plane_size(cfg, d):
    is3d = cfg['dim'] == 3
    return mp.Vector3(0, cfg['ly'] - 2*d, (cfg['lz'] - 2*d) if is3d else 0)


def _nearest(arr, frac):
    """Wert eines 1D/2D-Profils bei relativer Lage frac (0..1 je Achse)."""
    idx = tuple(min(max(int(round(f*(n - 1))), 0), n - 1) for f, n in zip(frac, arr.shape))
    return arr[idx]


def _sx_flux(ph, res, is3d):
    """Netto-Fluss in +x aus tangentialen Phasoren: S_x = 1/2 Re(Ey Hz* - Ez Hy*)."""
    s = 0.5*np.real(ph[mp.Ey]*np.conj(ph[mp.Hz]) - ph[mp.Ez]*np.conj(ph[mp.Hy]))
    return float(np.sum(s))*(1.0/res)**(2 if is3d else 1)


def run_stitch_vector(cfg, window_w_um, slide_um, n_dft_periods=4):
    """CW-Fenster-Stitch fuer Meep, 2D und 3D, planar/Linse/CAD (Pendant zum nativen
    Stitch). Fenster 0: physikalische Quelle (Gauss-Spot). Fenster w>0: Einspeisung des
    eingeschwungenen Feldes am Uebergabe-Querschnitt als Flaechenstroeme
        J = x^ x H = (0, -Hz, Hy),   M = -x^ x E = (0, Ez, -Ey)
    (Aequivalenzprinzip -> strahlt nur in +x). Meeps Normierung der Stroeme wird einmal
    im Ueberlapp von Fenster 0/1 kalibriert (komplexer Faktor C, linear -> exakt fuer
    alle weiteren Fenster). Ein Fenster im Speicher. NOCH NICHT gegen Meep-full validiert."""
    is3d = cfg['dim'] == 3
    comp = mp.Ez if cfg['pol'] == 's' else mp.Ey
    f0 = cfg['f0']; res = cfg['resolution']
    lx = cfg['lx']
    d = cfg.get('dpml') or cfg['lam_nm']/1000.0
    Wx = min(window_w_um, lx)
    if Wx < 2*d + 3.0 and Wx < lx:
        raise SystemExit(f'[meep-stitch] window {Wx:g} um too small (>= 2*PML + 3 um = {2*d + 3:g}).')
    x_inj = -Wx/2 + d + 0.3                      # Einspeise-Ebene (lokal)
    Sx = max(0.5, min(slide_um, Wx - 2*d - 2.0)) if lx > Wx else lx
    # Slide auf ganze Zellen: die Fensterstarts werden aufs Gitter gerundet; ist Sx kein
    # Vielfaches von dx, liegt die Einspeiseebene bis dx/2 neben der Uebergabeebene des
    # Vorgaengers -> Phasensprung beta*dx/2 an jeder Naht.
    Sx = max(1, int(round(Sx*res)))/res
    n_win = (int(np.ceil((lx - Wx)/Sx)) + 1) if lx > Wx else 1
    xR = x_inj + Sx                               # Uebergabe-Ebene (lokal)
    n_max = max(float(cfg['wg_n']), 1.5)
    T_win = 3.0*Wx*n_max + 10.0/f0
    T_dft = n_dft_periods/f0
    Nx_full = int(round(lx*res))
    Efull = None              # y/z-Groesse aus den Meep-Gitterkoordinaten des 1. Fensters
    x_off = 0.0               # Efull[i] liegt bei x = base + i/res (interpoliert)
    print(f'[meep-stitch/vector] {n_win} windows of {Wx:g} um, slide {Sx:.2f} um, '
          f'{"3D" if is3d else "2D"}, injection J=n x H, M=-n x E (Ey,Ez,Hy,Hz)', flush=True)

    C = 1.0 + 0j
    handoff = None
    prev = None                                   # (gx0, Evol, x-Koordinaten lokal) Fenster w-1
    S_in = S_out = None
    base = float(cfg.get('gx0', 0.0))            # native x-Aufteilung: Zelle beginnt bei 10dx - dpml
    for w in range(n_win):
        # Fenster immer um genau Sx weiter: die Einspeiseebene von Fenster w MUSS auf der
        # Uebergabeebene xR von Fenster w-1 liegen. Das letzte Fenster darf daher ueber
        # das Domaenenende hinausragen (Geometrie laeuft weiter, Rest wird verworfen).
        gx0 = base + w*Sx
        gx0 = round(gx0*res)/res                  # Fenster auf dem Gitter ausrichten
        wc = _window_cfg_v(cfg, gx0, Wx)
        psize = _plane_size(wc, d)
        if w == 0:
            wc0 = dict(wc); wc0['source_type'] = 'cw'
            sources = mb.build_source(wc0)
        else:
            src_t = mp.ContinuousSource(frequency=f0, width=5.0/f0)
            hy, hz, ey, ez = (handoff[mp.Hy], handoff[mp.Hz], handoff[mp.Ey], handoff[mp.Ez])
            hy_, hz_ = psize.y, psize.z

            def mk(arr, sign):
                def f(p, arr=arr, sign=sign):
                    fr = [(p.y + hy_/2)/max(hy_, 1e-12)]
                    if is3d:
                        fr.append((p.z + hz_/2)/max(hz_, 1e-12))
                    return complex(sign*_nearest(arr, fr))
                return f
            cen = mp.Vector3(x_inj, 0, 0)
            sources = [mp.Source(src=src_t, component=mp.Ey, center=cen, size=psize, amp_func=mk(hz, -1.0)),
                       mp.Source(src=src_t, component=mp.Ez, center=cen, size=psize, amp_func=mk(hy, +1.0)),
                       mp.Source(src=src_t, component=mp.Hy, center=cen, size=psize, amp_func=mk(ez, +1.0)),
                       mp.Source(src=src_t, component=mp.Hz, center=cen, size=psize, amp_func=mk(ey, -1.0))]
        kw = {}
        g = mb._grid(wc)
        if g is not None:
            h5 = mb._eps_file(g)
            kw = {'epsilon_input_file': h5} if h5 else {'default_material': mb._grid_material(wc, g)}
        else:
            kw = {'default_material': mp.Medium(index=1.0)}
        cell = mp.Vector3(Wx, wc['ly'], wc['lz'] if is3d else 0)
        sim = mp.Simulation(cell_size=cell, resolution=res, geometry=mb.build_geometry(wc),
                            sources=sources, boundary_layers=[mp.PML(d)],
                            force_complex_fields=True, **kw)
        sim.run(until=T_win)                       # einschwingen
        # DFT nur ueber die letzten Perioden (eingeschwungen)
        dft_vol = sim.add_dft_fields([comp], f0, 0, 1, center=mp.Vector3(), size=cell)
        dft_R = sim.add_dft_fields(list(_tang()), f0, 0, 1, center=mp.Vector3(xR, 0, 0), size=psize)
        x_in_l = cfg['det_in_x'] - wc['cx']; x_out_l = cfg['det_out_x'] - wc['cx']
        lo_ok = -Wx/2 + d + 0.4; hi_ok = Wx/2 - d - 0.2
        dft_in = (sim.add_dft_fields(list(_tang()), f0, 0, 1, center=mp.Vector3(x_in_l, 0, 0),
                                     size=psize) if (S_in is None and lo_ok <= x_in_l <= hi_ok
                                                     and (w == 0 or x_in_l > x_inj + 0.3)) else None)
        dft_out = (sim.add_dft_fields(list(_tang()), f0, 0, 1, center=mp.Vector3(x_out_l, 0, 0),
                                      size=psize) if lo_ok <= x_out_l <= hi_ok else None)
        _t0 = sim.meep_time()
        sim.run(until=T_dft)
        # Meep-DFT = 1/sqrt(2 pi) * sum f e^{i w t} dt -> Amplitude A = DFT*sqrt(2 pi)/Dauer
        # (tatsaechliche Dauer: ganze Zeitschritte), damit Stitch und Full gleich skaliert sind
        _nrm = np.sqrt(2*np.pi)/max(sim.meep_time() - _t0, 1e-12)
        Ev = np.asarray(sim.get_dft_array(dft_vol, comp, 0))*_nrm
        phR = {c: np.asarray(sim.get_dft_array(dft_R, c, 0))*_nrm for c in _tang()}
        # exakte Gitterkoordinaten (lokal) statt linspace (Meep liefert je nach Region
        # n oder n+1/n+2 Punkte -> gestreckte Achse = Phasenfehler am Rand)
        xs_l, ys_l, zs_l, _w = sim.get_array_metadata(dft_cell=dft_vol)
        xs_l = np.asarray(xs_l)
        if Efull is None:
            ys_l = np.asarray(ys_l)
            shp_full = (Nx_full, len(ys_l)) + ((len(zs_l),) if is3d else ())
            Efull = np.zeros(shp_full, dtype=np.complex64)
            cfg['view_yr'] = (float(ys_l[0] + wc['cy']), float(ys_l[-1] + wc['cy']))

        # Kalibrierung (einmal): Fenster 1 gegen Fenster 0 im Ueberlapp
        if w == 1 and prev is not None:
            g0, E0, xs0 = prev
            # globale Lagen: Fenster [gx0, gx0+Wx], lokale x relativ zur Fenstermitte
            a = gx0 + Wx/2 + x_inj + 0.5           # hinter der Einspeisung von Fenster 1
            b = g0 + Wx - d - 0.3                  # vor dem rechten PML von Fenster 0
            i1 = (xs_l + gx0 + Wx/2 >= a) & (xs_l + gx0 + Wx/2 <= b)
            if i1.sum() >= 3:
                xg = xs_l[i1] + gx0 + Wx/2
                i0 = np.clip(np.round((xg - g0 - Wx/2 - xs0[0])/(xs0[1] - xs0[0])).astype(int),
                             0, len(xs0) - 1)
                e1 = Ev[i1]; e0 = E0[i0]
                C = complex(np.vdot(e1, e0)/max(np.vdot(e1, e1).real, 1e-30))
                print(f'  [vector] calibration C = {abs(C):.4g} * exp(i {np.angle(C):+.3f}) '
                      f'(overlap {b - a:.2f} um)', flush=True)
            else:
                print('  [vector] WARNING: no overlap for the calibration -> C = 1')
        if w >= 1:
            Ev = Ev*C
            phR = {c: v*C for c, v in phR.items()}
        handoff = phR
        # Fluesse (Transmission wie native: ganzer Querschnitt)
        if dft_in is not None and S_in is None:
            ph = {c: np.asarray(sim.get_dft_array(dft_in, c, 0))*_nrm*(C if w else 1) for c in _tang()}
            S_in = _sx_flux(ph, res, is3d)
        if dft_out is not None:
            ph = {c: np.asarray(sim.get_dft_array(dft_out, c, 0))*_nrm*(C if w else 1) for c in _tang()}
            S_out = _sx_flux(ph, res, is3d)
        # Assemblieren: Fenster 0 ab x=0, sonst ab der Einspeisung; Ende vor der PML
        a_l = -Wx/2 if w == 0 else x_inj + 0.2
        b_l = Wx/2 if w == n_win - 1 else Wx/2 - d - 0.2
        # Linear auf das globale Gitter Efull[i] <-> x = base + i/res interpolieren: Meeps
        # DFT-Punkte liegen nicht exakt im Abstand 1/res, Runden liesse Spalten leer.
        p = (xs_l + gx0 + Wx/2 - base)*res               # Gleitkomma-Index je Meep-Punkt
        pa = (a_l + gx0 + Wx/2 - base)*res; pb = (b_l + gx0 + Wx/2 - base)*res
        t = np.arange(max(0, int(np.ceil(max(pa, p[0])))),
                      min(Nx_full - 1, int(np.floor(min(pb, p[-1])))) + 1)
        if t.size:
            k = np.clip(np.searchsorted(p, t), 1, len(p) - 1)
            wt = ((t - p[k - 1])/np.maximum(p[k] - p[k - 1], 1e-12)).astype(np.float32)
            wt = wt.reshape((-1,) + (1,)*(Ev.ndim - 1))
            blk = Ev[k - 1]*(1 - wt) + Ev[k]*wt
            ny_c = min(blk.shape[1], Efull.shape[1])
            if is3d:
                nz_c = min(blk.shape[2], Efull.shape[2])
                Efull[t, :ny_c, :nz_c] = blk[:, :ny_c, :nz_c]
            else:
                Efull[t, :ny_c] = blk[:, :ny_c]
        prev = (gx0, Ev, xs_l)
        print(f'  [vector] window {w+1}/{n_win} x0={gx0:.2f} um done', flush=True)
        del sim

    if cfg.get('view_x'):                         # Ausgabe auf [0, L] wie native
        vx0, vx1 = cfg['view_x']
        i0 = int(round((vx0 - base)*res)); n = int(round((vx1 - vx0)*res))
        Efull = Efull[i0:i0 + n]
        shp_full = Efull.shape
        cfg['view_xr'] = (base + (i0 + x_off)/res, base + (i0 + x_off + Efull.shape[0] - 1)/res)
    else:
        cfg['view_xr'] = (base + x_off/res, base + (x_off + Efull.shape[0] - 1)/res)
    Tr = (S_out/S_in) if (S_in and S_out is not None and abs(S_in) > 0) else float('nan')
    print(f'[meep-stitch/vector Result] {cfg["label"]}: Transmission (flux, full '
          f'cross-section) = {Tr:.4f}  ({n_win} windows)', flush=True)
    return dict(Efull=Efull, transmission=Tr, n_win=n_win, shape=shp_full, C=C)
