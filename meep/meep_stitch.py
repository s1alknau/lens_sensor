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
    parity = mp.NO_PARITY          # 2D-Mode: Meep die Mode selbst finden lassen
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
        raise SystemExit('[meep-stitch] Fenster zu klein fuer den Slide: '
                         '--window-um groesser oder --slide-um kleiner waehlen.')
    n_win = (int(np.ceil((lx - Wx)/Sx)) + 1) if lx > Wx else 1

    Nx_full = int(round(lx*res))
    Ny = int(round(ly*res))
    Efull = np.zeros((Nx_full, Ny), dtype=np.complex64)

    print(f'[meep-stitch] {n_win} Fenster a {Wx:g}um, Transfer-Slide {Sx:.2f}um, '
          f'{M} Mode(n), Grid/Fenster ~{int(Wx*res)}x{Ny}')
    A = np.zeros(M, dtype=complex)
    A[0] = 1.0             # Eingang: Fundamentalmode, Amplitude 1
    Ttot = np.eye(M, dtype=complex)
    for w in range(n_win):
        gx0 = min(w*Sx, max(0.0, lx - Wx))
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
        Efull[gi0 + l0:gi1, :fld.shape[1]] = fld[l0:l0 + (gi1 - gi0 - l0), :]

        A = Tcol @ A
        Ttot = Tcol @ Ttot
        print(f'  [meep-stitch] Fenster {w+1}/{n_win} x0={gx0:.1f}um  '
              f'gefuehrte Leistung |A|^2={float(np.sum(np.abs(A)**2)):.3f}', flush=True)

    Tr = float(np.sum(np.abs(A)**2))    # gefuehrte Gesamtleistung am Ende (Eingang=1)
    print(f'[meep-stitch Result] {cfg["label"]}: Transmission(gefuehrt, {M} Mode(n))'
          f'~{Tr:.4f} ({n_win} Fenster, nur EIN Fenster im RAM)')
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


def _calibrate_C(cfg, Wx):
    """Bestimmt den Aequivalenz-Skalenfaktor C selbst: extrahiert in einem
    uniformen Fenster das Ez/Hy-Profil + Mode-Amplitude a_ref an einer Ebene,
    praegt es dann per Zwei-Strom-Quelle (C=1) ein und misst a_test.
    C = a_ref/a_test macht die Einpraegung amplituden- UND phasenerhaltend.
    Robust gegen Aufloesung (C haengt von der Diskretisierung ab)."""
    comp = mp.Ez if cfg['pol'] == 's' else mp.Ey
    hcomp = mp.Hy if cfg['pol'] == 's' else mp.Hx
    parity = mp.NO_PARITY
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
    src_t = mp.GaussianSource(f0, fwidth=0.15*f0)
    aprobe, _, _, P_probe, _ = _win([
        mp.Source(src=src_t, component=comp, center=mp.Vector3(x_inj, y_off, 0),
                  size=mp.Vector3(0, mode_h, 0),
                  amp_func=lambda p: A_PROBE*_profile_interp(hy_p, ys, p.y)),
        mp.Source(src=src_t, component=hcomp, center=mp.Vector3(x_inj, y_off, 0),
                  size=mp.Vector3(0, mode_h, 0),
                  amp_func=lambda p: -A_PROBE*_profile_interp(ez_pf, ys, p.y))], xP)
    Cmag = A_PROBE*np.sqrt(abs(P_target)/max(abs(P_probe), 1e-30))
    phase = np.angle(aref) - np.angle(aprobe) if abs(aprobe) > 1e-30 else 0.0
    C = Cmag*np.exp(1j*phase)
    print(f'[meep-stitch/fullfield] Auto-Kalibrierung |C|={Cmag:.3g} '
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
    print('  [voll-feldbasierter Handoff] Aequivalenzprinzip (Ez+Hy), '
          'fluss-selbstkalibriert + Yee-Offset-phasenkorrigiert -> '
          'aufloesungsrobust (nm=15..60) fuer GEFUEHRT-DOMINIERTE Felder.')
    print('  !!! WICHTIG: Die skalare Kalibrierung ist auf die Grundmode getunt. '
          'Bei STARKER Streuung (radiation-lastiges Handoff-Feld, z.B. Bead im '
          'WG-Kern) skaliert sie die Strahlung falsch -> UNPHYSIKALISCH (Amplitude '
          'kann explodieren). Dann --meep-handoff mode (Multi-Mode) oder nativen '
          'Stitch nutzen. !!!')
    comp = mp.Ez if cfg['pol'] == 's' else mp.Ey
    hcomp = mp.Hy if cfg['pol'] == 's' else mp.Hx
    parity = mp.NO_PARITY
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

    print(f'[meep-stitch/fullfield] {n_win} Fenster a {Wx:g}um, Slide {Sx:.2f}um, C={C}')
    handoff = None      # (Ez_profile, Hy_profile) am Handoff-Plane des Vorgaengers
    afwd0 = None; afwd_last = None
    for w in range(n_win):
        gx0 = min(w*Sx, max(0.0, lx - Wx))
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
            def amp_e(p, hy_p=hy_p):     # elektrische Stromquelle (treibt Ez) ~ +C*Hy
                return C*_profile_interp(hy_p, ys, p.y)

            def amp_h(p, ez_pf=ez_pf):   # magnetische Stromquelle (treibt Hy) ~ -C*Ez
                return -C*_profile_interp(ez_pf, ys, p.y)
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
        Efull[gi0 + l0:gi1, :ez_full.shape[1]] = ez_full[l0:l0 + (gi1 - gi0 - l0), :]

        if w == 0:
            afwd0 = a_fwd
        afwd_last = a_fwd
        ratio = abs(a_bwd)/max(abs(a_fwd), 1e-30)
        print(f'  [fullfield] Fenster {w+1}/{n_win} x0={gx0:.1f}um  '
              f'|fwd|={abs(a_fwd):.3f} |bwd|={abs(a_bwd):.3f} '
              f'(bwd/fwd={ratio:.2f})', flush=True)
        del sim

    Tr = abs(afwd_last/afwd0)**2 if (afwd0 and abs(afwd0) > 1e-30) else 0.0
    print(f'[meep-stitch/fullfield Result] {cfg["label"]}: '
          f'Transmission(gefuehrt)~{Tr:.4f} ({n_win} Fenster, voll-feldbasiert)')
    if Tr > 1.05:
        print('  !!! WARNUNG: Transmission > 1 ist UNPHYSIKALISCH -> das Handoff-Feld '
              'ist radiation-dominiert (starke Streuung), die skalare Kalibrierung '
              'ueber-skaliert die Strahlung. Ergebnis NICHT belastbar. Nutze '
              '--meep-handoff mode oder --engine native --method stitch. !!!')
    return dict(Efull=Efull, transmission=Tr, n_win=n_win, shape=(Nx_full, Ny))


def mode_transfer_full(cfg):
    """Referenz: EIN Full-Domain-Lauf, Mode-Transfer |a_R/a_L|^2 zwischen zwei
    Planes nahe den Enden (fairer Vergleich zum Stitch, gleiche Groesse)."""
    comp = mp.Ez if cfg['pol'] == 's' else mp.Ey
    parity = mp.NO_PARITY
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
    print('=== FULL-DOMAIN Mode-Transfer (Referenz) ===')
    Tf, span = mode_transfer_full(cfg)
    print(f'\n[VERGLEICH] stitch T(mode)={st["transmission"]:.4f}  '
          f'vs full-domain T(mode ueber {span:.1f}um)={Tf:.4f}')
