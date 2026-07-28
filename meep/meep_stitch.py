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
