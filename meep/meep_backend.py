"""Meep-Backend: baut dieselbe Geometrie wie der eigene Solver in Meep-Objekten,
laesst sie rechnen und liefert Feld-Frames + Transmission zurueck.

Laeuft nur unter Linux/WSL (Meep ist Linux-only). Einheiten: a = 1 um, c = 1,
Frequenz f = 1/lambda[um]. Koordinaten (wie eigener Solver):
  x = Propagation, y = Schichtstapel (y=0 = WG-Unterseite), z = Tiefe (3D).
Meep rechnet zentriert -> phys. Punkt p wird um den Domain-Mittelpunkt versetzt.
"""
import os
import sys
import numpy as np
import meep as mp

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _layer_bounds(cfg):
    """y-Grenzen (phys) der Tear-Schichten unter dem WG (y<0)."""
    t_lip = cfg['t_lip'] or 0.0
    y_lip_bot = -t_lip
    y_aq_bot = y_lip_bot - cfg['t_aq']
    y_mu_bot = y_aq_bot - cfg['t_mu']
    return y_lip_bot, y_aq_bot, y_mu_bot


def _grid(cfg):
    """eps-Gitter, das die ganze Zelle beschreibt (CAD oder gekruemmte Linse), sonst None.
    Rueckgabe dict(eps, dx, y0)."""
    if cfg.get('cad'):
        c = cfg['cad']
        return dict(eps=c['eps'], dx=c['dx'], y0=c['y0'])
    if cfg.get('curved'):
        return _curved_grid(cfg)
    return None


def _layer_eps_y(cfg, y, core):
    """Schichtstapel eps(y) (phys. um) wie native, mit oder ohne Kern."""
    y_lip_bot, y_aq_bot, y_mu_bot = _layer_bounds(cfg)
    e = np.full(y.shape, 1.0)
    mc = y < y_mu_bot
    n_co = cfg['n_co']
    if hasattr(n_co, 'at_depth'):
        e[mc] = n_co.at_depth(y_mu_bot - y[mc])**2
    else:
        e[mc] = float(n_co)**2
    e[(y >= y_mu_bot) & (y < y_aq_bot)] = cfg['n_mu']**2
    e[(y >= y_aq_bot) & (y < y_lip_bot)] = cfg['n_aq']**2
    if (cfg['t_lip'] or 0.0) > 1e-6:
        e[(y >= y_lip_bot) & (y < 0.0)] = cfg['n_lip']**2
    if core:
        e[(y >= 0.0) & (y < cfg['t_wg'])] = cfg['wg_n']**2
    return e


def _curved_grid(cfg):
    """Gekruemmte Kontaktlinse (--curved) wie native: konforme Abbildung des Bogens auf
    einen geraden Leiter, n_eq = n*(1 + u/R), u = y - Kernmitte (u > 0 aussen).
    -> eps(y)*(1+u/R)^2 als Gitter ueber die ganze Zelle (Meep: epsilon_input_file)."""
    dx = 1.0/cfg['resolution']
    nx = max(2, int(round(cfg['lx']/dx))); ny = max(2, int(round(cfg['ly']/dx)))
    ys = cfg['y_bot'] + (np.arange(ny) + 0.5)*dx
    R = float(cfg.get('r_bend_um') or 8300.0)
    metric = (1.0 + (ys - cfg['t_wg']/2.0)/R)**2
    col_out = _layer_eps_y(cfg, ys, False)*metric
    col_core = _layer_eps_y(cfg, ys, True)*metric
    xs = cfg.get('gx0', 0.0) + (np.arange(nx) + 0.5)*dx      # Fenster: globale x-Lage
    x0 = (cfg['src_x'] + cfg['input_gap']) if (cfg.get('input_gap') or 0) > 0 else -1e30
    x1 = (cfg.get('lx_total', cfg['lx']) - cfg['end_facet']) if (cfg.get('end_facet') or 0) > 0 \
        else 1e30
    inwg = (xs >= x0) & (xs <= x1)
    eps2 = np.where(inwg[:, None], col_core[None, :], col_out[None, :])
    if cfg['dim'] == 3:
        nz = max(2, int(round(cfg['lz']/dx)))
        eps2 = np.repeat(eps2[:, :, None], nz, axis=2)
    return dict(eps=eps2, dx=dx, y0=cfg['y_bot'])


def _eps_file(g):
    """eps-Gitter als HDF5 fuer Meeps epsilon_input_file (deckt die ganze Zelle,
    Meep interpoliert). Rueckgabe 'pfad:eps' oder None (dann Materialfunktion)."""
    try:
        import h5py
    except Exception:
        return None
    import tempfile
    path = os.path.join(tempfile.gettempdir(), f"meep_eps_{os.getpid()}.h5")
    with h5py.File(path, 'w') as f:
        f.create_dataset('eps', data=np.asarray(g['eps'], dtype=np.float64))
    return path + ':eps'


def _grid_material(cfg, g):
    """Fallback ohne h5py: eps per naechstem Gitterpunkt (langsamer)."""
    eps = g['eps']; dx = g['dx']; y0 = g['y0']
    is3d = cfg['dim'] == 3

    def f(p):
        i = min(max(int(round((p.x + cfg['lx']/2.0)/dx)), 0), eps.shape[0] - 1)   # lokal (Fenster)
        j = min(max(int(round((p.y + cfg['cy'] - y0)/dx)), 0), eps.shape[1] - 1)
        if is3d:
            k = min(max(int(round((p.z + cfg['lz']/2)/dx)), 0), eps.shape[2] - 1)
            return mp.Medium(epsilon=float(eps[i, j, k]))
        return mp.Medium(epsilon=float(eps[i, j]))
    return f


def _med(cfg, n, key):
    """Meep-Medium fuer Brechzahl n und Absorption kappa = cfg['kappa'][key]:
    eps = (n + i kappa)^2 -> eps' = n^2 - kappa^2, Meep-D-Leitfaehigkeit
    sigma_D = omega0 * eps''/eps' (Meep: eps(w) = eps' (1 + i sigma_D/w)) - exakt bei f0."""
    k = float((cfg.get('kappa') or {}).get(key, 0.0))
    if k <= 0:
        return mp.Medium(index=n)
    e1 = n*n - k*k
    return mp.Medium(epsilon=e1, D_conductivity=2*np.pi*cfg['f0']*(2*n*k)/e1)


def build_geometry(cfg):
    """Liste von Meep-Objekten. Hintergrund = Luft; Bloecke fuer WG + Tear-Stack,
    optional Bead (2D Zylinder / 3D Kugel). CAD: leer (eps kommt aus dem Gitter)."""
    if cfg.get('cad'):
        return []
    if cfg.get('curved'):                        # Schichten/Kern kommen aus dem Gitter
        g = []
        b = cfg.get('bead')
        if b:
            cy, cx = cfg['cy'], cfg['cx']
            center = mp.Vector3(b['x'] - cx, b['y'] - cy, b.get('z', 0.0))
            R = float(cfg.get('r_bend_um') or 8300.0)
            nb = b['n']*(1.0 + (b['y'] - cfg['t_wg']/2.0)/R)   # gleiche Abbildung wie native
            if cfg['dim'] == 3:
                g.append(mp.Sphere(radius=b['r'], center=center, material=_med(cfg, nb, 'bead')))
            else:
                g.append(mp.Cylinder(radius=b['r'], height=mp.inf, axis=mp.Vector3(0, 0, 1),
                                     center=center, material=_med(cfg, nb, 'bead')))
        return g
    cy, cx = cfg['cy'], cfg['cx']
    lx, lz = cfg['lx'], cfg['lz']
    is3d = cfg['dim'] == 3
    x_inf = mp.inf
    z_size = lz if is3d else mp.inf

    def blk(y0, y1, n, key=None):
        yc = 0.5*(y0 + y1) - cy
        return mp.Block(size=mp.Vector3(x_inf, y1 - y0, z_size),
                        center=mp.Vector3(0, yc, 0),
                        material=_med(cfg, n, key))

    g = []
    y_lip_bot, y_aq_bot, y_mu_bot = _layer_bounds(cfg)
    y_bot = cfg['y_bot']
    # Cornea (Substrat) ganz unten, dann Mucin, Aqueous, (Lipid), WG-Kern.
    # Geschichtete Cornea (common.physics.CorneaProfile): als Folge von Bloecken,
    # Epithel/Bowman/Endothel konstant, Stroma (linear) in 20-um-Stufen.
    n_co = cfg['n_co']
    if hasattr(n_co, 'at_depth') and y_mu_bot > y_bot:
        from common.physics import CORNEA_LAYERS_UM as L
        b0 = L['epithelium']; b1 = b0 + L['bowman']
        edges = sorted(set([0.0, b0, b1, L['stroma_end'], L['endothelium_end']]
                           + list(np.arange(b1, L['stroma_end'], 20.0))))
        depth_max = y_mu_bot - y_bot
        edges = [e for e in edges if e < depth_max] + [max(depth_max, edges[-1] + 1.0)
                                                       if depth_max > edges[-1] else depth_max]
        for d0, d1 in zip(edges[:-1], edges[1:]):
            if d1 <= d0:
                continue
            nseg = float(n_co.at_depth(0.5*(d0 + d1)))
            g.append(blk(y_mu_bot - d1, y_mu_bot - d0, nseg, 'co'))
    else:
        g.append(blk(y_bot, y_mu_bot, float(n_co), 'co'))
    g.append(blk(y_mu_bot, y_aq_bot, cfg['n_mu'], 'mu'))
    g.append(blk(y_aq_bot, y_lip_bot, cfg['n_aq'], 'aq'))
    if (cfg['t_lip'] or 0.0) > 1e-6:
        g.append(blk(y_lip_bot, 0.0, cfg['n_lip'], 'lip'))
    # WG-Kern y in [0, t_wg] wie native: Einkoppelabstand (Luft zwischen Quelle und
    # WG-Eintritt) und optionale Endfacette (Luftzone am Ausgang) -> endlicher Block in x
    # (gilt auch fuer den 3D-Rechteck-Kanal; frueher dort ignoriert)
    x0 = (cfg['src_x'] + cfg['input_gap']) if (cfg.get('input_gap') or 0) > 0 else None
    x1 = (cfg.get('lx_total', lx) - cfg['end_facet']) if (cfg.get('end_facet') or 0) > 0 else None
    if is3d and cfg.get('wg_width'):
        w = cfg['wg_width']
        if x0 is None and x1 is None:
            g.append(mp.Block(size=mp.Vector3(x_inf, cfg['t_wg'], w),
                              center=mp.Vector3(0, cfg['t_wg']/2 - cy, 0),
                              material=_med(cfg, cfg['wg_n'], 'wg')))
        else:
            xa = x0 if x0 is not None else -lx
            xb = x1 if x1 is not None else 2*lx
            g.append(mp.Block(size=mp.Vector3(xb - xa, cfg['t_wg'], w),
                              center=mp.Vector3(0.5*(xa + xb) - cx, cfg['t_wg']/2 - cy, 0),
                              material=_med(cfg, cfg['wg_n'], 'wg')))
    else:
        if x0 is None and x1 is None:
            g.append(blk(0.0, cfg['t_wg'], cfg['wg_n'], 'wg'))
        else:
            xa = x0 if x0 is not None else -lx
            xb = x1 if x1 is not None else 2*lx
            g.append(mp.Block(size=mp.Vector3(xb - xa, cfg['t_wg'], z_size),
                              center=mp.Vector3(0.5*(xa + xb) - cx, cfg['t_wg']/2 - cy, 0),
                              material=_med(cfg, cfg['wg_n'], 'wg')))
    # Bead
    b = cfg.get('bead')
    if b:
        center = mp.Vector3(b['x'] - cx, b['y'] - cy, b.get('z', 0.0))
        if is3d:
            g.append(mp.Sphere(radius=b['r'], center=center,
                               material=_med(cfg, b['n'], 'bead')))
        else:
            g.append(mp.Cylinder(radius=b['r'], height=mp.inf,
                                 axis=mp.Vector3(0, 0, 1), center=center,
                                 material=_med(cfg, b['n'], 'bead')))
    return g


def build_source(cfg):
    """Gauss-Spot (Taille y, in 3D auch z) an x=src_x auf der getriebenen Komponente
    (Ez fuer s/TE, Ey fuer p/TM). CW oder Puls."""
    f0 = cfg['f0']
    comp = mp.Ez if cfg['pol'] == 's' else mp.Ey
    is3d = cfg['dim'] == 3
    wy = max(cfg['vcsel_waist'], 2.0/cfg['resolution'])
    wz = max(cfg.get('vcsel_waist_z', cfg['vcsel_waist']), 2.0/cfg['resolution'])
    # Spot-Mitte wie native: WG-Mitte + Versatz (y, in 3D auch z)
    y_c = cfg['src_y'] if 'src_y' in cfg else cfg['t_wg']/2   # CAD: auf die Eintrittsflaeche
    y_off = y_c + (cfg.get('vcsel_offset') or 0.0) - cfg['cy']
    z_off = ((cfg.get('src_z') or 0.0) + (cfg.get('vcsel_offset_z') or 0.0)) if is3d else 0.0
    tilt = (cfg.get('vcsel_tilt') or 0.0) + (cfg.get('cad_tilt') or 0.0)
    # Neigung wie native: Winkel im MEDIUM an der Quelle (k0*n*sin). Kern, wenn die Quelle
    # ohne Einkoppelabstand im Kern sitzt; sonst Luft (CAD: Quelle vor der Facette).
    y_abs_c = y_c + (cfg.get('vcsel_offset') or 0.0)
    in_core = ('src_y' not in cfg and not (cfg.get('input_gap') or 0) > 0
               and 0.0 <= y_abs_c < cfg['t_wg'])
    n_src = float(cfg['wg_n']) if in_core else 1.0
    ky = 2*np.pi*f0*n_src*np.sin(np.radians(tilt))     # Neigung: Phasenrampe (wie native)
    core_only = bool(cfg.get('source_core_only')) and 'src_y' not in cfg
    if cfg.get('source_core_only') and not core_only:
        print('[meep] Note: --source-core-only applies to the layer stack only, not to CAD')
    from common.physics import core_source_mask

    def amp(p):
        a = np.exp(-((p.y)/wy)**2)       # p ist relativ zum Quell-Zentrum
        if is3d:
            a *= np.exp(-((p.z)/wz)**2)
        if core_only:                    # nur durch den Kern einkoppeln (weiche Kante)
            a *= float(core_source_mask(p.y + y_off + cfg['cy'], cfg['t_wg'],
                                        z_um=(p.z + z_off) if is3d else None,
                                        wg_width_um=cfg.get('wg_width') if is3d else None))
        return complex(a)*np.exp(1j*ky*p.y)

    if cfg['source_type'] == 'pulse':
        # wie native: Huellkurve exp(-(t/sp)^2), sp = Perioden/f0 -> Meep-Breite
        # exp(-(t^2)/(2 w^2)) mit w = sp/sqrt(2), fwidth = 1/w
        ncyc = float(cfg.get('pulse_cycles') or 30.0)
        src = mp.GaussianSource(frequency=f0, fwidth=np.sqrt(2.0)*f0/ncyc)
    else:
        src = mp.ContinuousSource(frequency=f0)
    size = mp.Vector3(0, cfg['ly'] - 2*cfg.get('dpml', 0.0), cfg['lz'] if is3d else 0)
    center = mp.Vector3(cfg['src_x'] - cfg['cx'], y_off, z_off)
    return [mp.Source(src=src, component=comp, center=center, size=size, amp_func=amp)]


def _flux_region(cfg, x_phys):
    """Flussebene ueber den GANZEN Querschnitt (ohne PML) - wie die native
    Poynting-Transmission (Kern + evaneszenter Auslaeufer + Abstrahlung)."""
    is3d = cfg['dim'] == 3
    d = cfg.get('dpml', 0.0)
    hy = cfg['ly'] - 2*d
    hz = (cfg['lz'] - 2*d) if is3d else 0
    return mp.FluxRegion(center=mp.Vector3(x_phys - cfg['cx'], 0, 0),
                         size=mp.Vector3(0, hy, hz))


def run(cfg, progress=True):
    """Baut die Simulation, laeuft bis (nahe) Steady-State, sammelt Frames der
    getriebenen Komponente und Transmission (Flux out/in). Rueckgabe: dict."""
    is3d = cfg['dim'] == 3
    comp = mp.Ez if cfg['pol'] == 's' else mp.Ey
    # PML adaptiv: darf nie die halbe kleinste Domainkante ueberschreiten
    # (sonst "invalid boundary absorbers"). Ziel ~1 Wellenlaenge, gedeckelt.
    dims = [cfg['lx'], cfg['ly']] + ([cfg['lz']] if is3d else [])
    dpml = cfg.get('dpml') or max(0.1, min(cfg['lam_nm']/1000.0, 0.4*min(dims)))
    cfg['dpml'] = dpml
    cell = mp.Vector3(cfg['lx'], cfg['ly'], cfg['lz'] if is3d else 0)
    _tilt = abs((cfg.get('vcsel_tilt') or 0.0) + (cfg.get('cad_tilt') or 0.0)) > 0
    kw = {}
    _g = _grid(cfg)
    if _g is not None:
        _h5 = _eps_file(_g)
        if _h5:
            kw['epsilon_input_file'] = _h5
        else:
            kw['default_material'] = _grid_material(cfg, _g)
    if 'epsilon_input_file' not in kw:
        kw.setdefault('default_material', mp.Medium(index=1.0))
    # subpixel_maxeval: Meep-Default (1e5) braucht bei einem Bead an der duennen
    # Lipidschicht (30 nm) Minuten fuer set_epsilon; 1e4 aendert eps kaum
    kw.setdefault('subpixel_maxeval', 10000)
    sim = mp.Simulation(cell_size=cell, resolution=cfg['resolution'],
                        geometry=build_geometry(cfg), sources=build_source(cfg),
                        boundary_layers=[mp.PML(dpml)],
                        force_complex_fields=(cfg['source_type'] == 'cw' or _tilt), **kw)
    is_cw = cfg['source_type'] == 'cw'
    # Puls: Fluss-DFT ueber den ganzen Lauf. CW: erst NACH dem Einschwingen (s. u.),
    # sonst zaehlt die Eingangsebene den Einschaltvorgang laenger als die
    # Ausgangsebene (Laufzeit) und T wird systematisch zu klein.
    if not is_cw:
        f_in = sim.add_flux(cfg['f0'], 0, 1, _flux_region(cfg, cfg['det_in_x']))
        f_out = sim.add_flux(cfg['f0'], 0, 1, _flux_region(cfg, cfg['det_out_x']))

    n_grp = max(cfg['wg_n'], 1.4)
    T = cfg.get('run_time') or (2.2*cfg['lx']*n_grp + 20.0)
    if not is_cw and not cfg.get('run_time'):
        # GaussianSource ist erst nach 2*cutoff/fwidth (cutoff = 5) vorbei (Peak bei
        # 5/fwidth). Ohne diese Dauer laeuft der Pulsschwanz nicht ganz durch det_out
        # -> Fluss-DFT abgeschnitten, T zu klein (30 Perioden: ~180 Zeiteinheiten).
        _fw = np.sqrt(2.0)*cfg['f0']/float(cfg.get('pulse_cycles') or 30.0)
        T += 2*5.0/_fw
    n_snap = max(1, cfg['n_snapshots'])
    interval = T/n_snap
    # Ausgabebereich: ganze Zelle oder (native x-Aufteilung) nur [0, L]
    vx0, vx1 = cfg.get('view_x') or (cfg.get('gx0', 0.0), cfg.get('gx0', 0.0) + cfg['lx'])
    size = mp.Vector3(vx1 - vx0, cfg['ly'], cfg['lz'] if is3d else 0)
    vcen = mp.Vector3(0.5*(vx0 + vx1) - cfg['cx'], 0, 0)
    frames, times = [], []

    def cap(s):
        arr = s.get_array(center=vcen, size=size, component=comp)
        frames.append(np.real(np.asarray(arr)).astype(np.float32))
        times.append(s.meep_time())
        if progress:
            print(f'  [meep] t={s.meep_time():.1f}/{T:.1f}  frame {len(frames)}/{n_snap}',
                  flush=True)

    sim.run(mp.at_every(interval, cap), until=T)
    # exakte Koordinaten der ausgegebenen Gitterpunkte (Meep liefert fuer eine Region
    # der Laenge L oft L*res+1 oder +2 Punkte -> nicht per linspace(0, L) abbilden)
    _xm, _ym, _zm, _w = sim.get_array_metadata(center=vcen, size=size)
    cfg['view_xr'] = (float(_xm[0] + cfg['cx']), float(_xm[-1] + cfg['cx']))
    cfg['view_yr'] = (float(_ym[0] + cfg['cy']), float(_ym[-1] + cfg['cy']))

    # Eingeschwungenes Feld wie native: komplexe Amplitude, Konvergenztest zwischen
    # zwei aufeinanderfolgenden Perioden (< 1 %), sonst um 0.2 T verlaengern
    # (hoechstens + 1 T). Die CW-Felder sind komplex (E_c = A e^{-i w t}); die
    # Amplitude folgt daher EXAKT aus jedem Abtastwert, auch wenn die Abtastzeiten
    # (Vielfache von dt) eine Periode nicht genau teilen (das alte DFT ueber die
    # Realteile hatte deshalb ~4 % Scheinaenderung). Native-Konvention:
    # Ê = (2/N) sum E e^{-i w t} = conj(A).
    cw_frames = None; cw_rel = float('nan')
    if is_cw:
        Tp = 1.0/cfg['f0']; Ncw = max(8, n_snap)
        om = 2*np.pi*cfg['f0']

        def phase_set():
            acc = None
            for _ in range(Ncw):
                sim.run(until=Tp/Ncw)
                a = np.conj(np.asarray(sim.get_array(center=vcen, size=size,
                                                     component=comp)).astype(np.complex128))
                a = a*np.exp(-1j*om*sim.meep_time())
                acc = a if acc is None else acc + a
            return acc/Ncw

        prev = phase_set(); extra = 0.0
        while True:
            cur = phase_set()
            cw_rel = float(np.linalg.norm(cur - prev)/max(np.linalg.norm(cur), 1e-30))
            if cw_rel <= 0.01 or extra >= T:
                break
            sim.run(until=0.2*T); extra += 0.2*T
            prev = phase_set()
        print(f'  [meep CW] phasor change over last 2 periods: {100*cw_rel:.3f}%'
              + ('' if cw_rel <= 0.01 else '  (!! max. runtime reached)'), flush=True)
        tau = np.arange(Ncw)/(Ncw*cfg['f0'])
        cw_frames = np.stack([np.real(cur*np.exp(2j*np.pi*cfg['f0']*t)) for t in tau]
                             ).astype(np.float32)
        cw_times = list(tau)
        # Fluss im eingeschwungenen Zustand (komplexes Feld -> DFT bei f0 exakt)
        f_in = sim.add_flux(cfg['f0'], 0, 1, _flux_region(cfg, cfg['det_in_x']))
        f_out = sim.add_flux(cfg['f0'], 0, 1, _flux_region(cfg, cfg['det_out_x']))
        sim.run(until=4*Tp)

    pin = float(mp.get_fluxes(f_in)[0])
    pout = float(mp.get_fluxes(f_out)[0])
    Tr = pout/pin if abs(pin) > 1e-30 else 0.0
    print(f'[meep Result] {cfg["label"]}: P_in={pin:.3e} P_out={pout:.3e} '
          f'Transmission={Tr:.4f}')
    out = dict(frames=frames, times=times, P_in=pin, P_out=pout, transmission=Tr,
               transmission_def='poynting', cw_rel_change=cw_rel,
               shape=frames[0].shape if frames else None)
    if cw_frames is not None:
        out.update(cw_frames=cw_frames, cw_times=cw_times)
    return out
