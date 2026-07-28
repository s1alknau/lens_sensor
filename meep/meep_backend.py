"""Meep-Backend: baut dieselbe Geometrie wie der eigene Solver in Meep-Objekten,
laesst sie rechnen und liefert Feld-Frames + Transmission zurueck.

Laeuft nur unter Linux/WSL (Meep ist Linux-only). Einheiten: a = 1 um, c = 1,
Frequenz f = 1/lambda[um]. Koordinaten (wie eigener Solver):
  x = Propagation, y = Schichtstapel (y=0 = WG-Unterseite), z = Tiefe (3D).
Meep rechnet zentriert -> phys. Punkt p wird um den Domain-Mittelpunkt versetzt.
"""
import numpy as np
import meep as mp


def _layer_bounds(cfg):
    """y-Grenzen (phys) der Tear-Schichten unter dem WG (y<0)."""
    t_lip = cfg['t_lip'] or 0.0
    y_lip_bot = -t_lip
    y_aq_bot = y_lip_bot - cfg['t_aq']
    y_mu_bot = y_aq_bot - cfg['t_mu']
    return y_lip_bot, y_aq_bot, y_mu_bot


def build_geometry(cfg):
    """Liste von Meep-Objekten. Hintergrund = Luft; Bloecke fuer WG + Tear-Stack,
    optional Bead (2D Zylinder / 3D Kugel)."""
    cy, cx = cfg['cy'], cfg['cx']
    lx, lz = cfg['lx'], cfg['lz']
    is3d = cfg['dim'] == 3
    x_inf = mp.inf
    z_size = lz if is3d else mp.inf

    def blk(y0, y1, n):
        yc = 0.5*(y0 + y1) - cy
        return mp.Block(size=mp.Vector3(x_inf, y1 - y0, z_size),
                        center=mp.Vector3(0, yc, 0),
                        material=mp.Medium(index=n))

    g = []
    y_lip_bot, y_aq_bot, y_mu_bot = _layer_bounds(cfg)
    y_bot = cfg['y_bot']
    # Cornea (Substrat) ganz unten, dann Mucin, Aqueous, (Lipid), WG-Kern
    g.append(blk(y_bot, y_mu_bot, cfg['n_co']))
    g.append(blk(y_mu_bot, y_aq_bot, cfg['n_mu']))
    g.append(blk(y_aq_bot, y_lip_bot, cfg['n_aq']))
    if (cfg['t_lip'] or 0.0) > 1e-6:
        g.append(blk(y_lip_bot, 0.0, cfg['n_lip']))
    # WG-Kern y in [0, t_wg]; in 3D optional lateral (z) begrenzter Kanal
    if is3d and cfg.get('wg_width'):
        w = cfg['wg_width']
        g.append(mp.Block(size=mp.Vector3(x_inf, cfg['t_wg'], w),
                          center=mp.Vector3(0, cfg['t_wg']/2 - cy, 0),
                          material=mp.Medium(index=cfg['wg_n'])))
    else:
        g.append(blk(0.0, cfg['t_wg'], cfg['wg_n']))
    # Bead
    b = cfg.get('bead')
    if b:
        center = mp.Vector3(b['x'] - cx, b['y'] - cy, b.get('z', 0.0))
        if is3d:
            g.append(mp.Sphere(radius=b['r'], center=center,
                               material=mp.Medium(index=b['n'])))
        else:
            g.append(mp.Cylinder(radius=b['r'], height=mp.inf,
                                 axis=mp.Vector3(0, 0, 1), center=center,
                                 material=mp.Medium(index=b['n'])))
    return g


def build_source(cfg):
    """Gauss-Spot (Taille y, in 3D auch z) an x=src_x auf der getriebenen Komponente
    (Ez fuer s/TE, Ey fuer p/TM). CW oder Puls."""
    f0 = cfg['f0']
    comp = mp.Ez if cfg['pol'] == 's' else mp.Ey
    is3d = cfg['dim'] == 3
    wy = max(cfg['vcsel_waist'], 2.0/cfg['resolution'])
    wz = max(cfg.get('vcsel_waist_z', cfg['vcsel_waist']), 2.0/cfg['resolution'])
    y_off = cfg['t_wg']/2 - cfg['cy']   # WG-Mitte relativ zum Domain-Zentrum

    def amp(p):
        a = np.exp(-((p.y)/wy)**2)       # p ist relativ zum Quell-Zentrum
        if is3d:
            a *= np.exp(-((p.z)/wz)**2)
        return complex(a)

    if cfg['source_type'] == 'pulse':
        src = mp.GaussianSource(frequency=f0, fwidth=0.2*f0)
    else:
        src = mp.ContinuousSource(frequency=f0)
    size = mp.Vector3(0, cfg['ly'], cfg['lz'] if is3d else 0)
    center = mp.Vector3(cfg['src_x'] - cfg['cx'], y_off, 0)
    return [mp.Source(src=src, component=comp, center=center, size=size, amp_func=amp)]


def _flux_region(cfg, x_phys):
    is3d = cfg['dim'] == 3
    y_off = cfg['t_wg']/2 - cfg['cy']
    hy = cfg['t_wg'] + 2.0                      # WG + Rand in y
    hz = (cfg.get('wg_width') or cfg['lz']) if is3d else 0
    return mp.FluxRegion(center=mp.Vector3(x_phys - cfg['cx'], y_off, 0),
                         size=mp.Vector3(0, hy, hz))


def run(cfg, progress=True):
    """Baut die Simulation, laeuft bis (nahe) Steady-State, sammelt Frames der
    getriebenen Komponente und Transmission (Flux out/in). Rueckgabe: dict."""
    is3d = cfg['dim'] == 3
    comp = mp.Ez if cfg['pol'] == 's' else mp.Ey
    # PML adaptiv: darf nie die halbe kleinste Domainkante ueberschreiten
    # (sonst "invalid boundary absorbers"). Ziel ~1 Wellenlaenge, gedeckelt.
    dims = [cfg['lx'], cfg['ly']] + ([cfg['lz']] if is3d else [])
    dpml = max(0.1, min(cfg['lam_nm']/1000.0, 0.4*min(dims)))
    cell = mp.Vector3(cfg['lx'], cfg['ly'], cfg['lz'] if is3d else 0)
    sim = mp.Simulation(cell_size=cell, resolution=cfg['resolution'],
                        geometry=build_geometry(cfg), sources=build_source(cfg),
                        boundary_layers=[mp.PML(dpml)],
                        default_material=mp.Medium(index=1.0),
                        force_complex_fields=(cfg['source_type'] == 'cw'))
    f_in = sim.add_flux(cfg['f0'], 0, 1, _flux_region(cfg, cfg['det_in_x']))
    f_out = sim.add_flux(cfg['f0'], 0, 1, _flux_region(cfg, cfg['det_out_x']))

    n_grp = max(cfg['wg_n'], 1.4)
    T = cfg.get('run_time') or (2.2*cfg['lx']*n_grp + 20.0)
    n_snap = max(1, cfg['n_snapshots'])
    interval = T/n_snap
    size = mp.Vector3(cfg['lx'], cfg['ly'], cfg['lz'] if is3d else 0)
    frames, times = [], []

    def cap(s):
        arr = s.get_array(center=mp.Vector3(), size=size, component=comp)
        frames.append(np.real(np.asarray(arr)).astype(np.float32))
        times.append(s.meep_time())
        if progress:
            print(f'  [meep] t={s.meep_time():.1f}/{T:.1f}  frame {len(frames)}/{n_snap}',
                  flush=True)

    sim.run(mp.at_every(interval, cap), until=T)

    pin = float(mp.get_fluxes(f_in)[0])
    pout = float(mp.get_fluxes(f_out)[0])
    Tr = pout/pin if abs(pin) > 1e-30 else 0.0
    print(f'[meep Result] {cfg["label"]}: P_in={pin:.3e} P_out={pout:.3e} '
          f'Transmission={Tr:.4f}')
    return dict(frames=frames, times=times, P_in=pin, P_out=pout, transmission=Tr,
                shape=frames[0].shape if frames else None)
