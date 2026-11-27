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
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from common import vcsel                         # noqa: E402  (gemeinsames VCSEL-Modell)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _custom_stack(cfg):
    """Frei konfigurierter Stapel (--layers) als [(name, y_top, y_bot, n_re, n_im), ...]
    vom Kern (y=0) abwaerts; None im Legacy-Fall (drei feste Schichten)."""
    st = cfg.get('stack')
    if st is None:
        return None
    out, y = [], 0.0
    for L in st:
        t = float(L['t_um'])
        out.append((L['name'], y, y - t, float(L['n_re']), float(L.get('n_im') or 0.0)))
        y -= t
    return out


def stack_bottom(cfg):
    """y (phys) der Stapel-Unterseite = Oberkante des Substrats (Cornea)."""
    cs = _custom_stack(cfg)
    if cs is not None:
        return cs[-1][2] if cs else 0.0
    return _layer_bounds(cfg)[2]


def n_below_core(cfg, y):
    """Reelle Brechzahl des Stapels/Substrats an der Stelle y < 0 (phys. um)."""
    cs = _custom_stack(cfg)
    if cs is not None:
        for _nm, yt, yb, nr, _ni in cs:
            if yb <= y < yt:
                return nr
    else:
        y_lip_bot, y_aq_bot, y_mu_bot = _layer_bounds(cfg)
        if y >= y_lip_bot and (cfg['t_lip'] or 0.0) > 1e-6:
            return float(cfg['n_lip'])
        if y >= y_aq_bot:
            return float(cfg['n_aq'])
        if y >= y_mu_bot:
            return float(cfg['n_mu'])
    n_co = cfg['n_co']
    yb = stack_bottom(cfg)
    return float(n_co.at_depth(yb - y)) if hasattr(n_co, 'at_depth') else float(n_co)


def _layer_bounds(cfg):
    """y-Grenzen (phys) der Tear-Schichten unter dem WG (y<0) - nur Legacy-Stapel."""
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
    """Schichtstapel eps(y) (phys. um) wie native, mit oder ohne Kern. Reelles Gitter
    (epsilon_input_file) -> Absorption (n_im) der Schichten entfaellt hier."""
    cs = _custom_stack(cfg)
    y_sb = stack_bottom(cfg)
    e = np.full(y.shape, 1.0)
    mc = y < y_sb
    n_co = cfg['n_co']
    if hasattr(n_co, 'at_depth'):
        e[mc] = n_co.at_depth(y_sb - y[mc])**2
    else:
        e[mc] = float(n_co)**2
    if cs is not None:
        for _nm, yt, yb, nr, _ni in cs:
            e[(y >= yb) & (y < yt)] = nr**2
    else:
        y_lip_bot, y_aq_bot, y_mu_bot = _layer_bounds(cfg)
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


def _med_nk(cfg, n, k):
    """Meep-Medium fuer komplexe Brechzahl n + i k (Stapelschicht): wie _med, aber k direkt."""
    k = float(k or 0.0)
    if k <= 0:
        return mp.Medium(index=n)
    e1 = n*n - k*k
    return mp.Medium(epsilon=e1, D_conductivity=2*np.pi*cfg['f0']*(2*n*k)/e1)


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

    def blk(y0, y1, n, key=None, mat=None):
        yc = 0.5*(y0 + y1) - cy
        return mp.Block(size=mp.Vector3(x_inf, y1 - y0, z_size),
                        center=mp.Vector3(0, yc, 0),
                        material=mat if mat is not None else _med(cfg, n, key))

    g = []
    cs = _custom_stack(cfg)
    # Oberkante des Substrats: Legacy = Mucin-Unterseite, sonst Stapel-Unterseite
    y_mu_bot = stack_bottom(cfg)
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
    if cs is not None:
        # frei konfigurierter Stapel: ein Block je Schicht, Absorption (n_im = kappa)
        # ueber Meeps D-Leitfaehigkeit wie bei den --kappa-*-Schichten (exakt bei f0)
        for _nm, yt, yb, nr, ni in cs:
            g.append(blk(yb, yt, nr, mat=_med_nk(cfg, nr, ni)))
    else:
        y_lip_bot, y_aq_bot, _ = _layer_bounds(cfg)
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


def _src_center(cfg):
    """Quellmitte (Meep-Koordinaten) und Strahlparameter wie native (WG-Mitte + Versatz,
    CAD: auf die Eintrittsflaeche ausgerichtet)."""
    is3d = cfg['dim'] == 3
    y_c = cfg['src_y'] if 'src_y' in cfg else cfg['t_wg']/2
    y_off = y_c + (cfg.get('vcsel_offset') or 0.0) - cfg['cy']
    z_off = ((cfg.get('src_z') or 0.0) + (cfg.get('vcsel_offset_z') or 0.0)) if is3d else 0.0
    tilt = (cfg.get('vcsel_tilt') or 0.0) + (cfg.get('cad_tilt') or 0.0)
    return mp.Vector3(cfg['src_x'] - cfg['cx'], y_off, z_off), tilt


def _plain_air_background(cfg):
    """GaussianBeamSource braucht ein homogenes Medium an der Quelle; Meep nimmt dafuer
    default_material. Gilt fuer den eingebauten Stapel (Luft vor der Facette); CAD/gekruemmt
    (Gitter-Material) -> Fallback auf die weiche Quelle."""
    return _grid(cfg) is None


def build_source(cfg):
    """VCSEL (common/vcsel.py): Gauss-Strahl TEM00 in Luft vor der Facette, Taille
    vcsel_waist im Abstand vcsel_distance vor der Facette (Standard 0 = ebene Wellenfront
    auf der Facette), einseitig abstrahlend. Meep >= 1.24: GaussianBeam2D/3DSource (exakter
    Strahl ueber aequivalente E- und H-Stroeme). Fallback (Gitter-Material): weiche
    Quelle mit demselben Profil (Taille, Kruemmung, Neigung), beidseitig.
    Konvention wie native: positive Neigung -> Strahl Richtung -y (native Phasenrampe
    sin(w t + k sin(theta) y)); Meep-Felder ~ exp(-i w t) -> Phase konjugiert."""
    f0 = cfg['f0']
    comp = mp.Ez if cfg['pol'] == 's' else mp.Ey
    is3d = cfg['dim'] == 3
    center, tilt = _src_center(cfg)
    w0 = float(cfg['vcsel_waist'])
    w0z = float(cfg.get('vcsel_waist_z') or w0)
    zprop = float(cfg.get('vcsel_z', 0.0))          # Quellebene relativ zur Taille
    if cfg['source_type'] == 'pulse':
        # wie native: Huellkurve exp(-(t/sp)^2), sp = Perioden/f0 -> Meep-Breite
        # exp(-(t^2)/(2 w^2)) mit w = sp/sqrt(2), fwidth = 1/w
        ncyc = float(cfg.get('pulse_cycles') or 30.0)
        src = mp.GaussianSource(frequency=f0, fwidth=np.sqrt(2.0)*f0/ncyc)
    else:
        src = mp.ContinuousSource(frequency=f0)
    size = mp.Vector3(0, cfg['ly'] - 2*cfg.get('dpml', 0.0), cfg['lz'] if is3d else 0)
    th = np.radians(tilt)
    kdir = mp.Vector3(np.cos(th), -np.sin(th), 0)
    if cfg.get('source_core_only'):
        print('[meep] Note: --source-core-only is not applied to the VCSEL beam (source in air)')
    gb = getattr(mp, 'GaussianBeam3DSource' if is3d else 'GaussianBeam2DSource', None)         or getattr(mp, 'GaussianBeamSource', None)
    if gb is not None and _plain_air_background(cfg):
        if is3d and abs(w0z - w0) > 1e-9:
            print(f'[meep] Note: GaussianBeam source is round -> waist_z {w0z:g} ignored, '
                  f'w0 = {w0:g} um')
        pol = mp.Vector3(0, 0, 1) if cfg['pol'] == 's' else mp.Vector3(np.sin(th), np.cos(th), 0)
        x0 = kdir.scale(-zprop)                     # Fokus (Taille) relativ zur Quellmitte
        cfg['vcsel_impl'] = 'meep-gaussianbeam'
        return [gb(src=src, center=center, size=size, beam_x0=x0, beam_kdir=kdir,
                   beam_w0=w0, beam_E0=pol)]
    # Fallback: weiche Quelle mit dem VCSEL-Profil (beidseitig -> Leistung nicht exakt)
    print('[meep] Note: grid material (CAD/curved) -> VCSEL as soft source (emits both ways; '
          'emitted power not measured)')
    cfg['vcsel_impl'] = 'soft'
    lam_nm = cfg['lam_nm']

    def amp(p):                                      # p relativ zur Quellmitte
        if is3d:
            a, ph = vcsel.profile_2d([p.y], [p.z], 0.0, 0.0, zprop, w0, w0z, lam_nm, 1.0, tilt)
            a, ph = float(a[0, 0]), float(ph[0, 0])
        else:
            a, ph = vcsel.profile_1d([p.y], 0.0, zprop, w0, lam_nm, 1.0, tilt)
            a, ph = float(a[0]), float(ph[0])
        return complex(a)*np.exp(-1j*ph)             # native sin(wt+ph) <-> Meep exp(-i(wt+ph))

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


def _core_region(cfg, x_phys):
    """Flussebene NUR ueber den Kern (y in [0, t_wg], 3D: Rippenbreite oder ganze Tiefe)
    -> Leistung im Kern. None bei CAD/gekruemmt (Kernlage nicht achsparallel)."""
    if cfg.get('cad') is not None or cfg.get('curved'):
        return None
    is3d = cfg['dim'] == 3
    d = cfg.get('dpml', 0.0)
    hz = 0
    if is3d:
        hz = cfg['wg_width'] if cfg.get('wg_width') else cfg['lz'] - 2*d
    return mp.FluxRegion(center=mp.Vector3(x_phys - cfg['cx'], cfg['t_wg']/2 - cfg['cy'], 0),
                         size=mp.Vector3(0, cfg['t_wg'], hz))


def _add_core_fluxes(sim, cfg):
    r_in, r_out = _core_region(cfg, cfg['det_in_x']), _core_region(cfg, cfg['det_out_x'])
    if r_in is None:
        return None, None
    return sim.add_flux(cfg['f0'], 0, 1, r_in), sim.add_flux(cfg['f0'], 0, 1, r_out)


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
    _dsrc = 2.0/cfg['resolution']                    # Ebenen um die VCSEL-Quelle
    if not is_cw:
        f_in = sim.add_flux(cfg['f0'], 0, 1, _flux_region(cfg, cfg['det_in_x']))
        f_out = sim.add_flux(cfg['f0'], 0, 1, _flux_region(cfg, cfg['det_out_x']))
        f_up = sim.add_flux(cfg['f0'], 0, 1, _flux_region(cfg, cfg['src_x'] - _dsrc))
        f_dn = sim.add_flux(cfg['f0'], 0, 1, _flux_region(cfg, cfg['src_x'] + _dsrc))
        fc_in, fc_out = _add_core_fluxes(sim, cfg)

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
        f_up = sim.add_flux(cfg['f0'], 0, 1, _flux_region(cfg, cfg['src_x'] - _dsrc))
        f_dn = sim.add_flux(cfg['f0'], 0, 1, _flux_region(cfg, cfg['src_x'] + _dsrc))
        fc_in, fc_out = _add_core_fluxes(sim, cfg)
        sim.run(until=4*Tp)

    pin = float(mp.get_fluxes(f_in)[0])
    pout = float(mp.get_fluxes(f_out)[0])
    Tr = pout/pin if abs(pin) > 1e-30 else 0.0
    print(f'[meep Result] {cfg["label"]}: P_in={pin:.3e} P_out={pout:.3e} '
          f'Transmission={Tr:.4f}')
    out = dict(frames=frames, times=times, P_in=pin, P_out=pout, transmission=Tr,
               transmission_def='poynting', cw_rel_change=cw_rel,
               shape=frames[0].shape if frames else None)
    # VCSEL: abgestrahlte Leistung = Netto-Fluss aus dem Bereich um die Quelle
    pup = float(mp.get_fluxes(f_up)[0]); pdn = float(mp.get_fluxes(f_dn)[0])
    p_emit = pdn - pup
    if cfg.get('vcsel_impl') == 'meep-gaussianbeam' and p_emit > 0:
        eta = pin/p_emit
        out.update(p_emit=p_emit, coupling_efficiency=eta, facet_reflection=-pup/p_emit,
                   transmission_total=pout/p_emit)
        print(f'[meep VCSEL] emitted P={p_emit:.4e} (Meep units)  facet reflection='
              f'{-100*pup/p_emit:.2f}%  coupled through input plane: eta={100*eta:.2f}%  '
              f'T_total = out/laser = {pout/p_emit:.4f}')
        if fc_in is not None:
            cc = float(mp.get_fluxes(fc_in)[0])/p_emit
            co = float(mp.get_fluxes(fc_out)[0])/p_emit
            out.update(coupling_core=cc, core_out=co)
            print(f'[meep VCSEL] in core after facet={100*cc:.2f}%  in core at output={100*co:.2f}%')
    if cw_frames is not None:
        out.update(cw_frames=cw_frames, cw_times=cw_times)
    return out
