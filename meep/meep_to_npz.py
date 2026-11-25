"""Schreibt Meep-Ergebnisse in das NPZ-Format des eigenen Analyzers
(common/fdtd_analyzer.py), damit Meep- und Solver-Laeufe im selben Viewer
verglichen werden koennen. Deckt 2D (_frames.npz) und 3D (_frames.npz + _vol3d.npz)."""
import os
import numpy as np

_MAT_NAMES = ['WG', 'Lipid', 'Aqueous', 'Mucin', 'Cornea']


def _stack(cfg):
    """Stapel im common.layer_stack-Format: frei konfiguriert (cfg['stack']) oder aus den
    drei Legacy-Schichten (legacy_stack, Nullschichten entfallen)."""
    from common.layer_stack import legacy_stack
    if cfg.get('stack') is not None:
        return [dict(name=L['name'], t_um=float(L['t_um']),
                     n=complex(L['n_re'], L.get('n_im') or 0.0),
                     kappa=float(L.get('n_im') or 0.0), mat=L.get('mat'))
                for L in cfg['stack']]
    nm = list(cfg.get('layer_names') or _MAT_NAMES)
    return legacy_stack(cfg['t_lip'] or 0.0, cfg['n_lip'], cfg['t_aq'], cfg['n_aq'],
                        cfg['t_mu'], cfg['n_mu'], names=nm[1:4])


def _stack_kw(cfg):
    from common.layer_stack import stack_to_npz
    return stack_to_npz(_stack(cfg))


def _names(cfg):
    """Schichtnamen (Kern, 3 Schichten unter dem Kern, Substrat) - vom Nutzer
    vergebbar (--layer-names); Linse: Traenenfilm-Namen. Frei konfigurierter Stapel:
    [Kern, *Stapel, Substrat]."""
    nm = list(cfg.get('layer_names') or _MAT_NAMES)
    if cfg.get('stack') is not None:
        return np.array([nm[0]] + [str(L['name']) for L in cfg['stack']] + [nm[-1]])
    return np.array(nm)


def _mat_indices(cfg):
    if cfg.get('stack') is not None:
        return np.array([cfg['wg_n']] + [float(L['n_re']) for L in cfg['stack']]
                        + [float(cfg['n_co'])], dtype=np.float64)
    return np.array([cfg['wg_n'], cfg['n_lip'], cfg['n_aq'], cfg['n_mu'], float(cfg['n_co'])],
                    dtype=np.float64)


def _layers_m(cfg):
    if cfg.get('stack') is not None:
        from common.layer_stack import legacy_layers_tuple
        return np.array(legacy_layers_tuple(_stack(cfg)), dtype=np.float64)
    return np.array([(cfg['t_lip'] or 0.0)*1e-6, cfg['t_aq']*1e-6,
                     cfg['t_mu']*1e-6, cfg['n_aq']], dtype=np.float64)


def _bead_kw(cfg):
    b = cfg.get('bead') or {}
    return dict(bead_n=np.array([b.get('n', 0.0)]),
                bead_x_um=np.array([b.get('x', 0.0)]),
                bead_diameter_um=np.array([2*b['r'] if b else 0.0]),
                bead_y_um=np.array([b.get('y', 0.0)]),
                bead_z_um=np.array([b.get('z', 0.0)]))


def _flags(has_cw=False):
    return dict(has_transient=np.array([1]), has_cw=np.array([1 if has_cw else 0]),
                cw_only=np.array([0]), view_primary=np.array(['transient']))


def _extra(result, cfg=None, dim=None):
    """Transmission (Poynting-Fluss ueber den ganzen Querschnitt) + CW-Konvergenz."""
    out = dict(engine=np.array(['meep']))
    if 'transmission' in result:
        out['transmission'] = np.array([float(result['transmission'])])
        out['transmission_def'] = np.array([str(result.get('transmission_def', 'poynting'))])
    if result.get('cw_rel_change') is not None:
        out['cw_rel_change'] = np.array([float(result['cw_rel_change'])])
    # VCSEL-Leistung: Meep-Einheiten (eps0 = mu0 = c = 1, a = 1 um) -> SI-Aequivalent, mit den
    # Feldzahlen als V/m: P = P_meep * a^(d-1) / Z0 (3D in W, 2D in W/m) -> Analyzer skaliert
    # wie native auf die Laserleistung.
    if result.get('p_emit') and dim in (2, 3):
        _f = (1e-6**(dim - 1))/376.730313668
        out['p_src_si'] = np.array([float(result['p_emit'])*_f])
        out['p_in_si'] = np.array([float(result['P_in'])*_f])
        out['p_in_unit'] = np.array(['W' if dim == 3 else 'W/m'])
        out['coupling_efficiency'] = np.array([float(result['coupling_efficiency'])])
        out['facet_reflection'] = np.array([float(result['facet_reflection'])])
    if cfg is not None:
        out['laser_power_mw'] = np.array([float(cfg.get('laser_power_mw') or 20.0)])
        out['vcsel_distance_um'] = np.array([float(cfg.get('vcsel_distance') or 0.0)])
        out['vcsel_waist_um'] = np.array([float(cfg.get('vcsel_waist') or 0.0)])
    return out


def _meta(cfg, times):
    ys, ye = cfg.get('view_yr') or (cfg['y_bot'], cfg['y_top'])
    x0, x1 = cfg.get('view_xr') or cfg.get('view_x') or (0.0, cfg['lx'])
    return np.array([(x0, x1, ys, ye, 0, t, i)
                     for i, t in enumerate(times)], dtype=np.float64)


def _resample(reg, shape):
    """Regionskarte (CAD) auf die Form der Meep-Frames bringen (naechster Nachbar)."""
    idx = [np.clip(np.round(np.linspace(0, reg.shape[k] - 1, shape[k])).astype(int), 0,
                   reg.shape[k] - 1) for k in range(len(shape))]
    return reg[np.ix_(*idx)]


def _cad_kw(cfg, frame_shape):
    """CAD-Schluessel wie die native Speicherung (Analyzer: Koerper statt Schichten)."""
    c = cfg.get('cad')
    if not c:
        return {}
    m = c['meta']
    reg = c['region'] if c['region'].ndim == 2 else c['region'][:, :, c['region'].shape[2]//2]
    return dict(geometry=np.array(['cad']),
                material_names=np.array(m['cad_names']),
                material_indices=np.array(m['cad_n'], dtype=np.float64),
                cad_materials=np.array(m['cad_materials']),
                cad_core=np.array([m['cad_core']]), cad_tear=np.array([m['cad_tear']]),
                cad_file=np.array([m['cad_file']]),
                cad_tilt_deg=np.array([cfg.get('cad_tilt', 0.0)]),
                cad_stride=np.array([1, 1], dtype=np.int64),
                cad_x_shift_um=np.array([m['cad_x_shift_um']]),
                cad_plane_z_um=np.array([m.get('cad_plane_z_um', 0.0)]),
                cad_scale_um=np.array([m['cad_scale_um']]),
                cad_region=_resample(reg, frame_shape).astype(np.int8))


def _wg_x_kw(cfg):
    """WG-Kern in x wie in der Meep-Geometrie (build_geometry): Eintritt hinter dem
    Einkoppelabstand, Austritt vor der Endfacette. Frueher immer 0..L -> der Analyzer
    zeichnete den Kern bei --input-gap/--end-facet an der falschen Stelle."""
    L = float(cfg.get('lx_total', cfg['lx']))
    gap = float(cfg.get('input_gap') or 0.0)
    ef = float(cfg.get('end_facet') or 0.0)
    x0 = float(cfg.get('src_x', 0.0)) + gap if gap > 0 else 0.0
    return dict(x_wg_start_um=np.array([x0]), x_wg_end_um=np.array([L - ef if ef > 0 else L]),
                x_src_um=np.array([float(cfg.get('src_x', 0.0))]),
                end_facet_um=np.array([ef]))


def save_2d(cfg, result, out_dir, prefix='meep'):
    os.makedirs(out_dir, exist_ok=True)
    ez = np.stack(result['frames']).astype(np.float32)     # (n_fr, Nx, Ny)
    path = os.path.join(out_dir, f'{prefix}_{cfg["label"]}_frames.npz')
    cw = result.get('cw_frames')
    cw_kw = {}
    if cw is not None:                                     # eingeschwungene Phasen (wie native)
        cw = np.asarray(cw, dtype=np.float32)
        cw_kw = dict(Ez_cw=cw, meta_cw=_meta(cfg, result.get('cw_times') or range(len(cw))),
                     Iavg=(0.5*np.mean(cw.astype(np.float64)**2, axis=0)*2).astype(np.float32))
    _base = dict(material_names=_names(cfg), material_indices=_mat_indices(cfg))
    if cfg.get('cad'):
        _base = {}
    np.savez_compressed(
        path, Ez=ez, meta=_meta(cfg, result['times']),
        scenario=cfg['label'], layers=_layers_m(cfg), **_base, **_stack_kw(cfg),
        t_wg_um=np.array([cfg['t_wg']]),
        **_wg_x_kw(cfg),
        polarization=np.array([cfg['pol']]), lam_nm=np.array([cfg['lam_nm']]),
        **_bead_kw(cfg), **_flags(cw is not None), **cw_kw, **_extra(result, cfg, 2),
        **_cad_kw(cfg, ez.shape[1:]))
    print(f'  Saved: {path}  ({ez.shape})' + ('  +CW set' if cw is not None else ''))
    return path


def save_3d(cfg, result, out_dir, prefix='meep'):
    os.makedirs(out_dir, exist_ok=True)
    vol = np.stack(result['frames']).astype(np.float32)    # (n_fr, Nx, Ny, Nz)
    kmid = vol.shape[3]//2
    cw = result.get('cw_frames')
    cw = np.asarray(cw, dtype=np.float32) if cw is not None else None
    meta_cw = _meta(cfg, result.get('cw_times') or (range(len(cw)) if cw is not None else []))
    ez_slice = vol[:, :, :, kmid]
    dx_um = 1.0/cfg['resolution']
    meta = _meta(cfg, result['times'])
    bead = _bead_kw(cfg)
    # Frames-Slice (z-Mittelebene) -> 2D-Viewer
    fr_path = os.path.join(out_dir, f'{prefix}_{cfg["label"]}_frames.npz')
    np.savez_compressed(
        fr_path, Ez=ez_slice, meta=meta, scenario=cfg['label'],
        layers=_layers_m(cfg), **_stack_kw(cfg), **({} if cfg.get('cad') else
                                  dict(material_names=_names(cfg),
                                       material_indices=_mat_indices(cfg))),
        t_wg_um=np.array([cfg['t_wg']]),
        **_wg_x_kw(cfg),
        polarization=np.array([cfg['pol']]), lam_nm=np.array([cfg['lam_nm']]),
        **bead, **_flags(cw is not None), **_extra(result, cfg, 3),
        **(dict(Ez_cw=cw[:, :, :, kmid], meta_cw=meta_cw) if cw is not None else {}),
        **_cad_kw(cfg, ez_slice.shape[1:]))
    print(f'  Saved: {fr_path}  (z mid-plane {ez_slice.shape})')
    # Volles Volumen -> 3D-Viewer
    steps = np.arange(vol.shape[0], dtype=np.int64)
    # Zeitmittel: aus den eingeschwungenen Phasen (exakt fuer CW), sonst aus den Frames
    Iavg = np.mean((cw if cw is not None else vol).astype(np.float64)**2,
                   axis=0).astype(np.float32)
    vol_path = os.path.join(out_dir, f'{prefix}_{cfg["label"]}_vol3d.npz')
    np.savez_compressed(
        vol_path, Ez=vol, Iavg=Iavg, steps=steps,
        times=np.asarray(result['times'], dtype=np.float64), dx_um=dx_um,
        x0_um=0.0, y0_um=cfg['y_bot'], z0_um=-cfg['lz']/2.0,
        scenario=cfg['label'], layers=_layers_m(cfg), **_stack_kw(cfg),
        material_names=_names(cfg), material_indices=_mat_indices(cfg),
        t_wg_um=np.array([cfg['t_wg']]),
        wg_width_um=np.array([cfg.get('wg_width') or 0.0]),
        lx_um=np.array([cfg.get('lx_total', cfg['lx'])]), lam_nm=np.array([cfg['lam_nm']]),
        polarization=np.array([cfg['pol']]),
        **bead, **_flags(cw is not None), **_extra(result, cfg, 3),
        **(dict(Ez_cw=cw, meta_cw=meta_cw) if cw is not None else {}))
    print(f'  Saved: {vol_path}  ({vol.shape})' + ('  +CW set' if cw is not None else ''))
    return fr_path
