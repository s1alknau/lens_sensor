"""Schreibt Meep-Ergebnisse in das NPZ-Format des eigenen Analyzers
(common/fdtd_analyzer.py), damit Meep- und Solver-Laeufe im selben Viewer
verglichen werden koennen. Deckt 2D (_frames.npz) und 3D (_frames.npz + _vol3d.npz)."""
import os
import numpy as np

_MAT_NAMES = ['WG', 'Bead', 'Aqueous', 'Mucin', 'Cornea']


def _mat_indices(cfg):
    b = cfg.get('bead') or {}
    return np.array([cfg['wg_n'], b.get('n', cfg['n_lip']),
                     cfg['n_aq'], cfg['n_mu'], cfg['n_co']], dtype=np.float64)


def _layers_m(cfg):
    return np.array([(cfg['t_lip'] or 0.0)*1e-6, cfg['t_aq']*1e-6,
                     cfg['t_mu']*1e-6, cfg['n_aq']], dtype=np.float64)


def _bead_kw(cfg):
    b = cfg.get('bead') or {}
    return dict(bead_x_um=np.array([b.get('x', 0.0)]),
                bead_diameter_um=np.array([2*b['r'] if b else 0.0]),
                bead_y_um=np.array([b.get('y', 0.0)]),
                bead_z_um=np.array([b.get('z', 0.0)]))


def _flags():
    return dict(has_transient=np.array([1]), has_cw=np.array([0]),
                cw_only=np.array([0]), view_primary=np.array(['transient']))


def _meta(cfg, times):
    ys, ye = cfg['y_bot'], cfg['y_top']
    return np.array([(0.0, cfg['lx'], ys, ye, 0, t, i)
                     for i, t in enumerate(times)], dtype=np.float64)


def save_2d(cfg, result, out_dir, prefix='meep'):
    os.makedirs(out_dir, exist_ok=True)
    ez = np.stack(result['frames']).astype(np.float32)     # (n_fr, Nx, Ny)
    path = os.path.join(out_dir, f'{prefix}_{cfg["label"]}_frames.npz')
    np.savez_compressed(
        path, Ez=ez, meta=_meta(cfg, result['times']),
        scenario=cfg['label'], layers=_layers_m(cfg),
        material_names=np.array(_MAT_NAMES), material_indices=_mat_indices(cfg),
        t_wg_um=np.array([cfg['t_wg']]),
        x_wg_start_um=np.array([0.0]), x_wg_end_um=np.array([cfg['lx']]),
        polarization=np.array([cfg['pol']]), lam_nm=np.array([cfg['lam_nm']]),
        **_bead_kw(cfg), **_flags())
    print(f'  Saved: {path}  ({ez.shape})')
    return path


def save_3d(cfg, result, out_dir, prefix='meep'):
    os.makedirs(out_dir, exist_ok=True)
    vol = np.stack(result['frames']).astype(np.float32)    # (n_fr, Nx, Ny, Nz)
    kmid = vol.shape[3]//2
    ez_slice = vol[:, :, :, kmid]
    dx_um = 1.0/cfg['resolution']
    meta = _meta(cfg, result['times'])
    bead = _bead_kw(cfg)
    # Frames-Slice (z-Mittelebene) -> 2D-Viewer
    fr_path = os.path.join(out_dir, f'{prefix}_{cfg["label"]}_frames.npz')
    np.savez_compressed(
        fr_path, Ez=ez_slice, meta=meta, scenario=cfg['label'],
        layers=_layers_m(cfg), material_names=np.array(_MAT_NAMES),
        material_indices=_mat_indices(cfg), t_wg_um=np.array([cfg['t_wg']]),
        x_wg_start_um=np.array([0.0]), x_wg_end_um=np.array([cfg['lx']]),
        polarization=np.array([cfg['pol']]), lam_nm=np.array([cfg['lam_nm']]),
        **bead, **_flags())
    print(f'  Saved: {fr_path}  (z-Mittelebene {ez_slice.shape})')
    # Volles Volumen -> 3D-Viewer
    steps = np.arange(vol.shape[0], dtype=np.int64)
    Iavg = np.mean(vol.astype(np.float64)**2, axis=0).astype(np.float32)
    vol_path = os.path.join(out_dir, f'{prefix}_{cfg["label"]}_vol3d.npz')
    np.savez_compressed(
        vol_path, Ez=vol, Iavg=Iavg, steps=steps,
        times=np.asarray(result['times'], dtype=np.float64), dx_um=dx_um,
        x0_um=0.0, y0_um=cfg['y_bot'], z0_um=-cfg['lz']/2.0,
        scenario=cfg['label'], layers=_layers_m(cfg),
        material_names=np.array(_MAT_NAMES), material_indices=_mat_indices(cfg),
        t_wg_um=np.array([cfg['t_wg']]),
        wg_width_um=np.array([cfg.get('wg_width') or 0.0]),
        lx_um=np.array([cfg['lx']]), lam_nm=np.array([cfg['lam_nm']]),
        **bead, **_flags())
    print(f'  Saved: {vol_path}  ({vol.shape})')
    return fr_path
