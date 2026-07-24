"""Drehbare 3D-Ansicht eines FDTD-Volumens (_vol3d.npz) - reines Matplotlib.

Zeigt die Feld-"Hotspots" (|Feld| ueber einer Schwelle) als drehbare 3D-Punkt-
wolke, eingefaerbt nach Vorzeichen (RdBu). Zusaetzlich: Wellenleiter-Kern als
Drahtbox und der Bead als Drahtkugel zur Orientierung. Mit der Maus drehbar.

Optional: echte Isoflaeche via scikit-image (falls installiert, --iso).

Achsen:  x = LAENGE (Propagation), y = QUERSCHNITT (Schichten), z = TIEFE.

VERWENDUNG:
  python view3d.py results\\beads3d_..._vol3d.npz
  python view3d.py results\\..._vol3d.npz --frame 5 --thresh 0.3
  python view3d.py results\\..._vol3d.npz --avg           (Zeitmittel statt Frame)
  python view3d.py results\\..._vol3d.npz --comp absE      (Vektor-Dateien)
  python view3d.py results\\..._vol3d.npz --save bild.png  (ohne Fenster speichern)
"""
import argparse
import os
import sys
import numpy as np


def _load(path):
    d = np.load(path)
    g = lambda k, dflt=0.0: (float(np.asarray(d[k]).flat[0]) if k in d.files else dflt)
    out = dict(ez=d['Ez'], dx=g('dx_um', 0.05), x0=g('x0_um', 0.0),
               y0=g('y0_um', 0.0), z0=g('z0_um', 0.0),
               t_wg=g('t_wg_um', 0.0), wg_w=g('wg_width_um', 0.0),
               bead_x=g('bead_x_um', 0.0), bead_d=g('bead_diameter_um', 0.0),
               iavg=(np.asarray(d['Iavg']) if 'Iavg' in d.files else None),
               ex=(np.asarray(d['Ex']) if 'Ex' in d.files else None),
               ey=(np.asarray(d['Ey']) if 'Ey' in d.files else None),
               scenario=(str(d['scenario']) if 'scenario' in d.files else 'FDTD'))
    return out, d


def _field_volume(data, frame, comp, use_avg):
    """Liefert das 3D-Feld (Nx,Ny,Nz) fuer Frame/Komponente."""
    if use_avg and data['iavg'] is not None:
        return np.asarray(data['iavg'], dtype=np.float32), True  # Intensitaet >=0
    if comp == 'absE' and data['ex'] is not None:
        ez = data['ez'][frame].astype(np.float32)
        ex = data['ex'][frame].astype(np.float32)
        ey = data['ey'][frame].astype(np.float32)
        return np.sqrt(ex*ex + ey*ey + ez*ez), True
    if comp == 'Ex' and data['ex'] is not None:
        return data['ex'][frame].astype(np.float32), False
    if comp == 'Ey' and data['ey'] is not None:
        return data['ey'][frame].astype(np.float32), False
    return data['ez'][frame].astype(np.float32), False


def _wire_box(ax, x0, x1, y0, y1, z0, z1, color, lw=1.2, label=None):
    pts = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        ax.plot(*zip(pts[a], pts[b]), color=color, lw=lw, alpha=0.9)
    if label:
        ax.text(x1, y1, z1, ' '+label, color=color, fontsize=8)


def _wire_sphere(ax, cx, cy, cz, r, color):
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 12)
    x = cx + r*np.outer(np.cos(u), np.sin(v))
    y = cy + r*np.outer(np.sin(u), np.sin(v))
    z = cz + r*np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, lw=0.8, alpha=0.9)


def main():
    ap = argparse.ArgumentParser(description='Drehbare 3D-Ansicht (_vol3d.npz)')
    ap.add_argument('npz')
    ap.add_argument('--frame', type=int, default=-1,
                    help='Frame-Index (Default: staerkster Frame)')
    ap.add_argument('--comp', default='Ez', choices=['Ez', 'Ex', 'Ey', 'absE'])
    ap.add_argument('--avg', action='store_true',
                    help='Zeitgemittelte Intensitaet statt eines Frames')
    ap.add_argument('--thresh', type=float, default=0.30,
                    help='Schwelle als Anteil vom Max (0..1), Default 0.30')
    ap.add_argument('--stride', type=int, default=0,
                    help='Downsampling (0 = automatisch)')
    ap.add_argument('--max-points', type=int, default=80000)
    ap.add_argument('--iso', action='store_true',
                    help='Echte Isoflaeche via scikit-image (falls installiert)')
    ap.add_argument('--save', default='', help='PNG speichern statt Fenster')
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        print(f'Datei nicht gefunden: {args.npz}'); return 1
    data, d = _load(args.npz)
    ez_all = data['ez']
    if ez_all.ndim != 4:
        print('Das ist kein 3D-Volumen (_vol3d.npz noetig, nicht _frames.npz).')
        return 1
    nf = ez_all.shape[0]
    frame = args.frame if args.frame >= 0 else None
    if frame is None and not args.avg:
        # staerksten Frame suchen (max|Ez| ueber die Mittelebene, billig)
        kmid = ez_all.shape[3]//2
        amps = [np.abs(ez_all[f, :, :, kmid].astype(np.float32)).max() for f in range(nf)]
        frame = int(np.argmax(amps))
    frame = 0 if frame is None else max(0, min(nf-1, frame))

    vol, nonneg = _field_volume(data, frame, args.comp, args.avg)
    dx = data['dx']
    Nx, Ny, Nz = vol.shape

    import matplotlib
    if args.save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    vmax = float(np.abs(vol).max()) or 1.0
    lvl = args.thresh*vmax

    fig = plt.figure(figsize=(11, 8), facecolor='#0a0a0a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a0a')

    used_iso = False
    if args.iso:
        try:
            from skimage.measure import marching_cubes
            verts, faces, _, _ = marching_cubes(np.abs(vol), level=lvl)
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            # Indizes -> um
            vx = data['x0'] + verts[:, 0]*dx
            vy = data['y0'] + verts[:, 1]*dx
            vz = data['z0'] + verts[:, 2]*dx
            mesh = Poly3DCollection(np.stack([vx[faces], vy[faces], vz[faces]], axis=-1),
                                    alpha=0.35)
            mesh.set_facecolor('#FF8800'); mesh.set_edgecolor('none')
            ax.add_collection3d(mesh)
            used_iso = True
            print(f'Isoflaeche bei |Feld|={lvl:.3g} ({verts.shape[0]} Vertices)')
        except ImportError:
            print('scikit-image fehlt -> Punktwolke.')

    if not used_iso:
        # automatisches Downsampling
        stride = args.stride
        if stride <= 0:
            stride = max(1, int(round((Nx*Ny*Nz/2e6)**(1/3))))
        sub = vol[::stride, ::stride, ::stride]
        mask = np.abs(sub) >= lvl
        # zu viele Punkte? Schwelle anheben
        while mask.sum() > args.max_points and lvl < vmax:
            lvl *= 1.25
            mask = np.abs(sub) >= lvl
        ii, jj, kk = np.where(mask)
        xs = data['x0'] + ii*stride*dx
        ys = data['y0'] + jj*stride*dx
        zs = data['z0'] + kk*stride*dx
        vals = sub[mask]
        cmap = 'inferno' if nonneg else 'RdBu_r'
        cargs = dict(c=vals, cmap=cmap, s=6, alpha=0.35, linewidths=0)
        if not nonneg:
            cargs.update(vmin=-vmax*0.6, vmax=vmax*0.6)
        p = ax.scatter(xs, ys, zs, **cargs)
        fig.colorbar(p, ax=ax, shrink=0.5, pad=0.1, label=args.comp)
        print(f'Punktwolke: {mask.sum()} Punkte ueber |Feld|>={lvl:.3g} '
              f'(stride={stride})')

    # Kontext: Wellenleiter-Kern + Bead
    lx = data['x0'] + Nx*dx
    t_wg = data['t_wg'] or 5.0
    wg_w = data['wg_w']
    z0d, z1d = data['z0'], data['z0'] + Nz*dx
    if wg_w and wg_w > 0:
        zc0, zc1 = -wg_w/2, wg_w/2
    else:
        zc0, zc1 = z0d, z1d
    _wire_box(ax, data['x0'], lx, 0.0, t_wg, zc0, zc1, '#FFD75E', label='WG-Kern')
    if data['bead_d'] > 0:
        _wire_sphere(ax, data['bead_x'], -data['bead_d']/2, 0.0,
                     data['bead_d']/2, '#39FF14')

    ax.set_xlabel('Laenge x (um)', color='#ccc')
    ax.set_ylabel('Querschnitt y (um)', color='#ccc')
    ax.set_zlabel('Tiefe z (um)', color='#ccc')
    ax.tick_params(colors='#aaa')
    try:
        ax.set_box_aspect((Nx*dx, Ny*dx, Nz*dx))
    except Exception:
        pass
    tag = 'Zeitmittel' if args.avg else f'Frame {frame+1}/{nf}'
    ax.set_title(f"{data['scenario']}  -  {args.comp}  -  {tag}\n"
                 f"(mit Maus drehbar)", color='#fff', fontsize=10)

    if args.save:
        fig.savefig(args.save, dpi=110, facecolor='#0a0a0a')
        print(f'Gespeichert: {args.save}')
    else:
        plt.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
