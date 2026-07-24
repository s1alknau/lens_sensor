"""Geometrie-Montage des Bead-Setups mit MATERIAL-Farben und GLEICHER
Achsenskalierung (aspect=equal) -> Beads als echte Kreise.

Ein Panel pro Durchmesser; jeder Bead sitzt auf seiner flachen Mucin bei -d.

Verwendung:
  python plot_beads_geometry.py
  python plot_beads_geometry.py --diameters 0.2 0.5 1 2 4 --wg-thickness 5
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

import demo_beads as db

IDS = {'Air': 0, 'Waveguide': 1, 'Aqueous': 2, 'Mucin': 3, 'Cornea': 4, 'Bead': 5}
COLORS = ['#f2f2f2', '#8ec9ff', '#2aa6a0', '#ff9e2c', '#b98a5e', '#ffe14d']
NAMES = ['Air', 'Waveguide', 'Aqueous', 'Mucin', 'Cornea', 'Bead']


def local_map(d_um, T_WG, dx, x_half_um, y_lo_um, y_hi_um):
    """Kleines Material-Feld um den Bead (x relativ zum Bead-Zentrum)."""
    xs = np.arange(-x_half_um*1e-6, x_half_um*1e-6, dx)
    ys = np.arange(y_lo_um*1e-6, y_hi_um*1e-6, dx)
    Nx, Ny = len(xs), len(ys)
    t_aq = db.aqueous_thickness(d_um)
    y_aq_bot = -t_aq; y_mu_bot = y_aq_bot - db.T_MU
    col = np.zeros(Ny, dtype=np.int8)               # Air
    col[ys < y_mu_bot] = IDS['Cornea']
    col[(ys >= y_mu_bot) & (ys < y_aq_bot)] = IDS['Mucin']
    col[(ys >= y_aq_bot) & (ys < 0.0)] = IDS['Aqueous']
    col[(ys >= 0.0) & (ys <= T_WG)] = IDS['Waveguide']
    mid = np.tile(col, (Nx, 1))
    r = d_um*1e-6/2.0; yc = -r
    for i in range(Nx):
        dxx = xs[i]
        if abs(dxx) > r:
            continue
        dy = np.sqrt(max(0.0, r*r - dxx*dxx))
        mid[i, (ys >= yc-dy) & (ys <= yc+dy)] = IDS['Bead']
    ext = [xs[0]*1e6, xs[-1]*1e6, ys[0]*1e6, ys[-1]*1e6]
    return mid, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diameters', type=float, nargs='+', default=[0.2, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument('--wg-thickness', type=float, default=5.0)
    ap.add_argument('--resolution', type=float, default=15.0)
    args = ap.parse_args()
    T_WG = args.wg_thickness*1e-6; dx = args.resolution*1e-9
    cmap = ListedColormap(COLORS); norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
    ds = args.diameters
    # Feste Achsengrenzen fuer ALLE Panels -> Beads in echter relativer Groesse
    X_HALF = 3.5; Y_LO = -9.0; Y_HI = 2.0
    fig, axs = plt.subplots(1, len(ds), figsize=(2.7*len(ds), 5.1))
    if len(ds) == 1:
        axs = [axs]
    for ax, d in zip(axs, ds):
        mid, ext = local_map(d, T_WG, dx, X_HALF, Y_LO, Y_HI)
        ax.imshow(mid.T, origin='lower', extent=ext, aspect='equal',
                  cmap=cmap, norm=norm, interpolation='nearest')
        ax.axhline(0, color='w', lw=0.7, ls='--')
        ax.axhline(-db.aqueous_thickness(d)*1e6, color='#d11', lw=1.1)   # Mucin-Oberkante
        ax.set_xlim(-X_HALF, X_HALF); ax.set_ylim(Y_LO, Y_HI)
        ax.set_title(f'd={d:g} um (Mucin @ -{db.aqueous_thickness(d)*1e6:g})', fontsize=10)
        ax.set_xlabel('x-xc (um)')
    axs[0].set_ylabel('y (um)')
    handles = [Patch(facecolor=COLORS[IDS[n]], edgecolor='#666', label=n) for n in NAMES]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=9)
    fig.suptitle('Bead in Wasser eingebettet. Mucin (orange) gestaffelt: d>=1um -> -4um, d<1um -> -1um (rote Linie). aspect=equal.', y=0.99)
    fig.tight_layout(rect=[0, 0.13, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'beads_geometry_montage.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=115); print('Saved', out)


if __name__ == '__main__':
    main()
