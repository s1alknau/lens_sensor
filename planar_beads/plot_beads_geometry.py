"""Geometrie-Montage des Bead-Setups mit MATERIAL-Farben und GLEICHER
Achsenskalierung (aspect=equal) -> Beads als echte Kreise.

Ein Panel pro Durchmesser; jeder Bead sitzt auf seiner flachen Mucin bei -d.
Mit --layers wird statt Aqueous/Mucin ein frei konfigurierbarer Schichtstapel
(Format wie beim Solver, s. common/layer_stack.py) gezeichnet; der Bead liegt dann
an der Kern-Unterkante (y = -r).

Verwendung:
  python plot_beads_geometry.py
  python plot_beads_geometry.py --diameters 0.2 0.5 1 2 4 --wg-thickness 5
  python plot_beads_geometry.py --layers "Aqueous:3:aqueous,Mucin:0.5:mucin,Gel:1:1.40"
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

import sys
import fdtd2d_core as db
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from common.layer_stack import parse_layer_stack, stack_bounds, stack_total_um

IDS = {'Air': 0, 'Waveguide': 1, 'Aqueous': 2, 'Mucin': 3, 'Cornea': 4, 'Bead': 5}
COLORS = ['#f2f2f2', '#8ec9ff', '#2aa6a0', '#ff9e2c', '#b98a5e', '#ffe14d']
NAMES = ['Air', 'Waveguide', 'Aqueous', 'Mucin', 'Cornea', 'Bead']
LAYER_COLORS = ['#2aa6a0', '#ff9e2c', '#9b7fe0', '#7cc96b', '#e07f9b', '#5fb4d9', '#c9b25a']


def stack_layout(stack):
    """Namen/Farben/IDs fuer einen freien Stapel: Air 0, Waveguide 1, Schichten 2..,
    Substrat (Cornea), Bead zuletzt."""
    names = ['Air', 'Waveguide'] + [L['name'] for L in stack] + ['Cornea', 'Bead']
    colors = ['#f2f2f2', '#8ec9ff'] + [LAYER_COLORS[k % len(LAYER_COLORS)]
                                       for k in range(len(stack))] + ['#b98a5e', '#ffe14d']
    return names, colors


def local_map(d_um, T_WG, dx, x_half_um, y_lo_um, y_hi_um, stack=None):
    """Kleines Material-Feld um den Bead (x relativ zum Bead-Zentrum).
    stack=None: altes Setup (Aqueous je nach d, Mucin, Cornea); sonst freier Stapel
    (IDs wie stack_layout)."""
    xs = np.arange(-x_half_um*1e-6, x_half_um*1e-6, dx)
    ys = np.arange(y_lo_um*1e-6, y_hi_um*1e-6, dx)
    Nx, Ny = len(xs), len(ys)
    col = np.zeros(Ny, dtype=np.int8)               # Air
    if stack is None:
        t_aq = db.aqueous_thickness(d_um)
        y_aq_bot = -t_aq; y_mu_bot = y_aq_bot - db.T_MU
        col[ys < y_mu_bot] = IDS['Cornea']
        col[(ys >= y_mu_bot) & (ys < y_aq_bot)] = IDS['Mucin']
        col[(ys >= y_aq_bot) & (ys < 0.0)] = IDS['Aqueous']
        bead_id = IDS['Bead']
    else:
        nl = len(stack)
        col[ys < -stack_total_um(stack)*1e-6] = 2 + nl                    # Substrat
        for k, (_nm, yt, yb, _n) in enumerate(stack_bounds(stack, 0.0)):
            col[(ys >= yb*1e-6) & (ys < yt*1e-6)] = 2 + k
        bead_id = 3 + nl
    col[(ys >= 0.0) & (ys <= T_WG)] = IDS['Waveguide']
    mid = np.tile(col, (Nx, 1))
    r = d_um*1e-6/2.0; yc = -r
    for i in range(Nx):
        dxx = xs[i]
        if abs(dxx) > r:
            continue
        dy = np.sqrt(max(0.0, r*r - dxx*dxx))
        mid[i, (ys >= yc-dy) & (ys <= yc+dy)] = bead_id
    ext = [xs[0]*1e6, xs[-1]*1e6, ys[0]*1e6, ys[-1]*1e6]
    return mid, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diameters', type=float, nargs='+', default=[0.2, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument('--wg-thickness', type=float, default=5.0)
    ap.add_argument('--resolution', type=float, default=15.0)
    ap.add_argument('--layers', default=None,
                    help='freier Schichtstapel "Name:t_um:n,..." (wie beim Solver; "none" = leer)')
    ap.add_argument('--lambda-nm', type=float, default=850.0,
                    help='Wellenlaenge fuer Material-Namen in --layers')
    args = ap.parse_args()
    stack = parse_layer_stack(args.layers, args.lambda_nm) if args.layers is not None else None
    names, colors = (NAMES, COLORS) if stack is None else stack_layout(stack)
    T_WG = args.wg_thickness*1e-6; dx = args.resolution*1e-9
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), cmap.N)
    ds = args.diameters
    # Feste Achsengrenzen fuer ALLE Panels -> Beads in echter relativer Groesse
    X_HALF = 3.5; Y_LO = -9.0; Y_HI = 2.0
    fig, axs = plt.subplots(1, len(ds), figsize=(2.7*len(ds), 5.1))
    if len(ds) == 1:
        axs = [axs]
    for ax, d in zip(axs, ds):
        mid, ext = local_map(d, T_WG, dx, X_HALF, Y_LO, Y_HI, stack)
        ax.imshow(mid.T, origin='lower', extent=ext, aspect='equal',
                  cmap=cmap, norm=norm, interpolation='nearest')
        ax.axhline(0, color='w', lw=0.7, ls='--')
        ax.set_xlim(-X_HALF, X_HALF); ax.set_ylim(Y_LO, Y_HI)
        if stack is None:
            ax.axhline(-db.aqueous_thickness(d)*1e6, color='#d11', lw=1.1)   # Mucin-Oberkante
            ax.set_title(f'd={d:g} um (Mucin @ -{db.aqueous_thickness(d)*1e6:g})', fontsize=10)
        else:
            for _nm, _yt, yb, _n in stack_bounds(stack, 0.0):
                ax.axhline(yb, color='#d11', lw=0.8, ls='--')                # Schicht-Unterkanten
            ax.set_title(f'd={d:g} um', fontsize=10)
        ax.set_xlabel('x-xc (um)')
    axs[0].set_ylabel('y (um)')
    handles = [Patch(facecolor=c, edgecolor='#666', label=n) for n, c in zip(names, colors)]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(names), 6), frameon=False,
               fontsize=9)
    if stack is None:
        fig.suptitle('Bead in Wasser eingebettet. Mucin (orange) gestaffelt: d>=1um -> -4um, d<1um -> -1um (rote Linie). aspect=equal.', y=0.99)
    else:
        fig.suptitle(f'Freier Stapel ({len(stack)} Schichten), Bead am Kern', y=0.99)
    fig.tight_layout(rect=[0, 0.13, 1, 0.95])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                       'beads_geometry_montage.png' if stack is None else
                       'beads_geometry_montage_stack.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=115); print('Saved', out)


if __name__ == '__main__':
    main()
