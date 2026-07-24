"""Rebuild GIF aus einer Sliding-FDTD _frames.npz Datei.

Verwendung:
  python rebuild_gif.py results/sliding_Gesund_frames.npz
  python rebuild_gif.py results/sliding_Gesund_frames.npz --duration 0.15
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MPoly
import imageio.v2 as imageio

T_LENS = 250e-6
R_BEND = 8.3e-3
D_LENS = 14e-3
D1_S_CENTER = 2.0e-3
D3_S_CENTER = 1.8e-3
D1_H = 50e-6
D3_LEN = 50e-6
D3_QUERSCHN = 200e-6
N_PMMA = 1.491
N_LIPID = 1.480
N_MUCIN = 1.342
N_CORNEA = 1.376


def get_detector_polygons():
    x_d1_um = (D1_S_CENTER - D_LENS/2)*1e6
    sag_d1 = R_BEND - np.sqrt(R_BEND**2 - (x_d1_um*1e-6)**2)
    y_mid_d1_um = -sag_d1*1e6
    slope = -(x_d1_um*1e-6)/np.sqrt(R_BEND**2 - (x_d1_um*1e-6)**2)
    theta = np.arctan(slope)
    ct, st = np.cos(theta), np.sin(theta)
    L_2D = 100.0
    H_um = D1_H*1e6
    max_y_shift = L_2D/2*abs(st) + H_um/2*abs(ct)
    d_norm = max(max_y_shift/abs(ct), 50)
    def mk(xc, yc):
        return [(xc + ct*u - st*v, yc + st*u + ct*v)
                for u, v in [(-L_2D/2, -H_um/2), (+L_2D/2, -H_um/2),
                             (+L_2D/2, +H_um/2), (-L_2D/2, +H_um/2)]]
    polys = {}
    polys['D1'] = mk(x_d1_um + st*d_norm, y_mid_d1_um - ct*d_norm)
    polys['D2'] = mk(x_d1_um - st*d_norm, y_mid_d1_um + ct*d_norm)
    x_d3_um = (D3_S_CENTER - D_LENS/2)*1e6
    sag_d3 = R_BEND - np.sqrt(R_BEND**2 - (x_d3_um*1e-6)**2)
    y_mid_d3_um = -sag_d3*1e6
    slope_d3 = -(x_d3_um*1e-6)/np.sqrt(R_BEND**2 - (x_d3_um*1e-6)**2)
    th3 = np.arctan(slope_d3)
    ct3, st3 = np.cos(th3), np.sin(th3)
    L3 = D3_LEN*1e6
    Q3 = D3_QUERSCHN*1e6
    polys['D3'] = [
        (x_d3_um + ct3*(-L3/2) - st3*(-Q3/2), y_mid_d3_um + st3*(-L3/2) + ct3*(-Q3/2)),
        (x_d3_um + ct3*(+L3/2) - st3*(-Q3/2), y_mid_d3_um + st3*(+L3/2) + ct3*(-Q3/2)),
        (x_d3_um + ct3*(+L3/2) - st3*(+Q3/2), y_mid_d3_um + st3*(+L3/2) + ct3*(+Q3/2)),
        (x_d3_um + ct3*(-L3/2) - st3*(+Q3/2), y_mid_d3_um + st3*(-L3/2) + ct3*(+Q3/2)),
    ]
    return polys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz', help='Pfad zu sliding_*_frames.npz')
    ap.add_argument('--duration', type=float, default=0.18)
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        print(f'Fehler: {args.npz} nicht gefunden')
        sys.exit(1)

    print(f'Lade {args.npz} ...')
    data = np.load(args.npz)
    ezs = data['Ez']
    meta = data['meta']
    scenario = str(data['scenario'])
    n = len(ezs)
    if 'layers' in data.files:
        layers = data['layers']
    else:
        layers = np.array([0.030e-6, 3.5e-6, 0.5e-6, 1.336])
    print(f'  {n} Frames, Szenario: {scenario}')
    print(f'  Layers: lipid={layers[0]*1e6:.3f}um aq={layers[1]*1e6:.2f}um '
          f'mu={layers[2]*1e6:.2f}um n_aq={layers[3]}')

    cdict = {
        'red':   [(0.0, 0.10, 0.10), (0.30, 0.00, 0.00), (0.49, 0.04, 0.04),
                  (0.51, 0.10, 0.10), (0.70, 1.00, 1.00), (1.0, 1.00, 1.00)],
        'green': [(0.0, 0.85, 0.85), (0.30, 0.90, 0.90), (0.49, 0.06, 0.06),
                  (0.51, 0.10, 0.10), (0.70, 0.55, 0.55), (1.0, 0.95, 0.95)],
        'blue':  [(0.0, 1.00, 1.00), (0.30, 1.00, 1.00), (0.49, 0.10, 0.10),
                  (0.51, 0.04, 0.04), (0.70, 0.00, 0.00), (1.0, 0.30, 0.30)],
    }
    cmap_wave = LinearSegmentedColormap('wave', cdict)
    polys = get_detector_polygons()
    vmax_global = float(np.max(np.abs(ezs)))
    vmax = vmax_global*0.5
    if vmax < 1e-12:
        vmax = 1.0

    imgs = []
    for i in range(n):
        if meta.shape[1] >= 7:
            x_start, x_end, y_start, y_end, slide_i, t, step = meta[i]
        else:
            x_start, x_end, y_start, y_end, slide_i, t = meta[i]
            step = 0
        fig, ax = plt.subplots(figsize=(10, 5), dpi=85, facecolor='#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        ax.imshow(ezs[i].T, extent=[x_start, x_end, y_start, y_end],
                  origin='lower', cmap=cmap_wave, vmin=-vmax, vmax=vmax,
                  aspect='equal', interpolation='bilinear')

        xs_out = np.linspace(x_start, x_end, 200)
        xs_clip = np.clip(xs_out, -R_BEND*1e6 + 1, R_BEND*1e6 - 1)
        sag = (R_BEND - np.sqrt(R_BEND**2 - (xs_clip*1e-6)**2))*1e6
        y_mid_o = -sag
        y_lens_top = y_mid_o + T_LENS*1e6/2
        y_lens_bot = y_mid_o - T_LENS*1e6/2
        t_lip = layers[0]*1e6
        t_aq = layers[1]*1e6
        t_mu = layers[2]*1e6
        y_lipid_bot_o = y_lens_bot - t_lip
        y_aq_bot_o = y_lipid_bot_o - t_aq
        y_mu_bot_o = y_aq_bot_o - t_mu
        ax.plot(xs_out, y_lens_top, '-', color='#FFD75E', lw=1.2, alpha=0.9,
                label=f'PMMA (n={N_PMMA})')
        ax.plot(xs_out, y_lens_bot, '-', color='#FFD75E', lw=1.2, alpha=0.9)
        ax.plot(xs_out, y_lipid_bot_o, ':', color='#88AAFF', lw=0.8, alpha=0.7,
                label=f'Lipid {t_lip:.3f}um (n={N_LIPID})')
        ax.plot(xs_out, y_aq_bot_o, '--', color='#56C4FF', lw=1.0, alpha=0.8,
                label=f'Aqueous {t_aq:.2f}um (n={layers[3]})')
        ax.plot(xs_out, y_mu_bot_o, '--', color='#D88AFF', lw=1.0, alpha=0.8,
                label=f'Mucin {t_mu:.2f}um (n={N_MUCIN})')
        ax.fill_between(xs_out, y_start, y_mu_bot_o, color='#553030', alpha=0.18,
                        label=f'Cornea (n={N_CORNEA})')

        x_stirn_um = D_LENS*1e6/2
        if x_end > x_stirn_um - 400:
            sag_st = (R_BEND - np.sqrt(R_BEND**2 - (x_stirn_um*1e-6)**2))*1e6
            y_mid_st = -sag_st
            slope_st = -x_stirn_um*1e-6/np.sqrt(R_BEND**2 - (x_stirn_um*1e-6)**2)
            th_st = np.arctan(slope_st)
            ct_l, st_l = np.cos(th_st), np.sin(th_st)
            nx, ny = -st_l, ct_l
            x_top = x_stirn_um + (T_LENS*1e6/2)*nx
            y_top = y_mid_st + (T_LENS*1e6/2)*ny
            x_bot = x_stirn_um - (T_LENS*1e6/2)*nx
            y_bot = y_mid_st - (T_LENS*1e6/2)*ny
            ax.plot([x_bot, x_top], [y_bot, y_top], '-', color='#FF55FF', lw=3, alpha=0.95)
            vL, vW = 150, 160
            vxc = x_stirn_um + (vL/2)*ct_l
            vyc = y_mid_st + (vL/2)*st_l
            pts = []
            for u, v in [(-vL/2,-vW/2),(vL/2,-vW/2),(vL/2,vW/2),(-vL/2,vW/2)]:
                pts.append((vxc + ct_l*u + nx*v, vyc + st_l*u + ny*v))
            ax.add_patch(MPoly(pts, closed=True, fill=True,
                               fc='#440000', ec='#FF55FF', lw=1.5, alpha=0.85))
            ax.text(x_stirn_um + (vL+80)*ct_l, y_mid_st + (vL+80)*st_l,
                    'VCSEL\n850 nm', color='#FF55FF', fontsize=9,
                    fontweight='bold', ha='center', va='center')

        if any(x_start < p[0] < x_end for p in polys['D1']):
            ax.add_patch(MPoly(polys['D1'], closed=True, fill=False, ec='#FF8800', lw=2.0))
            cx = sum(p[0] for p in polys['D1'])/4
            cy = sum(p[1] for p in polys['D1'])/4
            ax.text(cx + 80, cy, 'D1', color='#FF8800', fontsize=10,
                    fontweight='bold', va='center')
        if any(x_start < p[0] < x_end for p in polys['D2']):
            ax.add_patch(MPoly(polys['D2'], closed=True, fill=False, ec='#00AAFF', lw=2.0))
            cx = sum(p[0] for p in polys['D2'])/4
            cy = sum(p[1] for p in polys['D2'])/4
            ax.text(cx - 80, cy, 'D2', color='#00AAFF', fontsize=10,
                    fontweight='bold', va='center', ha='right')
        if any(x_start < p[0] < x_end for p in polys['D3']):
            ax.add_patch(MPoly(polys['D3'], closed=True, fill=False, ec='#55FF55', lw=2.0))
            cx = sum(p[0] for p in polys['D3'])/4
            cy = sum(p[1] for p in polys['D3'])/4
            ax.text(cx, cy + 130, 'D3', color='#55FF55', fontsize=10,
                    fontweight='bold', ha='center')

        ax.set_xlim(x_start, x_end)
        ax.set_ylim(y_start, y_end)
        ax.set_xlabel('x (um)', color='#cccccc', fontsize=9)
        ax.set_ylabel('y (um)', color='#cccccc', fontsize=9)
        ax.tick_params(colors='#cccccc', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#444444')
        title = f"{scenario}  -  Slide {int(slide_i)+1}"
        if step:
            title += f"  step {int(step)}"
        title += f"  -  t = {t:.1f}s"
        ax.set_title(title, color='#ffffff', fontsize=10)
        leg = ax.legend(loc='upper left', fontsize=7, framealpha=0.55,
                        facecolor='#1a1a1a', edgecolor='#444444')
        for txt in leg.get_texts():
            txt.set_color('#dddddd')

        fig.tight_layout()
        fig.canvas.draw()
        try:
            rgba = np.asarray(fig.canvas.buffer_rgba())
            buf = rgba[..., :3].copy()
        except AttributeError:
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(),
                                dtype=np.uint8).reshape(h, w, 3).copy()
        imgs.append(buf)
        plt.close(fig)
        print(f'  Frame {i+1}/{n}', end='\r')

    gif_path = args.npz.replace('_frames.npz', '.gif')
    print(f'\nSchreibe {gif_path} ...')
    imageio.mimsave(gif_path, imgs, duration=args.duration, loop=0)
    print(f'  Saved: {gif_path}')


if __name__ == '__main__':
    main()
