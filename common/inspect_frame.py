"""Interaktiver Frame-Inspector fuer Sliding-FDTD _frames.npz

Laedt einen Frame in voller Aufloesung und zeigt ihn mit
Matplotlib's eingebauter Zoom-/Pan-Funktion. Damit kannst du
in die Schicht-Region reinzoomen und das evaneszente Feld
direkt an den Grenzflaechen beurteilen.

Verwendung:
  python inspect_frame.py results/sliding_Gesund_frames.npz
  python inspect_frame.py results/sliding_Gesund_frames.npz --frame 42
  python inspect_frame.py results/sliding_Gesund_frames.npz --slide 25 --step 4
  python inspect_frame.py results/sliding_Gesund_frames.npz --log
  python inspect_frame.py results/sliding_Gesund_frames.npz --abs --log

OPTIONEN:
  --frame N      Frame-Index direkt waehlen (0-basiert)
  --slide N      Slide-Nummer (1-basiert), zusammen mit --step
  --step N       Step-Nummer innerhalb des Slides
  --log          log10(|Ez|) Skala (evaneszent sichtbar machen)
  --abs          |Ez| statt Ez (Intensitaet)
  --vmax F       manuelles Symmetrie-Maximum (Default: auto)
  --save FN.png  Speichere statt anzuzeigen

ZOOM-BEDIENUNG (in der Matplotlib-Toolbar oben):
  - Lupe (+/-): klicke und ziehe ein Rechteck zum Reinzoomen
  - Hand-Symbol: Pan / Verschieben des Sichtfeldes
  - Home-Symbol: Zoom auf Ausgangsansicht zuruecksetzen
  - Disc-Symbol: aktuelle Ansicht als PNG speichern

WICHTIG fuer evaneszentes Feld:
  Bei dx=400 nm ist die 30 nm Lipid-Schicht weniger als 1 Pixel!
  Fuer aussagekraeftige evaneszente Beurteilung dx<=100 nm verwenden:
    python sliding_window_fdtd.py --scenario Gesund --gpu --resolution 100 ...
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm
from matplotlib.patches import Polygon as MPoly

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
    polys = {'D1': mk(x_d1_um + st*d_norm, y_mid_d1_um - ct*d_norm),
             'D2': mk(x_d1_um - st*d_norm, y_mid_d1_um + ct*d_norm)}
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
    ap.add_argument('npz', help='Pfad zu _frames.npz')
    ap.add_argument('--frame', type=int, default=None, help='Frame-Index (0-basiert)')
    ap.add_argument('--slide', type=int, default=None, help='Slide-Nummer (1-basiert)')
    ap.add_argument('--step', type=int, default=None,
                    help='Step-Nummer (zusammen mit --slide)')
    ap.add_argument('--log', action='store_true',
                    help='log-Skala fuer evaneszentes Feld')
    ap.add_argument('--abs', dest='take_abs', action='store_true',
                    help='|Ez| statt Ez')
    ap.add_argument('--vmax', type=float, default=None,
                    help='Manuelles Maximum (Standard: auto)')
    ap.add_argument('--save', default=None, help='Speichere als PNG statt anzeigen')
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
    print(f'  {n} Frames')

    # Frame auswaehlen
    if args.slide is not None:
        # Suche Frame mit gegebenem Slide (und optional Step)
        target_slide = args.slide - 1   # User gibt 1-basiert
        candidates = [i for i in range(n) if int(meta[i, 4]) == target_slide]
        if not candidates:
            print(f'Fehler: kein Frame mit Slide {args.slide}')
            sys.exit(1)
        if args.step is not None and meta.shape[1] >= 7:
            # exakter Step
            best = candidates[0]
            best_diff = abs(int(meta[best, 6]) - args.step)
            for c in candidates[1:]:
                d = abs(int(meta[c, 6]) - args.step)
                if d < best_diff:
                    best, best_diff = c, d
            frame_idx = best
        else:
            frame_idx = candidates[len(candidates)//2]   # mittlerer
    elif args.frame is not None:
        frame_idx = args.frame
    else:
        frame_idx = n // 2   # Default: mittlerer Frame
    frame_idx = max(0, min(n-1, frame_idx))

    if meta.shape[1] >= 7:
        x_start, x_end, y_start, y_end, slide_i, t, step = meta[frame_idx]
    else:
        x_start, x_end, y_start, y_end, slide_i, t = meta[frame_idx]
        step = 0

    print(f'  Frame {frame_idx+1}/{n}, Slide {int(slide_i)+1}, step {int(step)}, t={t:.1f}s')
    print(f'  Bereich: x=[{x_start:.0f}, {x_end:.0f}] um, y=[{y_start:.0f}, {y_end:.0f}] um')

    Ez = ezs[frame_idx]
    Nx, Ny = Ez.shape
    print(f'  Aufloesung: {Nx} x {Ny} Pixel  ->  dx ~ {(x_end-x_start)/Nx*1000:.0f} nm')

    # Daten vorbereiten
    if args.take_abs:
        data_show = np.abs(Ez).T
        clabel = '|Ez|'
    else:
        data_show = Ez.T
        clabel = 'Ez'

    # Colormap + Norm
    if args.log:
        if not args.take_abs:
            # SymLog: zeigt negativ + positiv in log
            vmax_eff = args.vmax if args.vmax else float(np.max(np.abs(data_show)))
            linthresh = vmax_eff * 1e-4
            norm = SymLogNorm(linthresh=linthresh, vmin=-vmax_eff, vmax=vmax_eff)
            cmap_name = 'RdBu_r'
        else:
            # log10(|Ez|)
            data_pos = np.maximum(data_show, 1e-30)
            vmax_eff = args.vmax if args.vmax else float(np.max(data_pos))
            norm = LogNorm(vmin=vmax_eff*1e-6, vmax=vmax_eff)
            cmap_name = 'inferno'
            data_show = data_pos
    else:
        vmax_eff = args.vmax if args.vmax else float(np.max(np.abs(data_show)))
        if args.take_abs:
            norm = None
            cmap_name = 'inferno'
        else:
            norm = None
            cmap_name = 'RdBu_r'

    # Plot
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    ext = [x_start, x_end, y_start, y_end]
    if norm is not None:
        im = ax.imshow(data_show, extent=ext, origin='lower', cmap=cmap_name,
                       norm=norm, aspect='equal', interpolation='nearest')
    else:
        if args.take_abs:
            im = ax.imshow(data_show, extent=ext, origin='lower', cmap=cmap_name,
                           vmin=0, vmax=vmax_eff, aspect='equal',
                           interpolation='nearest')
        else:
            vm = vmax_eff*0.5
            im = ax.imshow(data_show, extent=ext, origin='lower', cmap=cmap_name,
                           vmin=-vm, vmax=vm, aspect='equal',
                           interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label(clabel + (' (log)' if args.log else ''), color='#cccccc')
    cbar.ax.tick_params(colors='#cccccc', labelsize=8)

    # Schicht-Outlines
    xs_out = np.linspace(x_start, x_end, 400)
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
    ax.plot(xs_out, y_lens_top, '-', color='#FFD75E', lw=1.0, alpha=0.8,
            label=f'PMMA top (n={N_PMMA})')
    ax.plot(xs_out, y_lens_bot, '-', color='#FFD75E', lw=1.0, alpha=0.8,
            label=f'PMMA bot')
    ax.plot(xs_out, y_lipid_bot_o, ':', color='#88AAFF', lw=1.0, alpha=0.85,
            label=f'Lipid {t_lip:.3f}um (n={N_LIPID})')
    ax.plot(xs_out, y_aq_bot_o, '--', color='#56C4FF', lw=1.0, alpha=0.85,
            label=f'Aqueous {t_aq:.2f}um (n={layers[3]})')
    ax.plot(xs_out, y_mu_bot_o, '--', color='#D88AFF', lw=1.0, alpha=0.85,
            label=f'Mucin {t_mu:.2f}um (n={N_MUCIN})')

    # Detektoren (falls im Frame)
    polys = get_detector_polygons()
    for name, ec in [('D1','#FF8800'), ('D2','#00AAFF'), ('D3','#55FF55')]:
        if any(x_start < p[0] < x_end for p in polys[name]):
            ax.add_patch(MPoly(polys[name], closed=True, fill=False, ec=ec, lw=1.5))

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_start, y_end)
    ax.set_xlabel('x (um)', color='#cccccc')
    ax.set_ylabel('y (um)', color='#cccccc')
    ax.tick_params(colors='#cccccc')
    for sp in ax.spines.values():
        sp.set_color('#444444')
    title = f"{scenario}  -  Slide {int(slide_i)+1}"
    if step:
        title += f"  step {int(step)}"
    title += f"  -  t = {t:.1f}s"
    if args.log:
        title += '  [LOG]'
    if args.take_abs:
        title += '  |Ez|'
    ax.set_title(title, color='#ffffff', fontsize=11)
    leg = ax.legend(loc='upper left', fontsize=7, framealpha=0.6,
                    facecolor='#1a1a1a', edgecolor='#444444')
    for txt in leg.get_texts():
        txt.set_color('#dddddd')

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, facecolor=fig.get_facecolor())
        print(f'  Gespeichert: {args.save}')
    else:
        print('\nZoom-Toolbar oben verwenden:')
        print('  Lupe-Symbol: Rechteck-Zoom in die Schicht-Region')
        print('  Hand-Symbol: Pan / Verschieben')
        print('  Home-Symbol: zurueck zur Vollansicht')
        print('  Disk-Symbol: PNG der aktuellen Ansicht speichern\n')
        plt.show()


if __name__ == '__main__':
    main()
