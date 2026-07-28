r"""Interaktiver Frame-Browser fuer Sliding-FDTD-GIFs.

Zeigt das GIF mit einem Slider an. Du kannst frei durch die Slides scrollen,
einzelne Frames als PNG exportieren, oder einen Bereich als neues GIF ausschneiden.

VERWENDUNG:
  python view_gif.py results\sliding_Gesund.gif

STEUERUNG:
  - Slider unten:        Frame waehlen
  - Pfeiltaste LINKS:    1 Frame zurueck
  - Pfeiltaste RECHTS:   1 Frame vor
  - SHIFT+LINKS/RECHTS:  10 Frames zurueck/vor
  - Taste S:             aktuellen Frame als PNG speichern
  - Taste A:             "Start" fuer Trim-Bereich setzen (auf aktuellem Frame)
  - Taste B:             "Ende" fuer Trim-Bereich setzen (auf aktuellem Frame)
  - Taste T:             Trim-Bereich [A, B] als neues GIF speichern
  - Taste Q oder ESC:    schliessen
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import imageio.v2 as imageio


def view_gif(gif_path):
    if not os.path.exists(gif_path):
        print(f'Fehler: {gif_path} nicht gefunden')
        sys.exit(1)

    print(f'Lade {gif_path} ...')
    frames = imageio.mimread(gif_path, memtest=False)
    n = len(frames)
    print(f'  {n} Frames geladen')
    print(f'  Frame-Groesse: {frames[0].shape}\n')

    base = os.path.splitext(gif_path)[0]

    # State: aktueller Index, Trim-Range
    state = {'idx': 0, 'a': 0, 'b': n - 1}

    fig, ax = plt.subplots(figsize=(13, 7.5))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.18)
    img = ax.imshow(frames[0])
    ax.axis('off')

    def update_title():
        ax.set_title(
            f'{os.path.basename(gif_path)}    '
            f'Frame {state["idx"]+1} / {n}    '
            f'Trim-Bereich: [{state["a"]+1}, {state["b"]+1}]    '
            f'(S=Save  A/B=Trim-Start/Ende  T=Trim-GIF  Q=Quit)',
            fontsize=10)
    update_title()

    # Slider
    ax_slider = plt.axes([0.10, 0.07, 0.80, 0.035])
    slider = Slider(ax_slider, 'Slide', 0, n-1, valinit=0, valstep=1)

    # Trim-Indicator-Linien auf dem Slider (zeigt A und B)
    trim_ax = plt.axes([0.10, 0.03, 0.80, 0.02], facecolor='#eeeeee')
    trim_ax.set_xlim(0, n-1)
    trim_ax.set_ylim(0, 1)
    trim_ax.set_xticks([])
    trim_ax.set_yticks([])
    trim_a_line = trim_ax.axvline(state['a'], color='green', lw=2, label='A')
    trim_b_line = trim_ax.axvline(state['b'], color='red', lw=2, label='B')
    trim_span = trim_ax.axvspan(state['a'], state['b'], color='yellow', alpha=0.3)
    trim_ax.text(0.01, 0.5, 'Trim:', transform=trim_ax.transAxes, fontsize=8,
                  ha='left', va='center')

    def show_frame(idx):
        state['idx'] = idx
        img.set_data(frames[idx])
        update_title()
        slider.eventson = False
        slider.set_val(idx)
        slider.eventson = True
        fig.canvas.draw_idle()

    def update_trim_visual():
        trim_a_line.set_xdata([state['a'], state['a']])
        trim_b_line.set_xdata([state['b'], state['b']])
        trim_span.set_xy([[state['a'], 0], [state['a'], 1],
                          [state['b'], 1], [state['b'], 0]])
        update_title()
        fig.canvas.draw_idle()

    def slider_update(val):
        show_frame(int(val))
    slider.on_changed(slider_update)

    def save_current_frame():
        idx = state['idx']
        out = f'{base}_frame{idx:03d}.png'
        imageio.imwrite(out, frames[idx])
        print(f'[Save] Frame {idx+1} -> {out}')

    def save_trim_gif():
        a, b = min(state['a'], state['b']), max(state['a'], state['b'])
        if a == b:
            print('[Trim] Kein Bereich gewaehlt (A==B)')
            return
        out_frames = frames[a:b+1]
        out_path = f'{base}_trim_{a:03d}-{b:03d}.gif'
        imageio.mimsave(out_path, out_frames, duration=0.15, loop=0)
        print(f'[Trim] Frames {a+1}..{b+1} ({len(out_frames)} Bilder) -> {out_path}')

    def on_key(event):
        if event.key == 'right':
            show_frame(min(state['idx'] + 1, n - 1))
        elif event.key == 'left':
            show_frame(max(state['idx'] - 1, 0))
        elif event.key == 'shift+right':
            show_frame(min(state['idx'] + 10, n - 1))
        elif event.key == 'shift+left':
            show_frame(max(state['idx'] - 10, 0))
        elif event.key in ('s', 'S'):
            save_current_frame()
        elif event.key in ('a', 'A'):
            state['a'] = state['idx']
            print(f'[Trim] Start auf Frame {state["a"]+1} gesetzt')
            update_trim_visual()
        elif event.key in ('b', 'B'):
            state['b'] = state['idx']
            print(f'[Trim] Ende auf Frame {state["b"]+1} gesetzt')
            update_trim_visual()
        elif event.key in ('t', 'T'):
            save_trim_gif()
        elif event.key in ('q', 'Q', 'escape'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Save-Button (alternative zum 'S'-Key)
    ax_btn_save = plt.axes([0.92, 0.15, 0.07, 0.04])
    btn_save = Button(ax_btn_save, 'Save\nFrame')
    btn_save.on_clicked(lambda e: save_current_frame())

    # Trim-Button
    ax_btn_trim = plt.axes([0.92, 0.10, 0.07, 0.04])
    btn_trim = Button(ax_btn_trim, 'Save\nTrim')
    btn_trim.on_clicked(lambda e: save_trim_gif())

    print('Steuerung:')
    print('  PFEIL LINKS/RECHTS = +/- 1 Frame')
    print('  SHIFT+PFEIL        = +/- 10 Frames')
    print('  S                  = aktuellen Frame als PNG speichern')
    print('  A                  = Trim-Anfang setzen (auf aktuellem Frame)')
    print('  B                  = Trim-Ende setzen (auf aktuellem Frame)')
    print('  T                  = Trim-Bereich [A..B] als neues GIF speichern')
    print('  Q oder ESC         = Schliessen\n')

    plt.show()


def main():
    ap = argparse.ArgumentParser(description='Interaktiver Frame-Browser fuer Sliding-FDTD-GIFs')
    ap.add_argument('gif', help='Pfad zum GIF')
    args = ap.parse_args()
    view_gif(args.gif)


if __name__ == '__main__':
    main()
