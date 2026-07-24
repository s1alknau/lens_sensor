"""GIF-Tuning: existierende Sliding-FDTD-GIFs mit anderer Geschwindigkeit neu speichern.

Liest ein bereits erzeugtes .gif, extrahiert die Frames und speichert sie mit
neuer Bildwiederholrate. Keine Neuberechnung der FDTD noetig!

VERWENDUNG:
  # Einzelne GIF langsamer/schneller machen:
  python regenerate_gif.py results/sliding_Gesund.gif --duration 0.3
  python regenerate_gif.py results/sliding_Gesund.gif --fps 5
  python regenerate_gif.py results/sliding_Gesund.gif --fps 20 --suffix _fast

  # Alle GIFs im results-Ordner umstellen:
  python regenerate_gif.py results/sliding_*.gif --duration 0.4

  # Variabler Speed (langsamer Anfang, schneller Mitte, langsamer Ende):
  python regenerate_gif.py results/sliding_Gesund.gif --easing

  # Frames ueberspringen (kleineres GIF, schnellere Wiedergabe):
  python regenerate_gif.py results/sliding_Gesund.gif --skip 2

  # Reverse (wave runs backwards):
  python regenerate_gif.py results/sliding_Gesund.gif --reverse

PARAMETER:
  --duration FLOAT  Sekunden pro Frame (z.B. 0.1 = schnell, 0.5 = langsam)
  --fps INT         alternative: Frames per second (z.B. 10 = 0.1s pro Frame)
  --skip INT        nur jedes N-te Frame nehmen (z.B. 2 = haelfte der Frames)
  --easing          variable Geschwindigkeit (langsam-schnell-langsam)
  --reverse         Reihenfolge umkehren
  --suffix STR      Output-Datei-Suffix (Default: _tuned)
  --loop INT        0 = unendliche Schleife (Default), 1 = einmal abspielen
"""
import argparse
import glob
import os
import sys
import numpy as np
import imageio.v2 as imageio


def regenerate_gif(input_path, duration=0.18, skip=1, easing=False,
                    reverse=False, suffix='_tuned', loop=0):
    """Lese GIF, transformiere Frames, schreibe neu."""
    if not os.path.exists(input_path):
        print(f'[Fehler] Datei nicht gefunden: {input_path}')
        return None

    print(f'  Lade {input_path} ...', end=' ', flush=True)
    frames = imageio.mimread(input_path, memtest=False)
    n = len(frames)
    print(f'{n} Frames')

    if reverse:
        frames = frames[::-1]
        print('  Reverse: Reihenfolge umgekehrt')

    if skip > 1:
        frames = frames[::skip]
        print(f'  Skip: nur jedes {skip}. Frame ({len(frames)} Frames uebrig)')

    # Output-Pfad
    base, ext = os.path.splitext(input_path)
    out_path = f'{base}{suffix}{ext}'

    if easing:
        # Variabler Speed: langsam (Start), schnell (Mitte), langsam (Ende)
        n_out = len(frames)
        # Sinus-Kurve: pi*i/n gibt Werte zwischen 0 und 1
        speed = np.sin(np.linspace(0, np.pi, n_out))
        # Duration pro Frame: invertiert (hohe Geschwindigkeit = kurze duration)
        durations = (1.0 - 0.6*speed) * duration
        durations = durations.tolist()
        print(f'  Easing: Anfang/Ende langsam ({durations[0]*1000:.0f}ms), '
              f'Mitte schnell ({durations[n_out//2]*1000:.0f}ms)')
        imageio.mimsave(out_path, frames, duration=durations, loop=loop)
    else:
        print(f'  Speichere mit {duration*1000:.0f}ms/Frame ({1/duration:.1f} fps)')
        imageio.mimsave(out_path, frames, duration=duration, loop=loop)

    print(f'  Saved: {out_path}')
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='Pfad(e) zu input GIF(s) (Wildcards moeglich)')
    ap.add_argument('--duration', type=float, default=0.18,
                     help='Sekunden pro Frame (Default: 0.18)')
    ap.add_argument('--fps', type=float, default=None,
                     help='Frames per second (alternative zu --duration)')
    ap.add_argument('--skip', type=int, default=1,
                     help='Nur jedes N-te Frame nehmen (Default: 1)')
    ap.add_argument('--easing', action='store_true',
                     help='Variable Geschwindigkeit (langsam-schnell-langsam)')
    ap.add_argument('--reverse', action='store_true',
                     help='Frames in umgekehrter Reihenfolge')
    ap.add_argument('--suffix', default='_tuned',
                     help='Output-Suffix (Default: _tuned)')
    ap.add_argument('--loop', type=int, default=0,
                     help='Loops (0=unendlich, 1=einmal)')
    args = ap.parse_args()

    # Falls --fps gegeben, ueberschreibe duration
    if args.fps is not None:
        args.duration = 1.0/args.fps

    # Wildcards aufloesen
    all_files = []
    for inp in args.inputs:
        matches = glob.glob(inp)
        if matches:
            all_files.extend(matches)
        else:
            all_files.append(inp)

    if not all_files:
        print('Keine Dateien gefunden!')
        sys.exit(1)

    print(f'Verarbeite {len(all_files)} GIF(s)...\n')
    for i, fp in enumerate(all_files):
        print(f'[{i+1}/{len(all_files)}]')
        regenerate_gif(fp, duration=args.duration, skip=args.skip,
                        easing=args.easing, reverse=args.reverse,
                        suffix=args.suffix, loop=args.loop)
        print()


if __name__ == '__main__':
    main()
