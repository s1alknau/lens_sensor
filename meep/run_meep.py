"""Meep-Treiber: rechnet dieselben Faelle wie run_simulation.py, aber mit Meep.

Nutzt DENSELBEN Argument-Parser und dieselbe Parameter-Aufloesung wie
run_simulation.py (Single Source of Truth) und schreibt analyzer-kompatible NPZ.
Laeuft nur unter Linux/WSL. Normal wird es NICHT direkt aufgerufen, sondern ueber
`run_simulation.py --engine meep` (das unter Windows automatisch nach WSL auslagert).

Meep-Besonderheit: die Methoden 'sliding'/'stitch' des eigenen Solvers sind
Speicher-Tricks, die Meep nicht braucht -> hier auf 'full' abgebildet (mit Hinweis).
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                               # noqa: E402
import run_simulation as rs                     # noqa: E402  (Parser + Aufloesung)
from common.physics import n_at                 # noqa: E402
import meep_backend as mb                        # noqa: E402
import meep_to_npz as mnpz                       # noqa: E402


def _build_cfg(args, layers):
    """Baut das Meep-Config-Dict aus den (bereits aufgeloesten) Argumenten."""
    lam = args.lambda_nm
    # Schicht-/Index-Defaults exakt wie run_simulation._run_3d
    t_aq = layers['t_aq_um'] if layers['t_aq_um'] is not None else 4.0
    t_mu = layers['t_mu_um'] if layers['t_mu_um'] is not None else 2.0
    t_lip = layers['t_lip_um'] or 0.0
    n_aq = layers['n_aq'] if layers['n_aq'] is not None else n_at('aqueous', lam)
    n_mu = layers['n_mu'] if layers['n_mu'] is not None else n_at('mucin', lam)
    n_co = layers['n_co'] if layers['n_co'] is not None else n_at('cornea', lam)
    n_lip = layers['n_lip'] if layers['n_lip'] is not None else 1.480

    lx = args.length_um
    air, tear = args.air, args.tear
    t_wg = layers['t_wg_um']
    y_top = t_wg + air
    y_bot = -tear
    ly = y_top - y_bot
    lz = args.lz_um

    bead = None
    if rs._place_bead(args):
        r = args.bead_diameter/2.0
        by = getattr(args, 'bead_y_um', None)
        bead = dict(x=(args.bead_x if args.bead_x is not None else lx/2.0),
                    y=(by if by is not None else -r), z=0.0, r=r,
                    n=(args.bead_n if args.bead_n is not None
                       else n_at(args.bead_material, lam)))

    resolution = 1000.0/args.resolution_nm
    label = f'{args.geometry}{args.dim}d_meep_' + \
            (args.scenario if args.geometry == 'lens' else args.bead_material)
    # Quelle + Flux-Monitore konsistent zur (adaptiven) PML platzieren:
    # Quelle knapp hinter dem linken PML, Monitore STROMABWAERTS der Quelle,
    # det_out kurz vor dem rechten PML. (det_in links der Quelle -> negativer Flux.)
    dims = [lx, ly] + ([lz] if args.dim == 3 else [])
    dpml = max(0.1, min(lam/1000.0, 0.4*min(dims)))
    src_x = dpml + 0.5
    det_in_x = src_x + 0.5
    det_out_x = max(det_in_x + 0.5, lx - dpml - 0.5)
    return dict(
        label=label, dim=args.dim, lx=lx, ly=ly, lz=lz,
        y_bot=y_bot, y_top=y_top, cx=lx/2.0, cy=0.5*(y_top + y_bot),
        t_wg=t_wg, wg_n=layers['wg_n'], t_lip=t_lip, t_aq=t_aq, t_mu=t_mu,
        n_lip=n_lip, n_aq=n_aq, n_mu=n_mu, n_co=n_co, bead=bead,
        lam_nm=lam, f0=1000.0/lam, resolution=resolution,
        pol=args.polarization, source_type=args.source_type,
        n_snapshots=args.snapshots, vcsel_waist=args.vcsel_waist,
        vcsel_waist_z=args.vcsel_waist_z, wg_width=args.wg_width,
        src_x=src_x, det_in_x=det_in_x, det_out_x=det_out_x, run_time=None)


def main(argv=None):
    args = rs.build_parser().parse_args(argv)
    layers = rs.resolve_args(args)
    # Meep-Fenster-Stitch ist der planar-2D-Prototyp (Mode-Kaskade). Sonst full.
    use_meep_stitch = (args.method == 'stitch' and args.dim == 2
                       and args.geometry == 'planar')
    if args.method in ('sliding', 'stitch') and not use_meep_stitch:
        print(f'[meep] Methode "{args.method}" hier nicht als Stitch verfuegbar '
              f'(Meep-Fenster-Stitch nur planar 2D) -> Full-Domain.')

    cfg = _build_cfg(args, layers)
    print('===================== MEEP-Konfiguration =====================')
    print(f'  Geometrie : {"Kontaktlinse" if args.geometry=="lens" else "planar"}  '
          f'({args.dim}D)   Pol: {cfg["pol"]}')
    if args.geometry == 'lens':
        print(f'  Szenario  : {args.scenario}')
    print(f'  Domain    : x={cfg["lx"]:g} y={cfg["ly"]:g}'
          + (f' z={cfg["lz"]:g}' if args.dim == 3 else '') + ' um   '
          f'Aufloesung {cfg["resolution"]:g} px/um  (f0={cfg["f0"]:.3f})')
    bead_desc = 'nein' if not cfg['bead'] else f'd={2*cfg["bead"]["r"]:g}um'
    print(f'  Waveguide : {cfg["t_wg"]:g} um  n={cfg["wg_n"]:.3f}   Bead: {bead_desc}')
    if use_meep_stitch:
        print('  Methode   : Fenster-Stitch (Meep, Mode-Kaskade, 1 Fenster im RAM)')
    print('==============================================================')

    out_dir = os.path.join(_REPO_ROOT, 'results')

    # --- Meep-Fenster-Stitch (planar 2D): Speicher unabhaengig von der Laenge ---
    if use_meep_stitch:
        if args.dry_run:
            print('[dry-run] Meep-Stitch-Konfiguration aufgeloest - kein Lauf.')
            return 0
        import meep_stitch as ms
        if getattr(args, 'meep_handoff', 'mode') == 'field':
            st = ms.run_stitch_fullfield(cfg, args.window_um, args.slide_um)
        else:
            st = ms.run_stitch(cfg, args.window_um, args.slide_um,
                               modes=getattr(args, 'meep_modes', 1))
        if not args.no_save:
            Efull = st['Efull']
            nfr = max(2, args.snapshots)
            wct = 2*np.pi*cfg['f0']
            times = [k/(nfr*cfg['f0']) for k in range(nfr)]     # 1 Periode
            frames = [np.real(Efull*np.exp(1j*wct*t)).astype(np.float32)
                      for t in times]
            mnpz.save_2d(cfg, dict(frames=frames, times=times,
                                   transmission=st['transmission']), out_dir)
        return 0

    # --- Full-Domain: adaptiver Zellzahl-/RAM-Schutz (Speicher ~ Zellzahl) ---
    res = cfg['resolution']
    ncells = cfg['lx']*res * cfg['ly']*res * (cfg['lz']*res if args.dim == 3 else 1)
    # Realistischer Speicher pro Zelle. Meep-3D braucht weit mehr als die nackten
    # Feld-Arrays: 6 Feldkomponenten KOMPLEX (force_complex_fields) + D/B + PML-
    # Hilfsfelder + DFT-Monitore + die vollen 3D-Feld-Volumina fuers NPZ. Empirisch
    # hat ~18M Zellen eine 15-GB-Box in OOM getrieben -> ~700 B/Zelle statt 260.
    bpc = 700 if args.dim == 3 else 120
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except Exception:
        avail = 8e9
    max_cells = (0.5 if args.dim == 3 else 0.6)*avail/bpc
    print(f'  Gitter    : ~{ncells/1e6:.1f}M Zellen  (~{ncells*bpc/1e9:.1f} GB; '
          f'adaptives Limit ~{max_cells/1e6:.0f}M bei {avail/1e9:.1f} GB frei)')
    if ncells > max_cells and not args.dry_run and not getattr(args, 'allow_large', False):
        raise SystemExit(
            f'[ABBRUCH] ~{ncells/1e6:.0f}M Zellen (~{ncells*bpc/1e9:.1f} GB) '
            f'> adaptives Limit ~{max_cells/1e6:.0f}M ({avail/1e9:.1f} GB frei).\n'
            f'  Meep rechnet FULL-DOMAIN (kein Stitch) - Optionen:\n'
            f'   - groebere --resolution-nm / kuerzere --length-um '
            f'(3D auch kleinere --lz-um/--tear/--air)\n'
            f'   - lange/planare Domaene: Meep-Fenster-Stitch -> --method stitch\n'
            f'   - grosse Linse: eigener Solver -> --engine native --method stitch\n'
            f'   - bewusst erzwingen: --allow-large  (Risiko: OOM)')

    if args.dry_run:
        print('[dry-run] Meep-Konfiguration aufgeloest - kein Lauf gestartet.')
        return 0

    result = mb.run(cfg)
    if not args.no_save:
        if args.dim == 2:
            mnpz.save_2d(cfg, result, out_dir)
        else:
            mnpz.save_3d(cfg, result, out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
