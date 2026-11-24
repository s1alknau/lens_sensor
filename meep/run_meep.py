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
from common.physics import n_at, cornea_default   # noqa: E402
import meep_backend as mb                        # noqa: E402
import meep_to_npz as mnpz                       # noqa: E402


def _load_cad(args):
    """Vorberechnetes CAD-eps-Gitter (run_simulation._prepare_meep_cad, Windows-Seite)."""
    import json
    path = getattr(args, 'meep_cad_eps', None)
    if not path or not os.path.exists(path):
        raise SystemExit('[meep] --geometry cad: eps grid missing (prepared by run_simulation.py '
                         '--engine meep on Windows; --meep-cad-eps).')
    d = np.load(path)
    return dict(eps=np.asarray(d['eps']), region=np.asarray(d['region']),
                dx=float(d['dx_um']), lx=float(d['lx_um']), ly=float(d['ly_um']),
                lz=float(d['lz_um']), y0=float(d['y0_um']), z0=float(d['z0_um']),
                x_src=float(d['x_src_um']), src_y=float(d['src_y_um']),
                src_z=float(d['src_z_um']), src_tilt=float(d['src_tilt_deg']),
                det_in_x=float(d['det_in_x_um']), det_out_x=float(d['det_out_x_um']),
                meta=json.loads(str(d['meta_json'])),
                stem=os.path.splitext(os.path.basename(path))[0].replace('_meep_cad_', ''))


def _warn_unsupported(args):
    """GUI-/CLI-Optionen, die Meep (noch) nicht abbildet, klar melden statt still zu ignorieren."""
    msgs = []
    if args.boundary and str(args.boundary).lower() != 'cpml':
        msgs.append(f'--boundary {args.boundary}: Meep always uses its own PML (~1 lambda)')
    if args.method == 'sliding':
        msgs.append('--method sliding: not available in Meep (-> full)')
    if args.source_type == 'pulse' and args.method == 'stitch':
        msgs.append('Pulse stitch (Fourier/time domain): native only; Meep computes the pulse full-domain')
    if getattr(args, 'pec_faces', ''):
        msgs.append('--pec-faces: native only')
    if getattr(args, 'save_vector', False):
        msgs.append('--save-vector: Meep saves only the driven component')
    if getattr(args, 'kappa', None) and (getattr(args, 'curved', False) or args.geometry == 'cad'):
        msgs.append('Absorption (kappa) with --curved/CAD: Meep reads a real-valued eps grid there '
                    '-> computed without loss (native includes it)')
    for m in msgs:
        print('[meep] Note: ' + m)


def _build_cfg(args, layers):
    """Baut das Meep-Config-Dict aus den (bereits aufgeloesten) Argumenten."""
    lam = args.lambda_nm
    # Schicht-/Index-Defaults exakt wie run_simulation._run_3d (native): physiologischer
    # Traenenfilm (3.5 um Aqueous, 0.5 um Mucin) und geschichtete Cornea (Epithel ...).
    t_aq = layers['t_aq_um'] if layers['t_aq_um'] is not None else 3.5
    t_mu = layers['t_mu_um'] if layers['t_mu_um'] is not None else 0.5
    t_lip = layers['t_lip_um'] or 0.0
    n_aq = layers['n_aq'] if layers['n_aq'] is not None else n_at('aqueous', lam)
    n_mu = layers['n_mu'] if layers['n_mu'] is not None else n_at('mucin', lam)
    n_co = layers['n_co'] if layers['n_co'] is not None else cornea_default(lam)
    n_lip = layers['n_lip'] if layers['n_lip'] is not None else n_at('lipid', lam)

    lx = args.length_um
    t_wg = layers['t_wg_um']
    lz = args.lz_um
    # Meep-PML liegt INNERHALB der Zelle (Dicke ~1 lambda) -> Tear-Auto wie native:
    # Schichten + 1.5 um Cornea + PML-Dicke.
    dpml = lam/1000.0
    air = args.air
    if args.tear is not None:
        tear = args.tear
    elif getattr(args, 'cornea_full', False):       # bis ins Kammerwasser
        from common.physics import cornea_full_depth_um
        tear = t_lip + t_aq + t_mu + cornea_full_depth_um() + dpml
    else:
        tear = t_lip + t_aq + t_mu + 1.5 + dpml
    if args.dim == 2:
        # 2D wie native (fdtd2d_core: AIR_BUFFER 5 um, TEAR_BUFFER 10 um inkl. Absorber).
        # Eine kleinere Domaene laesst die Quellen-Abstrahlung anders im Gebiet stehen
        # (Engine-Vergleich: Streufeld am Bead +6 dB bei 3/6.4 um statt 5/10 um).
        if air == 3.0:                                # --air-Default ist der 3D-Wert
            air = 5.0
        if args.tear is None:
            tear = max(tear, 10.0)
    y_top = t_wg + air
    y_bot = -tear
    ly = y_top - y_bot

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
            (args.scenario if args.geometry == 'lens' else args.bead_material) + \
            (f'_d{args.bead_diameter:g}' if bead else '_ohneBead')
    # Quelle + Flux-Monitore konsistent zur (adaptiven) PML platzieren:
    # Quelle knapp hinter dem linken PML, Monitore STROMABWAERTS der Quelle,
    # det_out kurz vor dem rechten PML. (det_in links der Quelle -> negativer Flux.)
    dims = [lx, ly] + ([lz] if args.dim == 3 else [])
    dpml = max(0.1, min(dpml, 0.4*min(dims)))
    src_x = dpml + 0.5
    # Flussebenen wie native bei 10 % / 90 % der Laenge (ganzer Querschnitt),
    # mindestens 0.5 um hinter der Quelle bzw. vor dem rechten PML
    _fi, _fo = (0.1, 0.9) if args.dim == 2 else (0.2, 0.8)     # wie native (2D/3D)
    det_in_x = max(src_x + 0.5, _fi*lx)
    det_out_x = max(det_in_x + 0.5, min(_fo*lx, lx - dpml - 0.5))
    # GUI-Parameter, die native kennt: Einkoppelabstand, Endfacette (3D full)
    extra = dict(input_gap=float(args.input_gap or 0.0),
                 end_facet=float(getattr(args, 'end_facet', 0.0) or 0.0),
                 curved=bool(getattr(args, 'curved', False) and args.geometry == 'lens'),
                 r_bend_um=8300.0, lx_total=lx)   # R wie native (R_BEND = 8.3 mm)
    cadd = None
    if args.geometry == 'cad':
        cadd = _load_cad(args)
        lx, ly = cadd['lx'], cadd['ly']
        lz = cadd['lz'] if args.dim == 3 else lz
        y_bot, y_top = cadd['y0'], cadd['y0'] + cadd['ly']
        src_x = cadd['x_src']
        det_in_x, det_out_x = cadd['det_in_x'], cadd['det_out_x']
        label = 'cad' + str(args.dim) + 'd_meep_' + cadd['stem']
        extra.update(cad=cadd, src_y=cadd['src_y'], src_z=cadd['src_z'] - (cadd['z0'] + lz/2.0),
                     cad_tilt=cadd['src_tilt'])
    return dict(**extra,
        label=label, dim=args.dim, lx=lx, ly=ly, lz=lz, layer_names=layers.get('names'),
        y_bot=y_bot, y_top=y_top, cx=lx/2.0, cy=0.5*(y_top + y_bot),
        t_wg=t_wg, wg_n=layers['wg_n'], t_lip=t_lip, t_aq=t_aq, t_mu=t_mu,
        n_lip=n_lip, n_aq=n_aq, n_mu=n_mu, n_co=n_co, bead=bead,
        lam_nm=lam, f0=1000.0/lam, resolution=resolution,
        pol=args.polarization, source_type=args.source_type,
        n_snapshots=args.snapshots, vcsel_waist=args.vcsel_waist,
        vcsel_waist_z=args.vcsel_waist_z, wg_width=args.wg_width,
        src_x=src_x, det_in_x=det_in_x, det_out_x=det_out_x, run_time=None, dpml=dpml,
        vcsel_offset=args.vcsel_offset, vcsel_offset_z=args.vcsel_offset_z,
        vcsel_tilt=args.vcsel_tilt, pulse_cycles=args.pulse_cycles,
        source_core_only=bool(getattr(args, 'source_core_only', False)),
        kappa=dict(getattr(args, 'kappa', {}) or {}))


def _native_x_layout(cfg):
    """Full-Domain (nicht CAD): x-Aufteilung wie native. Native rechnet auf [0, L] mit
    10 CPML-Zellen an jedem Ende und der Quelle 2 (2D) bzw. 6 (3D) Zellen dahinter.
    Meep-PML (~1 lambda) wird AUSSERHALB dieser Absorberzone angefuegt: Zelle
    [10dx - dpml, L - 10dx + dpml], Quelle an derselben Stelle wie native. Ohne das
    liegt die Meep-Quelle ~1 um weiter rechts und das Schwebungsmuster der Kernmoden
    (und damit das lokale Feld am Bead) ist verschoben. Ausgegeben wird [0, L]."""
    dx = 1.0/cfg['resolution']
    L = cfg['lx']; dpml = cfg['dpml']
    xa = 10*dx - dpml; xb = L - 10*dx + dpml
    cfg.update(lx=xb - xa, cx=0.5*(xa + xb), gx0=xa, lx_total=L, view_x=(0.0, L),
               src_x=(12 if cfg['dim'] == 2 else 16)*dx)
    # Flussebenen wie native: 2D bei 10 %/90 %, 3D bei 20 %/80 % der Laenge
    f_in, f_out = (0.1, 0.9) if cfg['dim'] == 2 else (0.2, 0.8)
    cfg['det_in_x'] = max(cfg['src_x'] + 0.5, f_in*L)
    cfg['det_out_x'] = max(cfg['det_in_x'] + 0.5, f_out*L)
    return cfg


def main(argv=None):
    args = rs.build_parser().parse_args(argv)
    layers = rs.resolve_args(args)
    # Meep-Fenster-Stitch: planar 2D wahlweise Mode-Kaskade/Feld; 3D, Linse und CAD
    # (oder --meep-handoff vector) mit dem Vektor-Stitch (alle tangentialen E/H).
    use_meep_stitch = (args.method == 'stitch' and args.source_type != 'pulse')
    use_vector = use_meep_stitch and (args.dim == 3 or args.geometry != 'planar'
                                      or getattr(args, 'meep_handoff', 'mode') == 'vector')
    if args.method == 'sliding' or (args.method == 'stitch' and not use_meep_stitch):
        print(f'[meep] Method "{args.method}" not available with this source -> full domain.')

    _warn_unsupported(args)
    cfg = _build_cfg(args, layers)
    print('===================== MEEP configuration =====================')
    print(f'  Geometry  : {({"lens": "contact lens", "cad": "CAD"}).get(args.geometry, "planar")}  '
          f'({args.dim}D)   Pol: {cfg["pol"]}')
    if args.geometry == 'lens':
        print(f'  Scenario  : {args.scenario}')
    print(f'  Domain    : x={cfg["lx"]:g} y={cfg["ly"]:g}'
          + (f' z={cfg["lz"]:g}' if args.dim == 3 else '') + ' um   '
          f'resolution {cfg["resolution"]:g} px/um  (f0={cfg["f0"]:.3f})')
    bead_desc = 'none' if not cfg['bead'] else f'd={2*cfg["bead"]["r"]:g}um'
    print(f'  Waveguide : {cfg["t_wg"]:g} um  n={cfg["wg_n"]:.3f}   Bead: {bead_desc}')
    if use_meep_stitch:
        print('  Method    : window stitch (Meep, ' + ('vector equivalence principle' if use_vector
              else 'mode cascade/field') + ', 1 window in RAM)')
    print('==============================================================')

    out_dir = os.path.join(_REPO_ROOT, 'results')
    # x-Aufteilung wie native (Full und Vektor-Stitch; Mode-/Feld-Stitch behalten die
    # alte Zelle [0, L] mit PML innen)
    if args.geometry != 'cad' and (not use_meep_stitch or use_vector):
        cfg = _native_x_layout(cfg)

    # --- Meep-Fenster-Stitch (planar 2D): Speicher unabhaengig von der Laenge ---
    if use_meep_stitch:
        if args.dry_run:
            print('[dry-run] Meep stitch configuration resolved - no run.')
            return 0
        import meep_stitch as ms
        cfg['label'] += '_stitch'                    # nicht den Full-Lauf ueberschreiben
        if use_vector:
            st = ms.run_stitch_vector(cfg, args.window_um, args.slide_um)
        elif getattr(args, 'meep_handoff', 'mode') == 'field':
            st = ms.run_stitch_fullfield(cfg, args.window_um, args.slide_um)
        else:
            st = ms.run_stitch(cfg, args.window_um, args.slide_um,
                               modes=getattr(args, 'meep_modes', 1))
        if not args.no_save:
            Efull = st['Efull']
            nfr = max(2, args.snapshots)
            wct = 2*np.pi*cfg['f0']
            times = [k/(nfr*cfg['f0']) for k in range(nfr)]     # 1 Periode
            # Meep-DFT liefert A (Feld = Re(A e^{-i w t})) -> physikalische Zeitfolge
            # (mit e^{+i w t} liefe die Welle im Analyzer rueckwaerts)
            frames = [np.real(Efull*np.exp(-1j*wct*t)).astype(np.float32)
                      for t in times]
            res_d = dict(frames=frames, times=times, transmission=st['transmission'],
                         transmission_def='poynting' if use_vector else 'mode',
                         cw_frames=np.stack(frames), cw_times=times)
            if args.dim == 3:
                mnpz.save_3d(cfg, res_d, out_dir)
            else:
                mnpz.save_2d(cfg, res_d, out_dir)
        return 0

    # --- Full-Domain: adaptiver Zellzahl-/RAM-Schutz (Speicher ~ Zellzahl) ---
    res = cfg['resolution']
    ncells = cfg['lx']*res * cfg['ly']*res * (cfg['lz']*res if args.dim == 3 else 1)
    # Speicher pro Zelle, empirisch kalibriert: Meep-3D SERIELL ~394 B/Zelle
    # (gemessen: 4.27M Zellen -> 1.68 GB Peak-RSS, --no-save). Mit Full-Field-NPZ
    # + Reserve ~500. ACHTUNG: viele MPI-Raenge vervielfachen den Speicher (Overhead
    # je Rang + Einsammeln des vollen Volumens auf Rang 0) -> auf kleinem RAM 3D
    # SERIELL oder mit WENIGEN Raengen rechnen. (~18M Zellen mit -np 22 -> OOM.)
    bpc = 500 if args.dim == 3 else 120
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except Exception:
        avail = 8e9
    max_cells = (0.5 if args.dim == 3 else 0.6)*avail/bpc
    print(f'  Grid      : ~{ncells/1e6:.1f}M cells  (~{ncells*bpc/1e9:.1f} GB; '
          f'adaptive limit ~{max_cells/1e6:.0f}M at {avail/1e9:.1f} GB free)')
    if ncells > max_cells and not args.dry_run and not getattr(args, 'allow_large', False):
        raise SystemExit(
            f'[ABORT] ~{ncells/1e6:.0f}M cells (~{ncells*bpc/1e9:.1f} GB) '
            f'> adaptive limit ~{max_cells/1e6:.0f}M ({avail/1e9:.1f} GB free).\n'
            f'  Meep computes the FULL DOMAIN (no stitch) - options:\n'
            f'   - 3D with little RAM: SERIAL/few ranks (--meep-np 1..4); many '
            f'ranks multiply the memory (OOM risk)\n'
            f'   - coarser --resolution-nm / shorter --length-um '
            f'(3D also smaller --lz-um/--tear/--air)\n'
            f'   - long/planar domain: Meep window stitch -> --method stitch\n'
            f'   - large lens: native solver -> --engine native --method stitch\n'
            f'   - force deliberately: --allow-large  (risk: OOM)')

    if args.dry_run:
        print('[dry-run] Meep configuration resolved - no run started.')
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
