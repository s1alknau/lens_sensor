"""Vereinheitlichter FDTD-Einstieg fuer BEIDE Geometrien in 2D UND 3D.

Ein Skript unterscheidet die Faelle:

    Geometrie:  kontaktlinse (gekruemmte Linse, geflattet)  |  planar (Waveguide+Bead)
    Dimension:  2d  |  3d

Es werden die BESTEHENDEN, getesteten Solver aus planar_beads/planar_3d
wiederverwendet - kein eigener Kern:
    2d -> fdtd2d_core.run_beads(method=full|sliding|stitch)
    3d -> fdtd3d_core.run_3d (full)  /  run_3d_stitched (stitch)

Die Kontaktlinse ist im geflatteten Modell nur ein anderer Schichtstapel
(dicke PMMA-Linse statt duennem WG) mit Trockenauge-Szenario. Die GROSSE Linse
(volle ~14 mm) passt nicht in den Speicher und laeuft daher NUR im Stitch-Modus
- darauf weist das Skript beim Start hin und schaltet automatisch um.

Dieses Skript ist der EINZIGE Einstiegspunkt und deckt den vollen Funktions-
umfang der frueheren Einzel-CLIs ab (planar_beads/demo_beads, planar_3d/demo_beads_3d).

Beispiele:
    python run_simulation.py --geometry planar --dim 3 --method full
    python run_simulation.py --geometry planar --dim 2 --method stitch
    python run_simulation.py --geometry lens --dim 2 --scenario Gesund
    python run_simulation.py --geometry lens --dim 3 --scenario DED --length-um 60
    python run_simulation.py --geometry planar --dim 3 --calibrate 30   # nur Laufzeit-Schaetzung
    python run_simulation.py --geometry lens --dim 2 --dry-run          # nur Config zeigen
"""
import argparse
import os
import sys
import subprocess

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ("planar_beads", "planar_3d"):
    _p = os.path.join(_REPO_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common.physics import n_at, tear_index_at, cornea_default, parse_n, split_n   # noqa: E402
from common.vcsel import VCSEL_DEFAULTS                                    # noqa: E402

# WSL-Ziel fuer den Meep-Zweig (Meep laeuft nur unter Linux).
WSL_DISTRO = "Ubuntu-24.04"
WSL_ENV = "lens_sensor"
WSL_CONDA = "/opt/conda/bin/conda"   # conda ist im Non-Login-Shell nicht im PATH

# ---------------------------------------------------------------------------
# Geometrie-Konstanten der Kontaktlinse (aus kontaktlinse/config.py uebernommen).
# Im geflatteten Modell ist die Linse ein gerader Schicht-Wellenleiter der Dicke
# T_LENS ueber die Bogenlaenge D_LENS. Krummungseffekt (~T_LENS/R_BEND ~ 3%) wird
# - wie im bestehenden 2D-Linsenmodell - vernachlaessigt.
T_LENS_UM = 250.0          # Linsendicke (= "Waveguide"-Dicke der Linse)
D_LENS_UM = 14000.0        # volle Linsen-Bogenlaenge (14 mm)
N_LENS = 1.491             # PMMA

# Trockenauge-Szenarien: (lipid_um, aqueous_um, mucin_um, n_aqueous)
# 1:1 aus kontaktlinse/full_domain_fdtd.py (dort in Metern).
LENS_SCENARIOS = {
    'Gesund':       (0.030, 3.5, 0.5, 1.336),
    'DED':          (0.010, 1.5, 0.3, 1.336),
    'Frisch':       (0.050, 6.0, 1.0, 1.336),
    'Hyperosmolar': (0.020, 2.0, 0.4, 1.340),
    'MGD':          (0.005, 3.5, 0.5, 1.336),
    'Mucin-Mangel': (0.030, 3.5, 0.05, 1.336),
    'Mucin-reich':  (0.030, 3.5, 2.0, 1.336),
    'Lipid-reich':  (0.100, 3.5, 0.5, 1.336),
}

# Ab dieser Propagationslaenge gilt die Linse als "gross" -> nur Stitch rechenbar.
# Kuerzere Ausschnitte (ein paar zehn/hundert um) laufen auch full/sliding.
LENS_STITCH_ONLY_UM = 2000.0

# Geometrie-abhaengige Default-Laenge (um), wenn --length-um nicht gesetzt ist.
DEFAULT_LENGTH_UM = {
    ('lens', 2): D_LENS_UM,   # volle Linse -> wird auf Stitch gezwungen
    ('lens', 3): 60.0,        # 3D nur Ausschnitt (250um-Querschnitt ist teuer)
    ('planar', 2): 1000.0,    # langer planarer Waveguide
    ('planar', 3): 40.0,      # 3D-Ausschnitt um den Bead
}

_GEOM_ALIASES = {
    'lens': 'lens', 'linse': 'lens', 'kontaktlinse': 'lens', 'contact': 'lens',
    'planar': 'planar', 'waveguide': 'planar', 'wg': 'planar', 'bead': 'planar',
    'cad': 'cad', 'step': 'cad', 'stl': 'cad', 'inventor': 'cad', 'solidcraft': 'cad',
}


def _make_cad(args, absorber_um=None):
    """CAD-Szene + Rechenbox fuer --geometry cad. Materialzuordnung aus
    '<datei>.lens.json' (GUI) und/oder --cad-materials ('Kern=pmma,Film=aqueous,
    Prisma=n=1.52'); --cad-materials wird in die Konfigdatei uebernommen."""
    from common.cad_geometry import CadScene, CadRun
    if not args.cad:
        raise SystemExit('--geometry cad requires --cad <file> (.sc/.step/.stl/.ipt/...).')
    scene = CadScene.from_config(args.cad, scale_um=args.cad_scale, axis=args.cad_axis,
                                 core=args.cad_core, tear=args.cad_tear,
                                 background=args.cad_background)
    if args.cad_materials:
        for item in args.cad_materials.split(','):
            if '=' not in item:
                continue
            body, mat = item.split('=', 1)
            body = body.strip()
            if body not in scene.bodies:
                raise SystemExit(f'--cad-materials: body "{body}" does not exist. '
                                 f'Available: {", ".join(scene.bodies)}')
            scene.materials[body] = mat.strip()
        print(f'  CAD material assignment saved: {scene.save_config()}')
    if args.cad_open:
        try:
            scene.open_by_spec(args.cad_open, args.lambda_nm)
        except ValueError as e:
            raise SystemExit(str(e))
        print(f'  CAD open faces saved: {scene.save_config()}')
    missing = scene.unassigned()
    if missing:
        raise SystemExit('CAD: bodies without material: ' + ', '.join(missing) +
                         '\n  -> assign them in the GUI or pass --cad-materials "'
                         + ','.join(f'{b}=pmma' for b in missing) + '".')
    dx_um = args.resolution_nm/1000.0
    bnd = (args.boundary or 'cpml').lower()
    # 3D-Sponge: feste Dicke 1.2 um (mind. 8 Zellen), wie planar_3d._sponge_cells
    _ncell = max(8, -int(-1.2//dx_um)) if (args.dim == 3 and bnd == 'sponge') else 10
    absorber = _ncell*dx_um if bnd != 'mur1' else 0.0
    if absorber_um is not None:                     # z.B. Meep-PML (liegt in der Zelle)
        absorber = float(absorber_um)
    return CadRun(scene, args.dim, dx_um, args.lambda_nm, plane_z_um=args.cad_plane_z,
                  absorber_um=absorber)


def _prepare_meep_cad(args):
    """CAD fuer Meep: Konvertierung/Voxelisierung laeuft unter Windows (SolidCraft,
    Inventor), Meep laeuft in WSL. Deshalb hier das eps-Gitter der CAD-Box (gleiches dx
    wie Meep), Regionen, Quelle und Detektoren vorberechnen und als NPZ ablegen; der
    Meep-Treiber liest es (--meep-cad-eps). Meep-PML (~1 lambda) liegt in der Zelle."""
    import json as _json
    resolve_args(args)
    lam_um = args.lambda_nm/1000.0
    cad = _make_cad(args, absorber_um=lam_um)
    dx = cad.dx_um
    nx = max(2, int(round(cad.lx_um/dx))); ny = max(2, int(round(cad.ly_um/dx)))
    if args.dim == 3:
        nz = max(2, int(round(cad.lz_um/dx)))
        eps, reg = cad.eps3d(0.0, nx, ny, nz)
        reg_xy = reg[:, :, nz//2]
    else:
        eps, reg = cad.eps2d(0.0, nx, ny)
        reg_xy = reg
    x_src = lam_um + 0.5                             # wie Meep: knapp hinter der PML
    src = cad.source(x_src)
    lo, hi = cad.scene.bbox(cad.core)
    L = hi[0] - lo[0]
    path = os.path.join(_REPO_ROOT, 'results',
                        f'_meep_cad_{os.path.splitext(os.path.basename(args.cad))[0]}.npz')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = cad.meta()
    if np.iscomplexobj(eps) and np.any(np.abs(np.imag(eps)) > 0):
        # Meep liest hier ein REELLES eps-Gitter: absorbierende CAD-Materialien
        # (z.B. 'aqueous+0.001i') wuerden sonst stillschweigend verlustfrei gerechnet.
        print('[engine=meep] WARNING: CAD materials with absorption (Im(n) > 0) - Meep '
              'computes the CAD grid LOSSLESS (Re(eps) only). For absorption '
              'use --engine native.')
        eps = np.real(eps)
    np.savez_compressed(path, eps=np.asarray(eps, np.float32), region=np.asarray(reg, np.int8),
                        dx_um=dx, lx_um=cad.lx_um, ly_um=cad.ly_um, lz_um=cad.lz_um,
                        y0_um=cad.y0_um, z0_um=cad.z0_um, x_src_um=x_src,
                        src_y_um=src['y_um'], src_z_um=src['z_um'], src_tilt_deg=src['tilt_deg'],
                        src_air_um=float(np.linalg.norm(np.asarray(src['facet_center'], float)
                                                        - np.array([x_src, src['y_um'], src['z_um']])
                                                        [:len(src['facet_center'])])),
                        facet_x_um=float(src['facet_center'][0]),
                        det_in_x_um=lo[0] + 0.1*L, det_out_x_um=lo[0] + 0.9*L,
                        meta_json=_json.dumps(meta, default=float))
    print(f'[engine=meep] CAD prepared: {nx}x{ny}' + (f'x{nz}' if args.dim == 3 else '')
          + f' cells, core "{cad.core}", source y={src["y_um"]:.2f} um, '
          f'tilt {src["tilt_deg"]:.2f} deg -> {path}')
    return path


LENS_LAYER_NAMES = ['PMMA lens', 'Lipid', 'Aqueous', 'Mucin', 'Cornea']
GENERIC_LAYER_NAMES = ['Waveguide', 'Layer 1', 'Layer 2', 'Layer 3', 'Substrate']


def _layer_names(args):
    """Namen von Kern, den drei Schichten darunter (--t-lipid/--t-aqueous/--t-mucin)
    und dem Substrat (--n-cornea). Traenenfilm-Namen nur fuer die Kontaktlinse; sonst
    neutral, frei vergebbar mit --layer-names (fehlende Eintraege -> Standard)."""
    base = list(LENS_LAYER_NAMES if args.geometry == 'lens' else GENERIC_LAYER_NAMES)
    given = [t.strip() for t in str(getattr(args, 'layer_names', '') or '').split(',')]
    for i, t in enumerate(given[:5]):
        if t:
            base[i] = t
    return base


def _resolve_layers(args):
    """Liefert den Schichtstapel fuer die gewaehlte Geometrie. CLI-Overrides
    (--t-*/--n-*) haben Vorrang vor dem Geometrie-/Szenario-Default. None = der
    jeweilige Solver setzt seinen eigenen Default."""
    if args.geometry == 'lens':
        lip, aq, mu, n_aq = LENS_SCENARIOS[args.scenario]
        # Szenario-n_aq ist ein Wert im Sichtbaren -> auf lambda umrechnen (Review F5)
        base = dict(t_wg_um=T_LENS_UM, t_lip_um=lip, t_aq_um=aq, t_mu_um=mu,
                    n_aq=tear_index_at(n_aq, args.lambda_nm))
    else:
        # Einheitliche PHYSIOLOGISCHE Tränenfilm-Schichten (2D=3D): Lipid 30nm +
        # Aqueous 3.5um + Mucin 0.5um -> Tränenfilm gesamt ~4um. Einzeln via --t-*.
        base = dict(t_wg_um=args.wg_thickness_um, t_lip_um=0.030,
                    t_aq_um=3.5, t_mu_um=0.5, n_aq=None)
    # WG-Brechzahl: --wg-n ueberschreibt; sonst aus --wg-material + Wellenlaenge.
    base['wg_n'] = (args.wg_n if args.wg_n is not None
                    else n_at(getattr(args, 'wg_material', 'pmma'), args.lambda_nm))
    # CLI-Overrides (nur wenn explizit gesetzt)
    if args.t_aqueous is not None: base['t_aq_um'] = args.t_aqueous
    if args.t_mucin is not None:   base['t_mu_um'] = args.t_mucin
    if args.t_lipid is not None:   base['t_lip_um'] = args.t_lipid
    base['n_aq'] = args.n_aqueous if args.n_aqueous is not None else base['n_aq']
    base['n_mu'] = args.n_mucin
    base['n_co'] = args.n_cornea
    base['n_lip'] = args.n_lipid
    base['names'] = _layer_names(args)
    # Freely configurable stack (common/layer_stack.py). 'stack' is ALWAYS present:
    # with --layers the user's layers (stack_custom=True, solvers take the general path),
    # otherwise the three legacy layers with their resolved values (stack_custom=False,
    # solvers keep the legacy path -> identical results; stack only for saving/analyzer).
    from common.layer_stack import parse_layer_stack, legacy_stack
    spec = getattr(args, 'layers', None)
    if spec is not None:
        if args.geometry != 'planar':
            raise SystemExit('--layers is only available for --geometry planar')
        _given = [o for o in ('t_lipid', 't_aqueous', 't_mucin', 'n_lipid', 'n_aqueous', 'n_mucin',
                              'kappa_lipid', 'kappa_aqueous', 'kappa_mucin')
                  if getattr(args, o, None) is not None]
        if _given:
            print(f'  [WARN] --layers given: ignoring {", ".join("--" + o.replace("_", "-") for o in _given)}')
        try:
            base['stack'] = parse_layer_stack(spec, args.lambda_nm)
        except ValueError as e:
            raise SystemExit(f'[ERROR] {e}')
        base['stack_custom'] = True
    else:
        lam = args.lambda_nm
        _n = lambda v, mat: v if v is not None else n_at(mat, lam)
        base['stack'] = legacy_stack(
            base['t_lip_um'] or 0.0, _n(base['n_lip'], 'lipid'),
            base['t_aq_um'] if base['t_aq_um'] is not None else 3.5, _n(base['n_aq'], 'aqueous'),
            base['t_mu_um'] if base['t_mu_um'] is not None else 0.5, _n(base['n_mu'], 'mucin'),
            names=base['names'][1:4])
        base['stack_custom'] = False
    return base


def _enforce_lens_stitch(args):
    """Grosse Kontaktlinse: Hinweis + automatische Umschaltung auf Stitch."""
    if args.geometry == 'lens' and args.length_um > LENS_STITCH_ONLY_UM \
            and args.method != 'stitch':
        print('')
        print('  ############################################################')
        print('  #  NOTE: LARGE CONTACT LENS                               #')
        print(f'  #  Length {args.length_um:.0f} um (> {LENS_STITCH_ONLY_UM:.0f} um) does NOT fit')
        print('  #  into memory as a whole. The full lens works          #')
        print('  #  ONLY in STITCHED mode (domain decomposition).         #')
        print(f'  #  -> switching method "{args.method}" to "stitch" automatically.')
        print('  ############################################################')
        print('')
        args.method = 'stitch'


def _place_bead(args):
    """Bead nur im planaren Fall und nur wenn nicht --no-bead und d>0."""
    return (args.geometry == 'planar') and (not args.no_bead) and (args.bead_diameter > 0)


def _cornea_full_tear(layers, absorber_um):
    """Gebietstiefe unter dem Kern fuer --cornea-full: Lipid + Aqueous + Mucin + Cornea bis
    zur Rueckseite (~545 um) + 5 um Kammerwasser + Absorber."""
    from common.physics import cornea_full_depth_um
    if layers.get('stack_custom'):                   # frei konfigurierbarer Stapel
        from common.layer_stack import stack_total_um
        return stack_total_um(layers['stack']) + cornea_full_depth_um() + absorber_um
    t = (layers['t_lip_um'] or 0.0) + (layers['t_aq_um'] if layers['t_aq_um'] is not None else 3.5) \
        + (layers['t_mu_um'] if layers['t_mu_um'] is not None else 0.5)
    return t + cornea_full_depth_um() + absorber_um


def _run_2d(args, layers):
    import fdtd2d_core as d2
    d2.KAPPA = dict(getattr(args, 'kappa', {}) or {})   # Absorption je Schicht
    d2.SRC_CORE_ONLY = bool(getattr(args, 'source_core_only', False))   # Quelle nur im Kern
    d2.LASER_POWER_MW = float(args.laser_power)          # VCSEL-Leistung (Analyzer-Skalierung)
    result = d2.run_beads(
        wg_mat=args.wg_material, bead_mat=args.bead_material,
        bead_d_um=(args.bead_diameter if _place_bead(args) else 0.0),
        dx_nm=args.resolution_nm, save_frames=not args.no_save,
        n_snapshots=args.snapshots, wg_thickness_um=layers['t_wg_um'],
        lambda_nm=args.lambda_nm, vcsel_waist=args.vcsel_waist,
        vcsel_tilt=args.vcsel_tilt, vcsel_offset=args.vcsel_offset,
        source_type=args.source_type, method=args.method,
        window_w_um=args.window_um, slide_um=args.slide_um,
        place_bead=_place_bead(args), length_um=args.length_um,
        t_aqueous_um=layers['t_aq_um'], t_mucin_um=layers['t_mu_um'],
        t_lipid_um=(layers['t_lip_um'] or 0.0), input_gap_um=args.input_gap,
        bead_x_um=args.bead_x, bead_y_um=args.bead_y_um, wg_n=layers['wg_n'], bead_n=args.bead_n,
        n_aqueous=layers['n_aq'], n_mucin=layers['n_mu'],
        n_cornea=layers['n_co'], n_lipid=layers['n_lip'],
        polarization=args.polarization,
        curved=(args.curved and args.geometry == 'lens'),
        boundary=(args.boundary or 'cpml'),          # 2D-Default: cpml
        cad=getattr(args, '_cad', None),
        pulse_cycles=args.pulse_cycles, pulse_tmax_fs=args.pulse_tmax_fs,
        pulse_nfreq=args.pulse_nfreq, pulse_dispersion=not args.pulse_no_dispersion,
        n_aq_tear_offset=((layers['n_aq'] - n_at('aqueous', args.lambda_nm))
                          if (args.geometry == 'lens' and args.n_aqueous is None
                              and layers['n_aq'] is not None) else None),
        pulse_mode=args.pulse_stitch,
        tear_buffer_um=(_cornea_full_tear(layers, 0.2) if args.cornea_full else None),
        # eigener Stapel (--layers) -> allgemeiner Pfad; sonst Legacy-Pfad (unveraendert)
        stack=(layers['stack'] if layers.get('stack_custom') else None))
    result['layer_names'] = layers['names']
    if args.bead_y_um is not None:
        result['bead_y_um'] = args.bead_y_um          # Bead-Zentrum y (Analyzer-Marker)
    result['n_lip'] = (layers['n_lip'] if layers['n_lip'] is not None
                       else n_at('lipid', args.lambda_nm))
    if not args.no_save:
        d2.save_results(result, out_dir=os.path.join(_REPO_ROOT, 'results'))
    return result


def _run_3d(args, layers):
    import fdtd3d_core as f3d
    f3d.KAPPA = dict(getattr(args, 'kappa', {}) or {})  # Absorption je Schicht
    f3d.SRC_CORE_ONLY = bool(getattr(args, 'source_core_only', False))  # Quelle nur im Kern
    f3d.LASER_POWER_MW = float(args.laser_power)         # VCSEL-Leistung (Analyzer-Skalierung)
    bead = None
    if _place_bead(args):
        bead = dict(x_um=(args.bead_x if args.bead_x is not None else args.length_um/2.0),
                    d_um=args.bead_diameter,
                    n=(args.bead_n if args.bead_n is not None
                       else f3d.n_at(args.bead_material, args.lambda_nm)))
        if args.bead_y_um is not None:              # Bead-Zentrum in y (sonst aufliegend, -d/2)
            bead['y_um'] = args.bead_y_um
    # Layer-Defaults fuer 3D (run_3d verlangt konkrete Werte)
    t_aq = layers['t_aq_um'] if layers['t_aq_um'] is not None else 3.5
    t_mu = layers['t_mu_um'] if layers['t_mu_um'] is not None else 0.5
    n_aq = layers['n_aq'] if layers['n_aq'] is not None else f3d.n_at('aqueous', args.lambda_nm)
    n_mu = layers['n_mu'] if layers['n_mu'] is not None else f3d.n_at('mucin', args.lambda_nm)
    n_co = layers['n_co'] if layers['n_co'] is not None else cornea_default(args.lambda_nm)
    n_lip = layers['n_lip'] if layers['n_lip'] is not None else f3d.n_at('lipid', args.lambda_nm)
    # Dateiname <geometrie>3d_<label>_<methode>: Bead nur, wenn einer platziert ist; die
    # Methode im Namen verhindert, dass full- und stitch-Laeufe sich ueberschreiben.
    if args.geometry == 'lens':
        label = str(args.scenario)
    elif bead is not None:
        label = f'Bead-{args.bead_material}_d{args.bead_diameter:g}'
    else:
        label = 'ohneBead'
    _cad = getattr(args, '_cad', None)
    if _cad is not None:
        label = os.path.splitext(os.path.basename(args.cad))[0]   # Datei: cad3d_<name>
    label += f'_{args.method}'
    # Gemeinsame Argumente fuer run_3d UND run_3d_stitched
    common = dict(
        label=label, wg_n=layers['wg_n'], t_wg_um=layers['t_wg_um'],
        t_lip_um=(layers['t_lip_um'] or 0.0), t_aq_um=t_aq, t_mu_um=t_mu,
        n_lip=n_lip, n_aq=n_aq, n_mu=n_mu, n_co=n_co,
        lam_nm=args.lambda_nm, lx_um=args.length_um, air_um=args.air,
        tear_um=(args.tear if (args.tear is not None or not args.cornea_full) else
                 _cornea_full_tear(layers, 1.2 if (args.boundary or 'cpml') == 'sponge'
                                   else 0.2)),
        lz_um=args.lz_um, dx_nm=args.resolution_nm,
        vcsel_waist_um=args.vcsel_waist,
        vcsel_waist_z_um=(args.vcsel_waist_z if args.vcsel_waist_z is not None else args.vcsel_waist),
        source_type=args.source_type, n_snapshots=args.snapshots, bead=bead,
        vol_dtype=args.vol_dtype, wg_width_um=args.wg_width,
        n_clad_side=args.wg_clad_n, vcsel_tilt_deg=args.vcsel_tilt,
        vcsel_offset_y_um=args.vcsel_offset, vcsel_offset_z_um=args.vcsel_offset_z,
        input_gap_um=args.input_gap, polarization=args.polarization,
        # eigener Stapel (--layers) -> allgemeiner Pfad; sonst Legacy-Pfad (unveraendert)
        stack=(layers['stack'] if layers.get('stack_custom') else None))
    if args.method == 'stitch' and args.source_type == 'pulse' and args.pulse_stitch == 'td':
        result = f3d.run_3d_stitched_td(window_w_um=args.window_um, slide_um=args.slide_um,
                                        pulse_cycles=args.pulse_cycles,
                                        pulse_tmax_fs=args.pulse_tmax_fs, cad=_cad, **common)
    elif args.method == 'stitch' and args.source_type == 'pulse':
        # Fourier: nur NICHT explizit gesetzte Brechzahlen folgen n(lambda) je Frequenz
        disp = {}
        if args.wg_n is None:
            disp['wg_n'] = args.wg_material
        if args.n_aqueous is None:
            # Linse: Szenario-Traenen = n_aqueous(lambda) + fester Szenario-Offset
            disp['n_aq'] = (('aqueous', n_aq - f3d.n_at('aqueous', args.lambda_nm))
                            if args.geometry == 'lens' else 'aqueous')
        if args.n_mucin is None:
            disp['n_mu'] = 'mucin'
        if args.n_cornea is None:
            disp['n_co'] = 'cornea_layered'
        if args.n_lipid is None:
            disp['n_lip'] = 'lipid'
        if args.bead_n is None:
            disp['bead'] = args.bead_material
        cm = dict(common); cm.pop('source_type', None)
        result = f3d.run_3d_stitched_pulse(window_w_um=args.window_um, slide_um=args.slide_um,
                                           pulse_cycles=args.pulse_cycles,
                                           pulse_tmax_fs=args.pulse_tmax_fs,
                                           pulse_nfreq=args.pulse_nfreq,
                                           pulse_dispersion=not args.pulse_no_dispersion,
                                           disp_map=disp, cad=_cad,
                                           boundary=(args.boundary or 'cpml'), **cm)
    elif args.method == 'stitch':
        result = f3d.run_3d_stitched(window_w_um=args.window_um,
                                     slide_um=args.slide_um,
                                     boundary=(args.boundary or 'cpml'), cad=_cad, **common)
    else:
        # nur run_3d (full) kennt diese Zusatz-Optionen
        pec_faces = tuple(f.strip() for f in args.pec_faces.split(',') if f.strip())
        result = f3d.run_3d(
            steps_factor=args.steps_factor, calibrate_steps=args.calibrate,
            save_vector=args.save_vector, check_resources=args.check_resources,
            allow_large=args.allow_large, boundary=(args.boundary or 'cpml'),
            pec_faces=pec_faces, end_facet_um=args.end_facet, cad=_cad,
            pulse_cycles=args.pulse_cycles, **common)
        if args.calibrate:
            return result   # Kalibrierlauf: kein Speichern
    if not args.no_save:
        f3d.save_results_3d(result, out_dir=os.path.join(_REPO_ROOT, 'results'),
                            prefix=f'{args.geometry}3d', material_names=layers['names'])
    return result


def _to_wsl_path(win_path):
    """C:\\Users\\...  ->  /mnt/c/Users/... (Repo-Zugriff aus WSL)."""
    p = win_path.replace('\\', '/')
    if len(p) > 1 and p[1] == ':':
        p = '/mnt/' + p[0].lower() + p[2:]
    return p


def _strip_flags(raw, drop):
    """Entfernt die genannten Optionen (+ ihren Wert) aus der Argumentliste.
    'drop' ist ein Set von Optionsnamen (z.B. {'--engine','-e','--meep-np'})."""
    out = []
    skip = False
    for tok in raw:
        if skip:
            skip = False
            continue
        key = tok.split('=', 1)[0]
        if key in drop:
            skip = ('=' not in tok)   # eigener Token = naechster ist der Wert
            continue
        out.append(tok)
    return out


def _mpi_prefix(np_ranks):
    """Baut das mpirun-Praefix. Meep parallelisiert NUR ueber MPI (kein GPU).
    N<=1 -> seriell (kein Praefix); N==0 -> alle Kerne ($(nproc)); sonst -np N."""
    if np_ranks is None or np_ranks == 1:
        return ''
    n = '$(nproc)' if np_ranks == 0 else str(int(np_ranks))
    return 'mpirun -np %s ' % n


def _dispatch_meep(raw, np_ranks=1):
    """Startet den Meep-Treiber. Unter Windows via WSL (Meep-Only-Linux),
    unter Linux direkt. Reicht die urspruenglichen Argumente (ohne --engine/--meep-np)
    durch. np_ranks>1 (oder 0=alle Kerne) startet via 'mpirun -np N'."""
    args_fwd = _strip_flags(raw, {'--engine', '-e', '--meep-np'})
    mpi = _mpi_prefix(np_ranks)
    if os.name == 'nt':
        wsl_repo = _to_wsl_path(_REPO_ROOT)
        inner = ('cd "%s" && %s run -n %s %spython meep/run_meep.py %s'
                 % (wsl_repo, WSL_CONDA, WSL_ENV, mpi,
                    ' '.join(_q(a) for a in args_fwd)))
        cmd = ['wsl', '-d', WSL_DISTRO, '-u', 'root', 'bash', '-lc', inner]
        print(f'[engine=meep] Windows detected -> dispatching to WSL ({WSL_DISTRO})')
        if mpi:
            print(f'  MPI: {mpi.strip()} (Meep uses CPU cores only, no GPU)')
        print('  ' + ' '.join(cmd[:6]) + ' ...')
    else:
        base = [sys.executable, os.path.join(_REPO_ROOT, 'meep', 'run_meep.py')] + args_fwd
        if mpi:
            n = str(os.cpu_count() or 1) if np_ranks == 0 else str(int(np_ranks))
            cmd = ['mpirun', '-np', n] + base
        else:
            cmd = base
        print('[engine=meep] Linux -> running Meep directly' + (f' (mpirun -np {n})' if mpi else ''))
    try:
        return subprocess.call(cmd)
    except FileNotFoundError as e:
        raise SystemExit(f'[ERROR] Meep launch failed ({e}). '
                         f'Is WSL/distro "{WSL_DISTRO}" set up? See docs/meep_setup_wsl.md')


def _q(s):
    """Minimales Shell-Quoting fuer die Weitergabe an bash -lc."""
    return "'" + s.replace("'", "'\\''") + "'" if any(c in s for c in ' "\'$`\\') else s


def build_parser():
    ap = argparse.ArgumentParser(
        description='Unified FDTD entry point: contact lens | planar, in 2D | 3D.')
    # Argumente in benannte GRUPPEN -> dienen zugleich als Sektionen im GUI.
    g_case = ap.add_argument_group('Case')
    g_case.add_argument('--engine', '-e', choices=('native', 'meep'), default='native',
                        help='native = built-in NumPy/CuPy solver; meep = Meep toolbox '
                             '(runs only on Linux/WSL; on Windows it is dispatched to WSL '
                             'automatically)')
    g_case.add_argument('--geometry', '-g', default='planar',
                        help='kontaktlinse/lens OR planar/waveguide (default: planar)')
    g_case.add_argument('--dim', '-d', type=int, choices=(2, 3), default=3,
                        help='Dimension: 2 or 3 (default: 3)')
    g_case.add_argument('--method', '-m', choices=('full', 'sliding', 'stitch'),
                        default='full', help='full | sliding (2D only) | stitch')
    g_case.add_argument('--scenario', default='Gesund', choices=list(LENS_SCENARIOS),
                        help='Dry-eye scenario (--geometry lens only)')
    g_case.add_argument('--curved', action='store_true',
                        help='(lens only, 2D) TRUE curvature via conformal mapping '
                             'eps*(1+u/R)^2 instead of flattened. Applies to TE AND TM.')

    g_basic = ap.add_argument_group('Basic settings')
    g_basic.add_argument('--length-um', type=float, default=None,
                         help='Propagation length in um (default depends on geometry/dim)')
    g_basic.add_argument('--resolution-nm', type=float, default=None,
                         help='dx in nm (default: 2D=20, 3D=50)')
    g_basic.add_argument('--lambda-nm', type=float, default=850.0, help='Wavelength in nm')
    g_basic.add_argument('--polarization', choices=('s', 'p'), default='s',
                         help='s=TE (Ez) | p=TM (Ey)')
    g_basic.add_argument('--boundary', choices=('sponge', 'cpml', 'mur1'), default=None,
                         help='Absorbing OUTER boundary. Empty = default for the dimension. '
                              'Default cpml (2D and 3D; 10 cells, kappa 1 + CFS). 3D also: sponge (1.2 um). '
                              'mur1 = expert option (CLI only): first-order Mur with local '
                              'velocity (c0/n_eff), exact only for the fundamental mode at '
                              'normal incidence - use cpml for quantitative results. '
                              'stitch: inner window boundaries are always CPML.')
    g_basic.add_argument('--source-type', choices=('cw', 'pulse'), default='cw',
                         help='cw = continuous wave (steady state). pulse = Gaussian pulse; with '
                              '--method stitch (2D) as Fourier stitch: one CW stitch per frequency '
                              'with n(lambda), inverse transform -> true time trace + spectrum.')
    g_basic.add_argument('--pulse-cycles', type=float, default=30.0,
                         help='Pulse duration in optical periods (s = cycles/f0; 30 ~ 1.3 %% '
                              'bandwidth, 10 ~ 4 %%). Smaller = broader band = more frequencies.')
    g_basic.add_argument('--pulse-tmax-fs', type=float, default=None,
                         help='Fourier stitch: time window in fs (default: pulse + transit time). '
                              'Sets the frequency spacing (no aliasing up to t_max).')
    g_basic.add_argument('--pulse-nfreq', type=int, default=None,
                         help='Fourier stitch: fixed number of frequencies (default: auto).')
    g_basic.add_argument('--pulse-stitch', choices=('fourier', 'td'), default='fourier',
                         help='Pulse with --method stitch: fourier = one CW stitch per frequency + '
                              'inverse transform (with dispersion, also yields the spectrum); td = '
                              'directly in the time domain, windows coupled via recorded '
                              'Huygens planes (TF/SF), single run, no dispersion. Both in 2D and 3D.')
    g_basic.add_argument('--pulse-no-dispersion', action='store_true',
                         help='Fourier stitch: all refractive indices fixed at lambda0 (as in '
                              'the time domain) instead of n(lambda) per frequency.')
    g_basic.add_argument('--snapshots', type=int, default=8, help='Number of saved frames')

    g_mat = ap.add_argument_group('Material & layers')
    g_mat.add_argument('--cornea-full', action='store_true',
                       help='Compute the full cornea thickness (epithelium, Bowman, stroma, '
                            'endothelium, ~545 um) down into the aqueous humour instead of a few um '
                            'of epithelium. Expensive: about 28,000 extra rows at dx 20 nm; use a '
                            'coarser grid or stitch.')
    g_mat.add_argument('--layer-names', default=None,
                       help='Names: core, layer directly below the core (--t-lipid/--n-lipid), '
                            'second (--t-aqueous/--n-aqueous), third (--t-mucin/--n-mucin), '
                            'substrate (--n-cornea), comma-separated. Default for lens: '
                            'PMMA lens,Lipid,Aqueous,Mucin,Cornea; otherwise Waveguide,Layer 1,'
                            'Layer 2,Layer 3,Substrate. Shown in the analyzer and GUI.')
    g_mat.add_argument('--wg-material', choices=('pmma', 'polystyrol'), default='pmma',
                       help='Waveguide material (refractive index from dispersion; --wg-n overrides)')
    g_mat.add_argument('--wg-thickness-um', type=float, default=5.0,
                       help='Waveguide thickness (planar only; lens uses a fixed 250 um)')
    g_mat.add_argument('--wg-n', type=parse_n, default=None,
                       help='Custom waveguide refractive index (overrides --wg-material)')
    g_mat.add_argument('--kappa-wg', type=float, default=None,
                       help='Absorption of the waveguide core: extinction coefficient kappa, n = n_real + i*kappa (intensity decays as exp(-4 pi kappa z/lambda))')
    # Reihenfolge = Schichtstapel von oben nach unten (GUI zeigt die Felder so an):
    # Kern -> Schicht 1 (lipid) -> Schicht 2 (aqueous) -> Schicht 3 (mucin) -> Substrat
    g_mat.add_argument('--t-lipid', type=float, default=None,
                       help='Thickness of layer 1 directly below the core (lens: lipid), um')
    g_mat.add_argument('--n-lipid', type=parse_n, default=None,
                       help='Refractive index of layer 1 (lens: lipid)')
    g_mat.add_argument('--kappa-lipid', type=float, default=None,
                       help='Absorption of layer 1 (lens: lipid): extinction coefficient kappa, n = n_real + i*kappa (intensity decays as exp(-4 pi kappa z/lambda))')
    g_mat.add_argument('--t-aqueous', type=float, default=None,
                       help='Thickness of layer 2 (lens: aqueous), um')
    g_mat.add_argument('--n-aqueous', type=parse_n, default=None,
                       help='Refractive index of layer 2 (lens: aqueous)')
    g_mat.add_argument('--kappa-aqueous', type=float, default=None,
                       help='Absorption of layer 2 (lens: aqueous): extinction coefficient kappa, n = n_real + i*kappa (intensity decays as exp(-4 pi kappa z/lambda))')
    g_mat.add_argument('--t-mucin', type=float, default=None,
                       help='Thickness of layer 3 (lens: mucin), um')
    g_mat.add_argument('--n-mucin', type=parse_n, default=None,
                       help='Refractive index of layer 3 (lens: mucin)')
    g_mat.add_argument('--kappa-mucin', type=float, default=None,
                       help='Absorption of layer 3 (lens: mucin): extinction coefficient kappa, n = n_real + i*kappa (intensity decays as exp(-4 pi kappa z/lambda))')
    g_mat.add_argument('--layers', default=None,
                       help='Freely configurable layer stack below the core (planar only), from the '
                            'core downward: "Name:thickness_um:n,Name:thickness_um:n,...". n is a '
                            'number, a complex number (1.333+0.0001j = absorbing) or a material '
                            '(aqueous, mucin, lipid, pmma, polystyrol; evaluated at the wavelength). '
                            '"none" = core directly on the substrate. Replaces --t/--n-lipid, '
                            '-aqueous, -mucin. Example: "Lipid:0.03:lipid,Aqueous:3.5:aqueous,'
                            'Mucin:0.5:mucin"')
    g_mat.add_argument('--n-cornea', type=parse_n, default=None,
                       help='Refractive index of the substrate (lens: cornea), semi-infinite')
    g_mat.add_argument('--kappa-cornea', type=float, default=None,
                       help='Absorption of the substrate (lens: cornea): extinction coefficient kappa, n = n_real + i*kappa (intensity decays as exp(-4 pi kappa z/lambda))')

    g_cad = ap.add_argument_group('CAD geometry (--geometry cad)')
    g_cad.add_argument('--cad', default=None,
                       help='CAD file: .sc (SolidCraft), .step/.stp, .ipt/.iam (Inventor), .stl, '
                            '.3mf/.obj. Each body = one material; the source aligns itself '
                            'automatically to the entry face of the core. Methods: full, '
                            'stitch (2D+3D), sliding (2D). stitch only computes forward: strong '
                            'back-reflections (e.g. an end facet) are captured only by full - or '
                            'mark the face as open with --cad-open (GUI: mouse click).')
    g_cad.add_argument('--cad-scale', type=float, default=None,
                       help='um per CAD mm (default 1000 = true to scale; 1 = "1 mm in CAD '
                            '= 1 um"). Remembered in <file>.lens.json.')
    g_cad.add_argument('--cad-axis', choices=('xzy', 'xyz'), default=None,
                       help='Axis mapping CAD->simulation. xzy (default): CAD Z (up) = '
                            'layer stack y, CAD Y = depth z.')
    g_cad.add_argument('--cad-materials', default='',
                       help='Body=material, comma-separated, e.g. "Kern=pmma,Film=aqueous,'
                            'Prisma=n=1.52". Extends/overrides <file>.lens.json.')
    g_cad.add_argument('--cad-core', default=None,
                       help='Body name of the core (default: highest-index long body).')
    g_cad.add_argument('--cad-tear', default=None,
                       help='Body name of the tear film (sensor evaluation in the analyzer).')
    g_cad.add_argument('--cad-background', default=None,
                       help='Surrounding medium (default air), material or n=<number>.')
    g_cad.add_argument('--cad-open', default='',
                       help='Open faces (no reflection): "Kern:+x,Film:+x,Cornea:+x" extends '
                            'the last cross-section of these faces into the absorber (the '
                            'part "continues"). Axes x/y/z, sign = normal direction. '
                            'In the GUI by clicking the face. Remembered in .lens.json.')
    g_cad.add_argument('--cad-plane-z', type=float, default=None,
                       help='2D: position of the cut in depth z (um, simulation coordinates). '
                            'Default: center of the core.')

    g_bead = ap.add_argument_group('Bead (planar only)')
    g_bead.add_argument('--bead-material', choices=('polystyrol', 'pmma'),
                        default='polystyrol', help='Bead material (refractive index from dispersion)')
    g_bead.add_argument('--bead-diameter', type=float, default=0.5,
                        help='Bead diameter in um (0/--no-bead = no bead)')
    g_bead.add_argument('--bead-x', type=float, default=None,
                        help='Bead x position in um (default: center)')
    g_bead.add_argument('--bead-y-um', type=float, default=None,
                        help='Bead y position in um (default: below the WG at -d/2; '
                             '0..t_wg = inside the WG core -> strong scattering)')
    g_bead.add_argument('--bead-n', type=parse_n, default=None,
                        help='Custom bead refractive index (overrides --bead-material)')
    g_bead.add_argument('--kappa-bead', type=float, default=None,
                        help='Absorption of the bead: extinction coefficient kappa, n = n_real + i*kappa (intensity decays as exp(-4 pi kappa z/lambda))')
    g_bead.add_argument('--no-bead', action='store_true',
                        help='Planar run WITHOUT bead (reference)')
    g_bead.add_argument('--no-reference', action='store_true',
                        help='Do NOT also compute a reference run without bead. By default '
                             'BOTH runs are computed (with bead + reference without, '
                             'file _ref) for the bead difference. Check/set flag = '
                             'bead run only.')

    g_src = ap.add_argument_group(
        'Source (VCSEL)',
        'The VCSEL sits in air IN FRONT OF the coupling facet and emits a Gaussian TEM00 beam '
        '(waist w0 at the emitter, wavefront curvature, forward-only Huygens source). Air path, '
        'Fresnel reflection and mode coupling are computed rigorously by the FDTD. '
        'Defaults: typical single-mode 850 nm VCSEL (w0 = 3 um -> ~10 deg full divergence, 20 mW).')
    g_src.add_argument('--vcsel-waist', type=float, default=VCSEL_DEFAULTS['waist_um'],
                       help='Beam waist w0 (1/e^2 intensity radius) in y, in um. Full far-field '
                            'divergence = 2 atan(lambda/(pi w0)) (3 um -> 10.3 deg at 850 nm)')
    g_src.add_argument('--vcsel-waist-z', type=float, default=None,
                       help='Waist in z in um (3D only; default = --vcsel-waist, round beam)')
    g_src.add_argument('--vcsel-tilt', type=float, default=VCSEL_DEFAULTS['tilt_deg'],
                       help='Beam tilt in degrees')
    g_src.add_argument('--vcsel-distance', '--input-gap', dest='input_gap', type=float,
                       default=VCSEL_DEFAULTS['distance_um'],
                       help='Distance VCSEL waist (emitter) -> coupling facet in um. '
                            'Default 0 = waist with plane wavefront exactly on the facet. '
                            '(--input-gap is the old name.)')
    g_src.add_argument('--laser-power', type=float, default=VCSEL_DEFAULTS['power_mw'],
                       metavar='MW', help='Optical output power of the VCSEL in mW (default 20). '
                                          'Fields are scaled to it in the analyzer; the log '
                                          'shows the power coupled into the waveguide.')
    g_src.add_argument('--source-core-only', action='store_true',
                       help='Inject the source ONLY inside the waveguide core (soft 0.2 um edge): '
                            'no direct light from the Gaussian tails into air/tear film '
                            '(clean evanescent field). Layer stack only, not CAD.')
    g_src.add_argument('--vcsel-offset', type=float, default=0.0,
                       help='Spot offset in y (um); in 3D = offset y')
    g_src.add_argument('--vcsel-offset-z', type=float, default=0.0,
                       help='Spot offset in z (um, 3D only)')

    g_3d = ap.add_argument_group('3D specific')
    g_3d.add_argument('--lz-um', type=float, default=8.0,
                      help='TOTAL computational depth z (domain) in um, symmetric about z=0 '
                           '(-lz/2 .. +lz/2). NOT the waveguide width (that is --wg-width).')
    g_3d.add_argument('--air', type=float, default=3.0, help='Air above the WG in um (3D only)')
    g_3d.add_argument('--tear', type=float, default=None,
                      help='Tear + cornea region below the WG in um (3D only). Empty = AUTO: '
                           'lipid+aqueous+mucin + ~1.5 um cornea + absorber. Set explicitly '
                           'only to trim the cornea margin.')
    g_3d.add_argument('--wg-width', type=float, default=None,
                      help='WAVEGUIDE channel width in z (um), centered at z=0 (|z|<width/2) '
                           '-> rectangular channel, guiding in y AND z. Must be < --lz-um. '
                           'If omitted: slab (infinite in z, guiding in y only). 3D only.')
    g_3d.add_argument('--wg-clad-n', type=float, default=1.0,
                      help='Refractive index of the lateral cladding (3D only, default air)')
    g_3d.add_argument('--steps-factor', type=float, default=2.0, help='3D full only')
    g_3d.add_argument('--vol-dtype', choices=('float16', 'float32'), default='float32',
                      help='Storage format of the 3D volumes')
    g_3d.add_argument('--pec-faces', default='',
                      help='Comma list of PEC mirror faces, e.g. "xmax" (3D full only)')
    g_3d.add_argument('--end-facet', type=float, default=0.0,
                      help='Air zone at the WG end in um -> Fresnel (3D full only)')
    g_3d.add_argument('--save-vector', action='store_true',
                      help='Save Ex,Ey,Ez (3D full only)')

    g_stitch = ap.add_argument_group('Stitch / Meep')
    g_stitch.add_argument('--window-um', type=float, default=None,
                          help='Window width in um (--method stitch only)')
    g_stitch.add_argument('--slide-um', type=float, default=None,
                          help='Step size in um (--method stitch only)')
    g_stitch.add_argument('--meep-modes', type=int, default=1, metavar='M',
                          help='(engine meep, stitch, handoff mode) Number of guided modes '
                               'in the handoff. 1 = fundamental mode; >1 captures mode conversion.')
    g_stitch.add_argument('--meep-handoff', choices=('mode', 'field', 'vector'), default='mode',
                          help='(engine meep, stitch) mode = mode cascade (planar 2D); field = full '
                               'Ez/Hy field (planar 2D, guided-dominated only); vector = all tangential '
                               'E/H components via the equivalence principle (2D and 3D, planar, lens '
                               'and CAD; used automatically for 3D, lens and CAD).')
    g_stitch.add_argument('--meep-cad-eps', default=None, help=argparse.SUPPRESS)
    g_stitch.add_argument('--meep-np', type=int, default=1, metavar='N',
                          help='(engine meep) Number of MPI processes. Meep has NO GPU support '
                               'and parallelizes only via MPI (CPU cores). 1 = serial; '
                               '>1 launches the run via "mpirun -np N" (requires the MPI build of '
                               'pymeep). 0 = all available cores.')

    g_run = ap.add_argument_group('Run control')
    g_run.add_argument('--allow-large', action='store_true',
                       help='(engine meep) bypass the adaptive cell-count/RAM guard '
                            '(risk: OOM).')
    g_run.add_argument('--no-save', action='store_true', help='Do not save results')

    # Alle Parameter, die NICHT (voll) rechnen: Vorschau, Ressourcen-Check,
    # Kurz-Kalibrierung. Geordnet von "gar nicht rechnen" zu "kurz messen".
    g_prep = ap.add_argument_group('Calibration & preview')
    g_prep.add_argument('--dry-run', action='store_true',
                        help='Only show the resolved configuration, do NOT compute')
    g_prep.add_argument('--check-resources', action='store_true',
                        help='Check VRAM/RAM before the run (3D full only)')
    g_prep.add_argument('--calibrate', type=int, default=0, metavar='N',
                        help='Measure only N steps + extrapolate total runtime (3D full only)')
    return ap


def _split_absorption(args):
    """Komplexe Brechzahlen (--n-aqueous 1.331+0.001i) und --kappa-* zu reellen n plus
    args.kappa = {'wg','lip','aq','mu','co','bead'} (Extinktion kappa). Die Solver
    rechnen mit reellem n (Laufzeit, Mur, Ausgaben) und setzen kappa nur im Material
    als Leitfaehigkeit um (common.physics.loss_sigma)."""
    if getattr(args, '_kappa_done', False):
        return
    kap = {}
    for key, nd, kd in (('wg', 'wg_n', 'kappa_wg'), ('lip', 'n_lipid', 'kappa_lipid'),
                        ('aq', 'n_aqueous', 'kappa_aqueous'), ('mu', 'n_mucin', 'kappa_mucin'),
                        ('co', 'n_cornea', 'kappa_cornea'), ('bead', 'bead_n', 'kappa_bead')):
        n, k = split_n(getattr(args, nd, None))
        setattr(args, nd, n)
        kk = getattr(args, kd, None)
        k = float(kk) if kk is not None else k
        if k < 0:
            raise SystemExit(f'--{kd.replace("_", "-")} < 0 would mean gain.')
        if k > 0:
            kap[key] = k
    args.kappa = kap
    args._kappa_done = True
    if kap:
        print('[Absorption] kappa: ' + ', '.join(f'{k}={v:g}' for k, v in kap.items())
              + f'  (exact at {args.lambda_nm:g} nm; pulses: approximation)')


def resolve_args(args):
    """Normalisiert die geparsten Argumente (Geometrie-Alias, dim-abhaengige
    Defaults, sliding->stitch, grosse-Linse->stitch) und liefert den Schichtstapel.
    Wird von der nativen main() UND vom Meep-Treiber (meep/run_meep.py) genutzt,
    damit beide Engines exakt gleich parametrisiert werden."""
    # Geometrie-Alias aufloesen
    key = str(args.geometry).strip().lower()
    if key not in _GEOM_ALIASES:
        raise SystemExit(f'Unknown geometry "{args.geometry}". '
                         f'Allowed: lens/kontaktlinse or planar/waveguide.')
    args.geometry = _GEOM_ALIASES[key]

    _split_absorption(args)

    # Dimension-abhaengige Defaults
    if args.resolution_nm is None:
        args.resolution_nm = 20.0 if args.dim == 2 else 50.0
    if args.geometry == 'cad':
        args.length_um = args.length_um or 0.0   # Laenge kommt aus der CAD-Szene
    if args.length_um is None:
        args.length_um = DEFAULT_LENGTH_UM[(args.geometry, args.dim)]
    if args.window_um is None:
        args.window_um = 350.0 if args.dim == 2 else 20.0
    if args.slide_um is None:
        args.slide_um = 150.0 if args.dim == 2 else 12.0

    if str(args.boundary or '').lower() == 'mur1':
        print('[Warning] --boundary mur1: first-order Mur (local velocity c0/n_eff) is exact only for '
              'the fundamental mode at normal incidence; higher modes, radiation and '
              'bead scattering reflect more strongly (2D ~0.6%, 3D ~1% amplitude). '
              'Use cpml for quantitative results.')
    if args.dim == 2 and str(args.boundary or '').lower() == 'sponge':
        raise SystemExit('--boundary sponge is available in 3D only (2D: cpml or mur1).')

    # 'sliding' gibt es nur in 2D
    if args.dim == 3 and args.method == 'sliding':
        print('[Note] "sliding" exists in 2D only -> using "stitch" in 3D.')
        args.method = 'stitch'

    # Grosse Kontaktlinse: Stitch-Zwang (mit Hinweis)
    _enforce_lens_stitch(args)
    return _resolve_layers(args)


def main(argv=None):
    import sys as _sys
    raw = list(_sys.argv[1:]) if argv is None else list(argv)
    args = build_parser().parse_args(argv)

    # Engine-Weiche: Meep laeuft nur unter Linux/WSL. Auf Windows nach WSL auslagern.
    if args.engine == 'meep':
        if _GEOM_ALIASES.get(str(args.geometry).strip().lower()) == 'cad':
            # CAD-Geometrie unter Windows vorbereiten, Meep liest das eps-Gitter
            _p = _prepare_meep_cad(build_parser().parse_args(argv))
            raw = list(raw) + ['--meep-cad-eps', _to_wsl_path(_p) if os.name == 'nt' else _p]
        return _dispatch_meep(raw, np_ranks=args.meep_np)

    layers = resolve_args(args)
    args._cad = _make_cad(args) if args.geometry == 'cad' else None

    print('===================== FDTD configuration =====================')
    _gname = {'lens': 'contact lens (flattened)', 'planar': 'planar waveguide',
              'cad': f'CAD {os.path.basename(args.cad or "")}'}[args.geometry]
    print(f'  Geometry  : {_gname}')
    print(f'  Dimension : {args.dim}D')
    print(f'  Method    : {args.method}')
    if args._cad is not None:
        _c = args._cad
        print(_c.scene.summary(args.lambda_nm))
        _src = _c.source(0.0)
        print(f'  Box       : x 0..{_c.lx_um:.2f}  y {_c.y0_um:.2f}..{_c.y0_um + _c.ly_um:.2f}'
              + (f'  z {_c.z0_um:.2f}..{_c.z0_um + _c.lz_um:.2f} um' if args.dim == 3
                 else f' um, cut z={_c.z0_um:.2f} um'))
        print(f'  Core      : {_c.core}   entry face center {[round(v, 2) for v in _src["facet_center"]]} '
              f'normal {[round(v, 3) for v in _src["facet_normal"]]}')
        _op = _c.scene.open_faces
        print('  Open      : ' + (', '.join(f'{f["body"]}:{"+" if f["sign"] > 0 else "-"}{"xyz"[f["axis"]]}'
                                         for f in _op) if _op else 'none (all faces reflect)'))
        print(f'  Lambda: {args.lambda_nm:g} nm   Resolution: {args.resolution_nm:g} nm   Pol: {args.polarization}')
        print('==============================================================')
        if args.dry_run:
            print('[dry-run] Configuration resolved - no run started.')
            return
        if args.dim == 2:
            _run_2d(args, layers)
        else:
            _run_3d(args, layers)
        return
    if args.geometry == 'lens':
        print(f'  Scenario  : {args.scenario}  (lipid/aq/mu = '
              f'{layers["t_lip_um"]:g}/{layers["t_aq_um"]:g}/{layers["t_mu_um"]:g} um, '
              f'n_aq={layers["n_aq"]:.4f} @{args.lambda_nm:g}nm)')
    else:
        print(f'  Bead      : {"none" if not _place_bead(args) else f"{args.bead_material} d={args.bead_diameter:g} um"}')
    print(f'  Waveguide : thickness {layers["t_wg_um"]:g} um, n={layers["wg_n"]:.3f}')
    if args.geometry != 'cad':
        from common.layer_stack import describe as _describe_stack
        print(f'  Layers    : {_describe_stack(layers["stack"])}'
              + ('  [--layers]' if layers.get('stack_custom') else ''))
        _below = ((f'"{layers["stack"][-1]["name"]}"' if layers['stack'] else 'the core')
                  if layers.get('stack_custom') else 'the mucin')
        print('  Cornea    : ' + ('constant n=%.4f (--n-cornea)' % layers['n_co'] if layers['n_co'] is not None
              else ('layered, FULL thickness down to the aqueous humour (depth %.0f um)' % _cornea_full_tear(layers, 0.2)
                    if args.cornea_full else f'layered (epithelium directly below {_below})')))
    print(f'  Length    : {args.length_um:g} um   Resolution: {args.resolution_nm:g} nm   '
          f'Lambda: {args.lambda_nm:g} nm   Pol: {args.polarization}')
    if args.method == 'stitch':
        print(f'  Stitch    : window {args.window_um:g} um, Slide {args.slide_um:g} um')
    if args.dim == 3:
        print(f'  Depth z   : {args.lz_um:g} um   (air {args.air:g} / '
              f'tear {"auto" if args.tear is None else f"{args.tear:g} um"})')
        if args.calibrate:
            print(f'  Calibrate : measure {args.calibrate} steps (no full run)')
    print('==============================================================')

    if args.dry_run:
        print('[dry-run] Configuration resolved - no run started.')
        return

    _had_bead = _place_bead(args)
    if args.dim == 2:
        _run_2d(args, layers)
    else:
        _run_3d(args, layers)

    # STANDARD: nach dem Bead-Lauf automatisch den Referenzlauf OHNE Bead rechnen
    # (zweite Datei mit Suffix '_ref') -> mit EINEM Aufruf beide Laeufe fuer die
    # Bead-Differenz (Streufeld = mit - ohne). Mit --no-reference abschaltbar.
    if _had_bead and not getattr(args, 'no_reference', False):
        print('\n============ REFERENCE RUN (without bead) ============')
        args.no_bead = True                      # zweiter Lauf: kein Bead -> Datei _ref
        if args.dim == 2:
            _run_2d(args, layers)
        else:
            _run_3d(args, layers)
        print('  -> Reference (_ref) saved. In the analyzer: difference with - without bead.')


if __name__ == '__main__':
    main()
