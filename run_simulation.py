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

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ("planar_beads", "planar_3d"):
    _p = os.path.join(_REPO_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from common.physics import n_at            # noqa: E402  (Brechzahl aus Material+Lambda)

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
}


def _resolve_layers(args):
    """Liefert den Schichtstapel fuer die gewaehlte Geometrie. CLI-Overrides
    (--t-*/--n-*) haben Vorrang vor dem Geometrie-/Szenario-Default. None = der
    jeweilige Solver setzt seinen eigenen Default."""
    if args.geometry == 'lens':
        lip, aq, mu, n_aq = LENS_SCENARIOS[args.scenario]
        base = dict(t_wg_um=T_LENS_UM, t_lip_um=lip, t_aq_um=aq, t_mu_um=mu, n_aq=n_aq)
    else:
        base = dict(t_wg_um=args.wg_thickness_um, t_lip_um=0.0,
                    t_aq_um=None, t_mu_um=None, n_aq=None)
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
    return base


def _enforce_lens_stitch(args):
    """Grosse Kontaktlinse: Hinweis + automatische Umschaltung auf Stitch."""
    if args.geometry == 'lens' and args.length_um > LENS_STITCH_ONLY_UM \
            and args.method != 'stitch':
        print('')
        print('  ############################################################')
        print('  #  HINWEIS: GROSSE KONTAKTLINSE                            #')
        print(f'  #  Laenge {args.length_um:.0f} um (> {LENS_STITCH_ONLY_UM:.0f} um) passt NICHT')
        print('  #  komplett in den Speicher. Die volle Linse funktioniert  #')
        print('  #  NUR im STITCHED-Modus (Gebiets-Zerlegung).              #')
        print(f'  #  -> schalte Methode "{args.method}" automatisch auf "stitch".')
        print('  ############################################################')
        print('')
        args.method = 'stitch'


def _place_bead(args):
    """Bead nur im planaren Fall und nur wenn nicht --no-bead und d>0."""
    return (args.geometry == 'planar') and (not args.no_bead) and (args.bead_diameter > 0)


def _run_2d(args, layers):
    import fdtd2d_core as d2
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
        bead_x_um=args.bead_x, wg_n=layers['wg_n'], bead_n=args.bead_n,
        n_aqueous=layers['n_aq'], n_mucin=layers['n_mu'],
        n_cornea=layers['n_co'], n_lipid=layers['n_lip'],
        polarization=args.polarization,
        curved=(args.curved and args.geometry == 'lens'),
        boundary=args.boundary)
    if not args.no_save:
        d2.save_results(result, out_dir=os.path.join(_REPO_ROOT, 'results'))
    return result


def _run_3d(args, layers):
    import fdtd3d_core as f3d
    bead = None
    if _place_bead(args):
        bead = dict(x_um=(args.bead_x if args.bead_x is not None else args.length_um/2.0),
                    d_um=args.bead_diameter,
                    n=(args.bead_n if args.bead_n is not None
                       else f3d.n_at(args.bead_material, args.lambda_nm)))
    # Layer-Defaults fuer 3D (run_3d verlangt konkrete Werte)
    t_aq = layers['t_aq_um'] if layers['t_aq_um'] is not None else 4.0
    t_mu = layers['t_mu_um'] if layers['t_mu_um'] is not None else 2.0
    n_aq = layers['n_aq'] if layers['n_aq'] is not None else f3d.n_at('aqueous', args.lambda_nm)
    n_mu = layers['n_mu'] if layers['n_mu'] is not None else f3d.n_at('mucin', args.lambda_nm)
    n_co = layers['n_co'] if layers['n_co'] is not None else f3d.n_at('cornea', args.lambda_nm)
    n_lip = layers['n_lip'] if layers['n_lip'] is not None else 1.480
    label = f'{args.geometry}3d_{args.scenario if args.geometry=="lens" else args.bead_material}'
    # Gemeinsame Argumente fuer run_3d UND run_3d_stitched
    common = dict(
        label=label, wg_n=layers['wg_n'], t_wg_um=layers['t_wg_um'],
        t_lip_um=(layers['t_lip_um'] or 0.0), t_aq_um=t_aq, t_mu_um=t_mu,
        n_lip=n_lip, n_aq=n_aq, n_mu=n_mu, n_co=n_co,
        lam_nm=args.lambda_nm, lx_um=args.length_um, air_um=args.air,
        tear_um=args.tear, lz_um=args.lz_um, dx_nm=args.resolution_nm,
        vcsel_waist_um=args.vcsel_waist, vcsel_waist_z_um=args.vcsel_waist_z,
        source_type=args.source_type, n_snapshots=args.snapshots, bead=bead,
        vol_dtype=args.vol_dtype, wg_width_um=args.wg_width,
        n_clad_side=args.wg_clad_n, vcsel_tilt_deg=args.vcsel_tilt,
        vcsel_offset_y_um=args.vcsel_offset, vcsel_offset_z_um=args.vcsel_offset_z,
        input_gap_um=args.input_gap, polarization=args.polarization)
    if args.method == 'stitch':
        result = f3d.run_3d_stitched(window_w_um=args.window_um,
                                     slide_um=args.slide_um, **common)
    else:
        # nur run_3d (full) kennt diese Zusatz-Optionen
        pec_faces = tuple(f.strip() for f in args.pec_faces.split(',') if f.strip())
        result = f3d.run_3d(
            steps_factor=args.steps_factor, calibrate_steps=args.calibrate,
            save_vector=args.save_vector, check_resources=args.check_resources,
            allow_large=args.allow_large,
            pec_faces=pec_faces, end_facet_um=args.end_facet, **common)
        if args.calibrate:
            return result   # Kalibrierlauf: kein Speichern
    if not args.no_save:
        f3d.save_results_3d(result, out_dir=os.path.join(_REPO_ROOT, 'results'),
                            prefix=f'{args.geometry}3d')
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
        print(f'[engine=meep] Windows erkannt -> Auslagerung nach WSL ({WSL_DISTRO})')
        if mpi:
            print(f'  MPI: {mpi.strip()} (Meep nutzt nur CPU-Kerne, kein GPU)')
        print('  ' + ' '.join(cmd[:6]) + ' ...')
    else:
        base = [sys.executable, os.path.join(_REPO_ROOT, 'meep', 'run_meep.py')] + args_fwd
        if mpi:
            n = str(os.cpu_count() or 1) if np_ranks == 0 else str(int(np_ranks))
            cmd = ['mpirun', '-np', n] + base
        else:
            cmd = base
        print('[engine=meep] Linux -> Meep direkt' + (f' (mpirun -np {n})' if mpi else ''))
    try:
        return subprocess.call(cmd)
    except FileNotFoundError as e:
        raise SystemExit(f'[FEHLER] Meep-Start fehlgeschlagen ({e}). '
                         f'Ist WSL/Distro "{WSL_DISTRO}" eingerichtet? Siehe docs/meep_setup_wsl.md')


def _q(s):
    """Minimales Shell-Quoting fuer die Weitergabe an bash -lc."""
    return "'" + s.replace("'", "'\\''") + "'" if any(c in s for c in ' "\'$`\\') else s


def build_parser():
    ap = argparse.ArgumentParser(
        description='Vereinheitlichter FDTD-Einstieg: Kontaktlinse | planar, in 2D | 3D.')
    # Argumente in benannte GRUPPEN -> dienen zugleich als Sektionen im GUI.
    g_case = ap.add_argument_group('Fall-Auswahl')
    g_case.add_argument('--engine', '-e', choices=('native', 'meep'), default='native',
                        help='native = eigener NumPy/CuPy-Solver; meep = Meep-Toolbox '
                             '(laeuft nur unter Linux/WSL, wird auf Windows automatisch '
                             'nach WSL ausgelagert)')
    g_case.add_argument('--geometry', '-g', default='planar',
                        help='kontaktlinse/lens ODER planar/waveguide (Default: planar)')
    g_case.add_argument('--dim', '-d', type=int, choices=(2, 3), default=3,
                        help='Dimension: 2 oder 3 (Default: 3)')
    g_case.add_argument('--method', '-m', choices=('full', 'sliding', 'stitch'),
                        default='full', help='full | sliding (nur 2D) | stitch')
    g_case.add_argument('--scenario', default='Gesund', choices=list(LENS_SCENARIOS),
                        help='Trockenauge-Szenario (nur --geometry lens)')
    g_case.add_argument('--curved', action='store_true',
                        help='(nur lens, 2D) ECHTE Kruemmung via konformer Abbildung '
                             'eps*(1+u/R)^2 statt flattened. Wirkt in TE UND TM.')

    g_basic = ap.add_argument_group('Grundeinstellungen')
    g_basic.add_argument('--length-um', type=float, default=None,
                         help='Propagationslaenge in um (Default geometrie-/dim-abhaengig)')
    g_basic.add_argument('--resolution-nm', type=float, default=None,
                         help='dx in nm (Default: 2D=20, 3D=50)')
    g_basic.add_argument('--lambda-nm', type=float, default=850.0, help='Wellenlaenge in nm')
    g_basic.add_argument('--polarization', choices=('s', 'p'), default='s',
                         help='s=TE (Ez) | p=TM (Ey)')
    g_basic.add_argument('--boundary', choices=('mur1', 'mur2', 'cpml'), default='mur1',
                         help='Absorbierender Rand (alle 2D-Methoden): mur1 (Default) | '
                              'mur2 (besser schraeg) | cpml (beste Absorption). '
                              'full/sliding: alle Kanten; stitch: y-Kanten + echte '
                              'x-Bauteilenden (mur2->cpml wg. Handoff-Stabilitaet)')
    g_basic.add_argument('--source-type', choices=('cw', 'pulse'), default='cw')
    g_basic.add_argument('--snapshots', type=int, default=8, help='Anzahl gespeicherter Frames')

    g_mat = ap.add_argument_group('Material & Schichten')
    g_mat.add_argument('--wg-material', choices=('pmma', 'polystyrol'), default='pmma',
                       help='Waveguide-Material (Brechzahl aus Dispersion; --wg-n ueberschreibt)')
    g_mat.add_argument('--wg-thickness-um', type=float, default=5.0,
                       help='Waveguide-Dicke (nur planar; Linse nutzt fest 250um)')
    g_mat.add_argument('--wg-n', type=float, default=None,
                       help='Freie WG-Brechzahl (ueberschreibt --wg-material)')
    g_mat.add_argument('--t-aqueous', type=float, default=None, help='Aqueous-Dicke um')
    g_mat.add_argument('--t-mucin', type=float, default=None, help='Mucin-Dicke um')
    g_mat.add_argument('--t-lipid', type=float, default=None, help='Lipid-Dicke um')
    g_mat.add_argument('--n-aqueous', type=float, default=None, help='Freie Aqueous-Brechzahl')
    g_mat.add_argument('--n-mucin', type=float, default=None, help='Freie Mucin-Brechzahl')
    g_mat.add_argument('--n-cornea', type=float, default=None, help='Freie Cornea-Brechzahl')
    g_mat.add_argument('--n-lipid', type=float, default=None, help='Freie Lipid-Brechzahl')

    g_bead = ap.add_argument_group('Bead (nur planar)')
    g_bead.add_argument('--bead-material', choices=('polystyrol', 'pmma'),
                        default='polystyrol', help='Bead-Material (Brechzahl aus Dispersion)')
    g_bead.add_argument('--bead-diameter', type=float, default=0.5,
                        help='Bead-Durchmesser in um (0/--no-bead = kein Bead)')
    g_bead.add_argument('--bead-x', type=float, default=None,
                        help='Bead-x-Position in um (Default Mitte)')
    g_bead.add_argument('--bead-y-um', type=float, default=None,
                        help='Bead-y-Position in um (Default: unter WG bei -d/2; '
                             '0..t_wg = im WG-Kern -> starke Streuung)')
    g_bead.add_argument('--bead-n', type=float, default=None,
                        help='Freie Bead-Brechzahl (ueberschreibt --bead-material)')
    g_bead.add_argument('--no-bead', action='store_true',
                        help='planaren Lauf OHNE Bead (Referenz)')

    g_src = ap.add_argument_group('Quelle (VCSEL)')
    g_src.add_argument('--vcsel-waist', type=float, default=2.0, help='Taille (y) in um')
    g_src.add_argument('--vcsel-waist-z', type=float, default=2.0, help='Taille z in um (nur 3D)')
    g_src.add_argument('--vcsel-tilt', type=float, default=0.0, help='Strahlneigung in Grad')
    g_src.add_argument('--vcsel-offset', type=float, default=0.0,
                       help='Spot-Versatz in y (um); in 3D = Offset y')
    g_src.add_argument('--vcsel-offset-z', type=float, default=0.0,
                       help='Spot-Versatz in z (um, nur 3D)')
    g_src.add_argument('--input-gap', type=float, default=0.0,
                       help='Einkoppelabstand Laser-zu-WG in um (Luftweg + Fresnel-Eintritt)')

    g_3d = ap.add_argument_group('3D-spezifisch')
    g_3d.add_argument('--lz-um', type=float, default=8.0, help='Domain-Tiefe z in um (nur 3D)')
    g_3d.add_argument('--air', type=float, default=3.0, help='Luft ueber WG in um (nur 3D)')
    g_3d.add_argument('--tear', type=float, default=None,
                      help='Tear+Cornea-Gebiet unter WG in um (nur 3D). Leer = AUTO: '
                           'Lipid+Aqueous+Mucin + ~1.5um Cornea + Absorber. Nur zum '
                           'Trimmen der Cornea-Marge explizit setzen.')
    g_3d.add_argument('--wg-width', type=float, default=None,
                      help='WG-Kernbreite in z (um) -> Rechteck-Kanal, Fuehrung in y UND z '
                           '(ohne Angabe: Slab, nur y-Fuehrung; nur 3D)')
    g_3d.add_argument('--wg-clad-n', type=float, default=1.0,
                      help='Brechzahl seitliches Cladding (nur 3D, Default Luft)')
    g_3d.add_argument('--steps-factor', type=float, default=2.0, help='nur 3D full')
    g_3d.add_argument('--vol-dtype', choices=('float16', 'float32'), default='float32',
                      help='Speicherformat der 3D-Volumina')
    g_3d.add_argument('--pec-faces', default='',
                      help='Komma-Liste PEC-Spiegelflaechen, z.B. "xmax" (nur 3D full)')
    g_3d.add_argument('--end-facet', type=float, default=0.0,
                      help='Luftzone am WG-Ende in um -> Fresnel (nur 3D full)')
    g_3d.add_argument('--save-vector', action='store_true',
                      help='Ex,Ey,Ez speichern (nur 3D full)')

    g_stitch = ap.add_argument_group('Stitch / Meep')
    g_stitch.add_argument('--window-um', type=float, default=None,
                          help='Fensterbreite in um (nur --method stitch)')
    g_stitch.add_argument('--slide-um', type=float, default=None,
                          help='Schrittweite in um (nur --method stitch)')
    g_stitch.add_argument('--meep-modes', type=int, default=1, metavar='M',
                          help='(engine meep, stitch, handoff mode) Anzahl gefuehrter Moden '
                               'im Handoff. 1=Fundamentalmode; >1 erfasst Mode-Konversion.')
    g_stitch.add_argument('--meep-handoff', choices=('mode', 'field'), default='mode',
                          help='(engine meep, stitch) mode = Mode-Kaskade (robust, empfohlen); '
                               'field = VOLL-FELD (nur gefuehrt-dominiert; bricht bei starker '
                               'Streuung).')
    g_stitch.add_argument('--meep-np', type=int, default=1, metavar='N',
                          help='(engine meep) Anzahl MPI-Prozesse. Meep hat KEINEN GPU-Support '
                               'und parallelisiert nur ueber MPI (CPU-Kerne). 1=seriell; '
                               '>1 startet den Lauf via "mpirun -np N" (setzt den MPI-Build von '
                               'pymeep voraus). 0=alle verfuegbaren Kerne.')

    g_run = ap.add_argument_group('Ablauf')
    g_run.add_argument('--allow-large', action='store_true',
                       help='(engine meep) den adaptiven Zellzahl-/RAM-Schutz umgehen '
                            '(Risiko: OOM).')
    g_run.add_argument('--no-save', action='store_true', help='nicht speichern')

    # Alle Parameter, die NICHT (voll) rechnen: Vorschau, Ressourcen-Check,
    # Kurz-Kalibrierung. Geordnet von "gar nicht rechnen" zu "kurz messen".
    g_prep = ap.add_argument_group('Kalibrierung & Vorschau')
    g_prep.add_argument('--dry-run', action='store_true',
                        help='nur die aufgeloeste Konfiguration zeigen, NICHT rechnen')
    g_prep.add_argument('--check-resources', action='store_true',
                        help='vor dem Lauf VRAM/RAM pruefen (nur 3D full)')
    g_prep.add_argument('--calibrate', type=int, default=0, metavar='N',
                        help='nur N Steps messen + Gesamtlaufzeit hochrechnen (nur 3D full)')
    return ap


def resolve_args(args):
    """Normalisiert die geparsten Argumente (Geometrie-Alias, dim-abhaengige
    Defaults, sliding->stitch, grosse-Linse->stitch) und liefert den Schichtstapel.
    Wird von der nativen main() UND vom Meep-Treiber (meep/run_meep.py) genutzt,
    damit beide Engines exakt gleich parametrisiert werden."""
    # Geometrie-Alias aufloesen
    key = str(args.geometry).strip().lower()
    if key not in _GEOM_ALIASES:
        raise SystemExit(f'Unbekannte Geometrie "{args.geometry}". '
                         f'Erlaubt: lens/kontaktlinse oder planar/waveguide.')
    args.geometry = _GEOM_ALIASES[key]

    # Dimension-abhaengige Defaults
    if args.resolution_nm is None:
        args.resolution_nm = 20.0 if args.dim == 2 else 50.0
    if args.length_um is None:
        args.length_um = DEFAULT_LENGTH_UM[(args.geometry, args.dim)]
    if args.window_um is None:
        args.window_um = 350.0 if args.dim == 2 else 20.0
    if args.slide_um is None:
        args.slide_um = 150.0 if args.dim == 2 else 12.0

    # 'sliding' gibt es nur in 2D
    if args.dim == 3 and args.method == 'sliding':
        print('[Hinweis] "sliding" existiert nur in 2D -> nutze in 3D "stitch".')
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
        return _dispatch_meep(raw, np_ranks=args.meep_np)

    layers = resolve_args(args)

    print('===================== FDTD-Konfiguration =====================')
    print(f'  Geometrie : {"Kontaktlinse (geflattet)" if args.geometry=="lens" else "planarer Waveguide"}')
    print(f'  Dimension : {args.dim}D')
    print(f'  Methode   : {args.method}')
    if args.geometry == 'lens':
        print(f'  Szenario  : {args.scenario}  (lipid/aq/mu = '
              f'{layers["t_lip_um"]:g}/{layers["t_aq_um"]:g}/{layers["t_mu_um"]:g} um, '
              f'n_aq={layers["n_aq"]})')
    else:
        print(f'  Bead      : {"nein" if not _place_bead(args) else f"{args.bead_material} d={args.bead_diameter:g} um"}')
    print(f'  Waveguide : Dicke {layers["t_wg_um"]:g} um, n={layers["wg_n"]:.3f}')
    print(f'  Laenge    : {args.length_um:g} um   Aufloesung: {args.resolution_nm:g} nm   '
          f'Lambda: {args.lambda_nm:g} nm   Pol: {args.polarization}')
    if args.method == 'stitch':
        print(f'  Stitch    : Fenster {args.window_um:g} um, Slide {args.slide_um:g} um')
    if args.dim == 3:
        print(f'  Tiefe z   : {args.lz_um:g} um   (air {args.air:g} / '
              f'tear {"auto" if args.tear is None else f"{args.tear:g} um"})')
        if args.calibrate:
            print(f'  Kalibrier.: {args.calibrate} Steps messen (kein voller Lauf)')
    print('==============================================================')

    if args.dry_run:
        print('[dry-run] Konfiguration aufgeloest - kein Lauf gestartet.')
        return

    if args.dim == 2:
        _run_2d(args, layers)
    else:
        _run_3d(args, layers)


if __name__ == '__main__':
    main()
