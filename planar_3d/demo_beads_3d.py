"""Echte 3D-FDTD-Demo: EIN spaerischer Bead (Kugel!) am Waveguide-Tear-Interface.

Gegenstueck zu demo_beads.py, aber volle Yee-Zelle. Der Bead ist hier eine
echte Kugel (nicht ein unendlicher Zylinder wie in 2D), die Quelle ein
Gauss-Spot mit Taille in y und z.

Ausschnitt statt 1mm-Waveguide: rechnet nur ein paar zehn um um den Bead
herum (das physikalisch Interessante), damit 3D auf einer 4GB-GPU laeuft.

Beispiel:
  python demo_beads_3d.py --wg-material pmma --bead-material polystyrol \\
         --bead-diameter 0.5 --gpu
  python demo_beads_3d.py --wg-material pmma --bead-material polystyrol \\
         --bead-diameter 0.5 --no-bead   (Referenz ohne Bead)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fdtd3d_core as f3d


def main():
    ap = argparse.ArgumentParser(description='Echte 3D-FDTD mit einem Bead (Kugel)')
    # --- Materialien (Namen) ODER freie Brechzahlen (--*-n ueberschreibt Namen) ---
    ap.add_argument('--wg-material', choices=['polystyrol', 'pmma'], default='pmma')
    ap.add_argument('--bead-material', choices=['polystyrol', 'pmma'], default='polystyrol')
    ap.add_argument('--wg-n', type=float, default=None,
                    help='Freie WG-Brechzahl (ueberschreibt --wg-material)')
    ap.add_argument('--bead-n', type=float, default=None,
                    help='Freie Bead-Brechzahl (ueberschreibt --bead-material)')
    ap.add_argument('--n-aqueous', type=float, default=None,
                    help='Freie Brechzahl Aqueous-Schicht (Default ~1.336)')
    ap.add_argument('--n-mucin', type=float, default=None,
                    help='Freie Brechzahl Mucin-Schicht (Default ~1.342)')
    ap.add_argument('--n-cornea', type=float, default=None,
                    help='Freie Brechzahl Cornea/Substrat (Default ~1.376)')
    ap.add_argument('--n-lipid', type=float, default=None,
                    help='Freie Brechzahl Lipid-Schicht (nur falls --t-lipid>0)')
    # --- Bead ---
    ap.add_argument('--bead-diameter', type=float, required=True,
                    help='Bead-Durchmesser in um (echte Kugel)')
    ap.add_argument('--bead-x', type=float, default=None,
                    help='Bead-Position x (um). Default: Domainmitte lx/2')
    ap.add_argument('--bead-y', type=float, default=None,
                    help='Bead-Position y (um). Default: aufliegend an WG-Unterseite (-d/2)')
    ap.add_argument('--bead-z', type=float, default=None,
                    help='Bead-Position z (um). Default: Domainmitte 0')
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--resolution', type=float, default=50.0, help='dx in nm')
    ap.add_argument('--wg-thickness', type=float, default=5.0)
    ap.add_argument('--lambda-nm', type=float, default=850.0)
    # --- Laser / Einkopplung ---
    ap.add_argument('--vcsel-waist', type=float, default=2.0, help='Taille y in um')
    ap.add_argument('--vcsel-waist-z', type=float, default=2.0, help='Taille z in um')
    ap.add_argument('--vcsel-tilt', type=float, default=0.0,
                    help='Strahlneigung in der x-y-Ebene in Grad (0 = gerade in +x)')
    ap.add_argument('--vcsel-offset-y', type=float, default=0.0,
                    help='Versatz des Spot-Zentrums in y (um) relativ zur WG-Mitte')
    ap.add_argument('--vcsel-offset-z', type=float, default=0.0,
                    help='Versatz des Spot-Zentrums in z (um) relativ zur Domainmitte')
    ap.add_argument('--input-gap', type=float, default=0.0,
                    help='Abstand Laser <-> WG-Anfang in um (Luftweg vor der '
                         'Eintrittsfacette; 0 = Quelle direkt im WG)')
    ap.add_argument('--source-type', choices=['cw', 'pulse'], default='cw')
    ap.add_argument('--lx', type=float, default=40.0, help='Domain-Laenge x (um)')
    ap.add_argument('--lz', type=float, default=8.0, help='Domain-Tiefe z (um)')
    ap.add_argument('--wg-width', type=float, default=None,
                    help='WG-Kernbreite in Tiefe/z (um). Ohne Angabe: Slab '
                         '(nur y-Fuehrung). Mit Angabe: Rechteck-Kanal (y+z-Fuehrung)')
    ap.add_argument('--wg-clad-n', type=float, default=1.0,
                    help='Brechzahl des seitlichen Claddings (Default Luft=1.0)')
    ap.add_argument('--air', type=float, default=3.0, help='Luft ueber WG (um)')
    ap.add_argument('--tear', type=float, default=8.0, help='Tear+Cornea unter WG (um)')
    # --- Schichtdicken (um) ---
    ap.add_argument('--t-aqueous', type=float, default=None,
                    help='Dicke Aqueous-Schicht (um). Default: 4 fuer Bead>=1um, sonst 1')
    ap.add_argument('--t-mucin', type=float, default=2.0,
                    help='Dicke Mucin-Schicht (um)')
    ap.add_argument('--t-lipid', type=float, default=0.0,
                    help='Dicke Lipid-Schicht direkt unter WG (um, 0 = keine)')
    ap.add_argument('--snapshots', type=int, default=8)
    ap.add_argument('--steps-factor', type=float, default=2.0)
    ap.add_argument('--no-bead', action='store_true', help='Referenzlauf ohne Bead')
    ap.add_argument('--vol-dtype', choices=['float16', 'float32'], default='float16',
                    help='Speicherformat der 3D-Volumina (float16 halbiert Disk/RAM)')
    ap.add_argument('--calibrate', type=int, default=0, metavar='N',
                    help='Nur Kalibrierung: N Steps messen, Laufzeit hochrechnen, '
                         'NICHT den vollen Lauf starten (z.B. --calibrate 30)')
    ap.add_argument('--save-vector', action='store_true',
                    help='Vollen Vektor speichern (Ex,Ey,Ez -> |E| im Viewer). '
                         '~3x Volumen-Speicher/Disk.')
    ap.add_argument('--check-resources', action='store_true',
                    help='Vor dem Lauf VRAM/RAM-Bedarf pruefen; passt es nicht, '
                         'sauber abbrechen mit Empfehlung (statt CUDA-Absturz).')
    ap.add_argument('--pec-faces', default='',
                    help='Komma-Liste von Raendern als Spiegel (PEC), z.B. '
                         '"xmax" oder "xmin,xmax". Rest bleibt absorbierend.')
    ap.add_argument('--end-facet', type=float, default=0.0,
                    help='Luftzone am WG-Ende in um -> Fresnel-Reflexion an der '
                         'Material/Luft-Facette (0 = aus).')
    ap.add_argument('--polarization', choices=['s', 'p'], default='s',
                    help='s/TE = Quelle treibt Ez (E entlang Tiefe z); '
                         'p/TM = Quelle treibt Ey (E in x-y-Ebene, entlang Dicke).')
    ap.add_argument('--method', choices=['full', 'stitch'], default='full',
                    help='full = volle Domain; stitch = Gebiets-Zerlegung '
                         '(voller gefuellter WG, speicherschonend, CW-Handoff).')
    ap.add_argument('--window-w', type=float, default=20.0,
                    help='Fensterbreite in um (nur --method stitch)')
    ap.add_argument('--slide', type=float, default=12.0,
                    help='Schrittweite in um (< Fensterbreite; nur stitch)')
    args = ap.parse_args()
    pec_faces = tuple(f.strip() for f in args.pec_faces.split(',') if f.strip())

    if args.gpu and not f3d.GPU_AVAILABLE:
        print('[Warnung] --gpu angefordert aber CuPy fehlt - CPU-Fallback.')

    # Brechzahlen: freie Werte (--*-n) haben Vorrang vor Materialnamen.
    wg_n = args.wg_n if args.wg_n is not None else f3d.n_at(args.wg_material, args.lambda_nm)
    bead_n = args.bead_n if args.bead_n is not None else f3d.n_at(args.bead_material, args.lambda_nm)
    n_aq = args.n_aqueous if args.n_aqueous is not None else f3d.n_at('aqueous', args.lambda_nm)
    n_mu = args.n_mucin if args.n_mucin is not None else f3d.n_at('mucin', args.lambda_nm)
    n_co = args.n_cornea if args.n_cornea is not None else f3d.n_at('cornea', args.lambda_nm)
    n_lip = args.n_lipid if args.n_lipid is not None else 1.480

    # Aqueous grenzt direkt an den WG und hat eine FESTE, bead-unabhaengige Dicke.
    # Default: fuellt die Tear-Zone (tear - mucin - lipid), mind. Bead-Durchmesser
    # (damit der Bead in der Aqueous liegt); per --t-aqueous frei ueberschreibbar.
    t_aq = args.t_aqueous if args.t_aqueous is not None else \
        max(args.bead_diameter, args.tear - args.t_mucin - args.t_lipid)
    place = not args.no_bead
    bead_x = args.bead_x if args.bead_x is not None else args.lx/2.0
    bead = None
    if place:
        bead = dict(x_um=bead_x, d_um=args.bead_diameter, n=bead_n)
        if args.bead_y is not None:
            bead['y_um'] = args.bead_y
        if args.bead_z is not None:
            bead['z_um'] = args.bead_z

    label = (f'WG-{args.wg_material}_Bead-{args.bead_material}'
             f'_d{args.bead_diameter:g}_L{args.lambda_nm:g}_3D'
             + ('' if place else '_ref'))

    if args.method == 'stitch':
        label += '_stitch'
        result = f3d.run_3d_stitched(
            label=label, wg_n=wg_n, window_w_um=args.window_w, slide_um=args.slide,
            t_wg_um=args.wg_thickness, t_lip_um=args.t_lipid, t_aq_um=t_aq,
            t_mu_um=args.t_mucin, n_lip=n_lip, n_aq=n_aq, n_mu=n_mu, n_co=n_co,
            lam_nm=args.lambda_nm, lx_um=args.lx, air_um=args.air, tear_um=args.tear,
            lz_um=args.lz, dx_nm=args.resolution, vcsel_waist_um=args.vcsel_waist,
            vcsel_waist_z_um=args.vcsel_waist_z, source_type=args.source_type,
            n_snapshots=args.snapshots, bead=bead, vol_dtype=args.vol_dtype,
            wg_width_um=args.wg_width, n_clad_side=args.wg_clad_n,
            vcsel_tilt_deg=args.vcsel_tilt, vcsel_offset_y_um=args.vcsel_offset_y,
            vcsel_offset_z_um=args.vcsel_offset_z, input_gap_um=args.input_gap,
            polarization=args.polarization)
    else:
        result = f3d.run_3d(
            label=label, wg_n=wg_n, t_wg_um=args.wg_thickness,
            t_lip_um=args.t_lipid, t_aq_um=t_aq, t_mu_um=args.t_mucin,
            n_lip=n_lip, n_aq=n_aq, n_mu=n_mu, n_co=n_co, lam_nm=args.lambda_nm,
            lx_um=args.lx, air_um=args.air, tear_um=args.tear, lz_um=args.lz,
            dx_nm=args.resolution, vcsel_waist_um=args.vcsel_waist,
            vcsel_waist_z_um=args.vcsel_waist_z, source_type=args.source_type,
            n_snapshots=args.snapshots, steps_factor=args.steps_factor,
            bead=bead, vol_dtype=args.vol_dtype, calibrate_steps=args.calibrate,
            wg_width_um=args.wg_width, n_clad_side=args.wg_clad_n,
            save_vector=args.save_vector, check_resources=args.check_resources,
            pec_faces=pec_faces, end_facet_um=args.end_facet,
            vcsel_tilt_deg=args.vcsel_tilt, vcsel_offset_y_um=args.vcsel_offset_y,
            vcsel_offset_z_um=args.vcsel_offset_z, input_gap_um=args.input_gap,
            polarization=args.polarization)

    if args.calibrate:
        return   # Kalibrierlauf: kein Speichern

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    f3d.save_results_3d(result, out_dir=out_dir, prefix='beads3d',
                        material_names=['WG', 'Bead', 'Aqueous', 'Mucin', 'Cornea'])


if __name__ == '__main__':
    main()
