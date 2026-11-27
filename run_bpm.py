"""2D wide-angle beam propagation (BPM) through the contact lens / planar waveguide.

Uses the SAME parameters as run_simulation.py (geometry, scenario, layers, --curved,
VCSEL waist/tilt/offset, polarization, wavelength, length) plus a few BPM settings, so
a BPM run and an FDTD run of the same case are directly comparable.

    python run_bpm.py --geometry lens --curved --length-um 14000 --vcsel-tilt 20
    python run_bpm.py --geometry lens --length-um 2000 --polarization p --bpm-pade 2

The VCSEL beam is defined as in the FDTD (source plane --vcsel-distance / at least 1 um
before the facet), propagated through the air gap and transmitted through the facet
with the Fresnel coefficients of each plane-wave component (laser power = 1, power in
the core = coupling efficiency). Output: results/bpm_*.npz
with |E|^2 (TE: |Ez|^2, TM: |Ex|^2+|Ey|^2) on the non-uniform y grid every --bpm-rec-um, the
power in core / tear film / cornea / air along x, and the intensity at fixed depths
below the core.

Absorption (--kappa-aqueous, --kappa-lipid, ..., or a complex n in --layers, as in the
FDTD): the absorbed power per layer along x (A_cum_<layer>, A_rate_<layer> per um) and
the tear film's loss in dB/mm (ATR) are printed and saved.

2D means infinitely wide in z. The beam spreading in z is added analytically
(common.bpm.z_beam_width; --bpm-z-model): w_z(x) and the on-axis factor
sqrt(2/pi)/w_z are saved, and the log gives the on-axis intensity below the core in
mW/mm^2 at --laser-power (I3D = P_laser * I2D * factor).
"""
import contextlib
import io
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run_simulation as rs                                       # noqa: E402
from common import bpm as B                                       # noqa: E402

PROBE_DEPTHS_UM = (0.1, 0.5, 1.0, 2.0, 3.0)
R_BEND_UM = 8300.0


def build_parser():
    ap = rs.build_parser()
    ap.prog = 'run_bpm.py'
    g = ap.add_argument_group('BPM (run_bpm.py only)')
    g.add_argument('--bpm-dx-um', type=float, default=0.1,
                   help='Step along x in um (Crank-Nicolson; 0.1 is converged for beams '
                        'up to ~20 deg in PMMA)')
    g.add_argument('--bpm-dy-fine-nm', type=float, default=5.0,
                   help='y spacing at the tear film and the top surface (nm)')
    g.add_argument('--bpm-dy-core-nm', type=float, default=10.0,
                   help='y spacing inside the core (nm); sets the phase accuracy of steep '
                        'rays over long distances (10 nm: < 0.5 rad over 4 mm at 20 deg)')
    g.add_argument('--bpm-pade', type=int, default=1, choices=(1, 2, 3, 4),
                   help='Pade order of the wide-angle operator (1 = Hadley)')
    g.add_argument('--bpm-n-ref', type=float, default=None,
                   help='Reference index (default n_core*cos(beam angle in the core))')
    g.add_argument('--bpm-rec-um', type=float, default=1.0,
                   help='Store |E|^2 every ... um along x')
    g.add_argument('--bpm-z-model', default='auto', choices=('auto', 'sphere', 'flat'),
                   help='Beam spreading in z (the 2D run is infinitely wide in z): sphere = '
                        'great-circle convergence on the curved lens (R = 8.3 mm), flat = '
                        'free spreading; auto = sphere with --curved, else flat')
    g.add_argument('--bpm-start-fdtd-um', type=float, default=None, metavar='X',
                   help='Start the BPM with the field of a short 2D FDTD coupling run (VCSEL, '
                        'air gap, facet, near field exactly as run_simulation.py, same '
                        '--resolution-nm) taken X um after the facet, instead of the analytic '
                        'facet model. Costs ~1 min FDTD per 10 um at 40 nm.')
    g.add_argument('--bpm-out', default=None, help='Output file (default results/bpm_*.npz)')
    return ap


def z_model(args, c, x_um):
    """w_z(x) and the on-axis 3D factor sqrt(2/pi)/w_z for the BPM x positions."""
    m = args.bpm_z_model
    if m == 'auto':
        m = 'sphere' if c['curved'] else 'flat'
    wz0 = float(args.vcsel_waist_z if getattr(args, 'vcsel_waist_z', None) is not None
                else args.vcsel_waist)
    R = R_BEND_UM if m == 'sphere' else None
    w = B.z_beam_width(x_um, c['lam'], c['n_ref'], wz0, float(args.input_gap or 0.0), R)
    return dict(model=m, waist_z_um=wz0, w_z_um=w, z_factor_per_um=B.z_peak_factor(w))


def print_absolute(args, c, r, zm):
    """Peak on-axis intensity below the core at the laser power, with the z spreading."""
    P_W = float(args.laser_power)*1e-3
    F = zm['z_factor_per_um']
    print(f'[3D] z spreading ({zm["model"]}, w0z {zm["waist_z_um"]:g} um): w_z = '
          f'{zm["w_z_um"][-1]:.0f} um at x = {r["x_um"][-1]:g} um')
    for k, d in enumerate(PROBE_DEPTHS_UM):
        I3 = P_W*r['I_probe'][:, k]*F*1e9                      # W/um^2 -> mW/mm^2
        i = int(np.argmax(I3))
        print(f'[3D]   |E|^2 at {d:g} um below the core, on axis, {args.laser_power:g} mW laser: '
              f'max {I3[i]:.3e} mW/mm^2 at x = {r["x_um"][i]:.0f} um')


def prepare(args, quiet=False):
    """Resolve the run_simulation parameters and build the BPM for this case (also used
    by run_hybrid.py). Returns a dict with the solver, start field and geometry."""
    if args.dim != 2:
        args.dim = 2
    with contextlib.redirect_stdout(io.StringIO()):               # quiet the FDTD notes
        layers = rs.resolve_args(args)
    if args.geometry == 'cad':
        raise SystemExit('run_bpm.py supports --geometry lens / planar (layer stacks).')
    lam = float(args.lambda_nm)
    t_wg = float(layers['t_wg_um'])
    n_wg = complex(layers['wg_n']).real
    yb = B.stack_bottom_um(layers)
    curved = R_BEND_UM if (args.curved and args.geometry == 'lens') else None
    kappa = dict(getattr(args, 'kappa', {}) or {})
    eps = B.lens_eps_function(layers, lam, curved_R_um=curved, kappa=kappa)
    # domain as the FDTD: 10 um below the core (tear film + cornea), 5 um air above,
    # plus 2 um PML on each side
    y_lo, y_hi = -10.0 - 2.0, t_wg + 5.0 + 2.0
    y = B.graded_grid(y_lo, y_hi, [(yb - 0.5, 0.5), (t_wg - 0.5, t_wg + 0.5)],
                      args.bpm_dy_fine_nm/1000.0, args.bpm_dy_core_nm/1000.0)
    tilt = float(args.vcsel_tilt)
    th_core = math.asin(math.sin(math.radians(tilt))/n_wg)
    n_ref = args.bpm_n_ref if args.bpm_n_ref else n_wg*math.cos(th_core)
    pol = args.polarization
    bp = B.BPM2D(y, eps, lam, pol=pol, n_ref=n_ref, pml=(2.0, 2.0, 3.0), pade=args.bpm_pade)
    yc = t_wg/2.0 + float(args.vcsel_offset)
    # VCSEL beam as in the FDTD: defined in the source plane air_um before the facet,
    # through the air gap and the facet (Fresnel) -> power eta inside the core
    from common import vcsel as V
    dist = float(args.input_gap or 0.0)
    air = V.air_path_um(dist, float(args.resolution_nm)/1000.0)
    u0, eta = B.vcsel_facet_input(bp, yc, float(args.vcsel_waist), tilt, dist, air, n_wg,
                                  window=(0.0, t_wg))
    geo = ('lens ' + str(args.scenario) + (' curved R=8.3mm' if curved else ' flat')
           if args.geometry == 'lens' else 'planar')
    if not quiet:
        print('===================== BPM configuration =====================')
        print(f'  Geometry  : {geo}   core {t_wg:g} um n={n_wg:.4f}')
        from common.layer_stack import describe
        print(f'  Layers    : {describe(layers["stack"])}')
        print(f'  Beam      : waist {args.vcsel_waist:g} um, tilt {tilt:g} deg in air '
              f'({math.degrees(th_core):.2f} deg in the core), centre y={yc:g} um, pol {pol}')
        print(f'  Facet     : waist {dist:g} um before the facet, beam defined {air:g} um before it '
              f'(as the FDTD source); Fresnel -> coupled {100*eta:.2f} % '
              f'(reflection {100*(1 - eta):.2f} %)')
        print(f'  Grid      : {len(y)} nodes (y {y_lo:g}..{y_hi:g} um, fine {args.bpm_dy_fine_nm:g} nm, '
              f'core {args.bpm_dy_core_nm:g} nm), dx {args.bpm_dx_um:g} um, '
              f'Pade ({args.bpm_pade},{args.bpm_pade}), n_ref {n_ref:.4f}')
        print('==============================================================')
    regions = {'core': (0.0, t_wg), 'tear': (yb, 0.0), 'cornea': (y_lo + 2.0, yb),
               'air': (t_wg, y_hi - 2.0)}
    # absorption per layer: core, each tear-film layer, cornea (inside the domain)
    from common.layer_stack import stack_bounds
    absorb = {'core': (0.0, t_wg)}
    for nm, yt_, yb_, _n in stack_bounds(layers.get('stack') or [], 0.0):
        absorb[str(nm)] = (yb_, yt_)
    absorb['cornea'] = (y_lo + 2.0, yb)
    return dict(layers=layers, lam=lam, t_wg=t_wg, n_wg=n_wg, yb=yb, curved=curved, y=y,
                y_lo=y_lo, y_hi=y_hi, tilt=tilt, th_core=th_core, n_ref=n_ref, pol=pol, bp=bp,
                u0=u0, eta=eta, air_um=air, regions=regions, absorb_regions=absorb, kappa=kappa,
                probes=[-d for d in PROBE_DEPTHS_UM])


def print_absorption(r, L):
    """Absorbed power per layer (fraction of the laser power) and the tear film's loss
    rate in dB/mm (absorbed power per mm over the guided power)."""
    if 'A_cum' not in r:
        return
    x, P = r['x_um'], r['P_total']
    print(f'[ATR] absorbed by the end ({L:g} um), fraction of the laser power:')
    for k, v in r['A_cum'].items():
        if v[-1] > 0:
            print(f'[ATR]   {k:10s} {v[-1]:.4e}')
    tear = [k for k in r['A_rate'] if k not in ('core', 'cornea')]
    if not tear:
        return
    rate = sum(r['A_rate'][k] for k in tear)
    db = 10/math.log(10)*1e3*rate/np.maximum(P, 1e-300)
    A = sum(r['A_cum'][k][-1] for k in tear)
    i = int(np.argmax(db))
    print(f'[ATR]   tear film: max {db[i]:.3e} dB/mm at x = {x[i]:.0f} um, mean over the length '
          f'{10*math.log10(P[0]/max(P[0] - A, 1e-300))/(L/1e3):.3e} dB/mm')


def fdtd_start_field(args, c, x_start_um, margin_um=15.0):
    """Short 2D FDTD run of the coupling (same parameters, method full, no bead) and its
    complex CW field at x_start_um after the facet, as a BPM start field on c['y'].
    Returns (u, P): P = power at that plane as a fraction of the laser power = the FDTD's
    measured coupling through the facet times the ratio of the BPM power measure
    (TE |Ez|^2, TM eps|Ey|^2) at the start plane to that just behind the facet, so power
    radiated in between (thin cores) is not counted."""
    import copy
    import fdtd2d_core as d2
    a = copy.copy(args)
    a.dim, a.method, a._cad, a.no_bead, a.no_save = 2, 'full', None, True, True
    dx = float(a.resolution_nm)*1e-3
    # domain: source + air gap + facet + x_start + margin (core runs into the right PML)
    import time
    t0 = time.time()
    a.length_um = float(c['air_um']) + 2.0 + x_start_um + margin_um
    d2.RETURN_CW = True
    try:
        res = rs._run_2d(a, c['layers'])
    finally:
        d2.RETURN_CW = False
    E = res['E_cw']
    ix = int(round((res['x_facet_global_um'] + x_start_um)/dx))
    if ix >= E.shape[0] - 14:
        raise SystemExit('--bpm-start-fdtd-um: plane too close to the end of the FDTD domain')
    y_f = -d2.TEAR_BUFFER*1e6 + np.arange(E.shape[1])*dx
    col = np.conj(E[ix])                       # FDTD exp(-iwt)  -> BPM exp(+i(kx - wt))
    if c['pol'] == 'p':                        # FDTD Ey -> BPM Hz (inverse of run_hybrid)
        col = col*np.interp(y_f, c['y'], np.real(c['bp'].eps))/c['n_wg']**2
    y = c['y']
    u = (np.interp(y, y_f, col.real, left=0, right=0)
         + 1j*np.interp(y, y_f, col.imag, left=0, right=0))
    yf_in = (y_f > c['y_lo'] + 2.0) & (y_f < c['y_hi'] - 2.0)     # inside the BPM domain
    w = (np.interp(y_f[yf_in], c['y'], np.real(c['bp'].eps)) if c['pol'] == 'p'
         else np.ones(int(yf_in.sum())))
    S = lambda i: float((w*np.abs(E[i][yf_in])**2).sum())
    ix_f = int(round((res['x_facet_global_um'] + 0.2)/dx))        # just behind the facet
    P = float(res['coupling_efficiency'])*S(ix)/S(ix_f)
    from common.bpm import normalize_power
    u = normalize_power(c['bp'], u, P)
    eta = P
    print(f'  FDTD start: FDTD input-plane coupling {100*float(res.get("coupling_efficiency", np.nan)):.2f} %, '
          f'power at the start plane {100*P:.2f} %')
    print(f'  FDTD start: field at {x_start_um:g} um after the facet ({time.time() - t0:.0f} s FDTD); '
          f'start power (FDTD) {100*eta:.2f} % vs facet model coupling {100*c["eta"]:.2f} %')
    return u, eta


def save_npz(out, r, c, args, L, x0_um=0.0, zm=None):
    """Store a BPM result (propagate() output r, prepare() context c)."""
    np.savez(out, x_um=r['x_um'], y_um=c['y'], I=r['I'], eps=c['bp'].eps, P_total=r['P_total'],
             **{f'P_{k}': v for k, v in r['P_region'].items()},
             I_probe=r['I_probe'], probe_depths_um=np.asarray(PROBE_DEPTHS_UM),
             t_wg_um=c['t_wg'], stack_bottom_um=c['yb'], lam_nm=c['lam'], pol=c['pol'],
             n_ref=c['n_ref'], curved_R_um=(c['curved'] or 0.0), tilt_deg=c['tilt'],
             waist_um=float(args.vcsel_waist), offset_um=float(args.vcsel_offset),
             dx_um=args.bpm_dx_um, pade=args.bpm_pade, runtime_s=r['runtime_s'], x0_um=x0_um,
             coupling_eta=c['eta'], intensity='E', laser_power_mw=float(args.laser_power),
             **{f'A_cum_{k}': v for k, v in r.get('A_cum', {}).items()},
             **{f'A_rate_{k}': v for k, v in r.get('A_rate', {}).items()},
             absorb_layers=np.array(list(r.get('A_cum', {}).keys()), dtype=str),
             **({} if zm is None else dict(z_model=zm['model'], waist_z_um=zm['waist_z_um'],
                                           w_z_um=zm['w_z_um'],
                                           z_factor_per_um=zm['z_factor_per_um'])))
    print(f'  Saved: {out}')


def main(argv=None):
    args = build_parser().parse_args(argv)
    c = prepare(args)
    bp, u0, L = c['bp'], c['u0'], float(args.length_um)
    x0 = 0.0
    if args.bpm_start_fdtd_um is not None:
        x0 = float(args.bpm_start_fdtd_um)
        u0, c['eta'] = fdtd_start_field(args, c, x0)
    print(f'  Length    : {L:g} um ({int(round((L - x0)/args.bpm_dx_um))} steps)')
    r = bp.propagate(u0, L - x0, dx_um=args.bpm_dx_um, rec_dx_um=args.bpm_rec_um, x0_um=x0,
                     regions=c['regions'], probes_um=c['probes'], intensity='E',
                     absorb_regions=c['absorb_regions'])
    P0 = r['P_total'][0]
    core_end = r['P_region']['core'][-1]/P0
    print(f'[BPM] done in {r["runtime_s"]:.0f} s.  Coupled through the facet {c["eta"]:.4f}; '
          f'of that at the end: core {core_end:.4f}, total {r["P_total"][-1]/P0:.4f} '
          f'(rest left through cornea/air)')
    tear = r['P_region']['tear']/np.maximum(r['P_region']['core'], 1e-300)
    print(f'[BPM] tear film power / core power: max {tear.max():.3e} at x = '
          f'{r["x_um"][int(np.argmax(tear))]:.1f} um, mean {tear.mean():.3e}')
    print_absorption(r, L)
    zm = z_model(args, c, r['x_um'])
    print_absolute(args, c, r, zm)
    out = args.bpm_out
    if not out:
        os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
        tag = (f'bpm_{args.geometry}_{args.scenario if args.geometry == "lens" else ""}_'
               f'{"curved" if c["curved"] else "flat"}_{c["pol"]}_L{L:g}_tilt{c["tilt"]:g}'
               f'_off{args.vcsel_offset:g}')
        out = os.path.join(ROOT, 'results', tag.replace('__', '_') + '.npz')
    save_npz(out, r, c, args, L, x0_um=x0, zm=zm)
    return out


if __name__ == '__main__':
    main()
