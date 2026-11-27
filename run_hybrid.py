"""Hybrid BPM -> FDTD (2D): fast wide-angle BPM through the long lens, full FDTD only
near the sensor.

The BPM (common/bpm.py) carries the VCSEL beam from the coupling facet to the hand-off
plane x = --handoff-x-um. There its complex field E(y) becomes the source of a short
2D FDTD domain (planar_beads/fdtd2d_core.py, HANDOFF hook: one-sided Huygens source, no
entry facet, core running into the left PML) of length --fdtd-length-um. The BPM is
also continued through the FDTD region for comparison.

Same parameters as run_simulation.py / run_bpm.py, plus:

    python run_hybrid.py --geometry lens --curved --vcsel-tilt 20 --resolution-nm 30 ^
        --handoff-x-um 400 --fdtd-length-um 300

--fdtd-y-top-um crops the FDTD domain at the top (in the core): 'auto' (default) takes
the highest point the BPM beam reaches inside the FDTD region (1e-6 of the peak) plus
a margin; 'none' keeps the full lens thickness.

Output:
  results/hybrid_<case>.npz : BPM over 0..x_h+L (|E|^2 every --bpm-rec-um) and the FDTD
                              CW |E|^2 in lens coordinates, plus the power in core /
                              tear film along x for both, and the z spreading w_z(x)
                              (see run_bpm.py, --bpm-z-model).
  results/beads_*_handoff<x>_full.pkl/.npz : the usual FDTD result (analyzer), with x
                              counted from the FDTD domain start (source at ~0.5 um).
"""
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run_bpm                                                    # noqa: E402
import run_simulation as rs                                       # noqa: E402

Y_TOP_MARGIN_UM = 15.0


def build_parser():
    ap = run_bpm.build_parser()
    ap.prog = 'run_hybrid.py'
    g = ap.add_argument_group('Hybrid BPM -> FDTD (run_hybrid.py only)')
    g.add_argument('--handoff-x-um', type=float, required=True,
                   help='Hand-off plane: BPM from the facet to here, FDTD after it (um)')
    g.add_argument('--fdtd-length-um', type=float, default=300.0,
                   help='Length of the FDTD domain after the hand-off plane (um)')
    g.add_argument('--fdtd-y-top-um', default='auto',
                   help="Top of the FDTD domain in um ('auto' = BPM beam extent + "
                        f"{Y_TOP_MARGIN_UM:g} um, 'none' = full lens)")
    g.add_argument('--hybrid-out', default=None, help='Output file (default results/hybrid_*.npz)')
    return ap


def _region_sum(I, y, w, lo, hi):
    m = (y >= lo) & (y < hi)
    return (I[..., m]*w[m]).sum(-1)


def _auto_y_top(r, x_lo, x_hi, t_wg):
    """Highest y where the BPM |E|^2 inside [x_lo, x_hi] exceeds 1e-6 of its peak."""
    x, I, y = r['x_um'], r['I'], r['y_um']
    sel = (x >= x_lo) & (x <= x_hi)
    prof = I[sel].max(0)
    above = np.nonzero(prof > 1e-6*prof.max())[0]
    top = float(y[above[-1]]) + Y_TOP_MARGIN_UM
    return None if top >= t_wg - 1.0 else top


def main(argv=None):
    args = build_parser().parse_args(argv)
    x_h, L_f = float(args.handoff_x_um), float(args.fdtd_length_um)
    if x_h <= 0 or L_f <= 0:
        raise SystemExit('--handoff-x-um and --fdtd-length-um must be > 0')
    args.length_um = x_h + L_f
    c = run_bpm.prepare(args)
    bp, y, pol = c['bp'], c['y'], c['pol']
    if args.geometry != 'lens' and args.geometry != 'planar':
        raise SystemExit('run_hybrid.py supports --geometry lens / planar.')
    print(f'  Hybrid    : BPM 0 .. {x_h:g} um, FDTD {x_h:g} .. {x_h + L_f:g} um '
          f'(BPM continued for comparison)')

    # ---- 1. BPM through everything, complex field kept at the hand-off plane
    r = bp.propagate(c['u0'], x_h + L_f, dx_um=args.bpm_dx_um, rec_dx_um=args.bpm_rec_um,
                     regions=c['regions'], probes_um=c['probes'], field_planes_um=(x_h,),
                     intensity='E', absorb_regions=c['absorb_regions'])
    u_h = r['fields'][x_h]
    P_h = float(bp.power_density(u_h)[~bp.pml_mask].sum())
    print(f'[BPM] {r["runtime_s"]:.0f} s; coupled through the facet {c["eta"]:.4f}, power at '
          f'the hand-off plane {P_h:.4f} (laser = 1)')

    # ---- 2. FDTD source from the BPM field
    yt = str(args.fdtd_y_top_um).strip().lower()
    y_top = (None if yt == 'none' else
             _auto_y_top(r, x_h, x_h + L_f, c['t_wg']) if yt == 'auto' else float(yt))
    u_src = u_h/math.sqrt(P_h)                       # forward power 1 in BPM units
    if pol == 'p':                                   # BPM carries Hz; FDTD source drives Ey
        u_src = u_src*(c['n_wg']**2/np.real(bp.eps))   # Ey ~ kx Hz/(w eps0 eps)
    import fdtd2d_core as d2
    d2.HANDOFF = dict(y_um=y, u=u_src, n_src=c['n_wg'], tilt_core_deg=math.degrees(c['th_core']),
                      x0_um=x_h, y_top_um=y_top)
    print(f'  FDTD      : {L_f:g} um at dx {args.resolution_nm:g} nm, y '
          f'{-d2.TEAR_BUFFER*1e6:g} .. {y_top if y_top is not None else c["t_wg"] + 5:g} um'
          + ('' if y_top is None else ' (cropped above the beam)'))
    args.length_um = L_f
    args.method = 'full'
    args.input_gap = 0.0
    args._cad = None
    t0 = time.time()
    try:
        res = rs._run_2d(args, c['layers'])
    finally:
        d2.HANDOFF = None
    t_fdtd = time.time() - t0

    # ---- 3. Compare in lens coordinates
    E = res['E_cw']
    if E is None:
        raise SystemExit('FDTD returned no CW field (pulse source?)')
    I_f = np.abs(E)**2
    if pol == 'p' and res.get('Ex_cw_amp') is not None:
        I_f = I_f + np.abs(res['Ex_cw_amp'])**2
    dx = float(args.resolution_nm)*1e-3
    Nx, Ny = I_f.shape
    y_f = -d2.TEAR_BUFFER*1e6 + np.arange(Ny)*dx
    x_src = res['handoff']['src_x_um']               # source column = lens position x_h
    x_f = x_h + np.arange(Nx)*dx - x_src
    keep = (x_f >= x_h + 2*dx) & (x_f <= x_h + L_f - x_src - 12*dx)   # after the source pair,
                                                                      # in front of the right PML
    x_f, I_f = x_f[keep], I_f[keep]
    w_f = np.full(Ny, dx)
    yb, t_wg = c['yb'], c['t_wg']
    Pc_f = _region_sum(I_f, y_f, w_f, 0.0, t_wg)
    Pt_f = _region_sum(I_f, y_f, w_f, yb, 0.0)
    w_b = bp.w_real
    Pc_b = _region_sum(r['I'], y, w_b, 0.0, t_wg)
    Pt_b = _region_sum(r['I'], y, w_b, yb, 0.0)
    xb = r['x_um']
    in_f = (xb >= x_f[0]) & (xb <= x_f[-1])
    tf, tb = Pt_f/np.maximum(Pc_f, 1e-300), Pt_b[in_f]/np.maximum(Pc_b[in_f], 1e-300)
    print('\n================ Hybrid result ================')
    print(f'  BPM {r["runtime_s"]:.0f} s + FDTD {t_fdtd:.0f} s')
    print(f'  Tear film / core (|E|^2 integrals) in the FDTD region x = {x_f[0]:.0f}..{x_f[-1]:.0f} um:')
    print(f'    FDTD : max {tf.max():.3e} at x = {x_f[int(np.argmax(tf))]:.1f} um')
    print(f'    BPM  : max {tb.max():.3e} at x = {xb[in_f][int(np.argmax(tb))]:.1f} um')

    zb = run_bpm.z_model(args, c, xb); zf = run_bpm.z_model(args, c, x_f)
    run_bpm.print_absorption(r, x_h + L_f)
    run_bpm.print_absolute(args, c, r, zb)
    sx = max(1, int(round(0.2/dx)))                  # store the FDTD map every ~0.2 um
    out = args.hybrid_out
    if not out:
        os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
        tag = (f'hybrid_{args.geometry}_{"curved" if c["curved"] else "flat"}_{pol}'
               f'_x{x_h:g}_L{L_f:g}_tilt{c["tilt"]:g}_off{args.vcsel_offset:g}_dx{args.resolution_nm:g}')
        out = os.path.join(ROOT, 'results', tag + '.npz')
    np.savez_compressed(
        out, x_handoff_um=x_h, fdtd_length_um=L_f, y_top_um=(y_top if y_top is not None else np.nan),
        bpm_x_um=xb, bpm_y_um=y, bpm_I=r['I'], bpm_P_core=Pc_b, bpm_P_tear=Pt_b,
        bpm_I_probe=r['I_probe'], probe_depths_um=np.asarray(run_bpm.PROBE_DEPTHS_UM),
        fdtd_x_um=x_f[::sx], fdtd_y_um=y_f, fdtd_I=I_f[::sx].astype(np.float32),
        fdtd_P_core=Pc_f, fdtd_P_tear=Pt_f, fdtd_x_full_um=x_f,
        t_wg_um=t_wg, stack_bottom_um=yb, lam_nm=c['lam'], pol=pol, tilt_deg=c['tilt'],
        curved_R_um=(c['curved'] or 0.0), dx_nm=float(args.resolution_nm),
        bpm_runtime_s=r['runtime_s'], fdtd_runtime_s=t_fdtd, coupling_eta=c['eta'],
        laser_power_mw=float(args.laser_power), z_model=zb['model'], waist_z_um=zb['waist_z_um'],
        bpm_w_z_um=zb['w_z_um'], bpm_z_factor_per_um=zb['z_factor_per_um'],
        fdtd_w_z_um=zf['w_z_um'], fdtd_z_factor_per_um=zf['z_factor_per_um'],
        **{f'bpm_A_cum_{k}': v for k, v in r.get('A_cum', {}).items()},
        **{f'bpm_A_rate_{k}': v for k, v in r.get('A_rate', {}).items()},
        kappa=str(c['kappa']))
    print(f'  Saved: {out}')
    return out


if __name__ == '__main__':
    main()
