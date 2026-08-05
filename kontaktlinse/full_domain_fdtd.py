"""Full-Domain FDTD fuer 14mm Kontaktlinse (Flattened Coordinates).

In Flattened-Coords (y_rel = y_global - y_mid(x)) wird die gekruemmte Linse
zu einem geraden 14mm Planar-Waveguide. Das ermoeglicht eine durchgaengige
FDTD-Simulation der gesamten Linse in einem Lauf - ohne Sliding-Window.

VOR-/NACHTEILE:
  + Volle Maxwell-Physik inklusive ALLER Rueckreflexionen
  + Steady-State ueber gesamte 14mm
  + Direkter Vergleich zur sliding-Implementierung
  - Kruemmungs-Effekte vereinfacht (gut bei T_LENS/R = 250um/8300um = 3%)
  - Hoeherer VRAM-Bedarf (~1.9GB bei 300nm/600um)

VERWENDUNG:
  python full_domain_fdtd.py --scenario Gesund --gpu
  python full_domain_fdtd.py --scenario Custom --t-aqueous 5.0 --gpu
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np

# Repo-Root auf den Importpfad, damit das geteilte common/-Paket gefunden wird,
# unabhaengig davon, aus welchem Ordner das Skript gestartet wurde.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# Backend (GPU/CPU) und Fundamentalkonstanten aus den geteilten Modulen.
# Hinweis: EPS0/MU0 kommen jetzt in voller Genauigkeit aus common.physics
# (zuvor lokal auf ~8 Stellen gerundet) -> numerisch nur ~1e-8 relativ anders.
from common.backend import xp, cp, GPU_AVAILABLE, to_np   # noqa: E402
from common.physics import C0, EPS0, MU0                  # noqa: E402

LAM = 850e-9   # Default-Wellenlaenge dieses Legacy-Solvers

# Geometrie (kann via CLI ueberschrieben werden)
T_LENS = 250e-6
R_BEND = 8.3e-3
D_LENS = 14e-3
D1_S_CENTER = 2.0e-3
D3_S_CENTER = 1.8e-3
D1_H = 50e-6
D3_LEN = 50e-6
D3_QUERSCHN = 200e-6

# Brechungsindizes
N_PMMA = 1.491
N_LIPID = 1.480
N_MUCIN = 1.342
N_CORNEA = 1.376

# Tear-Szenarien: (lipid_m, aqueous_m, mucin_m, n_aqueous)
SCENARIOS = {
    'Gesund':       (0.030e-6, 3.5e-6, 0.5e-6, 1.336),
    'DED':          (0.010e-6, 1.5e-6, 0.3e-6, 1.336),
    'Frisch':       (0.050e-6, 6.0e-6, 1.0e-6, 1.336),
    'Hyperosmolar': (0.020e-6, 2.0e-6, 0.4e-6, 1.340),
    'MGD':          (0.005e-6, 3.5e-6, 0.5e-6, 1.336),
    'Mucin-Mangel': (0.030e-6, 3.5e-6, 0.05e-6, 1.336),
    'Mucin-reich':  (0.030e-6, 3.5e-6, 2.0e-6, 1.336),
    'Lipid-reich':  (0.100e-6, 3.5e-6, 0.5e-6, 1.336),
}


def estimate_vram_gb(Nx, Ny, n_arrays=5):
    """Speicher-Schaetzung: 5 float32-Arrays in Groesse Nx x Ny."""
    cells = Nx * Ny
    return cells * 4 * n_arrays / 1e9


def build_flat_materials(Nx, Ny, dx, y_offset_um, layers, n_aqueous,
                          n_pmma, n_lipid, n_mucin, n_cornea, t_lens_m,
                          curved=False):
    """Flattened material map: lens bei y_rel ∈ [-T/2, +T/2], Tear darunter, Air drueber.
    y_offset_um: y_rel bei iy=0 (in um)

    curved=True: echte Kruemmung via KONFORMER ABBILDUNG des gebogenen Wellenleiters
    auf einen geraden mit aequivalentem Index n_eq(u) = n(u)*(1 + u/R). u = y_rel ist
    der radiale Versatz von der Linsenmitte (Radius R_BEND); u>0 = aussen (Luftseite,
    groesserer Radius). eps_eq = eps*(1 + y_rel/R)^2. Erfasst Modenversatz nach aussen
    + Biegeverlust (1. Ordnung) - ohne Sliding-Window (ganze Linse in einem Lauf).
    """
    eps_r = np.ones((Nx, Ny), dtype=np.float32)
    sig = np.zeros((Nx, Ny), dtype=np.float32)
    dx_um = dx*1e6
    t_lens_um = t_lens_m*1e6
    t_lip_um = layers[0]*1e6
    t_aq_um = layers[1]*1e6
    t_mu_um = layers[2]*1e6
    # y_rel pro iy (vektorisiert, viel schneller als Schleife)
    y_um_arr = y_offset_um + np.arange(Ny)*dx_um
    # Layer-Masken (1D in y, broadcast auf 2D)
    mask_pmma = (y_um_arr >= -t_lens_um/2) & (y_um_arr <= t_lens_um/2)
    mask_lipid = (y_um_arr >= -t_lens_um/2 - t_lip_um) & (y_um_arr < -t_lens_um/2)
    mask_aq = (y_um_arr >= -t_lens_um/2 - t_lip_um - t_aq_um) & \
              (y_um_arr < -t_lens_um/2 - t_lip_um)
    mask_mu = (y_um_arr >= -t_lens_um/2 - t_lip_um - t_aq_um - t_mu_um) & \
              (y_um_arr < -t_lens_um/2 - t_lip_um - t_aq_um)
    mask_cornea = y_um_arr < -t_lens_um/2 - t_lip_um - t_aq_um - t_mu_um
    # Anwenden (broadcast)
    eps_r[:, mask_pmma] = n_pmma**2
    eps_r[:, mask_lipid] = n_lipid**2
    eps_r[:, mask_aq] = n_aqueous**2
    eps_r[:, mask_mu] = n_mucin**2
    eps_r[:, mask_cornea] = n_cornea**2
    if curved:
        # konforme Abbildung: eps_eq = eps * (1 + y_rel/R)^2  (R = R_BEND, y_rel in um)
        metric = (1.0 + y_um_arr/(R_BEND*1e6)).astype(np.float32)**2
        eps_r = eps_r*metric[None, :]
    return eps_r, sig


def run_full_domain_lens(scenario_name, dx_nm=300, window_h_um=600.0,
                         save_frames=True, n_frames=20, save_stride=4,
                         vcsel_waist=2e-6, vcsel_tilt=0.0,
                         vcsel_mode='single', vcsel_offset=0.0,
                         source_type='cw', curved=False):
    t_total = time.time()
    layers = SCENARIOS[scenario_name]
    print(f'\n========== Full-Domain-FDTD (Flattened): {scenario_name} ==========')
    print(f'Layers: lipid={layers[0]*1e6:.3f}um, aq={layers[1]*1e6:.2f}um, '
          f'mu={layers[2]*1e6:.2f}um, n_aq={layers[3]}')

    dx = dx_nm*1e-9
    dt = 0.5*dx/(C0*np.sqrt(2))
    print(f'Grid: dx={dx_nm} nm, dt={dt*1e15:.4f} fs')

    L_x = D_LENS  # 14 mm
    Nx = int(round(L_x / dx))
    Ny = int(round(window_h_um*1e-6 / dx))
    y_offset_um = -window_h_um/2

    vram_gb = estimate_vram_gb(Nx, Ny)
    print(f'Grid: {Nx} x {Ny} = {Nx*Ny/1e6:.1f} M cells, '
          f'estimated VRAM ~{vram_gb:.2f} GB')

    # Material map
    print('Building flattened material map ...', end=' ', flush=True)
    t_mat = time.time()
    eps_r_np, sig_np = build_flat_materials(
        Nx, Ny, dx, y_offset_um, layers, layers[3],
        N_PMMA, N_LIPID, N_MUCIN, N_CORNEA, T_LENS, curved=curved)
    # Austritts-Facet: Linse->Luft am linken Rand (Fresnel-Reflex), Mur dahinter
    _ig = int(round(40e-6/dx))
    if 0 < _ig < Nx:
        eps_r_np[:_ig, :] = 1.0
    eps_r = xp.asarray(eps_r_np)
    sig = xp.asarray(sig_np)
    del eps_r_np, sig_np
    print(f'{time.time()-t_mat:.1f}s')

    Ce_n = (1 - sig*dt/(2*EPS0*eps_r))
    Ce_d = (1 + sig*dt/(2*EPS0*eps_r))
    Ce_E = (Ce_n/Ce_d).astype(xp.float32)
    Ce_H = (dt/(EPS0*eps_r*dx)/Ce_d).astype(xp.float32)
    Ch = xp.float32(dt/(MU0*dx))
    del eps_r, sig, Ce_n, Ce_d

    Ez = xp.zeros((Nx, Ny), dtype=xp.float32)
    Hx = xp.zeros((Nx, Ny-1), dtype=xp.float32)
    Hy = xp.zeros((Nx-1, Ny), dtype=xp.float32)

    # VCSEL bei rechtem Linsen-Ende (x = +D/2 - 5um)
    x_src_um = D_LENS*1e6/2 - 5
    x_lens_left_um = -D_LENS*1e6/2
    src_ix = int(round((x_src_um - x_lens_left_um)/(dx*1e6)))
    src_ix = max(5, min(Nx - 5, src_ix))
    iy_lens_center = int(round((-y_offset_um + vcsel_offset*1e6)/(dx*1e6)))
    iy_lens_center = max(5, min(Ny - 5, iy_lens_center))
    waist_pixels = max(2, int(round(vcsel_waist/dx)))

    ys_local = xp.arange(Ny, dtype=xp.float32)
    src_p = xp.exp(-((ys_local - iy_lens_center)/waist_pixels)**2)
    src_p *= np.sqrt(1 - 0.162)  # Fresnel @ PMMA/GaAs

    f0 = C0/LAM
    sigma_t = 4/(2*np.pi*f0)
    tilt_rad = np.radians(vcsel_tilt)
    k0 = 2*np.pi/LAM
    y_pix_offset_m = (ys_local - iy_lens_center)*dx
    phase_y = float(np.sin(tilt_rad))*k0*y_pix_offset_m

    print(f'Source: ix={src_ix} (x={x_src_um:.0f}um), iy_lens={iy_lens_center}, '
          f'waist={vcsel_waist*1e6}um mode={vcsel_mode} tilt={vcsel_tilt}deg '
          f'offset={vcsel_offset*1e6}um type={source_type}')

    mur = xp.float32((C0*dt - dx)/(C0*dt + dx))

    steps_total = int(2 * D_LENS * N_PMMA / C0 / dt)
    snap_every = max(1, steps_total // n_frames) if save_frames else 999999
    print(f'Steps: {steps_total} ({steps_total*dt*1e12:.1f} ps total), '
          f'Snapshots: every {snap_every} (target {n_frames})')

    # Detektor-Indizes in Flat-Coords
    x_d1_um = (D1_S_CENTER - D_LENS/2)*1e6
    x_d3_um = (D3_S_CENTER - D_LENS/2)*1e6
    ix_d1 = int(round((x_d1_um - x_lens_left_um)/(dx*1e6)))
    ix_d3 = int(round((x_d3_um - x_lens_left_um)/(dx*1e6)))
    y_d1_rel_um = -T_LENS*1e6/2 + 25
    y_d2_rel_um = +T_LENS*1e6/2 - 25
    iy_d1 = int(round((y_d1_rel_um - y_offset_um)/(dx*1e6)))
    iy_d2 = int(round((y_d2_rel_um - y_offset_um)/(dx*1e6)))
    iy_d3 = int(round((0 - y_offset_um)/(dx*1e6)))
    ix_w = max(1, int(round(D1_H/2/dx)))
    iy_h = max(1, int(round(50e-6/2/dx)))
    iy_h_d3 = max(1, int(round(D3_QUERSCHN/2/dx)))

    P_D1 = 0.0; P_D2 = 0.0; P_D3 = 0.0
    # Snapshots (um save_stride reduziert) direkt auf Disk-Memmap streamen.
    frames = []
    ez_mm = None; mm_path = None; k_fr = 0
    if save_frames:
        _sx = (Nx + save_stride - 1)//save_stride
        _sy = (Ny + save_stride - 1)//save_stride
        os.makedirs('results', exist_ok=True)
        mm_path = os.path.join('results', f'_stream_full_{scenario_name}.npy')
        ez_mm = np.lib.format.open_memmap(mm_path, mode='w+', dtype=np.float32,
                                          shape=(max(1, n_frames + 2), _sx, _sy))
    t_start = time.time()
    for n in range(steps_total):
        Hx -= Ch*(Ez[:, 1:] - Ez[:, :-1])
        Hy += Ch*(Ez[1:, :] - Ez[:-1, :])
        ex1, ex2 = Ez[1, :].copy(), Ez[-2, :].copy()
        ey1, ey2 = Ez[:, 1].copy(), Ez[:, -2].copy()
        Ez[1:-1, 1:-1] = (Ce_E[1:-1, 1:-1]*Ez[1:-1, 1:-1]
                          + Ce_H[1:-1, 1:-1]*((Hy[1:, 1:-1] - Hy[:-1, 1:-1])
                                              - (Hx[1:-1, 1:] - Hx[1:-1, :-1])))
        t_phys = n*dt
        if source_type == 'pulse':
            sigma_p = 30 / f0
            t_center = 3*sigma_p
            envelope = float(np.exp(-((t_phys - t_center)/sigma_p)**2))
        else:
            envelope = float(1 - np.exp(-((t_phys/(2*sigma_t))**2)))
        omega_t = -2*np.pi*f0*t_phys
        if vcsel_tilt != 0.0:
            src_wave = xp.sin(omega_t + phase_y)
            Ez[src_ix, :] += envelope * src_wave * src_p
        else:
            amp = envelope * float(np.sin(omega_t))
            Ez[src_ix, :] += amp * src_p

        # Mur 1. Ordnung mit Ez^n der Innen-Nachbarn (ex1/ey1, vor E-Update
        # kopiert) - frueher wurde ein um 1 Step veralteter Wert benutzt.
        Ez[0, :]  = ex1 + mur*(Ez[1, :]  - Ez[0, :])
        Ez[-1, :] = ex2 + mur*(Ez[-2, :] - Ez[-1, :])
        Ez[:, 0]  = ey1 + mur*(Ez[:, 1]  - Ez[:, 0])
        Ez[:, -1] = ey2 + mur*(Ez[:, -2] - Ez[:, -1])

        # Detektor-Akkumulation
        for (ix_c, iy_c, ih, name) in [(ix_d1, iy_d1, iy_h, 'D1'),
                                        (ix_d1, iy_d2, iy_h, 'D2'),
                                        (ix_d3, iy_d3, iy_h_d3, 'D3')]:
            ix_lo = max(0, ix_c - ix_w); ix_hi = min(Nx, ix_c + ix_w)
            iy_lo = max(0, iy_c - ih); iy_hi = min(Ny, iy_c + ih)
            if ix_hi > ix_lo and iy_hi > iy_lo:
                box = Ez[ix_lo:ix_hi, iy_lo:iy_hi]
                p = float(xp.sum(box*box))*dx*dx
                if name == 'D1': P_D1 += p
                elif name == 'D2': P_D2 += p
                else: P_D3 += p

        if (save_frames and ez_mm is not None and ((n+1) % snap_every == 0)
                and k_fr < ez_mm.shape[0]):
            ez_np = to_np(Ez)
            ez_mm[k_fr] = ez_np[::save_stride, ::save_stride].astype(np.float32)
            frames.append({
                'x_start': x_lens_left_um,
                'x_end': x_lens_left_um + Nx*dx*1e6,
                'y_start': y_offset_um,
                'y_end': y_offset_um + Ny*dx*1e6,
                'slide_i': 0,
                'step': n+1,
                'time': time.time() - t_start,
            })
            k_fr += 1
            del ez_np
            print(f'  Step {n+1}/{steps_total} ({100*(n+1)/steps_total:.0f}%)  '
                  f't_sim={t_phys*1e12:.1f}ps  t_wall={time.time()-t_start:.0f}s  '
                  f'max|Ez|={float(xp.max(xp.abs(Ez))):.3e}')

    if ez_mm is not None:
        ez_mm.flush(); del ez_mm
    total_time = time.time() - t_total
    R_tear = P_D1/max(P_D2, 1e-30)
    print(f'\n[Result] {scenario_name}: total {total_time:.0f}s')
    print(f'  P_D1 = {P_D1:.3e}')
    print(f'  P_D2 = {P_D2:.3e}')
    print(f'  P_D3 = {P_D3:.3e}')
    print(f'  R_tear = P_D1/P_D2 = {R_tear:.4f}')

    return dict(scenario=scenario_name, P_D1=P_D1, P_D2=P_D2, P_D3=P_D3,
                R_tear=R_tear, frames=frames, frames_memmap=mm_path,
                total_time=total_time,
                layers=layers, flattened=True, dx=dx, method='full_domain')


def save_results(result, out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    scenario = result['scenario']
    suffix = '_full'
    summary = {k: v for k, v in result.items() if k != 'frames'}
    with open(f'{out_dir}/sliding_{scenario}{suffix}.pkl', 'wb') as f:
        pickle.dump(summary, f)
    print(f'  Saved: {out_dir}/sliding_{scenario}{suffix}.pkl')
    if result.get('frames'):
        npz_path = f'{out_dir}/sliding_{scenario}{suffix}_frames.npz'
        try:
            import cupy as _cp
            _cp.get_default_memory_pool().free_all_blocks()
            _cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        _frs = result['frames']
        mm_path = result.get('frames_memmap')
        ez_stream = None
        ezs = None
        _mm_full = None
        if mm_path and os.path.exists(mm_path):
            _mm_full = np.load(mm_path, mmap_mode='r')
            ez_stream = _mm_full[:len(_frs)]
            import shutil as _sh
            free_disk = _sh.disk_usage(out_dir or '.').free
            item = int(np.prod(ez_stream.shape[1:]))*4
            _ex = 1
            for s in (1, 2, 4, 8):
                if len(_frs)*item/(s*s)*0.5 <= free_disk/8*0.3:
                    _ex = s
                    break
            ezs = ez_stream[:, ::_ex, ::_ex]      # strided View -> streamt beim Speichern
        else:
            for _ex in (1, 2, 4, 8):
                try:
                    if _ex == 1:
                        ezs = np.stack([f['Ez'] for f in _frs], axis=0)
                    else:
                        ezs = np.stack([f['Ez'][::_ex, ::_ex] for f in _frs], axis=0)
                    break
                except MemoryError:
                    ezs = None
                    print(f'  [WARN] Stack zu gross fuer RAM, zusaetzlicher Stride {_ex*2}...')
            if ezs is None:
                print('  [FEHLER] Frames passen nicht in den RAM - NPZ nicht gespeichert.')
                return
        meta = np.array([(fr['x_start'], fr['x_end'], fr['y_start'],
                          fr['y_end'], fr['slide_i'], fr['time'],
                          fr.get('step', 0))
                         for fr in result['frames']], dtype=np.float64)
        layers_arr = np.array(result.get('layers',
                              (0.030e-6, 3.5e-6, 0.5e-6, 1.336)), dtype=np.float64)
        mat_names = result.get('material_names',
                               ['PMMA', 'Lipid', 'Aqueous', 'Mucin', 'Cornea'])
        mat_indices = np.array(result.get('material_indices',
                                          [N_PMMA, N_LIPID, 1.336, N_MUCIN, N_CORNEA]),
                               dtype=np.float64)
        np.savez_compressed(npz_path, Ez=ezs,
                            meta=meta, scenario=scenario, layers=layers_arr,
                            flattened=np.array([1.0]),
                            material_names=np.array(mat_names),
                            material_indices=mat_indices)
        print(f'  Saved: {npz_path} ({ezs.nbytes/1e6:.1f} MB raw)')
        # Temporaeres Stream-Memmap aufraeumen (Windows: Handle schliessen).
        try:
            del ezs
        except Exception:
            pass
        if mm_path and os.path.exists(mm_path):
            try:
                _mm_full._mmap.close()
            except Exception:
                pass
            try:
                del _mm_full, ez_stream
            except Exception:
                pass
            import gc
            gc.collect()
            try:
                os.remove(mm_path)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', default='Gesund')
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--curved', action='store_true',
                    help='echte Kruemmung via konformer Abbildung (eps*(1+y_rel/R)^2); '
                         'ohne Flag: flattened (Kruemmung vernachlaessigt)')
    ap.add_argument('--resolution', type=float, default=300)
    ap.add_argument('--window-h', type=float, default=600.0)
    ap.add_argument('--lambda-nm', type=float, default=850.0)
    ap.add_argument('--t-lens', type=float, default=250.0)
    ap.add_argument('--n-pmma', type=float, default=1.491)
    ap.add_argument('--n-lipid', type=float, default=1.480)
    ap.add_argument('--n-mucin', type=float, default=1.342)
    ap.add_argument('--n-cornea', type=float, default=1.376)
    ap.add_argument('--t-lipid', type=float, default=0.030)
    ap.add_argument('--t-aqueous', type=float, default=3.5)
    ap.add_argument('--t-mucin', type=float, default=0.5)
    ap.add_argument('--n-aqueous', type=float, default=1.336)
    ap.add_argument('--vcsel-waist', type=float, default=2.0)
    ap.add_argument('--vcsel-tilt', type=float, default=0.0)
    ap.add_argument('--vcsel-offset', type=float, default=0.0)
    ap.add_argument('--source-type', default='cw', choices=['cw', 'pulse'])
    ap.add_argument('--lens-radius', type=float, default=8.3)
    ap.add_argument('--lens-diameter', type=float, default=14.0)
    ap.add_argument('--d1-position', type=float, default=2.0)
    ap.add_argument('--d3-position', type=float, default=1.8)
    ap.add_argument('--d1-length', type=float, default=500.0)
    ap.add_argument('--d3-length', type=float, default=50.0)
    ap.add_argument('--no-frames', action='store_true')
    ap.add_argument('--vram-limit', type=float, default=4.0,
                    help='VRAM-Limit in GB (RTX A500 hat 4)')
    # Material-Namen
    ap.add_argument('--name-lens', default='PMMA')
    ap.add_argument('--name-lipid', default='Lipid')
    ap.add_argument('--name-aqueous', default='Aqueous')
    ap.add_argument('--name-mucin', default='Mucin')
    ap.add_argument('--name-cornea', default='Cornea')
    args = ap.parse_args()

    global N_PMMA, N_LIPID, N_MUCIN, N_CORNEA, T_LENS, LAM
    global R_BEND, D_LENS, D1_S_CENTER, D3_S_CENTER, D1_H, D3_LEN
    N_PMMA = args.n_pmma
    N_LIPID = args.n_lipid
    N_MUCIN = args.n_mucin
    N_CORNEA = args.n_cornea
    T_LENS = args.t_lens*1e-6
    LAM = args.lambda_nm*1e-9
    R_BEND = args.lens_radius*1e-3
    D_LENS = args.lens_diameter*1e-3
    D1_S_CENTER = args.d1_position*1e-3
    D3_S_CENTER = args.d3_position*1e-3
    D1_H = args.d1_length*1e-6
    D3_LEN = args.d3_length*1e-6

    if args.scenario == 'Custom':
        SCENARIOS['Custom'] = (args.t_lipid*1e-6, args.t_aqueous*1e-6,
                               args.t_mucin*1e-6, args.n_aqueous)
    elif args.scenario not in SCENARIOS:
        print(f'ERROR: scenario "{args.scenario}" unknown.')
        print(f'Bekannte: {list(SCENARIOS.keys())}')
        return 1

    # Pre-flight Memory-Check
    dx = args.resolution*1e-9
    Nx_est = int(round(D_LENS / dx))
    Ny_est = int(round(args.window_h*1e-6 / dx))
    vram_gb = estimate_vram_gb(Nx_est, Ny_est)
    print(f'\n[Memory] Grid: {Nx_est} x {Ny_est} = {Nx_est*Ny_est/1e6:.1f} M cells')
    print(f'[Memory] Estimated VRAM: ~{vram_gb:.2f} GB (5 float32-Arrays)')
    print(f'[Memory] Verfuegbar (deklariert): {args.vram_limit:.1f} GB')
    if vram_gb > args.vram_limit:
        print(f'[Memory] !!! WARNUNG: ~{vram_gb:.1f} GB > {args.vram_limit:.1f} GB !!!')
        # Empfehlungen
        new_res = args.resolution * np.sqrt(vram_gb/args.vram_limit)
        new_wh = args.window_h * args.vram_limit/vram_gb
        print(f'[Memory] Empfehlungen:')
        print(f'[Memory]   --resolution {new_res:.0f}  (statt {args.resolution:.0f})')
        print(f'[Memory]   --window-h {new_wh:.0f}  (statt {args.window_h:.0f})')
        try:
            ans = input('Trotzdem starten? (y/N): ').strip().lower()
        except EOFError:
            ans = 'n'
        if ans != 'y':
            print('Abgebrochen.')
            return 1

    result = run_full_domain_lens(
        scenario_name=args.scenario,
        dx_nm=args.resolution,
        window_h_um=args.window_h,
        save_frames=not args.no_frames,
        vcsel_waist=args.vcsel_waist*1e-6,
        vcsel_tilt=args.vcsel_tilt,
        vcsel_offset=args.vcsel_offset*1e-6,
        source_type=args.source_type, curved=args.curved)
    result['material_names'] = [args.name_lens, args.name_lipid,
                                 args.name_aqueous, args.name_mucin,
                                 args.name_cornea]
    result['material_indices'] = [args.n_pmma, args.n_lipid, args.n_aqueous,
                                   args.n_mucin, args.n_cornea]
    save_results(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
