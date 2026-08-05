"""Sliding-Window-FDTD fuer 14 mm Kontaktlinsen-Sensor (GPU oder CPU).

Features:
  - PMMA-Lens (n=1.491), R=8.3mm, D=14mm, t=250um
  - 3-Schicht-Tear (Lipid + Aqueous + Mucin) + Cornea
  - VCSEL an ORTHOGONALER Stirnflaeche (senkrecht zur Lens-Tangente bei x=+7000um)
  - D1 (Tear-Side) + D2 (Air-Side) bei x=-5000um, entlang Lens-Normalen versetzt
  - D3 (End-Fire, tangential rotiert) bei x=-5200um
  - 8 Snapshots pro Slide (waehrend Wellenpropagation, nicht am Ende)
  - GIF mit Schicht-Outlines + Legende + aspect='equal'
  - Frames werden separat als NPZ gespeichert (fuer rebuild_gif.py)

CO-MOVING WINDOW (seit Update):
  - Zwischen Slides werden Ez/Hx/Hy um N_shift = slide_um/dx Zellen geshiftet
    (xp.roll), Eingangsrand mit 0 gefuellt.
  - Effekt: die Welle behaelt ihre globale Lab-Frame-Position ueber alle Slides
    hinweg. Das Sliding-Window-Resultat ist damit aequivalent zu einem Full-
    Domain-FDTD ueber die gesamte 14mm-Linse (modulo Rueckreflexionen, die
    aus dem rechten Window-Rand verschwinden).
  - Konsistenz-Bedingung: slide_um < window_w_um (besser <= window/2),
    damit die ueberlappende Region die Welleninfo zwischen Slides traegt.
"""
import argparse
import os
import time
import pickle
import numpy as np

# ---------- GPU/CPU Backend ----------
try:
    import cupy as cp
    xp = cp
    GPU_AVAILABLE = True
    print('[Backend] CuPy detected - using NVIDIA GPU')
except ImportError:
    xp = np
    GPU_AVAILABLE = False
    print('[Backend] CuPy not available - fallback to NumPy (CPU)')


def to_xp(arr):
    return xp.asarray(arr) if GPU_AVAILABLE else arr


def to_np(arr):
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


# ---------- Physikalische Parameter ----------
LAM   = 850e-9
C0    = 2.99792458e8
EPS0  = 8.8541878128e-12
MU0   = 4.0*np.pi*1e-7

N_PMMA   = 1.491
N_AIR    = 1.000
N_LIPID  = 1.480
N_MUCIN  = 1.342
N_CORNEA = 1.376
N_BUFFER = 1.42
N_INGAAS_R = 3.5
N_INGAAS_K = 0.05
EPS_INGAAS = N_INGAAS_R**2 - N_INGAAS_K**2
SIG_INGAAS = 2*N_INGAAS_R*N_INGAAS_K*2*np.pi*C0/LAM*EPS0

T_LENS = 250e-6
R_BEND = 8.3e-3
D_LENS = 14e-3
T_BUF  = 20e-6
D1_LEN = 500e-6
D1_H   = 50e-6            # Detektor-DICKE normal zur Linse (= D1_DICKE)
L_DET_UM = 100.0         # tangentiale Detektor-Laenge im 2D-Fenster (wie plot_geometry Panel B)
D3_LEN = 50e-6
D3_QUERSCHN = 200e-6

D1_S_CENTER = 2.0e-3     # x = -5000 um
D2_S_CENTER = 2.0e-3
D3_S_CENTER = 1.8e-3     # x = -5200 um

# Frame-Save Downsample: speichert nur jeden N-ten Pixel zum Disk/RAM-Sparen.
# FDTD-Rechnung selbst laeuft in voller Aufloesung - Detektor-Flux unveraendert.
# Bei 2: 4x weniger RAM pro Frame, Display-Pixel = 2x dx (z.B. 300 nm bei 150 nm)
SAVE_STRIDE = 2
N_INTRA_FRAMES = 4   # Snapshots pro Slide (war 8, jetzt 4 fuer RAM-Schonung)

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


def build_window_materials(x_start_um, x_end_um, y_start_um, y_end_um,
                           dx_um, layers, n_aq):
    Nx = int(round((x_end_um - x_start_um)/dx_um))
    Ny = int(round((y_end_um - y_start_um)/dx_um))
    eps = np.full((Nx, Ny), N_AIR**2, dtype=np.float32)
    sig = np.zeros((Nx, Ny), dtype=np.float32)
    xs_global_um = x_start_um + np.arange(Nx)*dx_um
    ys_global_um = y_start_um + np.arange(Ny)*dx_um

    sag_um = (R_BEND - np.sqrt(R_BEND**2 - (xs_global_um*1e-6)**2))*1e6
    y_mid_um = -sag_um
    y_lens_top = y_mid_um + T_LENS*1e6/2
    y_lens_bot = y_mid_um - T_LENS*1e6/2
    t_lip = layers[0]*1e6
    t_aq = layers[1]*1e6
    t_mu = layers[2]*1e6
    y_lipid_bot = y_lens_bot - t_lip
    y_aq_bot = y_lipid_bot - t_aq
    y_mu_bot = y_aq_bot - t_mu

    for i in range(Nx):
        eps[i, ys_global_um < y_mu_bot[i]] = N_CORNEA**2
        eps[i, (ys_global_um >= y_mu_bot[i]) & (ys_global_um < y_aq_bot[i])] = N_MUCIN**2
        eps[i, (ys_global_um >= y_aq_bot[i]) & (ys_global_um < y_lipid_bot[i])] = n_aq**2
        eps[i, (ys_global_um >= y_lipid_bot[i]) & (ys_global_um < y_lens_bot[i])] = N_LIPID**2
        eps[i, (ys_global_um >= y_lens_bot[i]) & (ys_global_um <= y_lens_top[i])] = N_PMMA**2

    # ORTHOGONALE STIRNFLAECHE bei x = +7000 um
    x_stirn_apex_um = D_LENS*1e6/2
    if x_end_um > x_stirn_apex_um - 200:
        x_s_m = x_stirn_apex_um*1e-6
        sag_s = R_BEND - np.sqrt(R_BEND**2 - x_s_m**2)
        y_mid_s = -sag_s*1e6
        slope_s = -x_s_m/np.sqrt(R_BEND**2 - x_s_m**2)
        theta_s = np.arctan(slope_s)
        ct_s, st_s = np.cos(theta_s), np.sin(theta_s)
        XX, YY = np.meshgrid(xs_global_um, ys_global_um, indexing='ij')
        s_proj = (XX - x_stirn_apex_um)*ct_s + (YY - y_mid_s)*st_s
        eps[s_proj > 0] = N_AIR**2

    # D1/D2 (tangential rotiert, normal-versetzt)
    x_d1_global_um = (D1_S_CENTER - D_LENS/2)*1e6
    L_2D = L_DET_UM
    H_um = D1_H*1e6
    BUF_um = T_BUF*1e6

    if (x_start_um < x_d1_global_um + L_2D and
        x_d1_global_um - L_2D < x_end_um):
        sag_d1 = R_BEND - np.sqrt(R_BEND**2 - (x_d1_global_um*1e-6)**2)
        y_mid_d1_um = -sag_d1*1e6
        slope_d1 = -(x_d1_global_um*1e-6)/np.sqrt(R_BEND**2 - (x_d1_global_um*1e-6)**2)
        theta = np.arctan(slope_d1)
        ct, st = np.cos(theta), np.sin(theta)
        max_y_shift = L_2D/2*abs(st) + H_um/2*abs(ct)
        d_norm = max(max_y_shift/abs(ct), 50)
        XX, YY = np.meshgrid(xs_global_um, ys_global_um, indexing='ij')
        # D1
        xc1 = x_d1_global_um + st*d_norm
        yc1 = y_mid_d1_um - ct*d_norm
        du1 = ct*(XX - xc1) + st*(YY - yc1)
        dv1 = -st*(XX - xc1) + ct*(YY - yc1)
        m_d1 = (abs(du1) < L_2D/2) & (abs(dv1) < H_um/2)
        eps[m_d1] = EPS_INGAAS
        sig[m_d1] = SIG_INGAAS
        m_buf1 = (abs(du1) < L_2D/2) & (dv1 < -H_um/2) & (dv1 > -H_um/2 - BUF_um)
        eps[m_buf1] = N_BUFFER**2
        # D2
        xc2 = x_d1_global_um - st*d_norm
        yc2 = y_mid_d1_um + ct*d_norm
        du2 = ct*(XX - xc2) + st*(YY - yc2)
        dv2 = -st*(XX - xc2) + ct*(YY - yc2)
        m_d2 = (abs(du2) < L_2D/2) & (abs(dv2) < H_um/2)
        eps[m_d2] = EPS_INGAAS
        sig[m_d2] = SIG_INGAAS
        m_buf2 = (abs(du2) < L_2D/2) & (dv2 > H_um/2) & (dv2 < H_um/2 + BUF_um)
        eps[m_buf2] = N_BUFFER**2

    # D3 (tangential rotiert)
    x_d3_global_um = (D3_S_CENTER - D_LENS/2)*1e6
    if (x_start_um < x_d3_global_um + D3_QUERSCHN*1e6 and
        x_d3_global_um - D3_QUERSCHN*1e6 < x_end_um):
        sag_d3 = R_BEND - np.sqrt(R_BEND**2 - (x_d3_global_um*1e-6)**2)
        y_mid_d3_um = -sag_d3*1e6
        slope_d3 = -(x_d3_global_um*1e-6)/np.sqrt(R_BEND**2 - (x_d3_global_um*1e-6)**2)
        th3 = np.arctan(slope_d3)
        ct3, st3 = np.cos(th3), np.sin(th3)
        XX, YY = np.meshgrid(xs_global_um, ys_global_um, indexing='ij')
        du3 = ct3*(XX - x_d3_global_um) + st3*(YY - y_mid_d3_um)
        dv3 = -st3*(XX - x_d3_global_um) + ct3*(YY - y_mid_d3_um)
        m_d3 = (abs(du3) < D3_LEN*1e6/2) & (abs(dv3) < D3_QUERSCHN*1e6/2)
        eps[m_d3] = EPS_INGAAS
        sig[m_d3] = SIG_INGAAS

    return to_xp(eps), to_xp(sig)


def get_detector_polygons():
    x_d1_um = (D1_S_CENTER - D_LENS/2)*1e6
    sag_d1 = R_BEND - np.sqrt(R_BEND**2 - (x_d1_um*1e-6)**2)
    y_mid_d1_um = -sag_d1*1e6
    slope = -(x_d1_um*1e-6)/np.sqrt(R_BEND**2 - (x_d1_um*1e-6)**2)
    theta = np.arctan(slope)
    ct, st = np.cos(theta), np.sin(theta)
    L_2D = L_DET_UM
    H_um = D1_H*1e6
    max_y_shift = L_2D/2*abs(st) + H_um/2*abs(ct)
    d_norm = max(max_y_shift/abs(ct), 50)

    def mk(xc, yc):
        return [(xc + ct*u - st*v, yc + st*u + ct*v)
                for u, v in [(-L_2D/2, -H_um/2), (+L_2D/2, -H_um/2),
                             (+L_2D/2, +H_um/2), (-L_2D/2, +H_um/2)]]
    polys = {}
    polys['D1'] = mk(x_d1_um + st*d_norm, y_mid_d1_um - ct*d_norm)
    polys['D2'] = mk(x_d1_um - st*d_norm, y_mid_d1_um + ct*d_norm)
    x_d3_um = (D3_S_CENTER - D_LENS/2)*1e6
    sag_d3 = R_BEND - np.sqrt(R_BEND**2 - (x_d3_um*1e-6)**2)
    y_mid_d3_um = -sag_d3*1e6
    slope_d3 = -(x_d3_um*1e-6)/np.sqrt(R_BEND**2 - (x_d3_um*1e-6)**2)
    th3 = np.arctan(slope_d3)
    ct3, st3 = np.cos(th3), np.sin(th3)
    L3 = D3_LEN*1e6
    Q3 = D3_QUERSCHN*1e6
    polys['D3'] = [
        (x_d3_um + ct3*(-L3/2) - st3*(-Q3/2), y_mid_d3_um + st3*(-L3/2) + ct3*(-Q3/2)),
        (x_d3_um + ct3*(+L3/2) - st3*(-Q3/2), y_mid_d3_um + st3*(+L3/2) + ct3*(-Q3/2)),
        (x_d3_um + ct3*(+L3/2) - st3*(+Q3/2), y_mid_d3_um + st3*(+L3/2) + ct3*(+Q3/2)),
        (x_d3_um + ct3*(-L3/2) - st3*(+Q3/2), y_mid_d3_um + st3*(-L3/2) + ct3*(+Q3/2)),
    ]
    return polys


def run_sliding_fdtd(scenario_name, dx_nm=250.0, window_w_um=1000,
                    window_h_um=4000, slide_um=400, save_frames=True,
                    vcsel_waist=2.0, vcsel_tilt=0.0, vcsel_mode='single',
                    vcsel_offset=0.0, source_type='cw'):
    t_total = time.time()
    layers = SCENARIOS[scenario_name]
    print(f'\n========== Sliding-Window-FDTD: {scenario_name} ==========')
    print(f'Layers: lipid={layers[0]*1e6:.3f}um, aq={layers[1]*1e6:.2f}um, '
          f'mu={layers[2]*1e6:.2f}um, n_aq={layers[3]}')

    dx = dx_nm*1e-9
    dt = 0.5*dx/(C0*np.sqrt(2))
    print(f'Grid: dx={dx_nm} nm, dt={dt*1e15:.3f} fs')

    Nx = int(round(window_w_um*1e-6/dx))
    Ny = int(round(window_h_um*1e-6/dx))
    print(f'Window: {window_w_um} x {window_h_um} um = {Nx} x {Ny} cells '
          f'({Nx*Ny*4*5/1e9:.2f} GB peak)')

    x_vcsel_global_um = D_LENS*1e6/2 - 5
    sag_vcsel = R_BEND - np.sqrt(R_BEND**2 - (x_vcsel_global_um*1e-6)**2)
    y_vcsel_global_um = -sag_vcsel*1e6
    print(f'VCSEL bei x = +{x_vcsel_global_um:.0f} um, y = {y_vcsel_global_um:.0f} um')

    n_slides = int(np.ceil(D_LENS*1e6 / slide_um)) + 2
    print(f'Slides: {n_slides} x {slide_um} um')

    # === Co-Moving Window: Field-Shift zwischen Slides ===
    # Wenn das Window um slide_um in -x verschoben wird, muss das Feld um
    # N_shift Zellen in +ix-Richtung verschoben werden, damit es seine
    # globale Lab-Position behaelt. Der Eingangsrand (links, low ix) wird mit
    # 0 aufgefuellt -> dort kommt frisches, noch unsimuliertes Material rein.
    # Die rechte Window-Seite (high ix) verlaesst das Window -> alles was dort
    # noch an Welle ist (z.B. Rueck-Reflexionen) wird verworfen.
    # WICHTIG: slide_um sollte < window_w_um sein, damit Ueberlapp existiert
    # und die Welle ueber die Slide-Grenze hinweg propagieren kann.
    N_shift = int(round(slide_um*1e-6 / dx))
    overlap_um = window_w_um - slide_um
    if overlap_um <= 0:
        print(f'!!! WARNUNG: slide_um ({slide_um}um) >= window_w_um ({window_w_um}um)')
        print(f'    -> KEIN Ueberlapp -> Wellen-Information geht zwischen Slides verloren.')
        print(f'    -> Empfehlung: slide_um <= window_w_um/2 (z.B. slide=200, window=400)')
    else:
        print(f'Co-Moving: N_shift={N_shift} cells (={slide_um}um), '
              f'Ueberlapp={overlap_um}um (={int(overlap_um*1e-6/dx)} cells)')

    import gc
    Ez = xp.zeros((Nx, Ny), dtype=xp.float32)
    Hx = xp.zeros((Nx, Ny-1), dtype=xp.float32)
    Hy = xp.zeros((Nx-1, Ny), dtype=xp.float32)

    P_D1 = 0.0
    P_D2 = 0.0
    P_D3 = 0.0
    # Snapshots (bereits um SAVE_STRIDE reduziert) direkt auf Disk-Memmap streamen
    # statt in eine RAM-Liste zu stapeln -> vermeidet OOM bei langen 1mm-Laeufen.
    frames = []
    ez_mm = None; mm_path = None; k_fr = 0
    if save_frames:
        _sx = (Nx + SAVE_STRIDE - 1)//SAVE_STRIDE
        _sy = (Ny + SAVE_STRIDE - 1)//SAVE_STRIDE
        os.makedirs('results', exist_ok=True)
        mm_path = os.path.join('results', f'_stream_sliding_{scenario_name}.npy')
        ez_mm = np.lib.format.open_memmap(mm_path, mode='w+', dtype=np.float32,
                                          shape=(max(1, n_slides*N_INTRA_FRAMES + 2), _sx, _sy))

    y_start_prev = None
    dxu_shift = dx_nm/1000.0
    for slide_i in range(n_slides):
        x_end_global_um = x_vcsel_global_um + 50 - slide_i*slide_um
        x_start_global_um = x_end_global_um - window_w_um
        x_center_global_um = (x_start_global_um + x_end_global_um)/2
        if abs(x_center_global_um*1e-6) > R_BEND - 1e-6:
            break
        # y-Zentrum folgt der Kruemmung - VOR dem Shift berechnen (fuer y-Tracking)
        sag_c = R_BEND - np.sqrt(R_BEND**2 - (x_center_global_um*1e-6)**2)
        y_center_global_um = -sag_c*1e6
        y_start_global_um = y_center_global_um - window_h_um/2
        y_end_global_um = y_center_global_um + window_h_um/2

        # === Co-Moving Field-Shift (x UND y) ===
        # Das Fenster folgt der gekruemmten Linse in BEIDEN Achsen. Frueher wurde
        # das Feld nur in x gerollt; die y-Wanderung des Fensters (Kruemmung!) fehlte
        # -> der gefuehrte Strahl fiel aus dem Linsenband -> Amplituden-Kollaps.
        if slide_i > 0 and N_shift > 0 and N_shift < Nx:
            Ez = xp.roll(Ez, N_shift, axis=0); Ez[:N_shift, :] = 0
            Hx = xp.roll(Hx, N_shift, axis=0); Hx[:N_shift, :] = 0
            Hy = xp.roll(Hy, N_shift, axis=0); Hy[:min(N_shift, Hy.shape[0]), :] = 0
            diy = int(round((y_start_prev - y_start_global_um)/dxu_shift)) if y_start_prev is not None else 0
            if diy != 0:                                   # Fenster-Mitte in y verschoben -> Feld mitrollen
                Ez = xp.roll(Ez, diy, axis=1)
                Hx = xp.roll(Hx, diy, axis=1)
                Hy = xp.roll(Hy, diy, axis=1)
                if diy > 0:
                    Ez[:, :diy] = 0
                    Hx[:, :min(diy, Hx.shape[1])] = 0
                    Hy[:, :min(diy, Hy.shape[1])] = 0
                else:
                    Ez[:, diy:] = 0
                    Hx[:, diy:] = 0
                    Hy[:, diy:] = 0
        elif slide_i > 0 and N_shift >= Nx:
            Ez[:] = 0; Hx[:] = 0; Hy[:] = 0
        y_start_prev = y_start_global_um

        print(f'\n[Slide {slide_i+1}/{n_slides}]  '
              f'x=[{x_start_global_um:.0f}, {x_end_global_um:.0f}] um  '
              f'y_c={y_center_global_um:.0f} um')

        eps_r, sig = build_window_materials(
            x_start_global_um, x_end_global_um,
            y_start_global_um, y_end_global_um,
            dx_nm/1000, layers, layers[3])

        Ce_n = (1 - sig*dt/(2*EPS0*eps_r))
        Ce_d = (1 + sig*dt/(2*EPS0*eps_r))
        Ce_E = (Ce_n/Ce_d).astype(xp.float32)
        Ce_H = (dt/(EPS0*eps_r*dx)/Ce_d).astype(xp.float32)
        Ch = xp.float32(dt/(MU0*dx))

        src_active = (slide_i == 0)
        f0 = C0/LAM
        sigma_t = 4/(2*np.pi*f0)
        if src_active:
            # TANGENTIAL orientierte Quelle: Injektion IN-PHASE auf der zur Linse
            # ORTHOGONALEN Flaeche (senkrecht zur lokalen Tangente) -> die Wellenfront
            # laeuft entlang der Tangente in den gekruemmten Kern. Ersetzt die alte
            # vertikale Spalte, die am ~57deg-geneigten Rand fehlangepasst war (Kollaps).
            dxu = dx_nm/1000.0
            x_rim = D_LENS*1e6/2
            slope = -x_rim*1e-6/np.sqrt(R_BEND**2 - (x_rim*1e-6)**2)
            theta = np.arctan(slope)                         # Tangentenwinkel am Rand
            ct, st = np.cos(theta), np.sin(theta)
            x_src = x_rim - 6*ct                             # knapp innerhalb der Stirnflaeche
            y_mid_src = -(R_BEND - np.sqrt(R_BEND**2 - (x_src*1e-6)**2))*1e6
            fdx, fdy = -st, ct                              # Flaechenrichtung (normal zur Tangente)
            core_um = T_LENS*1e6
            nface = max(8, int(round(1.6*core_um/dxu)))
            vspan = np.linspace(-0.8*core_um, 0.8*core_um, nface)   # quer ueber die Dicke (um)
            amp_face = np.exp(-((vspan - vcsel_offset)/vcsel_waist)**2)*np.sqrt(1 - 0.162)
            ixf = np.round((x_src + vspan*fdx - x_start_global_um)/dxu).astype(int)
            iyf = np.round((y_mid_src + vspan*fdy - y_start_global_um)/dxu).astype(int)
            ok = (ixf >= 1) & (ixf < Nx - 1) & (iyf >= 1) & (iyf < Ny - 1)
            ixf, iyf, amp_face = ixf[ok], iyf[ok], amp_face[ok]
            flat = ixf*Ny + iyf                            # doppelte Zellen zusammenfassen
            uflat, inv = np.unique(flat, return_inverse=True)
            amp_u = np.zeros(len(uflat), dtype=np.float32); np.add.at(amp_u, inv, amp_face)
            src_ix = to_xp((uflat // Ny).astype(np.int64))
            src_iy = to_xp((uflat % Ny).astype(np.int64))
            src_amp = to_xp(amp_u)
            print(f'  Source: TANGENTIAL theta={np.degrees(theta):.0f}deg waist={vcsel_waist}um '
                  f'offset={vcsel_offset}um {len(uflat)} Flaechenzellen type={source_type}')

        mur = xp.float32((C0*dt - dx)/(C0*dt + dx))

        # Steps so dass Welle genau slide_um propagiert (Co-Moving Konsistenz)
        # In Slide 0 darf sie etwas weiter laufen (Aufbauphase), damit das Window
        # gefuellt ist bevor wir mit dem Shift starten.
        if src_active:
            # Aufbauphase: Welle laeuft window_w_um, plus etwas Reserve
            steps_per_slide = int(window_w_um*1e-6*N_PMMA/C0/dt) + 200
        else:
            # Steady: Welle propagiert genau slide_um pro Slide
            # (= Window-Verschiebung; im Co-Moving Frame bleibt Welle ungefaehr stationaer)
            steps_per_slide = int(slide_um*1e-6*N_PMMA/C0/dt) + 200
        n_intra_frames = N_INTRA_FRAMES if save_frames else 0
        snap_every = max(1, steps_per_slide // n_intra_frames) if n_intra_frames else 999999
        print(f'  Steps: {steps_per_slide}, Snapshots alle {snap_every} (stride={SAVE_STRIDE})')

        t_slide = time.time()
        for n in range(steps_per_slide):
            Hx -= Ch*(Ez[:, 1:] - Ez[:, :-1])
            Hy += Ch*(Ez[1:, :] - Ez[:-1, :])
            ex1, ex2 = Ez[1, :].copy(), Ez[-2, :].copy()
            ey1, ey2 = Ez[:, 1].copy(), Ez[:, -2].copy()
            Ez[1:-1, 1:-1] = (Ce_E[1:-1, 1:-1]*Ez[1:-1, 1:-1]
                              + Ce_H[1:-1, 1:-1]*((Hy[1:, 1:-1] - Hy[:-1, 1:-1])
                                                  - (Hx[1:-1, 1:] - Hx[1:-1, :-1])))
            if src_active:
                t_phys = n*dt
                if source_type == 'pulse':
                    sigma_p = 30 / f0
                    t_center = 3*sigma_p
                    envelope = float(np.exp(-((t_phys - t_center)/sigma_p)**2))
                else:  # cw
                    envelope = float(1 - np.exp(-((t_phys/(2*sigma_t))**2)))
                amp = envelope * float(np.sin(-2*np.pi*f0*t_phys))
                Ez[src_ix, src_iy] += amp * src_amp        # in-Phase auf der orthogonalen Flaeche
            # Mur 1. Ordnung mit Ez^n der Innen-Nachbarn (ex1/ey1, vor E-Update
            # kopiert) - frueher wurde ein um 1 Step veralteter Wert benutzt.
            Ez[0, :]  = ex1 + mur*(Ez[1, :]  - Ez[0, :])
            Ez[-1, :] = ex2 + mur*(Ez[-2, :] - Ez[-1, :])
            Ez[:, 0]  = ey1 + mur*(Ez[:, 1]  - Ez[:, 0])
            Ez[:, -1] = ey2 + mur*(Ez[:, -2] - Ez[:, -1])

            if (save_frames and ez_mm is not None and n_intra_frames
                    and ((n+1) % snap_every == 0) and k_fr < ez_mm.shape[0]):
                ez_np = to_np(Ez)
                # Downsample (SAVE_STRIDE) und direkt auf Disk-Memmap schreiben.
                ez_mm[k_fr] = ez_np[::SAVE_STRIDE, ::SAVE_STRIDE].astype(np.float32)
                frames.append({
                    'x_start': x_start_global_um,
                    'x_end': x_end_global_um,
                    'y_start': y_start_global_um,
                    'y_end': y_end_global_um,
                    'slide_i': slide_i,
                    'step': n+1,
                    'time': time.time() - t_total,
                })
                k_fr += 1
                del ez_np  # explizit aus RAM

        print(f'  FDTD-Time: {time.time()-t_slide:.1f}s, RAM-Frames: {len(frames)}')
        if slide_i % 5 == 0:
            gc.collect()

        # Detektor-Flux
        x_d1_global_um = (D1_S_CENTER - D_LENS/2)*1e6
        if x_start_global_um <= x_d1_global_um <= x_end_global_um:
            sag_d1 = R_BEND - np.sqrt(R_BEND**2 - (x_d1_global_um*1e-6)**2)
            y_mid_d1_um = -sag_d1*1e6
            slope_d1 = -(x_d1_global_um*1e-6)/np.sqrt(R_BEND**2 - (x_d1_global_um*1e-6)**2)
            th_d = np.arctan(slope_d1)
            ct_d, st_d = np.cos(th_d), np.sin(th_d)
            L_2D = 100e-6
            H_det = D1_H
            max_shift = L_2D/2*abs(st_d) + H_det/2*abs(ct_d)
            d_norm_m = max(max_shift/abs(ct_d), 50e-6)
            xc1_m = x_d1_global_um*1e-6 + st_d*d_norm_m
            yc1_m = y_mid_d1_um*1e-6 - ct_d*d_norm_m
            ix_d1 = int(round((xc1_m - x_start_global_um*1e-6)/dx))
            iy_d1 = int(round((yc1_m - y_start_global_um*1e-6)/dx))
            xc2_m = x_d1_global_um*1e-6 - st_d*d_norm_m
            yc2_m = y_mid_d1_um*1e-6 + ct_d*d_norm_m
            ix_d2 = int(round((xc2_m - x_start_global_um*1e-6)/dx))
            iy_d2 = int(round((yc2_m - y_start_global_um*1e-6)/dx))
            ix_w = int(round(L_2D/2/dx))
            iy_h = int(round(H_det/2/dx))
            for (ix, iy, accum) in [(ix_d1, iy_d1, 'D1'), (ix_d2, iy_d2, 'D2')]:
                ix_lo = max(0, ix - ix_w); ix_hi = min(Nx, ix + ix_w)
                iy_lo = max(0, iy - iy_h); iy_hi = min(Ny, iy + iy_h)
                if ix_hi > ix_lo and iy_hi > iy_lo:
                    box = Ez[ix_lo:ix_hi, iy_lo:iy_hi]
                    p = float(xp.sum(box*box))*dx*dx
                    if accum == 'D1':
                        P_D1 += p
                    else:
                        P_D2 += p

        x_d3_global_um = (D3_S_CENTER - D_LENS/2)*1e6
        if x_start_global_um <= x_d3_global_um <= x_end_global_um:
            ix_d3 = int(round((x_d3_global_um - x_start_global_um)*1e-6/dx))
            sag_d3 = R_BEND - np.sqrt(R_BEND**2 - (x_d3_global_um*1e-6)**2)
            y_mid_d3_um = -sag_d3*1e6
            iy_d3 = int(round((y_mid_d3_um - y_start_global_um)*1e-6/dx))
            ix_w = int(round(D3_LEN/2/dx))
            iy_h = int(round(D3_QUERSCHN/2/dx))
            ix_lo = max(0, ix_d3 - ix_w); ix_hi = min(Nx, ix_d3 + ix_w)
            iy_lo = max(0, iy_d3 - iy_h); iy_hi = min(Ny, iy_d3 + iy_h)
            if ix_hi > ix_lo and iy_hi > iy_lo:
                box = Ez[ix_lo:ix_hi, iy_lo:iy_hi]
                P_D3 += float(xp.sum(box*box))*dx*dx

    total_time = time.time() - t_total
    print(f'\n[Result] {scenario_name}: total {total_time:.0f}s')
    print(f'  P_D1 = {P_D1:.3e}')
    print(f'  P_D2 = {P_D2:.3e}')
    print(f'  P_D3 = {P_D3:.3e}')
    R_tear = P_D1/max(P_D2, 1e-30)
    print(f'  R_tear = P_D1/P_D2 = {R_tear:.4f}')

    if ez_mm is not None:
        ez_mm.flush(); del ez_mm
    return dict(scenario=scenario_name, P_D1=P_D1, P_D2=P_D2, P_D3=P_D3,
                R_tear=R_tear, frames=frames, frames_memmap=mm_path,
                total_time=total_time, layers=layers)


def save_results(result, out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    scenario = result['scenario']
    summary = {k: v for k, v in result.items() if k != 'frames'}
    with open(f'{out_dir}/sliding_{scenario}.pkl', 'wb') as f:
        pickle.dump(summary, f)
    print(f'  Saved: {out_dir}/sliding_{scenario}.pkl')

    if result.get('frames'):
        npz_path = f'{out_dir}/sliding_{scenario}_frames.npz'
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
            # Streamed: Frames liegen als Disk-Memmap vor (bereits SAVE_STRIDE-reduziert).
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
        # Material-Namen + Brechzahlen (5 Schichten: lens, lipid, aq, mucin, cornea)
        mat_names = result.get('material_names',
                               ['PMMA', 'Lipid', 'Aqueous', 'Mucin', 'Cornea'])
        mat_indices = np.array(result.get('material_indices',
                                          [N_PMMA, N_LIPID, 1.336, N_MUCIN, N_CORNEA]),
                               dtype=np.float64)
        np.savez_compressed(npz_path, Ez=ezs,
                            meta=meta, scenario=scenario, layers=layers_arr,
                            material_names=np.array(mat_names),
                            material_indices=mat_indices)
        print(f'  Saved: {npz_path} ({ezs.nbytes/1e6:.1f} MB raw)')

    if not result.get('frames'):
        return

    try:
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Polygon as MPoly

        cdict = {
            'red':   [(0.0, 0.10, 0.10), (0.30, 0.00, 0.00), (0.49, 0.04, 0.04),
                      (0.51, 0.10, 0.10), (0.70, 1.00, 1.00), (1.0, 1.00, 1.00)],
            'green': [(0.0, 0.85, 0.85), (0.30, 0.90, 0.90), (0.49, 0.06, 0.06),
                      (0.51, 0.10, 0.10), (0.70, 0.55, 0.55), (1.0, 0.95, 0.95)],
            'blue':  [(0.0, 1.00, 1.00), (0.30, 1.00, 1.00), (0.49, 0.10, 0.10),
                      (0.51, 0.04, 0.04), (0.70, 0.00, 0.00), (1.0, 0.30, 0.30)],
        }
        cmap_wave = LinearSegmentedColormap('wave', cdict)

        polys = get_detector_polygons()
        # Frames fuer das GIF: aus der Memmap (gestreamt) oder aus fr['Ez'] (Fallback).
        def _gif_ez(i, fr):
            return ez_stream[i] if ez_stream is not None else fr['Ez']
        vmax_global = max(float(np.max(np.abs(_gif_ez(i, fr))))
                          for i, fr in enumerate(result['frames']))
        vmax = vmax_global*0.5
        if vmax < 1e-12:
            vmax = 1.0
        layers_local = result.get('layers', (0.030e-6, 3.5e-6, 0.5e-6, 1.336))

        imgs = []
        for i, fr in enumerate(result['frames']):
            fig, ax = plt.subplots(figsize=(10, 5), dpi=85, facecolor='#0a0a0a')
            ax.set_facecolor('#0a0a0a')
            ext = [fr['x_start'], fr['x_end'], fr['y_start'], fr['y_end']]
            ax.imshow(_gif_ez(i, fr).T, extent=ext, origin='lower', cmap=cmap_wave,
                      vmin=-vmax, vmax=vmax, aspect='equal',
                      interpolation='bilinear')

            xs_out = np.linspace(fr['x_start'], fr['x_end'], 200)
            xs_clip = np.clip(xs_out, -R_BEND*1e6 + 1, R_BEND*1e6 - 1)
            sag = (R_BEND - np.sqrt(R_BEND**2 - (xs_clip*1e-6)**2))*1e6
            y_mid_o = -sag
            y_lens_top = y_mid_o + T_LENS*1e6/2
            y_lens_bot = y_mid_o - T_LENS*1e6/2
            t_lip = layers_local[0]*1e6
            t_aq = layers_local[1]*1e6
            t_mu = layers_local[2]*1e6
            y_lipid_bot_o = y_lens_bot - t_lip
            y_aq_bot_o = y_lipid_bot_o - t_aq
            y_mu_bot_o = y_aq_bot_o - t_mu
            ax.plot(xs_out, y_lens_top, '-', color='#FFD75E', lw=1.2, alpha=0.9,
                    label=f'PMMA (n={N_PMMA})')
            ax.plot(xs_out, y_lens_bot, '-', color='#FFD75E', lw=1.2, alpha=0.9)
            ax.plot(xs_out, y_lipid_bot_o, ':', color='#88AAFF', lw=0.8, alpha=0.7,
                    label=f'Lipid {t_lip:.3f}um (n={N_LIPID})')
            ax.plot(xs_out, y_aq_bot_o, '--', color='#56C4FF', lw=1.0, alpha=0.8,
                    label=f'Aqueous {t_aq:.2f}um (n={layers_local[3]})')
            ax.plot(xs_out, y_mu_bot_o, '--', color='#D88AFF', lw=1.0, alpha=0.8,
                    label=f'Mucin {t_mu:.2f}um (n={N_MUCIN})')
            ax.fill_between(xs_out, fr['y_start'], y_mu_bot_o,
                            color='#553030', alpha=0.18,
                            label=f'Cornea (n={N_CORNEA})')

            x_stirn_um = D_LENS*1e6/2
            if fr['x_end'] > x_stirn_um - 400:
                sag_st = (R_BEND - np.sqrt(R_BEND**2 - (x_stirn_um*1e-6)**2))*1e6
                y_mid_st = -sag_st
                slope_st = -x_stirn_um*1e-6/np.sqrt(R_BEND**2 - (x_stirn_um*1e-6)**2)
                th_st = np.arctan(slope_st)
                ct_l, st_l = np.cos(th_st), np.sin(th_st)
                nx, ny = -st_l, ct_l
                x_st_top = x_stirn_um + (T_LENS*1e6/2)*nx
                y_st_top = y_mid_st + (T_LENS*1e6/2)*ny
                x_st_bot = x_stirn_um - (T_LENS*1e6/2)*nx
                y_st_bot = y_mid_st - (T_LENS*1e6/2)*ny
                ax.plot([x_st_bot, x_st_top], [y_st_bot, y_st_top],
                        '-', color='#FF55FF', lw=3, alpha=0.95)
                vL, vW = 150, 160
                vxc = x_stirn_um + (vL/2)*ct_l
                vyc = y_mid_st + (vL/2)*st_l
                pts = []
                for u, v in [(-vL/2,-vW/2),(vL/2,-vW/2),(vL/2,vW/2),(-vL/2,vW/2)]:
                    pts.append((vxc + ct_l*u + nx*v, vyc + st_l*u + ny*v))
                ax.add_patch(MPoly(pts, closed=True, fill=True,
                                   fc='#440000', ec='#FF55FF', lw=1.5, alpha=0.85))
                ax.text(x_stirn_um + (vL+80)*ct_l, y_mid_st + (vL+80)*st_l,
                        'VCSEL\n850 nm', color='#FF55FF', fontsize=9,
                        fontweight='bold', ha='center', va='center')

            if any(fr['x_start'] < p[0] < fr['x_end'] for p in polys['D1']):
                ax.add_patch(MPoly(polys['D1'], closed=True, fill=False,
                                   ec='#FF8800', lw=2.0))
                cx = sum(p[0] for p in polys['D1'])/4
                cy = sum(p[1] for p in polys['D1'])/4
                ax.text(cx + 80, cy, 'D1', color='#FF8800', fontsize=10,
                        fontweight='bold', va='center')
            if any(fr['x_start'] < p[0] < fr['x_end'] for p in polys['D2']):
                ax.add_patch(MPoly(polys['D2'], closed=True, fill=False,
                                   ec='#00AAFF', lw=2.0))
                cx = sum(p[0] for p in polys['D2'])/4
                cy = sum(p[1] for p in polys['D2'])/4
                ax.text(cx - 80, cy, 'D2', color='#00AAFF', fontsize=10,
                        fontweight='bold', va='center', ha='right')
            if any(fr['x_start'] < p[0] < fr['x_end'] for p in polys['D3']):
                ax.add_patch(MPoly(polys['D3'], closed=True, fill=False,
                                   ec='#55FF55', lw=2.0))
                cx = sum(p[0] for p in polys['D3'])/4
                cy = sum(p[1] for p in polys['D3'])/4
                ax.text(cx, cy + 130, 'D3', color='#55FF55', fontsize=10,
                        fontweight='bold', ha='center')

            ax.set_xlim(fr['x_start'], fr['x_end'])
            ax.set_ylim(fr['y_start'], fr['y_end'])
            ax.set_xlabel('x (um)', color='#cccccc', fontsize=9)
            ax.set_ylabel('y (um)', color='#cccccc', fontsize=9)
            ax.tick_params(colors='#cccccc', labelsize=8)
            for sp in ax.spines.values():
                sp.set_color('#444444')
            step_info = f"  step {fr.get('step', '-')}" if 'step' in fr else ''
            ax.set_title(
                f"{result['scenario']}  -  Slide {fr['slide_i']+1}{step_info}"
                f"  -  t = {fr['time']:.1f}s",
                color='#ffffff', fontsize=10)
            leg = ax.legend(loc='upper left', fontsize=7, framealpha=0.55,
                            facecolor='#1a1a1a', edgecolor='#444444')
            for txt in leg.get_texts():
                txt.set_color('#dddddd')

            fig.tight_layout()
            fig.canvas.draw()
            try:
                rgba = np.asarray(fig.canvas.buffer_rgba())
                buf = rgba[..., :3].copy()
            except AttributeError:
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.tostring_rgb(),
                                    dtype=np.uint8).reshape(h, w, 3).copy()
            imgs.append(buf)
            plt.close(fig)

        gif_path = f'{out_dir}/sliding_{result["scenario"]}.gif'
        imageio.mimsave(gif_path, imgs, duration=0.18, loop=0)
        print(f'  Saved: {gif_path}')
    except Exception as e:
        print(f'  GIF skipped: {e}')

    # Temporaeres Stream-Memmap aufraeumen (nach NPZ + GIF).
    try:
        del ezs
    except Exception:
        pass
    _mm = result.get('frames_memmap')
    if _mm and os.path.exists(_mm):
        try:
            _mm_full._mmap.close()   # Windows: Handle schliessen vor os.remove
        except Exception:
            pass
        try:
            del _mm_full, ez_stream
        except Exception:
            pass
        import gc
        gc.collect()
        try:
            os.remove(_mm)
        except OSError:
            pass


def main():
    # global muss VOR jedem Lesezugriff stehen (Python-Syntax-Regel),
    # weil argparse-Defaults sonst lokale Lookups erzeugen wuerden.
    global N_PMMA, N_LIPID, N_MUCIN, N_CORNEA, T_LENS, LAM
    global R_BEND, D_LENS, D1_S_CENTER, D3_S_CENTER, D1_H, D3_LEN, L_DET_UM
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', default='Gesund')
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--resolution', type=float, default=300,
                    help='Grid resolution in nm')
    ap.add_argument('--window-w', type=float, default=400.0,
                    help='Sliding-Window-Breite in um')
    ap.add_argument('--window-h', type=float, default=2000.0,
                    help='Sliding-Window-Hoehe in um')
    ap.add_argument('--slide', type=float, default=200.0,
                    help='Slide-Schrittweite in um (< window_w fuer Co-Moving)')
    ap.add_argument('--lambda-nm', type=float, default=850.0)
    ap.add_argument('--t-lens', type=float, default=250.0)
    ap.add_argument('--n-pmma', type=float, default=N_PMMA)
    ap.add_argument('--n-lipid', type=float, default=N_LIPID)
    ap.add_argument('--n-mucin', type=float, default=N_MUCIN)
    ap.add_argument('--n-cornea', type=float, default=N_CORNEA)
    ap.add_argument('--t-lipid', type=float, default=0.030)
    ap.add_argument('--t-aqueous', type=float, default=3.5)
    ap.add_argument('--t-mucin', type=float, default=0.5)
    ap.add_argument('--n-aqueous', type=float, default=1.336)
    # Material-Namen fuer Plots/Analyzer
    ap.add_argument('--name-lens', default='PMMA', help='Linsenmaterial-Name')
    ap.add_argument('--name-lipid', default='Lipid', help='Lipid-Schicht Name')
    ap.add_argument('--name-aqueous', default='Aqueous', help='Aqueous-Schicht Name')
    ap.add_argument('--name-mucin', default='Mucin', help='Mucin-Schicht Name')
    ap.add_argument('--name-cornea', default='Cornea', help='Cornea Name')
    ap.add_argument('--vcsel-waist', type=float, default=2.0)
    ap.add_argument('--vcsel-tilt', type=float, default=0.0)
    ap.add_argument('--vcsel-offset', type=float, default=0.0)
    ap.add_argument('--source-type', default='cw', choices=['cw', 'pulse'])
    ap.add_argument('--lens-radius', type=float, default=8.3)
    ap.add_argument('--lens-diameter', type=float, default=14.0)
    ap.add_argument('--d1-position', type=float, default=2.0)
    ap.add_argument('--d3-position', type=float, default=1.8)
    ap.add_argument('--d1-length', type=float, default=100.0,
                    help='tangentiale Detektor-Laenge L_2D in um (Dicke fest = D1_DICKE=50um)')
    ap.add_argument('--d3-length', type=float, default=50.0)
    ap.add_argument('--no-frames', action='store_true',
                    help='Keine Frame-NPZ/GIF speichern')
    args = ap.parse_args()

    # Custom Scenario via globale Overrides
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
    # --d1-length steuert die TANGENTIALE Detektor-Laenge (L_2D), NICHT die Dicke!
    # (Frueher faelschlich D1_H=args.d1_length -> 500um dicke Detektoren weit ausserhalb
    # der Linse.) D1_H bleibt die Detektor-DICKE = 50 um.
    L_DET_UM = args.d1_length
    D1_H = 50e-6
    D3_LEN = args.d3_length*1e-6

    if args.scenario == 'Custom':
        SCENARIOS['Custom'] = (args.t_lipid*1e-6, args.t_aqueous*1e-6,
                               args.t_mucin*1e-6, args.n_aqueous)
    elif args.scenario not in SCENARIOS:
        print(f'ERROR: scenario "{args.scenario}" unknown.')
        print(f'Bekannte: {list(SCENARIOS.keys())}')
        return 1

    result = run_sliding_fdtd(
        scenario_name=args.scenario,
        dx_nm=args.resolution,
        window_w_um=args.window_w,
        window_h_um=args.window_h,
        slide_um=args.slide,
        save_frames=not args.no_frames,
        vcsel_waist=args.vcsel_waist*1e-6,
        vcsel_tilt=args.vcsel_tilt,
        vcsel_offset=args.vcsel_offset*1e-6,
        source_type=args.source_type)
    result['material_names'] = [args.name_lens, args.name_lipid,
                                 args.name_aqueous, args.name_mucin,
                                 args.name_cornea]
    result['material_indices'] = [args.n_pmma, args.n_lipid, args.n_aqueous,
                                                         args.n_mucin, args.n_cornea]
    save_results(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
