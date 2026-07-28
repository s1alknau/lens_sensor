"""2D-FDTD-Kern (planarer Waveguide + optionaler Bead am Waveguide-Tear-Interface).

Gegenstueck zu planar_3d/fdtd3d_core.py fuer 2D. Reine Solver-Library ohne CLI -
der Einstieg laeuft ueber run_simulation.py im Repo-Root (Geometrie/Dimension/Methode).

Schichtsystem: Air / Waveguide / (Lipid) / Aqueous / Mucin / Cornea. Ein optionaler
Bead (place_bead) sitzt in der Aqueous.

METHODEN:
  full    - ganzes Fenster auf einmal
  sliding - Co-Moving-Window (Felder + Mur-Rand verschoben, Material pro Slide neu)
  stitch  - Gebiets-Zerlegung -> voller gefuellter CW-Waveguide (Handoff)

2D-Hinweis: ein "Bead" ist ein unendlich langer Zylinder, keine echte Kugel.
"""
import os, sys, time, pickle
import numpy as np

# Repo-Root auf den Importpfad, damit das geteilte common/-Paket gefunden wird,
# unabhaengig davon, aus welchem Ordner das Skript gestartet wurde.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# Backend (GPU/CPU) und Physik/Materialdaten aus den geteilten Modulen.
from common.backend import xp, cp, GPU_AVAILABLE, to_np           # noqa: E402
from common.physics import C0, EPS0, MU0, N_AIR, DISPERSION, n_at  # noqa: E402

# ---------- demo_beads-spezifische Material-Defaults ----------
# Kurznamen-Konstanten fuer die 2D-Demo (Brechzahlen aus common.physics ableiten
# waere moeglich, die festen 850nm-Defaults bleiben hier aber bewusst explizit).
N_POLYSTYR = 1.590
N_PMMA = 1.491
N_AQ = 1.336
N_MUCIN = 1.342
N_CORNEA = 1.376
N_LIPID = 1.480
MAT = {'polystyrol': N_POLYSTYR, 'pmma': N_PMMA}

# ---------- Geometrie ----------
L_DEMO = 1.0e-3
AIR_BUFFER = 5.0e-6
TEAR_BUFFER = 10.0e-6
T_MU = 2.0e-6            # Mucin-Dicke
T_AQ = 6.0e-6            # Aqueous(Wasser)-Dicke, FEST (> groesster Bead) -> Bead eingebettet


def aqueous_thickness(bead_d_um):
    """Wasserdicke je nach Bead-Groesse: >=1um -> 4um (Mucin @ -4um),
    <1um -> 1um (Mucin @ -1um). Bead bleibt in Wasser eingebettet."""
    return 4.0e-6 if bead_d_um >= 1.0 else 1.0e-6
BEAD_X_UM = 500.0       # Bead-Position (Mitte des Waveguides)
AIR_END = 40e-6         # Luft-Zone am WG-Ende -> echtes Material/Luft-Facet (Fresnel-Reflex)
DET_LEN = 30e-6
X_IN_UM = 100.0
X_OUT_UM = 900.0


def build_materials(Nx, Ny, dx, wg_n, bead_n, wg_top, bead_x_um, bead_d_um, x_start_um=0.0, place_bead=True,
                    n_aq=N_AQ, n_mucin=N_MUCIN, n_cornea=N_CORNEA, t_aq_um=None,
                    t_mu_um=None, t_lip_um=0.0, n_lip=N_LIPID, input_gap_um=0.0,
                    length_um=None, bead_y_um=None):
    """Flache Schichten. AQUEOUS grenzt an den WG (y=0), feste Dicke (t_aq_um).
    Optional Lipidschicht (t_lip_um) direkt unter dem WG; freie Mucin-Dicke
    (t_mu_um). Bead liegt in der Aqueous (Default Oberkante an y=0). Optionaler
    Einkoppelabstand (input_gap_um): Luftzone bei x<gap vor dem WG-Eintritt.
    length_um: Propagationslaenge (Default L_DEMO) -> Position der End-Facette."""
    t_aq = (t_aq_um*1e-6) if t_aq_um is not None else T_AQ
    t_mu = (t_mu_um*1e-6) if t_mu_um is not None else T_MU
    t_lip = (t_lip_um or 0.0)*1e-6
    L = (length_um*1e-6) if length_um is not None else L_DEMO
    x_wg_start = (input_gap_um or 0.0)*1e-6      # WG-Eintrittsfacette (global, x)
    x_wg_end = L                                 # WG-Endfacette (global, x)
    if place_bead and bead_d_um*1e-6 > t_aq + 1e-12:
        print(f'  [WARN] Bead d={bead_d_um}um > Aqueous {t_aq*1e6:g}um -> ragt in Mucin. '
              f'T_AQUEOUS erhoehen.')
    eps = np.full((Nx, Ny), N_AIR**2, dtype=np.float32)
    sig = np.zeros((Nx, Ny), dtype=np.float32)
    ys = -TEAR_BUFFER + np.arange(Ny)*dx
    y_lip_bot = -t_lip
    y_aq_bot = y_lip_bot - t_aq
    y_mu_bot = y_aq_bot - t_mu
    # SUBSTRAT (unter y=0) ueber die GANZE Laenge: Wasser ist ueberall unter dem Slab
    # und daneben/dahinter. Ueber y=0 ist standardmaessig LUFT.
    col = np.full(Ny, N_AIR**2, dtype=np.float32)
    col[ys < y_mu_bot] = n_cornea**2
    col[(ys >= y_mu_bot) & (ys < y_aq_bot)] = n_mucin**2
    col[(ys >= y_aq_bot) & (ys < y_lip_bot)] = n_aq**2
    if t_lip > 1e-12:
        col[(ys >= y_lip_bot) & (ys < 0.0)] = n_lip**2
    eps[:, :] = col[None, :]
    # PMMA-KERN nur wo der Slab ist: x in [x_wg_start, x_wg_end], y in [0, t_wg].
    # Neben/hinter dem Slab ist ueber dem Wasser Luft -> echte Facetten an beiden Enden.
    xs_abs = x_start_um*1e-6 + np.arange(Nx)*dx
    in_wg_x = (xs_abs >= x_wg_start) & (xs_abs <= x_wg_end)
    core_y = (ys >= 0.0) & (ys <= wg_top)
    if in_wg_x.any() and core_y.any():
        eps[np.ix_(in_wg_x, core_y)] = wg_n**2
    # Einzel-Bead (nur wenn place_bead -> Referenzlauf ohne Bead moeglich)
    if place_bead:
        r = bead_d_um*1e-6/2.0
        xc = bead_x_um*1e-6
        x0 = x_start_um*1e-6
        xs = x0 + np.arange(Nx)*dx
        if not (xc + r < xs[0] or xc - r > xs[-1]):
            yc = (bead_y_um*1e-6) if bead_y_um is not None else -r
            ix0 = max(0, int((xc - r - x0)/dx) - 2)
            ix1 = min(Nx, int((xc + r - x0)/dx) + 2)
            for i in range(ix0, ix1):
                dxx = xs[i] - xc
                if abs(dxx) > r:
                    continue
                dy = np.sqrt(max(0.0, r*r - dxx*dxx))
                mask = (ys >= yc - dy) & (ys <= yc + dy)
                eps[i, mask] = bead_n**2
    return xp.asarray(eps), xp.asarray(sig)


def _coeffs(eps_r, sig, dt, dx):
    Ce_n = (1 - sig*dt/(2*EPS0*eps_r))
    Ce_d = (1 + sig*dt/(2*EPS0*eps_r))
    Ce_E = (Ce_n/Ce_d).astype(xp.float32)
    Ce_H = (dt/(EPS0*eps_r*dx)/Ce_d).astype(xp.float32)
    del Ce_n, Ce_d
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
    return Ce_E, Ce_H


def _src_setup(Ny, dx, T_WG, vcsel_offset, vcsel_waist):
    y_wg_center = T_WG/2 + vcsel_offset*1e-6
    iy_src = int(round((y_wg_center + TEAR_BUFFER)/dx))
    waist_pix = max(2, int(round(vcsel_waist*1e-6/dx)))
    ys_idx = xp.arange(Ny, dtype=xp.float32)
    src_p = xp.exp(-((ys_idx - iy_src)/waist_pix)**2)
    return iy_src, src_p, ys_idx


# ------------------------------------------------------------------ TM (p-Pol)
# 2D-TM-Solver: Felder Hz, Ex, Ey (E in der x-y-Ebene, H aus der Ebene). Die
# TE-Routinen nutzen Ez/Hx/Hy; TM ist dazu vollstaendig entkoppelt und braucht
# ein eigenes Yee-Layout:
#   Ex : (Nx-1, Ny)    an (i+1/2, j)
#   Ey : (Nx,   Ny-1)  an (i,     j+1/2)
#   Hz : (Nx-1, Ny-1)  an (i+1/2, j+1/2)
def _tm_coeffs(eps_r, dt, dx):
    """eps_r auf Knoten (Nx,Ny) -> Update-Koeffizienten fuer Ex/Ey an ihren
    versetzten Positionen (eps dort gemittelt) und fuer Hz."""
    eps_ex = 0.5*(eps_r[:-1, :] + eps_r[1:, :])          # (Nx-1, Ny)
    eps_ey = 0.5*(eps_r[:, :-1] + eps_r[:, 1:])          # (Nx,  Ny-1)
    Ca_ex = (dt/(EPS0*eps_ex*dx)).astype(xp.float32)
    Ca_ey = (dt/(EPS0*eps_ey*dx)).astype(xp.float32)
    Ch_tm = xp.float32(dt/(MU0*dx))
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
    return Ca_ex, Ca_ey, Ch_tm


def _tm_node_field(Ey):
    """Ey (Nx,Ny-1) -> auf Knotengitter (Nx,Ny) gemittelt, damit die
    Frame-/Analyzer-Pipeline (erwartet (Nx,Ny)) unveraendert bleibt. Ey ist die
    dominante transversale E-Komponente des gefuehrten TM-Mode."""
    out = xp.zeros((Ey.shape[0], Ey.shape[1] + 1), dtype=xp.float32)
    out[:, 1:-1] = 0.5*(Ey[:, :-1] + Ey[:, 1:])
    out[:, 0] = Ey[:, 0]; out[:, -1] = Ey[:, -1]
    return out


def run_beads(wg_mat, bead_mat, bead_d_um, dx_nm=20.0, save_frames=True, n_snapshots=12,
              wg_thickness_um=5.0, lambda_nm=850.0, vcsel_waist=2.0, vcsel_tilt=0.0,
              vcsel_offset=0.0, source_type='cw', method='full',
              window_w_um=350.0, slide_um=150.0, place_bead=True, t_aqueous_um=None,
              t_mucin_um=None, t_lipid_um=0.0, input_gap_um=0.0, length_um=None,
              bead_x_um=None, wg_n=None, bead_n=None, n_aqueous=None, n_mucin=None,
              n_cornea=None, n_lipid=None, polarization='s'):
    if method == 'sliding':
        return run_beads_sliding(wg_mat, bead_mat, bead_d_um, dx_nm, save_frames, n_snapshots,
                                 wg_thickness_um, lambda_nm, vcsel_waist, vcsel_tilt,
                                 vcsel_offset, source_type, window_w_um, slide_um, place_bead,
                                 t_aqueous_um, t_mucin_um=t_mucin_um, t_lipid_um=t_lipid_um,
                                 input_gap_um=input_gap_um, length_um=length_um,
                                 bead_x_um=bead_x_um, wg_n=wg_n, bead_n=bead_n,
                                 n_aqueous=n_aqueous, n_mucin=n_mucin, n_cornea=n_cornea,
                                 n_lipid=n_lipid, polarization=polarization)
    if method == 'stitch':
        return run_beads_stitched(wg_mat, bead_mat, bead_d_um, dx_nm, save_frames, n_snapshots,
                                  wg_thickness_um, lambda_nm, vcsel_waist, vcsel_tilt,
                                  vcsel_offset, source_type, window_w_um, slide_um, place_bead,
                                  t_aqueous_um, t_mucin_um=t_mucin_um, t_lipid_um=t_lipid_um,
                                  input_gap_um=input_gap_um, length_um=length_um,
                                  bead_x_um=bead_x_um, wg_n=wg_n, bead_n=bead_n,
                                  n_aqueous=n_aqueous, n_mucin=n_mucin, n_cornea=n_cornea,
                                  n_lipid=n_lipid, polarization=polarization)
    _pol = 'p' if str(polarization).lower() in ('p', 'tm') else 's'
    t_aq_um = t_aqueous_um if t_aqueous_um is not None else T_AQ*1e6   # feste Aqueous-Dicke
    t_total = time.time()
    wg_n = wg_n if wg_n is not None else n_at(wg_mat, lambda_nm)
    bead_n = bead_n if bead_n is not None else n_at(bead_mat, lambda_nm)
    n_aq_l = n_aqueous if n_aqueous is not None else n_at('aqueous', lambda_nm)
    n_mu_l = n_mucin if n_mucin is not None else n_at('mucin', lambda_nm)
    n_co_l = n_cornea if n_cornea is not None else n_at('cornea', lambda_nm)
    n_lip_l = n_lipid if n_lipid is not None else N_LIPID
    L = (length_um*1e-6) if length_um is not None else L_DEMO
    L_um = L*1e6
    bx_um = bead_x_um if bead_x_um is not None else L_um/2.0
    T_WG = wg_thickness_um*1e-6; LAM = lambda_nm*1e-9
    label = f'WG-{wg_mat}_Bead-{bead_mat}_d{bead_d_um:g}_L{lambda_nm:g}' + ('' if place_bead else '_ref')
    print(f'\n===== Bead-Demo [full]: {label} =====')
    dx = dx_nm*1e-9
    dt = 0.5*dx/(C0*np.sqrt(2))
    H_total = T_WG + AIR_BUFFER + TEAR_BUFFER
    Nx = int(round(L/dx)); Ny = int(round(H_total/dx))
    print(f'Grid: dx={dx_nm}nm  {Nx}x{Ny}  WG n={wg_n:.3f} t={wg_thickness_um}um  Bead d={bead_d_um}um n={bead_n:.3f}')
    print(f'Laenge={L_um:g}um  Aqueous={t_aq_um:g}um (fest, an WG)  Lipid={(t_lipid_um or 0):g}um  '
          f'Gap={input_gap_um:g}um  Bead-x={bx_um:g}um')

    eps_r, sig = build_materials(Nx, Ny, dx, wg_n, bead_n, T_WG, bx_um, bead_d_um, 0.0, place_bead,
                                 n_aq=n_aq_l, n_mucin=n_mu_l, n_cornea=n_co_l, t_aq_um=t_aq_um,
                                 t_mu_um=t_mucin_um, t_lip_um=t_lipid_um, n_lip=n_lip_l,
                                 input_gap_um=input_gap_um, length_um=length_um)
    Ce_E, Ce_H = _coeffs(eps_r, sig, dt, dx)
    if _pol == 'p':                                  # TM-Koeffizienten aus eps_r
        Ca_ex, Ca_ey, Ch_tm = _tm_coeffs(eps_r, dt, dx)
    del eps_r, sig
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
    Ch = xp.float32(dt/(MU0*dx))
    Ez = xp.zeros((Nx, Ny), dtype=xp.float32)         # TE-Hauptfeld
    Hx = xp.zeros((Nx, Ny-1), dtype=xp.float32)
    Hy = xp.zeros((Nx-1, Ny), dtype=xp.float32)
    # TM-Felder (nur bei p): Ex, Ey, Hz
    Ex_t = xp.zeros((Nx-1, Ny), dtype=xp.float32)
    Ey_t = xp.zeros((Nx, Ny-1), dtype=xp.float32)
    Hz_t = xp.zeros((Nx-1, Ny-1), dtype=xp.float32)
    src_ix = 5
    iy_src, src_p, ys_idx = _src_setup(Ny, dx, T_WG, vcsel_offset, vcsel_waist)
    src_p_ey = 0.5*(src_p[:-1] + src_p[1:])           # Ey-Gitter (Ny-1)
    f0 = C0/LAM; sigma_t = 4/(2*np.pi*f0); k0 = 2*np.pi/LAM
    phase_y = float(np.sin(np.radians(vcsel_tilt)))*k0*(ys_idx - iy_src)*dx
    phase_y_ey = 0.5*(phase_y[:-1] + phase_y[1:])
    mur = xp.float32((C0*dt - dx)/(C0*dt + dx))
    print(f'Polarisation: {_pol}-Pol -> {"TM (Ey/Ex/Hz), Quelle treibt Ey" if _pol=="p" else "TE (Ez/Hx/Hy), Quelle treibt Ez"}')
    steps_total = int(L*wg_n/C0/dt)*2
    snap_every = max(1, steps_total//n_snapshots)
    print(f'Steps: {steps_total}')
    iy_lo = int(round((0.0 + TEAR_BUFFER)/dx)); iy_hi = int(round((T_WG + TEAR_BUFFER)/dx))
    def xwin(xu):
        c = int(round(xu*1e-6/dx))
        return max(0, c-int(round(DET_LEN/2/dx))), min(Nx, c+int(round(DET_LEN/2/dx)))
    # Detektorfenster skalieren mit der Laenge (10% Eingang / 90% Ausgang)
    ixin0, ixin1 = xwin(0.1*L_um); ixout0, ixout1 = xwin(0.9*L_um)
    P_in = 0.0; P_out = 0.0
    # CW-Steady per laufendem DFT bei f0 (letzte ~8 Perioden) -> zweiter,
    # eingeschwungener Frame-Satz (Phasen ueber 1 Periode), zusaetzlich zum
    # transienten (einlaufenden) Satz. Akkumulation auf dem Device.
    omega = 2*np.pi*f0
    steps_per_T = max(4, int(round((1.0/f0)/dt)))
    n_acc_cw = 8*steps_per_T
    acc_cw = xp.zeros((Nx, Ny), dtype=xp.complex64); nacc_cw = 0
    # Snapshots direkt auf Disk-Memmap (kein RAM-Stapel).
    frames = []
    ez_mm = None; mm_path = None; k_fr = 0
    if save_frames:
        os.makedirs('results', exist_ok=True)
        _safe = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in label)
        mm_path = os.path.join('results', f'_stream_beads_{_safe}_full.npy')
        ez_mm = np.lib.format.open_memmap(mm_path, mode='w+', dtype=np.float32,
                                          shape=(max(1, n_snapshots + 2), Nx, Ny))
    t_start = time.time()
    for n in range(steps_total):
        t_phys = n*dt
        if source_type == 'pulse':
            sp = 30/f0; envelope = float(np.exp(-((t_phys - 3*sp)/sp)**2))
        else:
            envelope = float(1 - np.exp(-((t_phys/(2*sigma_t))**2)))
        if _pol == 'p':
            # --- TM (Hz, Ex, Ey) ---
            Hz_t += -Ch_tm*((Ey_t[1:, :] - Ey_t[:-1, :]) - (Ex_t[:, 1:] - Ex_t[:, :-1]))
            ey_l = Ey_t[1, :].copy(); ey_r = Ey_t[-2, :].copy()
            ex_b = Ex_t[:, 1].copy(); ex_t = Ex_t[:, -2].copy()
            Ex_t[:, 1:-1] += Ca_ex[:, 1:-1]*(Hz_t[:, 1:] - Hz_t[:, :-1])
            Ey_t[1:-1, :] += -Ca_ey[1:-1, :]*(Hz_t[1:, :] - Hz_t[:-1, :])
            if vcsel_tilt != 0.0:
                Ey_t[src_ix, :] += envelope*xp.sin(2*np.pi*f0*t_phys + phase_y_ey)*src_p_ey
            else:
                Ey_t[src_ix, :] += envelope*float(np.sin(2*np.pi*f0*t_phys))*src_p_ey
            # Mur 1. Ordnung: Ey an x-Raendern, Ex an y-Raendern (tangential)
            Ey_t[0, :] = ey_l + mur*(Ey_t[1, :] - Ey_t[0, :])
            Ey_t[-1, :] = ey_r + mur*(Ey_t[-2, :] - Ey_t[-1, :])
            Ex_t[:, 0] = ex_b + mur*(Ex_t[:, 1] - Ex_t[:, 0])
            Ex_t[:, -1] = ex_t + mur*(Ex_t[:, -2] - Ex_t[:, -1])
            field = _tm_node_field(Ey_t)              # (Nx,Ny) fuer Frames/Detektor
        else:
            # --- TE (Ez, Hx, Hy) ---
            Hx -= Ch*(Ez[:, 1:] - Ez[:, :-1])
            Hy += Ch*(Ez[1:, :] - Ez[:-1, :])
            ex1, ex2 = Ez[1, :].copy(), Ez[-2, :].copy()
            ey1, ey2 = Ez[:, 1].copy(), Ez[:, -2].copy()
            Ez[1:-1, 1:-1] = (Ce_E[1:-1, 1:-1]*Ez[1:-1, 1:-1]
                              + Ce_H[1:-1, 1:-1]*((Hy[1:, 1:-1] - Hy[:-1, 1:-1])
                                                  - (Hx[1:-1, 1:] - Hx[1:-1, :-1])))
            if vcsel_tilt != 0.0:
                Ez[src_ix, :] += envelope*xp.sin(2*np.pi*f0*t_phys + phase_y)*src_p
            else:
                Ez[src_ix, :] += envelope*float(np.sin(2*np.pi*f0*t_phys))*src_p
            Ez[0, :] = ex1 + mur*(Ez[1, :] - Ez[0, :]); Ez[-1, :] = ex2 + mur*(Ez[-2, :] - Ez[-1, :])
            Ez[:, 0] = ey1 + mur*(Ez[:, 1] - Ez[:, 0]); Ez[:, -1] = ey2 + mur*(Ez[:, -2] - Ez[:, -1])
            field = Ez
        bi = field[ixin0:ixin1, iy_lo:iy_hi]; bo = field[ixout0:ixout1, iy_lo:iy_hi]
        P_in += float(xp.sum(bi*bi))*dx*dx; P_out += float(xp.sum(bo*bo))*dx*dx
        if n >= steps_total - n_acc_cw:                 # eingeschwungenes Ê akkumulieren
            acc_cw += field.astype(xp.complex64)*np.exp(-1j*omega*t_phys)
            nacc_cw += 1
        if save_frames and ez_mm is not None and ((n+1) % snap_every == 0) and k_fr < ez_mm.shape[0]:
            ez_mm[k_fr] = to_np(field).astype(np.float32)
            frames.append({'x_start': 0.0,
                           'x_end': L_um, 'y_start': -TEAR_BUFFER*1e6,
                           'y_end': (T_WG + AIR_BUFFER)*1e6, 'slide_i': 0,
                           'step': n+1, 'time': time.time()-t_start})
            k_fr += 1
            print(f'  Step {n+1}/{steps_total} ({100*(n+1)/steps_total:.0f}%)  max|E|={float(xp.max(xp.abs(field))):.3e}')
    if ez_mm is not None:
        ez_mm.flush(); del ez_mm
    # eingeschwungenen (CW) Frame-Satz aus Ê schreiben: nfr Phasen ueber 1 Periode
    cw_path = None; cw_frames = []
    nfr_cw = max(1, n_snapshots)
    if save_frames and nacc_cw > 0:
        Ecw = to_np((2.0/nacc_cw)*acc_cw)             # komplexe Amplitude (Nx,Ny)
        cw_path = os.path.join('results', f'_stream_beads_{_safe}_full_cw.npy')
        cw_mm = np.lib.format.open_memmap(cw_path, mode='w+', dtype=np.float32,
                                          shape=(nfr_cw, Nx, Ny))
        for k in range(nfr_cw):
            tk = (k/nfr_cw)*(1.0/f0)
            cw_mm[k] = np.real(Ecw*np.exp(1j*omega*tk)).astype(np.float32)
            cw_frames.append({'x_start': 0.0, 'x_end': L_um, 'y_start': -TEAR_BUFFER*1e6,
                              'y_end': (T_WG + AIR_BUFFER)*1e6, 'slide_i': 0,
                              'step': k, 'time': tk})
        cw_mm.flush(); del cw_mm
    total_time = time.time() - t_total
    Tr = P_out/max(P_in, 1e-30)
    print(f'\n[Result] {label} [full]: total {total_time:.0f}s  Transmission={Tr:.4f}')
    res = _result_dict(label, P_in, P_out, Tr, frames, total_time, bead_d_um,
                       wg_mat, bead_mat, wg_n, bead_n, wg_thickness_um, lambda_nm,
                       vcsel_waist, vcsel_tilt, vcsel_offset, source_type, 'full', place_bead, n_aq_l, n_mu_l, n_co_l,
                       frames_memmap=mm_path, t_aq_um=t_aq_um, t_mu_um=t_mucin_um,
                       t_lip_um=t_lipid_um, bead_x_um=bx_um,
                       x_wg_start_um=(input_gap_um or 0.0), x_wg_end_um=L_um,
                       polarization=_pol)
    res['cw_frames'] = cw_frames
    res['cw_frames_memmap'] = cw_path
    res['view_primary'] = 'transient'      # full: einlaufend ist der Hauptsatz
    return res


def run_beads_sliding(wg_mat, bead_mat, bead_d_um, dx_nm, save_frames, n_snapshots,
                      wg_thickness_um, lambda_nm, vcsel_waist, vcsel_tilt,
                      vcsel_offset, source_type, window_w_um, slide_um, place_bead=True,
                      t_aqueous_um=None, t_mucin_um=None, t_lipid_um=0.0, input_gap_um=0.0,
                      length_um=None, bead_x_um=None, wg_n=None, bead_n=None,
                      n_aqueous=None, n_mucin=None, n_cornea=None, n_lipid=None,
                      polarization='s'):
    _pol = 'p' if str(polarization).lower() in ('p', 'tm') else 's'
    t_total = time.time()
    t_aq_um = t_aqueous_um if t_aqueous_um is not None else T_AQ*1e6   # feste Aqueous-Dicke
    wg_n = wg_n if wg_n is not None else n_at(wg_mat, lambda_nm)
    bead_n = bead_n if bead_n is not None else n_at(bead_mat, lambda_nm)
    n_aq_l = n_aqueous if n_aqueous is not None else n_at('aqueous', lambda_nm)
    n_mu_l = n_mucin if n_mucin is not None else n_at('mucin', lambda_nm)
    n_co_l = n_cornea if n_cornea is not None else n_at('cornea', lambda_nm)
    n_lip_l = n_lipid if n_lipid is not None else N_LIPID
    L = (length_um*1e-6) if length_um is not None else L_DEMO
    T_WG = wg_thickness_um*1e-6; LAM = lambda_nm*1e-9
    label = f'WG-{wg_mat}_Bead-{bead_mat}_d{bead_d_um:g}_L{lambda_nm:g}' + ('' if place_bead else '_ref')
    dx = dx_nm*1e-9
    dt = 0.5*dx/(C0*np.sqrt(2))
    H_total = T_WG + AIR_BUFFER + TEAR_BUFFER
    Nx = int(round(window_w_um*1e-6/dx)); Ny = int(round(H_total/dx))
    L_um = L*1e6
    bx_um = bead_x_um if bead_x_um is not None else L_um/2.0
    n_slides = int(np.ceil(L_um/slide_um))
    N_shift = int(round(slide_um*1e-6/dx))
    print(f'\n===== Bead-Demo [sliding]: {label} =====')
    print(f'Window {window_w_um}um = {Nx}x{Ny}, {n_slides} Slides, shift={N_shift} cells')
    Ch = xp.float32(dt/(MU0*dx))
    Ez = xp.zeros((Nx, Ny), dtype=xp.float32)
    Hx = xp.zeros((Nx, Ny-1), dtype=xp.float32)
    Hy = xp.zeros((Nx-1, Ny), dtype=xp.float32)
    Ex_t = xp.zeros((Nx-1, Ny), dtype=xp.float32)     # TM-Felder (nur bei p)
    Ey_t = xp.zeros((Nx, Ny-1), dtype=xp.float32)
    Hz_t = xp.zeros((Nx-1, Ny-1), dtype=xp.float32)
    src_ix = 5
    iy_src, src_p, ys_idx = _src_setup(Ny, dx, T_WG, vcsel_offset, vcsel_waist)
    src_p_ey = 0.5*(src_p[:-1] + src_p[1:])
    f0 = C0/LAM; sigma_t = 4/(2*np.pi*f0); k0 = 2*np.pi/LAM
    phase_y = float(np.sin(np.radians(vcsel_tilt)))*k0*(ys_idx - iy_src)*dx
    phase_y_ey = 0.5*(phase_y[:-1] + phase_y[1:])
    mur = xp.float32((C0*dt - dx)/(C0*dt + dx))
    print(f'Polarisation: {_pol}-Pol ({"TM/Ey" if _pol=="p" else "TE/Ez"})')
    iy_lo = int(round((0.0 + TEAR_BUFFER)/dx)); iy_hi = int(round((T_WG + TEAR_BUFFER)/dx))
    iy_h_det = int(round(DET_LEN/2/dx))
    P_in = 0.0; P_out = 0.0
    frames = []
    ez_mm = None; mm_path = None; k_fr = 0
    if save_frames:
        os.makedirs('results', exist_ok=True)
        _safe = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in label)
        mm_path = os.path.join('results', f'_stream_beads_{_safe}_sliding.npy')
        ez_mm = np.lib.format.open_memmap(mm_path, mode='w+', dtype=np.float32,
                                          shape=(max(1, n_snapshots + n_slides + 4), Nx, Ny))
    t_start = time.time()
    for slide_i in range(n_slides):
        x_start_um = slide_i*slide_um
        if x_start_um >= L_um:
            break
        x_end_um = x_start_um + window_w_um
        eps_r, sig = build_materials(Nx, Ny, dx, wg_n, bead_n, T_WG, bx_um, bead_d_um, x_start_um, place_bead,
                                     n_aq=n_aq_l, n_mucin=n_mu_l, n_cornea=n_co_l, t_aq_um=t_aq_um,
                                     t_mu_um=t_mucin_um, t_lip_um=t_lipid_um, n_lip=n_lip_l,
                                     input_gap_um=input_gap_um, length_um=length_um)
        Ce_E, Ce_H = _coeffs(eps_r, sig, dt, dx)
        if _pol == 'p':
            Ca_ex, Ca_ey, Ch_tm = _tm_coeffs(eps_r, dt, dx)
        del eps_r, sig
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
        if slide_i > 0 and 0 < N_shift < Nx:
            if _pol == 'p':
                Ex_t = xp.roll(Ex_t, -N_shift, axis=0); Ex_t[Ex_t.shape[0]-N_shift:, :] = 0
                Ey_t = xp.roll(Ey_t, -N_shift, axis=0); Ey_t[Nx-N_shift:, :] = 0
                Hz_t = xp.roll(Hz_t, -N_shift, axis=0); Hz_t[Hz_t.shape[0]-N_shift:, :] = 0
            else:
                Ez = xp.roll(Ez, -N_shift, axis=0); Ez[Nx-N_shift:, :] = 0
                Hx = xp.roll(Hx, -N_shift, axis=0); Hx[Nx-N_shift:, :] = 0
                Hy = xp.roll(Hy, -N_shift, axis=0); Hy[Hy.shape[0]-N_shift:, :] = 0
        src_active = (slide_i == 0)
        steps_slide = (int(window_w_um*1e-6*wg_n/C0/dt) if src_active
                       else int(slide_um*1e-6*wg_n/C0/dt)) + 200
        snap_every = max(1, steps_slide//max(1, n_snapshots//n_slides + 1))
        print(f'[Slide {slide_i+1}/{n_slides}] x=[{x_start_um:.0f},{x_end_um:.0f}]um src={src_active} {steps_slide} steps')
        for n in range(steps_slide):
            t_phys = n*dt
            if src_active:
                if source_type == 'pulse':
                    sp = 30/f0; envelope = float(np.exp(-((t_phys - 3*sp)/sp)**2))
                else:
                    envelope = float(1 - np.exp(-((t_phys/(2*sigma_t))**2)))
            if _pol == 'p':
                Hz_t += -Ch_tm*((Ey_t[1:, :] - Ey_t[:-1, :]) - (Ex_t[:, 1:] - Ex_t[:, :-1]))
                ey_l = Ey_t[1, :].copy(); ey_r = Ey_t[-2, :].copy()
                ex_b = Ex_t[:, 1].copy(); ex_tp = Ex_t[:, -2].copy()
                Ex_t[:, 1:-1] += Ca_ex[:, 1:-1]*(Hz_t[:, 1:] - Hz_t[:, :-1])
                Ey_t[1:-1, :] += -Ca_ey[1:-1, :]*(Hz_t[1:, :] - Hz_t[:-1, :])
                if src_active:
                    if vcsel_tilt != 0.0:
                        Ey_t[src_ix, :] += envelope*xp.sin(2*np.pi*f0*t_phys + phase_y_ey)*src_p_ey
                    else:
                        Ey_t[src_ix, :] += envelope*float(np.sin(2*np.pi*f0*t_phys))*src_p_ey
                Ey_t[0, :] = ey_l + mur*(Ey_t[1, :] - Ey_t[0, :])
                Ey_t[-1, :] = ey_r + mur*(Ey_t[-2, :] - Ey_t[-1, :])
                Ex_t[:, 0] = ex_b + mur*(Ex_t[:, 1] - Ex_t[:, 0])
                Ex_t[:, -1] = ex_tp + mur*(Ex_t[:, -2] - Ex_t[:, -1])
                field = _tm_node_field(Ey_t)
            else:
                Hx -= Ch*(Ez[:, 1:] - Ez[:, :-1])
                Hy += Ch*(Ez[1:, :] - Ez[:-1, :])
                ex1, ex2 = Ez[1, :].copy(), Ez[-2, :].copy()
                ey1, ey2 = Ez[:, 1].copy(), Ez[:, -2].copy()
                Ez[1:-1, 1:-1] = (Ce_E[1:-1, 1:-1]*Ez[1:-1, 1:-1]
                                  + Ce_H[1:-1, 1:-1]*((Hy[1:, 1:-1] - Hy[:-1, 1:-1])
                                                      - (Hx[1:-1, 1:] - Hx[1:-1, :-1])))
                if src_active:
                    if vcsel_tilt != 0.0:
                        Ez[src_ix, :] += envelope*xp.sin(2*np.pi*f0*t_phys + phase_y)*src_p
                    else:
                        Ez[src_ix, :] += envelope*float(np.sin(2*np.pi*f0*t_phys))*src_p
                Ez[0, :] = ex1 + mur*(Ez[1, :] - Ez[0, :]); Ez[-1, :] = ex2 + mur*(Ez[-2, :] - Ez[-1, :])
                Ez[:, 0] = ey1 + mur*(Ez[:, 1] - Ez[:, 0]); Ez[:, -1] = ey2 + mur*(Ez[:, -2] - Ez[:, -1])
                field = Ez
            for (xg, which) in ((0.1*L_um, 'in'), (0.9*L_um, 'out')):
                if x_start_um <= xg <= x_end_um:
                    cl = int(round((xg - x_start_um)*1e-6/dx))
                    a = max(0, cl-iy_h_det); b = min(Nx, cl+iy_h_det)
                    v = float(xp.sum(field[a:b, iy_lo:iy_hi]**2))*dx*dx
                    if which == 'in':
                        P_in += v
                    else:
                        P_out += v
            if save_frames and ez_mm is not None and ((n+1) % snap_every == 0) and k_fr < ez_mm.shape[0]:
                ez_mm[k_fr] = to_np(field).astype(np.float32)
                frames.append({'x_start': x_start_um,
                               'x_end': x_end_um, 'y_start': -TEAR_BUFFER*1e6,
                               'y_end': (T_WG + AIR_BUFFER)*1e6, 'slide_i': slide_i,
                               'step': n+1, 'time': time.time()-t_start})
                k_fr += 1
        print(f'  Slide done: max|E|={float(xp.max(xp.abs(field))):.3e}')
    if ez_mm is not None:
        ez_mm.flush(); del ez_mm
    total_time = time.time() - t_total
    Tr = P_out/max(P_in, 1e-30)
    print(f'\n[Result] {label} [sliding]: total {total_time:.0f}s  Transmission={Tr:.4f}')
    return _result_dict(label, P_in, P_out, Tr, frames, total_time, bead_d_um,
                        wg_mat, bead_mat, wg_n, bead_n, wg_thickness_um, lambda_nm,
                        vcsel_waist, vcsel_tilt, vcsel_offset, source_type, 'sliding', place_bead, n_aq_l, n_mu_l, n_co_l,
                        frames_memmap=mm_path, t_aq_um=t_aq_um, t_mu_um=t_mucin_um,
                        t_lip_um=t_lipid_um, bead_x_um=bx_um,
                        x_wg_start_um=(input_gap_um or 0.0), x_wg_end_um=L_um,
                        polarization=_pol)


def run_beads_stitched(wg_mat, bead_mat, bead_d_um, dx_nm, save_frames, n_snapshots,
                       wg_thickness_um, lambda_nm, vcsel_waist, vcsel_tilt,
                       vcsel_offset, source_type, window_w_um, slide_um, place_bead=True,
                       t_aqueous_um=None, t_mucin_um=None, t_lipid_um=0.0, input_gap_um=0.0,
                       length_um=None, bead_x_um=None, wg_n=None, bead_n=None,
                       n_aqueous=None, n_mucin=None, n_cornea=None, n_lipid=None,
                       polarization='s'):
    """Voller gefuellter CW-Waveguide per GEBIETS-ZERLEGUNG (Hard-Overlap-Handoff).
    Jedes Fenster wird bis zum Steady-State gerechnet; im Ueberlappbereich wird das
    zeitharmonische Feld (komplexe Amplitude, DFT bei f0) des Vorgaengers hart
    eingepraegt -> treibt das naechste Fenster mit dem exakt einlaufenden Feld.
    Ergebnis: full-domain Steady-State, aber immer nur EIN Fenster im Speicher.
    v1 - MUSS auf Windows gegen einen Full-Domain-Lauf gleicher (kurzer) Laenge
    validiert werden (Felder muessen uebereinstimmen)."""
    _pol = 'p' if str(polarization).lower() in ('p', 'tm') else 's'
    t_total = time.time()
    t_aq_um = t_aqueous_um if t_aqueous_um is not None else T_AQ*1e6
    wg_n = wg_n if wg_n is not None else n_at(wg_mat, lambda_nm)
    bead_n = bead_n if bead_n is not None else n_at(bead_mat, lambda_nm)
    n_aq_l = n_aqueous if n_aqueous is not None else n_at('aqueous', lambda_nm)
    n_mu_l = n_mucin if n_mucin is not None else n_at('mucin', lambda_nm)
    n_co_l = n_cornea if n_cornea is not None else n_at('cornea', lambda_nm)
    n_lip_l = n_lipid if n_lipid is not None else N_LIPID
    L = (length_um*1e-6) if length_um is not None else L_DEMO
    L_um = L*1e6
    T_WG = wg_thickness_um*1e-6; LAM = lambda_nm*1e-9
    bx_um = bead_x_um if bead_x_um is not None else L_um/2.0
    label = f'WG-{wg_mat}_Bead-{bead_mat}_d{bead_d_um:g}_L{lambda_nm:g}' + ('' if place_bead else '_ref')
    dx = dx_nm*1e-9
    dt = 0.5*dx/(C0*np.sqrt(2))
    H_total = T_WG + AIR_BUFFER + TEAR_BUFFER
    Ny = int(round(H_total/dx))
    Nx_win = int(round(window_w_um*1e-6/dx))
    S_cells = int(round(slide_um*1e-6/dx))
    O_cells = Nx_win - S_cells                     # Ueberlappung (Yee-Zellen)
    Nx_full = int(round(L/dx))
    if O_cells < 8:
        print(f'  [WARN] Ueberlappung {O_cells} Zellen sehr klein -> window_w >> slide waehlen.')
        O_cells = max(O_cells, 1)
    f0 = C0/LAM; omega = 2*np.pi*f0; sigma_t = 4/(2*np.pi*f0); k0 = 2*np.pi/LAM
    transit = window_w_um*1e-6*wg_n/C0
    steps_win = int(3.0*transit/dt) + 400          # ~3 Transitzeiten -> Steady-State
    steps_per_T = max(4, int(round((1.0/f0)/dt)))
    n_acc = steps_per_T*8                           # DFT ueber 8 Perioden am Ende
    n_win = (int(np.ceil((Nx_full - Nx_win)/S_cells)) + 1) if Nx_full > Nx_win else 1
    print(f'\n===== Bead-Demo [stitch/Gebiets-Zerlegung]: {label} =====')
    print(f'Laenge {L_um:g}um -> {n_win} Fenster a {window_w_um}um (Ueberlapp {O_cells*dx*1e6:.1f}um), '
          f'Grid/Fenster {Nx_win}x{Ny}, Steps/Fenster {steps_win}')
    Ch = xp.float32(dt/(MU0*dx))
    mur = xp.float32((C0*dt - dx)/(C0*dt + dx))
    iy_src, src_p, ys_idx = _src_setup(Ny, dx, T_WG, vcsel_offset, vcsel_waist)
    phase_y = float(np.sin(np.radians(vcsel_tilt)))*k0*(ys_idx - iy_src)*dx
    src_p_ey = 0.5*(src_p[:-1] + src_p[1:])
    phase_y_ey = 0.5*(phase_y[:-1] + phase_y[1:])
    src_ix = 5
    _Nyc = (Ny-1) if _pol == 'p' else Ny                  # y-Groesse der Hauptkomp.
    print(f'Polarisation: {_pol}-Pol ({"TM/Ey" if _pol=="p" else "TE/Ez"})')
    # v2: weiche cosinus-getaperte Einpraegung des Overlaps (1 an linker Kante ->
    # 0 an Innenkante) statt hartem Dirichlet -> vermeidet Naht/Reflexion am
    # Uebergang zur frei gerechneten Region.
    _itap = np.arange(O_cells)
    _wtap = xp.asarray((0.5*(1.0 + np.cos(np.pi*_itap/max(O_cells-1, 1)))
                        ).astype(np.float32))[:, None]
    Efull = np.zeros((Nx_full, _Nyc), dtype=np.complex64)  # assemblierte Amplitude
    Ehand = None                                          # Overlap-Amplitude Vorgaenger
    for w in range(n_win):
        gx0 = w*S_cells
        x_start_um = gx0*dx*1e6
        eps_r, sig = build_materials(Nx_win, Ny, dx, wg_n, bead_n, T_WG, bx_um, bead_d_um,
                                     x_start_um, place_bead, n_aq=n_aq_l, n_mucin=n_mu_l,
                                     n_cornea=n_co_l, t_aq_um=t_aq_um, t_mu_um=t_mucin_um,
                                     t_lip_um=t_lipid_um, n_lip=n_lip_l,
                                     input_gap_um=input_gap_um, length_um=length_um)
        Ce_E, Ce_H = _coeffs(eps_r, sig, dt, dx)
        if _pol == 'p':
            Ca_ex, Ca_ey, Ch_tm = _tm_coeffs(eps_r, dt, dx)
        del eps_r, sig
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
        Ez = xp.zeros((Nx_win, Ny), dtype=xp.float32)
        Hx = xp.zeros((Nx_win, Ny-1), dtype=xp.float32)
        Hy = xp.zeros((Nx_win-1, Ny), dtype=xp.float32)
        Ex_t = xp.zeros((Nx_win-1, Ny), dtype=xp.float32)
        Ey_t = xp.zeros((Nx_win, Ny-1), dtype=xp.float32)
        Hz_t = xp.zeros((Nx_win-1, Ny-1), dtype=xp.float32)
        acc = np.zeros((Nx_win, _Nyc), dtype=np.complex64); acc_n = 0
        _drive = xp.asarray(Ehand.astype(np.complex64)) if (w > 0 and Ehand is not None) else None
        for n in range(steps_win):
            t_phys = n*dt
            env = float(1 - np.exp(-((t_phys/(2*sigma_t))**2)))    # sanfter Anlauf
            if _pol == 'p':
                # --- TM (Hz, Ex, Ey) ---
                Hz_t += -Ch_tm*((Ey_t[1:, :] - Ey_t[:-1, :]) - (Ex_t[:, 1:] - Ex_t[:, :-1]))
                ey_r = Ey_t[-2, :].copy(); ex_b = Ex_t[:, 1].copy(); ex_tp = Ex_t[:, -2].copy()
                ey_l = Ey_t[1, :].copy()
                Ex_t[:, 1:-1] += Ca_ex[:, 1:-1]*(Hz_t[:, 1:] - Hz_t[:, :-1])
                Ey_t[1:-1, :] += -Ca_ey[1:-1, :]*(Hz_t[1:, :] - Hz_t[:-1, :])
                if w == 0:
                    if vcsel_tilt != 0.0:
                        Ey_t[src_ix, :] += env*xp.sin(2*np.pi*f0*t_phys + phase_y_ey)*src_p_ey
                    else:
                        Ey_t[src_ix, :] += env*float(np.sin(2*np.pi*f0*t_phys))*src_p_ey
                    Ey_t[0, :] = ey_l + mur*(Ey_t[1, :] - Ey_t[0, :])
                else:
                    _din = env*xp.real(_drive*np.exp(1j*omega*t_phys))
                    Ey_t[:O_cells, :] = _wtap*_din + (1.0 - _wtap)*Ey_t[:O_cells, :]
                Ey_t[-1, :] = ey_r + mur*(Ey_t[-2, :] - Ey_t[-1, :])
                Ex_t[:, 0] = ex_b + mur*(Ex_t[:, 1] - Ex_t[:, 0])
                Ex_t[:, -1] = ex_tp + mur*(Ex_t[:, -2] - Ex_t[:, -1])
                if n >= steps_win - n_acc:
                    acc += (to_np(Ey_t)*np.exp(-1j*omega*t_phys)).astype(np.complex64)
                    acc_n += 1
            else:
                # --- TE (Ez, Hx, Hy) ---
                Hx -= Ch*(Ez[:, 1:] - Ez[:, :-1])
                Hy += Ch*(Ez[1:, :] - Ez[:-1, :])
                ex2 = Ez[-2, :].copy(); ey1 = Ez[:, 1].copy(); ey2 = Ez[:, -2].copy()
                ex1 = Ez[1, :].copy()
                Ez[1:-1, 1:-1] = (Ce_E[1:-1, 1:-1]*Ez[1:-1, 1:-1]
                                  + Ce_H[1:-1, 1:-1]*((Hy[1:, 1:-1] - Hy[:-1, 1:-1])
                                                      - (Hx[1:-1, 1:] - Hx[1:-1, :-1])))
                if w == 0:
                    if vcsel_tilt != 0.0:
                        Ez[src_ix, :] += env*xp.sin(2*np.pi*f0*t_phys + phase_y)*src_p
                    else:
                        Ez[src_ix, :] += env*float(np.sin(2*np.pi*f0*t_phys))*src_p
                    Ez[0, :] = ex1 + mur*(Ez[1, :] - Ez[0, :])    # linker Mur nur Fenster 0
                else:
                    _din = env*xp.real(_drive*np.exp(1j*omega*t_phys))
                    Ez[:O_cells, :] = _wtap*_din + (1.0 - _wtap)*Ez[:O_cells, :]
                Ez[-1, :] = ex2 + mur*(Ez[-2, :] - Ez[-1, :])
                Ez[:, 0] = ey1 + mur*(Ez[:, 1] - Ez[:, 0])
                Ez[:, -1] = ey2 + mur*(Ez[:, -2] - Ez[:, -1])
                if n >= steps_win - n_acc:
                    acc += (to_np(Ez)*np.exp(-1j*omega*t_phys)).astype(np.complex64)
                    acc_n += 1
        Ew = (2.0/max(acc_n, 1))*acc                              # komplexe Amplitude
        Ehand = Ew[S_cells:S_cells+O_cells, :].copy()             # Overlap fuers naechste
        if w == 0:
            g_lo, l_lo = 0, 0
        else:
            g_lo, l_lo = gx0 + O_cells, O_cells
        g_hi = min(gx0 + Nx_win, Nx_full)
        l_hi = l_lo + (g_hi - g_lo)
        if g_hi > g_lo:
            Efull[g_lo:g_hi, :] = Ew[l_lo:l_hi, :]
        print(f'[Fenster {w+1}/{n_win}] x0={x_start_um:.0f}um  max|E|={np.abs(Ew).max():.3e}')

    # Ausgabe-Frames: n_snapshots reale Phasen ueber eine Periode (CW-Animation)
    total_time = time.time() - t_total
    frames = []
    ez_mm = None; mm_path = None
    nfr = max(1, n_snapshots)
    if save_frames:
        os.makedirs('results', exist_ok=True)
        _safe = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in label)
        mm_path = os.path.join('results', f'_stream_beads_{_safe}_stitch.npy')
        ez_mm = np.lib.format.open_memmap(mm_path, mode='w+', dtype=np.float32,
                                          shape=(nfr, Nx_full, Ny))
        def _node2d(F):                               # (Nx,Ny-1)->(Nx,Ny) fuer TM
            o = np.zeros((F.shape[0], F.shape[1]+1), dtype=np.float32)
            o[:, 1:-1] = 0.5*(F[:, :-1] + F[:, 1:]); o[:, 0] = F[:, 0]; o[:, -1] = F[:, -1]
            return o
        for k in range(nfr):
            tk = (k/nfr)*(1.0/f0)
            _fr = np.real(Efull*np.exp(1j*omega*tk)).astype(np.float32)
            ez_mm[k] = _node2d(_fr) if _pol == 'p' else _fr
            frames.append({'x_start': 0.0, 'x_end': L_um, 'y_start': -TEAR_BUFFER*1e6,
                           'y_end': (T_WG + AIR_BUFFER)*1e6, 'slide_i': 0,
                           'step': k, 'time': tk})
        ez_mm.flush(); del ez_mm
    # Sensor-Kennzahlen aus der assemblierten Amplitude (|E|^2 integriert)
    iy_lo = int(round((0.0 + TEAR_BUFFER)/dx)); iy_hi = int(round((T_WG + TEAR_BUFFER)/dx))
    ix_in = int(round(0.1*Nx_full)); ix_out = int(round(0.9*Nx_full))
    P_in = float(np.sum(np.abs(Efull[ix_in, iy_lo:iy_hi])**2))
    P_out = float(np.sum(np.abs(Efull[ix_out, iy_lo:iy_hi])**2))
    Tr = P_out/max(P_in, 1e-30)
    print(f'\n[Result] {label} [stitch]: total {total_time:.0f}s  Transmission~{Tr:.4f}')
    res = _result_dict(label, P_in, P_out, Tr, frames, total_time, bead_d_um,
                       wg_mat, bead_mat, wg_n, bead_n, wg_thickness_um, lambda_nm,
                       vcsel_waist, vcsel_tilt, vcsel_offset, source_type, 'stitch', place_bead,
                       n_aq_l, n_mu_l, n_co_l, frames_memmap=mm_path, t_aq_um=t_aq_um,
                       t_mu_um=t_mucin_um, t_lip_um=t_lipid_um, bead_x_um=bx_um,
                       x_wg_start_um=(input_gap_um or 0.0), x_wg_end_um=L_um,
                       polarization=_pol)
    # Stitch = eingeschwungener Hauptsatz; 'einlaufend' erzeugt der Analyzer
    # illustrativ aus diesem CW-Feld (kein echter Transient vorhanden).
    res['view_primary'] = 'cw'
    res['cw_only'] = True
    return res


def _result_dict(label, P_in, P_out, Tr, frames, total_time, bead_d_um,
                 wg_mat, bead_mat, wg_n, bead_n, wg_th, lam, waist, tilt, off, src, method, place_bead=True,
                 n_aq=N_AQ, n_mucin=N_MUCIN, n_cornea=N_CORNEA, frames_memmap=None, t_aq_um=None,
                 t_mu_um=None, t_lip_um=0.0, bead_x_um=None, x_wg_start_um=0.0, x_wg_end_um=None,
                 polarization='s'):
    _t_aq = (t_aq_um*1e-6) if t_aq_um is not None else T_AQ
    _t_mu = (t_mu_um*1e-6) if t_mu_um is not None else T_MU
    _t_lip = (t_lip_um or 0.0)*1e-6
    return dict(scenario=label, P_in=P_in, P_out=P_out, transmission=Tr,
                frames=frames, frames_memmap=frames_memmap,
                total_time=total_time, layers=(_t_lip, _t_aq, _t_mu, n_aq), n_aq=n_aq, n_mucin=n_mucin, n_cornea=n_cornea,
                wg_material=wg_mat, bead_material=bead_mat, wg_n=wg_n, bead_n=bead_n,
                bead_diameter_um=(bead_d_um if place_bead else 0.0),
                bead_x_um=(bead_x_um if bead_x_um is not None else BEAD_X_UM),
                x_wg_start_um=x_wg_start_um, x_wg_end_um=x_wg_end_um,
                wg_thickness_um=wg_th, lambda_nm=lam, vcsel_waist=waist,
                vcsel_tilt=tilt, vcsel_offset=off, source_type=src, method=method,
                polarization=polarization)


def save_results(result, out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{result['scenario']}_{result.get('method','full')}"
    summary = {k: v for k, v in result.items() if k != 'frames'}
    with open(f'{out_dir}/beads_{tag}.pkl', 'wb') as f:
        pickle.dump(summary, f)
    print(f'  Saved: {out_dir}/beads_{tag}.pkl')
    frames = result.get('frames')
    if not frames:
        return
    if GPU_AVAILABLE:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    n_fr = len(frames)
    mm_path = result.get('frames_memmap')
    ez_stream = None
    _mm_full = None
    if mm_path and os.path.exists(mm_path):
        # Streamed: Frames liegen als Disk-Memmap vor -> nicht in den RAM laden.
        _mm_full = np.load(mm_path, mmap_mode='r')
        ez_stream = _mm_full[:n_fr]
        shape = ez_stream.shape[1:]
    else:
        shape = frames[0]['Ez'].shape
    ezs = None; used_stride = None
    # NUR die grosse x-Kante (Propagation) deckeln; y (Querschnitt, klein) bleibt
    # VOLL -> die Mode bleibt vertikal scharf (kein blockiges Downsampling).
    _MAXX = 5000     # max. gespeicherte x-Punkte (Welle ueber lange Strecke sonst riesig)
    _MAXY = 2000     # y nur deckeln, falls wirklich extrem hoch (sonst voll)
    _sy = 1
    for _s in (1, 2, 4, 8):
        if shape[1]//_s <= _MAXY:
            _sy = _s
            break
    if ez_stream is not None:
        _sx = 1
        for _s in (1, 2, 3, 4, 6, 8, 12, 16, 32):
            if shape[0]//_s <= _MAXX:
                _sx = _s
                break
        used_stride = _sx
        ezs = ez_stream[:, ::_sx, ::_sy]
    else:
        for _sx in (1, 2, 3, 4, 6, 8, 12, 16, 32):
            if shape[0]//_sx > _MAXX:
                continue                       # x zu breit fuer Viewer -> groeber
            try:
                buf = np.empty((n_fr, shape[0]//_sx, shape[1]//_sy), dtype=np.float32)
                for i, fr in enumerate(frames):
                    buf[i] = fr['Ez'][::_sx, ::_sy]
                ezs = buf; used_stride = _sx
                break
            except MemoryError:
                buf = None
                print(f'  [WARN] x-stride={_sx} passt nicht in RAM, versuche groeber...')
        if ezs is None:
            print('  [FEHLER] Frames passen nicht in den RAM - NPZ nicht gespeichert.')
            return
        for fr in frames:
            fr['Ez'] = None
    meta = np.array([(fr['x_start'], fr['x_end'], fr['y_start'], fr['y_end'],
                      fr['slide_i'], fr['time'], fr.get('step', 0)) for fr in frames], dtype=np.float64)
    # --- zweiter (eingeschwungener) Frame-Satz + Ansichts-Flags -----------------
    cw_kw = {}; _cw_mm = None; cw_path = result.get('cw_frames_memmap')
    cw_frames = result.get('cw_frames') or []
    if cw_path and os.path.exists(cw_path) and cw_frames:
        _cw_mm = np.load(cw_path, mmap_mode='r')
        ezs_cw = _cw_mm[:len(cw_frames), ::(used_stride or 1), ::_sy]
        meta_cw = np.array([(fr['x_start'], fr['x_end'], fr['y_start'], fr['y_end'],
                             fr['slide_i'], fr['time'], fr.get('step', 0)) for fr in cw_frames],
                           dtype=np.float64)
        cw_kw = dict(Ez_cw=ezs_cw, meta_cw=meta_cw)
    _cwonly = bool(result.get('cw_only', False)) or result.get('method') == 'stitch'
    _flags = dict(has_transient=np.array([0 if _cwonly else 1]),
                  has_cw=np.array([1 if (_cwonly or cw_kw) else 0]),
                  cw_only=np.array([1 if _cwonly else 0]),
                  view_primary=np.array([str(result.get('view_primary', 'transient'))]))
    npz = f'{out_dir}/beads_{tag}_frames.npz'
    np.savez_compressed(npz, Ez=ezs, meta=meta, scenario=result['scenario'],
                        **cw_kw, **_flags,
                        layers=np.array(result['layers'], dtype=np.float64),
                        material_names=np.array(['WG', 'Bead', 'Aqueous', 'Mucin', 'Cornea']),
                        material_indices=np.array([result['wg_n'], result['bead_n'],
                                                   result['n_aq'], result['n_mucin'], result['n_cornea']], dtype=np.float64),
                        bead_x_um=np.array([result.get('bead_x_um', BEAD_X_UM)]),
                        bead_diameter_um=np.array([result.get('bead_diameter_um', 0.0)]),
                        t_wg_um=np.array([result.get('wg_thickness_um', 5.0)]),
                        x_wg_start_um=np.array([result.get('x_wg_start_um', 0.0) or 0.0]),
                        x_wg_end_um=np.array([result.get('x_wg_end_um', 0.0) or 0.0]),
                        polarization=np.array([str(result.get('polarization', 's'))]),
                        lam_nm=np.array([result.get('lambda_nm', 850.0)]))
    print(f'  Saved: {npz} (stride={used_stride}, {ezs.nbytes/1e6:.1f} MB)'
          + ('  +CW-Satz' if cw_kw else '') + ('  [cw_only]' if _cwonly else ''))
    del ezs
    # CW-Memmap schliessen + Temp entfernen
    if _cw_mm is not None:
        try: _cw_mm._mmap.close()
        except Exception: pass
        try: os.remove(cw_path)
        except OSError: pass
    # Temporaeres Stream-Memmap aufraeumen (falls gestreamt wurde). Auf Windows
    # muss das Memmap-Handle explizit geschlossen werden, sonst schlaegt os.remove
    # fehl und die (grosse) Temp-Datei bleibt liegen.
    if mm_path and os.path.exists(mm_path):
        try:
            _mm_full._mmap.close()
        except Exception:
            pass
        try:
            del _mm_full, ez_stream, ezs
        except Exception:
            pass
        import gc
        gc.collect()
        try:
            os.remove(mm_path)
        except OSError:
            pass
