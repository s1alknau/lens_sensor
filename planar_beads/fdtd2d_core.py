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
from common.physics import (C0, EPS0, MU0, N_AIR, DISPERSION, n_at, cornea_default,  # noqa: E402
                            loss_sigma, core_source_mask, SRC_CORE_RAMP_UM)
from common.subpixel import circle_fraction, blend, cell_average_column  # noqa: E402
from common.flux import PlaneFluxDFT, pairs_2d, pairs_3d, guard_in_plane  # noqa: E402,F401
BEAD_SUBPIXEL = True     # Bead-Rand per Volumenanteil glaetten (False = harte Maske)
from planar_beads.boundaries import (make_boundary, mur1_coeff, mur_speeds, MurSet,     # noqa: E402
                                     CPML_TE, CPML_TM)

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
# Absorption je Schicht (Extinktion kappa, n = n' + i kappa), gesetzt vom Treiber
# (run_simulation: args.kappa). Schluessel: wg, lip, aq, mu, co, bead. Leer = verlustfrei.
KAPPA = {}
# Quelle nur im Kern einpraegen (--source-core-only), gesetzt vom Treiber
SRC_CORE_ONLY = False
LAM_LOSS = 850e-9       # Wellenlaenge, bei der kappa -> sigma umgerechnet wird (je Lauf gesetzt)
T_MU = 0.5e-6            # Mucin-Dicke (physiologisch ~0.2-0.5 um)
T_AQ = 3.5e-6            # Aqueous(Wasser)-Dicke (physiologisch ~3-4 um; > typ. Bead)


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
                    length_um=None, bead_y_um=None, curved=False, r_bend_m=8.3e-3,
                    tm_patch=None):
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
        print(f'  [WARN] Bead d={bead_d_um}um > Aqueous {t_aq*1e6:g}um -> extends into mucin. '
              f'Increase T_AQUEOUS.')
    # Absorption (KAPPA): intern mit komplexem n rechnen, am Ende eps' + sigma trennen
    _k = dict(KAPPA or {})
    lossy = any(v > 0 for v in _k.values())
    if lossy:
        wg_n = complex(wg_n, _k.get('wg', 0.0)); bead_n_c = complex(bead_n, _k.get('bead', 0.0))
        n_aq = complex(n_aq, _k.get('aq', 0.0)); n_mucin = complex(n_mucin, _k.get('mu', 0.0))
        n_lip = complex(n_lip, _k.get('lip', 0.0))
        _k_co = _k.get('co', 0.0)
    else:
        bead_n_c = bead_n; _k_co = 0.0
    _edt = np.complex128 if lossy else np.float64
    eps = np.full((Nx, Ny), N_AIR**2, dtype=np.complex128 if lossy else np.float32)
    sig = np.zeros((Nx, Ny), dtype=np.float32)
    ys = -TEAR_BUFFER + np.arange(Ny)*dx
    y_lip_bot = -t_lip
    y_aq_bot = y_lip_bot - t_aq
    y_mu_bot = y_aq_bot - t_mu
    # SUBSTRAT (unter y=0) ueber die GANZE Laenge: Wasser ist ueberall unter dem Slab
    # und daneben/dahinter. Ueber y=0 ist standardmaessig LUFT.
    def _eps_y(y, core):
        """Schichtstapel eps(y) (exakt, ohne Gitter) - mit oder ohne Kern."""
        e = np.full(y.shape, N_AIR**2, dtype=_edt)
        mc = y < y_mu_bot
        if hasattr(n_cornea, 'at_depth'):            # geschichtete Cornea (Epithel, Stroma ...)
            e[mc] = (n_cornea.at_depth((y_mu_bot - y[mc])*1e6) + 1j*_k_co)**2 if lossy                 else n_cornea.at_depth((y_mu_bot - y[mc])*1e6)**2
        else:
            e[mc] = (float(n_cornea) + 1j*_k_co)**2 if lossy else n_cornea**2
        e[(y >= y_mu_bot) & (y < y_aq_bot)] = n_mucin**2
        e[(y >= y_aq_bot) & (y < y_lip_bot)] = n_aq**2
        if t_lip > 1e-12:
            e[(y >= y_lip_bot) & (y < 0.0)] = n_lip**2
        if core:
            e[(y >= 0.0) & (y < wg_top)] = wg_n**2
        return e
    # Subpixel an den Schichtgrenzen: jede Zelle = Mittel ueber ihre Hoehe (Review F12)
    col = cell_average_column(ys, dx, lambda y: _eps_y(y, False))
    col_core = cell_average_column(ys, dx, lambda y: _eps_y(y, True))
    eps[:, :] = col[None, :]
    # PMMA-KERN nur wo der Slab ist: x in [x_wg_start, x_wg_end], y in [0, t_wg].
    # Neben/hinter dem Slab ist ueber dem Wasser Luft -> echte Facetten an beiden Enden.
    xs_abs = x_start_um*1e-6 + np.arange(Nx)*dx
    in_wg_x = (xs_abs >= x_wg_start) & (xs_abs <= x_wg_end)
    if in_wg_x.any():
        eps[in_wg_x, :] = col_core[None, :]
    if tm_patch is not None and not curved:
        # TM: Ey steht SENKRECHT auf den Schichten -> harmonisches Mittel auf den
        # Ey-Positionen (y_j + dy/2); _apply_bead_tm setzt es in Ca_ey ein.
        tm_patch['layer'] = dict(
            in_wg_x=in_wg_x,
            h_out=cell_average_column(ys[:-1], dx, lambda y: _eps_y(y, False).real,
                                      harmonic=True, shift=0.5*dx),
            h_core=cell_average_column(ys[:-1], dx, lambda y: _eps_y(y, True).real,
                                       harmonic=True, shift=0.5*dx))
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
            if BEAD_SUBPIXEL:
                # Subpixel (Review F12): Randzellen bekommen den Flaechenanteil des Beads
                # -> Position/Abstand kontinuierlich statt auf das Gitter eingerastet.
                jy0 = max(0, int((yc - r - ys[0])/dx) - 2)
                jy1 = min(Ny, int((yc + r - ys[0])/dx) + 3)
                if ix1 > ix0 and jy1 > jy0:
                    if tm_patch is not None and not curved:
                        # fuer die TM-Tensor-Mittelung: eps OHNE Bead im Bead-Rahmen
                        tm_patch.update(eps_bg=eps[ix0:ix1, jy0:jy1].real.astype(np.float64),
                                        ix0=ix0, jy0=jy0, xs=xs[ix0:ix1], ys=ys[jy0:jy1],
                                        xc=xc, yc=yc, r=r, n=bead_n)
                    blend(eps[ix0:ix1, jy0:jy1],
                          circle_fraction(xs[ix0:ix1], ys[jy0:jy1], xc, yc, r, dx), bead_n_c)
            else:                                    # alte harte Maske (Vergleich)
                for i in range(ix0, ix1):
                    dxx = xs[i] - xc
                    if abs(dxx) > r:
                        continue
                    dy = np.sqrt(max(0.0, r*r - dxx*dxx))
                    mask = (ys >= yc - dy) & (ys <= yc + dy)
                    eps[i, mask] = bead_n_c**2
    if curved:
        # Echte Kruemmung der Kontaktlinse via KONFORMER ABBILDUNG des gebogenen
        # Wellenleiters auf einen geraden: n_eq = n*(1 + u/R), u = y - Kernmitte
        # (Radius R_BEND); u>0 = aussen (Luftseite). eps_eq = eps*(1 + u/R)^2.
        u = ys - wg_top/2.0                          # radialer Versatz von der Kernmitte (m)
        metric = (1.0 + u/r_bend_m).astype(np.float32)**2
        eps = eps*metric[None, :]
    if lossy:                                        # eps' = n'^2 - k^2, sigma = w0 eps0 eps''
        sig = loss_sigma(eps.imag, LAM_LOSS).astype(np.float32)
        eps = eps.real.astype(np.float32)
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


def _split_eps(e):
    """CAD-eps (reell oder komplex bei absorbierenden Koerpern) -> (eps', sigma) auf dem Device."""
    e = np.asarray(to_np(e))
    if np.iscomplexobj(e):
        return (xp.asarray(e.real.astype(np.float32)),
                xp.asarray(loss_sigma(e.imag, LAM_LOSS).astype(np.float32)))
    er = xp.asarray(e.astype(np.float32))
    return er, xp.zeros_like(er)


def _tm_loss(eps_r, sig, Ca_ex, Ca_ey, dt):
    """Absorption fuer TM (Ex, Ey): semi-implizit wie TE (_coeffs):
        E^{n+1} = D E^n + Ca/(1+a) curl H,  a = sigma dt/(2 eps0 eps'),  D = (1-a)/(1+a).
    Skaliert Ca_ex/Ca_ey IN PLACE (nach dem Bead-Tensor) und liefert (D_ex, D_ey) oder
    (None, None) ohne Verlust. D wird VOR dem E-Update (nach dem H-Update) angewandt."""
    if sig is None or not bool(xp.any(sig > 0)):
        return None, None
    s_ex = 0.5*(sig[:-1, :] + sig[1:, :]); e_ex = 0.5*(eps_r[:-1, :] + eps_r[1:, :])
    s_ey = 0.5*(sig[:, :-1] + sig[:, 1:]); e_ey = 0.5*(eps_r[:, :-1] + eps_r[:, 1:])
    a_ex = s_ex*dt/(2*EPS0*e_ex); a_ey = s_ey*dt/(2*EPS0*e_ey)
    Ca_ex /= (1 + a_ex).astype(xp.float32); Ca_ey /= (1 + a_ey).astype(xp.float32)
    return ((1 - a_ex)/(1 + a_ex)).astype(xp.float32), ((1 - a_ey)/(1 + a_ey)).astype(xp.float32)


def _apply_bead_tm(bp, Ca_ex, Ca_ey, dt, dx):
    """Anisotrope Subpixel-Mittelung (Farjadpour et al., Opt. Lett. 31, 2972 (2006);
    Review F12) fuer die TM-Komponenten Ex/Ey am Bead-Rand: Feldanteil SENKRECHT zur
    Oberflaeche sieht das harmonische Mittel <1/eps>, der TANGENTIALE das arithmetische
    1/<eps>:  (1/eps)_ii = n_i^2 <1/eps> + (1 - n_i^2)/<eps>   (n = Oberflaechennormale).
    TE (Ez, immer tangential an der 2D-Kreisscheibe) braucht nur das arithmetische
    Mittel - das macht bereits build_materials."""
    if not bp:
        return
    c0 = dt/(EPS0*dx)
    lp = bp.get('layer')
    h = None
    if lp is not None:                               # Schichtgrenzen: Ey harmonisch
        h = np.where(lp['in_wg_x'][:, None], lp['h_core'][None, :], lp['h_out'][None, :])
        Ca_ey[...] = xp.asarray((c0/h).astype(np.float32))
    if 'eps_bg' not in bp:                           # kein Bead -> fertig
        return
    from common.subpixel import circle_fraction
    e_bg, ix0, jy0 = bp['eps_bg'], bp['ix0'], bp['jy0']
    xs, ys, xc, yc, r, eb = bp['xs'], bp['ys'], bp['xc'], bp['yc'], bp['r'], bp['n']**2
    c = dt/(EPS0*dx)

    def tensor(xp_, yp_, ebg_a, ebg_h, comp):
        """ebg_a: arithmetisches, ebg_h: harmonisches Hintergrund-Mittel an der
        Komponentenposition (Anteil tangential bzw. senkrecht zur Bead-Oberflaeche)."""
        f = circle_fraction(xp_, yp_, xc, yc, r, dx)
        X = xp_[:, None] - xc; Y = yp_[None, :] - yc
        d = np.maximum(np.sqrt(X*X + Y*Y), 1e-30)
        nc2 = (X/d)**2 if comp == 'x' else (Y/d)**2
        mean = (1 - f)*ebg_a + f*eb
        hmean = (1 - f)/ebg_h + f/eb
        return f, nc2*hmean + (1 - nc2)/mean

    # Ex an (i+1/2, j): eps_bg in x gemittelt (wie _tm_coeffs)
    f, inv = tensor(xs[:-1] + 0.5*dx, ys, 0.5*(e_bg[:-1] + e_bg[1:]),
                    1.0/(0.5*(1.0/e_bg[:-1] + 1.0/e_bg[1:])), 'x')
    sl = (slice(ix0, ix0 + inv.shape[0]), slice(jy0, jy0 + inv.shape[1]))
    blk = to_np(Ca_ex[sl]).astype(np.float64)
    m = f > 0
    blk[m] = c*inv[m]
    Ca_ex[sl] = xp.asarray(blk.astype(np.float32))
    # Ey an (i, j+1/2): Normalkomponente an den Schichten -> harmonisches Hintergrund-
    # Mittel, wenn bekannt exakt aus dem Schichtstapel (Beruehrpunkt Bead/Kern bei y=0)
    eh = 1.0/(0.5*(1.0/e_bg[:, :-1] + 1.0/e_bg[:, 1:]))
    if h is not None:
        hb = h[ix0:ix0 + eh.shape[0], jy0:jy0 + eh.shape[1]]
        eh[:hb.shape[0], :hb.shape[1]] = hb
    f, inv = tensor(xs, ys[:-1] + 0.5*dx, 0.5*(e_bg[:, :-1] + e_bg[:, 1:]), eh, 'y')
    sl = (slice(ix0, ix0 + inv.shape[0]), slice(jy0, jy0 + inv.shape[1]))
    blk = to_np(Ca_ey[sl]).astype(np.float64)
    m = f > 0
    blk[m] = c*inv[m]
    Ca_ey[sl] = xp.asarray(blk.astype(np.float32))


def _n_source_2d(cad, src_x_um, Ny, iy_src, y0_m, dx, wg_n, T_WG, n_below, gap_um):
    """Brechzahl am Quellpunkt: die Strahlneigung (vcsel_tilt) gilt im MEDIUM, in dem die
    Quelle sitzt (Phasenrampe k0*n*sin(theta)); frueher immer Vakuum-k0 (Winkel in Luft).
    CAD: aus der Szene; eingebauter Stapel: Kern, Luft (Einkoppelabstand / ueber dem Kern)
    oder die Schicht unter dem Kern."""
    try:
        if cad is not None:
            e = np.asarray(cad.eps2d(src_x_um - dx*1e6, 3, Ny)[0])[1]
            return float(np.sqrt(max(float(np.real(e[min(max(iy_src, 0), Ny - 1)])), 1.0)))
    except Exception:
        return 1.0
    if gap_um and src_x_um < gap_um:
        return float(N_AIR)
    y = y0_m + iy_src*dx
    if 0.0 <= y < T_WG:
        return float(np.real(wg_n))
    return float(N_AIR) if y >= T_WG else float(np.real(n_below))


def _src_setup(Ny, dx, T_WG, vcsel_offset, vcsel_waist, y0_m=None, y_center_m=None):
    """Gauss-Profil der Quelle in y. Standard: WG-Mitte + Versatz. y_center_m
    (CAD: automatisch ausgerichtete Mitte) ersetzt die WG-Mitte; y0_m = y bei j=0."""
    if y0_m is None:                                 # zur Laufzeit (TEAR_BUFFER variabel)
        y0_m = -TEAR_BUFFER
    y_wg_center = (T_WG/2 if y_center_m is None else y_center_m) + vcsel_offset*1e-6
    iy_src = int(round((y_wg_center - y0_m)/dx))
    waist_pix = max(2, int(round(vcsel_waist*1e-6/dx)))
    ys_idx = xp.arange(Ny, dtype=xp.float32)
    src_p = xp.exp(-((ys_idx - iy_src)/waist_pix)**2)
    if SRC_CORE_ONLY:
        if y_center_m is None:
            # nur durch den Kern einkoppeln (weiche Kante), kein direkter Ausläufer in
            # Luft/Traenenfilm. Gewicht auf den Knoten y_j (Ez bzw. Mittel fuer Ey).
            y_um = (y0_m + np.arange(Ny)*dx)*1e6
            src_p = src_p*xp.asarray(core_source_mask(y_um, T_WG*1e6).astype(np.float32))
            print('  Source in core only (--source-core-only, soft edge '
                  f'{SRC_CORE_RAMP_UM:g} um)')
        else:
            print('  [Note] --source-core-only applies to the layer stack only, not to CAD')
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


def _tm_ex_node(Ex):
    """Ex (Nx-1,Ny) VORZEICHENRICHTIG auf das Knotengitter (Nx,Ny) gemittelt (fuer die
    komplexe CW-Amplitude von Ex: tangentiale, an Grenzflaechen stetige TM-Komponente)."""
    out = xp.zeros((Ex.shape[0] + 1, Ex.shape[1]), dtype=xp.float32)
    out[1:-1, :] = 0.5*(Ex[:-1, :] + Ex[1:, :])
    out[0, :] = Ex[0, :]; out[-1, :] = Ex[-1, :]
    return out


def _tm_ex2_node(Ex):
    """Ex^2 (Gitter (Nx-1,Ny)) per Quadrat-Mittelung auf das Knotengitter (Nx,Ny).
    Fuer die VOLLE TM-Intensitaet |E|^2 = Ex^2 + Ey^2: Ex ist im evaneszenten Feld
    (Tear-Film) nicht vernachlaessigbar, und Ey allein springt an den Schicht-
    grenzen (Normalkomponente, eps1*Ey1 = eps2*Ey2)."""
    s = Ex*Ex
    out = xp.zeros((Ex.shape[0] + 1, Ex.shape[1]), dtype=xp.float32)
    out[1:-1, :] = 0.5*(s[:-1, :] + s[1:, :])
    out[0, :] = s[0, :]; out[-1, :] = s[-1, :]
    return out


def _sx_plane_2d(pol, il, Ez=None, Hy=None, Ey=None, Hz=None):
    """Momentaner Netto-Fluss in +x durch die Ebene x=il (Summe ueber den GANZEN
    Querschnitt): TE S_x = -Ez*Hy, TM S_x = Ey*Hz; H auf die E-Ebene gemittelt
    (Hy/Hz liegen bei i+1/2). Device-Skalar (kein Sync)."""
    if pol == 's':
        hy = 0.5*(Hy[il - 1] + Hy[il])
        return -xp.sum(Ez[il]*hy, dtype=xp.float64)
    hz = 0.5*(Hz[il - 1] + Hz[il])
    return xp.sum(Ey[il]*hz, dtype=xp.float64)


def poynting_flux(E, ix, eps_col=None):
    """Netto-Leistungsfluss in +x durch die Ebene x=ix, integriert ueber den GANZEN
    Querschnitt (Kern + evaneszenter Auslaeufer + Abstrahlung), aus der komplexen
    CW-Amplitude Ê (Konvention: Feld = Re(Ê e^{+i w t})):
      TE (Ez):  Sx = -1/2 Re(Ez Hy*), Hy = (1/(i w mu)) dEz/dx -> Sx ~ Im(Ê dÊ*/dx)  (exakt)
      TM (Ey):  Sx ~ eps(y) Im(Ê dÊ*/dx)   (eps_col uebergeben; Naeherung gefuehrter Mode)
    Konstanten kuerzen sich in Verhaeltnissen. Rueckwaerts laufende Anteile zaehlen negativ."""
    E = np.asarray(E)
    d = 0.5*(E[ix + 1] - E[ix - 1])
    s = np.imag(E[ix]*np.conj(d))
    if eps_col is not None:
        s = s*np.asarray(eps_col, dtype=np.float64)[:s.shape[0]]
    return float(np.sum(s))


def _eps_col_ey(eps_col):
    """eps auf den Knoten (Ny) -> auf das Ey-Gitter (Ny-1)."""
    e = np.asarray(eps_col, dtype=np.float64)
    return 0.5*(e[:-1] + e[1:])


# Laufzeit-Marge: Gruppenindex der (hoeheren) Moden ~ 1.03-1.1 x n_Kern
N_G_FACTOR = 1.10
# CW-Konvergenz: relative Aenderung des DFT-Phasors zwischen zwei Fenstern a 8 Perioden
CW_TOL = 0.01
CW_MAX_EXTEND = 2.0


_RUN_BEADS_ARGS = None


def run_beads(wg_mat, bead_mat, bead_d_um, dx_nm=20.0, save_frames=True, n_snapshots=12,
              wg_thickness_um=5.0, lambda_nm=850.0, vcsel_waist=2.0, vcsel_tilt=0.0,
              vcsel_offset=0.0, source_type='cw', method='full',
              window_w_um=350.0, slide_um=150.0, place_bead=True, t_aqueous_um=None,
              t_mucin_um=None, t_lipid_um=0.0, input_gap_um=0.0, length_um=None,
              bead_x_um=None, bead_y_um=None, wg_n=None, bead_n=None, n_aqueous=None, n_mucin=None,
              n_cornea=None, n_lipid=None, polarization='s', curved=False, boundary='cpml',
              cad=None, pulse_cycles=30.0, pulse_tmax_fs=None, pulse_nfreq=None,
              pulse_dispersion=True, pulse_mode='fourier', n_aq_tear_offset=None,
              tear_buffer_um=None):
    """cad: optional common.cad_geometry.CadRun -> freie CAD-Geometrie statt des
    eingebauten Schichtstapels (Rechenbox, eps, Quelle und Detektoren aus der Szene).
    pulse_cycles: Gauss-Puls exp(-((t-3s)/s)^2) sin(w0 t) mit s = pulse_cycles/f0
    (source_type='pulse'). Mit method='stitch' -> Fourier-Stitch (run_beads_stitched_pulse)."""
    if tear_buffer_um is not None:
        # Tiefe unter dem Kern (z.B. --cornea-full bis ins Kammerwasser): Modul-Puffer
        # fuer DIESEN Lauf setzen (alle Builder lesen ihn zur Laufzeit), danach zuruecksetzen.
        global TEAR_BUFFER
        _old = TEAR_BUFFER
        TEAR_BUFFER = float(tear_buffer_um)*1e-6
        try:
            _kw = dict(locals()); _kw.pop('_old'); _kw['tear_buffer_um'] = None
            return run_beads(**{k: _kw[k] for k in _RUN_BEADS_ARGS})
        finally:
            TEAR_BUFFER = _old

    if method == 'sliding':
        return run_beads_sliding(wg_mat, bead_mat, bead_d_um, dx_nm, save_frames, n_snapshots,
                                 wg_thickness_um, lambda_nm, vcsel_waist, vcsel_tilt,
                                 vcsel_offset, source_type, window_w_um, slide_um, place_bead,
                                 t_aqueous_um, t_mucin_um=t_mucin_um, t_lipid_um=t_lipid_um,
                                 input_gap_um=input_gap_um, length_um=length_um,
                                 bead_x_um=bead_x_um, bead_y_um=bead_y_um, wg_n=wg_n, bead_n=bead_n,
                                 n_aqueous=n_aqueous, n_mucin=n_mucin, n_cornea=n_cornea,
                                 n_lipid=n_lipid, polarization=polarization, curved=curved,
                                 boundary=boundary, pulse_cycles=pulse_cycles, cad=cad)
    if method == 'stitch' and str(source_type).lower() == 'pulse' and pulse_mode == 'td':
        return run_beads_stitched_td(
            wg_mat, bead_mat, bead_d_um, dx_nm, save_frames, n_snapshots,
            wg_thickness_um, lambda_nm, vcsel_waist, vcsel_tilt, vcsel_offset,
            window_w_um, slide_um, place_bead, pulse_cycles=pulse_cycles,
            pulse_tmax_fs=pulse_tmax_fs, cad=cad, t_aqueous_um=t_aqueous_um,
            t_mucin_um=t_mucin_um, t_lipid_um=t_lipid_um, input_gap_um=input_gap_um,
            length_um=length_um, bead_x_um=bead_x_um, bead_y_um=bead_y_um, wg_n=wg_n, bead_n=bead_n,
            n_aqueous=n_aqueous, n_mucin=n_mucin, n_cornea=n_cornea, n_lipid=n_lipid,
            polarization=polarization, curved=curved, boundary=boundary)
    if method == 'stitch' and str(source_type).lower() == 'pulse':
        return run_beads_stitched_pulse(
            wg_mat, bead_mat, bead_d_um, dx_nm, save_frames, n_snapshots,
            wg_thickness_um, lambda_nm, vcsel_waist, vcsel_tilt, vcsel_offset,
            window_w_um, slide_um, place_bead, t_aqueous_um=t_aqueous_um,
            t_mucin_um=t_mucin_um, t_lipid_um=t_lipid_um, input_gap_um=input_gap_um,
            length_um=length_um, bead_x_um=bead_x_um, bead_y_um=bead_y_um, wg_n=wg_n, bead_n=bead_n,
            n_aqueous=n_aqueous, n_mucin=n_mucin, n_cornea=n_cornea, n_lipid=n_lipid,
            polarization=polarization, curved=curved, boundary=boundary, cad=cad,
            pulse_cycles=pulse_cycles, pulse_tmax_fs=pulse_tmax_fs, pulse_nfreq=pulse_nfreq,
            pulse_dispersion=pulse_dispersion, n_aq_tear_offset=n_aq_tear_offset)
    if method == 'stitch':
        return run_beads_stitched(wg_mat, bead_mat, bead_d_um, dx_nm, save_frames, n_snapshots,
                                  wg_thickness_um, lambda_nm, vcsel_waist, vcsel_tilt,
                                  vcsel_offset, source_type, window_w_um, slide_um, place_bead,
                                  t_aqueous_um, t_mucin_um=t_mucin_um, t_lipid_um=t_lipid_um,
                                  input_gap_um=input_gap_um, length_um=length_um,
                                  bead_x_um=bead_x_um, bead_y_um=bead_y_um, wg_n=wg_n, bead_n=bead_n,
                                  n_aqueous=n_aqueous, n_mucin=n_mucin, n_cornea=n_cornea,
                                  n_lipid=n_lipid, polarization=polarization, curved=curved,
                                  boundary=boundary, cad=cad)
    _pol = 'p' if str(polarization).lower() in ('p', 'tm') else 's'
    t_aq_um = t_aqueous_um if t_aqueous_um is not None else T_AQ*1e6   # feste Aqueous-Dicke
    t_total = time.time()
    wg_n = wg_n if wg_n is not None else n_at(wg_mat, lambda_nm)
    bead_n = bead_n if bead_n is not None else n_at(bead_mat, lambda_nm)
    n_aq_l = n_aqueous if n_aqueous is not None else n_at('aqueous', lambda_nm)
    n_mu_l = n_mucin if n_mucin is not None else n_at('mucin', lambda_nm)
    n_co_l = n_cornea if n_cornea is not None else cornea_default(lambda_nm)
    n_lip_l = n_lipid if n_lipid is not None else n_at('lipid', lambda_nm)
    L = (length_um*1e-6) if length_um is not None else L_DEMO
    L_um = L*1e6
    bx_um = bead_x_um if bead_x_um is not None else L_um/2.0
    T_WG = wg_thickness_um*1e-6; LAM = lambda_nm*1e-9
    global LAM_LOSS
    LAM_LOSS = LAM                                   # kappa -> sigma bei dieser Wellenlaenge
    label = f'WG-{wg_mat}_Bead-{bead_mat}_d{bead_d_um:g}_L{lambda_nm:g}' + ('' if place_bead else '_ref')
    dx = dx_nm*1e-9
    dt = 0.5*dx/(C0*np.sqrt(2))
    if cad is not None:                              # freie CAD-Geometrie
        _cm = cad.meta()
        label = f'CAD-{os.path.splitext(_cm["cad_file"])[0]}_z{cad.z0_um:g}_L{lambda_nm:g}'
        L = cad.lx_um*1e-6; L_um = L*1e6
        y0_m = cad.y0_um*1e-6; H_total = cad.ly_um*1e-6
        wg_n = float(_cm['cad_n'][cad.core_index - 1])   # Kern-Index (Schrittzahl, Ausgabe)
    else:
        y0_m = -TEAR_BUFFER                          # y bei j=0
        H_total = T_WG + AIR_BUFFER + TEAR_BUFFER
    print(f'\n===== Bead-Demo [full]: {label} =====')
    Nx = int(round(L/dx)); Ny = int(round(H_total/dx))
    cad_region = None
    _bp = {}                                   # Bead-Daten fuer TM-Tensor
    if cad is not None:
        print(f'Grid: dx={dx_nm}nm  {Nx}x{Ny}  CAD {_cm["cad_file"]} cut z={cad.z0_um:g}um  '
              f'core "{cad.core}" n={wg_n:.3f}')
        print(cad.scene.summary(lambda_nm))
        _eps_np, cad_region = cad.eps2d(0.0, Nx, Ny)
        eps_r, sig = _split_eps(_eps_np)
        del _eps_np
    else:
        print(f'Grid: dx={dx_nm}nm  {Nx}x{Ny}  WG n={wg_n:.3f} t={wg_thickness_um}um  Bead d={bead_d_um}um n={bead_n:.3f}')
        print(f'Length={L_um:g}um  Aqueous={t_aq_um:g}um (fixed, at WG)  Lipid={(t_lipid_um or 0):g}um  '
              f'Gap={input_gap_um:g}um  Bead-x={bx_um:g}um')
        eps_r, sig = build_materials(Nx, Ny, dx, wg_n, bead_n, T_WG, bx_um, bead_d_um, 0.0, place_bead,
                                     n_aq=n_aq_l, n_mucin=n_mu_l, n_cornea=n_co_l, t_aq_um=t_aq_um,
                                     t_mu_um=t_mucin_um, t_lip_um=t_lipid_um, n_lip=n_lip_l,
                                     input_gap_um=input_gap_um, length_um=length_um, curved=curved, tm_patch=_bp, bead_y_um=bead_y_um)
    Ce_E, Ce_H = _coeffs(eps_r, sig, dt, dx)
    if _pol == 'p':                                  # TM-Koeffizienten aus eps_r
        Ca_ex, Ca_ey, Ch_tm = _tm_coeffs(eps_r, dt, dx)
        _apply_bead_tm(_bp, Ca_ex, Ca_ey, dt, dx)   # TM-Tensor am Bead-Rand
        if cad is not None:                         # CAD-TM: Ey harmonisch in y
            Ca_ey[...] = xp.asarray((dt/(EPS0*dx)/cad.eps2d_ey(0.0, Ca_ey.shape[0], Ca_ey.shape[1] + 1)).astype(np.float32))
        Da_ex, Da_ey = _tm_loss(eps_r, sig, Ca_ex, Ca_ey, dt)   # Absorption (kappa)
    # Mur mit lokaler Geschwindigkeit (x-Enden: n_eff der gefuehrten Welle, y: n der
    # Randzellen). Mit c0 reflektiert Mur im Dielektrikum |r|~(n-1)/(n+1) ~ 0.2.
    _msp = mur_speeds(eps_r, dx, LAM) if str(boundary).lower() != 'cpml' else None
    ms = MurSet(dx, dt, _msp)
    _eps_h = to_np(eps_r) if _pol == 'p' else None   # TM: eps fuer Poynting-Gewicht
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
    NPML = 10
    _cpml = boundary.lower() == 'cpml'
    src_ix = (NPML + 2) if _cpml else 5              # Quelle nicht in die PML legen
    tilt = vcsel_tilt
    if cad is not None:
        # Quelle automatisch auf die Eintrittsflaeche des Kerns: Zentrum so, dass der
        # Strahl die Facettenmitte trifft, Neigung = Innennormale (Versatz/Neigung
        # der Eingabe wirken als Korrektur obendrauf).
        _s = cad.source(src_ix*dx*1e6)
        tilt = vcsel_tilt + _s['tilt_deg']
        iy_src, src_p, ys_idx = _src_setup(Ny, dx, T_WG, vcsel_offset, vcsel_waist,
                                           y0_m=y0_m, y_center_m=_s['y_um']*1e-6)
        print(f'Source (auto): facet {np.round(_s["facet_center"][:2], 2)} um, '
              f'normal {np.round(_s["facet_normal"][:2], 3)} -> spot y={_s["y_um"] + vcsel_offset:.2f}um '
              f'at x={src_ix*dx*1e6:.2f}um, tilt {tilt:.2f} deg')
    else:
        iy_src, src_p, ys_idx = _src_setup(Ny, dx, T_WG, vcsel_offset, vcsel_waist)
    src_p_ey = 0.5*(src_p[:-1] + src_p[1:])           # Ey-Gitter (Ny-1)
    f0 = C0/LAM; sigma_t = 4/(2*np.pi*f0); k0 = 2*np.pi/LAM
    _n_src = _n_source_2d(cad, src_ix*dx*1e6, Ny, iy_src, y0_m, dx, wg_n, T_WG, n_aq_l, input_gap_um)
    phase_y = float(np.sin(np.radians(tilt)))*k0*_n_src*(ys_idx - iy_src)*dx   # Winkel im Medium an der Quelle
    phase_y_ey = 0.5*(phase_y[:-1] + phase_y[1:])
    mur = mur1_coeff(dx, dt)
    bnd = make_boundary(boundary, _pol, Nx, Ny, dx, dt, npml=NPML, lam_m=LAM)
    if not _cpml:
        print('  ' + ms.describe())
    print(f'Boundary: {boundary}' + (f' (PML {NPML} cells, src@ix={src_ix})' if _cpml else ''))
    print(f'Polarization: {_pol}-pol -> {"TM (Ey/Ex/Hz), source drives Ey" if _pol=="p" else "TE (Ez/Hx/Hy), source drives Ez"}')
    # Laufzeit: Anlauf + 2 Transits beim GRUPPENindex (die reflektierte/gestreute Welle
    # laeuft mit v_g; gemessen n_g ~ 1.53 > n_Kern bei hoeheren Moden) -> Marge N_G_FACTOR.
    # Danach CW-Konvergenztest (zwei aufeinanderfolgende DFT-Fenster); falls nicht
    # eingeschwungen, wird um je ein Fenster verlaengert (hoechstens CW_MAX_EXTEND x).
    steps_total = int((6*sigma_t + 2*L*N_G_FACTOR*wg_n/C0)/dt)
    steps_max = int(CW_MAX_EXTEND*steps_total)
    snap_every = max(1, steps_total//n_snapshots)
    print(f'Steps: {steps_total}')
    if cad is not None:
        # Detektoren am Kern-Koerper: 10 % / 90 % seiner Laenge, quer = sein Querschnitt
        (ixin0, ixin1, iyi0, iyi1), (ixout0, ixout1, iyo0, iyo1) = cad.detectors(cad_region, 0.0)
    else:
        iy_lo = int(round((0.0 + TEAR_BUFFER)/dx)); iy_hi = int(round((T_WG + TEAR_BUFFER)/dx))

        # Fensterbreite hoechstens 10 % der Laenge: sonst ueberdecken sich Ein- und
        # Ausgangsfenster bei kurzen Domaenen (L < 2*DET_LEN) -> P_in = P_out, T = 1.
        _hw = max(1, int(round(min(DET_LEN/2, 0.05*L_um*1e-6)/dx)))

        def xwin(xu):
            c = int(round(xu*1e-6/dx))
            return max(0, c - _hw), min(Nx, c + _hw)
        # Detektorfenster skalieren mit der Laenge (10% Eingang / 90% Ausgang)
        ixin0, ixin1 = xwin(0.1*L_um); ixout0, ixout1 = xwin(0.9*L_um)
        iyi0, iyi1, iyo0, iyo1 = iy_lo, iy_hi, iy_lo, iy_hi
    # Eingangsfenster stromab der Quelle halten (sonst misst es die Rueckwaerts-Haelfte)
    _ci = (ixin0 + ixin1)//2
    _sh = guard_in_plane(_ci, src_ix, Nx) - _ci
    ixin0, ixin1 = ixin0 + _sh, min(Nx, ixin1 + _sh)
    P_in = 0.0; P_out = 0.0
    P_in_d = xp.zeros((), dtype=xp.float64); P_out_d = xp.zeros((), dtype=xp.float64)
    # Exakter CW-Poynting-Fluss (E und H) durch die Mitten der Detektorfenster
    _flux = PlaneFluxDFT(2*np.pi*f0, dt)
    _fl_planes = {'in': max(1, (ixin0 + ixin1)//2), 'out': max(1, (ixout0 + ixout1)//2)}
    # CW-Steady per laufendem DFT bei f0 (letzte ~8 Perioden) -> zweiter,
    # eingeschwungener Frame-Satz (Phasen ueber 1 Periode), zusaetzlich zum
    # transienten (einlaufenden) Satz. Akkumulation auf dem Device.
    omega = 2*np.pi*f0
    steps_per_T = max(4, int(round((1.0/f0)/dt)))
    n_acc_cw = 8*steps_per_T
    acc_cw = xp.zeros((Nx, Ny), dtype=xp.complex64); nacc_cw = 0
    # TM: komplexe CW-Amplitude von Ex (tangential) zusaetzlich -> Analyzer kann Ex/Ey/|E|
    acc_cw_ex = xp.zeros((Nx, Ny), dtype=xp.complex64) if _pol == 'p' else None
    acc_prev = xp.zeros((Nx, Ny), dtype=xp.complex64); nacc_prev = 0
    _conv = source_type != 'pulse'; cw_rel_change = float('nan')
    # EXAKTES Zeitmittel <E^2>: jeden Zeitschritt der 2. Lauf-Haelfte akkumulieren.
    # Der Analyzer nutzt das fuer 'Int-Mittel'; ein Mittel ueber die wenigen
    # Snapshots waere verfaelscht (die liegen oft fast phasengleich -> es bliebe
    # der cos^2-Traeger als lambda/2-Streifenmuster stehen).
    n_avg0 = steps_total//2
    acc_I = xp.zeros((Nx, Ny), dtype=xp.float32); nacc_I = 0
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
    n = -1
    while n + 1 < steps_total:
        n += 1
        t_phys = n*dt
        if source_type == 'pulse':
            sp = pulse_cycles/f0; envelope = float(np.exp(-((t_phys - 3*sp)/sp)**2))
        else:
            envelope = float(1 - np.exp(-((t_phys/(2*sigma_t))**2)))
        if _pol == 'p':
            # --- TM (Hz, Ex, Ey) ---
            if _cpml:
                bnd.update_H(Ex_t, Ey_t, Hz_t, Ch_tm)
            else:
                Hz_t += -Ch_tm*((Ey_t[1:, :] - Ey_t[:-1, :]) - (Ex_t[:, 1:] - Ex_t[:, :-1]))
            ey_l = Ey_t[1, :].copy(); ey_r = Ey_t[-2, :].copy()
            ex_b = Ex_t[:, 1].copy(); ex_t = Ex_t[:, -2].copy()
            if Da_ex is not None:                    # Absorption: E <- D E vor dem Update
                Ex_t *= Da_ex; Ey_t *= Da_ey
            if _cpml:
                bnd.update_E(Ex_t, Ey_t, Hz_t, Ca_ex, Ca_ey)
            else:
                Ex_t[:, 1:-1] += Ca_ex[:, 1:-1]*(Hz_t[:, 1:] - Hz_t[:, :-1])
                Ey_t[1:-1, :] += -Ca_ey[1:-1, :]*(Hz_t[1:, :] - Hz_t[:-1, :])
            if tilt != 0.0:
                Ey_t[src_ix, :] += envelope*xp.sin(2*np.pi*f0*t_phys + phase_y_ey)*src_p_ey
            else:
                Ey_t[src_ix, :] += envelope*float(np.sin(2*np.pi*f0*t_phys))*src_p_ey
            if _cpml:
                bnd.terminate(Ex_t, Ey_t)            # PEC hinter der PML
            else:                                    # Mur 1. Ordnung (Default)
                Ey_t[0, :] = ey_l + ms.xl*(Ey_t[1, :] - Ey_t[0, :])
                Ey_t[-1, :] = ey_r + ms.xr*(Ey_t[-2, :] - Ey_t[-1, :])
                Ex_t[:, 0] = ex_b + ms.bh*(Ex_t[:, 1] - Ex_t[:, 0])
                Ex_t[:, -1] = ex_t + ms.th*(Ex_t[:, -2] - Ex_t[:, -1])
            field = _tm_node_field(Ey_t)              # (Nx,Ny) fuer Frames/Detektor
            I_E = field*field + _tm_ex2_node(Ex_t)    # volle Intensitaet |E|^2
        else:
            # --- TE (Ez, Hx, Hy) ---
            if _cpml:
                bnd.update_H(Ez, Hx, Hy, Ch)
            else:
                Hx -= Ch*(Ez[:, 1:] - Ez[:, :-1])
                Hy += Ch*(Ez[1:, :] - Ez[:-1, :])
            ex1, ex2 = Ez[1, :].copy(), Ez[-2, :].copy()
            ey1, ey2 = Ez[:, 1].copy(), Ez[:, -2].copy()
            if _cpml:
                bnd.update_E(Ez, Hx, Hy, Ce_E, Ce_H)
            else:
                Ez[1:-1, 1:-1] = (Ce_E[1:-1, 1:-1]*Ez[1:-1, 1:-1]
                                  + Ce_H[1:-1, 1:-1]*((Hy[1:, 1:-1] - Hy[:-1, 1:-1])
                                                      - (Hx[1:-1, 1:] - Hx[1:-1, :-1])))
            if tilt != 0.0:
                Ez[src_ix, :] += envelope*xp.sin(2*np.pi*f0*t_phys + phase_y)*src_p
            else:
                Ez[src_ix, :] += envelope*float(np.sin(2*np.pi*f0*t_phys))*src_p
            if _cpml:
                bnd.terminate(Ez)                    # PEC hinter der PML
            else:                                    # Mur 1. Ordnung (Default)
                Ez[0, :] = ex1 + ms.xl*(Ez[1, :] - Ez[0, :]); Ez[-1, :] = ex2 + ms.xr*(Ez[-2, :] - Ez[-1, :])
                Ez[:, 0] = ey1 + ms.b*(Ez[:, 1] - Ez[:, 0]); Ez[:, -1] = ey2 + ms.t*(Ez[:, -2] - Ez[:, -1])
            field = Ez
            I_E = field*field                         # TE: |E|^2 = Ez^2
        P_in_d += xp.sum(I_E[ixin0:ixin1, iyi0:iyi1], dtype=xp.float64)   # Device, kein Sync
        P_out_d += xp.sum(I_E[ixout0:ixout1, iyo0:iyo1], dtype=xp.float64)
        if n >= steps_total - n_acc_cw:                 # eingeschwungenes Ê akkumulieren
            acc_cw += field.astype(xp.complex64)*np.exp(-1j*omega*t_phys)
            nacc_cw += 1
            if acc_cw_ex is not None:
                acc_cw_ex += _tm_ex_node(Ex_t).astype(xp.complex64)*np.exp(-1j*omega*t_phys)
            for _k, _il in _fl_planes.items():
                _flux.add(_k, t_phys, pairs_2d(_pol, _il, Ez=Ez, Hy=Hy, Ey=Ey_t, Hz=Hz_t))
        elif _conv and n >= steps_total - 2*n_acc_cw:   # Fenster davor (Konvergenztest)
            acc_prev += field.astype(xp.complex64)*np.exp(-1j*omega*t_phys)
            nacc_prev += 1
        if n >= n_avg0:                                 # exaktes Zeitmittel <|E|^2>
            acc_I += I_E
            nacc_I += 1
        if save_frames and ez_mm is not None and ((n+1) % snap_every == 0) and k_fr < ez_mm.shape[0]:
            ez_mm[k_fr] = to_np(field).astype(np.float32)
            frames.append({'x_start': 0.0,
                           'x_end': L_um, 'y_start': y0_m*1e6,
                           'y_end': (y0_m + H_total)*1e6, 'slide_i': 0,
                           'step': n+1, 'time': time.time()-t_start})
            k_fr += 1
            print(f'  Step {n+1}/{steps_total} ({100*(n+1)/steps_total:.0f}%)  max|E|={float(xp.max(xp.abs(field))):.3e}')
        if _conv and n + 1 == steps_total and nacc_prev > 0 and nacc_cw > 0:
            _a = acc_cw/nacc_cw; _b = acc_prev/nacc_prev
            cw_rel_change = float(xp.linalg.norm(_a - _b)/max(float(xp.linalg.norm(_a)), 1e-30))
            if cw_rel_change > CW_TOL and steps_total + n_acc_cw <= steps_max:
                print(f'  [CW] not yet in steady state (change {100*cw_rel_change:.2f}% > '
                      f'{100*CW_TOL:.1f}%) -> +{n_acc_cw} Steps')
                acc_prev = acc_cw; nacc_prev = nacc_cw
                acc_cw = xp.zeros_like(acc_prev); nacc_cw = 0
                if acc_cw_ex is not None:
                    acc_cw_ex[...] = 0
                _flux.reset()
                steps_total += n_acc_cw
            else:
                print(f'  [CW] phasor change over last 2 DFT windows: {100*cw_rel_change:.3f}%'
                      + ('' if cw_rel_change <= CW_TOL else '  (!! max. runtime reached)'))
    if ez_mm is not None:
        ez_mm.flush(); del ez_mm
    # eingeschwungenen (CW) Frame-Satz aus Ê schreiben: nfr Phasen ueber 1 Periode
    cw_path = None; cw_frames = []
    nfr_cw = max(1, n_snapshots)
    Ecw = to_np((2.0/nacc_cw)*acc_cw) if nacc_cw > 0 else None   # komplexe Amplitude (Nx,Ny)
    if save_frames and nacc_cw > 0:
        cw_path = os.path.join('results', f'_stream_beads_{_safe}_full_cw.npy')
        cw_mm = np.lib.format.open_memmap(cw_path, mode='w+', dtype=np.float32,
                                          shape=(nfr_cw, Nx, Ny))
        for k in range(nfr_cw):
            tk = (k/nfr_cw)*(1.0/f0)
            cw_mm[k] = np.real(Ecw*np.exp(1j*omega*tk)).astype(np.float32)
            cw_frames.append({'x_start': 0.0, 'x_end': L_um, 'y_start': y0_m*1e6,
                              'y_end': (y0_m + H_total)*1e6, 'slide_i': 0,
                              'step': k, 'time': tk})
        cw_mm.flush(); del cw_mm
    P_in = float(P_in_d)*dx*dx; P_out = float(P_out_d)*dx*dx   # einmaliger Readout
    total_time = time.time() - t_total
    Tr_I = P_out/max(P_in, 1e-30)                 # alt: zeitintegriertes |E|^2 in Boxen
    # Einheitliche Transmission (alle Methoden): CW-Poynting-Fluss durch die Ebenen
    # x_in/x_out (Mitte der Detektorfenster), ueber den ganzen Querschnitt integriert.
    Tr = Tr_I; tr_def = 'intensity'; _p_si = None
    if Ecw is not None and source_type != 'pulse' and _flux.has('in') and _flux.has('out'):
        S_in, S_out = _flux.flux('in'), _flux.flux('out')   # exakt aus E und H
        _p_si = S_in*dx          # eingekoppelte Leistung [W/m] (2D: pro Meter Tiefe in z)
        if abs(S_in) > 0:
            Tr = S_out/S_in; tr_def = 'poynting'
    print(f'\n[Result] {label} [full]: total {total_time:.0f}s  Transmission={Tr:.4f} ({tr_def})'
          f'  [intensity boxes: {Tr_I:.4f}]')
    res = _result_dict(label, P_in, P_out, Tr, frames, total_time, bead_d_um,
                       wg_mat, bead_mat, wg_n, bead_n, wg_thickness_um, lambda_nm,
                       vcsel_waist, vcsel_tilt, vcsel_offset, source_type, 'full', place_bead, n_aq_l, n_mu_l, n_co_l,
                       frames_memmap=mm_path, t_aq_um=t_aq_um, t_mu_um=t_mucin_um,
                       t_lip_um=t_lipid_um, bead_x_um=bx_um,
                       x_wg_start_um=(input_gap_um or 0.0), x_wg_end_um=L_um,
                       polarization=_pol)
    res['Iavg'] = (to_np(acc_I)/max(1, nacc_I)) if nacc_I else None
    res['cw_frames'] = cw_frames
    res['cw_frames_memmap'] = cw_path
    res['cw_rel_change'] = cw_rel_change
    if acc_cw_ex is not None and nacc_cw > 0:        # TM: komplexe Ex-Amplitude (Knoten)
        res['Ex_cw_amp'] = to_np((2.0/nacc_cw)*acc_cw_ex)
    res['transmission_intensity'] = Tr_I; res['transmission_def'] = tr_def
    res['steps_total'] = steps_total
    res['p_in_si'] = _p_si; res['p_in_unit'] = 'W/m'   # fuer Leistungs-Skalierung (Analyzer)
    if cad is not None:
        res['cad'] = dict(cad.meta(), cad_tilt_deg=tilt,
                          cad_det=[[ixin0, ixin1, iyi0, iyi1], [ixout0, ixout1, iyo0, iyo1]])
        res['cad_region'] = cad_region
    res['view_primary'] = 'transient'      # full: einlaufend ist der Hauptsatz
    return res


def run_beads_sliding(wg_mat, bead_mat, bead_d_um, dx_nm, save_frames, n_snapshots,
                      wg_thickness_um, lambda_nm, vcsel_waist, vcsel_tilt,
                      vcsel_offset, source_type, window_w_um, slide_um, place_bead=True,
                      t_aqueous_um=None, t_mucin_um=None, t_lipid_um=0.0, input_gap_um=0.0,
                      length_um=None, bead_x_um=None, bead_y_um=None, wg_n=None, bead_n=None,
                      n_aqueous=None, n_mucin=None, n_cornea=None, n_lipid=None,
                      polarization='s', curved=False, boundary='cpml', pulse_cycles=30.0,
                      cad=None):
    _bnd_kind = (boundary or 'cpml').lower()
    _use_bnd = _bnd_kind == 'cpml'                   # Rand-Objekt (sonst inline Mur-1)
    _is_cpml = _bnd_kind == 'cpml'
    _pol = 'p' if str(polarization).lower() in ('p', 'tm') else 's'
    t_total = time.time()
    t_aq_um = t_aqueous_um if t_aqueous_um is not None else T_AQ*1e6   # feste Aqueous-Dicke
    wg_n = wg_n if wg_n is not None else n_at(wg_mat, lambda_nm)
    bead_n = bead_n if bead_n is not None else n_at(bead_mat, lambda_nm)
    n_aq_l = n_aqueous if n_aqueous is not None else n_at('aqueous', lambda_nm)
    n_mu_l = n_mucin if n_mucin is not None else n_at('mucin', lambda_nm)
    n_co_l = n_cornea if n_cornea is not None else cornea_default(lambda_nm)
    n_lip_l = n_lipid if n_lipid is not None else n_at('lipid', lambda_nm)
    L = (length_um*1e-6) if length_um is not None else L_DEMO
    T_WG = wg_thickness_um*1e-6; LAM = lambda_nm*1e-9
    global LAM_LOSS
    LAM_LOSS = LAM                                   # kappa -> sigma bei dieser Wellenlaenge
    label = f'WG-{wg_mat}_Bead-{bead_mat}_d{bead_d_um:g}_L{lambda_nm:g}' + ('' if place_bead else '_ref')
    dx = dx_nm*1e-9
    dt = 0.5*dx/(C0*np.sqrt(2))
    if cad is not None:                              # freie CAD-Geometrie
        _cm = cad.meta()
        label = f'CAD-{os.path.splitext(_cm["cad_file"])[0]}_z{cad.z0_um:g}_L{lambda_nm:g}'
        L = cad.lx_um*1e-6
        y0_m = cad.y0_um*1e-6; H_total = cad.ly_um*1e-6
        wg_n = float(_cm['cad_n'][cad.core_index - 1])
    else:
        y0_m = -TEAR_BUFFER
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
    src_ix = (10 + 2) if _is_cpml else 5             # Quelle nicht in die PML legen
    tilt = vcsel_tilt
    cad_region = None
    if cad is not None:
        _s = cad.source(src_ix*dx*1e6)               # Quelle auf die Eintrittsflaeche
        tilt = vcsel_tilt + _s['tilt_deg']
        iy_src, src_p, ys_idx = _src_setup(Ny, dx, T_WG, vcsel_offset, vcsel_waist,
                                           y0_m=y0_m, y_center_m=_s['y_um']*1e-6)
        Nx_full = int(round(L/dx))
        cad_region = np.zeros((Nx_full, Ny), dtype=np.int8)
        for _x0 in range(0, Nx_full, 4096):
            _x1 = min(_x0 + 4096, Nx_full)
            cad_region[_x0:_x1] = cad.eps2d(_x0*dx*1e6, _x1 - _x0, Ny, smooth=1)[1]
        (_a0, _a1, iyi0, iyi1), (_b0, _b1, iyo0, iyo1) = cad.detectors(cad_region, 0.0)
        det_x = ((0.5*(_a0 + _a1)*dx*1e6, 'in', iyi0, iyi1),
                 (0.5*(_b0 + _b1)*dx*1e6, 'out', iyo0, iyo1))
        print(f'Source (auto): spot y={_s["y_um"] + vcsel_offset:.2f}um, tilt {tilt:.2f} deg')
    else:
        iy_src, src_p, ys_idx = _src_setup(Ny, dx, T_WG, vcsel_offset, vcsel_waist)
    src_p_ey = 0.5*(src_p[:-1] + src_p[1:])
    f0 = C0/LAM; sigma_t = 4/(2*np.pi*f0); k0 = 2*np.pi/LAM
    _n_src = _n_source_2d(cad, src_ix*dx*1e6, Ny, iy_src, y0_m, dx, wg_n, T_WG, n_aq_l, input_gap_um)
    phase_y = float(np.sin(np.radians(tilt)))*k0*_n_src*(ys_idx - iy_src)*dx   # Winkel im Medium an der Quelle
    phase_y_ey = 0.5*(phase_y[:-1] + phase_y[1:])
    mur = mur1_coeff(dx, dt)
    NPML = 10
    bnd = make_boundary(_bnd_kind, _pol, Nx, Ny, dx, dt, npml=NPML, lam_m=LAM) if _use_bnd else None
    if _is_cpml:
        src_ix = max(src_ix, NPML + 2)               # Quelle nicht in die PML legen
    print(f'Polarization: {_pol}-pol ({"TM/Ey" if _pol=="p" else "TE/Ez"})  Boundary: {_bnd_kind}'
          + (' (all edges, psi rolled along)' if _is_cpml else ''))
    if cad is None:
        iy_lo = int(round((0.0 + TEAR_BUFFER)/dx)); iy_hi = int(round((T_WG + TEAR_BUFFER)/dx))
        det_x = ((0.1*L_um, 'in', iy_lo, iy_hi), (0.9*L_um, 'out', iy_lo, iy_hi))
    # Eingangsebene stromab der Quelle halten (Quelle sitzt im ersten Fenster bei src_ix)
    _xi_g = guard_in_plane(int(round(det_x[0][0]/(dx*1e6))), src_ix, int(round(L/dx)))*dx*1e6
    det_x = ((_xi_g,) + tuple(det_x[0][1:]), det_x[1])
    iy_h_det =max(1, int(round(min(DET_LEN/2, 0.05*L_um*1e-6)/dx)))   # halbe Fensterbreite in x
    P_in = xp.zeros((), dtype=xp.float64); P_out = xp.zeros((), dtype=xp.float64)   # Device
    S_in = xp.zeros((), dtype=xp.float64); S_out = xp.zeros((), dtype=xp.float64)   # Poynting
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
        _bp = {}                                   # Bead-Daten fuer TM-Tensor
        if cad is not None:
            eps_r, sig = _split_eps(cad.eps2d(x_start_um, Nx, Ny)[0])
        else:
            eps_r, sig = build_materials(Nx, Ny, dx, wg_n, bead_n, T_WG, bx_um, bead_d_um, x_start_um,
                                         place_bead, n_aq=n_aq_l, n_mucin=n_mu_l, n_cornea=n_co_l,
                                         t_aq_um=t_aq_um, t_mu_um=t_mucin_um, t_lip_um=t_lipid_um,
                                         n_lip=n_lip_l, input_gap_um=input_gap_um,
                                         length_um=length_um, curved=curved, tm_patch=_bp, bead_y_um=bead_y_um)
        Ce_E, Ce_H = _coeffs(eps_r, sig, dt, dx)
        if _pol == 'p':
            Ca_ex, Ca_ey, Ch_tm = _tm_coeffs(eps_r, dt, dx)
            _apply_bead_tm(_bp, Ca_ex, Ca_ey, dt, dx)   # TM-Tensor am Bead-Rand
            if cad is not None:                         # CAD-TM: Ey harmonisch in y
                Ca_ey[...] = xp.asarray((dt/(EPS0*dx)/cad.eps2d_ey(x_start_um, Ca_ey.shape[0], Ca_ey.shape[1] + 1)).astype(np.float32))
            Da_ex, Da_ey = _tm_loss(eps_r, sig, Ca_ex, Ca_ey, dt)   # Absorption (kappa)
        _msp = None if _is_cpml else mur_speeds(eps_r, dx, LAM)   # je Slide (Material neu)
        ms = MurSet(dx, dt, _msp)
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
            if _is_cpml:
                bnd.roll(N_shift)                    # psi mit dem Fenster mitziehen
        src_active = (slide_i == 0)
        steps_slide = (int(window_w_um*1e-6*wg_n/C0/dt) if src_active
                       else int(slide_um*1e-6*wg_n/C0/dt)) + 200
        snap_every = max(1, steps_slide//max(1, n_snapshots//n_slides + 1))
        print(f'[Slide {slide_i+1}/{n_slides}] x=[{x_start_um:.0f},{x_end_um:.0f}]um src={src_active} {steps_slide} steps')
        for n in range(steps_slide):
            t_phys = n*dt
            if src_active:
                if source_type == 'pulse':
                    sp = pulse_cycles/f0; envelope = float(np.exp(-((t_phys - 3*sp)/sp)**2))
                else:
                    envelope = float(1 - np.exp(-((t_phys/(2*sigma_t))**2)))
            if _pol == 'p':
                if _is_cpml:
                    bnd.update_H(Ex_t, Ey_t, Hz_t, Ch_tm)
                else:
                    Hz_t += -Ch_tm*((Ey_t[1:, :] - Ey_t[:-1, :]) - (Ex_t[:, 1:] - Ex_t[:, :-1]))
                ey_l = Ey_t[1, :].copy(); ey_r = Ey_t[-2, :].copy()
                ex_b = Ex_t[:, 1].copy(); ex_tp = Ex_t[:, -2].copy()
                if Da_ex is not None:                    # Absorption: E <- D E vor dem Update
                    Ex_t *= Da_ex; Ey_t *= Da_ey
                if _is_cpml:
                    bnd.update_E(Ex_t, Ey_t, Hz_t, Ca_ex, Ca_ey)
                else:
                    Ex_t[:, 1:-1] += Ca_ex[:, 1:-1]*(Hz_t[:, 1:] - Hz_t[:, :-1])
                    Ey_t[1:-1, :] += -Ca_ey[1:-1, :]*(Hz_t[1:, :] - Hz_t[:-1, :])
                if src_active:
                    if tilt != 0.0:
                        Ey_t[src_ix, :] += envelope*xp.sin(2*np.pi*f0*t_phys + phase_y_ey)*src_p_ey
                    else:
                        Ey_t[src_ix, :] += envelope*float(np.sin(2*np.pi*f0*t_phys))*src_p_ey
                if _is_cpml:
                    bnd.terminate(Ex_t, Ey_t)
                else:
                    Ey_t[0, :] = ey_l + ms.xl*(Ey_t[1, :] - Ey_t[0, :])
                    Ey_t[-1, :] = ey_r + ms.xr*(Ey_t[-2, :] - Ey_t[-1, :])
                    Ex_t[:, 0] = ex_b + ms.bh*(Ex_t[:, 1] - Ex_t[:, 0])
                    Ex_t[:, -1] = ex_tp + ms.th*(Ex_t[:, -2] - Ex_t[:, -1])
                field = _tm_node_field(Ey_t)
                exI = _tm_ex2_node(Ex_t)              # Ex^2 fuer volle |E|^2-Detektion
            else:
                exI = None
                if _is_cpml:
                    bnd.update_H(Ez, Hx, Hy, Ch)
                else:
                    Hx -= Ch*(Ez[:, 1:] - Ez[:, :-1])
                    Hy += Ch*(Ez[1:, :] - Ez[:-1, :])
                ex1, ex2 = Ez[1, :].copy(), Ez[-2, :].copy()
                ey1, ey2 = Ez[:, 1].copy(), Ez[:, -2].copy()
                if _is_cpml:
                    bnd.update_E(Ez, Hx, Hy, Ce_E, Ce_H)
                else:
                    Ez[1:-1, 1:-1] = (Ce_E[1:-1, 1:-1]*Ez[1:-1, 1:-1]
                                      + Ce_H[1:-1, 1:-1]*((Hy[1:, 1:-1] - Hy[:-1, 1:-1])
                                                          - (Hx[1:-1, 1:] - Hx[1:-1, :-1])))
                if src_active:
                    if tilt != 0.0:
                        Ez[src_ix, :] += envelope*xp.sin(2*np.pi*f0*t_phys + phase_y)*src_p
                    else:
                        Ez[src_ix, :] += envelope*float(np.sin(2*np.pi*f0*t_phys))*src_p
                if _is_cpml:
                    bnd.terminate(Ez)
                else:
                    Ez[0, :] = ex1 + ms.xl*(Ez[1, :] - Ez[0, :]); Ez[-1, :] = ex2 + ms.xr*(Ez[-2, :] - Ez[-1, :])
                    Ez[:, 0] = ey1 + ms.b*(Ez[:, 1] - Ez[:, 0]); Ez[:, -1] = ey2 + ms.t*(Ez[:, -2] - Ez[:, -1])
                field = Ez
            for (xg, which, iy_lo, iy_hi) in det_x:
                if x_start_um <= xg <= x_end_um:
                    cl = int(round((xg - x_start_um)*1e-6/dx))
                    a = max(0, cl-iy_h_det); b = min(Nx, cl+iy_h_det)
                    v = xp.sum(field[a:b, iy_lo:iy_hi]**2, dtype=xp.float64)   # Device, kein Sync
                    if exI is not None:                # TM: |E|^2 = Ey^2 + Ex^2
                        v = v + xp.sum(exI[a:b, iy_lo:iy_hi], dtype=xp.float64)
                    if 1 <= cl < Nx - 1:                # Poynting-Fluss durch die Ebene x=cl
                        _s = (_sx_plane_2d('s', cl, Ez=Ez, Hy=Hy) if _pol == 's'
                              else _sx_plane_2d('p', cl, Ey=Ey_t, Hz=Hz_t))
                    else:
                        _s = 0.0
                    if which == 'in':
                        P_in += v; S_in += _s
                    else:
                        P_out += v; S_out += _s
            if save_frames and ez_mm is not None and ((n+1) % snap_every == 0) and k_fr < ez_mm.shape[0]:
                ez_mm[k_fr] = to_np(field).astype(np.float32)
                frames.append({'x_start': x_start_um,
                               'x_end': x_end_um, 'y_start': y0_m*1e6,
                               'y_end': (y0_m + H_total)*1e6, 'slide_i': slide_i,
                               'step': n+1, 'time': time.time()-t_start})
                k_fr += 1
        print(f'  Slide done: max|E|={float(xp.max(xp.abs(field))):.3e}')
    if ez_mm is not None:
        ez_mm.flush(); del ez_mm
    total_time = time.time() - t_total
    P_in = float(P_in)*dx*dx; P_out = float(P_out)*dx*dx      # einmaliger Readout
    Tr_I = P_out/max(P_in, 1e-30)
    S_in = float(S_in); S_out = float(S_out)
    Tr = S_out/S_in if abs(S_in) > 0 else Tr_I                 # Pulsenergie (Poynting)
    tr_def = 'poynting' if abs(S_in) > 0 else 'intensity'
    print(f'\n[Result] {label} [sliding]: total {total_time:.0f}s  Transmission={Tr:.4f}')
    res = _result_dict(label, P_in, P_out, Tr, frames, total_time, bead_d_um,
                        wg_mat, bead_mat, wg_n, bead_n, wg_thickness_um, lambda_nm,
                        vcsel_waist, vcsel_tilt, vcsel_offset, source_type, 'sliding', place_bead, n_aq_l, n_mu_l, n_co_l,
                        frames_memmap=mm_path, t_aq_um=t_aq_um, t_mu_um=t_mucin_um,
                        t_lip_um=t_lipid_um, bead_x_um=bx_um,
                        x_wg_start_um=(input_gap_um or 0.0), x_wg_end_um=L_um,
                        polarization=_pol)
    res['transmission_intensity'] = Tr_I; res['transmission_def'] = tr_def
    if cad is not None:
        res['cad'] = dict(cad.meta(), cad_tilt_deg=tilt,
                          cad_det=[[int(det_x[0][0]/(dx*1e6)), int(det_x[0][0]/(dx*1e6)) + 1,
                                    det_x[0][2], det_x[0][3]],
                                   [int(det_x[1][0]/(dx*1e6)), int(det_x[1][0]/(dx*1e6)) + 1,
                                    det_x[1][2], det_x[1][3]]])
        res['cad_region'] = None        # Sliding-Frames sind Fensterausschnitte (keine Gesamtkarte)
    return res


def run_beads_stitched(wg_mat, bead_mat, bead_d_um, dx_nm, save_frames, n_snapshots,
                       wg_thickness_um, lambda_nm, vcsel_waist, vcsel_tilt,
                       vcsel_offset, source_type, window_w_um, slide_um, place_bead=True,
                       t_aqueous_um=None, t_mucin_um=None, t_lipid_um=0.0, input_gap_um=0.0,
                       length_um=None, bead_x_um=None, bead_y_um=None, wg_n=None, bead_n=None,
                       n_aqueous=None, n_mucin=None, n_cornea=None, n_lipid=None,
                       polarization='s', curved=False, boundary='cpml', cad=None,
                       return_efull=False):
    _bnd_kind = (boundary or 'cpml').lower()
    if _bnd_kind not in ('cpml', 'mur1'):
        raise ValueError(f"stitch: unknown boundary '{_bnd_kind}' (cpml|mur1)")
    # Trennung: die gewaehlte Randbedingung (mur1|cpml) gilt fuer die AEUSSEREN Raender
    # (y oben/unten, Eintritt in Fenster 0, Austritt im letzten Fenster). Die inneren
    # Fenstergrenzen sind KEINE physikalischen Raender und bekommen IMMER eine PML in
    # einer nicht assemblierten Verlaengerung: Mur (auf c0 abgestimmt) reflektiert dort
    # die gefuehrte Welle mit |r|~0.2 (gemessen 4.4% Leistung, 14-20% Feldfehler).
    _ucpml = _bnd_kind == 'cpml'                     # Nutzerwahl fuer die Aussenraender
    _use_bnd_st = True                               # Rand-Objekt immer CPML (ggf. 0 Zellen)
    _is_cpml_st = True
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
    n_co_l = n_cornea if n_cornea is not None else cornea_default(lambda_nm)
    n_lip_l = n_lipid if n_lipid is not None else n_at('lipid', lambda_nm)
    L = (length_um*1e-6) if length_um is not None else L_DEMO
    L_um = L*1e6
    T_WG = wg_thickness_um*1e-6; LAM = lambda_nm*1e-9
    global LAM_LOSS
    LAM_LOSS = LAM                                   # kappa -> sigma bei dieser Wellenlaenge
    bx_um = bead_x_um if bead_x_um is not None else L_um/2.0
    label = f'WG-{wg_mat}_Bead-{bead_mat}_d{bead_d_um:g}_L{lambda_nm:g}' + ('' if place_bead else '_ref')
    dx = dx_nm*1e-9
    dt = 0.5*dx/(C0*np.sqrt(2))
    if cad is not None:                              # freie CAD-Geometrie
        _cm = cad.meta()
        label = f'CAD-{os.path.splitext(_cm["cad_file"])[0]}_z{cad.z0_um:g}_L{lambda_nm:g}'
        L = cad.lx_um*1e-6; L_um = L*1e6
        y0_m = cad.y0_um*1e-6; H_total = cad.ly_um*1e-6
        wg_n = float(_cm['cad_n'][cad.core_index - 1])
    else:
        y0_m = -TEAR_BUFFER
        H_total = T_WG + AIR_BUFFER + TEAR_BUFFER
    Ny = int(round(H_total/dx))
    Nx_win = int(round(window_w_um*1e-6/dx))
    S_cells = int(round(slide_um*1e-6/dx))
    O_cells = Nx_win - S_cells                     # Ueberlappung (Yee-Zellen)
    Nx_full = int(round(L/dx))
    if O_cells < 8:
        print(f'  [WARN] overlap of {O_cells} cells is very small -> choose window_w >> slide.')
        O_cells = max(O_cells, 1)
    f0 = C0/LAM; omega = 2*np.pi*f0; sigma_t = 4/(2*np.pi*f0); k0 = 2*np.pi/LAM
    transit = window_w_um*1e-6*N_G_FACTOR*wg_n/C0     # Gruppenindex-Marge (F8)
    steps_win = int(3.0*transit/dt) + 400          # ~3 Transitzeiten -> Steady-State
    steps_per_T = max(4, int(round((1.0/f0)/dt)))
    n_acc = steps_per_T*8                           # DFT ueber 8 Perioden am Ende
    n_win = (int(np.ceil((Nx_full - Nx_win)/S_cells)) + 1) if Nx_full > Nx_win else 1
    print(f'\n===== Bead-Demo [stitch/domain decomposition]: {label} =====')
    print(f'Length {L_um:g}um -> {n_win} windows of {window_w_um}um (overlap {O_cells*dx*1e6:.1f}um), '
          f'grid/window {Nx_win}x{Ny}, steps/window {steps_win}')
    Ch = xp.float32(dt/(MU0*dx))
    mur = xp.float32((C0*dt - dx)/(C0*dt + dx))
    NPML_ST = 10
    src_ix = (NPML_ST + 2) if _ucpml else 5               # Quelle nicht in die x-PML
    tilt = vcsel_tilt
    cad_region = None
    if cad is not None:
        _s = cad.source(src_ix*dx*1e6)                    # Quelle auf die Eintrittsflaeche
        tilt = vcsel_tilt + _s['tilt_deg']
        iy_src, src_p, ys_idx = _src_setup(Ny, dx, T_WG, vcsel_offset, vcsel_waist,
                                           y0_m=y0_m, y_center_m=_s['y_um']*1e-6)
        print(f'Source (auto): facet {np.round(_s["facet_center"][:2], 2)} um -> spot '
              f'y={_s["y_um"] + vcsel_offset:.2f}um, tilt {tilt:.2f} deg')
        # Regionskarte der ganzen Laenge (fuer Detektoren + Analyzer), blockweise
        cad_region = np.zeros((Nx_full, Ny), dtype=np.int8)
        for _x0 in range(0, Nx_full, 4096):
            _x1 = min(_x0 + 4096, Nx_full)
            cad_region[_x0:_x1] = cad.eps2d(_x0*dx*1e6, _x1 - _x0, Ny, smooth=1)[1]
    else:
        iy_src, src_p, ys_idx = _src_setup(Ny, dx, T_WG, vcsel_offset, vcsel_waist)
    _n_src = _n_source_2d(cad, src_ix*dx*1e6, Ny, iy_src, y0_m, dx, wg_n, T_WG, n_aq_l, input_gap_um)
    phase_y = float(np.sin(np.radians(tilt)))*k0*_n_src*(ys_idx - iy_src)*dx   # Winkel im Medium an der Quelle
    src_p_ey = 0.5*(src_p[:-1] + src_p[1:])
    phase_y_ey = 0.5*(phase_y[:-1] + phase_y[1:])
    _Nyc = (Ny-1) if _pol == 'p' else Ny                  # y-Groesse der Hauptkomp.
    print(f'Polarization: {_pol}-pol ({"TM/Ey" if _pol=="p" else "TE/Ez"})'
          + f'  outer boundary: {_bnd_kind}; inner window boundaries: CPML (extension)')
    # v2: weiche cosinus-getaperte Einpraegung des Overlaps (1 an linker Kante ->
    # 0 an Innenkante) statt hartem Dirichlet -> vermeidet Naht/Reflexion am
    # Uebergang zur frei gerechneten Region.
    _itap = np.arange(O_cells)
    _wtap = xp.asarray((0.5*(1.0 + np.cos(np.pi*_itap/max(O_cells-1, 1)))
                        ).astype(np.float32))[:, None]
    Efull = np.zeros((Nx_full, _Nyc), dtype=np.complex64)  # assemblierte Amplitude
    Ehand = None                                          # Overlap-Amplitude Vorgaenger
    # TM: <Ex^2> (Knotengitter) mitassemblieren -> volle Intensitaet |E|^2 im Iavg.
    # Disk-Memmap (Groesse wie halbes Efull), damit grosse Linsen-Laeufe nicht OOM gehen.
    ExI_full = None
    if _pol == 'p':
        os.makedirs('results', exist_ok=True)
        _safe_i = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in label)
        # eindeutiger Name: ein frueheres Ergebnis kann seine Memmap noch offen halten
        # (Windows sperrt dann dieselbe Datei -> OSError beim zweiten Lauf im Prozess)
        ExI_full = np.lib.format.open_memmap(
            os.path.join('results', f'_stream_beads_{_safe_i}_stitch_Iavg_'
                                    f'{os.getpid()}_{int(time.time()*1000)}.npy'),
            mode='w+', dtype=np.float32, shape=(Nx_full, Ny))
        # komplexe CW-Amplitude von Ex (Knotengitter) -> Analyzer: Ex/Ey/|E| umschaltbar
        ExA_full = np.lib.format.open_memmap(
            os.path.join('results', f'_stream_beads_{_safe_i}_stitch_ExA_'
                                    f'{os.getpid()}_{int(time.time()*1000)}.npy'),
            mode='w+', dtype=np.complex64, shape=(Nx_full, Ny))
    else:
        ExA_full = None
    # Fensterstart: w*S, das LETZTE Fenster buendig am Domaenenende (sonst ragt es
    # ueber L hinaus und sieht dort eine WG-Endfacette, die der full-Lauf nicht hat).
    def _gx0(w):
        return min(w*S_cells, max(Nx_full - Nx_win, 0))
    # Innere Fenster rechts um eine CPML-Schicht verlaengert (wird NICHT assembliert):
    # reflexionsfreie Naht statt Mur. WG laeuft wie im full-Lauf in die PML hinein.
    _EXT = NPML_ST + 4
    _len_mat = L_um + _EXT*dx*1e6
    # Detektor-Ebenen (global) schon vor der Fensterschleife: dort wird der exakte
    # CW-Poynting-Fluss (E und H) im Fenster mitgeschrieben, das die Ebene assembliert.
    if cad is not None:
        (_a0, _a1, iyi0, iyi1), (_b0, _b1, iyo0, iyo1) = cad.detectors(cad_region, 0.0)
        ix_in = (_a0 + _a1)//2; ix_out = (_b0 + _b1)//2
    else:
        iy_lo = int(round((0.0 + TEAR_BUFFER)/dx)); iy_hi = int(round((T_WG + TEAR_BUFFER)/dx))
        ix_in = int(round(0.1*Nx_full)); ix_out = int(round(0.9*Nx_full))
        iyi0, iyi1, iyo0, iyo1 = iy_lo, iy_hi, iy_lo, iy_hi
    ix_in = guard_in_plane(ix_in, src_ix, Nx_full)       # stromab der Quelle
    _flux = PlaneFluxDFT(omega, dt)
    for w in range(n_win):
        gx0 = _gx0(w)
        x_start_um = gx0*dx*1e6
        _last = (w == n_win - 1)
        # assemblierter Bereich dieses Fensters (global) -> lokale Flussebenen
        _glo = 0 if w == 0 else gx0 + O_cells
        _ghi = min(gx0 + Nx_win, Nx_full)
        _fpl = {k: g - gx0 for k, g in (('in', ix_in), ('out', ix_out))
                if _glo <= g < _ghi and g - gx0 >= 1}
        Nxw = Nx_win + (0 if _last else _EXT)       # Rechengebiet dieses Fensters
        _bp = {}                                   # Bead-Daten fuer TM-Tensor
        if cad is not None:
            eps_r, sig = _split_eps(cad.eps2d(x_start_um, Nxw, Ny)[0])
        else:
            eps_r, sig = build_materials(Nxw, Ny, dx, wg_n, bead_n, T_WG, bx_um, bead_d_um,
                                         x_start_um, place_bead, n_aq=n_aq_l, n_mucin=n_mu_l,
                                         n_cornea=n_co_l, t_aq_um=t_aq_um, t_mu_um=t_mucin_um,
                                         t_lip_um=t_lipid_um, n_lip=n_lip_l,
                                         input_gap_um=input_gap_um, length_um=_len_mat,
                                         curved=curved, tm_patch=_bp, bead_y_um=bead_y_um)
        Ce_E, Ce_H = _coeffs(eps_r, sig, dt, dx)
        if _pol == 'p':
            Ca_ex, Ca_ey, Ch_tm = _tm_coeffs(eps_r, dt, dx)
            _apply_bead_tm(_bp, Ca_ex, Ca_ey, dt, dx)   # TM-Tensor am Bead-Rand
            if cad is not None:                         # CAD-TM: Ey harmonisch in y
                Ca_ey[...] = xp.asarray((dt/(EPS0*dx)/cad.eps2d_ey(x_start_um, Ca_ey.shape[0], Ca_ey.shape[1] + 1)).astype(np.float32))
            Da_ex, Da_ey = _tm_loss(eps_r, sig, Ca_ex, Ca_ey, dt)   # Absorption (kappa)
        ms = MurSet(dx, dt, mur_speeds(eps_r, dx, LAM))   # Mur-Enden: lokale Geschw.
        del eps_r, sig
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
        Ez = xp.zeros((Nxw, Ny), dtype=xp.float32)
        Hx = xp.zeros((Nxw, Ny-1), dtype=xp.float32)
        Hy = xp.zeros((Nxw-1, Ny), dtype=xp.float32)
        Ex_t = xp.zeros((Nxw-1, Ny), dtype=xp.float32)
        Ey_t = xp.zeros((Nxw, Ny-1), dtype=xp.float32)
        Hz_t = xp.zeros((Nxw-1, Ny-1), dtype=xp.float32)
        # Rand-Objekt immer CPML; Kanten ohne PML (0 Zellen) bekommen Mur-1 inline.
        #  links : Fenster 0 = Eintritt (Nutzerwahl); sonst Handoff (eingepraegt)
        #  rechts: innere Fenster IMMER PML (Verlaengerung); letztes = Nutzerwahl
        #  y     : Nutzerwahl
        _pml_l = (w == 0) and _ucpml
        _pml_r = (not _last) or _ucpml
        bnd = make_boundary('cpml', _pol, Nxw, Ny, dx, dt,
                            npml_x=(NPML_ST if _pml_l else 0, NPML_ST if _pml_r else 0),
                            npml_y=(NPML_ST if _ucpml else 0), lam_m=LAM)
        acc = xp.zeros((Nxw, _Nyc), dtype=xp.complex64); acc_n = 0   # Device-DFT
        acc_ex2 = xp.zeros((Nxw, Ny), dtype=xp.float32) if _pol == 'p' else None
        acc_exa = xp.zeros((Nxw, Ny), dtype=xp.complex64) if _pol == 'p' else None
        _drive = xp.asarray(Ehand.astype(np.complex64)) if (w > 0 and Ehand is not None) else None
        for n in range(steps_win):
            t_phys = n*dt
            env = float(1 - np.exp(-((t_phys/(2*sigma_t))**2)))    # sanfter Anlauf
            if _pol == 'p':
                # --- TM (Hz, Ex, Ey) ---
                if _is_cpml_st:
                    bnd.update_H(Ex_t, Ey_t, Hz_t, Ch_tm)
                else:
                    Hz_t += -Ch_tm*((Ey_t[1:, :] - Ey_t[:-1, :]) - (Ex_t[:, 1:] - Ex_t[:, :-1]))
                ey_r = Ey_t[-2, :].copy(); ey_l = Ey_t[1, :].copy()
                ex_b = Ex_t[:, 1].copy(); ex_tp = Ex_t[:, -2].copy()
                if Da_ex is not None:                    # Absorption: E <- D E vor dem Update
                    Ex_t *= Da_ex; Ey_t *= Da_ey
                if _is_cpml_st:
                    bnd.update_E(Ex_t, Ey_t, Hz_t, Ca_ex, Ca_ey)
                else:
                    Ex_t[:, 1:-1] += Ca_ex[:, 1:-1]*(Hz_t[:, 1:] - Hz_t[:, :-1])
                    Ey_t[1:-1, :] += -Ca_ey[1:-1, :]*(Hz_t[1:, :] - Hz_t[:-1, :])
                if w == 0:
                    if tilt != 0.0:
                        Ey_t[src_ix, :] += env*xp.sin(2*np.pi*f0*t_phys + phase_y_ey)*src_p_ey
                    else:
                        Ey_t[src_ix, :] += env*float(np.sin(2*np.pi*f0*t_phys))*src_p_ey
                    if not _pml_l:                   # linkes Ende: Mur, ausser echtes CPML-Ende
                        Ey_t[0, :] = ey_l + ms.xl*(Ey_t[1, :] - Ey_t[0, :])
                else:
                    _din = env*xp.real(_drive*np.exp(1j*omega*t_phys))
                    Ey_t[:O_cells, :] = _wtap*_din + (1.0 - _wtap)*Ey_t[:O_cells, :]
                if not _pml_r:                       # rechtes Ende: Mur, ausser echtes CPML-Ende
                    Ey_t[-1, :] = ey_r + ms.xr*(Ey_t[-2, :] - Ey_t[-1, :])
                bnd.terminate(Ex_t, Ey_t)            # PEC nur an Kanten MIT PML
                if not _ucpml:                       # y-Raender: Mur-1 (Nutzerwahl)
                    Ex_t[:, 0] = ex_b + ms.bh*(Ex_t[:, 1] - Ex_t[:, 0])
                    Ex_t[:, -1] = ex_tp + ms.th*(Ex_t[:, -2] - Ex_t[:, -1])
                if n >= steps_win - n_acc:
                    acc += Ey_t.astype(xp.complex64)*np.complex64(np.exp(-1j*omega*t_phys))
                    acc_ex2 += _tm_ex2_node(Ex_t)    # <Ex^2> fuer volle |E|^2
                    acc_exa += (_tm_ex_node(Ex_t).astype(xp.complex64)
                                *np.complex64(np.exp(-1j*omega*t_phys)))
                    acc_n += 1
                    for _k, _il in _fpl.items():
                        _flux.add(_k, t_phys, pairs_2d('p', _il, Ey=Ey_t, Hz=Hz_t))
            else:
                # --- TE (Ez, Hx, Hy) ---
                if _is_cpml_st:
                    bnd.update_H(Ez, Hx, Hy, Ch)
                else:
                    Hx -= Ch*(Ez[:, 1:] - Ez[:, :-1])
                    Hy += Ch*(Ez[1:, :] - Ez[:-1, :])
                ex2 = Ez[-2, :].copy(); ey1 = Ez[:, 1].copy(); ey2 = Ez[:, -2].copy()
                ex1 = Ez[1, :].copy()
                if _is_cpml_st:
                    bnd.update_E(Ez, Hx, Hy, Ce_E, Ce_H)
                else:
                    Ez[1:-1, 1:-1] = (Ce_E[1:-1, 1:-1]*Ez[1:-1, 1:-1]
                                      + Ce_H[1:-1, 1:-1]*((Hy[1:, 1:-1] - Hy[:-1, 1:-1])
                                                          - (Hx[1:-1, 1:] - Hx[1:-1, :-1])))
                if w == 0:
                    if tilt != 0.0:
                        Ez[src_ix, :] += env*xp.sin(2*np.pi*f0*t_phys + phase_y)*src_p
                    else:
                        Ez[src_ix, :] += env*float(np.sin(2*np.pi*f0*t_phys))*src_p
                    if not _pml_l:                   # linkes Ende: Mur, ausser echtes CPML-Ende
                        Ez[0, :] = ex1 + ms.xl*(Ez[1, :] - Ez[0, :])
                else:
                    _din = env*xp.real(_drive*np.exp(1j*omega*t_phys))
                    Ez[:O_cells, :] = _wtap*_din + (1.0 - _wtap)*Ez[:O_cells, :]
                if not _pml_r:                       # rechtes Ende: Mur, ausser echtes CPML-Ende
                    Ez[-1, :] = ex2 + ms.xr*(Ez[-2, :] - Ez[-1, :])
                bnd.terminate(Ez)                    # PEC nur an Kanten MIT PML
                if not _ucpml:                       # y-Raender: Mur-1 (Nutzerwahl)
                    Ez[:, 0] = ey1 + ms.b*(Ez[:, 1] - Ez[:, 0])
                    Ez[:, -1] = ey2 + ms.t*(Ez[:, -2] - Ez[:, -1])
                if n >= steps_win - n_acc:
                    acc += Ez.astype(xp.complex64)*np.complex64(np.exp(-1j*omega*t_phys))
                    acc_n += 1
                    for _k, _il in _fpl.items():
                        _flux.add(_k, t_phys, pairs_2d('s', _il, Ez=Ez, Hy=Hy))
        Ew = to_np((2.0/max(acc_n, 1))*acc)                       # komplexe Amplitude
        if not _last:                                             # Overlap fuers naechste
            _off = _gx0(w + 1) - gx0                              # (letztes Fenster: <= S)
            Ehand = Ew[_off:_off+O_cells, :].copy()
        if w == 0:
            g_lo, l_lo = 0, 0
        else:
            g_lo, l_lo = gx0 + O_cells, O_cells
        g_hi = min(gx0 + Nx_win, Nx_full)
        l_hi = l_lo + (g_hi - g_lo)
        if g_hi > g_lo:
            Efull[g_lo:g_hi, :] = Ew[l_lo:l_hi, :]
            if ExI_full is not None:
                ExI_full[g_lo:g_hi, :] = to_np(acc_ex2[l_lo:l_hi, :])/max(acc_n, 1)
            if ExA_full is not None:
                ExA_full[g_lo:g_hi, :] = to_np(acc_exa[l_lo:l_hi, :])*(2.0/max(acc_n, 1))
        print(f'[Window {w+1}/{n_win}] x0={x_start_um:.0f}um  max|E|={np.abs(Ew).max():.3e}')

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
            frames.append({'x_start': 0.0, 'x_end': L_um, 'y_start': y0_m*1e6,
                           'y_end': (y0_m + H_total)*1e6, 'slide_i': 0,
                           'step': k, 'time': tk})
        ez_mm.flush(); del ez_mm
    # Sensor-Kennzahlen aus der assemblierten Amplitude (|E|^2 integriert;
    # Detektor-Ebenen ix_in/ix_out wurden vor der Fensterschleife festgelegt)
    P_in = float(np.sum(np.abs(Efull[ix_in, iyi0:iyi1])**2))
    P_out = float(np.sum(np.abs(Efull[ix_out, iyo0:iyo1])**2))
    if ExI_full is not None:
        # TM: |Ê|^2 = 2<E^2> -> Ex-Anteil 2*<Ex^2> addieren (volle Intensitaet)
        P_in += 2.0*float(np.sum(ExI_full[ix_in, iyi0:iyi1]))
        P_out += 2.0*float(np.sum(ExI_full[ix_out, iyo0:iyo1]))
        # Iavg = <Ey^2> + <Ex^2> auf dem Knotengitter, blockweise in x (kein RAM-Peak)
        for x0 in range(0, Nx_full, 4096):
            x1 = min(x0 + 4096, Nx_full)
            s = 0.5*np.abs(Efull[x0:x1])**2                       # <Ey^2>, (.., Ny-1)
            ey2 = np.empty((x1 - x0, Ny), dtype=np.float32)
            ey2[:, 1:-1] = 0.5*(s[:, :-1] + s[:, 1:]); ey2[:, 0] = s[:, 0]; ey2[:, -1] = s[:, -1]
            ExI_full[x0:x1] += ey2
        ExI_full.flush()
    Tr_I = P_out/max(P_in, 1e-30)                 # alt: |E_cw|^2 in einer Spalte (Kern)
    # Einheitliche Transmission: Poynting-Fluss ueber den ganzen Querschnitt (wie full),
    # exakt aus E und H (PlaneFluxDFT). Fallback (Ebene nicht erfasst): Schaetzung aus E.
    _ec = {}; _p_si = None
    if _flux.has('in') and _flux.has('out'):
        S_in, S_out = _flux.flux('in'), _flux.flux('out')
        _p_si = S_in*dx          # eingekoppelte Leistung [W/m] (2D: pro Meter Tiefe in z)
    elif _pol == 'p':                              # TM: eps-Spalten an x_in/x_out
        for _ix in (ix_in, ix_out):
            _xs = (_ix - 1)*dx*1e6
            if cad is not None:
                _e = cad.eps2d(_xs, 3, Ny)[0][1]
            else:
                _e = to_np(build_materials(3, Ny, dx, wg_n, bead_n, T_WG, bx_um, bead_d_um, _xs,
                                           place_bead, n_aq=n_aq_l, n_mucin=n_mu_l,
                                           n_cornea=n_co_l, t_aq_um=t_aq_um, t_mu_um=t_mucin_um,
                                           t_lip_um=t_lipid_um, n_lip=n_lip_l,
                                           input_gap_um=input_gap_um, length_um=L_um,
                                           curved=curved, bead_y_um=bead_y_um)[0])[1]
            _ec[_ix] = _eps_col_ey(_e)
    if not (_flux.has('in') and _flux.has('out')):
        S_in = poynting_flux(Efull, ix_in, _ec.get(ix_in))
        S_out = poynting_flux(Efull, ix_out, _ec.get(ix_out))
    Tr = S_out/S_in if abs(S_in) > 0 else Tr_I
    print(f'\n[Result] {label} [stitch]: total {total_time:.0f}s  Transmission={Tr:.4f} (poynting)'
          f'  [intensity core column: {Tr_I:.4f}]')
    res = _result_dict(label, P_in, P_out, Tr, frames, total_time, bead_d_um,
                       wg_mat, bead_mat, wg_n, bead_n, wg_thickness_um, lambda_nm,
                       vcsel_waist, vcsel_tilt, vcsel_offset, source_type, 'stitch', place_bead,
                       n_aq_l, n_mu_l, n_co_l, frames_memmap=mm_path, t_aq_um=t_aq_um,
                       t_mu_um=t_mucin_um, t_lip_um=t_lipid_um, bead_x_um=bx_um,
                       x_wg_start_um=(input_gap_um or 0.0), x_wg_end_um=L_um,
                       polarization=_pol)
    res['transmission_intensity'] = Tr_I; res['transmission_def'] = 'poynting'
    # Stitch = eingeschwungener Hauptsatz; 'einlaufend' erzeugt der Analyzer
    # illustrativ aus diesem CW-Feld (kein echter Transient vorhanden).
    res['view_primary'] = 'cw'
    res['cw_only'] = True
    res['p_in_si'] = _p_si; res['p_in_unit'] = 'W/m'   # fuer Leistungs-Skalierung (Analyzer)
    if ExI_full is not None:
        res['Iavg'] = ExI_full          # TM: volles <|E|^2> (TE: Analyzer mittelt die CW-Phasen exakt)
    if ExA_full is not None:
        ExA_full.flush()
        res['Ex_cw_amp'] = ExA_full     # TM: komplexe Ex-Amplitude (Knotengitter)
    if cad is not None:
        res['cad'] = dict(cad.meta(), cad_tilt_deg=tilt,
                          cad_det=[[ix_in, ix_in + 1, iyi0, iyi1], [ix_out, ix_out + 1, iyo0, iyo1]])
        res['cad_region'] = cad_region
    if return_efull:
        res['Efull'] = Efull            # komplexe CW-Amplitude der ganzen Laenge (Fourier-Stitch)
    return res


def run_beads_stitched_td(wg_mat, bead_mat, bead_d_um, dx_nm, save_frames, n_snapshots,
                          wg_thickness_um, lambda_nm, vcsel_waist, vcsel_tilt, vcsel_offset,
                          window_w_um, slide_um, place_bead, pulse_cycles=30.0,
                          pulse_tmax_fs=None, cad=None, t_eval_s=None, t_aqueous_um=None,
                          t_mucin_um=None, t_lipid_um=0.0, input_gap_um=0.0, length_um=None,
                          bead_x_um=None, bead_y_um=None, wg_n=None, bead_n=None, n_aqueous=None, n_mucin=None,
                          n_cornea=None, n_lipid=None, polarization='s', curved=False,
                          boundary='cpml', **_ignored):
    """TRANSIENTER Stitch direkt im ZEITBEREICH (Huygens-Ebenen, TF/SF).

    Alle Fenster laufen auf derselben absoluten Zeitachse t = n dt. Fenster w zeichnet in
    jedem Schritt an der Stelle, an der Fenster w+1 beginnt, eine Ebene auf: TE Ez(n) und
    Hy(n+1/2), TM Ey(n) und Hz(n+1/2). Fenster w+1 spielt sie als Total-Field/Scattered-
    Field-Grenze ab (i0 = NPML+2): links davon nur Streufeld (-> CPML), rechts das volle
    Feld. Weil beide Fenster an der Ebene dieselbe Geometrie haben, ist die Einkopplung
    exakt. Jedes Fenster rechnet nur solange Signal da ist (Ankunft bis Abklingen).
    Speicher: ein Fenster + zwei Zeilen je Zeitschritt. Grenze: nur Vorwaertsfluss.
    Materialien: ein n je Material (Zeitbereich, keine Dispersion)."""
    _pol = 'p' if str(polarization).lower() in ('p', 'tm') else 's'
    t_total = time.time()
    t_aq_um = t_aqueous_um if t_aqueous_um is not None else T_AQ*1e6
    wg_n = wg_n if wg_n is not None else n_at(wg_mat, lambda_nm)
    bead_n = bead_n if bead_n is not None else n_at(bead_mat, lambda_nm)
    n_aq_l = n_aqueous if n_aqueous is not None else n_at('aqueous', lambda_nm)
    n_mu_l = n_mucin if n_mucin is not None else n_at('mucin', lambda_nm)
    n_co_l = n_cornea if n_cornea is not None else cornea_default(lambda_nm)
    n_lip_l = n_lipid if n_lipid is not None else n_at('lipid', lambda_nm)
    L = (length_um*1e-6) if length_um is not None else L_DEMO
    L_um = L*1e6
    T_WG = wg_thickness_um*1e-6; LAM = lambda_nm*1e-9
    global LAM_LOSS
    LAM_LOSS = LAM                                   # kappa -> sigma bei dieser Wellenlaenge
    bx_um = bead_x_um if bead_x_um is not None else L_um/2.0
    label = f'WG-{wg_mat}_Bead-{bead_mat}_d{bead_d_um:g}_L{lambda_nm:g}' + ('' if place_bead else '_ref')
    dx = dx_nm*1e-9
    dt = 0.5*dx/(C0*np.sqrt(2))
    if cad is not None:
        _cm = cad.meta()
        label = f'CAD-{os.path.splitext(_cm["cad_file"])[0]}_z{cad.z0_um:g}_L{lambda_nm:g}'
        L = cad.lx_um*1e-6; L_um = L*1e6
        y0_m = cad.y0_um*1e-6; H_total = cad.ly_um*1e-6
        wg_n = float(_cm['cad_n'][cad.core_index - 1])
    else:
        y0_m = -TEAR_BUFFER
        H_total = T_WG + AIR_BUFFER + TEAR_BUFFER
    Ny = int(round(H_total/dx))
    Nx_win = int(round(window_w_um*1e-6/dx))
    S_cells = int(round(slide_um*1e-6/dx))
    O_cells = Nx_win - S_cells
    Nx_full = int(round(L/dx))
    NPML = 10
    i0 = NPML + 2                                   # TF/SF-Ebene (Zelle) im Folgefenster
    if O_cells < i0 + 4:
        raise ValueError(f'Overlap {O_cells} cells < {i0 + 4}: choose window_w >> slide.')
    f0 = C0/LAM; omega = 2*np.pi*f0; k0 = 2*np.pi/LAM
    sp = pulse_cycles/f0; tc = 3*sp
    t_transit = L*wg_n*1.15/C0
    t_max = (pulse_tmax_fs*1e-15) if pulse_tmax_fs else (tc + 3*sp + t_transit)
    N = int(np.ceil(t_max/dt)) + 1
    n_win = (int(np.ceil((Nx_full - Nx_win)/S_cells)) + 1) if Nx_full > Nx_win else 1

    def _gx0(w):
        return min(w*S_cells, max(Nx_full - Nx_win, 0))
    _EXT = NPML + 4
    _len_mat = L_um + _EXT*dx*1e6
    if t_eval_s is not None:
        t_j = np.asarray(t_eval_s, dtype=np.float64)
    else:
        t_j = np.linspace(0.0, t_max, max(int(n_snapshots), 24))
    n_j = np.clip(np.round(t_j/dt).astype(np.int64), 0, N - 1)   # Schritt je Frame
    print(f'\n===== Time-domain stitch (Huygens/TF-SF): {label} =====')
    print(f'Pulse {pulse_cycles:g} periods (s={sp*1e15:.1f} fs), t_max={t_max*1e15:.0f} fs = {N} steps, '
          f'{n_win} windows of {window_w_um}um (overlap {O_cells*dx*1e6:.2f}um), {len(n_j)} frames')
    # Quelle (nur Fenster 0)
    tilt = vcsel_tilt
    src_ix = NPML + 2
    if cad is not None:
        _s = cad.source(src_ix*dx*1e6)
        tilt = vcsel_tilt + _s['tilt_deg']
        iy_src, src_p, ys_idx = _src_setup(Ny, dx, T_WG, vcsel_offset, vcsel_waist,
                                           y0_m=y0_m, y_center_m=_s['y_um']*1e-6)
    else:
        iy_src, src_p, ys_idx = _src_setup(Ny, dx, T_WG, vcsel_offset, vcsel_waist)
    _n_src = _n_source_2d(cad, src_ix*dx*1e6, Ny, iy_src, y0_m, dx, wg_n, T_WG, n_aq_l, input_gap_um)
    phase_y = float(np.sin(np.radians(tilt)))*k0*_n_src*(ys_idx - iy_src)*dx   # Winkel im Medium an der Quelle
    src_p_ey = 0.5*(src_p[:-1] + src_p[1:]); phase_y_ey = 0.5*(phase_y[:-1] + phase_y[1:])
    env_t = np.exp(-((np.arange(N)*dt - tc)/sp)**2)
    os.makedirs('results', exist_ok=True)
    _safe = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in label)
    mm_path = os.path.join('results', f'_stream_beads_{_safe}_td_{os.getpid()}.npy')
    fr_mm = np.lib.format.open_memmap(mm_path, mode='w+', dtype=np.float32,
                                      shape=(len(n_j), Nx_full, Ny))
    fr_mm[:] = 0.0
    Ch = xp.float32(dt/(MU0*dx))
    _Nyc = (Ny - 1) if _pol == 'p' else Ny
    rec_in = None                                   # (E(n), H(n+1/2)) fuer dieses Fenster
    P_in = P_out = 0.0
    S_in = S_out = 0.0                                          # Poynting (Pulsenergie)
    cad_region = None
    if cad is not None:
        cad_region = np.zeros((Nx_full, Ny), dtype=np.int8)
        for _x0 in range(0, Nx_full, 4096):
            _x1 = min(_x0 + 4096, Nx_full)
            cad_region[_x0:_x1] = cad.eps2d(_x0*dx*1e6, _x1 - _x0, Ny, smooth=1)[1]
        (_a0, _a1, iyi0, iyi1), (_b0, _b1, iyo0, iyo1) = cad.detectors(cad_region, 0.0)
        ix_in = (_a0 + _a1)//2; ix_out = (_b0 + _b1)//2
    else:
        iyi0 = iyo0 = int(round(TEAR_BUFFER/dx)); iyi1 = iyo1 = int(round((T_WG + TEAR_BUFFER)/dx))
        ix_in = int(round(0.1*Nx_full)); ix_out = int(round(0.9*Nx_full))
    ix_in = guard_in_plane(ix_in, src_ix, Nx_full)       # stromab der Quelle
    for w in range(n_win):
        t1 = time.time()
        gx0 = _gx0(w); last = (w == n_win - 1)
        Nxw = Nx_win + (0 if last else _EXT)
        x_start_um = gx0*dx*1e6
        _bp = {}                                   # Bead-Daten fuer TM-Tensor
        if cad is not None:
            eps_r, sig = _split_eps(cad.eps2d(x_start_um, Nxw, Ny)[0])
        else:
            eps_r, sig = build_materials(Nxw, Ny, dx, wg_n, bead_n, T_WG, bx_um, bead_d_um,
                                         x_start_um, place_bead, n_aq=n_aq_l, n_mucin=n_mu_l,
                                         n_cornea=n_co_l, t_aq_um=t_aq_um, t_mu_um=t_mucin_um,
                                         t_lip_um=t_lipid_um, n_lip=n_lip_l,
                                         input_gap_um=input_gap_um, length_um=_len_mat, curved=curved, tm_patch=_bp, bead_y_um=bead_y_um)
        Ce_E, Ce_H = _coeffs(eps_r, sig, dt, dx)
        if _pol == 'p':
            Ca_ex, Ca_ey, Ch_tm = _tm_coeffs(eps_r, dt, dx)
            _apply_bead_tm(_bp, Ca_ex, Ca_ey, dt, dx)   # TM-Tensor am Bead-Rand
            if cad is not None:                         # CAD-TM: Ey harmonisch in y
                Ca_ey[...] = xp.asarray((dt/(EPS0*dx)/cad.eps2d_ey(x_start_um, Ca_ey.shape[0], Ca_ey.shape[1] + 1)).astype(np.float32))
            Da_ex, Da_ey = _tm_loss(eps_r, sig, Ca_ex, Ca_ey, dt)   # Absorption (kappa)
        del eps_r, sig
        bnd = make_boundary('cpml', _pol, Nxw, Ny, dx, dt, npml_x=(NPML, NPML), npml_y=NPML, lam_m=LAM)
        Ez = xp.zeros((Nxw, Ny), dtype=xp.float32); Hx = xp.zeros((Nxw, Ny-1), dtype=xp.float32)
        Hy = xp.zeros((Nxw-1, Ny), dtype=xp.float32)
        Ex_t = xp.zeros((Nxw-1, Ny), dtype=xp.float32); Ey_t = xp.zeros((Nxw, Ny-1), dtype=xp.float32)
        Hz_t = xp.zeros((Nxw-1, Ny-1), dtype=xp.float32)
        # Aufzeichnung fuer das Folgefenster: Ebene bei dessen i0 (globale Zelle)
        i_rec = (_gx0(w + 1) - gx0 + i0) if not last else None
        if i_rec is not None:
            rec_E = np.zeros((N, _Nyc), dtype=np.float32)
            rec_H = np.zeros((N, _Nyc if _pol == 's' else Ny - 1), dtype=np.float32)
        # aktiver Zeitbereich: Fenster 0 ab 0, sonst ab Eintreffen des Signals
        if w == 0:
            n_start = 0
        else:
            amp = np.abs(rec_in[0]).max(axis=1)
            nz = np.nonzero(amp > 1e-7*max(amp.max(), 1e-30))[0]
            n_start = int(nz[0]) if len(nz) else N
            n_in_last = int(nz[-1]) if len(nz) else 0
        if w == 0:
            n_in_last = int(np.nonzero(env_t > 1e-7)[0][-1])
        Ein = Hin = None
        if w > 0:
            Ein = xp.asarray(rec_in[0]); Hin = xp.asarray(rec_in[1])
        a_g, b_g = (0, Nx_win) if w == 0 else (O_cells, Nx_win)   # assemblierter Bereich (lokal)
        if last:
            b_g = Nx_full - gx0
        peak = 0.0; n_stop = N
        for n in range(n_start, N):
            t_phys = n*dt
            if _pol == 's':
                if i_rec is not None:
                    rec_E[n] = to_np(Ez[i_rec])
                bnd.update_H(Ez, Hx, Hy, Ch)
                if w > 0:
                    Hy[i0 - 1] -= Ch*Ein[n]                    # TF/SF: Hy links ist Streufeld
                if i_rec is not None:
                    rec_H[n] = to_np(Hy[i_rec - 1])
                bnd.update_E(Ez, Hx, Hy, Ce_E, Ce_H)
                if w > 0:
                    Ez[i0, 1:-1] -= Ce_H[i0, 1:-1]*Hin[n, 1:-1]   # TF/SF: Ez rechts ist Gesamtfeld
                else:
                    e = float(env_t[n])
                    if e > 1e-9:
                        Ez[src_ix, :] += e*xp.sin(omega*t_phys + phase_y)*src_p
                bnd.terminate(Ez)
                field = Ez
            else:
                if i_rec is not None:
                    rec_E[n] = to_np(Ey_t[i_rec])
                bnd.update_H(Ex_t, Ey_t, Hz_t, Ch_tm)
                if w > 0:
                    Hz_t[i0 - 1] += Ch_tm*Ein[n]
                if i_rec is not None:
                    rec_H[n] = to_np(Hz_t[i_rec - 1])
                if Da_ex is not None:                    # Absorption: E <- D E vor dem Update
                    Ex_t *= Da_ex; Ey_t *= Da_ey
                bnd.update_E(Ex_t, Ey_t, Hz_t, Ca_ex, Ca_ey)
                if w > 0:
                    Ey_t[i0] += Ca_ey[i0]*Hin[n]
                else:
                    e = float(env_t[n])
                    if e > 1e-9:
                        Ey_t[src_ix, :] += e*xp.sin(omega*t_phys + phase_y_ey)*src_p_ey
                bnd.terminate(Ex_t, Ey_t)
                field = None
            hit = np.nonzero(n_j == n)[0]
            if len(hit):
                F = to_np(_tm_node_field(Ey_t) if _pol == 'p' else Ez)
                for jj in hit:
                    fr_mm[jj, gx0 + a_g:gx0 + b_g] = F[a_g:b_g]
            # Detektoren (zeitintegriert, wie full): Kern-Ebene bei 10 % / 90 %
            for ixg, y_a, y_b, which in ((ix_in, iyi0, iyi1, 'in'), (ix_out, iyo0, iyo1, 'out')):
                il = ixg - gx0
                if a_g <= il < b_g:
                    if _pol == 's':
                        v = xp.sum(Ez[il, y_a:y_b]**2, dtype=xp.float64)        # Device, kein Sync
                    else:
                        v = xp.sum(_tm_node_field(Ey_t)[il, y_a:y_b]**2, dtype=xp.float64)
                    _s = ((_sx_plane_2d('s', il, Ez=Ez, Hy=Hy) if _pol == 's'
                           else _sx_plane_2d('p', il, Ey=Ey_t, Hz=Hz_t)) if il >= 1 else 0.0)
                    if which == 'in':
                        P_in += v; S_in += _s
                    else:
                        P_out += v; S_out += _s
            # Abbruch, wenn das Signal durch ist
            if n > n_in_last and (n - n_in_last) % 200 == 0:
                cur = float(xp.max(xp.abs(Ez if _pol == 's' else Ey_t)))
                peak = max(peak, cur)
                if cur < 1e-4*max(peak, 1e-30):
                    n_stop = n
                    break
            if n % 200 == 0:
                peak = max(peak, float(xp.max(xp.abs(Ez if _pol == 's' else Ey_t))))
        print(f'[Window {w+1}/{n_win}] x0={x_start_um:.1f}um  steps {n_start}..{n_stop} '
              f'({n_stop - n_start} of {N})  {time.time()-t1:.0f}s', flush=True)
        rec_in = (rec_E, rec_H) if i_rec is not None else None
        del Ez, Hx, Hy, Ex_t, Ey_t, Hz_t, Ce_E, Ce_H, bnd
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
    fr_mm.flush(); del fr_mm
    total_time = time.time() - t_total
    P_in = float(P_in); P_out = float(P_out)                 # einmaliger Readout
    Tr_I = P_out/max(P_in, 1e-30)
    S_in = float(S_in); S_out = float(S_out)
    Tr = S_out/S_in if abs(S_in) > 0 else Tr_I                 # Pulsenergie (Poynting)
    tr_def = 'poynting' if abs(S_in) > 0 else 'intensity'
    print(f'\n[Result] {label} [time-domain stitch]: total {total_time:.0f}s  Transmission={Tr:.4f}')
    frames = [{'x_start': 0.0, 'x_end': L_um, 'y_start': y0_m*1e6, 'y_end': (y0_m + H_total)*1e6,
               'slide_i': 0, 'step': int(n), 'time': float(n*dt)} for n in n_j]
    res = _result_dict(label, P_in, P_out, Tr, frames, total_time, bead_d_um,
                       wg_mat, bead_mat, wg_n, bead_n, wg_thickness_um, lambda_nm,
                       vcsel_waist, tilt, vcsel_offset, 'pulse', 'stitch-td', place_bead,
                       n_aq_l, n_mu_l, n_co_l, frames_memmap=mm_path, t_aq_um=t_aq_um,
                       t_mu_um=t_mucin_um, t_lip_um=t_lipid_um, bead_x_um=bx_um,
                       x_wg_start_um=(input_gap_um or 0.0), x_wg_end_um=L_um, polarization=_pol)
    res['transmission_intensity'] = Tr_I; res['transmission_def'] = tr_def
    res.update(view_primary='transient', cw_only=False,
               pulse=dict(cycles=pulse_cycles, s_fs=sp*1e15, tmax_fs=t_max*1e15, dispersion=False))
    if cad is not None:
        res['cad'] = dict(cad.meta(), cad_tilt_deg=tilt,
                          cad_det=[[ix_in, ix_in + 1, iyi0, iyi1], [ix_out, ix_out + 1, iyo0, iyo1]])
        res['cad_region'] = cad_region
    return res


def run_beads_stitched_pulse(wg_mat, bead_mat, bead_d_um, dx_nm, save_frames, n_snapshots,
                             wg_thickness_um, lambda_nm, vcsel_waist, vcsel_tilt, vcsel_offset,
                             window_w_um, slide_um, place_bead, pulse_cycles=30.0,
                             pulse_tmax_fs=None, pulse_nfreq=None, pulse_dispersion=True,
                             cad=None, t_eval_s=None, n_aq_tear_offset=None, **kw):
    """TRANSIENTER Stitch per Fourier-Synthese.

    Der Quellpuls s(t) = exp(-((t-t_c)/s)^2) sin(w0 t), s = pulse_cycles/f0, t_c = 3 s
    (identisch zur Pulsquelle des full-Laufs) wird in N Frequenzen w_k zerlegt. Fuer
    jede laeuft ein normaler CW-Stitch (mit n(lambda_k) -> echte Materialdispersion,
    abschaltbar). Dessen Amplitude E_k ist die Antwort auf die CW-Quelle sin(w t) =
    Re(-i e^{iwt}), also H_k = i E_k. Zeitverlauf:

        E(r, t) = (1/pi) Re sum_k H_k(r) S(w_k) e^{i w_k t} dw

    mit S(w) = [G(w-w0) - G(w+w0)]/(2i), G(v) = sqrt(pi) s exp(-(v s)^2/4) e^{-i v t_c}.
    dw = 2 pi/T_rep mit T_rep = 1.6 t_max (kein zeitliches Aliasing bis t_max); das Band
    umfasst |w-w0| < 5.26/s (|S| > 1e-3). Die Frames werden Frequenz fuer Frequenz
    aufsummiert -> konstanter Speicher. Nebenprodukt: Spektrum T(lambda).
    Grenze wie beim CW-Stitch: nur Vorwaertsfluss (Rueckreflexion bleibt im Fenster)."""
    import contextlib
    import io
    t_total = time.time()
    lam0 = lambda_nm*1e-9; f0 = C0/lam0; w0 = 2*np.pi*f0
    sp = pulse_cycles/f0; tc = 3*sp
    pol = 'p' if str(kw.get('polarization', 's')).lower() in ('p', 'tm') else 's'
    if cad is not None:
        _cm = cad.meta()
        L_um = cad.lx_um; n_core = float(_cm['cad_n'][cad.core_index - 1])
        y0_um, y1_um = cad.y0_um, cad.y0_um + cad.ly_um
        lam_cad0 = cad.lam_nm
    else:
        L_um = kw.get('length_um') or L_DEMO*1e6
        n_core = kw.get('wg_n') or n_at(wg_mat, lambda_nm)
        y0_um = -TEAR_BUFFER*1e6; y1_um = (wg_thickness_um*1e-6 + AIR_BUFFER)*1e6
    t_transit = L_um*1e-6*n_core*1.15/C0                 # Gruppenindex ~ n + 15 %
    t_max = (pulse_tmax_fs*1e-15) if pulse_tmax_fs else (tc + 3*sp + t_transit)
    half = 5.26/sp                                       # |S| > 1e-3 innerhalb +-half
    if pulse_nfreq:
        dw = 2*half/max(int(pulse_nfreq) - 1, 1)
        if 2*np.pi/dw < t_max:
            print(f'  [WARN] {pulse_nfreq} frequencies -> repetition period {2*np.pi/dw*1e15:.0f} fs '
                  f'< t_max {t_max*1e15:.0f} fs: temporal aliasing.')
    else:
        dw = 2*np.pi/(1.6*t_max)
    kk = np.arange(-int(np.ceil(half/dw)), int(np.ceil(half/dw)) + 1)
    w_k = w0 + kk*dw
    w_k = w_k[w_k > 0]
    lam_k = 2*np.pi*C0/w_k*1e9

    def G(v):
        return np.sqrt(np.pi)*sp*np.exp(-(v*sp)**2/4.0)*np.exp(-1j*v*tc)
    S_k = (G(w_k - w0) - G(w_k + w0))/(2j)
    # ohne Dispersion: alle Brechzahlen fest bei lambda0 (fuer Vergleich mit full-Lauf)
    over = {}
    if not pulse_dispersion and cad is None:
        over = dict(wg_n=kw.get('wg_n') or n_at(wg_mat, lambda_nm),
                    bead_n=kw.get('bead_n') or n_at(bead_mat, lambda_nm),
                    n_aqueous=kw.get('n_aqueous') or n_at('aqueous', lambda_nm),
                    n_mucin=kw.get('n_mucin') or n_at('mucin', lambda_nm),
                    n_cornea=kw.get('n_cornea') or cornea_default(lambda_nm),
                    n_lipid=kw.get('n_lipid') or n_at('lipid', lambda_nm))
    kw2 = {k: v for k, v in kw.items() if k not in over}
    # Linsen-Szenario: Traenen-n folgt n_aqueous(lambda_k) + Szenario-Offset (statt fest)
    _tear_disp = pulse_dispersion and n_aq_tear_offset is not None and cad is None
    n_t = max(int(n_snapshots), 24)
    t_j = np.linspace(0.0, t_max, n_t)
    if t_eval_s is not None:                             # feste Zeitpunkte (z.B. Vergleich mit full)
        t_j = np.asarray(t_eval_s, dtype=np.float64); n_t = len(t_j)
    label = (f'CAD-{os.path.splitext(_cm["cad_file"])[0]}_z{cad.z0_um:g}_L{lambda_nm:g}'
             if cad is not None else
             f'WG-{wg_mat}_Bead-{bead_mat}_d{bead_d_um:g}_L{lambda_nm:g}' + ('' if place_bead else '_ref'))
    print(f'\n===== Fourier-Stitch (transient): {label} =====')
    print(f'Pulse: {pulse_cycles:g} periods (s={sp*1e15:.1f} fs), Band {lam_k.min():.1f}..{lam_k.max():.1f} nm, '
          f'{len(w_k)} frequencies, dw/2pi={dw/2/np.pi/1e12:.2f} THz, t_max={t_max*1e15:.0f} fs, '
          f'{n_t} time frames, dispersion {"on" if pulse_dispersion else "off (n at lambda0)"}')
    frames_t = None; spec_T = []; spec_P = []; r_c = None; E_c = None
    k_c = int(np.argmin(np.abs(w_k - w0)))
    for k, (wk, lk, Sk) in enumerate(zip(w_k, lam_k, S_k)):
        if cad is not None and pulse_dispersion:
            cad.lam_nm = float(lk)                       # CAD-Materialien bei lambda_k
        t1 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            _kwk = dict(kw2)
            if _tear_disp:
                _kwk['n_aqueous'] = n_at('aqueous', float(lk)) + float(n_aq_tear_offset)
            r = run_beads_stitched(wg_mat, bead_mat, bead_d_um, dx_nm, False, 4,
                                   wg_thickness_um, float(lk), vcsel_waist, vcsel_tilt,
                                   vcsel_offset, 'cw', window_w_um, slide_um, place_bead,
                                   cad=cad, return_efull=True, **over, **_kwk)
        E = np.asarray(r.pop('Efull'))
        if pol == 'p':                                   # Ey (Nx,Ny-1) -> Knotengitter (Nx,Ny)
            En = np.empty((E.shape[0], E.shape[1] + 1), dtype=np.complex64)
            En[:, 1:-1] = 0.5*(E[:, :-1] + E[:, 1:]); En[:, 0] = E[:, 0]; En[:, -1] = E[:, -1]
            E = En
        if frames_t is None:
            os.makedirs('results', exist_ok=True)
            _safe = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in label)
            mm_path = os.path.join('results', f'_stream_beads_{_safe}_pulse_{os.getpid()}.npy')
            frames_t = np.lib.format.open_memmap(mm_path, mode='w+', dtype=np.float32,
                                                 shape=(n_t,) + E.shape)
            frames_t[:] = 0.0
        c = (1j*E)*(Sk*dw/np.pi)                         # H_k S_k dw/pi
        for j, tj in enumerate(t_j):
            frames_t[j] += np.real(c*np.exp(1j*wk*tj)).astype(np.float32)
        spec_T.append(float(r['transmission'])); spec_P.append(float(r['P_out']))
        if k == k_c:
            r_c, E_c = r, E
        print(f'  [{k+1}/{len(w_k)}] lambda={lk:.2f} nm  |S|={abs(Sk)/abs(S_k).max():.3f}  '
              f'T={r["transmission"]:.4f}  ({time.time()-t1:.0f}s)', flush=True)
    if cad is not None:
        cad.lam_nm = lam_cad0
    frames_t.flush()
    frames = [{'x_start': 0.0, 'x_end': L_um, 'y_start': y0_um, 'y_end': y1_um,
               'slide_i': 0, 'step': j, 'time': float(tj)} for j, tj in enumerate(t_j)]
    # eingeschwungener Satz bei der Mittenfrequenz (zweite Ansicht im Analyzer)
    cw_path = os.path.join('results', f'_stream_beads_{_safe}_pulse_cw_{os.getpid()}.npy')
    nfr_cw = max(1, n_snapshots)
    cw_mm = np.lib.format.open_memmap(cw_path, mode='w+', dtype=np.float32,
                                      shape=(nfr_cw,) + E_c.shape)
    cw_frames = []
    for k in range(nfr_cw):
        tk = (k/nfr_cw)*(1.0/(w_k[k_c]/2/np.pi))
        cw_mm[k] = np.real(E_c*np.exp(1j*w_k[k_c]*tk)).astype(np.float32)
        cw_frames.append({'x_start': 0.0, 'x_end': L_um, 'y_start': y0_um, 'y_end': y1_um,
                          'slide_i': 0, 'step': k, 'time': tk})
    cw_mm.flush(); del cw_mm
    del frames_t
    res = dict(r_c)
    res.update(scenario=label, method='stitch-pulse', frames=frames, frames_memmap=mm_path,
               cw_frames=cw_frames, cw_frames_memmap=cw_path, view_primary='transient',
               cw_only=False, Iavg=None, source_type='pulse', lambda_nm=lambda_nm,
               total_time=time.time() - t_total,
               spectrum=dict(lam_nm=lam_k, T=np.array(spec_T), P_out=np.array(spec_P),
                             weight=np.abs(S_k)/np.abs(S_k).max()),
               pulse=dict(cycles=pulse_cycles, s_fs=sp*1e15, tmax_fs=t_max*1e15,
                          nfreq=len(w_k), dispersion=bool(pulse_dispersion)))
    w = np.abs(S_k)**2
    res['transmission'] = float(np.sum(w*np.array(spec_T))/np.sum(w))   # spektral gewichtet
    print(f'\n[Result] {label} [Fourier-Stitch]: {len(w_k)} frequencies in {res["total_time"]:.0f}s  '
          f'T(weighted)={res["transmission"]:.4f}')
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
    summary = {k: v for k, v in result.items()
               if k not in ('frames', 'cad_region', 'Iavg', 'Ex_cw_amp')}
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
                print(f'  [WARN] x-stride={_sx} does not fit in RAM, trying coarser...')
        if ezs is None:
            print('  [ERROR] Frames do not fit in RAM - NPZ not saved.')
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
    # exaktes Zeitmittel <E^2> mitspeichern (gleiche Strides wie die Frames), damit
    # 'Int-Mittel' im Analyzer die echte Envelope zeigt statt eines Snapshot-Mittels
    _ia = result.get('Iavg'); _kw_ia = {}
    if _ia is not None:
        try:
            _kw_ia = dict(Iavg=np.asarray(_ia)[::(used_stride or 1), ::_sy].astype(np.float32),
                          iavg_full=np.array([1]))   # Iavg = <|E|^2> aller Komponenten
        except Exception:
            _kw_ia = {}
    # TM: komplexe CW-Amplitude von Ex (Knotengitter, gleiche Strides) -> im Analyzer
    # sind Ex (tangential, stetig), Ey (normal, springt) und |E| getrennt waehlbar
    _exa = result.get('Ex_cw_amp')
    if _exa is not None:
        try:
            _kw_ia['Ex_cw_amp'] = np.asarray(_exa)[::(used_stride or 1), ::_sy].astype(np.complex64)
        except Exception:
            pass
    if result.get('p_in_si') is not None:      # eingekoppelte Leistung (SI) -> Leistungs-Skalierung
        _kw_ia['p_in_si'] = np.array([float(result['p_in_si'])])
        _kw_ia['p_in_unit'] = np.array([str(result.get('p_in_unit', 'W/m'))])
    npz = f'{out_dir}/beads_{tag}_frames.npz'
    kw = dict(Ez=ezs, meta=meta, scenario=result['scenario'],
              layers=np.array(result['layers'], dtype=np.float64),
              # Kern, 3 Schichten unter dem Kern, Substrat - Namen frei (--layer-names);
              # der Bead steht getrennt (bead_n/bead_material), nicht mehr auf Platz 2.
              material_names=np.array(result.get('layer_names')
                                      or ['WG', 'Lipid', 'Aqueous', 'Mucin', 'Cornea']),
              material_indices=np.array([result['wg_n'], result.get('n_lip', N_LIPID),
                                         result['n_aq'], result['n_mucin'], result['n_cornea']],
                                        dtype=np.float64),
              bead_n=np.array([result.get('bead_n', 0.0)]),
              bead_material=np.array([str(result.get('bead_material', ''))]),
              bead_x_um=np.array([result.get('bead_x_um', BEAD_X_UM)]),
              **({'bead_y_um': np.array([result['bead_y_um']])} if result.get('bead_y_um') is not None else {}),
              bead_diameter_um=np.array([result.get('bead_diameter_um', 0.0)]),
              t_wg_um=np.array([result.get('wg_thickness_um', 5.0)]),
              x_wg_start_um=np.array([result.get('x_wg_start_um', 0.0) or 0.0]),
              x_wg_end_um=np.array([result.get('x_wg_end_um', 0.0) or 0.0]),
              polarization=np.array([str(result.get('polarization', 's'))]),
              lam_nm=np.array([result.get('lambda_nm', 850.0)]))
    _cad = result.get('cad')
    if _cad:
        # Freie CAD-Geometrie: Regionskarte (gleiche Strides wie die Frames) + Tabelle der
        # Koerper. Kein t_wg_um/Bead (gibt es hier nicht) -> der Analyzer nutzt die Regionen.
        _reg = result.get('cad_region')
        _st = (used_stride or 1, _sy)
        kw.update(geometry=np.array(['cad']),
                  material_names=np.array(_cad['cad_names']),
                  material_indices=np.array(_cad['cad_n'], dtype=np.float64),
                  cad_materials=np.array(_cad['cad_materials']),
                  cad_core=np.array([_cad['cad_core']]), cad_tear=np.array([_cad['cad_tear']]),
                  cad_file=np.array([_cad['cad_file']]),
                  cad_tilt_deg=np.array([_cad['cad_tilt_deg']]),
                  cad_det=np.array(_cad['cad_det'], dtype=np.int64),
                  cad_stride=np.array(_st, dtype=np.int64),
                  cad_x_shift_um=np.array([_cad['cad_x_shift_um']]),
                  cad_plane_z_um=np.array([_cad['cad_plane_z_um']]),
                  cad_scale_um=np.array([_cad['cad_scale_um']]))
        if _reg is not None:
            kw['cad_region'] = np.asarray(_reg)[::_st[0], ::_st[1]].astype(np.int8)
        for k in ('t_wg_um', 'bead_x_um', 'bead_diameter_um', 'x_wg_start_um', 'x_wg_end_um',
                  'bead_n', 'bead_material', 'bead_y_um'):
            kw.pop(k, None)
    _sp = result.get('spectrum')
    if _sp:                                       # Fourier-Stitch: Spektrum + Pulsdaten
        _pu = result.get('pulse') or {}
        kw.update(spec_lam_nm=np.asarray(_sp['lam_nm'], dtype=np.float64),
                  spec_T=np.asarray(_sp['T'], dtype=np.float64),
                  spec_weight=np.asarray(_sp['weight'], dtype=np.float64),
                  pulse_cycles=np.array([_pu.get('cycles', 0.0)]),
                  pulse_tmax_fs=np.array([_pu.get('tmax_fs', 0.0)]),
                  pulse_dispersion=np.array([int(bool(_pu.get('dispersion', True)))]))
    np.savez_compressed(npz, **kw, **cw_kw, **_flags, **_kw_ia)
    print(f'  Saved: {npz} (stride={used_stride}, {ezs.nbytes/1e6:.1f} MB)'
          + ('  +CW set' if cw_kw else '') + ('  [cw_only]' if _cwonly else ''))
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


import inspect as _inspect                                  # noqa: E402
_RUN_BEADS_ARGS = [k for k in _inspect.signature(run_beads).parameters]
