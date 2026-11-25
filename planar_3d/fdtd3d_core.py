"""Echter 3D-FDTD-Kern (volle Yee-Zelle) fuer die planaren Demos.

Rechnet alle 6 Feldkomponenten (Ex, Ey, Ez, Hx, Hy, Hz) auf einem
gestaggerten Yee-Gitter. Im Gegensatz zu den 2D-Demos ist der Bead hier
eine ECHTE KUGEL und die Quelle ein echter Gauss-Spot (Taille in y UND z).

Koordinaten (wie 2D-Demos):
  x : Propagationsrichtung entlang Waveguide
  y : Schichtstapel (y=0 = WG-Unterseite, Tear-Film bei y<0, Luft oben)
  z : NEU - laterale Richtung (z=0 = Domainmitte, Bead-Zentrum)

Yee-Staggering (E auf Kanten, H auf Flaechen):
  Ex (Nx-1, Ny,   Nz  )    Hx (Nx,   Ny-1, Nz-1)
  Ey (Nx,   Ny-1, Nz  )    Hy (Nx-1, Ny,   Nz-1)
  Ez (Nx,   Ny,   Nz-1)    Hz (Nx-1, Ny-1, Nz  )

Absorber: statt Mur-1.Ordnung (in 3D 12 Face/Komponenten-Updates) wird ein
GRADED SPONGE benutzt: in den aeussersten n_sponge Zellen werden E- UND
H-Felder pro Step mit exp(-alpha*(tiefe/n_sponge)^3) gedaempft.
Gleiche Daempfung auf E und H wirkt wie ein angepasstes verlustbehaftetes
Medium -> Restreflexion typ. <1% Amplitude, auch bei schraegem Einfall.

Polarisation der Quelle: Ez (parallel zu den Schichten, senkrecht zur
Propagation) -> entspricht der TMz/TE-Polarisation der 2D-Demos.

Domaingroesse beachten: Zellen = (lx*ly*lz)/dx^3.
Default 40 x 16 x 8 um @ 50 nm = 41M Zellen ~= 1.4 GB GPU-RAM.
"""
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
# Backend (GPU/CPU) und Physik/Materialdaten aus den geteilten Modulen.
from common.backend import xp, cp, GPU_AVAILABLE, to_np           # noqa: E402
from common.physics import C0, EPS0, MU0, N_AIR, DISPERSION, n_at, cornea_default  # noqa: E402
from common.physics import core_source_mask, SRC_CORE_RAMP_UM  # noqa: E402
from common.subpixel import sphere_fraction, blend, cell_average_column  # noqa: E402
from common.flux import PlaneFluxDFT, pairs_3d, guard_in_plane  # noqa: E402
from common import vcsel                                         # noqa: E402
from common.layer_stack import (stack_total_um, legacy_stack, legacy_layers_tuple,  # noqa: E402
                                stack_to_npz, describe as describe_stack)
BEAD_SUBPIXEL = True     # Bead-Rand per Volumenanteil glaetten (False = harte Maske)
# Absorption je Schicht (Extinktion kappa, n = n' + i kappa), gesetzt vom Treiber
# (run_simulation: args.kappa). Schluessel wg, lip, aq, mu, co, bead. Leer = verlustfrei.
KAPPA = {}
# Quelle nur im Kern einpraegen (--source-core-only), gesetzt vom Treiber
SRC_CORE_ONLY = False
# optische VCSEL-Leistung [mW] (--laser-power), gesetzt vom Treiber (Analyzer-Skalierung)
LASER_POWER_MW = vcsel.VCSEL_DEFAULTS['power_mw']
LAM_LOSS = 850e-9        # Wellenlaenge fuer kappa -> sigma (je Lauf gesetzt)
from planar_3d.boundaries3d import make_boundary_3d, mur_speeds_3d               # noqa: E402


def _stack_lossy(stack=None):
    """Absorption aktiv? KAPPA (Legacy-Schluessel) oder eine Stapelschicht mit Im(n) > 0."""
    if any(v > 0 for v in (KAPPA or {}).values()):
        return True
    return bool(stack) and any(complex(Ly['n']).imag > 0 for Ly in stack)


def _stack_banner(stack, bead=None):
    """Konsolen-Info zum Stapel + Hinweise (ignorierte kappa, Bead ausserhalb Schicht 1)."""
    if stack is None:
        return
    print(f'Layers: {describe_stack(stack)}')
    _ig = [k for k in ('lip', 'aq', 'mu') if (KAPPA or {}).get(k, 0) > 0]
    if _ig:
        print(f'  [Note] layer stack given: kappa for {", ".join(_ig)} ignored '
              f'(set absorption as complex n in --layers)')
    if bead and bead.get('d_um', 0) > 0:
        r = 0.5*float(bead['d_um']); yc = float(bead.get('y_um', -r))
        if not stack:
            print(f'  [WARN] Bead d={bead["d_um"]:g}um: no layers below the core -> the bead '
                  f'sits in the substrate.')
        elif yc - r < -float(stack[0]['t_um']) - 1e-6 or yc + r > 1e-6:
            print(f'  [WARN] Bead d={bead["d_um"]:g}um (y {yc - r:.3g}..{yc + r:.3g}um) sticks out '
                  f'of the first layer "{stack[0]["name"]}" ({stack[0]["t_um"]:g}um below the core).')


def _stack_n_below(stack, n_sub):
    """Reelle Brechzahl direkt unter dem Kern: erste Stapelschicht, sonst Substrat."""
    return float(complex(stack[0]['n']).real) if stack else float(n_sub)


def stack_at_lambda(stack, lam_nm):
    """Stapel bei einer anderen Wellenlaenge: Schichten mit Material (``mat``) folgen
    n_at(mat, lambda) (kappa bleibt), numerische n bleiben konstant."""
    if stack is None:
        return None
    out = []
    for Ly in stack:
        Ly = dict(Ly)
        if Ly.get('mat'):
            n = complex(n_at(Ly['mat'], lam_nm), complex(Ly['n']).imag)
            Ly.update(n=n, kappa=n.imag)
        out.append(Ly)
    return out


def _result_stack(stack, t_lip_um, t_aq_um, t_mu_um, n_lip, n_aq, n_mu):
    """(stack, stack_custom, layers-4-Tupel) fuers Ergebnis-Dict. Legacy: Stapel aus den
    drei festen Schichten (inkl. KAPPA), layers wie bisher."""
    if stack is not None:
        return [dict(Ly) for Ly in stack], True, legacy_layers_tuple(stack)
    _kp = dict(KAPPA or {})
    st = legacy_stack(t_lip_um, complex(float(np.real(n_lip)), _kp.get('lip', 0.0)),
                      t_aq_um, complex(float(np.real(n_aq)), _kp.get('aq', 0.0)),
                      t_mu_um, complex(float(np.real(n_mu)), _kp.get('mu', 0.0)))
    return st, False, (t_lip_um*1e-6, t_aq_um*1e-6, t_mu_um*1e-6, n_aq)


def build_eps3d(Nx, Ny, Nz, dx_um, y0_um, z0_um,
                wg_n, t_wg_um, t_lip_um, t_aq_um, t_mu_um,
                n_lip, n_aq, n_mu, n_co, bead=None,
                wg_width_um=None, n_clad_side=N_AIR, end_facet_um=0.0,
                x_wg_start_um=0.0, x0_um=None, x_wg_end_um=None, tensor_patch=None,
                stack=None):
    """eps_r auf (Nx,Ny,Nz). Schichtstapel in y, optional Bead-KUGEL.
    bead = dict(x_um=..., d_um=..., n=...) - Kugel-Zentrum bei
    (x_um, -d/2, 0), d.h. aufliegend an der WG-Unterseite wie im 2D-Setup.

    wg_width_um: Breite des WG-KERNS in Tiefenrichtung (z). None oder >= Domain
      -> Kern fuellt die ganze Tiefe (Slab, nur y-Fuehrung, wie zuvor).
      Endlicher Wert -> echter RECHTECK-KANAL: Kern nur bei |z|<wg_width/2,
      seitlich (in z) von Cladding n_clad_side umgeben -> Fuehrung in y UND z.
      Die Tear-/Cornea-Schichten UNTER dem Kern bleiben ueber die volle Tiefe
      erhalten (Substrat).

    stack: frei konfigurierbarer Schichtstapel unter dem Kern (common.layer_stack, vom
      Kern abwaerts, n komplex = absorbierend). None = Legacy Lipid/Aqueous/Mucin aus
      t_*/n_* (unveraendert). Mit stack werden t_lip/aq/mu, n_lip/aq/mu und
      KAPPA['lip'/'aq'/'mu'] ignoriert; das Substrat n_co beginnt am Stapelboden."""
    # Globaler x-Modus (Stitch): xs bezieht sich auf globale Koordinaten mit
    # Offset x0_um; der WG-KERN existiert nur im Bereich [x_wg_start, x_wg_end]
    # (endlicher Slab mit Facetten an BEIDEN Enden). Ist x0_um None -> altes
    # Verhalten (lokale x ab 0, WG fuellt ganze Laenge, nur Front-Gap/Endfacette).
    _global_x = x0_um is not None
    xs_g = (float(x0_um) if _global_x else 0.0) + np.arange(Nx)*dx_um
    # Absorption: intern komplexes n (Rueckgabe complex, _split_eps3 trennt eps'/eps'')
    _k = dict(KAPPA or {})
    lossy = _stack_lossy(stack)
    _k_co = _k.get('co', 0.0) if lossy else 0.0
    if lossy:
        wg_n = complex(wg_n, _k.get('wg', 0.0)); n_lip = complex(n_lip, _k.get('lip', 0.0))
        n_aq = complex(n_aq, _k.get('aq', 0.0)); n_mu = complex(n_mu, _k.get('mu', 0.0))
        if bead is not None:
            bead = dict(bead, n_c=complex(bead['n'], _k.get('bead', 0.0)))
    _edt = np.complex128 if lossy else np.float64
    ys = y0_um + np.arange(Ny)*dx_um
    y_lip_bot = -t_lip_um
    y_aq_bot = y_lip_bot - t_aq_um
    y_mu_bot = y_aq_bot - t_mu_um
    _sb = None
    if stack is not None:
        # Schichtgrenzen (um) vom Kern (y=0) abwaerts; Substrat beginnt am Stapelboden
        _sb, _yt = [], 0.0
        for Ly in stack:
            _yb = _yt - float(Ly['t_um'])
            _n = complex(Ly['n'])
            _sb.append((_yt, _yb, _n if lossy else _n.real))
            _yt = _yb
        y_mu_bot = _yt

    def _eps_y(y, core, n_core=None):
        """Schichtstapel eps(y) exakt (um) - mit oder ohne Kern (n_core: Kern-Brechzahl,
        Default wg_n; fuer die Cladding-Spalten des Rechteck-Kanals n_clad_side)."""
        e = np.full(y.shape, N_AIR**2, dtype=_edt)
        mc = y < y_mu_bot
        if hasattr(n_co, 'at_depth'):                # geschichtete Cornea (Epithel, Stroma ...)
            e[mc] = ((n_co.at_depth(y_mu_bot - y[mc]) + 1j*_k_co)**2 if lossy
                     else n_co.at_depth(y_mu_bot - y[mc])**2)
        else:
            e[mc] = (float(n_co) + 1j*_k_co)**2 if lossy else n_co**2
        if _sb is not None:                          # frei konfigurierbarer Stapel
            for _yt_, _yb_, _n_ in _sb:
                e[(y >= _yb_) & (y < _yt_)] = _n_**2
        else:                                        # Legacy: Lipid / Aqueous / Mucin
            e[(y >= y_mu_bot) & (y < y_aq_bot)] = n_mu**2
            e[(y >= y_aq_bot) & (y < y_lip_bot)] = n_aq**2
            if t_lip_um > 1e-6:
                e[(y >= y_lip_bot) & (y < 0.0)] = n_lip**2
        if core:
            e[(y >= 0.0) & (y < t_wg_um)] = (wg_n if n_core is None else n_core)**2
        return e
    # Subpixel an den Schichtgrenzen: arithmetisches Zellmittel an den Knoten; die
    # E-Koeffizienten werden daraus in _ce_arrays an die versetzten Positionen gemittelt
    # (harmonisch entlang der Versatzachse -> Ey senkrecht auf den Schichten korrekt)
    col_out = cell_average_column(ys, dx_um, lambda y: _eps_y(y, False))
    col_core = cell_average_column(ys, dx_um, lambda y: _eps_y(y, True))
    col = col_out if _global_x else col_core
    eps = np.broadcast_to(col[None, :, None], (Nx, Ny, Nz)).copy()
    core_y = (ys >= 0.0) & (ys <= t_wg_um)
    if _global_x:
        # WG-Kern (in y) nur wo der Slab in x liegt: [x_wg_start, x_wg_end]
        in_wg_x = (xs_g >= (x_wg_start_um or 0.0))
        if x_wg_end_um is not None:
            in_wg_x &= (xs_g <= x_wg_end_um)
        if in_wg_x.any():
            eps[in_wg_x, :, :] = col_core[None, :, None]

    # Laterale (z-)Fuehrung: WG-Kern auf endliche Tiefe begrenzen -> Rechteck-Kanal
    lz_um = Nz*dx_um
    if wg_width_um is not None and 0 < wg_width_um < lz_um:
        zs = z0_um + np.arange(Nz)*dx_um
        core_y = (ys >= 0.0) & (ys <= t_wg_um)      # Kern-Hoehenbereich
        outside_z = np.abs(zs) > (wg_width_um/2.0)   # ausserhalb der Kernbreite
        # nur die KERN-Zellen ausserhalb |z|<w/2 auf Cladding setzen; im globalen
        # x-Modus nur dort, wo der Kern in x ueberhaupt existiert.
        yz = core_y[None, :, None] & outside_z[None, None, :]
        if _global_x:
            _inx = (xs_g >= (x_wg_start_um or 0.0))
            if x_wg_end_um is not None:
                _inx &= (xs_g <= x_wg_end_um)
            yzc = _inx[:, None, None] & yz
            eps[np.broadcast_to(yzc, eps.shape)] = n_clad_side**2
        else:
            eps[np.broadcast_to(yz, eps.shape)] = n_clad_side**2

    # Im globalen x-Modus ist der WG bereits endlich [x_wg_start, x_wg_end] gesetzt
    # (Luft davor/dahinter) -> Front-Gap und Endfacette sind implizit vorhanden.
    if not _global_x:
        # Endfacette: Material->Luft am WG-Ende (grosses x) -> Fresnel-Reflexion.
        if end_facet_um and end_facet_um > 0:
            lx_um = Nx*dx_um
            xs = np.arange(Nx)*dx_um
            eps[xs >= (lx_um - end_facet_um), :, :] = N_AIR**2
        # Einkoppelabstand: Luftzone am Anfang (x < x_wg_start).
        if x_wg_start_um and x_wg_start_um > 0:
            xs0 = np.arange(Nx)*dx_um
            eps[xs0 < x_wg_start_um, :, :] = N_AIR**2

    if tensor_patch is not None:
        # Ey bei (i, j+1/2, k) steht SENKRECHT auf den Schichten -> exaktes harmonisches
        # Mittel 1/<1/eps> ueber [y_j, y_{j+1}] (wie der 2D-TM-Kern). Je (x,z)-Spalte
        # einer von vier Stapeltypen: 0 ohne Kern, 1 mit Kern, 2 Kanal-Cladding, 3 Luft.
        # _ce_arrays setzt daraus ce_y.
        def _hcol(core, n_core=None):
            return cell_average_column(ys[:-1], dx_um,
                                       lambda y: np.real(_eps_y(y, core, n_core)),
                                       harmonic=True, shift=0.5*dx_um).astype(np.float64)
        T = np.full((Nx, Nz), 0 if _global_x else 1, dtype=np.int8)
        if _global_x and in_wg_x.any():
            T[in_wg_x, :] = 1
        if wg_width_um is not None and 0 < wg_width_um < lz_um:
            rows = _inx if _global_x else np.ones(Nx, dtype=bool)
            T[np.ix_(rows, outside_z)] = 2
        if not _global_x:
            if end_facet_um and end_facet_um > 0:
                T[xs >= (lx_um - end_facet_um), :] = 3
            if x_wg_start_um and x_wg_start_um > 0:
                T[xs0 < x_wg_start_um, :] = 3
        tensor_patch['layer'] = dict(
            T=T, cols=np.stack([_hcol(False), _hcol(True), _hcol(True, n_clad_side),
                                np.full(Ny - 1, N_AIR**2)]))

    if bead is not None and bead.get('d_um', 0) > 0:
        r = bead['d_um']/2.0
        xc = bead['x_um']
        yc = bead.get('y_um', -r)   # Default: aufliegend an WG-Unterseite (y=-r)
        zc = bead.get('z_um', 0.0)  # Default: Domainmitte (z=0)
        _x0 = float(x0_um) if _global_x else 0.0
        xs = _x0 + np.arange(Nx)*dx_um
        zs = z0_um + np.arange(Nz)*dx_um
        # Bounding-Box (Speicher schonen), in globalen x-Koordinaten
        i0 = max(0, int((xc - r - _x0)/dx_um) - 2); i1 = min(Nx, int((xc + r - _x0)/dx_um) + 3)
        j0 = max(0, int((yc - r - y0_um)/dx_um) - 2); j1 = min(Ny, int((yc + r - y0_um)/dx_um) + 3)
        k0 = max(0, int((zc - r - z0_um)/dx_um) - 2); k1 = min(Nz, int((zc + r - z0_um)/dx_um) + 3)
        if i1 > i0 and j1 > j0 and k1 > k0:
            XX = xs[i0:i1, None, None] - xc
            YY = ys[None, j0:j1, None] - yc
            ZZ = zs[None, None, k0:k1] - zc
            sub = eps[i0:i1, j0:j1, k0:k1]
            if BEAD_SUBPIXEL and tensor_patch is not None:
                # fuer die anisotrope Mittelung je E-Komponente: eps OHNE Bead
                tensor_patch.update(eps_bg=sub.real.astype(np.float64), i0=i0, j0=j0, k0=k0,
                                    xs=xs[i0:i1], ys=ys[j0:j1], zs=zs[k0:k1],
                                    xc=xc, yc=yc, zc=zc, r=r, n=bead['n'])
            if BEAD_SUBPIXEL:
                # Subpixel (Review F12): Volumenanteil der Kugel in den Randzellen
                blend(sub, sphere_fraction(xs[i0:i1], ys[j0:j1], zs[k0:k1], xc, yc, zc, r,
                                           dx_um), bead.get('n_c', bead['n']))
            else:                                    # alte harte Maske (Vergleich)
                mask = (XX*XX + YY*YY + ZZ*ZZ) <= r*r
                sub[mask] = bead.get('n_c', bead['n'])**2
    return eps


def _staggered_inv(inv, ax):
    """inv (Knoten) -> Position der E-Komponente ax (um dx/2 entlang ax versetzt):
    Mittel der beiden Nachbarknoten = HARMONISCHES eps-Mittel. Entlang ax steht die
    Komponente senkrecht auf Grenzflaechen normal zu ax (Ey auf den Schichten, Ex auf
    den Facetten, Ez auf den Kanalwaenden) -> dort ist <1/eps> das richtige Mittel."""
    a = [slice(None)]*3; b = [slice(None)]*3
    a[ax] = slice(None, -1); b[ax] = slice(1, None)
    return (xp.float32(0.5)*(inv[tuple(a)] + inv[tuple(b)])).astype(xp.float32)


def _ce_arrays(inv, tp, dt, dx, dx_um, da=None):
    """Update-Koeffizienten ce_x/ce_y/ce_z an den VERSETZTEN Yee-Positionen der
    E-Komponenten (eigene Arrays, 3 Volumen; inv kann danach freigegeben werden).
    Frueher wurde inv am Knoten i benutzt -> Ey (TM-Hauptkomponente) sah den
    Schichtstapel um dx/2 verschoben und arithmetisch statt harmonisch gemittelt
    (2D macht das in _tm_coeffs/_apply_bead_tm richtig).
    Mit Bead (tp gefuellt) am Bead-Rand zusaetzlich die anisotrope Subpixel-Mittelung
    (Farjadpour et al., Opt. Lett. 31, 2972 (2006); Review F12):
    (1/eps)_ii = n_i^2 <1/eps> + (1 - n_i^2)/<eps>,  n = Kugelnormale,
    <..> = Volumenmittel ueber die Zelle um die jeweilige Komponentenposition.
    tp['layer'] (aus build_eps3d): exakte harmonische Schichtmittelung fuer Ey; der
    Verlustfaktor 1/(1+a) = (1+D)/2 (da = D aus _inv_loss) wird dabei mitgenommen."""
    ces = (_staggered_inv(inv, 0), _staggered_inv(inv, 1), _staggered_inv(inv, 2))
    tp = tp or {}
    lp = tp.get('layer')
    if lp is not None:
        cy = ces[1]
        hinv = xp.asarray((dt/(EPS0*dx)/lp['cols']).astype(np.float32))   # (4, Ny-1)
        T = xp.asarray(lp['T'])[:, None, :]                                # (Nx, 1, Nz)
        for t in range(hinv.shape[0]):
            xp.copyto(cy, hinv[t][None, :, None], where=(T == t))
        if da is not None:                                                 # Absorption
            cy *= _staggered_inv(xp.float32(0.5)*(1 + da), 1)
    if 'eps_bg' not in tp:
        return ces
    from common.subpixel import sphere_fraction
    c = dt/(EPS0*dx)
    eb = tp['n']**2
    out = []
    for ax, ce in enumerate(ces):
        # Hintergrund-eps an die versetzte Position: harmonisch (<1/eps>, fuer den Anteil
        # senkrecht zur Kugel) und arithmetisch (<eps>, tangential). Fuer Ey (ax=1) ist das
        # harmonische Mittel der Schichten exakt bekannt (tp['layer']) -> am Beruehrpunkt
        # Bead/Kern stimmt dann auch die Normalkomponente an der Schichtgrenze.
        e0 = tp['eps_bg']
        nb_ax = e0.shape[ax]
        e_nx = np.take(e0, np.minimum(np.arange(nb_ax) + 1, nb_ax - 1), axis=ax)
        e_bg_a = 0.5*(e0 + e_nx)
        e_bg_h = 1.0/(0.5*(1.0/e0 + 1.0/e_nx))
        if ax == 1 and lp is not None:
            i0_, j0_, k0_ = tp['i0'], tp['j0'], tp['k0']
            Tb = lp['T'][i0_:i0_ + e0.shape[0], k0_:k0_ + e0.shape[2]]      # (nx, nz)
            cols = lp['cols'][:, j0_:j0_ + e0.shape[1]]                      # (4, <=ny)
            hb = np.transpose(cols[Tb], (0, 2, 1))                           # (nx, <=ny, nz)
            e_bg_h[:, :hb.shape[1], :] = hb
        fac = None
        if da is not None:                                   # Verlustfaktor 1/(1+a), versetzt
            fac = _staggered_inv(xp.float32(0.5)*(1 + da), ax)
        o = [0.0, 0.0, 0.0]; o[ax] = 0.5*dx_um              # Komponente bei +dx/2 in Achse ax
        xs, ys, zs = tp['xs'] + o[0], tp['ys'] + o[1], tp['zs'] + o[2]
        a0, b0, c0 = tp['i0'], tp['j0'], tp['k0']
        na = min(len(xs), ce.shape[0] - a0); nb = min(len(ys), ce.shape[1] - b0)
        nc = min(len(zs), ce.shape[2] - c0)
        if na <= 0 or nb <= 0 or nc <= 0:
            out.append(ce); continue
        xs, ys, zs = xs[:na], ys[:nb], zs[:nc]
        f = sphere_fraction(xs, ys, zs, tp['xc'], tp['yc'], tp['zc'], tp['r'], dx_um)
        m = f > 0
        if m.any():
            X = xs[:, None, None] - tp['xc']; Y = ys[None, :, None] - tp['yc']
            Z = zs[None, None, :] - tp['zc']
            d = np.maximum(np.sqrt(X*X + Y*Y + Z*Z), 1e-30)
            nc2 = ((X, Y, Z)[ax]/d)**2
            ebh = e_bg_h[:na, :nb, :nc]; eba = e_bg_a[:na, :nb, :nc]
            inv_t = nc2*((1 - f)/ebh + f/eb) + (1 - nc2)/((1 - f)*eba + f*eb)
            sl = (slice(a0, a0 + na), slice(b0, b0 + nb), slice(c0, c0 + nc))
            blk = to_np(ce[sl]).astype(np.float64)
            val = c*np.broadcast_to(inv_t, f.shape)
            if fac is not None:                              # Absorption auch am Bead-Rand
                val = val*to_np(fac[sl]).astype(np.float64)
            blk[m] = val[m]
            ce[sl] = xp.asarray(blk.astype(np.float32))
        out.append(ce)
    return tuple(out)


def _sponge_cells(n_sponge, sponge_um, dx_um, shape=None, warn=True):
    """Sponge-Dicke in Zellen. Standard: feste Dicke in um (sponge_um), damit die
    Reflexion bei Verfeinerung von dx NICHT steigt (24 feste Zellen waren bei 20 nm nur
    0.48 um -> |r|~1e-2 statt 1e-4). n_sponge (Zellen) ueberschreibt, falls gesetzt.
    shape: Gitter -> hoechstens N/6 je Achse (mit Warnung, wenn das kappt)."""
    n = int(n_sponge) if n_sponge else int(np.ceil(float(sponge_um)/dx_um))
    n = max(8, n)
    if shape is not None:
        cap = max(8, min(s//6 for s in shape))
        if n > cap and warn:
            print(f'  [WARN] Sponge {n} cells ({n*dx_um:.2f} um) capped to {cap} (grid too small) '
                  f'-> weaker absorption; enlarge the domain.')
        n = min(n, cap)
    return n


def _sponge_profile(n_sp, alpha):
    """Daempfungsfaktoren pro Step, Index 0 = aeusserste Zelle."""
    u = (n_sp - np.arange(n_sp, dtype=np.float64))/n_sp
    return xp.asarray(np.exp(-alpha*u**3), dtype=xp.float32)


ALL_FACES = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')


def _apply_sponge(F, g, n_sp, faces=ALL_FACES):
    """Daempft Feld F in den Randslabs - nur auf den angegebenen (absorbierenden)
    Flaechen. PEC-Flaechen werden hier ausgelassen (dort keine Daempfung)."""
    if 'xmin' in faces: F[:n_sp] *= g[:, None, None]
    if 'xmax' in faces: F[-n_sp:] *= g[::-1][:, None, None]
    if 'ymin' in faces: F[:, :n_sp] *= g[None, :, None]
    if 'ymax' in faces: F[:, -n_sp:] *= g[::-1][None, :, None]
    if 'zmin' in faces: F[:, :, :n_sp] *= g[None, None, :]
    if 'zmax' in faces: F[:, :, -n_sp:] *= g[None, None, ::-1]


def _apply_pec(Ex, Ey, Ez, faces):
    """Perfekter elektrischer Leiter (Spiegel): tangentiales E = 0 auf der Flaeche
    -> ~100%% Reflexion. Ex(Nx-1,Ny,Nz), Ey(Nx,Ny-1,Nz), Ez(Nx,Ny,Nz-1)."""
    if 'xmin' in faces: Ey[0] = 0; Ez[0] = 0
    if 'xmax' in faces: Ey[-1] = 0; Ez[-1] = 0
    if 'ymin' in faces: Ex[:, 0] = 0; Ez[:, 0] = 0
    if 'ymax' in faces: Ex[:, -1] = 0; Ez[:, -1] = 0
    if 'zmin' in faces: Ex[:, :, 0] = 0; Ey[:, :, 0] = 0
    if 'zmax' in faces: Ex[:, :, -1] = 0; Ey[:, :, -1] = 0


def _colocate_E(Ex, Ey, Ez):
    """Mittelt die versetzten Yee-E-Komponenten auf gemeinsame ZELLZENTREN
    -> alle Shape (Nx-1, Ny-1, Nz-1), damit Ex,Ey,Ez,|E| am selben Punkt
    vergleichbar sind.  Ex(Nx-1,Ny,Nz), Ey(Nx,Ny-1,Nz), Ez(Nx,Ny,Nz-1)."""
    exc = 0.25*(Ex[:, :-1, :-1] + Ex[:, 1:, :-1] + Ex[:, :-1, 1:] + Ex[:, 1:, 1:])
    eyc = 0.25*(Ey[:-1, :, :-1] + Ey[1:, :, :-1] + Ey[:-1, :, 1:] + Ey[1:, :, 1:])
    ezc = 0.25*(Ez[:-1, :-1, :] + Ez[1:, :-1, :] + Ez[:-1, 1:, :] + Ez[1:, 1:, :])
    return exc, eyc, ezc


def _sq_on_grid(F, tshape, x0, x1):
    """F^2 auf das Zielgitter tshape gebracht, nur Zeilen x0..x1 (Zielindizes).
    Versetzte Achsen werden per QUADRAT-Mittelung angeglichen: eine Zelle zu kurz
    -> Nachbarn mitteln + Rand kopieren; eine zu lang -> benachbarte mitteln."""
    nF = F.shape[0]
    if nF == tshape[0]:
        S = F[x0:x1]**2
    else:                                          # nF == tshape[0]-1 (Ex auf Knoten-x)
        i = xp.arange(x0, x1)
        S = 0.5*(F[xp.clip(i - 1, 0, nF - 1)]**2 + F[xp.clip(i, 0, nF - 1)]**2)
    for ax in (1, 2):
        n, t = S.shape[ax], tshape[ax]
        if n == t:
            continue
        a = [slice(None)]*3; b = [slice(None)]*3
        a[ax] = slice(None, -1); b[ax] = slice(1, None)
        mid = 0.5*(S[tuple(a)] + S[tuple(b)])
        if n == t + 1:
            S = mid
        else:                                      # n == t-1: auf t aufweiten
            first = [slice(None)]*3; last = [slice(None)]*3
            first[ax] = slice(0, 1); last[ax] = slice(-1, None)
            S = xp.concatenate([S[tuple(first)], mid, S[tuple(last)]], axis=ax)
    return S


def _accum_E2(I, Ex, Ey, Ez):
    """I (Gitter der getriebenen Hauptkomponente) += |E|^2 = Ex^2+Ey^2+Ez^2.
    x-blockweise, damit die Temporaries klein bleiben (VRAM). Alle drei
    Komponenten: auch die nicht getriebenen (Beugung, TM-Ex im evaneszenten Feld)
    tragen zur Intensitaet bei."""
    Nx = I.shape[0]
    ch = max(32, Nx//8)
    for x0 in range(0, Nx, ch):
        x1 = min(x0 + ch, Nx)
        I[x0:x1] += (_sq_on_grid(Ex, I.shape, x0, x1) + _sq_on_grid(Ey, I.shape, x0, x1)
                     + _sq_on_grid(Ez, I.shape, x0, x1))


def _box_E2(Ex, Ey, Ez, xs, ys, zs):
    """Summe |E|^2 aller drei Komponenten ueber eine Detektor-Box (Index-Slices;
    der halbe Yee-Versatz ist fuer Box-Integrale vernachlaessigbar)."""
    # Device-Skalar (kein GPU->Host-Sync pro Zeitschritt; Readout einmal am Ende)
    return sum(xp.sum(F[xs, ys, zs]**2, dtype=xp.float64) for F in (Ex, Ey, Ez))


def _resolve_pol(polarization, Ny, Nz):
    """Normalisiert die Polarisation und liefert das Gitter der getriebenen
    Hauptkomponente. 's'/TE -> Ez (Ny, Nz-1); 'p'/TM -> Ey (Ny-1, Nz).
    Rueckgabe (pol, Nj, Nk). Geteilt von run_3d und run_3d_stitched."""
    pol = str(polarization).lower()
    pol = 'p' if pol in ('p', 'tm') else 's'
    Nj, Nk = (Ny-1, Nz) if pol == 'p' else (Ny, Nz-1)
    return pol, Nj, Nk


def _build_gauss_source(n_sp, t_wg_um, y0_um, z0_um, dx_um, dx, LAM, Nj, Nk,
                        vcsel_waist_um, vcsel_waist_z_um,
                        vcsel_offset_y_um, vcsel_offset_z_um, vcsel_tilt_deg,
                        center_y_um=None, center_z_um=None, n_src_fn=None, pol='s',
                        wg_width_um=None, dt=None, z_prop_um=0.0):
    """VCSEL-Gauss-Strahl (Taillen y/z, Wellenfrontkruemmung, Neigung) als einseitige
    Huygens-Quelle (common/vcsel.py) auf der getriebenen Komponente in der Ebene src_ix.
    Zentrum standardmaessig WG-Mitte (y=t_wg/2) und z=0; center_y_um/center_z_um
    (CAD: automatisch auf die Eintrittsflaeche ausgerichtet) ersetzen das. z_prop_um:
    Lage der Quellebene relativ zur Taille (z < 0: Taille stromab). Rueckgabe
    (src_ix, j_src, k_src, src). Geteilt von allen 3D-Methoden."""
    src_ix = n_sp + 6
    cy = t_wg_um/2 if center_y_um is None else center_y_um
    cz = 0.0 if center_z_um is None else center_z_um
    yc = cy + vcsel_offset_y_um; zc = cz + vcsel_offset_z_um
    j_src = min(max(int(round((yc - y0_um)/dx_um)), 0), Nj-1)
    k_src = min(max(int(round((zc - z0_um)/dx_um)), 0), Nk-1)
    # Gitterlagen der getriebenen Komponente: Ey (p) bei j+1/2, Ez (s) bei k+1/2
    _yv = y0_um + (np.arange(Nj) + (0.5 if pol == 'p' else 0.0))*dx_um
    _zv = z0_um + (np.arange(Nk) + (0.5 if pol == 's' else 0.0))*dx_um
    _n = float(n_src_fn(j_src, k_src)) if n_src_fn is not None else 1.0
    lam_nm = LAM*1e9
    amp, ph = vcsel.profile_2d(_yv, _zv, yc, zc, z_prop_um, vcsel_waist_um, vcsel_waist_z_um,
                               lam_nm, _n, vcsel_tilt_deg)
    if SRC_CORE_ONLY:
        if center_y_um is None:
            _m = core_source_mask(_yv[:, None], t_wg_um, z_um=_zv[None, :],
                                  wg_width_um=wg_width_um)
            amp = amp*_m
            print(f'  Source in core only (--source-core-only, soft edge {SRC_CORE_RAMP_UM:g} um)')
        else:
            print('  [Note] --source-core-only applies to the layer stack only, not to CAD')
    if abs(vcsel_tilt_deg) > 1e-9:
        print(f'  Source tilt {vcsel_tilt_deg:g} deg in the medium n={_n:.3f} at the source')
    kdx = vcsel.numerical_kx_dx(lam_nm, dx, C0*dt, _n, vcsel_tilt_deg)
    src = vcsel.HuygensSource(src_ix, amp, ph, 2*np.pi*C0/LAM, kdx)
    return src_ix, j_src, k_src, src


def _vcsel_geom_3d(n_sp, dx_um, distance_um):
    """VCSEL vor der Facette: (x_src_um, x_facette_um, Luftweg_um, z_um an der Quellebene)."""
    x_src_um = (n_sp + 6)*dx_um
    air = vcsel.air_path_um(distance_um, dx_um)
    return x_src_um, x_src_um + air, air, vcsel.z_at_source(distance_um, air)


def _cad_air_um_3d(_s, x_src_um):
    """CAD: Abstand Quellpunkt -> Facettenmitte [um]."""
    fc = np.asarray(_s['facet_center'], dtype=float)
    return float(np.sqrt((fc[0] - x_src_um)**2 + (fc[1] - _s['y_um'])**2
                         + (fc[2] - _s['z_um'])**2))


def _vcsel_res(p_emit, p_refl, p_in, distance_um, air_um, x_facet_um, waist_um, waist_z_um,
               energy=False):
    """VCSEL-Kennwerte fuers Ergebnis (W; energy=True: Pulsenergie J, keine Skalierung)."""
    r = dict(laser_power_mw=LASER_POWER_MW, vcsel_distance_um=float(distance_um or 0.0),
             vcsel_air_um=float(air_um), x_facet_um=float(x_facet_um),
             vcsel_waist_um=float(waist_um), vcsel_waist_z_um=float(waist_z_um),
             p_src_si=(None if energy else p_emit))
    if p_emit and p_emit > 0:
        eta = (p_in/p_emit) if p_in is not None else float('nan')
        refl = (p_refl/p_emit) if p_refl is not None else float('nan')
        r.update(coupling_efficiency=eta, facet_reflection=refl)
        print(f'[VCSEL] emitted {"energy" if energy else "P"}={p_emit:.4e} {"J" if energy else "W"}  '
              f'facet reflection={100*refl:.2f}%  coupled through input plane: eta={100*eta:.2f}%'
              f'  -> at {LASER_POWER_MW:g} mW: {LASER_POWER_MW*eta:.3f} mW')
    return r


def _n_source_fn_3d(cad, x_src_um, Ny, Nz, y0_um, dx_um, wg_n, t_wg_um, n_below,
                    x_wg_start_um, pol):
    """Liefert n(j, k) am Quellpunkt (fuer die Strahlneigung im Medium). CAD: aus der
    Szene; eingebauter Stapel: Luft vor dem WG-Eintritt / ueber dem Kern, Kern, sonst
    die Schicht darunter. j zaehlt auf dem Gitter der getriebenen Komponente."""
    def fn(j, k):
        try:
            if cad is not None:
                e = np.asarray(cad.eps3d(x_src_um - dx_um, 3, Ny, Nz)[0])[1]
                return float(np.sqrt(max(float(np.real(e[min(j, Ny - 1), min(k, Nz - 1)])), 1.0)))
        except Exception:
            return 1.0
        if x_wg_start_um and x_src_um < x_wg_start_um:
            return float(N_AIR)
        y = y0_um + (j + (0.5 if pol == 'p' else 0.0))*dx_um
        if 0.0 <= y < t_wg_um:
            return float(np.real(wg_n))
        return float(N_AIR) if y >= t_wg_um else float(np.real(n_below))
    return fn


def _split_eps3(eps):
    """eps (reell oder komplex: Absorption) -> (eps' float32 Host, eps'' float32 oder None)."""
    eps = np.asarray(to_np(eps))
    if np.iscomplexobj(eps):
        return eps.real.astype(np.float32), eps.imag.astype(np.float32)
    return eps, None


def _inv_loss(eps, eim, dt, dx, lam_m):
    """Update-Faktor inv = dt/(eps0 eps' dx) und Daempfung D (Absorption, wie TE-2D):
        E^{n+1} = D E^n + inv/(1+a) curl H,  a = sigma dt/(2 eps0 eps') = w0 dt eps''/(2 eps'),
    D = (1-a)/(1+a). Rueckgabe (inv, D) auf dem Device; D = None ohne Verlust."""
    e = xp.asarray(eps)
    inv = xp.float32(dt/(EPS0*dx))/e
    if eim is None or not np.any(eim > 0):
        return inv, None
    a = (xp.float32(np.pi*C0/lam_m*dt)*xp.asarray(eim))/e      # w0 dt eps''/(2 eps')
    inv /= (1 + a)
    return inv, ((1 - a)/(1 + a)).astype(xp.float32)


def _stagger_da(da):
    """D (Knoten) -> (D_x, D_y, D_z) an den versetzten Positionen der E-Komponenten
    (Mittel der Nachbarknoten entlang der Versatzachse, wie die ce-Koeffizienten).
    Frueher wurde D am Knoten angesetzt -> absorbierende Schicht fuer Ey um dx/2
    verschoben. None bleibt None (verlustfrei)."""
    if da is None:
        return None
    return tuple(_staggered_inv(da, ax) for ax in range(3))


def _damp(Ex, Ey, Ez, da):
    """Absorption: E <- D E (zwischen H- und E-Update). da = (D_x, D_y, D_z) aus
    _stagger_da (je Komponente auf deren versetzter Position) oder None."""
    if da is None:
        return
    Ex *= da[0]
    Ey *= da[1]
    Ez *= da[2]


def _yee_step(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, Ch, da=None):
    """Ein voller Yee-Leapfrog-Schritt: H-Update, dann E-Update (Innenbereich).
    Aktualisiert alle sechs Felder IN PLACE. Identisch fuer run_3d und
    run_3d_stitched (zuvor im heissen Loop dupliziert)."""
    _yee_H(Ex, Ey, Ez, Hx, Hy, Hz, Ch)
    _damp(Ex, Ey, Ez, da)
    _yee_E(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z)


def _yee_H(Ex, Ey, Ez, Hx, Hy, Hz, Ch):
    """H-Halbschritt (n-1/2 -> n+1/2). Getrennt, damit die TF/SF-Kopplung des
    Zeitbereichs-Stitch zwischen H- und E-Update eingreifen kann."""
    Hx += Ch*((Ey[:, :, 1:] - Ey[:, :, :-1]) - (Ez[:, 1:, :] - Ez[:, :-1, :]))
    Hy += Ch*((Ez[1:, :, :] - Ez[:-1, :, :]) - (Ex[:, :, 1:] - Ex[:, :, :-1]))
    Hz += Ch*((Ex[:, 1:, :] - Ex[:, :-1, :]) - (Ey[1:, :, :] - Ey[:-1, :, :]))


def _yee_E(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z):
    """E-Halbschritt (n -> n+1), Innenbereich."""
    Ex[:, 1:-1, 1:-1] += ce_x[:, 1:-1, 1:-1]*(
        (Hz[:, 1:, 1:-1] - Hz[:, :-1, 1:-1]) - (Hy[:, 1:-1, 1:] - Hy[:, 1:-1, :-1]))
    Ey[1:-1, :, 1:-1] += ce_y[1:-1, :, 1:-1]*(
        (Hx[1:-1, :, 1:] - Hx[1:-1, :, :-1]) - (Hz[1:, :, 1:-1] - Hz[:-1, :, 1:-1]))
    Ez[1:-1, 1:-1, :] += ce_z[1:-1, 1:-1, :]*(
        (Hy[1:, 1:-1, :] - Hy[:-1, 1:-1, :]) - (Hx[1:-1, 1:, :] - Hx[1:-1, :-1, :]))


def _gpu_free_gb():
    """(frei, gesamt) GPU-VRAM in GB, oder None wenn keine GPU/nicht abfragbar."""
    if not GPU_AVAILABLE:
        return None
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        return free/1e9, total/1e9
    except Exception:
        return None


def _host_free_gb():
    """Freier System-RAM in GB (via psutil), oder None wenn nicht ermittelbar."""
    try:
        import psutil
        return psutil.virtual_memory().available/1e9
    except Exception:
        return None


def check_resources_3d(Nx, Ny, Nz, n_snapshots, vol_dtype, save_vector,
                       dx_nm, safety=0.85, boundary='cpml', bead=False, lossy=None):
    """Schaetzt VRAM- (Felder) und Host-RAM-Bedarf (Snapshot-Puffer) und
    vergleicht mit dem tatsaechlich Freien. Rueckgabe (ok, report, hints).
    Aendert NICHTS - nur Pruefung + konkrete Empfehlungen.
    boundary='cpml' rechnet die psi-Randstreifen mit ein.
    lossy (None = aus KAPPA): Absorption haelt D_x/D_y/D_z dauerhaft (3 Volumen) und
    beim Aufbau kurzzeitig D + Verlustfaktor (2 weitere) -> +5 Volumen im Peak."""
    if lossy is None:
        lossy = any(v > 0 for v in (KAPPA or {}).values())
    cells = Nx*Ny*Nz
    # GPU-Peak: 6 Felder + inv_eps + Iavg (8) + Temporaere aus den Curl-Updates
    # + CuPy-Pool-Overhead/Fragmentierung. Erfahrungswert ~16x float32 (real gemessen:
    # 55M Zellen -> OOM bei 3.5 GB frei, also Peak > 3.5 GB).
    # CPML: psi liegt nur in den PML-Randstreifen (4 psi je Achse, 2*npml dick)
    # -> 32*npml*(NyNz+NxNz+NxNy) Byte, statt 12 vollen Volumina.
    _cpml_b = (32*10*(Ny*Nz + Nx*Nz + Nx*Ny)
               if str(boundary or '').lower() == 'cpml' else 0)
    # Gemessen (CuPy-Pool-Peak, 25M Zellen, CPML, Vektor-Modus): ~99 Byte/Zelle = ~25
    # float32-Volumen (6 Felder, ce_x/y/z, CW-/Iavg-Akkumulatoren, Curl-Temporaere je
    # Komponente). Frueher 18 angesetzt -> 36M-Zellen-Fall als "passt" gemeldet, dann OOM.
    vram_need = (cells*(26 + (5 if lossy else 0))*4/1e9   # Absorption: D_x/y/z (+Aufbau)
                 + _cpml_b/1e9)
    bpp = 2 if vol_dtype == 'float16' else 4
    ncomp = 3 if save_vector else 1
    # Host: build_eps3d legt das volle eps-Array (float32) im RAM an (broadcast+copy)
    # -> das ist die ERSTE grosse Host-Allokation (haeufigste OOM-Ursache). Dazu die
    # Snapshot-Puffer (+30% fuer Kompression/Save).
    host_eps = cells*4/1e9
    host_need = host_eps + n_snapshots*cells*bpp*ncomp/1e9*1.3
    lines = [f'[Resource check] Domain {Nx}x{Ny}x{Nz} = {cells/1e6:.0f}M cells, '
             f'{n_snapshots} snapshots, vector={"yes" if save_vector else "no"}']
    ok = True
    hints = []
    g = _gpu_free_gb()
    if g:
        free, total = g
        lines.append(f'  GPU VRAM:  need ~{vram_need:.1f} GB   '
                     f'free {free:.1f} / {total:.1f} GB')
        if vram_need > safety*free:
            ok = False
            f = vram_need/(safety*max(free, 1e-6))
            hints.append(f'increase RES to ~{dx_nm*f**(1/3):.0f} nm '
                         f'OR reduce LENGTH/DEPTH together by a factor of ~{f**0.5:.2f}')
    else:
        lines.append(f'  GPU VRAM:  need ~{vram_need:.1f} GB   (free: unknown)')
    h = _host_free_gb()
    if h is not None:
        lines.append(f'  Host RAM:  need ~{host_need:.1f} GB   free {h:.1f} GB '
                     f'(eps array ~{host_eps:.1f} GB + snapshot buffer)')
        if host_need > safety*h:
            ok = False
            new_ns = max(1, int(safety*h/(host_need/max(n_snapshots, 1))))
            hints.append(f'reduce SNAPSHOTS to ~{new_ns}'
                         + (' or set VECTOR=0' if save_vector else ''))
    else:
        lines.append(f'  Host RAM:  need ~{host_need:.1f} GB   (free: unknown, '
                     f'psutil missing)')
    return ok, '\n'.join(lines), hints


def _resolve_tear_um(tear_um, t_lip_um, t_aq_um, t_mu_um, n_sponge, dx_um, cornea_um=1.5,
                     stack=None):
    """tear_um (Gebiets-Hoehe unter dem WG) aufloesen. None -> AUTO aus den
    physikalischen Schichten + Cornea-Marge + Absorber-Zone (2D macht das ueber
    einen festen Puffer; 3D leitet es hier ab, damit man es nicht doppelt angeben
    muss). Explizit gesetzt -> unveraendert (zum Trimmen der Cornea-Marge)."""
    if tear_um is not None:
        return float(tear_um)
    absorber = n_sponge*dx_um
    if stack is not None:                          # frei konfigurierbarer Stapel
        layers = stack_total_um(stack)
        tot = layers + cornea_um + absorber
        print(f'  [Auto] tear = {tot:.2f}um (layer stack {layers:.2f} ({len(stack)} layers) + '
              f'Cornea {cornea_um:g} + Absorber {absorber:.2f}um)')
        return tot
    layers = t_lip_um + t_aq_um + t_mu_um
    tot = layers + cornea_um + absorber
    print(f'  [Auto] tear = {tot:.2f}um (Lipid+Aqueous+Mucin {layers:.2f} + '
          f'Cornea {cornea_um:g} + Absorber {absorber:.2f}um)')
    return tot


# Laufzeit-Marge (Gruppenindex) und CW-Konvergenz (wie planar_beads.fdtd2d_core)
N_G_FACTOR = 1.10
CW_TOL = 0.01
CW_MAX_EXTEND = 2.0


def _sx_plane_3d(il, Ey, Ez, Hy, Hz):
    """Momentaner Netto-Fluss in +x durch die Ebene x=il (ganzer Querschnitt):
    S_x = Ey*Hz - Ez*Hy, H (bei i+1/2) auf die E-Ebene gemittelt. Device-Skalar."""
    hz = 0.5*(Hz[il - 1] + Hz[il])                 # (Ny-1, Nz)  passt zu Ey[il]
    hy = 0.5*(Hy[il - 1] + Hy[il])                 # (Ny, Nz-1)  passt zu Ez[il]
    return (xp.sum(Ey[il]*hz, dtype=xp.float64) - xp.sum(Ez[il]*hy, dtype=xp.float64))


def _poynting_flux_3d(planes, eps_plane=None):
    """Netto-Fluss in +x durch die mittlere von drei x-Ebenen (planes: (3, Nj, Nk) komplex),
    ueber den ganzen Querschnitt: Sx ~ Im(Ê dÊ*/dx) (TE exakt, TM mit eps(y,z) gewichtet)."""
    d = 0.5*(planes[2] - planes[0])
    s = np.imag(planes[1]*np.conj(d))
    if eps_plane is not None:
        s = s*np.asarray(eps_plane, dtype=np.float64)[:s.shape[0], :s.shape[1]]
    return float(np.sum(s))


def _eps_planes_3d(cad, ix, dx_um, Ny, Nz, y0_um, z0_um, wg_n, t_wg_um, t_lip_um, t_aq_um,
                   t_mu_um, n_lip, n_aq, n_mu, n_co, bead, wg_width_um, n_clad_side,
                   end_facet_um, x_wg_start_um, stack=None):
    """eps(y,z) der x-Ebene ix (fuer die TM-Gewichtung des Poynting-Flusses)."""
    x0 = (ix - 1)*dx_um
    if cad is not None:
        return np.asarray(cad.eps3d(x0, 3, Ny, Nz)[0][1]).real
    e = build_eps3d(3, Ny, Nz, dx_um, y0_um, z0_um, wg_n, t_wg_um, t_lip_um, t_aq_um, t_mu_um,
                    n_lip, n_aq, n_mu, n_co, bead, wg_width_um=wg_width_um,
                    n_clad_side=n_clad_side, x_wg_start_um=x_wg_start_um, x0_um=x0, stack=stack)
    return np.asarray(e[1]).real


def run_3d(label, wg_n, t_wg_um=5.0,
           t_lip_um=0.0, t_aq_um=3.5, t_mu_um=0.5,
           n_lip=1.480, n_aq=1.336, n_mu=1.342, n_co=1.376,
           lam_nm=850.0,
           lx_um=40.0, air_um=3.0, tear_um=None, lz_um=8.0,
           dx_nm=50.0, vcsel_waist_um=3.0, vcsel_waist_z_um=3.0,
           source_type='cw', n_snapshots=8, steps_factor=2.0,
           bead=None, n_sponge=None, sponge_um=1.2, sponge_alpha=0.3,
           det_in_um=None, det_out_um=None, vol_dtype='float32',
           calibrate_steps=0, wg_width_um=None, n_clad_side=N_AIR,
           save_vector=False, check_resources=False,
           pec_faces=(), end_facet_um=0.0,
           vcsel_tilt_deg=0.0, vcsel_offset_y_um=0.0, vcsel_offset_z_um=0.0,
           input_gap_um=0.0, polarization='s', allow_large=False, boundary='cpml',
           cad=None, pulse_cycles=30.0, stack=None):
    """Volle 3D-FDTD-Simulation. Rueckgabe: result-dict mit 3D-Volumina.

    stack: frei konfigurierbarer Schichtstapel unter dem Kern (common.layer_stack);
    None = Legacy Lipid/Aqueous/Mucin aus t_*/n_* (unveraendert).

    cad: optional common.cad_geometry.CadRun (dim=3) -> freie CAD-Geometrie: Box,
    eps, Quelle (auf die Eintrittsflaeche) und Detektoren kommen aus der Szene.

    polarization: 's'/'TE' -> Quelle treibt Ez (E entlang Tiefe z, senkrecht zur
    Einfallsebene x-y). 'p'/'TM' -> Quelle treibt Ey (E in der Einfallsebene, ent-
    lang Schichtdicke y). Der Solver ist voll vektoriell; nur die getriebene und
    die gespeicherte Hauptkomponente wechseln.

    calibrate_steps>0: nur kurze Messung (Warmup + N Steps), druckt die
    gemessene Zell-Update-Rate und die HOCHRECHNUNG auf die volle Laufzeit,
    kehrt dann zurueck OHNE vollen Lauf/Speichern."""
    t_total = time.time()
    # GPU-Speicherpools VOR dem Lauf leeren (Hygiene). ACHTUNG: befreit nur den
    # Speicher DIESES Prozesses - Reste aus fruehreren, abgestuerzten Prozessen
    # bekommt nur ein GPU-Reset / Neustart zurueck.
    if GPU_AVAILABLE:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            free0, total0 = cp.cuda.runtime.memGetInfo()
            print(f'[GPU memory] before run: free {free0/1e9:.1f} / {total0/1e9:.1f} GB')
        except Exception:
            pass
    dx = dx_nm*1e-9
    dx_um = dx_nm/1000.0
    dt = 0.5*dx/(C0*np.sqrt(3))          # CFL 3D: dt <= dx/(c*sqrt(3)), Faktor 0.5
    global LAM_LOSS
    LAM = lam_nm*1e-9; LAM_LOSS = LAM
    f0 = C0/LAM
    sigma_t = 4/(2*np.pi*f0)

    if cad is not None:                   # freie CAD-Geometrie: Box aus der Szene
        _cm = cad.meta()
        lx_um, ly_um, lz_um = cad.lx_um, cad.ly_um, cad.lz_um
        y0_um, z0_um = cad.y0_um, cad.z0_um
        wg_n = float(_cm['cad_n'][cad.core_index - 1])
        t_wg_um = 0.0
    else:
        tear_um = _resolve_tear_um(tear_um, t_lip_um, t_aq_um, t_mu_um,
                                   (10 if str(boundary or 'cpml').lower() == 'cpml' else
                                    0 if str(boundary or '').lower() == 'mur1' else
                                    _sponge_cells(n_sponge, sponge_um, dx_um)), dx_um,
                                   stack=stack)
        ly_um = t_wg_um + air_um + tear_um
        y0_um = -tear_um                  # y bei j=0
        z0_um = -lz_um/2.0                # z bei k=0
    Nx = int(round(lx_um/dx_um))
    Ny = int(round(ly_um/dx_um))
    Nz = int(round(lz_um/dx_um))
    n_sp = _sponge_cells(n_sponge, sponge_um, dx_um, (Nx, Ny, Nz),
                         warn=str(boundary or 'cpml').lower() == 'sponge')

    cells = Nx*Ny*Nz
    gb = cells*4*8/1e9                    # 6 Felder + inv_eps + Iavg
    cpl = (LAM/wg_n)/dx
    print(f'\n===== 3D-FDTD: {label} =====')
    print(f'Axes:    x = LENGTH        = LIGHT PROPAGATION / COUPLING DIRECTION (source at x=0)')
    print(f'         y = CROSS-SECTION = layer stack (air / waveguide / tear film)')
    print(f'         z = DEPTH         = 3rd dimension (not present in 2D)')
    print(f'Domain:  length {lx_um} x cross-section {ly_um} x depth {lz_um} um '
          f'= {Nx} x {Ny} x {Nz} ({cells/1e6:.1f}M cells, ~{gb:.1f} GB fields)')
    print(f'dx={dx_nm}nm ({cpl:.1f} cells/lambda in WG), dt={dt*1e15:.4f} fs, '
          f'Sponge={n_sp} cells')
    if cpl < 8:
        print('!!! WARNING: <8 cells/wavelength in WG - result distorted by numerical dispersion.')
    if wg_width_um is not None and 0 < wg_width_um < lz_um:
        print(f'Waveguide: RECTANGULAR CHANNEL {t_wg_um} um (cross-section/y) '
              f'x {wg_width_um} um (depth/z), cladding n={n_clad_side:.3f} '
              f'-> guiding in y AND z')
    else:
        print(f'Waveguide: SLAB (guided in cross-section/y only, unbounded in depth/z)')
    if cad is None:
        _stack_banner(stack, bead)

    # Ressourcen-Check IMMER (nicht nur bei --check-resources): verhindert den
    # rohen numpy/CuPy-MemoryError beim eps-Aufbau. Bei Nichtpassen harter Abbruch
    # mit konkreten Vorschlaegen, ausser --allow-large erzwingt es.
    ok, report, hints = check_resources_3d(
        Nx, Ny, Nz, n_snapshots, vol_dtype, save_vector, dx_nm, boundary=boundary,
        bead=bool(bead and BEAD_SUBPIXEL), lossy=(None if stack is None else _stack_lossy(stack)))
    if check_resources or not ok:
        print(report)
    if not ok and not allow_large:
        print('  [ABORT] Configuration does NOT safely fit into the available resources '
              '(would cause a MemoryError while building eps).')
        for hnt in hints:
            print('   -> ' + hnt)
        print('   -> or force it anyway with --allow-large (risk of OOM/crash).')
        raise SystemExit(2)
    if not ok and allow_large:
        print('  [WARNING] --allow-large set: forcing run despite resource warning.')
    elif check_resources:
        print('  [OK] Configuration fits into the available resources.')

    pec_faces = tuple(pec_faces or ())
    absorbing_faces = tuple(f for f in ALL_FACES if f not in pec_faces)
    _bnd3d = make_boundary_3d(boundary, dx, dt, absorbing_faces, shape=(Nx, Ny, Nz), lam_m=LAM)
    _cpml3d = _bnd3d is not None and hasattr(_bnd3d, 'step')   # cpml: eigener Yee-Schritt
    print(f'Absorber: {"Sponge (graded)" if _bnd3d is None else boundary} '
          f'on {list(absorbing_faces)}')
    if pec_faces:
        print(f'Boundaries: PEC (mirror) on {list(pec_faces)}, '
              f'absorber on {list(absorbing_faces)}')
    if end_facet_um and end_facet_um > 0:
        print(f'End facet: material->air {end_facet_um} um before the exit '
              f'(Fresnel reflection)')
    # Einkoppelabstand: Quelle sitzt bei x=(n_sp+6)*dx; der WG beginnt input_gap
    # dahinter -> Laser laeuft input_gap um durch Luft bis zur Eintrittsfacette.
    # VCSEL in Luft vor der Facette: input_gap_um = Abstand Taille -> Facette
    _vd = float(input_gap_um or 0.0)
    x_src_um, x_wg_start_um, _air, _zsrc = _vcsel_geom_3d(n_sp, dx_um, _vd)
    print(f'VCSEL: source plane x={x_src_um:.2f}um, WG entry (facet) at '
          f'x={x_wg_start_um:.2f}um -> {_air:.2f}um air path (Fresnel entry)')
    cad_region = None
    _tp = {}                                  # Bead-Rand fuer Tensor-Mittelung
    if cad is not None:
        print(f'CAD {_cm["cad_file"]}: core "{cad.core}" n={wg_n:.3f}')
        print(cad.scene.summary(lam_nm))
        eps, cad_region = cad.eps3d(0.0, Nx, Ny, Nz)
    else:
        eps = build_eps3d(Nx, Ny, Nz, dx_um, y0_um, z0_um, wg_n, t_wg_um,
                          t_lip_um, t_aq_um, t_mu_um, n_lip, n_aq, n_mu, n_co, bead,
                          wg_width_um=wg_width_um, n_clad_side=n_clad_side,
                          end_facet_um=end_facet_um, x_wg_start_um=x_wg_start_um,
                          tensor_patch=_tp, stack=stack)
    eps, _eim = _split_eps3(eps)                     # Absorption: eps' + eps''
    # eps ist float32 -> nur float32 uebertragen (halber Host-Puffer statt float64),
    # Division dann auf dem Device. Vermeidet grossen Pinned-Transfer.
    if _bnd3d is not None and hasattr(_bnd3d, 'set_speeds'):
        # Mur mit lokaler Geschwindigkeit (x: n_eff, y/z: n der Randzellen) statt c0:
        # mit c0 reflektiert Mur die gefuehrte Welle im Dielektrikum mit |r|~0.2.
        _msp3 = mur_speeds_3d(eps, dx, LAM)
        _bnd3d.set_speeds(_msp3)
        if _msp3:
            print(f'  Mur local: x faces n_eff={_msp3["xmin"]:.4f}/{_msp3["xmax"]:.4f}')
    inv, _da = _inv_loss(eps, _eim, dt, dx, LAM)
    del eps, _eim
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
    ce_x, ce_y, ce_z = _ce_arrays(inv, _tp, dt, dx, dx_um, da=_da)   # versetzt gemittelt; Bead: Tensor-Rand
    _da = _stagger_da(_da)                 # Daempfung an den versetzten Positionen
    inv = None                            # ce_* sind eigene Arrays -> inv freigeben
    Ch = xp.float32(dt/(MU0*dx))

    Ex = xp.zeros((Nx-1, Ny,   Nz),   dtype=xp.float32)
    Ey = xp.zeros((Nx,   Ny-1, Nz),   dtype=xp.float32)
    Ez = xp.zeros((Nx,   Ny,   Nz-1), dtype=xp.float32)
    Hx = xp.zeros((Nx,   Ny-1, Nz-1), dtype=xp.float32)
    Hy = xp.zeros((Nx-1, Ny,   Nz-1), dtype=xp.float32)
    Hz = xp.zeros((Nx-1, Ny-1, Nz),   dtype=xp.float32)
    # Polarisation: 's'/TE -> Quelle+Hauptfeld = Ez (Gitter Ny x Nz-1);
    #               'p'/TM -> Quelle+Hauptfeld = Ey (Gitter Ny-1 x Nz).
    _pol, _Nj, _Nk = _resolve_pol(polarization, Ny, Nz)
    src_field = Ey if _pol == 'p' else Ez
    Iavg = xp.zeros_like(src_field)

    # Quelle: Gauss-Spot (Taille in y und z) an x = src_ix auf der getriebenen
    # Komponente. Zentrum standardmaessig WG-Mitte (y=t_wg/2) und z=0; bei CAD
    # automatisch so, dass der Strahl die Eintrittsflaeche mittig und senkrecht trifft.
    _cy = _cz = None
    if cad is not None:
        _s = cad.source((n_sp + 6)*dx_um)
        _cy, _cz = _s['y_um'], _s['z_um']
        _air = _cad_air_um_3d(_s, x_src_um); _zsrc = vcsel.z_at_source(_vd, _air)
        x_wg_start_um = float(_s['facet_center'][0])   # Facette (nur Ausgabe/Detektor-Schutz)
        vcsel_tilt_deg = vcsel_tilt_deg + _s['tilt_deg']
        print(f'Source (auto): facet {np.round(_s["facet_center"], 2)} um, normal '
              f'{np.round(_s["facet_normal"], 3)} -> spot y={_cy:.2f} z={_cz:.2f} um, '
              f'tilt {vcsel_tilt_deg:.2f} deg')
        if abs(_s['tilt_z_deg']) > 1.0:
            print(f'  [Note] Facet is also tilted in z ({_s["tilt_z_deg"]:.1f} deg): the '
                  f'source tilts only in the x-y plane.')
    src_ix, j_src, k_src, _vs = _build_gauss_source(
        n_sp, t_wg_um, y0_um, z0_um, dx_um, dx, LAM, _Nj, _Nk,
        vcsel_waist_um, vcsel_waist_z_um,
        vcsel_offset_y_um, vcsel_offset_z_um, vcsel_tilt_deg,
        center_y_um=_cy, center_z_um=_cz,
        n_src_fn=_n_source_fn_3d(cad, x_src_um, Ny, Nz, y0_um, dx_um, wg_n, t_wg_um,
                                 n_aq if stack is None else _stack_n_below(stack, n_co),
                                 x_wg_start_um, _pol),
        pol=_pol, wg_width_um=wg_width_um, dt=dt, z_prop_um=_zsrc)
    print('  ' + vcsel.describe(vcsel_waist_um, lam_nm, _vd, _air, LASER_POWER_MW,
                                waist_z_um=vcsel_waist_z_um))
    print(f'Polarization: {_pol}-pol -> source drives '
          f'{"Ey (TM, E in x-y plane)" if _pol=="p" else "Ez (TE, E along z)"}')

    g_sp = _sponge_profile(n_sp, sponge_alpha)

    # Detektor-Boxen: WG-Querschnitt (y=0..t_wg, z innerhalb Sponge), +-1um in x
    if cad is not None:                   # am Kern: 10 % / 90 % seiner Laenge
        (iin0, iin1, j_wg0, j_wg1), (iout0, iout1, _jo0, _jo1) = cad.detectors(
            cad_region[:, :, cad.core_k(Nz)], 0.0)
        det_in_um = 0.5*(iin0 + iin1)*dx_um; det_out_um = 0.5*(iout0 + iout1)*dx_um
    else:
        if det_in_um is None:
            det_in_um = 0.2*lx_um
        if det_out_um is None:
            det_out_um = 0.8*lx_um
        j_wg0 = int(round((0.0 - y0_um)/dx_um))
        j_wg1 = int(round((t_wg_um - y0_um)/dx_um))
        _jo0, _jo1 = j_wg0, j_wg1                  # Ausgangszeilen = Eingangszeilen (gerader Kern)
        hw = max(2, int(round(1.0/dx_um)))    # +-1 um

        def xrange_at(xu):
            c = int(round(xu/dx_um))
            return max(0, c - hw), min(Nx, c + hw)

        iin0, iin1 = xrange_at(det_in_um)
        iout0, iout1 = xrange_at(det_out_um)
    # Eingangsbox stromab der Quelle halten (sonst misst sie die Rueckwaerts-Haelfte)
    _ci = (iin0 + iin1)//2
    _sh = guard_in_plane(_ci, max(src_ix, int(round(x_wg_start_um/dx_um))), Nx) - _ci   # im WG
    iin0, iin1 = iin0 + _sh, min(Nx, iin1 + _sh)
    det_in_um = 0.5*(iin0 + iin1)*dx_um
    _flux = PlaneFluxDFT(2*np.pi*f0, dt)          # exakter CW-Poynting-Fluss (E und H)
    _fl_planes = {'in': max(1, (iin0 + iin1)//2), 'out': max(1, (iout0 + iout1)//2),
                  'src_up': _vs.plane_up, 'src_dn': _vs.plane_dn}   # VCSEL-Leistung

    # Laufzeit: Anlauf + steps_factor Transits beim GRUPPENindex (Marge N_G_FACTOR, die
    # reflektierte Welle laeuft mit v_g < c/n). Danach CW-Konvergenztest (s.u.).
    steps_total = int((24.0/(2*np.pi*f0) + steps_factor*lx_um*1e-6*N_G_FACTOR*wg_n/C0)/dt)
    steps_max = int(CW_MAX_EXTEND*steps_total)
    snap_every = max(1, steps_total//max(1, n_snapshots))
    n_avg0 = steps_total//2               # Iavg ueber die 2. Haelfte (steady)
    # CW-Steady per laufendem DFT bei f0 (letzte ~8 Perioden) -> eingeschwungener
    # Frame-Satz zusaetzlich zum transienten. Akkumulator auf getriebener Komponente.
    _omega = 2*np.pi*f0
    _steps_per_T = max(4, int(round((1.0/f0)/dt)))
    n_acc_cw = min(steps_total, 8*_steps_per_T)
    acc_cw = xp.zeros_like(src_field, dtype=xp.complex64); nacc_cw = 0
    # Konvergenztest nur auf der z-Mittelebene (spart ein zweites Volumen im VRAM)
    _kc = src_field.shape[2]//2
    acc_pl_a = xp.zeros(src_field.shape[:2], dtype=xp.complex64); n_pl_a = 0   # Fenster davor
    acc_pl_b = xp.zeros(src_field.shape[:2], dtype=xp.complex64); n_pl_b = 0   # letztes Fenster
    _conv = (source_type != 'pulse') and not calibrate_steps; cw_rel_change = float('nan')
    _m_I = max(1, _steps_per_T//8); n_I = 0       # Iavg-Abtastung (>=8 Phasen/Periode)
    print(f'Steps: {steps_total}, snapshots every {snap_every}, '
          f'source x={src_ix*dx_um:.1f}um, det in/out x={det_in_um:.0f}/{det_out_um:.0f}um')
    print(f'Source: Gaussian spot waist(y={vcsel_waist_um:g}um, z={vcsel_waist_z_um:g}um) '
          f'@ y={(j_src*dx_um + y0_um):.2f}um, z={(k_src*dx_um + z0_um):.2f}um, '
          f'tilt {vcsel_tilt_deg:g}deg')
    if bead is not None:
        _byc = bead.get('y_um', -bead['d_um']/2.0)
        _bzc = bead.get('z_um', 0.0)
        print(f"Bead: SPHERE d={bead['d_um']:g}um n={bead['n']:.3f} "
              f"at x={bead['x_um']:g}um, y={_byc:g}um, z={_bzc:g}um")

    P_in = 0.0
    P_out = 0.0
    vol_dt = np.float16 if vol_dtype == 'float16' else np.float32
    _vdt_xp = xp.float16 if vol_dtype == 'float16' else xp.float32  # Device-Cast
    # Snapshots direkt auf DISK-MEMMAP schreiben (statt Host-RAM-Puffer) -> vermeidet
    # Host-OOM bei grossen Volumina. Beim Speichern wird chunk-weise ins NPZ gestreamt;
    # die Temp-Memmaps werden danach von save_results_3d geloescht.
    os.makedirs('results', exist_ok=True)
    _safe = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in str(label))
    _stream_paths = []

    def _mmap(shape, tag):
        p = os.path.join('results', f'_stream3d_{_safe}_{tag}.npy')
        _stream_paths.append(p)
        return np.lib.format.open_memmap(p, mode='w+', dtype=vol_dt, shape=shape)

    snap_at = [] if calibrate_steps else \
        [nn for nn in range(steps_total) if (nn+1) % snap_every == 0]
    n_snaps = len(snap_at)
    if n_snaps and save_vector:
        # Vektor-Modus: Ex,Ey,Ez auf gemeinsame Zellzentren (Nx-1,Ny-1,Nz-1)
        vshape = (n_snaps, Nx-1, Ny-1, Nz-1)
        vbuf_x = _mmap(vshape, 'ex')
        vbuf_y = _mmap(vshape, 'ey')
        vol_buf = _mmap(vshape, 'ez')             # = Ez (zellzentriert)
        print(f'  [Vector mode] saving Ex,Ey,Ez (cell-centered, disk stream) '
              f'-> ~3x volume on disk')
    else:
        vbuf_x = vbuf_y = None
        # Hauptfeld = getriebene Komponente (Ez fuer s, Ey fuer p) -> passende Shape
        vol_buf = (_mmap((n_snaps, Nx, _Nj, _Nk), 'ez') if n_snaps else None)
    snap_steps = []
    snap_times = []
    ksnap = 0
    t_start = time.time()

    if calibrate_steps:
        warmup = 5
        n_run = min(steps_total, warmup + calibrate_steps)
        print(f'[CALIBRATION] warmup {warmup} + measurement {calibrate_steps} steps '
              f'(of {steps_total} real steps) ...')
    else:
        n_run = steps_total
    t_cal0 = None

    n = -1
    while n + 1 < n_run:
        n += 1
        if calibrate_steps and n == warmup:
            if GPU_AVAILABLE:
                cp.cuda.runtime.deviceSynchronize()
            t_cal0 = time.time()
        # --- Mur: E^n-Randebenen VOR dem Update sichern ---
        if _bnd3d is not None and not _cpml3d:
            _bnd3d.capture(Ex, Ey, Ez)
        # --- Yee-Schritt (CPML: eigener Schritt mit psi-Korrektur) ---
        if _cpml3d:
            _bnd3d.step(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, Ch, da=_da)
        else:
            _yee_step(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, Ch, da=_da)
        # --- Quelle (soft) ---
        t_phys = n*dt
        if source_type == 'pulse':
            sp = pulse_cycles/f0
            envelope = float(np.exp(-((t_phys - 3*sp)/sp)**2))
        else:
            envelope = float(1 - np.exp(-((t_phys/(2*sigma_t))**2)))
        _vs.inject(src_field, t_phys, envelope)     # VCSEL (einseitig, Huygens)
        # --- Absorber: CPML (im step) / Mur (Kanten) / gradierter Sponge ---
        if _cpml3d:
            pass                                 # Absorption + PEC-Abschluss im step
        elif _bnd3d is not None:
            _bnd3d.apply(Ex, Ey, Ez)
        else:
            for F in (Ex, Ey, Ez, Hx, Hy, Hz):
                _apply_sponge(F, g_sp, n_sp, absorbing_faces)
        # --- PEC-Spiegel: tangentiales E = 0 auf gewaehlten Flaechen ---
        if pec_faces:
            _apply_pec(Ex, Ey, Ez, pec_faces)
        # --- Detektoren (zeitintegriert), volle Intensitaet |E|^2 aller Komponenten ---
        _zs = slice(n_sp, -n_sp)
        P_in += _box_E2(Ex, Ey, Ez, slice(iin0, iin1), slice(j_wg0, j_wg1), _zs)
        P_out += _box_E2(Ex, Ey, Ez, slice(iout0, iout1), slice(_jo0, _jo1), _zs)   # eigene Ausgangszeilen (CAD)
        # --- Zeitmittel-Intensitaet <|E|^2> (2. Haelfte), alle drei Komponenten.
        # Abgetastet alle _m_I Steps (>=8 Phasen/Periode -> exakt fuer CW), das
        # spart die blockweise Kolokation in den meisten Steps.
        if n >= n_avg0 and (n - n_avg0) % _m_I == 0:
            _accum_E2(Iavg, Ex, Ey, Ez)
            n_I += 1
        # --- eingeschwungenes Ê akkumulieren (letzte ~8 Perioden) ---
        if n >= steps_total - n_acc_cw:
            _ph = np.exp(-1j*_omega*t_phys)
            acc_cw += src_field.astype(xp.complex64)*_ph
            nacc_cw += 1
            for _k, _il in _fl_planes.items():          # exakter Fluss (E und H)
                _flux.add(_k, t_phys, pairs_3d(_il, Ey, Ez, Hy, Hz))
            if _conv:
                acc_pl_b += src_field[:, :, _kc].astype(xp.complex64)*_ph; n_pl_b += 1
        elif _conv and n >= steps_total - 2*n_acc_cw:
            acc_pl_a += src_field[:, :, _kc].astype(xp.complex64)*np.exp(-1j*_omega*t_phys)
            n_pl_a += 1
        if _conv and n + 1 == steps_total and n_pl_a > 0 and n_pl_b > 0:
            _a = acc_pl_b/n_pl_b; _b = acc_pl_a/n_pl_a
            cw_rel_change = float(xp.linalg.norm(_a - _b)/max(float(xp.linalg.norm(_a)), 1e-30))
            if cw_rel_change > CW_TOL and steps_total + n_acc_cw <= steps_max:
                print(f'  [CW] not yet in steady state (change {100*cw_rel_change:.2f}% > '
                      f'{100*CW_TOL:.1f}%) -> +{n_acc_cw} Steps')
                acc_pl_a = acc_pl_b; n_pl_a = n_pl_b
                acc_pl_b = xp.zeros_like(acc_pl_a); n_pl_b = 0
                acc_cw[...] = 0; nacc_cw = 0
                _flux.reset()
                steps_total += n_acc_cw; n_run += n_acc_cw
            else:
                print(f'  [CW] phasor change over last 2 DFT windows (z center): '
                      f'{100*cw_rel_change:.3f}%'
                      + ('' if cw_rel_change <= CW_TOL else '  (!! max. runtime reached)'))
        # --- Snapshot: direkt in vorallokierten Puffer (kein np.stack noetig) ---
        # Cast schon auf dem Device in vol_dt -> halber Host-Transfer (float16).
        # Ein fehlgeschlagener Snapshot (Speicher) beendet nur die Snapshot-Sammlung,
        # der Lauf laeuft weiter und speichert die bereits gesammelten Frames.
        if vol_buf is not None and (n+1) % snap_every == 0 and ksnap < n_snaps:
            try:
                if save_vector:
                    exc, eyc, ezc = _colocate_E(Ex, Ey, Ez)   # auf Zellzentren
                    vbuf_x[ksnap] = to_np(exc.astype(_vdt_xp))
                    vbuf_y[ksnap] = to_np(eyc.astype(_vdt_xp))
                    vol_buf[ksnap] = to_np(ezc.astype(_vdt_xp))
                else:
                    vol_buf[ksnap] = to_np(src_field.astype(_vdt_xp))
                snap_steps.append(n+1)
                snap_times.append(time.time() - t_start)
                ksnap += 1
                print(f'  Step {n+1}/{steps_total} ({100*(n+1)/steps_total:.0f}%)  '
                      f'max|E_{_pol}|={float(xp.max(xp.abs(src_field))):.3e}')
            except Exception as _snap_e:
                print(f'  [WARN] Snapshot {ksnap+1} skipped (memory: {_snap_e}). '
                      f'Run continues, {ksnap} frames already saved.')
                n_snaps = ksnap        # keine weiteren Snapshots mehr versuchen

    if calibrate_steps:
        if GPU_AVAILABLE:
            cp.cuda.runtime.deviceSynchronize()
        n_measured = n_run - warmup
        dt_cal = time.time() - t_cal0
        per_step = dt_cal/max(1, n_measured)
        rate = cells*n_measured/max(dt_cal, 1e-9)        # Zell-Updates/s
        est_full = per_step*steps_total
        print(f'\n===== CALIBRATION {label} =====')
        print(f'  Measured: {n_measured} steps in {dt_cal:.2f}s '
              f'-> {per_step*1000:.1f} ms/step')
        print(f'  Rate: {rate/1e9:.2f} G cell updates/s '
              f'({rate/1e6:.0f} Mcell/s)')
        m, s = divmod(int(est_full), 60)
        print(f'  EXTRAPOLATION full run ({steps_total} steps): '
              f'~{est_full:.0f}s = {m}min {s}s')
        print(f'  (Do NOT set calibrate to start the full run.)')
        return dict(scenario=label, dim='3D', calibrated=True,
                    per_step_s=per_step, cell_update_rate=rate,
                    steps_total=steps_total, est_full_s=est_full,
                    cells=cells, dx_nm=dx_nm)

    Iavg_np = to_np(Iavg)/max(1, n_I)
    P_in = float(P_in)*dx**3                 # einmaliger Readout (Device-Skalar)
    P_out = float(P_out)*dx**3
    Tr_I = P_out/max(P_in, 1e-30)                 # alt: zeitintegriertes |E|^2 in Kern-Boxen
    Tr = Tr_I; tr_def = 'intensity'
    _p_si = None
    if nacc_cw > 0 and source_type != 'pulse' and _flux.has('in') and _flux.has('out'):
        # Einheitliche Transmission: CW-Poynting-Fluss ueber den ganzen Querschnitt (y,z),
        # exakt aus allen tangentialen E- und H-Komponenten (PlaneFluxDFT)
        _S = [_flux.flux('in'), _flux.flux('out')]
        _p_si = _S[0]*dx*dx      # eingekoppelte Leistung [W]
        if abs(_S[0]) > 0:
            Tr = _S[1]/_S[0]; tr_def = 'poynting'
    elif nacc_cw > 0 and source_type != 'pulse':
        # Fallback: Schaetzung aus der Hauptkomponente
        _E = (2.0/nacc_cw)*acc_cw
        _S = []
        for _ix in ((iin0 + iin1)//2, (iout0 + iout1)//2):
            _pl = to_np(_E[_ix - 1:_ix + 2])
            _e = None
            if _pol == 'p':
                _e = _eps_planes_3d(cad, _ix, dx_um, Ny, Nz, y0_um, z0_um, wg_n, t_wg_um,
                                    t_lip_um, t_aq_um, t_mu_um, n_lip, n_aq, n_mu, n_co, bead,
                                    wg_width_um, n_clad_side, end_facet_um, x_wg_start_um,
                                    stack=stack)
                _e = 0.5*(_e[:-1] + _e[1:])           # auf das Ey-Gitter (Ny-1)
            _S.append(_poynting_flux_3d(_pl, _e))
        if abs(_S[0]) > 0:
            Tr = _S[1]/_S[0]; tr_def = 'poynting'
        del _E
    total_time = time.time() - t_total
    print(f'\n[Result] {label} [3D]: total {total_time:.0f}s  '
          f'P_in={P_in:.3e}  P_out={P_out:.3e}  Transmission={Tr:.4f} ({tr_def})'
          f'  [intensity boxes: {Tr_I:.4f}]')

    # Memmaps auf Disk sichern, damit save_results_3d konsistent liest.
    for _b in (vol_buf, vbuf_x, vbuf_y):
        if _b is not None:
            try:
                _b.flush()
            except Exception:
                pass
    # Nur die tatsaechlich gefuellten Snapshots zurueckgeben (ksnap kann < n_snaps
    # sein, falls ein Snapshot wegen Speicher uebersprungen wurde).
    _va = vol_buf[:ksnap] if vol_buf is not None else None
    _vx = vbuf_x[:ksnap] if vbuf_x is not None else None
    _vy = vbuf_y[:ksnap] if vbuf_y is not None else None
    # eingeschwungenes (CW) Volumen aus Ê: nfr Phasen ueber 1 Periode (getriebene
    # Hauptkomponente), auf DISK-Memmap gestreamt.
    _cw_vol = None
    if nacc_cw > 0:
        Ecw = (2.0/nacc_cw)*acc_cw
        nfr_cw = max(1, ksnap if ksnap else n_snapshots)
        _cw_path = os.path.join('results', f'_stream3d_{_safe}_cwvol.npy')
        _stream_paths.append(_cw_path)
        _cw_vol = np.lib.format.open_memmap(_cw_path, mode='w+', dtype=vol_dt,
                                            shape=(nfr_cw,) + tuple(src_field.shape))
        for k in range(nfr_cw):
            tk = (k/nfr_cw)*(1.0/f0)
            _cw_vol[k] = to_np(xp.real(Ecw*np.exp(1j*_omega*tk)).astype(_vdt_xp))
        _cw_vol.flush()
    _rs = _result_stack(stack, t_lip_um, t_aq_um, t_mu_um, n_lip, n_aq, n_mu)
    _cw_ok = source_type != 'pulse' and _flux.has('src_up') and _flux.has('src_dn')
    _vres = _vcsel_res(((_flux.flux('src_dn') - _flux.flux('src_up'))*dx*dx) if _cw_ok else None,
                       (-_flux.flux('src_up')*dx*dx) if _cw_ok else None, _p_si, _vd, _air,
                       x_wg_start_um, vcsel_waist_um, vcsel_waist_z_um)
    return dict(**_vres, scenario=label, dim='3D', P_in=P_in, P_out=P_out, p_in_si=_p_si, p_in_unit='W',
                transmission=Tr, vols_array=_va,
                vols_ex=_vx, vols_ey=_vy, vector=bool(save_vector),
                _stream_paths=_stream_paths,
                snap_steps=snap_steps, snap_times=snap_times, Iavg=Iavg_np,
                cw_vol=_cw_vol, view_primary='transient', f0=f0,
                transmission_intensity=Tr_I, transmission_def=tr_def,
                cw_rel_change=cw_rel_change,
                total_time=total_time, steps_total=steps_total, dt=dt,
                dx_nm=dx_nm, lx_um=lx_um, ly_um=ly_um, lz_um=lz_um,
                y0_um=y0_um, z0_um=z0_um, t_wg_um=t_wg_um, wg_n=wg_n,
                layers=_rs[2], stack=_rs[0], stack_custom=_rs[1],
                n_lip=n_lip, n_aq=n_aq, n_mu=n_mu, n_co=n_co,
                lam_nm=lam_nm, source_type=source_type,
                bead=(bead if bead else {}),
                det_in_um=det_in_um, det_out_um=det_out_um,
                x_wg_start_um=x_wg_start_um, end_facet_um=(end_facet_um or 0.0),
                x_src_um=x_src_um, polarization=_pol,
                cad=(dict(cad.meta(), cad_tilt_deg=vcsel_tilt_deg,
                          cad_det=[[iin0, iin1, j_wg0, j_wg1], [iout0, iout1, _jo0, _jo1]])
                     if cad is not None else None),
                cad_region=cad_region)


def run_3d_stitched(label, wg_n, window_w_um=20.0, slide_um=12.0,
                    t_wg_um=5.0, t_lip_um=0.0, t_aq_um=3.5, t_mu_um=0.5,
                    n_lip=1.480, n_aq=1.336, n_mu=1.342, n_co=1.376,
                    lam_nm=850.0, lx_um=40.0, air_um=3.0, tear_um=None, lz_um=8.0,
                    dx_nm=50.0, vcsel_waist_um=3.0, vcsel_waist_z_um=3.0,
                    source_type='cw', n_snapshots=8,
                    bead=None, n_sponge=None, sponge_um=1.2, sponge_alpha=0.3,
                    det_in_um=None, det_out_um=None, vol_dtype='float32',
                    wg_width_um=None, n_clad_side=N_AIR,
                    vcsel_tilt_deg=0.0, vcsel_offset_y_um=0.0, vcsel_offset_z_um=0.0,
                    input_gap_um=0.0, polarization='s', boundary='cpml', cad=None,
                    return_efull=False, stack=None):
    """Voller gefuellter CW-Waveguide in 3D per GEBIETS-ZERLEGUNG (Stitch).
    cad: optional CadRun (dim=3) -> freie CAD-Geometrie (Box, eps je Fenster, Quelle,
    Detektoren aus der Szene).
    polarization: 's'/TE treibt+assembliert Ez, 'p'/TM treibt+assembliert Ey.
    Analog zum in 2D validierten run_beads_stitched: Fenster entlang x, jedes bis
    zum Steady-State; im Ueberlappbereich wird das zeitharmonische Ez (komplexe
    Amplitude, DFT bei f0) des Vorgaengers WEICH (cosinus-getapert) eingepraegt und
    treibt das naechste Fenster. Assembliert ein komplexes Ez-Amplitudenvolumen und
    gibt reale CW-Phasen-Frames aus. Immer nur EIN Fenster im Speicher.
    v1 - gegen einen 3D-Full-Domain-Lauf gleicher (kurzer) Laenge validieren."""
    t_total = time.time()
    if GPU_AVAILABLE:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    dx = dx_nm*1e-9; dx_um = dx_nm/1000.0
    dt = 0.5*dx/(C0*np.sqrt(3))
    global LAM_LOSS
    LAM = lam_nm*1e-9; LAM_LOSS = LAM; f0 = C0/LAM; omega = 2*np.pi*f0
    sigma_t = 4/(2*np.pi*f0); k0 = 2*np.pi/LAM
    if cad is not None:                        # freie CAD-Geometrie: Box aus der Szene
        _cm = cad.meta()
        lx_um, ly_um, lz_um = cad.lx_um, cad.ly_um, cad.lz_um
        y0_um, z0_um = cad.y0_um, cad.z0_um
        wg_n = float(_cm['cad_n'][cad.core_index - 1])
        t_wg_um = 0.0
    else:
        tear_um = _resolve_tear_um(tear_um, t_lip_um, t_aq_um, t_mu_um,
                                   (10 if str(boundary or 'cpml').lower() == 'cpml' else
                                    0 if str(boundary or '').lower() == 'mur1' else
                                    _sponge_cells(n_sponge, sponge_um, dx_um)), dx_um,
                                   stack=stack)
        ly_um = t_wg_um + air_um + tear_um
        y0_um = -tear_um; z0_um = -lz_um/2.0
    Ny = int(round(ly_um/dx_um)); Nz = int(round(lz_um/dx_um))
    Nx_full = int(round(lx_um/dx_um))
    Nx_win = int(round(window_w_um/dx_um))
    if Nx_win > Nx_full:                       # Fenster nie groesser als Domaene
        Nx_win = Nx_full
        print(f'  [Info] window width > domain -> limited to {lx_um}um (=1 window).')
    S_cells = int(round(slide_um/dx_um))
    S_cells = max(1, min(S_cells, Nx_win))
    O_cells = Nx_win - S_cells
    n_sp = _sponge_cells(n_sponge, sponge_um, dx_um, (Nx_win, Ny, Nz),
                         warn=str(boundary or 'cpml').lower() == 'sponge')
    if O_cells < max(8, n_sp):
        print(f'  [WARN] overlap {O_cells} < sponge {n_sp} -> choose window_w >> slide.')
    # Quelle/WG-Geometrie in globalen Koordinaten
    # VCSEL in Luft vor der Facette: input_gap_um = Abstand Taille -> Facette
    _vd = float(input_gap_um or 0.0)
    x_src_um, x_wg_start_um, _air, _zsrc = _vcsel_geom_3d(n_sp, dx_um, _vd)
    # Trennung: die gewaehlte Randbedingung gilt fuer die AEUSSEREN Flaechen (y, z,
    # Eintritt, Austritt). Die inneren Fenstergrenzen (xmax innerer Fenster) bekommen
    # IMMER eine CPML in einer Verlaengerung (NICHT assembliert, NICHT uebergeben):
    # Mur (auf c0 abgestimmt) reflektiert dort die gefuehrte Welle mit |r|~0.2.
    _EXT = max(n_sp, 10) + 4
    x_wg_end_um = lx_um + _EXT*dx_um            # WG laeuft wie im full-Lauf in den Absorber
    n_win = (int(np.ceil((Nx_full - Nx_win)/S_cells)) + 1) if Nx_full > Nx_win else 1

    def _gx0(w):                                # letztes Fenster buendig am Domaenenende
        return min(w*S_cells, max(Nx_full - Nx_win, 0))
    transit = window_w_um*1e-6*N_G_FACTOR*wg_n/C0     # Gruppenindex-Marge (F8)
    steps_win = int(3.0*transit/dt) + 400
    steps_per_T = max(4, int(round((1.0/f0)/dt)))
    n_acc = steps_per_T*8
    cpl = (LAM/wg_n)/dx
    print(f'\n===== 3D-FDTD [stitch/domain decomposition]: {label} =====')
    print(f'Axes: x=length (propagation), y=layer stack, z=depth')
    print(f'Domain {lx_um}x{ly_um}x{lz_um}um -> {n_win} windows of {window_w_um}um '
          f'(overlap {O_cells*dx_um:.1f}um), grid/window {Nx_win}x{Ny}x{Nz}')
    print(f'dx={dx_nm}nm ({cpl:.1f} cells/lambda in WG), steps/window {steps_win}, Sponge {n_sp}')
    if cad is None:
        _stack_banner(stack, bead)
    if cpl < 8:
        print('!!! WARNING: <8 cells/wavelength in WG - distorted by numerical dispersion.')
    # Groesse des ASSEMBLIERTEN Ergebnisses abschaetzen. Der Stitch spart Speicher
    # bei der RECHNUNG (ein Fenster zur Zeit), aber das volle 3D-Feld ueber die
    # ganze Laenge ist intrinsisch gross -> auf Disk gestreamt (nicht RAM).
    _vcell = Nx_full*Ny*(Nz-1)
    _vb = 2 if vol_dtype == 'float16' else 4
    gb_ef = _vcell*8/1e9                       # Efull complex64 (Disk)
    gb_fr = max(1, n_snapshots)*_vcell*_vb/1e9  # Frames (Disk)
    gb_ia = _vcell*4/1e9                        # Iavg float32 (Disk)
    gb_tot = gb_ef + gb_fr + gb_ia
    print(f'[Memory] assembled volume is streamed to DISK: '
          f'Efull {gb_ef:.1f} + Frames {gb_fr:.1f} + Iavg {gb_ia:.1f} = ~{gb_tot:.1f} GB Disk')
    # Per-Fenster GPU-VRAM abschaetzen: 6 Felder + ce_x/y/z + acc_re + acc_im + acc_I
    # (+ blockweise |E|^2-Temps) ~ 12.5 float32-Volumina der Fenstergroesse (+ Pool-/Temp-Overhead). Passt das
    # nicht in den freien VRAM -> sauber abbrechen statt mitten im Lauf OOM.
    _wcell = (Nx_win + _EXT)*Ny*Nz              # inkl. Absorber-Verlaengerung
    _nvol = 12.5                    # 6 Felder + ce_x/y/z + acc_re/im + acc_I + |E|^2-Temps
    if _stack_lossy(stack):
        _nvol += 5                  # Absorption: D_x/D_y/D_z (+2 kurzzeitig beim Aufbau)
    _bnd_kind = (boundary or 'cpml').lower()
    # CPML haelt psi NUR in den PML-Randstreifen (_PsiStrip): 4 psi je Achse,
    # jeweils 2*npml Zellen dick -> 32*npml*(NyNz+NxNz+NxNy) Byte.
    _NPML3D = 10
    gb_cpml = (32*_NPML3D*(Ny*Nz + Nx_win*Nz + Nx_win*Ny)/1e9
               if _bnd_kind == 'cpml'
               else 32*_NPML3D*Ny*Nz/1e9)        # sonst nur Naht-CPML (xmax-Streifen)
    gb_win_gpu = _wcell*4*_nvol/1e9 + gb_cpml
    print(f'[GPU] requirement per window ~{gb_win_gpu:.1f} GB VRAM '
          f'(window grid {Nx_win}x{Ny}x{Nz}'
          + (f', of which {gb_cpml:.2f} GB CPML psi (strips)' if gb_cpml else '')
          + ')')
    if GPU_AVAILABLE:
        try:
            _free, _tot = cp.cuda.runtime.memGetInfo()
            print(f'[GPU] free {_free/1e9:.1f} / {_tot/1e9:.1f} GB')
            if gb_win_gpu > 0.85*(_free/1e9):    # >85% des freien VRAM
                raise MemoryError
        except MemoryError:
            # maximal moegliche Fensterbreite fuer diese Konfiguration ausrechnen,
            # damit die Empfehlung konkret ist (statt "waehle kleiner").
            # Bedarf pro x-Zelle (Felder + der x-abhaengige CPML-Anteil)
            _per_x = (Ny*Nz*4*_nvol + (32*_NPML3D*(Nz + Ny) if gb_cpml else 0))/1e9
            _fix = (32*_NPML3D*Ny*Nz/1e9) if gb_cpml else 0.0   # x-Streifen (fest)
            _nx_ok = int(max(0.0, 0.85*(_free/1e9) - _fix)/max(_per_x, 1e-12))
            _w_ok = _nx_ok*dx_um
            print(f'!!! ABORT: one window (~{gb_win_gpu:.1f} GB) does not fit into the '
                  f'free VRAM ({_free/1e9:.1f} GB).')
            print(f'    -> choose window width <= ~{_w_ok:.1f} um '
                  f'(--window-um {max(1, int(_w_ok*0.9))} --slide {max(1, int(_w_ok*0.55))})')
            if gb_cpml:
                print(f'    -> or --boundary sponge (saves the '
                      f'{gb_cpml:.2f} GB of CPML psi strips)')
            print(f'    -> or coarser resolution / smaller cross-section (air/tear/lz)')
            raise SystemExit(2)
        except Exception:
            pass
    # DISK: passt das assemblierte Volumen + Frames + Iavg auf die Platte? Sonst
    # crasht der Lauf spaeter beim Schreiben -> hier hart abbrechen.
    try:
        import shutil
        _free_disk = shutil.disk_usage(os.getcwd()).free/1e9
        print(f'[Disk] requirement ~{gb_tot:.1f} GB   free {_free_disk:.1f} GB')
        if gb_tot > 0.95*_free_disk:
            print(f'!!! ABORT: ~{gb_tot:.1f} GB of output does not fit on the disk '
                  f'({_free_disk:.1f} GB free). Omit --save-vector (frames /3), '
                  f'reduce --snapshots, or use a shorter --length-um.')
            raise SystemExit(2)
    except SystemExit:
        raise
    except Exception:
        pass
    # Host-RAM (nur noch Fenster-Transfer, kein Akkumulator mehr): Warnung, kein Abbruch.
    _hfree = _host_free_gb()
    _host_need = _wcell*16/1e9                  # Ew (complex64) + 2 float32-Transferpuffer
    if _hfree is not None and _host_need > 0.85*_hfree:
        print(f'!!! WARNING: host RAM is tight (~{_host_need:.1f} GB window transfer vs '
              f'{_hfree:.1f} GB free). On OOM: reduce the cross-section (tear/lz) or the window.')
    if gb_tot > 40:
        print(f'!!! WARNING: ~{gb_tot:.0f} GB disk requirement. 3D over such long distances '
              f'is expensive. Consider a smaller lx (3D = short distance) or a coarser '
              f'resolution. Long propagation is the domain of the 2D stitch method.')
    Ch = xp.float32(dt/(MU0*dx))
    g_sp = _sponge_profile(n_sp, sponge_alpha)
    # Polarisation: 's'/TE -> getriebene+assemblierte Komponente Ez (Ny x Nz-1),
    #               'p'/TM -> Ey (Ny-1 x Nz).
    _pol, _Nj, _Nk = _resolve_pol(polarization, Ny, Nz)
    print(f'Polarization: {_pol}-pol -> drives/assembles '
          f'{"Ey (TM)" if _pol=="p" else "Ez (TE)"}')
    # Quelle (Fenster 0): Gauss-Spot auf der getriebenen Komponente; bei CAD auf die
    # Eintrittsflaeche des Kerns ausgerichtet.
    _cy = _cz = None
    cad_region = None
    if cad is not None:
        _s = cad.source((n_sp + 6)*dx_um)
        _cy, _cz = _s['y_um'], _s['z_um']
        _air = _cad_air_um_3d(_s, x_src_um); _zsrc = vcsel.z_at_source(_vd, _air)
        x_wg_start_um = float(_s['facet_center'][0])   # Facette (nur Ausgabe/Detektor-Schutz)
        vcsel_tilt_deg = vcsel_tilt_deg + _s['tilt_deg']
        print(f'CAD {_cm["cad_file"]}: core "{cad.core}" n={wg_n:.3f}\n{cad.scene.summary(lam_nm)}')
        print(f'Source (auto): facet {np.round(_s["facet_center"], 2)} um -> spot '
              f'y={_cy:.2f} z={_cz:.2f} um, tilt {vcsel_tilt_deg:.2f} deg')
        cad_region = np.zeros((Nx_full, Ny, Nz), dtype=np.int8)   # fuer Detektoren + Analyzer
        for _x0 in range(0, Nx_full, 256):
            _x1 = min(_x0 + 256, Nx_full)
            cad_region[_x0:_x1] = cad.eps3d(_x0*dx_um, _x1 - _x0, Ny, Nz)[1]
    src_ix, j_src, k_src, _vs = _build_gauss_source(
        n_sp, t_wg_um, y0_um, z0_um, dx_um, dx, LAM, _Nj, _Nk,
        vcsel_waist_um, vcsel_waist_z_um,
        vcsel_offset_y_um, vcsel_offset_z_um, vcsel_tilt_deg,
        center_y_um=_cy, center_z_um=_cz,
        n_src_fn=_n_source_fn_3d(cad, x_src_um, Ny, Nz, y0_um, dx_um, wg_n, t_wg_um,
                                 n_aq if stack is None else _stack_n_below(stack, n_co),
                                 x_wg_start_um, _pol),
        pol=_pol, wg_width_um=wg_width_um, dt=dt, z_prop_um=_zsrc)
    print('  ' + vcsel.describe(vcsel_waist_um, lam_nm, _vd, _air, LASER_POWER_MW,
                                waist_z_um=vcsel_waist_z_um))
    # cosinus-Taper fuer weiche Overlap-Einpraegung (1 -> 0 ueber O_cells)
    _wtap = xp.asarray((0.5*(1.0 + np.cos(np.pi*np.arange(O_cells)/max(O_cells-1, 1)))
                        ).astype(np.float32))[:, None, None]
    # Detektor-Ebenen (global)
    if cad is not None:                        # Detektoren am Kern (10 % / 90 % seiner Laenge)
        (_a0, _a1, j_wg0, j_wg1), (_b0, _b1, _jo0, _jo1) = cad.detectors(
            cad_region[:, :, cad.core_k(Nz)], 0.0)
        det_in_um = 0.5*(_a0 + _a1)*dx_um; det_out_um = 0.5*(_b0 + _b1)*dx_um
    else:
        if det_in_um is None: det_in_um = 0.2*lx_um
        if det_out_um is None: det_out_um = 0.8*lx_um
        j_wg0 = int(round((0.0 - y0_um)/dx_um)); j_wg1 = int(round((t_wg_um - y0_um)/dx_um))
        _jo0, _jo1 = j_wg0, j_wg1
    vol_dt = np.float16 if vol_dtype == 'float16' else np.float32
    os.makedirs('results', exist_ok=True)
    _safe = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in str(label))
    _stream_paths = []
    # assembliertes komplexes Amplitudenvolumen auf DISK-Memmap (Shape = Hauptkomp.)
    _ef_path = os.path.join('results', f'_stream3d_{_safe}_stitch_Efull.npy')
    Efull = np.lib.format.open_memmap(_ef_path, mode='w+', dtype=np.complex64,
                                      shape=(Nx_full, _Nj, _Nk))
    # <|E|^2> aller drei Komponenten, pro Fenster gemittelt und wie Efull assembliert
    _iavg_path = os.path.join('results', f'_stream3d_{_safe}_stitch_Iavg.npy')
    _stream_paths.append(_iavg_path)
    Iavg = np.lib.format.open_memmap(_iavg_path, mode='w+', dtype=np.float32,
                                     shape=(Nx_full, _Nj, _Nk))
    _m_I = max(1, int(round((1.0/f0)/dt))//8)     # >=8 Phasen/Periode (exakt fuer CW)
    Ehand = None                                 # Handoff: komplexe Ex,Ey,Ez des Overlaps
    # Detektor-Ebenen (global) fuer den exakten CW-Poynting-Fluss (E und H), im Fenster
    # mitgeschrieben, das die Ebene assembliert (gleiche Ebenen wie die Auswertung unten).
    _gi = guard_in_plane(min(max(int(round(det_in_um/dx_um)), 1), Nx_full - 2),
                         max(src_ix, int(round(x_wg_start_um/dx_um))), Nx_full)   # im WG
    det_in_um = _gi*dx_um                        # auch fuer die Intensitaets-Kennzahl unten
    _go = min(max(int(round(det_out_um/dx_um)), 1), Nx_full - 2)
    _flux = PlaneFluxDFT(omega, dt)
    for w in range(n_win):
        gx0 = _gx0(w)
        x0_um = gx0*dx_um
        last = (w == n_win - 1)
        Nxw = Nx_win + (0 if last else _EXT)     # Rechengebiet (inkl. Absorber-Verlaengerung)
        _glo = 0 if w == 0 else gx0 + O_cells
        _ghi = Nx_full if last else min(gx0 + Nx_win, Nx_full)
        _fpl = {k: g - gx0 for k, g in (('in', _gi), ('out', _go))
                if _glo <= g < _ghi and g - gx0 >= 1}
        if w == 0:                                # VCSEL-Leistung um die Quelle
            _fpl.update(src_up=_vs.plane_up, src_dn=_vs.plane_dn)
        _off = (_gx0(w + 1) - gx0) if not last else 0   # Overlap-Start fuers naechste Fenster
        _tp = {}                                  # Bead-Rand fuer Tensor-Mittelung
        if cad is not None:
            eps = cad.eps3d(x0_um, Nxw, Ny, Nz)[0]
        else:
            eps = build_eps3d(Nxw, Ny, Nz, dx_um, y0_um, z0_um, wg_n, t_wg_um,
                              t_lip_um, t_aq_um, t_mu_um, n_lip, n_aq, n_mu, n_co, bead,
                              wg_width_um=wg_width_um, n_clad_side=n_clad_side,
                              x0_um=x0_um, x_wg_start_um=x_wg_start_um,
                              x_wg_end_um=x_wg_end_um, tensor_patch=_tp, stack=stack)
        eps, _eim = _split_eps3(eps)
        _msp3 = (mur_speeds_3d(eps, dx, LAM)
                 if str(boundary or 'cpml').lower() == 'mur1' else None)
        inv, _da = _inv_loss(eps, _eim, dt, dx, LAM); del eps, _eim
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
        ce_x, ce_y, ce_z = _ce_arrays(inv, _tp, dt, dx, dx_um, da=_da)   # versetzt; mit Bead: Tensor-Rand
        _da = _stagger_da(_da)                 # Daempfung an den versetzten Positionen
        inv = None
        Ex = xp.zeros((Nxw-1, Ny,   Nz),   dtype=xp.float32)
        Ey = xp.zeros((Nxw,   Ny-1, Nz),   dtype=xp.float32)
        Ez = xp.zeros((Nxw,   Ny,   Nz-1), dtype=xp.float32)
        Hx = xp.zeros((Nxw,   Ny-1, Nz-1), dtype=xp.float32)
        Hy = xp.zeros((Nxw-1, Ny,   Nz-1), dtype=xp.float32)
        Hz = xp.zeros((Nxw-1, Ny-1, Nz),   dtype=xp.float32)
        src_field = Ey if _pol == 'p' else Ez     # getriebene/assemblierte Komponente
        # DFT-Akkumulation Ê = Σ E(t) e^{-iωt} auf der GPU als Real-/Imagteil (float32).
        # Kein Host-Akkumulator und KEIN to_np pro Step -> Transfer nur EINMAL pro Fenster.
        # Nur der assemblierte Bereich [0, Nx_win) (die Verlaengerung ist Absorber).
        acc_re = xp.zeros((Nx_win, _Nj, _Nk), dtype=xp.float32)
        acc_im = xp.zeros((Nx_win, _Nj, _Nk), dtype=xp.float32); acc_n = 0
        acc_I = xp.zeros((Nx_win, _Nj, _Nk), dtype=xp.float32); n_I = 0
        # Handoff-DFT ALLER drei E-Komponenten im Overlap des naechsten Fensters: in 3D
        # beschreibt die Hauptkomponente allein das Feld nicht (Ex/Ey der Mode) ->
        # Einpraegen nur von Ez/Ey kostete ~9% Amplitude pro Naht.
        hand_acc = ([xp.zeros((O_cells,) + F.shape[1:], dtype=xp.complex64)
                     for F in (Ex, Ey, Ez)] if not last else None)
        _drive = Ehand if (w > 0 and Ehand is not None) else None
        # Aeussere Flaechen (Nutzerwahl): links nur bei Fenster 0 (sonst getrieben),
        # rechts nur beim letzten Fenster (sonst innere Naht -> eigene CPML).
        aface = tuple(f for f in ALL_FACES
                      if not ((w > 0 and f == 'xmin') or (not last and f == 'xmax')))
        _kind = str(boundary or 'cpml').lower()
        if _kind == 'cpml':                      # eine CPML fuer aussen + Naht
            _pml = make_boundary_3d('cpml', dx, dt, aface + (() if last else ('xmax',)),
                                    shape=(Nxw, Ny, Nz), lam_m=LAM)
            _bnd3d = None
        else:                                    # Naht-CPML nur auf xmax + Nutzerrand aussen
            _pml = (None if last else
                    make_boundary_3d('cpml', dx, dt, ('xmax',), shape=(Nxw, Ny, Nz), lam_m=LAM))
            _bnd3d = make_boundary_3d(_kind, dx, dt, aface, shape=(Nxw, Ny, Nz),
                                      speeds=_msp3)  # None=sponge
        _cpml3d = _pml is not None
        if w == 0:
            print(f'Outer absorber: {_kind}; inner window boundaries: CPML (extension)')
        for n in range(steps_win):
            if _bnd3d is not None:
                _bnd3d.capture(Ex, Ey, Ez)
            if _cpml3d:
                _pml.step(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, Ch, da=_da)
            else:
                _yee_step(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, Ch, da=_da)
            t_phys = n*dt
            env = float(1 - np.exp(-((t_phys/(2*sigma_t))**2)))
            if w == 0:
                _vs.inject(src_field, t_phys, env)     # VCSEL (einseitig, Huygens)
            else:
                _ph = np.complex64(np.exp(1j*omega*t_phys))
                for F, D in zip((Ex, Ey, Ez), _drive):      # alle drei E-Komponenten
                    F[:O_cells] = _wtap*(env*xp.real(D*_ph)) + (1.0 - _wtap)*F[:O_cells]
            # Naht-/Aussen-CPML: Absorption + PEC-Abschluss bereits im step.
            # Nutzerrand auf den uebrigen Aussenflaechen:
            if _bnd3d is not None:
                _bnd3d.apply(Ex, Ey, Ez)          # Mur-1
            elif _kind == 'sponge':
                for F in (Ex, Ey, Ez, Hx, Hy, Hz):
                    _apply_sponge(F, g_sp, n_sp, aface)
            if n >= steps_win - n_acc:
                # Ê += E(t)*e^{-iωt} = E*cosωt - i E*sinωt, komplett auf der GPU (float32).
                acc_re += src_field[:Nx_win]*np.float32(np.cos(omega*t_phys))
                acc_im -= src_field[:Nx_win]*np.float32(np.sin(omega*t_phys))
                acc_n += 1
                for _k, _il in _fpl.items():             # exakter Fluss (E und H)
                    _flux.add(_k, t_phys, pairs_3d(_il, Ey, Ez, Hy, Hz))
                if hand_acc is not None:
                    _e = np.complex64(np.exp(-1j*omega*t_phys))
                    for A, F in zip(hand_acc, (Ex, Ey, Ez)):
                        A += F[_off:_off+O_cells]*_e
                if (n - (steps_win - n_acc)) % _m_I == 0:
                    _accum_E2(acc_I, Ex[:Nx_win-1], Ey[:Nx_win], Ez[:Nx_win])  # <|E|^2>
                    n_I += 1
            # Fortschritt INNERHALB des Fensters (~alle 10%) -> im Log sichtbar
            if (n+1) % max(1, steps_win//10) == 0:
                print(f'  [Window {w+1}/{n_win}] Step {n+1}/{steps_win} '
                      f'({100*(n+1)/steps_win:.0f}%)  max|E|={float(xp.max(xp.abs(src_field))):.3e}',
                      flush=True)
        # Ê = (2/N)*(acc_re + i*acc_im): je ein DtoH-Transfer, direkt in complex64
        # (kein complex128-Zwischenprodukt).
        _sc = np.float32(2.0/max(acc_n, 1))
        Ew = np.empty((Nx_win, _Nj, _Nk), dtype=np.complex64)
        Ew.real = to_np(acc_re*_sc)
        Ew.imag = to_np(acc_im*_sc)
        if hand_acc is not None:                   # Handoff (bleibt auf dem Device)
            Ehand = [A*np.complex64(2.0/max(acc_n, 1)) for A in hand_acc]
            del hand_acc
        # Absorber liegt bei inneren Fenstern in der (nicht assemblierten) Verlaengerung
        # -> [0, Nx_win) ist ungedaempft und wird bis zum Rand uebernommen. Links wird
        # der eingepraegte Overlap (O Zellen) ausgelassen; das letzte Fenster laeuft bis
        # zur realen Domaingrenze (echter Absorber, wie im full-Lauf).
        if w == 0:
            g_lo, l_lo = 0, 0
        else:
            l_lo = O_cells
            g_lo = gx0 + l_lo
        g_hi = Nx_full if last else min(gx0 + Nx_win, Nx_full)
        l_hi = l_lo + (g_hi - g_lo)
        if g_hi > g_lo:
            Efull[g_lo:g_hi, :, :] = Ew[l_lo:l_hi, :, :]
            Iavg[g_lo:g_hi, :, :] = to_np(acc_I[l_lo:l_hi]*np.float32(1.0/max(n_I, 1)))
        # Fortschritts-Max per Subsample (voll |Ew| waere ein 229MB-Host-Temp).
        _mxe = float(np.abs(Ew[::4, ::4, ::4]).max())
        print(f'[Window {w+1}/{n_win}] x0={x0_um:.1f}um  max|Ez|={_mxe:.3e}')
        src_field = None; Ew = None
        del Ex, Ey, Ez, Hx, Hy, Hz, inv, ce_x, ce_y, ce_z, acc_re, acc_im, acc_I
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()   # Host-Staging der DtoH-Transfers
        Efull.flush(); Iavg.flush()                # Fenster-Slice auf Disk -> Page-Cache freigeben

    Efull.flush()
    total_time = time.time() - t_total
    nfr = max(1, n_snapshots)
    p = os.path.join('results', f'_stream3d_{_safe}_stitch_ez.npy')
    _stream_paths.append(p)
    vol_buf = np.lib.format.open_memmap(p, mode='w+', dtype=vol_dt,
                                        shape=(nfr, Nx_full, _Nj, _Nk))
    snap_times = [(k/nfr)*(1.0/f0) for k in range(nfr)]
    snap_steps = [k+1 for k in range(nfr)]
    # Re(E*e^{iωt}) = Re*cos - Im*sin ohne complex128-Promotion (float32-Views).
    _cos = [float(np.cos(omega*tk)) for tk in snap_times]
    _sin = [float(np.sin(omega*tk)) for tk in snap_times]
    # Frames blockweise in x aus dem Efull-Memmap erzeugen (kein 22-GB-Peak).
    # Iavg (<|E|^2> aller Komponenten) ist bereits pro Fenster assembliert.
    _CHUNK = 512
    for x0 in range(0, Nx_full, _CHUNK):
        x1 = min(x0 + _CHUNK, Nx_full)
        blk = np.asarray(Efull[x0:x1])                 # (chunk,Ny,Nz-1) complex64
        for k in range(nfr):
            vol_buf[k, x0:x1] = (blk.real*_cos[k] - blk.imag*_sin[k]).astype(vol_dt)
    vol_buf.flush(); Iavg.flush()
    # Sensor-Kennzahlen aus der vollen Intensitaet an Detektor-Ebenen (Einzel-x-
    # Ebenen); 2*<|E|^2> = |Ê|^2 -> gleiche Skala wie frueher.
    iin = int(round(det_in_um/dx_um)); iout = int(round(det_out_um/dx_um))
    iin = min(max(iin, 0), Nx_full-1); iout = min(max(iout, 0), Nx_full-1)
    P_in = 2.0*float(np.sum(np.asarray(Iavg[iin, j_wg0:j_wg1, n_sp:-n_sp])))*dx**3
    P_out = 2.0*float(np.sum(np.asarray(Iavg[iout, _jo0:_jo1, n_sp:-n_sp])))*dx**3
    Tr_I = P_out/max(P_in, 1e-30)                 # alt: 2<|E|^2> in Kern-Ebenen
    # Einheitliche Transmission (wie 2D und 3D full): CW-Poynting-Fluss durch die
    # Detektor-Ebenen ueber den ganzen (y, z)-Querschnitt, TM mit eps gewichtet.
    Tr = Tr_I; tr_def = 'intensity'; _p_si = None
    try:
        if _flux.has('in') and _flux.has('out'):   # exakt aus E und H
            _S = [_flux.flux('in'), _flux.flux('out')]
            _p_si = _S[0]*dx*dx  # eingekoppelte Leistung [W]
            if abs(_S[0]) > 0:
                Tr = _S[1]/_S[0]; tr_def = 'poynting'
            raise StopIteration
        _S = []
        for _ix in (min(max(iin, 1), Nx_full - 2), min(max(iout, 1), Nx_full - 2)):
            _pl = np.asarray(Efull[_ix - 1:_ix + 2])
            _e = None
            if _pol == 'p':
                _e = _eps_planes_3d(cad, _ix, dx_um, Ny, Nz, y0_um, z0_um, wg_n, t_wg_um,
                                    t_lip_um, t_aq_um, t_mu_um, n_lip, n_aq, n_mu, n_co, bead,
                                    wg_width_um, n_clad_side, 0.0, x_wg_start_um,
                                    stack=stack)
                _e = 0.5*(_e[:-1] + _e[1:])
            _S.append(_poynting_flux_3d(_pl, _e))
        if abs(_S[0]) > 0:
            Tr = _S[1]/_S[0]; tr_def = 'poynting'
    except StopIteration:
        pass
    except Exception as _ex:
        print(f'  [WARN] Poynting transmission could not be computed ({_ex}) -> intensity')
    # return_efull (Fourier-Stitch): komplexe Amplitude als RAM-Kopie zurueckgeben
    _Efull_ret = np.array(Efull) if return_efull else None
    # Efull-Temp schliessen + loeschen (rein intermediaer, nicht gespeichert)
    try:
        Efull._mmap.close()
    except Exception:
        pass
    try:
        if os.path.exists(_ef_path):
            os.remove(_ef_path)
    except OSError:
        pass
    print(f'\n[Result] {label} [3D stitch]: total {total_time:.0f}s  '
          f'Transmission={Tr:.4f} ({tr_def})  [intensity: {Tr_I:.4f}]  ({n_win} windows)')
    _rs = _result_stack(stack, t_lip_um, t_aq_um, t_mu_um, n_lip, n_aq, n_mu)
    _cw_ok = source_type != 'pulse' and _flux.has('src_up') and _flux.has('src_dn')
    _vres = _vcsel_res(((_flux.flux('src_dn') - _flux.flux('src_up'))*dx*dx) if _cw_ok else None,
                       (-_flux.flux('src_up')*dx*dx) if _cw_ok else None, _p_si, _vd, _air,
                       x_wg_start_um, vcsel_waist_um, vcsel_waist_z_um)
    return dict(**_vres, scenario=label, dim='3D', P_in=P_in, P_out=P_out, p_in_si=_p_si, p_in_unit='W',
                transmission=Tr, vols_array=vol_buf, vols_ex=None, vols_ey=None,
                vector=False, _stream_paths=_stream_paths,
                snap_steps=snap_steps, snap_times=snap_times, Iavg=Iavg,
                total_time=total_time, steps_total=steps_win*n_win, dt=dt,
                dx_nm=dx_nm, lx_um=lx_um, ly_um=ly_um, lz_um=lz_um,
                y0_um=y0_um, z0_um=z0_um, t_wg_um=t_wg_um, wg_n=wg_n,
                layers=_rs[2], stack=_rs[0], stack_custom=_rs[1],
                n_lip=n_lip, n_aq=n_aq, n_mu=n_mu, n_co=n_co,
                lam_nm=lam_nm, source_type=source_type,
                bead=(bead if bead else {}),
                det_in_um=det_in_um, det_out_um=det_out_um,
                x_wg_start_um=x_wg_start_um, end_facet_um=0.0, x_src_um=x_src_um,
                polarization=_pol, view_primary='cw', cw_only=True,
                cad=(dict(cad.meta(), cad_tilt_deg=vcsel_tilt_deg,
                          cad_det=[[iin, iin + 1, j_wg0, j_wg1], [iout, iout + 1, _jo0, _jo1]])
                     if cad is not None else None),
                cad_region=cad_region, Efull=_Efull_ret,
                transmission_intensity=Tr_I, transmission_def=tr_def)


def run_3d_stitched_td(label, wg_n, window_w_um=20.0, slide_um=12.0,
                       t_wg_um=5.0, t_lip_um=0.0, t_aq_um=3.5, t_mu_um=0.5,
                       n_lip=1.480, n_aq=1.336, n_mu=1.342, n_co=1.376,
                       lam_nm=850.0, lx_um=40.0, air_um=3.0, tear_um=None, lz_um=8.0,
                       dx_nm=50.0, vcsel_waist_um=3.0, vcsel_waist_z_um=3.0,
                       source_type='pulse', n_snapshots=8, bead=None, n_sponge=None, sponge_um=1.2,
                       sponge_alpha=0.3, det_in_um=None, det_out_um=None, vol_dtype='float32',
                       wg_width_um=None, n_clad_side=N_AIR, vcsel_tilt_deg=0.0,
                       vcsel_offset_y_um=0.0, vcsel_offset_z_um=0.0, input_gap_um=0.0,
                       polarization='s', boundary='cpml', cad=None, pulse_cycles=30.0,
                       pulse_tmax_fs=None, t_eval_s=None, stack=None):
    """TRANSIENTER 3D-Stitch im ZEITBEREICH (Huygens-Ebenen, TF/SF) - wie
    planar_beads.run_beads_stitched_td, mit allen vier tangentialen Komponenten:
    Fenster w zeichnet an der i0-Ebene des Folgefensters Ey,Ez(n) und Hy,Hz(n+1/2) auf
    (Disk-Memmap); Fenster w+1 praegt sie als TF/SF-Grenze ein:
        Hy[i0-1] -= Ch Ez_inc,  Hz[i0-1] += Ch Ey_inc         (zwischen H- und E-Update)
        Ey[i0]   += ce_y Hz_inc, Ez[i0]  -= ce_z Hy_inc       (nach dem E-Update)
    Gemeinsame Zeitachse, jedes Fenster nur aktiv solange Signal da ist. Absorber:
    CPML (Default) bzw. Sponge auf allen Flaechen (TF/SF-Ebene liegt dahinter). Ausgabe: transiente
    Volumen-Frames (Hauptkomponente) + zeitgemitteltes <|E|^2>. Ohne Dispersion."""
    t_total = time.time()
    if GPU_AVAILABLE:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    dx = dx_nm*1e-9; dx_um = dx_nm/1000.0
    dt = 0.5*dx/(C0*np.sqrt(3))
    global LAM_LOSS
    LAM = lam_nm*1e-9; LAM_LOSS = LAM; f0 = C0/LAM; omega = 2*np.pi*f0
    if cad is not None:
        _cm = cad.meta()
        lx_um, ly_um, lz_um = cad.lx_um, cad.ly_um, cad.lz_um
        y0_um, z0_um = cad.y0_um, cad.z0_um
        wg_n = float(_cm['cad_n'][cad.core_index - 1]); t_wg_um = 0.0
    else:
        tear_um = _resolve_tear_um(tear_um, t_lip_um, t_aq_um, t_mu_um,
                                   (10 if str(boundary or 'cpml').lower() == 'cpml' else
                                    0 if str(boundary or '').lower() == 'mur1' else
                                    _sponge_cells(n_sponge, sponge_um, dx_um)), dx_um,
                                   stack=stack)
        ly_um = t_wg_um + air_um + tear_um
        y0_um = -tear_um; z0_um = -lz_um/2.0
    Ny = int(round(ly_um/dx_um)); Nz = int(round(lz_um/dx_um))
    Nx_full = int(round(lx_um/dx_um))
    Nx_win = min(int(round(window_w_um/dx_um)), Nx_full)
    S_cells = max(1, min(int(round(slide_um/dx_um)), Nx_win))
    O_cells = Nx_win - S_cells
    n_sp = _sponge_cells(n_sponge, sponge_um, dx_um, (Nx_win, Ny, Nz),
                         warn=str(boundary or 'cpml').lower() == 'sponge')
    i0 = n_sp + 2                                   # TF/SF-Ebene im Folgefenster
    if O_cells < i0 + 4:
        raise ValueError(f'Overlap {O_cells} cells < {i0 + 4}: choose window_w >> slide.')
    # VCSEL in Luft vor der Facette: input_gap_um = Abstand Taille -> Facette
    _vd = float(input_gap_um or 0.0)
    x_src_um, x_wg_start_um, _air, _zsrc = _vcsel_geom_3d(n_sp, dx_um, _vd)
    _EXT = n_sp + 4
    x_wg_end_um = lx_um + _EXT*dx_um
    n_win = (int(np.ceil((Nx_full - Nx_win)/S_cells)) + 1) if Nx_full > Nx_win else 1

    def _gx0(w):
        return min(w*S_cells, max(Nx_full - Nx_win, 0))
    sp = pulse_cycles/f0; tc = 3*sp
    t_max = (pulse_tmax_fs*1e-15) if pulse_tmax_fs else (tc + 3*sp + lx_um*1e-6*wg_n*1.15/C0)
    N = int(np.ceil(t_max/dt)) + 1
    t_j = (np.asarray(t_eval_s, dtype=np.float64) if t_eval_s is not None
           else np.linspace(0.0, t_max, max(int(n_snapshots), 8)))
    n_j = np.clip(np.round(t_j/dt).astype(np.int64), 0, N - 1)
    _pol, _Nj, _Nk = _resolve_pol(polarization, Ny, Nz)
    print(f'\n===== 3D time-domain stitch (Huygens/TF-SF): {label} =====')
    print(f'Domain {lx_um:.1f}x{ly_um:.1f}x{lz_um:.1f}um, {n_win} windows of {window_w_um}um '
          f'(overlap {O_cells*dx_um:.1f}um), grid/window {Nx_win}x{Ny}x{Nz}, Sponge {n_sp}')
    print(f'Pulse {pulse_cycles:g} periods (s={sp*1e15:.1f} fs), t_max={t_max*1e15:.0f} fs = {N} steps, '
          f'{len(n_j)} frames; Huygens plane per seam {4*Ny*Nz*4*N/1e9:.2f} GB (disk)')
    if cad is None:
        _stack_banner(stack, bead)
    _cy = _cz = None
    cad_region = None
    if cad is not None:
        _s = cad.source(x_src_um)
        _cy, _cz = _s['y_um'], _s['z_um']
        _air = _cad_air_um_3d(_s, x_src_um); _zsrc = vcsel.z_at_source(_vd, _air)
        x_wg_start_um = float(_s['facet_center'][0])   # Facette (nur Ausgabe/Detektor-Schutz)
        vcsel_tilt_deg = vcsel_tilt_deg + _s['tilt_deg']
        cad_region = np.zeros((Nx_full, Ny, Nz), dtype=np.int8)
        for _x0 in range(0, Nx_full, 256):
            _x1 = min(_x0 + 256, Nx_full)
            cad_region[_x0:_x1] = cad.eps3d(_x0*dx_um, _x1 - _x0, Ny, Nz)[1]
    src_ix, j_src, k_src, _vs = _build_gauss_source(
        n_sp, t_wg_um, y0_um, z0_um, dx_um, dx, LAM, _Nj, _Nk,
        vcsel_waist_um, vcsel_waist_z_um, vcsel_offset_y_um, vcsel_offset_z_um,
        vcsel_tilt_deg, center_y_um=_cy, center_z_um=_cz,
        n_src_fn=_n_source_fn_3d(cad, x_src_um, Ny, Nz, y0_um, dx_um, wg_n, t_wg_um,
                                 n_aq if stack is None else _stack_n_below(stack, n_co),
                                 x_wg_start_um, _pol),
        pol=_pol, wg_width_um=wg_width_um, dt=dt, z_prop_um=_zsrc)
    print('  ' + vcsel.describe(vcsel_waist_um, lam_nm, _vd, _air, LASER_POWER_MW,
                                waist_z_um=vcsel_waist_z_um))
    env_t = np.exp(-((np.arange(N)*dt - tc)/sp)**2)
    if cad is not None:
        (_a0, _a1, j_wg0, j_wg1), (_b0, _b1, _jo0, _jo1) = cad.detectors(cad_region[:, :, cad.core_k(Nz)], 0.0)
        iin, iout = (_a0 + _a1)//2, (_b0 + _b1)//2
    else:
        if det_in_um is None: det_in_um = 0.2*lx_um
        if det_out_um is None: det_out_um = 0.8*lx_um
        iin = int(round(det_in_um/dx_um)); iout = int(round(det_out_um/dx_um))
        j_wg0 = int(round((0.0 - y0_um)/dx_um)); j_wg1 = int(round((t_wg_um - y0_um)/dx_um))
        _jo0, _jo1 = j_wg0, j_wg1
    iin = guard_in_plane(iin, max(src_ix, int(round(x_wg_start_um/dx_um))), Nx_full)   # im WG
    g_sp = _sponge_profile(n_sp, sponge_alpha)
    vol_dt = np.float16 if vol_dtype == 'float16' else np.float32
    _vdt_xp = xp.float16 if vol_dtype == 'float16' else xp.float32
    os.makedirs('results', exist_ok=True)
    _safe = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in str(label))
    _stream_paths = []
    p = os.path.join('results', f'_stream3d_{_safe}_td_ez.npy'); _stream_paths.append(p)
    vol_buf = np.lib.format.open_memmap(p, mode='w+', dtype=vol_dt,
                                        shape=(len(n_j), Nx_full, _Nj, _Nk))
    vol_buf[:] = 0
    p = os.path.join('results', f'_stream3d_{_safe}_td_Iavg.npy'); _stream_paths.append(p)
    Iavg = np.lib.format.open_memmap(p, mode='w+', dtype=np.float32, shape=(Nx_full, _Nj, _Nk))
    Iavg[:] = 0
    _m_I = max(1, int(round((1.0/f0)/dt))//8)
    n_I_glob = max(1, (N + _m_I - 1)//_m_I)        # Zeitmittel ueber die GANZE Zeitachse
    Ch = xp.float32(dt/(MU0*dx))
    shp = {'Ey': (Ny - 1, Nz), 'Ez': (Ny, Nz - 1), 'Hy': (Ny, Nz - 1), 'Hz': (Ny - 1, Nz)}
    rec_in = None
    P_in = P_out = 0.0
    S_in = S_out = 0.0                                          # Poynting (Pulsenergie)
    S_up = S_dn = 0.0                                           # um die VCSEL-Quelle (Fenster 0)
    for w in range(n_win):
        t1 = time.time()
        gx0 = _gx0(w); last = (w == n_win - 1)
        Nxw = Nx_win + (0 if last else _EXT)
        x0_um = gx0*dx_um
        _tp = {}                                  # Bead-Rand fuer Tensor-Mittelung
        if cad is not None:
            eps = cad.eps3d(x0_um, Nxw, Ny, Nz)[0]
        else:
            eps = build_eps3d(Nxw, Ny, Nz, dx_um, y0_um, z0_um, wg_n, t_wg_um,
                              t_lip_um, t_aq_um, t_mu_um, n_lip, n_aq, n_mu, n_co, bead,
                              wg_width_um=wg_width_um, n_clad_side=n_clad_side,
                              x0_um=x0_um, x_wg_start_um=x_wg_start_um, x_wg_end_um=x_wg_end_um,
                              tensor_patch=_tp, stack=stack)
        eps, _eim = _split_eps3(eps)
        inv, _da = _inv_loss(eps, _eim, dt, dx, LAM); del eps, _eim
        ce_x, ce_y, ce_z = _ce_arrays(inv, _tp, dt, dx, dx_um, da=_da)   # versetzt; mit Bead: Tensor-Rand
        _da = _stagger_da(_da)                 # Daempfung an den versetzten Positionen
        inv = None
        Ex = xp.zeros((Nxw-1, Ny, Nz), dtype=xp.float32); Ey = xp.zeros((Nxw, Ny-1, Nz), dtype=xp.float32)
        Ez = xp.zeros((Nxw, Ny, Nz-1), dtype=xp.float32); Hx = xp.zeros((Nxw, Ny-1, Nz-1), dtype=xp.float32)
        Hy = xp.zeros((Nxw-1, Ny, Nz-1), dtype=xp.float32); Hz = xp.zeros((Nxw-1, Ny-1, Nz), dtype=xp.float32)
        src_field = Ey if _pol == 'p' else Ez
        acc_I = xp.zeros((Nx_win, _Nj, _Nk), dtype=xp.float32)
        i_rec = (_gx0(w + 1) - gx0 + i0) if not last else None
        rec = None
        if i_rec is not None:
            rec = {}
            for c, s in shp.items():
                pth = os.path.join('results', f'_stream3d_{_safe}_td_rec{w}_{c}.npy')
                rec[c] = np.lib.format.open_memmap(pth, mode='w+', dtype=np.float32, shape=(N,) + s)
                rec[c][:] = 0
                _stream_paths.append(pth)
        if w == 0:
            n_start = 0; n_in_last = int(np.nonzero(env_t > 1e-7)[0][-1])
        else:
            amp = np.array([float(np.abs(rec_in['Ez'][n]).max() + np.abs(rec_in['Ey'][n]).max())
                            for n in range(N)])
            nz_ = np.nonzero(amp > 1e-7*max(amp.max(), 1e-30))[0]
            n_start = int(nz_[0]) if len(nz_) else N
            n_in_last = int(nz_[-1]) if len(nz_) else 0
        a_g = 0 if w == 0 else O_cells
        b_g = (Nx_full - gx0) if last else Nx_win
        peak = 0.0; n_stop = N
        # Absorber: CPML (Default) auf allen 6 Flaechen, sonst gradierter Sponge. Die
        # TF/SF-Ebene i0 = n_sp+2 liegt hinter der 10-Zellen-PML (n_sp >= 8 -> i0 >= 10).
        _pml = (make_boundary_3d('cpml', dx, dt, ALL_FACES, shape=(Nxw, Ny, Nz), lam_m=LAM)
                if str(boundary or 'cpml').lower() == 'cpml' else None)
        if _pml is not None and i0 < 11:
            _pml = None
            print('  [WARN] TF/SF plane too close to the boundary for CPML -> sponge')
        for n in range(n_start, N):
            if rec is not None:
                rec['Ey'][n] = to_np(Ey[i_rec]); rec['Ez'][n] = to_np(Ez[i_rec])
            if _pml is not None:
                _pml.step_H(Ex, Ey, Ez, Hx, Hy, Hz, Ch)
            else:
                _yee_H(Ex, Ey, Ez, Hx, Hy, Hz, Ch)
            if w > 0:
                Hy[i0 - 1] -= Ch*xp.asarray(rec_in['Ez'][n])
                Hz[i0 - 1] += Ch*xp.asarray(rec_in['Ey'][n])
            if rec is not None:
                rec['Hy'][n] = to_np(Hy[i_rec - 1]); rec['Hz'][n] = to_np(Hz[i_rec - 1])
            if _pml is not None:
                _pml.step_E(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, da=_da)
            else:
                _damp(Ex, Ey, Ez, _da)
                _yee_E(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z)
            t_phys = n*dt
            if w > 0:
                Ey[i0, :, 1:-1] += ce_y[i0, :, 1:-1]*xp.asarray(rec_in['Hz'][n])[:, 1:-1]
                Ez[i0, 1:-1, :] -= ce_z[i0, 1:-1, :]*xp.asarray(rec_in['Hy'][n])[1:-1, :]
            else:
                e = float(env_t[n])
                if e > 1e-9:
                    _vs.inject(src_field, t_phys, e)   # VCSEL (einseitig, Huygens)
            if _pml is None:
                for F in (Ex, Ey, Ez, Hx, Hy, Hz):
                    _apply_sponge(F, g_sp, n_sp, ALL_FACES)
            hit = np.nonzero(n_j == n)[0]
            if len(hit):
                blk = to_np(src_field[a_g:b_g].astype(_vdt_xp))
                for jj in hit:
                    vol_buf[jj, gx0 + a_g:gx0 + b_g] = blk
            if n % _m_I == 0:
                _accum_E2(acc_I, Ex[:Nx_win-1], Ey[:Nx_win], Ez[:Nx_win])
            if w == 0:                                 # Energie um die VCSEL-Quelle
                S_up += _sx_plane_3d(_vs.plane_up, Ey, Ez, Hy, Hz)
                S_dn += _sx_plane_3d(_vs.plane_dn, Ey, Ez, Hy, Hz)
            for ixg, which in ((iin, 'in'), (iout, 'out')):
                il = ixg - gx0
                if a_g <= il < b_g:
                    _jr = slice(j_wg0, j_wg1) if which == 'in' else slice(_jo0, _jo1)
                    v = _box_E2(Ex, Ey, Ez, slice(il, il + 1), _jr, slice(n_sp, -n_sp))
                    _s = _sx_plane_3d(il, Ey, Ez, Hy, Hz) if il >= 1 else 0.0
                    if which == 'in':
                        P_in += v; S_in += _s
                    else:
                        P_out += v; S_out += _s
            if n % 200 == 0 or (n > n_in_last and (n - n_in_last) % 200 == 0):
                cur = float(xp.max(xp.abs(src_field)))
                peak = max(peak, cur)
                if n > n_in_last and cur < 1e-4*max(peak, 1e-30):
                    n_stop = n
                    break
        Iavg[gx0 + a_g:gx0 + b_g] = to_np(acc_I[a_g:b_g]*np.float32(1.0/n_I_glob))
        print(f'[Window {w+1}/{n_win}] x0={x0_um:.1f}um  steps {n_start}..{n_stop} '
              f'({n_stop - n_start} of {N})  {time.time()-t1:.0f}s', flush=True)
        if rec_in is not None:                       # Vorgaenger-Ebene verbraucht -> Disk frei
            for c in list(rec_in):
                pth = rec_in[c].filename
                try:
                    rec_in[c]._mmap.close()
                except Exception:
                    pass
                rec_in[c] = None
                try:
                    os.remove(pth)
                    _stream_paths.remove(pth)
                except (OSError, ValueError):
                    pass
        rec_in = rec
        del Ex, Ey, Ez, Hx, Hy, Hz, inv, ce_x, ce_y, ce_z, acc_I, src_field
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
    vol_buf.flush(); Iavg.flush()
    P_in = float(P_in)*dx**3; P_out = float(P_out)*dx**3   # einmaliger Readout
    Tr_I = P_out/max(P_in, 1e-30)
    S_in = float(S_in); S_out = float(S_out)
    Tr = S_out/S_in if abs(S_in) > 0 else Tr_I                 # Pulsenergie (Poynting)
    tr_def = 'poynting' if abs(S_in) > 0 else 'intensity'
    total_time = time.time() - t_total
    print(f'\n[Result] {label} [3D time-domain stitch]: total {total_time:.0f}s  '
          f'Transmission={Tr:.4f} ({tr_def})  [intensity: {Tr_I:.4f}]  ({n_win} windows)')
    _rs = _result_stack(stack, t_lip_um, t_aq_um, t_mu_um, n_lip, n_aq, n_mu)
    _ea = dx*dx*dt                                              # Flaeche x Zeit -> Energie [J]
    _vres = _vcsel_res((float(S_dn) - float(S_up))*_ea, -float(S_up)*_ea, float(S_in)*_ea, _vd,
                       _air, x_wg_start_um, vcsel_waist_um, vcsel_waist_z_um, energy=True)
    return dict(**_vres, scenario=label, dim='3D', P_in=P_in, P_out=P_out, transmission=Tr,
                vols_array=vol_buf, vols_ex=None, vols_ey=None, vector=False,
                _stream_paths=_stream_paths, snap_steps=[int(n) for n in n_j],
                snap_times=[float(n*dt) for n in n_j], Iavg=Iavg, total_time=total_time,
                steps_total=N, dt=dt, dx_nm=dx_nm, lx_um=lx_um, ly_um=ly_um, lz_um=lz_um,
                y0_um=y0_um, z0_um=z0_um, t_wg_um=t_wg_um, wg_n=wg_n,
                layers=_rs[2], stack=_rs[0], stack_custom=_rs[1],
                n_lip=n_lip, n_aq=n_aq, n_mu=n_mu, n_co=n_co, lam_nm=lam_nm,
                source_type='pulse', bead=(bead if bead else {}),
                det_in_um=iin*dx_um, det_out_um=iout*dx_um, x_wg_start_um=x_wg_start_um,
                end_facet_um=0.0, x_src_um=x_src_um, polarization=_pol,
                view_primary='transient', cw_only=False,
                cad=(dict(cad.meta(), cad_tilt_deg=vcsel_tilt_deg,
                          cad_det=[[iin, iin + 1, j_wg0, j_wg1], [iout, iout + 1, _jo0, _jo1]])
                     if cad is not None else None),
                cad_region=cad_region, transmission_intensity=Tr_I, transmission_def=tr_def)


def _drop_stream_files(res):
    """Temporaere Memmaps eines Ergebnisses schliessen und loeschen."""
    for key in ('vols_array', 'vols_ex', 'vols_ey', 'cw_vol', 'Iavg'):
        a = res.get(key)
        try:
            a._mmap.close()
        except Exception:
            pass
        res[key] = None
    for pth in res.get('_stream_paths') or []:
        try:
            os.remove(pth)
        except OSError:
            pass


def run_3d_stitched_pulse(label, wg_n, window_w_um=20.0, slide_um=12.0, pulse_cycles=30.0,
                          pulse_tmax_fs=None, pulse_nfreq=None, pulse_dispersion=True,
                          disp_map=None, t_eval_s=None, cad=None, stack=None, **kw):
    """TRANSIENTER 3D-Stitch per Fourier-Synthese (wie planar_beads.run_beads_stitched_pulse).

    Je Frequenz w_k ein CW-run_3d_stitched (Hauptkomponente Ez bzw. Ey); H_k = i E_k;
    E(t) = (1/pi) Re sum_k H_k S(w_k) e^{i w_k t} dw, aufsummiert direkt in die Frames
    (Disk-Memmap, x-blockweise). Dispersion: disp_map {'wg_n': 'pmma', 'n_aq': 'aqueous',
    'n_mu': 'mucin', 'n_co': 'cornea', 'n_lip': 'lipid', 'bead': 'polystyrol'} -> diese
    Brechzahlen werden je Frequenz aus n(lambda_k) gesetzt (explizit gesetzte nicht);
    CAD: Szene bei lambda_k. Nebenprodukt: Spektrum T(lambda)."""
    import contextlib
    import io
    t_total = time.time()
    lam_nm = kw.pop('lam_nm', 850.0)
    lam0 = lam_nm*1e-9; f0 = C0/lam0; w0 = 2*np.pi*f0
    sp = pulse_cycles/f0; tc = 3*sp
    lx = cad.lx_um if cad is not None else kw.get('lx_um', 40.0)
    n_core = float(cad.meta()['cad_n'][cad.core_index - 1]) if cad is not None else wg_n
    t_max = (pulse_tmax_fs*1e-15) if pulse_tmax_fs else (tc + 3*sp + lx*1e-6*n_core*1.15/C0)
    half = 5.26/sp
    if pulse_nfreq:
        dw = 2*half/max(int(pulse_nfreq) - 1, 1)
        if 2*np.pi/dw < t_max:
            print(f'  [WARN] {pulse_nfreq} frequencies -> repetition period {2*np.pi/dw*1e15:.0f} fs '
                  f'< t_max {t_max*1e15:.0f} fs: temporal aliasing.')
    else:
        dw = 2*np.pi/(1.6*t_max)
    kk = np.arange(-int(np.ceil(half/dw)), int(np.ceil(half/dw)) + 1)
    w_k = w0 + kk*dw; w_k = w_k[w_k > 0]
    lam_k = 2*np.pi*C0/w_k*1e9

    def G(v):
        return np.sqrt(np.pi)*sp*np.exp(-(v*sp)**2/4.0)*np.exp(-1j*v*tc)
    S_k = (G(w_k - w0) - G(w_k + w0))/(2j)
    n_snap = int(kw.pop('n_snapshots', 8))
    t_j = (np.asarray(t_eval_s, dtype=np.float64) if t_eval_s is not None
           else np.linspace(0.0, t_max, max(n_snap, 8)))
    print(f'\n===== 3D-Fourier-Stitch (transient): {label} =====')
    print(f'Pulse {pulse_cycles:g} periods (s={sp*1e15:.1f} fs), band {lam_k.min():.1f}..{lam_k.max():.1f} nm, '
          f'{len(w_k)} frequencies, t_max={t_max*1e15:.0f} fs, {len(t_j)} frames, '
          f'dispersion {"on" if pulse_dispersion else "off"}')
    if stack is not None and cad is None:
        print(f'Layers: {describe_stack(stack)}'
              + ('  (material layers follow n(lambda_k))' if pulse_dispersion else ''))
    disp_map = dict(disp_map or {}) if pulse_dispersion else {}
    lam_cad0 = cad.lam_nm if cad is not None else None
    fr = None; spec_T = []; r_c = None; E_c = None
    k_c = int(np.argmin(np.abs(w_k - w0)))
    os.makedirs('results', exist_ok=True)
    _safe = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in str(label))
    for k, (wk, lk, Sk) in enumerate(zip(w_k, lam_k, S_k)):
        t1 = time.time()
        kwk = dict(kw); wg_k = wg_n
        for par, mat in disp_map.items():
            if par == 'wg_n':
                wg_k = n_at(mat, lk)
            elif par == 'bead':
                if kwk.get('bead'):
                    kwk['bead'] = dict(kwk['bead'], n=n_at(mat, lk))
            elif mat == 'cornea_layered':               # geschichtete Cornea je Frequenz
                kwk[par] = cornea_default(lk)
            elif isinstance(mat, (tuple, list)):        # (Material, Offset), z.B. Traenen-Szenario
                kwk[par] = n_at(mat[0], lk) + float(mat[1])
            else:
                kwk[par] = n_at(mat, lk)
        if cad is not None and pulse_dispersion:
            cad.lam_nm = float(lk)
        # Stapel: Material-Schichten folgen n(lambda_k), numerische n bleiben fest
        _stk = stack_at_lambda(stack, float(lk)) if (stack is not None and pulse_dispersion) else stack
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_3d_stitched(f'{label}_f{k}', wg_k, window_w_um=window_w_um, slide_um=slide_um,
                                lam_nm=float(lk), n_snapshots=1, return_efull=True, cad=cad,
                                stack=_stk, **kwk)
        E = r.pop('Efull')
        if fr is None:
            pth = os.path.join('results', f'_stream3d_{_safe}_fourier_ez.npy')
            fr = np.lib.format.open_memmap(pth, mode='w+', dtype=np.float32,
                                           shape=(len(t_j),) + E.shape)
            fr[:] = 0.0
            fr_path = pth
        c = (1j*E)*np.complex64(Sk*dw/np.pi)
        for x0 in range(0, E.shape[0], 64):
            cb = c[x0:x0 + 64]
            for j, tj in enumerate(t_j):
                fr[j, x0:x0 + 64] += np.real(cb*np.complex64(np.exp(1j*wk*tj))).astype(np.float32)
        spec_T.append(float(r['transmission']))
        if k == k_c:
            E_c = E
            r_c = r
        else:
            del E
        _drop_stream_files(r)
        print(f'  [{k+1}/{len(w_k)}] lambda={lk:.2f} nm  |S|={abs(Sk)/abs(S_k).max():.3f}  '
              f'T={r["transmission"]:.4f}  ({time.time()-t1:.0f}s)', flush=True)
    if cad is not None:
        cad.lam_nm = lam_cad0
    fr.flush()
    # eingeschwungener Satz bei der Mittenfrequenz + grobes Zeitmittel aus den Frames
    nfr_cw = 8
    cw_path = os.path.join('results', f'_stream3d_{_safe}_fourier_cwvol.npy')
    cw_vol = np.lib.format.open_memmap(cw_path, mode='w+', dtype=np.float32,
                                       shape=(nfr_cw,) + E_c.shape)
    for q in range(nfr_cw):
        cw_vol[q] = np.real(E_c*np.complex64(np.exp(2j*np.pi*q/nfr_cw))).astype(np.float32)
    cw_vol.flush()
    ia_path = os.path.join('results', f'_stream3d_{_safe}_fourier_Iavg.npy')
    Iavg = np.lib.format.open_memmap(ia_path, mode='w+', dtype=np.float32, shape=E_c.shape)
    for x0 in range(0, E_c.shape[0], 64):
        Iavg[x0:x0 + 64] = np.mean(np.asarray(fr[:, x0:x0 + 64], dtype=np.float32)**2, axis=0)
    Iavg.flush()
    res = dict(r_c)
    dt_c = r_c['dt']
    w = np.abs(S_k)**2
    res.update(scenario=label, vols_array=fr, vols_ex=None, vols_ey=None, vector=False,
               _stream_paths=[fr_path, cw_path, ia_path],
               snap_steps=[int(round(t/dt_c)) for t in t_j], snap_times=[float(t) for t in t_j],
               Iavg=Iavg, cw_vol=cw_vol, view_primary='transient', cw_only=False,
               source_type='pulse', lam_nm=lam_nm, total_time=time.time() - t_total,
               transmission=float(np.sum(w*np.array(spec_T))/np.sum(w)),
               spectrum=dict(lam_nm=lam_k, T=np.array(spec_T), weight=np.abs(S_k)/np.abs(S_k).max()),
               pulse=dict(cycles=pulse_cycles, tmax_fs=t_max*1e15, nfreq=len(w_k),
                          dispersion=bool(pulse_dispersion)))
    if stack is not None:                          # Stapel bei lambda0 (nicht lambda_k_c)
        res.update(stack=[dict(Ly) for Ly in stack], stack_custom=True,
                   layers=legacy_layers_tuple(stack))
    print(f'\n[Result] {label} [3D-Fourier-Stitch]: {len(w_k)} frequencies in {res["total_time"]:.0f}s  '
          f'T(weighted)={res["transmission"]:.4f}')
    return res


def save_results_3d(result, out_dir='results', prefix='fdtd3d',
                    material_names=None):
    """Speichert: PKL-Summary, Volumen-NPZ (3D) und Analyzer-kompatible
    Slice-NPZ (z-Mittelebene, lesbar mit fdtd_analyzer.py)."""
    os.makedirs(out_dir, exist_ok=True)
    tag = result['scenario']
    if material_names is None:
        material_names = ['WG', 'Lipid', 'Aqueous', 'Mucin', 'Cornea']
    # Kern, 3 Schichten unter dem Kern, Substrat (Namen: material_names); Bead getrennt
    mat_idx = np.array([result['wg_n'], result['n_lip'],
                        result['n_aq'], result['n_mu'], result['n_co']],
                       dtype=np.float64)
    # Schichtstapel (immer gespeichert): eigener Stapel -> Namen/Indizes variabler Laenge
    # [Kern, *Schichten, Substrat]; Legacy -> wie bisher, Stapel aus den Legacy-Werten.
    _stk = result.get('stack')
    if _stk is None:                              # aeltere Ergebnis-Dicts
        _l = result.get('layers') or (0.0, 3.5e-6, 0.5e-6, result.get('n_aq', 1.336))
        _stk = legacy_stack(_l[0]*1e6, result.get('n_lip', 1.48), _l[1]*1e6,
                            result.get('n_aq', 1.336), _l[2]*1e6, result.get('n_mu', 1.342))
    _pos = {'lipid': 1, 'aqueous': 2, 'mucin': 3}
    if result.get('stack_custom'):
        material_names = ([str(material_names[0])] + [str(Ly['name']) for Ly in _stk]
                          + [str(material_names[-1])])
        mat_idx = np.array([float(np.real(result['wg_n']))]
                           + [complex(Ly['n']).real for Ly in _stk]
                           + [float(result['n_co'])], dtype=np.float64)
    elif len(material_names) > 3:
        _stk = [dict(Ly, name=str(material_names[_pos[Ly['mat']]]))
                if Ly.get('mat') in _pos else dict(Ly) for Ly in _stk]
    result['stack'] = _stk
    _kw_stk = stack_to_npz(_stk)
    # ALLE grossen Feld-Arrays vom PKL-Summary ausschliessen (die kommen ins NPZ).
    # 'cw_vol' fehlte hier -> das volle CW-Volumen wurde ins Pickle geladen -> OOM.
    _big = ('vols', 'vols_array', 'vols_ex', 'vols_ey', 'Iavg', 'cw_vol', 'cad_region')
    summary = {k: v for k, v in result.items() if k not in _big}
    try:
        with open(f'{out_dir}/{prefix}_{tag}.pkl', 'wb') as f:
            pickle.dump(summary, f)
        print(f'  Saved: {out_dir}/{prefix}_{tag}.pkl')
    except Exception as e:                      # PKL darf NIE die NPZ-Speicherung verhindern
        print(f'  [WARN] PKL summary not saved ({e}); continuing with NPZ.')

    # ezs3d ist bereits der vorallokierte Puffer (kein np.stack -> kein RAM-Peak)
    ezs3d = result.get('vols_array')
    if ezs3d is None or ezs3d.shape[0] == 0:
        return
    steps = np.array(result.get('snap_steps', []), dtype=np.int64)
    times = np.array(result.get('snap_times', []), dtype=np.float64)

    # WICHTIG: Frames-Slice (klein) ZUERST speichern, damit ein RAM-Engpass beim
    # grossen Volumen-NPZ den Lauf nicht komplett wertlos macht.
    kmid = ezs3d.shape[3]//2
    # Hauptkomponente fuer den 2D-Schnitt: im Vektor-Modus ist vols_array = Ez
    # (zellzentriert), bei TM ist die getriebene Komponente aber Ey (vols_ey). Frueher
    # landete immer Ez im Schnitt -> bei TM ~0 (nur Bead-Nahfeld), als 'Ey' beschriftet.
    _prim_vol = ezs3d
    if (result.get('vector') and result.get('vols_ey') is not None
            and str(result.get('polarization', 's')).lower() in ('p', 'tm')):
        _prim_vol = result['vols_ey']
    ez_xy = _prim_vol[:, :, :, kmid].astype(np.float32)
    x_end = result['lx_um']
    y_start = result['y0_um']
    y_end = result['y0_um'] + result['ly_um']
    meta = np.array([(0.0, x_end, y_start, y_end, 0, t, s)
                     for t, s in zip(times, steps)], dtype=np.float64)
    # --- eingeschwungener (CW) Satz + Ansichts-Flags -------------------------
    cw_vol = result.get('cw_vol'); _cwonly = bool(result.get('cw_only', False))
    kw_cw_fr = {}; kw_cw_vol = {}
    if cw_vol is not None and not _cwonly:
        # Nur die z-Mittelebene laden (Memmap-Slice), NICHT das ganze Volumen -> kein
        # RAM-Peak fuer das (kleine) Frames-NPZ. Das volle Volumen geht als Memmap
        # ins (spaetere, mit Fallback abgesicherte) Volumen-NPZ.
        cw_xy = np.asarray(cw_vol[:, :, :, cw_vol.shape[3]//2]).astype(np.float32)
        _f0 = result.get('f0') or (2.99792458e8/(result.get('lam_nm', 850.0)*1e-9))
        ncw = cw_xy.shape[0]
        meta_cw = np.array([(0.0, x_end, y_start, y_end, 0, (k/ncw)/_f0, k)
                            for k in range(ncw)], dtype=np.float64)
        kw_cw_fr = dict(Ez_cw=cw_xy, meta_cw=meta_cw)
        kw_cw_vol = dict(Ez_cw=cw_vol)
    _flags = dict(has_transient=np.array([0 if _cwonly else 1]),
                  has_cw=np.array([1 if (_cwonly or cw_vol is not None) else 0]),
                  cw_only=np.array([1 if _cwonly else 0]),
                  view_primary=np.array([str(result.get('view_primary', 'transient'))]))
    _sp = result.get('spectrum')
    if _sp:                                       # Fourier-Stitch: Spektrum in beide NPZ
        _pu = result.get('pulse') or {}
        _flags.update(spec_lam_nm=np.asarray(_sp['lam_nm'], dtype=np.float64),
                      spec_T=np.asarray(_sp['T'], dtype=np.float64),
                      spec_weight=np.asarray(_sp['weight'], dtype=np.float64),
                      pulse_cycles=np.array([_pu.get('cycles', 0.0)]),
                      pulse_dispersion=np.array([int(bool(_pu.get('dispersion', True)))]))
    if material_names is None:
        material_names = ['WG', 'Lipid', 'Aqueous', 'Mucin', 'Cornea']
    bead = result.get('bead') or {}
    _bd = bead.get('d_um', 0.0)
    bead_yc = bead.get('y_um', -_bd/2.0)   # Default: aufliegend (Zentrum bei y=-r)
    bead_zc = bead.get('z_um', 0.0)        # Default: Domainmitte z=0
    # Freie CAD-Geometrie: Koerpertabelle statt Schichtstapel + Regionskarte
    _cad = result.get('cad'); _reg = result.get('cad_region')
    _kw_cad = {}
    if _cad:
        material_names = list(_cad['cad_names'])
        mat_idx = np.array(_cad['cad_n'], dtype=np.float64)
        _kw_cad = dict(geometry=np.array(['cad']),
                       cad_materials=np.array(_cad['cad_materials']),
                       cad_core=np.array([_cad['cad_core']]), cad_tear=np.array([_cad['cad_tear']]),
                       cad_file=np.array([_cad['cad_file']]),
                       cad_tilt_deg=np.array([_cad['cad_tilt_deg']]),
                       cad_det=np.array(_cad['cad_det'], dtype=np.int64),
                       cad_stride=np.array([1, 1], dtype=np.int64),
                       cad_x_shift_um=np.array([_cad['cad_x_shift_um']]),
                       cad_scale_um=np.array([_cad['cad_scale_um']]))
    # exakt akkumuliertes Zeitmittel <E^2> auch in die (kleine) Slice-NPZ legen:
    # der Analyzer nutzt es fuer 'Int-Mittel' (statt ueber wenige, oft phasen-
    # gleiche Snapshots zu mitteln -> sonst lambda/2-Streifen als Artefakt).
    _ia = result.get('Iavg')
    _kw_ia = {}
    try:
        _ia = np.asarray(_ia)
        if _ia.ndim == 3:
            _kw_ia = dict(Iavg=_ia[:, :, _ia.shape[2]//2].astype(np.float32),
                          iavg_full=np.array([1]))   # Iavg = <|E|^2> aller Komponenten
    except Exception:
        _kw_ia = {}
    # eingekoppelte Leistung (SI, ganzer 3D-Querschnitt) -> Leistungs-Skalierung im Analyzer
    _kw_p = ({'p_in_si': np.array([float(result['p_in_si'])]),
              'p_in_unit': np.array([str(result.get('p_in_unit', 'W'))])}
             if result.get('p_in_si') is not None else {})
    _kw_p.update(vcsel.npz_keys(result))                # VCSEL: Laserleistung, Kopplung
    _kw_ia.update(_kw_p)
    fr_path = f'{out_dir}/{prefix}_{tag}_frames.npz'
    np.savez_compressed(
        fr_path, Ez=ez_xy, meta=meta, scenario=tag, **_kw_ia,
        layers=np.array(result['layers'], dtype=np.float64),
        material_names=np.array(material_names),
        material_indices=mat_idx,
        bead_x_um=np.array([bead.get('x_um', 0.0)]),
        bead_diameter_um=np.array([bead.get('d_um', 0.0)]),
        bead_y_um=np.array([bead_yc]), bead_z_um=np.array([bead_zc]),
        bead_n=np.array([bead.get('n', 0.0)]),
        x_wg_start_um=np.array([result.get('x_wg_start_um', 0.0) or 0.0]),
        end_facet_um=np.array([result.get('end_facet_um', 0.0) or 0.0]),
        x_src_um=np.array([result.get('x_src_um', 0.0) or 0.0]),
        lam_nm=np.array([result.get('lam_nm', 850.0)]),
        polarization=np.array([str(result.get('polarization', 's'))]),
        **_kw_stk, **kw_cw_fr, **_flags, **_kw_cad,
        **({'cad_region': np.asarray(_reg)[:, :ez_xy.shape[2], kmid].astype(np.int8)}
           if (_cad and _reg is not None) else {}))
    print(f'  Saved: {fr_path} (z mid-plane, compatible with fdtd_analyzer.py)'
          + ('  +CW' if kw_cw_fr else '') + ('  [cw_only]' if _cwonly else ''))

    # Volles 3D-Volumen (gross) - danach, mit Fallback falls RAM knapp wird.
    # Geometrie-Metadaten mitspeichern, damit der 3D-Viewer Achsen/Overlays kennt.
    vol_path = f'{out_dir}/{prefix}_{tag}_vol3d.npz'
    _iavg = result['Iavg']
    _iavg = _iavg if getattr(_iavg, 'dtype', None) == np.float32 else _iavg.astype(np.float32)
    vol_kw = dict(
        **_kw_p,
        Ez=ezs3d, Iavg=_iavg, iavg_full=np.array([1]),
        steps=steps, times=times, dx_um=result['dx_nm']/1000.0,
        x0_um=0.0, y0_um=result['y0_um'], z0_um=result['z0_um'],
        scenario=tag, dt=result['dt'],
        layers=np.array(result['layers'], dtype=np.float64),
        material_names=np.array(material_names),
        material_indices=mat_idx,
        t_wg_um=np.array([result.get('t_wg_um', 0.0)]),
        wg_width_um=np.array([(result.get('bead') or {}).get('wg_width_um',
                              result.get('wg_width_um', 0.0)) or 0.0]),
        bead_x_um=np.array([bead.get('x_um', 0.0)]),
        bead_diameter_um=np.array([bead.get('d_um', 0.0)]),
        bead_y_um=np.array([bead_yc]), bead_z_um=np.array([bead_zc]),
        bead_n=np.array([bead.get('n', 0.0)]),
        x_wg_start_um=np.array([result.get('x_wg_start_um', 0.0) or 0.0]),
        end_facet_um=np.array([result.get('end_facet_um', 0.0) or 0.0]),
        lx_um=np.array([result.get('lx_um', 0.0)]),
        x_src_um=np.array([result.get('x_src_um', 0.0) or 0.0]),
        lam_nm=np.array([result.get('lam_nm', 850.0)]),
        # Polarisation: fehlte im Volumen -> Analyzer hielt TM-Volumen fuer TE ('Ez')
        polarization=np.array([str(result.get('polarization', 's'))]),
        **_kw_stk, **kw_cw_vol, **_flags)
    if _cad:
        vol_kw.update(_kw_cad)
        vol_kw.pop('t_wg_um', None)                 # kein Schichtstapel -> Analyzer nutzt Regionen
        if _reg is not None:
            _s3 = ezs3d.shape[1:]
            vol_kw['cad_region'] = np.asarray(_reg)[:_s3[0], :_s3[1], :_s3[2]].astype(np.int8)
    # Vektor-Modus: Ex, Ey zusaetzlich (zellzentriert, gleiche Shape wie Ez)
    if result.get('vector') and result.get('vols_ex') is not None:
        vol_kw['Ex'] = result['vols_ex']
        vol_kw['Ey'] = result['vols_ey']
        vol_kw['vector'] = np.array([1])
        print(f'  [Vector mode] saving Ex,Ey,Ez')
    try:
        np.savez_compressed(vol_path, **vol_kw)
        print(f'  Saved: {vol_path} ({ezs3d.nbytes/1e6:.0f} MB raw, '
              f'{ezs3d.shape[0]} volumes {ezs3d.shape[1:]}, dtype={ezs3d.dtype})')
    except (MemoryError, OSError) as e:
        # Kompression braucht Zusatz-RAM. Fallback: unkomprimiert (streamt auf Disk).
        print(f'  [WARN] compressed volume NPZ failed ({e}); '
              f'trying uncompressed ...')
        try:
            np.savez(vol_path, **vol_kw)
            print(f'  Saved: {vol_path} (uncompressed)')
        except (MemoryError, OSError) as e2:
            print(f'  [ERROR] Volume NPZ could not be saved ({e2}). '
                  f'The frame slice + PKL are safe, however.')

    # Temporaere 3D-Stream-Memmaps aufraeumen (Datei lag auf Disk). Auf Windows
    # muessen die Memmap-Handles explizit geschlossen werden, sonst schlaegt
    # os.remove fehl und die grossen Temp-Dateien bleiben liegen.
    _paths = result.get('_stream_paths') or []
    if _paths:
        for _key in ('vols_array', 'vols_ex', 'vols_ey', 'cw_vol'):
            _arr = result.get(_key)
            try:
                _m = getattr(_arr, '_mmap', None) or \
                    getattr(getattr(_arr, 'base', None), '_mmap', None)
                if _m is not None:
                    _m.close()
            except Exception:
                pass
        vol_kw.clear()
        # auch das Iavg-Memmap-Handle schliessen (sonst bleibt Temp-Datei liegen)
        try:
            _ia = result.get('Iavg')
            _m = getattr(_ia, '_mmap', None) or getattr(getattr(_ia, 'base', None), '_mmap', None)
            if _m is not None:
                _m.close()
        except Exception:
            pass
        try:
            del ezs3d
        except Exception:
            pass
        result['vols_array'] = result['vols_ex'] = result['vols_ey'] = None
        result['Iavg'] = None
        import gc
        gc.collect()
        for _p in _paths:
            try:
                if os.path.exists(_p):
                    os.remove(_p)
            except OSError:
                pass
