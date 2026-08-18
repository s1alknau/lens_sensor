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
from common.physics import C0, EPS0, MU0, N_AIR, DISPERSION, n_at  # noqa: E402
from planar_3d.boundaries3d import make_boundary_3d               # noqa: E402


def build_eps3d(Nx, Ny, Nz, dx_um, y0_um, z0_um,
                wg_n, t_wg_um, t_lip_um, t_aq_um, t_mu_um,
                n_lip, n_aq, n_mu, n_co, bead=None,
                wg_width_um=None, n_clad_side=N_AIR, end_facet_um=0.0,
                x_wg_start_um=0.0, x0_um=None, x_wg_end_um=None):
    """eps_r auf (Nx,Ny,Nz). Schichtstapel in y, optional Bead-KUGEL.
    bead = dict(x_um=..., d_um=..., n=...) - Kugel-Zentrum bei
    (x_um, -d/2, 0), d.h. aufliegend an der WG-Unterseite wie im 2D-Setup.

    wg_width_um: Breite des WG-KERNS in Tiefenrichtung (z). None oder >= Domain
      -> Kern fuellt die ganze Tiefe (Slab, nur y-Fuehrung, wie zuvor).
      Endlicher Wert -> echter RECHTECK-KANAL: Kern nur bei |z|<wg_width/2,
      seitlich (in z) von Cladding n_clad_side umgeben -> Fuehrung in y UND z.
      Die Tear-/Cornea-Schichten UNTER dem Kern bleiben ueber die volle Tiefe
      erhalten (Substrat)."""
    # Globaler x-Modus (Stitch): xs bezieht sich auf globale Koordinaten mit
    # Offset x0_um; der WG-KERN existiert nur im Bereich [x_wg_start, x_wg_end]
    # (endlicher Slab mit Facetten an BEIDEN Enden). Ist x0_um None -> altes
    # Verhalten (lokale x ab 0, WG fuellt ganze Laenge, nur Front-Gap/Endfacette).
    _global_x = x0_um is not None
    xs_g = (float(x0_um) if _global_x else 0.0) + np.arange(Nx)*dx_um
    ys = y0_um + np.arange(Ny)*dx_um
    col = np.full(Ny, N_AIR**2, dtype=np.float32)
    y_lip_bot = -t_lip_um
    y_aq_bot = y_lip_bot - t_aq_um
    y_mu_bot = y_aq_bot - t_mu_um
    col[ys < y_mu_bot] = n_co**2
    col[(ys >= y_mu_bot) & (ys < y_aq_bot)] = n_mu**2
    col[(ys >= y_aq_bot) & (ys < y_lip_bot)] = n_aq**2
    if t_lip_um > 1e-6:
        col[(ys >= y_lip_bot) & (ys < 0.0)] = n_lip**2
    if not _global_x:
        col[(ys >= 0.0) & (ys <= t_wg_um)] = wg_n**2
    eps = np.broadcast_to(col[None, :, None], (Nx, Ny, Nz)).copy()
    if _global_x:
        # WG-Kern (in y) nur wo der Slab in x liegt: [x_wg_start, x_wg_end]
        core_y = (ys >= 0.0) & (ys <= t_wg_um)
        in_wg_x = (xs_g >= (x_wg_start_um or 0.0))
        if x_wg_end_um is not None:
            in_wg_x &= (xs_g <= x_wg_end_um)
        yz_core = in_wg_x[:, None, None] & core_y[None, :, None]
        eps[np.broadcast_to(yz_core, eps.shape)] = wg_n**2

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
            mask = (XX*XX + YY*YY + ZZ*ZZ) <= r*r
            sub = eps[i0:i1, j0:j1, k0:k1]
            sub[mask] = bead['n']**2
    return eps


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
                        vcsel_offset_y_um, vcsel_offset_z_um, vcsel_tilt_deg):
    """Gauss-Spot-Quelle (Taille in y und z) auf der getriebenen Komponente.
    Zentrum standardmaessig WG-Mitte (y=t_wg/2) und z=0. Rueckgabe
    (src_ix, j_src, k_src, src_p, use_tilt, src_phase); src_phase ist None ohne
    Neigung. Geteilt von run_3d und run_3d_stitched (zuvor doppelt)."""
    src_ix = n_sp + 6
    j_src = min(max(int(round((t_wg_um/2 + vcsel_offset_y_um - y0_um)/dx_um)), 0), Nj-1)
    k_src = min(max(int(round((0.0 + vcsel_offset_z_um - z0_um)/dx_um)), 0), Nk-1)
    wy = max(2.0, vcsel_waist_um/dx_um)
    wz = max(2.0, vcsel_waist_z_um/dx_um)
    jj = xp.arange(Nj, dtype=xp.float32)[:, None]
    kk = xp.arange(Nk, dtype=xp.float32)[None, :]
    src_p = xp.exp(-((jj - j_src)/wy)**2 - ((kk - k_src)/wz)**2)
    # Strahlneigung in der x-y-Ebene: lineare Phasenrampe entlang y ueber die
    # Quellebene -> der Gauss laeuft unter dem Winkel vcsel_tilt_deg zur x-Achse.
    use_tilt = abs(vcsel_tilt_deg) > 1e-9
    src_phase = None
    if use_tilt:
        _ky = 2*np.pi/LAM*np.sin(np.deg2rad(vcsel_tilt_deg))*dx   # Phase pro Zelle in y
        src_phase = xp.asarray(_ky*(np.arange(Nj) - j_src), dtype=xp.float32)[:, None]
    return src_ix, j_src, k_src, src_p, use_tilt, src_phase


def _yee_step(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, Ch):
    """Ein voller Yee-Leapfrog-Schritt: H-Update, dann E-Update (Innenbereich).
    Aktualisiert alle sechs Felder IN PLACE. Identisch fuer run_3d und
    run_3d_stitched (zuvor im heissen Loop dupliziert)."""
    Hx += Ch*((Ey[:, :, 1:] - Ey[:, :, :-1]) - (Ez[:, 1:, :] - Ez[:, :-1, :]))
    Hy += Ch*((Ez[1:, :, :] - Ez[:-1, :, :]) - (Ex[:, :, 1:] - Ex[:, :, :-1]))
    Hz += Ch*((Ex[:, 1:, :] - Ex[:, :-1, :]) - (Ey[1:, :, :] - Ey[:-1, :, :]))
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
                       dx_nm, safety=0.85):
    """Schaetzt VRAM- (Felder) und Host-RAM-Bedarf (Snapshot-Puffer) und
    vergleicht mit dem tatsaechlich Freien. Rueckgabe (ok, report, hints).
    Aendert NICHTS - nur Pruefung + konkrete Empfehlungen."""
    cells = Nx*Ny*Nz
    # GPU-Peak: 6 Felder + inv_eps + Iavg (8) + Temporaere aus den Curl-Updates
    # + CuPy-Pool-Overhead/Fragmentierung. Erfahrungswert ~16x float32 (real gemessen:
    # 55M Zellen -> OOM bei 3.5 GB frei, also Peak > 3.5 GB).
    vram_need = cells*16*4/1e9
    bpp = 2 if vol_dtype == 'float16' else 4
    ncomp = 3 if save_vector else 1
    # Host: build_eps3d legt das volle eps-Array (float32) im RAM an (broadcast+copy)
    # -> das ist die ERSTE grosse Host-Allokation (haeufigste OOM-Ursache). Dazu die
    # Snapshot-Puffer (+30% fuer Kompression/Save).
    host_eps = cells*4/1e9
    host_need = host_eps + n_snapshots*cells*bpp*ncomp/1e9*1.3
    lines = [f'[Ressourcen-Check] Domain {Nx}x{Ny}x{Nz} = {cells/1e6:.0f}M Zellen, '
             f'{n_snapshots} Snapshots, Vektor={"ja" if save_vector else "nein"}']
    ok = True
    hints = []
    g = _gpu_free_gb()
    if g:
        free, total = g
        lines.append(f'  GPU-VRAM:  brauche ~{vram_need:.1f} GB   '
                     f'frei {free:.1f} / {total:.1f} GB')
        if vram_need > safety*free:
            ok = False
            f = vram_need/(safety*max(free, 1e-6))
            hints.append(f'RES auf ~{dx_nm*f**(1/3):.0f} nm erhoehen '
                         f'ODER LAENGE/TIEFE zusammen um Faktor ~{f**0.5:.2f} verkleinern')
    else:
        lines.append(f'  GPU-VRAM:  brauche ~{vram_need:.1f} GB   (frei: unbekannt)')
    h = _host_free_gb()
    if h is not None:
        lines.append(f'  Host-RAM:  brauche ~{host_need:.1f} GB   frei {h:.1f} GB '
                     f'(eps-Array ~{host_eps:.1f} GB + Snapshot-Puffer)')
        if host_need > safety*h:
            ok = False
            new_ns = max(1, int(safety*h/(host_need/max(n_snapshots, 1))))
            hints.append(f'SNAPSHOTS auf ~{new_ns} reduzieren'
                         + (' oder VECTOR=0 setzen' if save_vector else ''))
    else:
        lines.append(f'  Host-RAM:  brauche ~{host_need:.1f} GB   (frei: unbekannt, '
                     f'psutil fehlt)')
    return ok, '\n'.join(lines), hints


def _resolve_tear_um(tear_um, t_lip_um, t_aq_um, t_mu_um, n_sponge, dx_um, cornea_um=1.5):
    """tear_um (Gebiets-Hoehe unter dem WG) aufloesen. None -> AUTO aus den
    physikalischen Schichten + Cornea-Marge + Absorber-Zone (2D macht das ueber
    einen festen Puffer; 3D leitet es hier ab, damit man es nicht doppelt angeben
    muss). Explizit gesetzt -> unveraendert (zum Trimmen der Cornea-Marge)."""
    if tear_um is not None:
        return float(tear_um)
    layers = t_lip_um + t_aq_um + t_mu_um
    absorber = n_sponge*dx_um
    tot = layers + cornea_um + absorber
    print(f'  [Auto] tear = {tot:.2f}um (Lipid+Aqueous+Mucin {layers:.2f} + '
          f'Cornea {cornea_um:g} + Absorber {absorber:.2f}um)')
    return tot


def run_3d(label, wg_n, t_wg_um=5.0,
           t_lip_um=0.0, t_aq_um=4.0, t_mu_um=2.0,
           n_lip=1.480, n_aq=1.336, n_mu=1.342, n_co=1.376,
           lam_nm=850.0,
           lx_um=40.0, air_um=3.0, tear_um=None, lz_um=8.0,
           dx_nm=50.0, vcsel_waist_um=2.0, vcsel_waist_z_um=2.0,
           source_type='cw', n_snapshots=8, steps_factor=2.0,
           bead=None, n_sponge=24, sponge_alpha=0.3,
           det_in_um=None, det_out_um=None, vol_dtype='float32',
           calibrate_steps=0, wg_width_um=None, n_clad_side=N_AIR,
           save_vector=False, check_resources=False,
           pec_faces=(), end_facet_um=0.0,
           vcsel_tilt_deg=0.0, vcsel_offset_y_um=0.0, vcsel_offset_z_um=0.0,
           input_gap_um=0.0, polarization='s', allow_large=False, boundary='sponge'):
    """Volle 3D-FDTD-Simulation. Rueckgabe: result-dict mit 3D-Volumina.

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
            print(f'[GPU-Speicher] vor Lauf: frei {free0/1e9:.1f} / {total0/1e9:.1f} GB')
        except Exception:
            pass
    dx = dx_nm*1e-9
    dx_um = dx_nm/1000.0
    dt = 0.5*dx/(C0*np.sqrt(3))          # CFL 3D: dt <= dx/(c*sqrt(3)), Faktor 0.5
    LAM = lam_nm*1e-9
    f0 = C0/LAM
    sigma_t = 4/(2*np.pi*f0)

    tear_um = _resolve_tear_um(tear_um, t_lip_um, t_aq_um, t_mu_um, n_sponge, dx_um)
    ly_um = t_wg_um + air_um + tear_um
    Nx = int(round(lx_um/dx_um))
    Ny = int(round(ly_um/dx_um))
    Nz = int(round(lz_um/dx_um))
    y0_um = -tear_um                      # y bei j=0
    z0_um = -lz_um/2.0                    # z bei k=0
    n_sp = max(8, min(n_sponge, Nx//6, Ny//6, Nz//6))

    cells = Nx*Ny*Nz
    gb = cells*4*8/1e9                    # 6 Felder + inv_eps + Iavg
    cpl = (LAM/wg_n)/dx
    print(f'\n===== 3D-FDTD: {label} =====')
    print(f'Achsen:  x = LAENGE      = LICHT-PROPAGATION / EINKOPPELRICHTUNG (Quelle bei x=0)')
    print(f'         y = QUERSCHNITT = Schichtstapel (Luft / Wellenleiter / Tear-Film)')
    print(f'         z = TIEFE       = 3. Dimension (in 2D nicht vorhanden)')
    print(f'Domain:  Laenge {lx_um} x Querschnitt {ly_um} x Tiefe {lz_um} um '
          f'= {Nx} x {Ny} x {Nz} ({cells/1e6:.1f}M Zellen, ~{gb:.1f} GB Felder)')
    print(f'dx={dx_nm}nm ({cpl:.1f} Zellen/Lambda im WG), dt={dt*1e15:.4f} fs, '
          f'Sponge={n_sp} Zellen')
    if cpl < 8:
        print('!!! WARNUNG: <8 Zellen/Wellenlaenge im WG - Ergebnis dispersiv verfaelscht.')
    if wg_width_um is not None and 0 < wg_width_um < lz_um:
        print(f'Wellenleiter: RECHTECK-KANAL {t_wg_um} um (Querschnitt/y) '
              f'x {wg_width_um} um (Tiefe/z), Cladding n={n_clad_side:.3f} '
              f'-> Fuehrung in y UND z')
    else:
        print(f'Wellenleiter: SLAB (nur Querschnitt/y gefuehrt, in Tiefe/z unbegrenzt)')

    # Ressourcen-Check IMMER (nicht nur bei --check-resources): verhindert den
    # rohen numpy/CuPy-MemoryError beim eps-Aufbau. Bei Nichtpassen harter Abbruch
    # mit konkreten Vorschlaegen, ausser --allow-large erzwingt es.
    ok, report, hints = check_resources_3d(
        Nx, Ny, Nz, n_snapshots, vol_dtype, save_vector, dx_nm)
    if check_resources or not ok:
        print(report)
    if not ok and not allow_large:
        print('  [ABBRUCH] Konfiguration passt NICHT sicher in die Ressourcen '
              '(sonst MemoryError beim eps-Aufbau).')
        for hnt in hints:
            print('   -> ' + hnt)
        print('   -> oder mit --allow-large trotzdem erzwingen (Risiko OOM/Absturz).')
        raise SystemExit(2)
    if not ok and allow_large:
        print('  [WARNUNG] --allow-large gesetzt: erzwinge trotz Ressourcen-Warnung.')
    elif check_resources:
        print('  [OK] Konfiguration passt in die vorhandenen Ressourcen.')

    pec_faces = tuple(pec_faces or ())
    absorbing_faces = tuple(f for f in ALL_FACES if f not in pec_faces)
    _bnd3d = make_boundary_3d(boundary, dx, dt, absorbing_faces, shape=(Nx, Ny, Nz))
    _cpml3d = _bnd3d is not None and hasattr(_bnd3d, 'step')   # cpml: eigener Yee-Schritt
    print(f'Absorber: {"Sponge (gradiert)" if _bnd3d is None else boundary} '
          f'auf {list(absorbing_faces)}')
    if pec_faces:
        print(f'Raender: PEC (Spiegel) auf {list(pec_faces)}, '
              f'Absorber auf {list(absorbing_faces)}')
    if end_facet_um and end_facet_um > 0:
        print(f'Endfacette: Material->Luft {end_facet_um} um vor dem Ausgang '
              f'(Fresnel-Reflexion)')
    # Einkoppelabstand: Quelle sitzt bei x=(n_sp+6)*dx; der WG beginnt input_gap
    # dahinter -> Laser laeuft input_gap um durch Luft bis zur Eintrittsfacette.
    x_src_um = (n_sp + 6)*dx_um
    x_wg_start_um = (x_src_um + input_gap_um) if (input_gap_um and input_gap_um > 0) else 0.0
    if x_wg_start_um:
        print(f'Einkoppelabstand: Laser bei x={x_src_um:.2f}um, WG-Eintritt bei '
              f'x={x_wg_start_um:.2f}um -> {input_gap_um:g}um Luftweg (Fresnel-Eintritt)')
    eps = build_eps3d(Nx, Ny, Nz, dx_um, y0_um, z0_um, wg_n, t_wg_um,
                      t_lip_um, t_aq_um, t_mu_um, n_lip, n_aq, n_mu, n_co, bead,
                      wg_width_um=wg_width_um, n_clad_side=n_clad_side,
                      end_facet_um=end_facet_um, x_wg_start_um=x_wg_start_um)
    # eps ist float32 -> nur float32 uebertragen (halber Host-Puffer statt float64),
    # Division dann auf dem Device. Vermeidet grossen Pinned-Transfer.
    inv = xp.float32(dt/(EPS0*dx)) / xp.asarray(eps)
    del eps
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
    ce_x = inv[:-1, :, :]                 # Views, kein Extra-RAM
    ce_y = inv[:, :-1, :]
    ce_z = inv[:, :, :-1]
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
    # Komponente. Zentrum standardmaessig WG-Mitte (y=t_wg/2) und z=0.
    src_ix, j_src, k_src, src_p, _use_tilt, src_phase = _build_gauss_source(
        n_sp, t_wg_um, y0_um, z0_um, dx_um, dx, LAM, _Nj, _Nk,
        vcsel_waist_um, vcsel_waist_z_um,
        vcsel_offset_y_um, vcsel_offset_z_um, vcsel_tilt_deg)
    print(f'Polarisation: {_pol}-Pol -> Quelle treibt '
          f'{"Ey (TM, E in x-y-Ebene)" if _pol=="p" else "Ez (TE, E entlang z)"}')

    g_sp = _sponge_profile(n_sp, sponge_alpha)

    # Detektor-Boxen: WG-Querschnitt (y=0..t_wg, z innerhalb Sponge), +-1um in x
    if det_in_um is None:
        det_in_um = 0.2*lx_um
    if det_out_um is None:
        det_out_um = 0.8*lx_um
    j_wg0 = int(round((0.0 - y0_um)/dx_um))
    j_wg1 = int(round((t_wg_um - y0_um)/dx_um))
    hw = max(2, int(round(1.0/dx_um)))    # +-1 um

    def xrange_at(xu):
        c = int(round(xu/dx_um))
        return max(0, c - hw), min(Nx, c + hw)

    iin0, iin1 = xrange_at(det_in_um)
    iout0, iout1 = xrange_at(det_out_um)

    steps_total = int(steps_factor*lx_um*1e-6*wg_n/C0/dt)
    snap_every = max(1, steps_total//max(1, n_snapshots))
    n_avg0 = steps_total//2               # Iavg ueber die 2. Haelfte (steady)
    # CW-Steady per laufendem DFT bei f0 (letzte ~8 Perioden) -> eingeschwungener
    # Frame-Satz zusaetzlich zum transienten. Akkumulator auf getriebener Komponente.
    _omega = 2*np.pi*f0
    _steps_per_T = max(4, int(round((1.0/f0)/dt)))
    n_acc_cw = min(steps_total, 8*_steps_per_T)
    acc_cw = xp.zeros_like(src_field, dtype=xp.complex64); nacc_cw = 0
    print(f'Steps: {steps_total}, Snapshots alle {snap_every}, '
          f'Quelle x={src_ix*dx_um:.1f}um, Det in/out x={det_in_um:.0f}/{det_out_um:.0f}um')
    print(f'Quelle: Gauss-Spot Taille(y={vcsel_waist_um:g}um, z={vcsel_waist_z_um:g}um) '
          f'@ y={(j_src*dx_um + y0_um):.2f}um, z={(k_src*dx_um + z0_um):.2f}um, '
          f'Neigung {vcsel_tilt_deg:g}Grad')
    if bead is not None:
        _byc = bead.get('y_um', -bead['d_um']/2.0)
        _bzc = bead.get('z_um', 0.0)
        print(f"Bead: KUGEL d={bead['d_um']:g}um n={bead['n']:.3f} "
              f"bei x={bead['x_um']:g}um, y={_byc:g}um, z={_bzc:g}um")

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
        print(f'  [Vektor-Modus] speichere Ex,Ey,Ez (zellzentriert, Disk-Stream) '
              f'-> ~3x Volumen-Disk')
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
        print(f'[KALIBRIERUNG] Warmup {warmup} + Messung {calibrate_steps} Steps '
              f'(von {steps_total} echten Steps) ...')
    else:
        n_run = steps_total
    t_cal0 = None

    for n in range(n_run):
        if calibrate_steps and n == warmup:
            if GPU_AVAILABLE:
                cp.cuda.runtime.deviceSynchronize()
            t_cal0 = time.time()
        # --- Mur: E^n-Randebenen VOR dem Update sichern ---
        if _bnd3d is not None and not _cpml3d:
            _bnd3d.capture(Ex, Ey, Ez)
        # --- Yee-Schritt (CPML: eigener Schritt mit psi-Korrektur) ---
        if _cpml3d:
            _bnd3d.step(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, Ch)
        else:
            _yee_step(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, Ch)
        # --- Quelle (soft) ---
        t_phys = n*dt
        if source_type == 'pulse':
            sp = 30/f0
            envelope = float(np.exp(-((t_phys - 3*sp)/sp)**2))
        else:
            envelope = float(1 - np.exp(-((t_phys/(2*sigma_t))**2)))
        if _use_tilt:
            src_field[src_ix, :, :] += envelope*xp.sin(2*np.pi*f0*t_phys + src_phase)*src_p
        else:
            src_field[src_ix, :, :] += envelope*float(np.sin(2*np.pi*f0*t_phys))*src_p
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
        # --- Detektoren (zeitintegriert) auf der getriebenen Hauptkomponente ---
        P_in += float(xp.sum(src_field[iin0:iin1, j_wg0:j_wg1, n_sp:-n_sp]**2))
        P_out += float(xp.sum(src_field[iout0:iout1, j_wg0:j_wg1, n_sp:-n_sp]**2))
        # --- Zeitmittel-Intensitaet (2. Haelfte) ---
        if n >= n_avg0:
            Iavg += src_field*src_field
        # --- eingeschwungenes Ê akkumulieren (letzte ~8 Perioden) ---
        if n >= steps_total - n_acc_cw:
            acc_cw += src_field.astype(xp.complex64)*np.exp(-1j*_omega*t_phys)
            nacc_cw += 1
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
                print(f'  [WARN] Snapshot {ksnap+1} uebersprungen (Speicher: {_snap_e}). '
                      f'Lauf laeuft weiter, {ksnap} Frames bereits gesichert.')
                n_snaps = ksnap        # keine weiteren Snapshots mehr versuchen

    if calibrate_steps:
        if GPU_AVAILABLE:
            cp.cuda.runtime.deviceSynchronize()
        n_measured = n_run - warmup
        dt_cal = time.time() - t_cal0
        per_step = dt_cal/max(1, n_measured)
        rate = cells*n_measured/max(dt_cal, 1e-9)        # Zell-Updates/s
        est_full = per_step*steps_total
        print(f'\n===== KALIBRIERUNG {label} =====')
        print(f'  Gemessen: {n_measured} Steps in {dt_cal:.2f}s '
              f'-> {per_step*1000:.1f} ms/Step')
        print(f'  Rate: {rate/1e9:.2f} Mrd Zell-Updates/s '
              f'({rate/1e6:.0f} Mcell/s)')
        m, s = divmod(int(est_full), 60)
        print(f'  HOCHRECHNUNG voller Lauf ({steps_total} Steps): '
              f'~{est_full:.0f}s = {m}min {s}s')
        print(f'  (Setze calibrate NICHT, um den vollen Lauf zu starten.)')
        return dict(scenario=label, dim='3D', calibrated=True,
                    per_step_s=per_step, cell_update_rate=rate,
                    steps_total=steps_total, est_full_s=est_full,
                    cells=cells, dx_nm=dx_nm)

    n_avg = max(1, steps_total - n_avg0)
    Iavg_np = to_np(Iavg)/n_avg
    P_in *= dx**3
    P_out *= dx**3
    Tr = P_out/max(P_in, 1e-30)
    total_time = time.time() - t_total
    print(f'\n[Result] {label} [3D]: total {total_time:.0f}s  '
          f'P_in={P_in:.3e}  P_out={P_out:.3e}  Transmission={Tr:.4f}')

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
    return dict(scenario=label, dim='3D', P_in=P_in, P_out=P_out,
                transmission=Tr, vols_array=_va,
                vols_ex=_vx, vols_ey=_vy, vector=bool(save_vector),
                _stream_paths=_stream_paths,
                snap_steps=snap_steps, snap_times=snap_times, Iavg=Iavg_np,
                cw_vol=_cw_vol, view_primary='transient', f0=f0,
                total_time=total_time, steps_total=steps_total, dt=dt,
                dx_nm=dx_nm, lx_um=lx_um, ly_um=ly_um, lz_um=lz_um,
                y0_um=y0_um, z0_um=z0_um, t_wg_um=t_wg_um, wg_n=wg_n,
                layers=(t_lip_um*1e-6, t_aq_um*1e-6, t_mu_um*1e-6, n_aq),
                n_lip=n_lip, n_aq=n_aq, n_mu=n_mu, n_co=n_co,
                lam_nm=lam_nm, source_type=source_type,
                bead=(bead if bead else {}),
                det_in_um=det_in_um, det_out_um=det_out_um,
                x_wg_start_um=x_wg_start_um, end_facet_um=(end_facet_um or 0.0),
                x_src_um=x_src_um, polarization=_pol)


def run_3d_stitched(label, wg_n, window_w_um=20.0, slide_um=12.0,
                    t_wg_um=5.0, t_lip_um=0.0, t_aq_um=4.0, t_mu_um=2.0,
                    n_lip=1.480, n_aq=1.336, n_mu=1.342, n_co=1.376,
                    lam_nm=850.0, lx_um=40.0, air_um=3.0, tear_um=None, lz_um=8.0,
                    dx_nm=50.0, vcsel_waist_um=2.0, vcsel_waist_z_um=2.0,
                    source_type='cw', n_snapshots=8,
                    bead=None, n_sponge=24, sponge_alpha=0.3,
                    det_in_um=None, det_out_um=None, vol_dtype='float32',
                    wg_width_um=None, n_clad_side=N_AIR,
                    vcsel_tilt_deg=0.0, vcsel_offset_y_um=0.0, vcsel_offset_z_um=0.0,
                    input_gap_um=0.0, polarization='s', boundary='sponge'):
    """Voller gefuellter CW-Waveguide in 3D per GEBIETS-ZERLEGUNG (Stitch).
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
    LAM = lam_nm*1e-9; f0 = C0/LAM; omega = 2*np.pi*f0
    sigma_t = 4/(2*np.pi*f0); k0 = 2*np.pi/LAM
    tear_um = _resolve_tear_um(tear_um, t_lip_um, t_aq_um, t_mu_um, n_sponge, dx_um)
    ly_um = t_wg_um + air_um + tear_um
    Ny = int(round(ly_um/dx_um)); Nz = int(round(lz_um/dx_um))
    y0_um = -tear_um; z0_um = -lz_um/2.0
    Nx_full = int(round(lx_um/dx_um))
    Nx_win = int(round(window_w_um/dx_um))
    if Nx_win > Nx_full:                       # Fenster nie groesser als Domaene
        Nx_win = Nx_full
        print(f'  [Info] Fensterbreite > Domaene -> auf {lx_um}um (=1 Fenster) begrenzt.')
    S_cells = int(round(slide_um/dx_um))
    S_cells = max(1, min(S_cells, Nx_win))
    O_cells = Nx_win - S_cells
    n_sp = max(8, min(n_sponge, Nx_win//6, Ny//6, Nz//6))
    if O_cells < max(8, n_sp):
        print(f'  [WARN] Ueberlapp {O_cells} < Sponge {n_sp} -> window_w >> slide waehlen.')
    # Quelle/WG-Geometrie in globalen Koordinaten
    x_src_um = (n_sp + 6)*dx_um
    x_wg_start_um = (x_src_um + input_gap_um) if (input_gap_um and input_gap_um > 0) else 0.0
    x_wg_end_um = lx_um
    n_win = (int(np.ceil((Nx_full - Nx_win)/S_cells)) + 1) if Nx_full > Nx_win else 1
    transit = window_w_um*1e-6*wg_n/C0
    steps_win = int(3.0*transit/dt) + 400
    steps_per_T = max(4, int(round((1.0/f0)/dt)))
    n_acc = steps_per_T*8
    cpl = (LAM/wg_n)/dx
    print(f'\n===== 3D-FDTD [stitch/Gebiets-Zerlegung]: {label} =====')
    print(f'Achsen: x=Laenge(Propagation), y=Schichtstapel, z=Tiefe')
    print(f'Domain {lx_um}x{ly_um}x{lz_um}um -> {n_win} Fenster a {window_w_um}um '
          f'(Ueberlapp {O_cells*dx_um:.1f}um), Grid/Fenster {Nx_win}x{Ny}x{Nz}')
    print(f'dx={dx_nm}nm ({cpl:.1f} Zellen/Lambda im WG), Steps/Fenster {steps_win}, Sponge {n_sp}')
    if cpl < 8:
        print('!!! WARNUNG: <8 Zellen/Wellenlaenge im WG - dispersiv verfaelscht.')
    # Groesse des ASSEMBLIERTEN Ergebnisses abschaetzen. Der Stitch spart Speicher
    # bei der RECHNUNG (ein Fenster zur Zeit), aber das volle 3D-Feld ueber die
    # ganze Laenge ist intrinsisch gross -> auf Disk gestreamt (nicht RAM).
    _vcell = Nx_full*Ny*(Nz-1)
    _vb = 2 if vol_dtype == 'float16' else 4
    gb_ef = _vcell*8/1e9                       # Efull complex64 (Disk)
    gb_fr = max(1, n_snapshots)*_vcell*_vb/1e9  # Frames (Disk)
    gb_ia = _vcell*4/1e9                        # Iavg float32 (Disk)
    gb_tot = gb_ef + gb_fr + gb_ia
    print(f'[Speicher] assembliertes Volumen streamt auf DISK: '
          f'Efull {gb_ef:.1f} + Frames {gb_fr:.1f} + Iavg {gb_ia:.1f} = ~{gb_tot:.1f} GB Disk')
    # Per-Fenster GPU-VRAM abschaetzen: 6 Felder + inv(eps) + acc_re + acc_im
    # ~ 9 float32-Volumina der Fenstergroesse (+ Pool-/Temp-Overhead). Passt das
    # nicht in den freien VRAM -> sauber abbrechen statt mitten im Lauf OOM.
    _wcell = Nx_win*Ny*Nz
    gb_win_gpu = _wcell*4*9/1e9
    print(f'[GPU] Bedarf pro Fenster ~{gb_win_gpu:.1f} GB VRAM '
          f'(Fenstergitter {Nx_win}x{Ny}x{Nz})')
    if GPU_AVAILABLE:
        try:
            _free, _tot = cp.cuda.runtime.memGetInfo()
            print(f'[GPU] frei {_free/1e9:.1f} / {_tot/1e9:.1f} GB')
            if gb_win_gpu > 0.85*(_free/1e9):    # >85% des freien VRAM
                raise MemoryError
        except MemoryError:
            print(f'!!! ABBRUCH: Ein Fenster (~{gb_win_gpu:.1f} GB) passt nicht sicher '
                  f'in den freien VRAM. Waehle KLEINERE Fensterbreite (z.B. 8-15um), '
                  f'groebere Aufloesung, oder kleineren Querschnitt (air/tear/lz).')
            raise SystemExit(2)
        except Exception:
            pass
    # DISK: passt das assemblierte Volumen + Frames + Iavg auf die Platte? Sonst
    # crasht der Lauf spaeter beim Schreiben -> hier hart abbrechen.
    try:
        import shutil
        _free_disk = shutil.disk_usage(os.getcwd()).free/1e9
        print(f'[Disk] Bedarf ~{gb_tot:.1f} GB   frei {_free_disk:.1f} GB')
        if gb_tot > 0.95*_free_disk:
            print(f'!!! ABBRUCH: ~{gb_tot:.1f} GB Ausgabe passen nicht auf die Platte '
                  f'({_free_disk:.1f} GB frei). --save-vector weglassen (Frames /3), '
                  f'--snapshots reduzieren, oder kuerzere --length-um.')
            raise SystemExit(2)
    except SystemExit:
        raise
    except Exception:
        pass
    # Host-RAM (nur noch Fenster-Transfer, kein Akkumulator mehr): Warnung, kein Abbruch.
    _hfree = _host_free_gb()
    _host_need = _wcell*16/1e9                  # Ew (complex64) + 2 float32-Transferpuffer
    if _hfree is not None and _host_need > 0.85*_hfree:
        print(f'!!! WARNUNG: Host-RAM knapp (~{_host_need:.1f} GB Fenster-Transfer vs '
              f'{_hfree:.1f} GB frei). Bei OOM: Querschnitt (tear/lz) oder Fenster verkleinern.')
    if gb_tot > 40:
        print(f'!!! WARNUNG: ~{gb_tot:.0f} GB Disk-Bedarf. 3D ueber so lange Strecken '
              f'ist teuer. Erwaege kleineres lx (3D = kurze Strecke) oder groebere '
              f'Aufloesung. Lange Propagation ist die Domaene der 2D-Stitch-Methode.')
    Ch = xp.float32(dt/(MU0*dx))
    g_sp = _sponge_profile(n_sp, sponge_alpha)
    # Polarisation: 's'/TE -> getriebene+assemblierte Komponente Ez (Ny x Nz-1),
    #               'p'/TM -> Ey (Ny-1 x Nz).
    _pol, _Nj, _Nk = _resolve_pol(polarization, Ny, Nz)
    print(f'Polarisation: {_pol}-Pol -> treibt/assembliert '
          f'{"Ey (TM)" if _pol=="p" else "Ez (TE)"}')
    # Quelle (Fenster 0): Gauss-Spot auf der getriebenen Komponente
    src_ix, j_src, k_src, src_p, _use_tilt, src_phase = _build_gauss_source(
        n_sp, t_wg_um, y0_um, z0_um, dx_um, dx, LAM, _Nj, _Nk,
        vcsel_waist_um, vcsel_waist_z_um,
        vcsel_offset_y_um, vcsel_offset_z_um, vcsel_tilt_deg)
    # cosinus-Taper fuer weiche Overlap-Einpraegung (1 -> 0 ueber O_cells)
    _wtap = xp.asarray((0.5*(1.0 + np.cos(np.pi*np.arange(O_cells)/max(O_cells-1, 1)))
                        ).astype(np.float32))[:, None, None]
    # Detektor-Ebenen (global)
    if det_in_um is None: det_in_um = 0.2*lx_um
    if det_out_um is None: det_out_um = 0.8*lx_um
    j_wg0 = int(round((0.0 - y0_um)/dx_um)); j_wg1 = int(round((t_wg_um - y0_um)/dx_um))
    vol_dt = np.float16 if vol_dtype == 'float16' else np.float32
    os.makedirs('results', exist_ok=True)
    _safe = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in str(label))
    _stream_paths = []
    # assembliertes komplexes Amplitudenvolumen auf DISK-Memmap (Shape = Hauptkomp.)
    _ef_path = os.path.join('results', f'_stream3d_{_safe}_stitch_Efull.npy')
    Efull = np.lib.format.open_memmap(_ef_path, mode='w+', dtype=np.complex64,
                                      shape=(Nx_full, _Nj, _Nk))
    Ehand = None
    for w in range(n_win):
        gx0 = w*S_cells
        x0_um = gx0*dx_um
        eps = build_eps3d(Nx_win, Ny, Nz, dx_um, y0_um, z0_um, wg_n, t_wg_um,
                          t_lip_um, t_aq_um, t_mu_um, n_lip, n_aq, n_mu, n_co, bead,
                          wg_width_um=wg_width_um, n_clad_side=n_clad_side,
                          x0_um=x0_um, x_wg_start_um=x_wg_start_um,
                          x_wg_end_um=x_wg_end_um)
        inv = xp.float32(dt/(EPS0*dx)) / xp.asarray(eps); del eps
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
        ce_x = inv[:-1, :, :]; ce_y = inv[:, :-1, :]; ce_z = inv[:, :, :-1]
        Ex = xp.zeros((Nx_win-1, Ny,   Nz),   dtype=xp.float32)
        Ey = xp.zeros((Nx_win,   Ny-1, Nz),   dtype=xp.float32)
        Ez = xp.zeros((Nx_win,   Ny,   Nz-1), dtype=xp.float32)
        Hx = xp.zeros((Nx_win,   Ny-1, Nz-1), dtype=xp.float32)
        Hy = xp.zeros((Nx_win-1, Ny,   Nz-1), dtype=xp.float32)
        Hz = xp.zeros((Nx_win-1, Ny-1, Nz),   dtype=xp.float32)
        src_field = Ey if _pol == 'p' else Ez     # getriebene/assemblierte Komponente
        # DFT-Akkumulation Ê = Σ E(t) e^{-iωt} auf der GPU als Real-/Imagteil (float32).
        # Kein Host-Akkumulator und KEIN to_np pro Step -> Transfer nur EINMAL pro Fenster.
        acc_re = xp.zeros((Nx_win, _Nj, _Nk), dtype=xp.float32)
        acc_im = xp.zeros((Nx_win, _Nj, _Nk), dtype=xp.float32); acc_n = 0
        _drive = xp.asarray(Ehand.astype(np.complex64)) if (w > 0 and Ehand is not None) else None
        # Fenster w>0: linke Flaeche wird getrieben -> NICHT absorbieren
        aface = tuple(f for f in ALL_FACES if not (w > 0 and f == 'xmin'))
        _bnd3d = make_boundary_3d(boundary, dx, dt, aface, shape=(Nx_win, Ny, Nz))
        _cpml3d = _bnd3d is not None and hasattr(_bnd3d, 'step')
        if w == 0:
            print(f'Absorber: {"Sponge (gradiert)" if _bnd3d is None else boundary}')
        for n in range(steps_win):
            if _bnd3d is not None and not _cpml3d:
                _bnd3d.capture(Ex, Ey, Ez)
            if _cpml3d:
                _bnd3d.step(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, Ch)
            else:
                _yee_step(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, Ch)
            t_phys = n*dt
            env = float(1 - np.exp(-((t_phys/(2*sigma_t))**2)))
            if w == 0:
                if _use_tilt:
                    src_field[src_ix, :, :] += env*xp.sin(2*np.pi*f0*t_phys + src_phase)*src_p
                else:
                    src_field[src_ix, :, :] += env*float(np.sin(2*np.pi*f0*t_phys))*src_p
            else:
                _din = env*xp.real(_drive*np.complex64(np.exp(1j*omega*t_phys)))
                src_field[:O_cells, :, :] = _wtap*_din + (1.0 - _wtap)*src_field[:O_cells, :, :]
            if _cpml3d:
                pass                             # Absorption + PEC-Abschluss im step
            elif _bnd3d is not None:
                _bnd3d.apply(Ex, Ey, Ez)
            else:
                for F in (Ex, Ey, Ez, Hx, Hy, Hz):
                    _apply_sponge(F, g_sp, n_sp, aface)
            if n >= steps_win - n_acc:
                # Ê += E(t)*e^{-iωt} = E*cosωt - i E*sinωt, komplett auf der GPU (float32).
                acc_re += src_field*np.float32(np.cos(omega*t_phys))
                acc_im -= src_field*np.float32(np.sin(omega*t_phys))
                acc_n += 1
            # Fortschritt INNERHALB des Fensters (~alle 10%) -> im Log sichtbar
            if (n+1) % max(1, steps_win//10) == 0:
                print(f'  [Fenster {w+1}/{n_win}] Step {n+1}/{steps_win} '
                      f'({100*(n+1)/steps_win:.0f}%)  max|E|={float(xp.max(xp.abs(src_field))):.3e}',
                      flush=True)
        # Ê = (2/N)*(acc_re + i*acc_im): je ein DtoH-Transfer, direkt in complex64
        # (kein complex128-Zwischenprodukt).
        _sc = np.float32(2.0/max(acc_n, 1))
        Ew = np.empty((Nx_win, _Nj, _Nk), dtype=np.complex64)
        Ew.real = to_np(acc_re*_sc)
        Ew.imag = to_np(acc_im*_sc)
        Ehand = Ew[S_cells:S_cells+O_cells, :, :].copy()
        # WICHTIG: die rechte SPONGE-Zone (n_sp Zellen) eines Fensters ist kuenstlich
        # gedaempft -> NICHT assemblieren, sonst entsteht an jeder Naht eine gedaempfte
        # (weisse) Spalte. Sie wird vom sauberen Innenbereich des naechsten Fensters
        # ueberdeckt. Verlangt Ueberlapp O >= n_sp (sonst kleinere Kappung).
        last = (w == n_win - 1)
        tr = min(n_sp, max(0, O_cells - 1))        # Sponge-Kappung (Standard)
        # LINKS immer um tr einruecken -> schliesst nahtlos an das getrimmte rechte
        # Ende des Vorgaengers an. RECHTS nur bei nicht-letztem Fenster kappen;
        # das letzte Fenster laeuft bis zur realen Domaingrenze (echter Absorber).
        if w == 0:
            g_lo, l_lo = 0, 0
        else:
            l_lo = max(0, O_cells - tr)
            g_lo = gx0 + l_lo
        g_hi = Nx_full if last else min(gx0 + Nx_win - tr, Nx_full)
        l_hi = l_lo + (g_hi - g_lo)
        if g_hi > g_lo:
            Efull[g_lo:g_hi, :, :] = Ew[l_lo:l_hi, :, :]
        # Fortschritts-Max per Subsample (voll |Ew| waere ein 229MB-Host-Temp).
        _mxe = float(np.abs(Ew[::4, ::4, ::4]).max())
        print(f'[Fenster {w+1}/{n_win}] x0={x0_um:.1f}um  max|Ez|={_mxe:.3e}')
        src_field = None; Ew = None
        del Ex, Ey, Ez, Hx, Hy, Hz, inv, ce_x, ce_y, ce_z, acc_re, acc_im
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()   # Host-Staging der DtoH-Transfers
        Efull.flush()                              # Fenster-Slice auf Disk -> Page-Cache freigeben

    Efull.flush()
    total_time = time.time() - t_total
    nfr = max(1, n_snapshots)
    p = os.path.join('results', f'_stream3d_{_safe}_stitch_ez.npy')
    _stream_paths.append(p)
    vol_buf = np.lib.format.open_memmap(p, mode='w+', dtype=vol_dt,
                                        shape=(nfr, Nx_full, _Nj, _Nk))
    _iavg_path = os.path.join('results', f'_stream3d_{_safe}_stitch_Iavg.npy')
    _stream_paths.append(_iavg_path)
    Iavg = np.lib.format.open_memmap(_iavg_path, mode='w+', dtype=np.float32,
                                     shape=(Nx_full, _Nj, _Nk))
    snap_times = [(k/nfr)*(1.0/f0) for k in range(nfr)]
    snap_steps = [k+1 for k in range(nfr)]
    # Re(E*e^{iωt}) = Re*cos - Im*sin ohne complex128-Promotion (float32-Views).
    _cos = [float(np.cos(omega*tk)) for tk in snap_times]
    _sin = [float(np.sin(omega*tk)) for tk in snap_times]
    # Frames + Iavg blockweise in x aus dem Efull-Memmap erzeugen (kein 22-GB-Peak)
    _CHUNK = 512
    for x0 in range(0, Nx_full, _CHUNK):
        x1 = min(x0 + _CHUNK, Nx_full)
        blk = np.asarray(Efull[x0:x1])                 # (chunk,Ny,Nz-1) complex64
        Iavg[x0:x1] = (0.5*np.abs(blk)**2).astype(np.float32)
        for k in range(nfr):
            vol_buf[k, x0:x1] = (blk.real*_cos[k] - blk.imag*_sin[k]).astype(vol_dt)
    vol_buf.flush(); Iavg.flush()
    # Sensor-Kennzahlen aus |Efull|^2 an Detektor-Ebenen (Einzel-x-Ebenen)
    iin = int(round(det_in_um/dx_um)); iout = int(round(det_out_um/dx_um))
    iin = min(max(iin, 0), Nx_full-1); iout = min(max(iout, 0), Nx_full-1)
    P_in = float(np.sum(np.abs(np.asarray(Efull[iin, j_wg0:j_wg1, n_sp:-n_sp]))**2))*dx**3
    P_out = float(np.sum(np.abs(np.asarray(Efull[iout, j_wg0:j_wg1, n_sp:-n_sp]))**2))*dx**3
    Tr = P_out/max(P_in, 1e-30)
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
          f'Transmission~{Tr:.4f}  ({n_win} Fenster)')
    return dict(scenario=label, dim='3D', P_in=P_in, P_out=P_out,
                transmission=Tr, vols_array=vol_buf, vols_ex=None, vols_ey=None,
                vector=False, _stream_paths=_stream_paths,
                snap_steps=snap_steps, snap_times=snap_times, Iavg=Iavg,
                total_time=total_time, steps_total=steps_win*n_win, dt=dt,
                dx_nm=dx_nm, lx_um=lx_um, ly_um=ly_um, lz_um=lz_um,
                y0_um=y0_um, z0_um=z0_um, t_wg_um=t_wg_um, wg_n=wg_n,
                layers=(t_lip_um*1e-6, t_aq_um*1e-6, t_mu_um*1e-6, n_aq),
                n_lip=n_lip, n_aq=n_aq, n_mu=n_mu, n_co=n_co,
                lam_nm=lam_nm, source_type=source_type,
                bead=(bead if bead else {}),
                det_in_um=det_in_um, det_out_um=det_out_um,
                x_wg_start_um=x_wg_start_um, end_facet_um=0.0, x_src_um=x_src_um,
                polarization=_pol, view_primary='cw', cw_only=True)


def save_results_3d(result, out_dir='results', prefix='fdtd3d',
                    material_names=None):
    """Speichert: PKL-Summary, Volumen-NPZ (3D) und Analyzer-kompatible
    Slice-NPZ (z-Mittelebene, lesbar mit fdtd_analyzer.py)."""
    os.makedirs(out_dir, exist_ok=True)
    tag = result['scenario']
    summary = {k: v for k, v in result.items()
               if k not in ('vols', 'vols_array', 'vols_ex', 'vols_ey', 'Iavg')}
    with open(f'{out_dir}/{prefix}_{tag}.pkl', 'wb') as f:
        pickle.dump(summary, f)
    print(f'  Saved: {out_dir}/{prefix}_{tag}.pkl')

    # ezs3d ist bereits der vorallokierte Puffer (kein np.stack -> kein RAM-Peak)
    ezs3d = result.get('vols_array')
    if ezs3d is None or ezs3d.shape[0] == 0:
        return
    steps = np.array(result.get('snap_steps', []), dtype=np.int64)
    times = np.array(result.get('snap_times', []), dtype=np.float64)

    # WICHTIG: Frames-Slice (klein) ZUERST speichern, damit ein RAM-Engpass beim
    # grossen Volumen-NPZ den Lauf nicht komplett wertlos macht.
    kmid = ezs3d.shape[3]//2
    ez_xy = ezs3d[:, :, :, kmid].astype(np.float32)
    x_end = result['lx_um']
    y_start = result['y0_um']
    y_end = result['y0_um'] + result['ly_um']
    meta = np.array([(0.0, x_end, y_start, y_end, 0, t, s)
                     for t, s in zip(times, steps)], dtype=np.float64)
    # --- eingeschwungener (CW) Satz + Ansichts-Flags -------------------------
    cw_vol = result.get('cw_vol'); _cwonly = bool(result.get('cw_only', False))
    kw_cw_fr = {}; kw_cw_vol = {}
    if cw_vol is not None and not _cwonly:
        cw3d = np.asarray(cw_vol)
        cw_xy = cw3d[:, :, :, cw3d.shape[3]//2].astype(np.float32)
        _f0 = result.get('f0') or (2.99792458e8/(result.get('lam_nm', 850.0)*1e-9))
        ncw = cw_xy.shape[0]
        meta_cw = np.array([(0.0, x_end, y_start, y_end, 0, (k/ncw)/_f0, k)
                            for k in range(ncw)], dtype=np.float64)
        kw_cw_fr = dict(Ez_cw=cw_xy, meta_cw=meta_cw)
        kw_cw_vol = dict(Ez_cw=cw3d)
    _flags = dict(has_transient=np.array([0 if _cwonly else 1]),
                  has_cw=np.array([1 if (_cwonly or cw_vol is not None) else 0]),
                  cw_only=np.array([1 if _cwonly else 0]),
                  view_primary=np.array([str(result.get('view_primary', 'transient'))]))
    if material_names is None:
        material_names = ['WG', 'Bead', 'Aqueous', 'Mucin', 'Cornea']
    bead = result.get('bead') or {}
    _bd = bead.get('d_um', 0.0)
    bead_yc = bead.get('y_um', -_bd/2.0)   # Default: aufliegend (Zentrum bei y=-r)
    bead_zc = bead.get('z_um', 0.0)        # Default: Domainmitte z=0
    mat_idx = np.array([result['wg_n'], bead.get('n', result['n_lip']),
                        result['n_aq'], result['n_mu'], result['n_co']],
                       dtype=np.float64)
    fr_path = f'{out_dir}/{prefix}_{tag}_frames.npz'
    np.savez_compressed(
        fr_path, Ez=ez_xy, meta=meta, scenario=tag,
        layers=np.array(result['layers'], dtype=np.float64),
        material_names=np.array(material_names),
        material_indices=mat_idx,
        bead_x_um=np.array([bead.get('x_um', 0.0)]),
        bead_diameter_um=np.array([bead.get('d_um', 0.0)]),
        bead_y_um=np.array([bead_yc]), bead_z_um=np.array([bead_zc]),
        x_wg_start_um=np.array([result.get('x_wg_start_um', 0.0) or 0.0]),
        end_facet_um=np.array([result.get('end_facet_um', 0.0) or 0.0]),
        x_src_um=np.array([result.get('x_src_um', 0.0) or 0.0]),
        lam_nm=np.array([result.get('lam_nm', 850.0)]),
        polarization=np.array([str(result.get('polarization', 's'))]),
        **kw_cw_fr, **_flags)
    print(f'  Saved: {fr_path} (z-Mittelebene, kompatibel zu fdtd_analyzer.py)'
          + ('  +CW' if kw_cw_fr else '') + ('  [cw_only]' if _cwonly else ''))

    # Volles 3D-Volumen (gross) - danach, mit Fallback falls RAM knapp wird.
    # Geometrie-Metadaten mitspeichern, damit der 3D-Viewer Achsen/Overlays kennt.
    vol_path = f'{out_dir}/{prefix}_{tag}_vol3d.npz'
    _iavg = result['Iavg']
    _iavg = _iavg if getattr(_iavg, 'dtype', None) == np.float32 else _iavg.astype(np.float32)
    vol_kw = dict(
        Ez=ezs3d, Iavg=_iavg,
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
        x_wg_start_um=np.array([result.get('x_wg_start_um', 0.0) or 0.0]),
        end_facet_um=np.array([result.get('end_facet_um', 0.0) or 0.0]),
        lx_um=np.array([result.get('lx_um', 0.0)]),
        x_src_um=np.array([result.get('x_src_um', 0.0) or 0.0]),
        lam_nm=np.array([result.get('lam_nm', 850.0)]),
        **kw_cw_vol, **_flags)
    # Vektor-Modus: Ex, Ey zusaetzlich (zellzentriert, gleiche Shape wie Ez)
    if result.get('vector') and result.get('vols_ex') is not None:
        vol_kw['Ex'] = result['vols_ex']
        vol_kw['Ey'] = result['vols_ey']
        vol_kw['vector'] = np.array([1])
        print(f'  [Vektor-Modus] Ex,Ey,Ez werden gespeichert')
    try:
        np.savez_compressed(vol_path, **vol_kw)
        print(f'  Saved: {vol_path} ({ezs3d.nbytes/1e6:.0f} MB raw, '
              f'{ezs3d.shape[0]} Volumina {ezs3d.shape[1:]}, dtype={ezs3d.dtype})')
    except (MemoryError, OSError) as e:
        # Kompression braucht Zusatz-RAM. Fallback: unkomprimiert (streamt auf Disk).
        print(f'  [WARN] komprimiertes Volumen-NPZ fehlgeschlagen ({e}); '
              f'versuche unkomprimiert ...')
        try:
            np.savez(vol_path, **vol_kw)
            print(f'  Saved: {vol_path} (unkomprimiert)')
        except (MemoryError, OSError) as e2:
            print(f'  [FEHLER] Volumen-NPZ nicht speicherbar ({e2}). '
                  f'Frames-Slice + PKL sind aber sicher.')

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
