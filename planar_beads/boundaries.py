"""Absorbierende Raender fuer die 2D-FDTD-Solver (fdtd2d_core).

Auswahl (Parameter ``boundary``):
  'cpml'  - Convolutional PML (Roden-Gedney, Default): gestreckte Koordinaten mit
            rekursiver Faltung (psi-Hilfsfelder). Mediumunabhaengig, beste Absorption
            auch bei schraegem/streifendem Einfall.
  'mur1'  - Mur 1. Ordnung (nur CLI, Experten-Option): inline im Solver, Koeffizient
            mit LOKALER Geschwindigkeit (MurSet: x-Enden c0/n_eff der Grundmode,
            y-Kanten c0/n der Randzellen). Exakt nur fuer senkrechten Einfall der
            Grundmode; hoehere Moden/Streuung reflektieren staerker.
  (mur2 entfernt: kaum besser als mur1 mit lokaler Geschw., mit Stitch-Handoff instabil.)

Beide Polarisationen: TE (Ez, Hx, Hy) -> CPML_TE, TM (Hz, Ex, Ey) -> CPML_TM.
"""
import numpy as np

from common.backend import xp                       # noqa: E402
from common.physics import C0, EPS0                  # noqa: E402
from planar_beads import fused2d                     # noqa: E402  (CUDA-Kernel)


def mur1_coeff(dx, dt, n=1.0):
    """Koeffizient fuer Mur 1. Ordnung bei Phasengeschwindigkeit c0/n.
    n=1 (Vakuum) ist nur in Luft richtig: im Dielektrikum reflektiert Mur mit c0
    die Welle mit |r| ~ (n-1)/(n+1) (PMMA: 0.2). Fuer gefuehrte Wellen gilt c0/n_eff."""
    c = C0/np.asarray(n, dtype=np.float64)
    m = (c*dt - dx)/(c*dt + dx)
    return xp.float32(m) if np.ndim(m) == 0 else xp.asarray(m.astype(np.float32))


def guided_neff(eps_col, dx, lam_m):
    """n_eff der Grundmode eines 1D-Schichtprofils eps(y) (TE-Skalargleichung
    E'' + k0^2 eps E = beta^2 E, finite Differenzen). Liegt das Profil ohne Fuehrung
    vor (z.B. Luft hinter einer Facette), kommt ~sqrt(max eps) der Umgebung heraus."""
    e = np.asarray(eps_col.get() if hasattr(eps_col, 'get') else eps_col, dtype=np.float64)
    k0 = 2*np.pi/lam_m
    try:
        import scipy.sparse as sps
        from scipy.sparse.linalg import eigsh
        N = len(e)
        d = -2.0/dx**2 + k0**2*e
        o = np.full(N - 1, 1.0/dx**2)
        A = sps.diags([o, d, o], [-1, 0, 1], format='csc')
        smax = k0**2*float(e.max())
        val = eigsh(A, k=1, sigma=smax, which='LM', return_eigenvectors=False)[0]
        ne = float(np.sqrt(max(val, 0.0))/k0)
        if np.isfinite(ne) and 1.0 <= ne <= np.sqrt(e.max()) + 1e-6:
            return ne
    except Exception:
        pass
    return float(np.sqrt(e.max()))


def mur_speeds(eps_r, dx, lam_m):
    """Lokale Brechzahlen fuer die Mur-Raender aus eps_r (Nx, Ny):
    x-Enden -> n_eff der dort gefuehrten Grundmode (Welle laeuft mit c0/n_eff hinaus),
    y-Kanten -> lokales n der Randzellen (Abstrahlung ~ senkrecht durch das Medium)."""
    e = eps_r.get() if hasattr(eps_r, 'get') else np.asarray(eps_r)
    e = np.asarray(e, dtype=np.float64)
    return dict(xl=guided_neff(e[1, :], dx, lam_m), xr=guided_neff(e[-2, :], dx, lam_m),
                b=np.sqrt(e[:, 0]), t=np.sqrt(e[:, -1]))


class MurSet:
    """Mur-1-Koeffizienten je Kante (Skalar an den x-Enden, Vektor entlang y-Kanten).
    b/t: fuer Felder auf den Knoten (Nx); bh/th: auf Halbzellen (Nx-1, TM-Ex)."""
    def __init__(self, dx, dt, speeds=None):
        sp = speeds or {}
        self.xl = mur1_coeff(dx, dt, sp.get('xl', 1.0))
        self.xr = mur1_coeff(dx, dt, sp.get('xr', 1.0))
        if 'b' in sp:
            nb, nt = np.asarray(sp['b']), np.asarray(sp['t'])
            self.b = mur1_coeff(dx, dt, nb); self.t = mur1_coeff(dx, dt, nt)
            self.bh = mur1_coeff(dx, dt, 0.5*(nb[:-1] + nb[1:]))
            self.th = mur1_coeff(dx, dt, 0.5*(nt[:-1] + nt[1:]))
        else:
            self.b = self.t = self.bh = self.th = mur1_coeff(dx, dt)

        self.speeds = sp

    def describe(self):
        sp = self.speeds
        if not sp:
            return 'Mur (vacuum c0)'
        s = 'Mur with local velocity: x ends n_eff=%.4f / %.4f' % (sp.get('xl', 1), sp.get('xr', 1))
        if 'b' in sp:
            s += ', y edges n=%.3f..%.3f / %.3f..%.3f' % (np.min(sp['b']), np.max(sp['b']),
                                                         np.min(sp['t']), np.max(sp['t']))
        return s


# ======================================================================
# CPML-Profile (Roden-Gedney, Taflove Kap. 7)
# ======================================================================
def cfs_a_max(lam_m=None):
    """CFS-Parameter a_max [S/m] = 2*pi*eps0*f0/10, f0 = c0/lam (Standard 850 nm).
    CFS-Term a (complex frequency shifted, Roden-Gedney): a_max = 2*pi*eps0*f0/10
    (f0 = c0/lam). Der fruehere Wert 0.05 S/m war gegen sigma_max ~ 4e5 S/m praktisch
    null -> kein CFS. Mit a ~ 2e3 S/m werden evaneszente und streifend einfallende
    Anteile besser absorbiert, die laufende Welle (omega*eps0 ~ 2e4 S/m) kaum anders."""
    return 2.0*np.pi*EPS0*(C0/(lam_m or 850e-9))/10.0


def _cpml_profiles(N, npml_lo, npml_hi, dx, dt, m=3, kappa_max=1.0, a_max=None, R0=1e-6,
                   lam_m=None):
    """1D-Leitfaehigkeits-/Streck-Profile fuer eine Koordinate mit npml_lo Zellen
    am unteren (Index 0) und npml_hi am oberen (Index N-1) Ende. Rueckgabe an
    Ganzzahl-Knoten (E) und Halbzahl-Knoten (H):
      (kappa_e, b_e, c_e)  Laenge N     an i
      (kappa_h, b_h, c_h)  Laenge N-1   an i+1/2
    Ausserhalb der PML: sigma=0, kappa=1, a=0 -> b=1, c=0 (psi bleibt 0). Ein
    Ende mit npml=0 ist kein PML (z.B. innere Handoff-Naht bei stitch)."""
    if npml_lo <= 0 and npml_hi <= 0:                # kein PML in dieser Richtung
        one_e = xp.ones(N, dtype=xp.float32); zero_e = xp.zeros(N, dtype=xp.float32)
        one_h = xp.ones(N - 1, dtype=xp.float32); zero_h = xp.zeros(N - 1, dtype=xp.float32)
        return (one_e, one_e, zero_e), (one_h, one_h, zero_h)
    eta0 = np.sqrt(4e-7*np.pi/EPS0)                  # ~376.73 Ohm
    if a_max is None:                                # CFS-Term aus der Wellenlaenge
        a_max = cfs_a_max(lam_m)

    def _smax(npml):
        return (-(m + 1)*np.log(R0)/(2.0*eta0*npml*dx)) if npml > 0 else 0.0
    smx_lo = _smax(npml_lo); smx_hi = _smax(npml_hi)

    def build(pos):                                  # pos: Knotenpositionen (Zellen)
        depth = np.zeros_like(pos, dtype=np.float64)
        sgmax = np.zeros_like(pos, dtype=np.float64)
        if npml_lo > 0:
            lo = pos < npml_lo
            depth[lo] = (npml_lo - pos[lo])/npml_lo
            sgmax[lo] = smx_lo
        if npml_hi > 0:
            hi = pos > (N - 1 - npml_hi)
            depth[hi] = (pos[hi] - (N - 1 - npml_hi))/npml_hi
            sgmax[hi] = smx_hi
        depth = np.clip(depth, 0.0, 1.0)
        sigma = sgmax*depth**m
        kappa = 1.0 + (kappa_max - 1.0)*depth**m
        a = np.where(depth > 0.0, a_max*(1.0 - depth), 0.0)
        b = np.exp(-(sigma/kappa + a)*dt/EPS0)
        denom = sigma*kappa + kappa*kappa*a
        c = np.zeros_like(sigma)                      # denom=0 (ausserhalb PML) -> c=0
        nz = denom > 0.0
        c[nz] = sigma[nz]*(b[nz] - 1.0)/denom[nz]
        return (xp.asarray(kappa.astype(np.float32)),
                xp.asarray(b.astype(np.float32)),
                xp.asarray(c.astype(np.float32)))

    ke, be, ce = build(np.arange(N, dtype=np.float64))
    kh, bh, ch = build(np.arange(N - 1, dtype=np.float64) + 0.5)
    return (ke, be, ce), (kh, bh, ch)


def _pml_pair(v, default):
    """npml-Angabe -> (lo, hi). Skalar = beide Enden gleich; (lo,hi) = getrennt;
    None = default."""
    if v is None:
        v = default
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


class CPML_TE:
    """CPML fuer TE (Ez, Hx, Hy). psi-Felder ueber die volle Domain (fuer das
    'full'-Fenster ausreichend; PML nur an den 4 Raendern nichttrivial). Der
    aeussere E-Rand wird als PEC (Ez=0) abgeschlossen.

    npml_x/npml_y erlauben richtungsselektives PML (Default beide = npml). Jeder
    Wert ist Skalar (beide Enden) oder (lo,hi) fuer getrennte Enden -> so kann
    stitch CPML nur an den ECHTEN Bauteilenden setzen (Fenster 0 links, letztes
    Fenster rechts) und die inneren Handoff-Naehte auf 0 lassen."""
    def __init__(self, Nx, Ny, dx, dt, npml=10, npml_x=None, npml_y=None, **kw):
        self.npx_lo, self.npx_hi = _pml_pair(npml_x, npml)
        self.npy_lo, self.npy_hi = _pml_pair(npml_y, npml)
        (self.kex, self.bex, self.cex), (self.khx, self.bhx, self.chx) = \
            _cpml_profiles(Nx, self.npx_lo, self.npx_hi, dx, dt, **kw)
        (self.key, self.bey, self.cey), (self.khy, self.bhy, self.chy) = \
            _cpml_profiles(Ny, self.npy_lo, self.npy_hi, dx, dt, **kw)
        z = xp.zeros
        self.psi_Hxy = z((Nx, Ny - 1), dtype=xp.float32)
        self.psi_Hyx = z((Nx - 1, Ny), dtype=xp.float32)
        self.psi_Ezx = z((Nx - 2, Ny - 2), dtype=xp.float32)
        self.psi_Ezy = z((Nx - 2, Ny - 2), dtype=xp.float32)

    def update_H(self, Ez, Hx, Hy, Ch):
        if fused2d.available(Ez, Hx, Hy, self.psi_Hxy, self.psi_Hyx):
            return fused2d.te_update_H(self, Ez, Hx, Hy, Ch)     # 1 Kernel je Feld
        dEzy = Ez[:, 1:] - Ez[:, :-1]                # d Ez/dy an Halb-y
        self.psi_Hxy = self.bhy[None, :]*self.psi_Hxy + self.chy[None, :]*dEzy
        Hx -= Ch*(dEzy/self.khy[None, :] + self.psi_Hxy)
        dEzx = Ez[1:, :] - Ez[:-1, :]                # d Ez/dx an Halb-x
        self.psi_Hyx = self.bhx[:, None]*self.psi_Hyx + self.chx[:, None]*dEzx
        Hy += Ch*(dEzx/self.khx[:, None] + self.psi_Hyx)

    def update_E(self, Ez, Hx, Hy, Ce_E, Ce_H):
        if fused2d.available(Ez, Hx, Hy, Ce_E, Ce_H, self.psi_Ezx, self.psi_Ezy):
            return fused2d.te_update_E(self, Ez, Hx, Hy, Ce_E, Ce_H)
        dHy = Hy[1:, 1:-1] - Hy[:-1, 1:-1]           # d Hy/dx an Ganz-x (innen)
        dHx = Hx[1:-1, 1:] - Hx[1:-1, :-1]           # d Hx/dy an Ganz-y (innen)
        self.psi_Ezx = self.bex[1:-1, None]*self.psi_Ezx + self.cex[1:-1, None]*dHy
        self.psi_Ezy = self.bey[None, 1:-1]*self.psi_Ezy + self.cey[None, 1:-1]*dHx
        Ez[1:-1, 1:-1] = (Ce_E[1:-1, 1:-1]*Ez[1:-1, 1:-1]
                          + Ce_H[1:-1, 1:-1]*(dHy/self.kex[1:-1, None] + self.psi_Ezx
                                              - dHx/self.key[None, 1:-1] - self.psi_Ezy))

    def terminate(self, Ez):                         # PEC-Abschluss (nur PML-Enden)
        if self.npx_lo > 0:
            Ez[0, :] = 0.0
        if self.npx_hi > 0:
            Ez[-1, :] = 0.0
        if self.npy_lo > 0:
            Ez[:, 0] = 0.0
        if self.npy_hi > 0:
            Ez[:, -1] = 0.0

    def roll(self, n_shift):                          # Co-Moving (sliding): psi mitziehen
        for nm in ('psi_Hxy', 'psi_Hyx', 'psi_Ezx', 'psi_Ezy'):
            a = xp.roll(getattr(self, nm), -n_shift, axis=0)
            a[a.shape[0]-n_shift:, :] = 0.0
            setattr(self, nm, a)


class CPML_TM:
    """CPML fuer TM (Hz, Ex, Ey). Aeusserer E-Rand PEC (tangentiales E=0:
    Ey an x-Raendern, Ex an y-Raendern). npml_x/npml_y: Skalar oder (lo,hi)."""
    def __init__(self, Nx, Ny, dx, dt, npml=10, npml_x=None, npml_y=None, **kw):
        self.npx_lo, self.npx_hi = _pml_pair(npml_x, npml)
        self.npy_lo, self.npy_hi = _pml_pair(npml_y, npml)
        (self.kex, self.bex, self.cex), (self.khx, self.bhx, self.chx) = \
            _cpml_profiles(Nx, self.npx_lo, self.npx_hi, dx, dt, **kw)
        (self.key, self.bey, self.cey), (self.khy, self.bhy, self.chy) = \
            _cpml_profiles(Ny, self.npy_lo, self.npy_hi, dx, dt, **kw)
        z = xp.zeros
        self.psi_Hzx = z((Nx - 1, Ny - 1), dtype=xp.float32)
        self.psi_Hzy = z((Nx - 1, Ny - 1), dtype=xp.float32)
        self.psi_Exy = z((Nx - 1, Ny - 2), dtype=xp.float32)
        self.psi_Eyx = z((Nx - 2, Ny - 1), dtype=xp.float32)

    def update_H(self, Ex, Ey, Hz, Ch_tm):
        if fused2d.available(Ex, Ey, Hz, self.psi_Hzx, self.psi_Hzy):
            return fused2d.tm_update_H(self, Ex, Ey, Hz, Ch_tm)
        dEyx = Ey[1:, :] - Ey[:-1, :]                # d Ey/dx an Halb-x
        dExy = Ex[:, 1:] - Ex[:, :-1]                # d Ex/dy an Halb-y
        self.psi_Hzx = self.bhx[:, None]*self.psi_Hzx + self.chx[:, None]*dEyx
        self.psi_Hzy = self.bhy[None, :]*self.psi_Hzy + self.chy[None, :]*dExy
        Hz -= Ch_tm*((dEyx/self.khx[:, None] + self.psi_Hzx)
                     - (dExy/self.khy[None, :] + self.psi_Hzy))

    def update_E(self, Ex, Ey, Hz, Ca_ex, Ca_ey):
        if fused2d.available(Ex, Ey, Hz, Ca_ex, Ca_ey, self.psi_Exy, self.psi_Eyx):
            return fused2d.tm_update_E(self, Ex, Ey, Hz, Ca_ex, Ca_ey)
        dHzy = Hz[:, 1:] - Hz[:, :-1]                # d Hz/dy an Ganz-y (innen)
        self.psi_Exy = self.bey[None, 1:-1]*self.psi_Exy + self.cey[None, 1:-1]*dHzy
        Ex[:, 1:-1] += Ca_ex[:, 1:-1]*(dHzy/self.key[None, 1:-1] + self.psi_Exy)
        dHzx = Hz[1:, :] - Hz[:-1, :]                # d Hz/dx an Ganz-x (innen)
        self.psi_Eyx = self.bex[1:-1, None]*self.psi_Eyx + self.cex[1:-1, None]*dHzx
        Ey[1:-1, :] += -Ca_ey[1:-1, :]*(dHzx/self.kex[1:-1, None] + self.psi_Eyx)

    def terminate(self, Ex, Ey):                     # PEC (nur PML-Enden)
        if self.npx_lo > 0:
            Ey[0, :] = 0.0
        if self.npx_hi > 0:
            Ey[-1, :] = 0.0
        if self.npy_lo > 0:
            Ex[:, 0] = 0.0
        if self.npy_hi > 0:
            Ex[:, -1] = 0.0

    def roll(self, n_shift):                          # Co-Moving (sliding): psi mitziehen
        for nm in ('psi_Hzx', 'psi_Hzy', 'psi_Exy', 'psi_Eyx'):
            a = xp.roll(getattr(self, nm), -n_shift, axis=0)
            a[a.shape[0]-n_shift:, :] = 0.0
            setattr(self, nm, a)


def make_boundary(kind, pol, Nx, Ny, dx, dt, npml=10, npml_x=None, npml_y=None, lam_m=None):
    """Fabrik: CPML-Objekt fuer kind='cpml', None fuer 'mur1' (Mur 1. Ordnung inline im
    Solver, Koeffizienten via MurSet). pol in {'s'/'te','p'/'tm'}; npml_x/npml_y = CPML-Profil.
    mur2 wurde entfernt (kaum besser als mur1 mit lokaler Geschwindigkeit, mit dem
    Stitch-Handoff instabil) -> cpml verwenden."""
    kind = (kind or 'cpml').lower()
    te = str(pol).lower() in ('s', 'te')
    if kind == 'mur1':
        return None
    if kind == 'cpml':
        return ((CPML_TE if te else CPML_TM)
                (Nx, Ny, dx, dt, npml=npml, npml_x=npml_x, npml_y=npml_y, lam_m=lam_m))
    raise ValueError(f"unknown boundary '{kind}' (cpml|mur1; mur2 removed -> cpml)")
