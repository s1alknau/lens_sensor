"""Absorbierende Raender fuer die 2D-FDTD-Solver (fdtd2d_core).

Drei Auswahlmoeglichkeiten (Parameter ``boundary``):
  'mur1'  - Mur 1. Ordnung (Default, wie bisher; billig, schwach bei schraegem Einfall)
  'mur2'  - Mur 2. Ordnung (transversaler Korrekturterm; deutlich besser bei
            schraegem Einfall, nur Kantenspeicher, kaum Mehrkosten)
  'cpml'  - Convolutional PML (Roden-Gedney): gestreckte Koordinaten mit
            rekursiver Faltung (psi-Hilfsfelder). Beste Absorption, auch bei
            streifendem Einfall; modifiziert die H/E-Updates in den Randstreifen.

Beide Polarisationen:
  TE (Ez, Hx, Hy)   -> Mur2TE  / CPML_TE
  TM (Hz, Ex, Ey)   -> Mur2TM  / CPML_TM

Die Klassen kapseln ihren Zustand (psi-Felder bzw. Kanten-Historie) und werden
pro Fenster einmal gebaut. mur1 bleibt inline im Solver (ein Koeffizient).
"""
import numpy as np

from common.backend import xp                       # noqa: E402
from common.physics import C0, EPS0                  # noqa: E402


def mur1_coeff(dx, dt):
    """Koeffizient fuer Mur 1. Ordnung (Vakuum-Phasengeschw.)."""
    return xp.float32((C0*dt - dx)/(C0*dt + dx))


# ======================================================================
# CPML-Profile (Roden-Gedney, Taflove Kap. 7)
# ======================================================================
def _cpml_profiles(N, npml, dx, dt, m=3, kappa_max=7.0, a_max=0.05, R0=1e-6):
    """1D-Leitfaehigkeits-/Streck-Profile fuer eine Koordinate mit npml Zellen
    an JEDEM Ende. Rueckgabe an Ganzzahl-Knoten (E) und Halbzahl-Knoten (H):
      (kappa_e, b_e, c_e)  Laenge N     an i
      (kappa_h, b_h, c_h)  Laenge N-1   an i+1/2
    Ausserhalb der PML: sigma=0, kappa=1, a=0 -> b=1, c=0 (psi bleibt 0).
    npml=0 -> triviale Profile (kappa=1,b=1,c=0): diese Richtung ist kein PML
    (z.B. stitch: nur y-PML, x bleibt Mur/Handoff)."""
    if npml <= 0:
        one_e = xp.ones(N, dtype=xp.float32); zero_e = xp.zeros(N, dtype=xp.float32)
        one_h = xp.ones(N - 1, dtype=xp.float32); zero_h = xp.zeros(N - 1, dtype=xp.float32)
        return (one_e, one_e, zero_e), (one_h, one_h, zero_h)
    eta0 = np.sqrt(4e-7*np.pi/EPS0)                  # ~376.73 Ohm
    d_pml = npml*dx
    sigma_max = -(m + 1)*np.log(R0)/(2.0*eta0*d_pml)

    def build(pos):                                  # pos: Knotenpositionen (Zellen)
        depth = np.zeros_like(pos, dtype=np.float64)
        left = pos < npml
        depth[left] = (npml - pos[left])/npml
        right = pos > (N - 1 - npml)
        depth[right] = (pos[right] - (N - 1 - npml))/npml
        depth = np.clip(depth, 0.0, 1.0)
        sigma = sigma_max*depth**m
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


class CPML_TE:
    """CPML fuer TE (Ez, Hx, Hy). psi-Felder ueber die volle Domain (fuer das
    'full'-Fenster ausreichend; PML nur an den 4 Raendern nichttrivial). Der
    aeussere E-Rand wird als PEC (Ez=0) abgeschlossen.

    npml_x/npml_y erlauben richtungsselektives PML (Default beide = npml).
    npml_x=0 -> nur y-PML (stitch: x bleibt Mur/Handoff)."""
    def __init__(self, Nx, Ny, dx, dt, npml=10, npml_x=None, npml_y=None, **kw):
        self.npml_x = npml if npml_x is None else npml_x
        self.npml_y = npml if npml_y is None else npml_y
        (self.kex, self.bex, self.cex), (self.khx, self.bhx, self.chx) = \
            _cpml_profiles(Nx, self.npml_x, dx, dt, **kw)
        (self.key, self.bey, self.cey), (self.khy, self.bhy, self.chy) = \
            _cpml_profiles(Ny, self.npml_y, dx, dt, **kw)
        z = xp.zeros
        self.psi_Hxy = z((Nx, Ny - 1), dtype=xp.float32)
        self.psi_Hyx = z((Nx - 1, Ny), dtype=xp.float32)
        self.psi_Ezx = z((Nx - 2, Ny - 2), dtype=xp.float32)
        self.psi_Ezy = z((Nx - 2, Ny - 2), dtype=xp.float32)

    def update_H(self, Ez, Hx, Hy, Ch):
        dEzy = Ez[:, 1:] - Ez[:, :-1]                # d Ez/dy an Halb-y
        self.psi_Hxy = self.bhy[None, :]*self.psi_Hxy + self.chy[None, :]*dEzy
        Hx -= Ch*(dEzy/self.khy[None, :] + self.psi_Hxy)
        dEzx = Ez[1:, :] - Ez[:-1, :]                # d Ez/dx an Halb-x
        self.psi_Hyx = self.bhx[:, None]*self.psi_Hyx + self.chx[:, None]*dEzx
        Hy += Ch*(dEzx/self.khx[:, None] + self.psi_Hyx)

    def update_E(self, Ez, Hx, Hy, Ce_E, Ce_H):
        dHy = Hy[1:, 1:-1] - Hy[:-1, 1:-1]           # d Hy/dx an Ganz-x (innen)
        dHx = Hx[1:-1, 1:] - Hx[1:-1, :-1]           # d Hx/dy an Ganz-y (innen)
        self.psi_Ezx = self.bex[1:-1, None]*self.psi_Ezx + self.cex[1:-1, None]*dHy
        self.psi_Ezy = self.bey[None, 1:-1]*self.psi_Ezy + self.cey[None, 1:-1]*dHx
        Ez[1:-1, 1:-1] = (Ce_E[1:-1, 1:-1]*Ez[1:-1, 1:-1]
                          + Ce_H[1:-1, 1:-1]*(dHy/self.kex[1:-1, None] + self.psi_Ezx
                                              - dHx/self.key[None, 1:-1] - self.psi_Ezy))

    def terminate(self, Ez):                         # PEC-Abschluss (nur PML-Seiten)
        if self.npml_x > 0:
            Ez[0, :] = 0.0; Ez[-1, :] = 0.0
        if self.npml_y > 0:
            Ez[:, 0] = 0.0; Ez[:, -1] = 0.0

    def roll(self, n_shift):                          # Co-Moving (sliding): psi mitziehen
        for nm in ('psi_Hxy', 'psi_Hyx', 'psi_Ezx', 'psi_Ezy'):
            a = xp.roll(getattr(self, nm), -n_shift, axis=0)
            a[a.shape[0]-n_shift:, :] = 0.0
            setattr(self, nm, a)


class CPML_TM:
    """CPML fuer TM (Hz, Ex, Ey). Aeusserer E-Rand PEC (tangentiales E=0:
    Ey an x-Raendern, Ex an y-Raendern). npml_x=0 -> nur y-PML (stitch)."""
    def __init__(self, Nx, Ny, dx, dt, npml=10, npml_x=None, npml_y=None, **kw):
        self.npml_x = npml if npml_x is None else npml_x
        self.npml_y = npml if npml_y is None else npml_y
        (self.kex, self.bex, self.cex), (self.khx, self.bhx, self.chx) = \
            _cpml_profiles(Nx, self.npml_x, dx, dt, **kw)
        (self.key, self.bey, self.cey), (self.khy, self.bhy, self.chy) = \
            _cpml_profiles(Ny, self.npml_y, dx, dt, **kw)
        z = xp.zeros
        self.psi_Hzx = z((Nx - 1, Ny - 1), dtype=xp.float32)
        self.psi_Hzy = z((Nx - 1, Ny - 1), dtype=xp.float32)
        self.psi_Exy = z((Nx - 1, Ny - 2), dtype=xp.float32)
        self.psi_Eyx = z((Nx - 2, Ny - 1), dtype=xp.float32)

    def update_H(self, Ex, Ey, Hz, Ch_tm):
        dEyx = Ey[1:, :] - Ey[:-1, :]                # d Ey/dx an Halb-x
        dExy = Ex[:, 1:] - Ex[:, :-1]                # d Ex/dy an Halb-y
        self.psi_Hzx = self.bhx[:, None]*self.psi_Hzx + self.chx[:, None]*dEyx
        self.psi_Hzy = self.bhy[None, :]*self.psi_Hzy + self.chy[None, :]*dExy
        Hz -= Ch_tm*((dEyx/self.khx[:, None] + self.psi_Hzx)
                     - (dExy/self.khy[None, :] + self.psi_Hzy))

    def update_E(self, Ex, Ey, Hz, Ca_ex, Ca_ey):
        dHzy = Hz[:, 1:] - Hz[:, :-1]                # d Hz/dy an Ganz-y (innen)
        self.psi_Exy = self.bey[None, 1:-1]*self.psi_Exy + self.cey[None, 1:-1]*dHzy
        Ex[:, 1:-1] += Ca_ex[:, 1:-1]*(dHzy/self.key[None, 1:-1] + self.psi_Exy)
        dHzx = Hz[1:, :] - Hz[:-1, :]                # d Hz/dx an Ganz-x (innen)
        self.psi_Eyx = self.bex[1:-1, None]*self.psi_Eyx + self.cex[1:-1, None]*dHzx
        Ey[1:-1, :] += -Ca_ey[1:-1, :]*(dHzx/self.kex[1:-1, None] + self.psi_Eyx)

    def terminate(self, Ex, Ey):                     # PEC (nur PML-Seiten)
        if self.npml_x > 0:
            Ey[0, :] = 0.0; Ey[-1, :] = 0.0
        if self.npml_y > 0:
            Ex[:, 0] = 0.0; Ex[:, -1] = 0.0

    def roll(self, n_shift):                          # Co-Moving (sliding): psi mitziehen
        for nm in ('psi_Hzx', 'psi_Hzy', 'psi_Exy', 'psi_Eyx'):
            a = xp.roll(getattr(self, nm), -n_shift, axis=0)
            a[a.shape[0]-n_shift:, :] = 0.0
            setattr(self, nm, a)


# ======================================================================
# Mur 2. Ordnung (transversaler Korrekturterm; nur Kanten-Historie)
# ======================================================================
def _mur2_coeffs(dx, dt):
    den = C0*dt + dx
    ca = (C0*dt - dx)/den
    cb = 2.0*dx/den
    cc = (C0*dt)**2/(2.0*dx*den)                     # dy=dx
    return (xp.float32(ca), xp.float32(cb), xp.float32(cc),
            xp.float32((C0*dt - dx)/(C0*dt + dx)))    # + mur1 fuer Ecken


class _Mur2Edge:
    """Mur-2 fuer EIN skalares Feld an seinen Aussenkanten. Braucht die zwei
    aeussersten Zell-Lagen zu den Zeiten n (cur) und n-1 (prev).
    sides: 'xy' (alle 4 Kanten), 'x' (nur x) oder 'y' (nur y-Kanten; z.B. stitch,
    wo die x-Kanten Quelle/Handoff sind)."""
    def __init__(self, dx, dt, sides='xy'):
        self.ca, self.cb, self.cc, self.mur1 = _mur2_coeffs(dx, dt)
        self.dox = 'x' in sides; self.doy = 'y' in sides
        self.prev = None; self.cur = None            # dicts der Kantenbaender

    def reset(self):                                 # Historie verwerfen (z.B. nach Slide-Roll)
        self.prev = None; self.cur = None

    def capture(self, F):                            # VOR dem Innen-Update (F=F^n)
        self.prev = self.cur
        self.cur = {
            'L': F[0:2, :].copy(),   'R': F[-1:-3:-1, :].copy(),   # R: [ -1, -2 ]
            'B': F[:, 0:2].copy(),   'T': F[:, -1:-3:-1].copy(),   # T: [ -1, -2 ]
        }

    def apply(self, F):                              # NACH Innen-Update (Innen=F^{n+1})
        if self.prev is None:                        # erster Schritt: Mur-1
            self._mur1(F); return
        ca, cb, cc = self.ca, self.cb, self.cc
        cL, cR, cB, cT = self.cur['L'], self.cur['R'], self.cur['B'], self.cur['T']
        pL, pR, pB, pT = self.prev['L'], self.prev['R'], self.prev['B'], self.prev['T']
        if self.dox:                                 # x-Kanten (links i=0, rechts i=-1)
            F[0, 1:-1] = (-pL[1, 1:-1] + ca*(F[1, 1:-1] + pL[0, 1:-1])
                          + cb*(cL[0, 1:-1] + cL[1, 1:-1])
                          + cc*(cL[0, 2:] - 2*cL[0, 1:-1] + cL[0, :-2]
                                + cL[1, 2:] - 2*cL[1, 1:-1] + cL[1, :-2]))
            F[-1, 1:-1] = (-pR[1, 1:-1] + ca*(F[-2, 1:-1] + pR[0, 1:-1])
                           + cb*(cR[0, 1:-1] + cR[1, 1:-1])
                           + cc*(cR[0, 2:] - 2*cR[0, 1:-1] + cR[0, :-2]
                                 + cR[1, 2:] - 2*cR[1, 1:-1] + cR[1, :-2]))
        if self.doy:                                 # y-Kanten (unten j=0, oben j=-1)
            F[1:-1, 0] = (-pB[1:-1, 1] + ca*(F[1:-1, 1] + pB[1:-1, 0])
                          + cb*(cB[1:-1, 0] + cB[1:-1, 1])
                          + cc*(cB[2:, 0] - 2*cB[1:-1, 0] + cB[:-2, 0]
                                + cB[2:, 1] - 2*cB[1:-1, 1] + cB[:-2, 1]))
            F[1:-1, -1] = (-pT[1:-1, 1] + ca*(F[1:-1, -2] + pT[1:-1, 0])
                           + cb*(cT[1:-1, 0] + cT[1:-1, 1])
                           + cc*(cT[2:, 0] - 2*cT[1:-1, 0] + cT[:-2, 0]
                                 + cT[2:, 1] - 2*cT[1:-1, 1] + cT[:-2, 1]))
        if self.dox and self.doy:
            self._corners(F)

    def _mur1(self, F):
        m = self.mur1
        if self.dox:
            F[0, :] = self.cur['L'][1, :] + m*(F[1, :] - self.cur['L'][0, :])
            F[-1, :] = self.cur['R'][1, :] + m*(F[-2, :] - self.cur['R'][0, :])
        if self.doy:
            F[:, 0] = self.cur['B'][:, 1] + m*(F[:, 1] - self.cur['B'][:, 0])
            F[:, -1] = self.cur['T'][:, 1] + m*(F[:, -2] - self.cur['T'][:, 0])

    def _corners(self, F):                           # Ecken per Mur-1 stabil halten
        m = self.mur1; cL, cR = self.cur['L'], self.cur['R']
        F[0, 0] = cL[1, 0] + m*(F[1, 0] - cL[0, 0])
        F[0, -1] = cL[1, -1] + m*(F[1, -1] - cL[0, -1])
        F[-1, 0] = cR[1, 0] + m*(F[-2, 0] - cR[0, 0])
        F[-1, -1] = cR[1, -1] + m*(F[-2, -1] - cR[0, -1])


class Mur2TE:
    """Mur 2. Ordnung fuer TE: wirkt auf Ez. sides in {'xy','x','y'}."""
    def __init__(self, dx, dt, sides='xy'):
        self.ez = _Mur2Edge(dx, dt, sides=sides)

    def reset(self):
        self.ez.reset()

    def capture(self, Ez):
        self.ez.capture(Ez)

    def apply(self, Ez):
        self.ez.apply(Ez)


class Mur2TM:
    """Mur 2. Ordnung fuer TM: Ey an x-Raendern, Ex an y-Raendern (tangential).
    sides in {'xy','x','y'} waehlt, welche Kanten absorbieren (stitch: 'y')."""
    def __init__(self, dx, dt, sides='xy'):
        self.ca, self.cb, self.cc, self.mur1 = _mur2_coeffs(dx, dt)
        self.dox = 'x' in sides; self.doy = 'y' in sides
        self.pEy = self.cEy = None                   # Ey-Baender an x-Kanten
        self.pEx = self.cEx = None                   # Ex-Baender an y-Kanten

    def reset(self):
        self.pEy = self.cEy = None; self.pEx = self.cEx = None

    def capture(self, Ex, Ey):
        self.pEy = self.cEy; self.cEy = (Ey[0:2, :].copy(), Ey[-1:-3:-1, :].copy())
        self.pEx = self.cEx; self.cEx = (Ex[:, 0:2].copy(), Ex[:, -1:-3:-1].copy())

    def apply(self, Ex, Ey):
        ca, cb, cc, m = self.ca, self.cb, self.cc, self.mur1
        if self.pEy is None:                         # erster Schritt: Mur-1
            cL, cR = self.cEy; cB, cT = self.cEx
            if self.dox:
                Ey[0, :] = cL[1, :] + m*(Ey[1, :] - cL[0, :])
                Ey[-1, :] = cR[1, :] + m*(Ey[-2, :] - cR[0, :])
            if self.doy:
                Ex[:, 0] = cB[:, 1] + m*(Ex[:, 1] - cB[:, 0])
                Ex[:, -1] = cT[:, 1] + m*(Ex[:, -2] - cT[:, 0])
            return
        if self.dox:
            (cL, cR) = self.cEy; (pL, pR) = self.pEy
            Ey[0, 1:-1] = (-pL[1, 1:-1] + ca*(Ey[1, 1:-1] + pL[0, 1:-1])
                           + cb*(cL[0, 1:-1] + cL[1, 1:-1])
                           + cc*(cL[0, 2:] - 2*cL[0, 1:-1] + cL[0, :-2]
                                 + cL[1, 2:] - 2*cL[1, 1:-1] + cL[1, :-2]))
            Ey[-1, 1:-1] = (-pR[1, 1:-1] + ca*(Ey[-2, 1:-1] + pR[0, 1:-1])
                            + cb*(cR[0, 1:-1] + cR[1, 1:-1])
                            + cc*(cR[0, 2:] - 2*cR[0, 1:-1] + cR[0, :-2]
                                  + cR[1, 2:] - 2*cR[1, 1:-1] + cR[1, :-2]))
        if self.doy:
            (bB, tT) = self.cEx; (pB, pT) = self.pEx
            Ex[1:-1, 0] = (-pB[1:-1, 1] + ca*(Ex[1:-1, 1] + pB[1:-1, 0])
                           + cb*(bB[1:-1, 0] + bB[1:-1, 1])
                           + cc*(bB[2:, 0] - 2*bB[1:-1, 0] + bB[:-2, 0]
                                 + bB[2:, 1] - 2*bB[1:-1, 1] + bB[:-2, 1]))
            Ex[1:-1, -1] = (-pT[1:-1, 1] + ca*(Ex[1:-1, -2] + pT[1:-1, 0])
                            + cb*(tT[1:-1, 0] + tT[1:-1, 1])
                            + cc*(tT[2:, 0] - 2*tT[1:-1, 0] + tT[:-2, 0]
                                  + tT[2:, 1] - 2*tT[1:-1, 1] + tT[:-2, 1]))


def make_boundary(kind, pol, Nx, Ny, dx, dt, npml=10, npml_x=None, npml_y=None, sides='xy'):
    """Fabrik: liefert das Rand-Objekt (oder None fuer mur1) fuer kind in
    {'mur1','mur2','cpml'} und pol in {'s'/'te','p'/'tm'}.
    sides waehlt die Mur-2-Kanten; npml_x/npml_y das CPML-Richtungsprofil."""
    kind = (kind or 'mur1').lower()
    te = str(pol).lower() in ('s', 'te')
    if kind == 'mur1':
        return None
    if kind == 'mur2':
        return Mur2TE(dx, dt, sides=sides) if te else Mur2TM(dx, dt, sides=sides)
    if kind == 'cpml':
        return ((CPML_TE if te else CPML_TM)
                (Nx, Ny, dx, dt, npml=npml, npml_x=npml_x, npml_y=npml_y))
    raise ValueError(f"unbekannte boundary '{kind}' (mur1|mur2|cpml)")
