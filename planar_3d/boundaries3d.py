"""Absorbierende Raender fuer den 3D-FDTD-Solver (fdtd3d_core).

Auswahl ueber ``boundary``:
  'sponge' - gradierter Volumen-Absorber (Default, wie bisher)
  'mur1'   - Mur 1. Ordnung auf den 6 Aussenflaechen (tangentiales E)
  'mur2'   - Mur 2. Ordnung (transversaler Korrekturterm)   [Stufe 2]
  'cpml'   - Convolutional PML (ψ-Streifen, alle 6 Komponenten) [Stufe 3]

Yee-Layout (Zellgitter Nx x Ny x Nz):
  Ex(Nx-1, Ny,   Nz)    Ey(Nx,   Ny-1, Nz)    Ez(Nx,   Ny,   Nz-1)
  Hx(Nx,   Ny-1, Nz-1)  Hy(Nx-1, Ny,   Nz-1)  Hz(Nx-1, Ny-1, Nz)

Tangentiale E-Komponenten je Aussenflaeche (die der Innen-Update NICHT setzt und
die der Rand fuellt):
  x-Flaechen (xmin/xmax): Ey, Ez
  y-Flaechen (ymin/ymax): Ex, Ez
  z-Flaechen (zmin/zmax): Ex, Ey
"""
import numpy as np                                     # noqa: E402
from common.physics import C0                          # noqa: E402


def mur1_coeff(dx, dt):
    return (C0*dt - dx)/(C0*dt + dx)


# Tangentiale E-Felder pro Flaeche: (Feldname, Achse, Randindex, Innenindex)
# Achse ist die Flaechen-Normale (0=x,1=y,2=z); der Mur-Update laeuft entlang ihr.
_TANG = {
    'xmin': (('Ey', 0, 0, 1),  ('Ez', 0, 0, 1)),
    'xmax': (('Ey', 0, -1, -2), ('Ez', 0, -1, -2)),
    'ymin': (('Ex', 1, 0, 1),  ('Ez', 1, 0, 1)),
    'ymax': (('Ex', 1, -1, -2), ('Ez', 1, -1, -2)),
    'zmin': (('Ex', 2, 0, 1),  ('Ey', 2, 0, 1)),
    'zmax': (('Ex', 2, -1, -2), ('Ey', 2, -1, -2)),
}


def _plane(F, axis, idx):
    """Die 2D-Ebene von F bei Index idx entlang axis (0/1/2)."""
    if axis == 0:
        return F[idx]
    if axis == 1:
        return F[:, idx]
    return F[:, :, idx]


def _set_plane(F, axis, idx, val):
    if axis == 0:
        F[idx] = val
    elif axis == 1:
        F[:, idx] = val
    else:
        F[:, :, idx] = val


class Mur1_3D:
    """Mur 1. Ordnung auf den absorbierenden Aussenflaechen. Fuellt die
    tangentialen E-Aussenebenen (die der Innen-Update auslaesst) aus dem
    Feld eine Zelle innen + der Historie: F0^{n+1} = F1^n + m (F1^{n+1} - F0^n)."""
    def __init__(self, dx, dt, faces):
        self.m = mur1_coeff(dx, dt)
        self.faces = tuple(faces)
        self._prev = None                               # gespeicherte Ebenen (F0^n, F1^n)

    def _fields(self, Ex, Ey, Ez):
        return {'Ex': Ex, 'Ey': Ey, 'Ez': Ez}

    def capture(self, Ex, Ey, Ez):                      # VOR dem Yee-Schritt (E = E^n)
        F = self._fields(Ex, Ey, Ez)
        prev = {}
        for face in self.faces:
            for name, axis, i0, i1 in _TANG[face]:
                prev[(face, name)] = (_plane(F[name], axis, i0).copy(),
                                      _plane(F[name], axis, i1).copy())
        self._prev = prev

    def apply(self, Ex, Ey, Ez):                        # NACH dem Yee-Schritt (Innen = E^{n+1})
        if self._prev is None:
            return
        m = self.m
        F = self._fields(Ex, Ey, Ez)
        for face in self.faces:
            for name, axis, i0, i1 in _TANG[face]:
                f0_prev, f1_prev = self._prev[(face, name)]
                f1_new = _plane(F[name], axis, i1)      # eine Zelle innen, bereits n+1
                _set_plane(F[name], axis, i0, f1_prev + m*(f1_new - f0_prev))


def _mur2_coeffs(dx, dt):
    den = C0*dt + dx
    return ((C0*dt - dx)/den,          # ca
            2.0*dx/den,                # cb
            (C0*dt)**2/(2.0*dx*den),   # cc  (dy=dz=dx)
            (C0*dt - dx)/(C0*dt + dx)) # m (mur1, Border/Ecken)


def _lap2d(a):
    """Transversaler 2D-Laplace (x-Nachbarn + y-Nachbarn) auf dem Innenbereich."""
    return (a[2:, 1:-1] - 2*a[1:-1, 1:-1] + a[:-2, 1:-1]
            + a[1:-1, 2:] - 2*a[1:-1, 1:-1] + a[1:-1, :-2])


class Mur2_3D:
    """Mur 2. Ordnung auf den absorbierenden Aussenflaechen. Wie Mur-1, aber mit
    transversalem 2D-Laplace-Korrekturterm auf jeder Randebene (bessere Absorption
    bei schraegem Einfall). Ebenen-Raender/Ecken: Mur-1-Fallback. Braucht die
    Randebenen zu n (cur) UND n-1 (prev)."""
    def __init__(self, dx, dt, faces):
        self.ca, self.cb, self.cc, self.m = _mur2_coeffs(dx, dt)
        self.faces = tuple(faces)
        self._prev = None
        self._cur = None

    def _fields(self, Ex, Ey, Ez):
        return {'Ex': Ex, 'Ey': Ey, 'Ez': Ez}

    def reset(self):
        self._prev = None; self._cur = None

    def capture(self, Ex, Ey, Ez):                      # VOR dem Yee-Schritt (E=E^n)
        F = self._fields(Ex, Ey, Ez)
        self._prev = self._cur
        cur = {}
        for face in self.faces:
            for name, axis, i0, i1 in _TANG[face]:
                cur[(face, name)] = (_plane(F[name], axis, i0).copy(),
                                     _plane(F[name], axis, i1).copy())
        self._cur = cur

    def apply(self, Ex, Ey, Ez):                        # NACH dem Yee-Schritt
        if self._cur is None:
            return
        m = self.m
        F = self._fields(Ex, Ey, Ez)
        if self._prev is None:                          # erster Schritt: Mur-1
            for face in self.faces:
                for name, axis, i0, i1 in _TANG[face]:
                    cf0, cf1 = self._cur[(face, name)]
                    f1n = _plane(F[name], axis, i1)
                    _set_plane(F[name], axis, i0, cf1 + m*(f1n - cf0))
            return
        ca, cb, cc = self.ca, self.cb, self.cc
        for face in self.faces:
            for name, axis, i0, i1 in _TANG[face]:
                cf0, cf1 = self._cur[(face, name)]
                pf0, pf1 = self._prev[(face, name)]
                f1n = _plane(F[name], axis, i1)
                out = cf1 + m*(f1n - cf0)               # Mur-1 ueberall (Border)
                if cf0.shape[0] >= 3 and cf0.shape[1] >= 3:
                    s = (slice(1, -1), slice(1, -1))
                    out[s] = (-pf1[s] + ca*(f1n[s] + pf0[s]) + cb*(cf0[s] + cf1[s])
                              + cc*(_lap2d(cf0) + _lap2d(cf1)))
                _set_plane(F[name], axis, i0, out)


from common.physics import EPS0                            # noqa: E402
try:
    from common.backend import xp                          # noqa: E402
except Exception:                                          # pragma: no cover
    import numpy as xp


def _cpml_bc_1d(N, npml_lo, npml_hi, dx, dt, m=3, a_max=0.05, R0=1e-6):
    """b,c-Profile (kappa=1) an Ganz- (Laenge N) UND Halbknoten (N-1) fuer EINE
    Achse. npml_lo/hi = PML-Zellen am unteren/oberen Ende. Ausserhalb: b=1,c=0."""
    if npml_lo <= 0 and npml_hi <= 0:
        one_e = xp.ones(N, dtype=xp.float32); zero_e = xp.zeros(N, dtype=xp.float32)
        one_h = xp.ones(N-1, dtype=xp.float32); zero_h = xp.zeros(N-1, dtype=xp.float32)
        return (one_e, zero_e), (one_h, zero_h)
    eta0 = np.sqrt(4e-7*np.pi/EPS0)

    def smax(npml):
        return (-(m+1)*np.log(R0)/(2.0*eta0*npml*dx)) if npml > 0 else 0.0
    smx_lo, smx_hi = smax(npml_lo), smax(npml_hi)

    def build(pos):
        depth = np.zeros_like(pos); sg = np.zeros_like(pos)
        if npml_lo > 0:
            lo = pos < npml_lo
            depth[lo] = (npml_lo - pos[lo])/npml_lo; sg[lo] = smx_lo
        if npml_hi > 0:
            hi = pos > (N-1-npml_hi)
            depth[hi] = (pos[hi]-(N-1-npml_hi))/npml_hi; sg[hi] = smx_hi
        depth = np.clip(depth, 0, 1)
        sigma = sg*depth**m
        a = np.where(depth > 0, a_max*(1.0-depth), 0.0)
        b = np.exp(-(sigma + a)*dt/EPS0)
        denom = sigma + a
        c = np.zeros_like(sigma); nz = denom > 0
        c[nz] = sigma[nz]*(b[nz]-1.0)/denom[nz]
        return xp.asarray(b.astype(np.float32)), xp.asarray(c.astype(np.float32))

    be, ce = build(np.arange(N, dtype=np.float64))
    bh, ch = build(np.arange(N-1, dtype=np.float64) + 0.5)
    return (be, ce), (bh, ch)


def _npml_for_axis(faces, lo_name, hi_name, npml):
    return (npml if lo_name in faces else 0, npml if hi_name in faces else 0)


class CPML_3D:
    """Convolutional PML (Roden-Gedney, kappa=1) fuer den 3D-Yee-Schritt. Fuer jede
    der 6 Curl-Komponenten werden die zwei Ableitungsterme per rekursiver Faltung
    (psi = b*psi + c*dF) korrigiert. psi voll-domaenig (float32); ausserhalb der PML
    ist c=0 -> psi bleibt 0. Aeusserer E-Rand wird als PEC abgeschlossen.
    NB: 12 psi-Felder ~ speicherintensiv -> fuer kleine/mittlere 3D-Domaenen.
    (Streifen-Speicher fuer sehr grosse Fenster ist der naechste Ausbau.)"""
    def __init__(self, Nx, Ny, Nz, dx, dt, faces, npml=10, **kw):
        nxl, nxh = _npml_for_axis(faces, 'xmin', 'xmax', npml)
        nyl, nyh = _npml_for_axis(faces, 'ymin', 'ymax', npml)
        nzl, nzh = _npml_for_axis(faces, 'zmin', 'zmax', npml)
        self.px = nxl > 0 or nxh > 0
        self.py = nyl > 0 or nyh > 0
        self.pz = nzl > 0 or nzh > 0
        (self.bxe, self.cxe), (self.bxh, self.cxh) = _cpml_bc_1d(Nx, nxl, nxh, dx, dt, **kw)
        (self.bye, self.cye), (self.byh, self.cyh) = _cpml_bc_1d(Ny, nyl, nyh, dx, dt, **kw)
        (self.bze, self.cze), (self.bzh, self.czh) = _cpml_bc_1d(Nz, nzl, nzh, dx, dt, **kw)
        self.faces = tuple(faces)
        z = xp.zeros
        # H-Update-psi (an den H-Positionen)
        self.p_Hx_z = z((Nx, Ny-1, Nz-1), xp.float32); self.p_Hx_y = z((Nx, Ny-1, Nz-1), xp.float32)
        self.p_Hy_x = z((Nx-1, Ny, Nz-1), xp.float32); self.p_Hy_z = z((Nx-1, Ny, Nz-1), xp.float32)
        self.p_Hz_y = z((Nx-1, Ny-1, Nz), xp.float32); self.p_Hz_x = z((Nx-1, Ny-1, Nz), xp.float32)
        # E-Update-psi (an den E-Innenpositionen [1:-1])
        self.p_Ex_y = z((Nx-1, Ny-2, Nz-2), xp.float32); self.p_Ex_z = z((Nx-1, Ny-2, Nz-2), xp.float32)
        self.p_Ey_z = z((Nx-2, Ny-1, Nz-2), xp.float32); self.p_Ey_x = z((Nx-2, Ny-1, Nz-2), xp.float32)
        self.p_Ez_x = z((Nx-2, Ny-2, Nz-1), xp.float32); self.p_Ez_y = z((Nx-2, Ny-2, Nz-1), xp.float32)

    def step(self, Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, Ch):
        # ---- H-Update (nutzt E^n) ----
        dEy_dz = Ey[:, :, 1:] - Ey[:, :, :-1]      # -> Hx (Halb-z)
        dEz_dy = Ez[:, 1:, :] - Ez[:, :-1, :]      # -> Hx (Halb-y)
        if self.pz:
            self.p_Hx_z = self.bzh[None, None, :]*self.p_Hx_z + self.czh[None, None, :]*dEy_dz
        if self.py:
            self.p_Hx_y = self.byh[None, :, None]*self.p_Hx_y + self.cyh[None, :, None]*dEz_dy
        Hx += Ch*((dEy_dz + self.p_Hx_z) - (dEz_dy + self.p_Hx_y))
        dEz_dx = Ez[1:, :, :] - Ez[:-1, :, :]      # -> Hy (Halb-x)
        dEx_dz = Ex[:, :, 1:] - Ex[:, :, :-1]      # -> Hy (Halb-z)
        if self.px:
            self.p_Hy_x = self.bxh[:, None, None]*self.p_Hy_x + self.cxh[:, None, None]*dEz_dx
        if self.pz:
            self.p_Hy_z = self.bzh[None, None, :]*self.p_Hy_z + self.czh[None, None, :]*dEx_dz
        Hy += Ch*((dEz_dx + self.p_Hy_x) - (dEx_dz + self.p_Hy_z))
        dEx_dy = Ex[:, 1:, :] - Ex[:, :-1, :]      # -> Hz (Halb-y)
        dEy_dx = Ey[1:, :, :] - Ey[:-1, :, :]      # -> Hz (Halb-x)
        if self.py:
            self.p_Hz_y = self.byh[None, :, None]*self.p_Hz_y + self.cyh[None, :, None]*dEx_dy
        if self.px:
            self.p_Hz_x = self.bxh[:, None, None]*self.p_Hz_x + self.cxh[:, None, None]*dEy_dx
        Hz += Ch*((dEx_dy + self.p_Hz_y) - (dEy_dx + self.p_Hz_x))
        # ---- E-Update (nutzt H^{n+1/2}, Innenbereich) ----
        dHz_dy = Hz[:, 1:, 1:-1] - Hz[:, :-1, 1:-1]   # -> Ex (Ganz-y innen)
        dHy_dz = Hy[:, 1:-1, 1:] - Hy[:, 1:-1, :-1]   # -> Ex (Ganz-z innen)
        if self.py:
            self.p_Ex_y = self.bye[None, 1:-1, None]*self.p_Ex_y + self.cye[None, 1:-1, None]*dHz_dy
        if self.pz:
            self.p_Ex_z = self.bze[None, None, 1:-1]*self.p_Ex_z + self.cze[None, None, 1:-1]*dHy_dz
        Ex[:, 1:-1, 1:-1] += ce_x[:, 1:-1, 1:-1]*((dHz_dy + self.p_Ex_y) - (dHy_dz + self.p_Ex_z))
        dHx_dz = Hx[1:-1, :, 1:] - Hx[1:-1, :, :-1]   # -> Ey (Ganz-z innen)
        dHz_dx = Hz[1:, :, 1:-1] - Hz[:-1, :, 1:-1]   # -> Ey (Ganz-x innen)
        if self.pz:
            self.p_Ey_z = self.bze[None, None, 1:-1]*self.p_Ey_z + self.cze[None, None, 1:-1]*dHx_dz
        if self.px:
            self.p_Ey_x = self.bxe[1:-1, None, None]*self.p_Ey_x + self.cxe[1:-1, None, None]*dHz_dx
        Ey[1:-1, :, 1:-1] += ce_y[1:-1, :, 1:-1]*((dHx_dz + self.p_Ey_z) - (dHz_dx + self.p_Ey_x))
        dHy_dx = Hy[1:, 1:-1, :] - Hy[:-1, 1:-1, :]   # -> Ez (Ganz-x innen)
        dHx_dy = Hx[1:-1, 1:, :] - Hx[1:-1, :-1, :]   # -> Ez (Ganz-y innen)
        if self.px:
            self.p_Ez_x = self.bxe[1:-1, None, None]*self.p_Ez_x + self.cxe[1:-1, None, None]*dHy_dx
        if self.py:
            self.p_Ez_y = self.bye[None, 1:-1, None]*self.p_Ez_y + self.cye[None, 1:-1, None]*dHx_dy
        Ez[1:-1, 1:-1, :] += ce_z[1:-1, 1:-1, :]*((dHy_dx + self.p_Ez_x) - (dHx_dy + self.p_Ez_y))
        # ---- PEC-Abschluss (tangentiales E=0) auf den PML-Aussenflaechen ----
        if 'xmin' in self.faces: Ey[0] = 0; Ez[0] = 0
        if 'xmax' in self.faces: Ey[-1] = 0; Ez[-1] = 0
        if 'ymin' in self.faces: Ex[:, 0] = 0; Ez[:, 0] = 0
        if 'ymax' in self.faces: Ex[:, -1] = 0; Ez[:, -1] = 0
        if 'zmin' in self.faces: Ex[:, :, 0] = 0; Ey[:, :, 0] = 0
        if 'zmax' in self.faces: Ex[:, :, -1] = 0; Ey[:, :, -1] = 0


def make_boundary_3d(kind, dx, dt, faces, shape=None, npml=10):
    """Rand-Objekt (oder None fuer 'sponge'). kind in {'sponge','mur1','mur2','cpml'};
    faces = absorbierende Flaechen. shape=(Nx,Ny,Nz) nur fuer cpml noetig."""
    kind = (kind or 'sponge').lower()
    if kind == 'sponge':
        return None
    if kind == 'mur1':
        return Mur1_3D(dx, dt, faces)
    if kind == 'mur2':
        return Mur2_3D(dx, dt, faces)
    if kind == 'cpml':
        if shape is None:
            raise ValueError('cpml (3D) braucht shape=(Nx,Ny,Nz)')
        return CPML_3D(shape[0], shape[1], shape[2], dx, dt, faces, npml=npml)
    raise ValueError(f"unbekannte boundary '{kind}' (sponge|mur1|mur2|cpml)")
