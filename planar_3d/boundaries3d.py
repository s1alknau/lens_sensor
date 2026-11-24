"""Absorbierende Raender fuer den 3D-FDTD-Solver (fdtd3d_core).

Auswahl ueber ``boundary``:
  'cpml'   - Convolutional PML (Default; psi-Streifen, alle 6 Komponenten), mediumunabhaengig
             (Benchmark tests/boundary_reflection_3d.py: ~-78 dB gegen ~-16 dB Sponge)
  'sponge' - gradierter Volumen-Absorber, Dicke fest 1.2 um
  'mur1'   - Mur 1. Ordnung (nur CLI): lokale Geschwindigkeit je Flaeche
             (x: c0/n_eff, y/z: c0/n der Randzellen; mur_speeds_3d)
  (mur2 entfernt)

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


def mur1_coeff(dx, dt, n=1.0):
    """Mur-1-Koeffizient bei Phasengeschwindigkeit c0/n (n Skalar oder Array)."""
    c = C0/np.asarray(n, dtype=np.float64)
    m = (c*dt - dx)/(c*dt + dx)
    return float(m) if np.ndim(m) == 0 else m


def mur_speeds_3d(eps, dx, lam_m):
    """Lokale Brechzahlen je Aussenflaeche aus eps (Nx, Ny, Nz), Host-Array.
    x-Flaechen: n_eff der gefuehrten Grundmode des y-Profils (Mitte z) - die Welle
    laeuft dort mit c0/n_eff hinaus. y/z-Flaechen: lokales n der Randzellen."""
    from planar_beads.boundaries import guided_neff
    e = eps.get() if hasattr(eps, 'get') else np.asarray(eps)
    kz = e.shape[2]//2
    return {'xmin': guided_neff(e[1, :, kz], dx, lam_m),
            'xmax': guided_neff(e[-2, :, kz], dx, lam_m),
            'ymin': np.sqrt(e[:, 0, :].astype(np.float64)),
            'ymax': np.sqrt(e[:, -1, :].astype(np.float64)),
            'zmin': np.sqrt(e[:, :, 0].astype(np.float64)),
            'zmax': np.sqrt(e[:, :, -1].astype(np.float64))}


def _face_n(speeds, face, shape):
    """n fuer die Randebene einer Feldkomponente (Array auf deren Ebenenform zugeschnitten;
    der halbe Zellversatz der Yee-Komponenten wird vernachlaessigt)."""
    n = (speeds or {}).get(face, 1.0)
    if np.ndim(n) == 0:
        return float(n)
    n = np.asarray(n)
    a, b = shape
    out = np.empty((a, b))
    out[:min(a, n.shape[0]), :min(b, n.shape[1])] = n[:a, :b]
    if a > n.shape[0]:
        out[n.shape[0]:, :] = out[n.shape[0]-1:n.shape[0], :]
    if b > n.shape[1]:
        out[:, n.shape[1]:] = out[:, n.shape[1]-1:n.shape[1]]
    return out


def _dev(v):
    """Skalar bleibt float; Array -> float32 auf dem Rechen-Backend (GPU/CPU)."""
    return v if np.ndim(v) == 0 else xp.asarray(np.asarray(v, dtype=np.float32))


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
    def __init__(self, dx, dt, faces, speeds=None):
        self.dx, self.dt = dx, dt
        self.faces = tuple(faces)
        self._prev = None                               # gespeicherte Ebenen (F0^n, F1^n)
        self.set_speeds(speeds)

    def set_speeds(self, speeds):
        """Lokale Brechzahlen je Flaeche (mur_speeds_3d); None = Vakuum c0."""
        self.speeds = speeds; self._mk = {}

    def _m(self, face, name, shape):
        k = (face, name)
        if k not in self._mk:
            self._mk[k] = _dev(mur1_coeff(self.dx, self.dt, _face_n(self.speeds, face, shape)))
        return self._mk[k]

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
        F = self._fields(Ex, Ey, Ez)
        for face in self.faces:
            for name, axis, i0, i1 in _TANG[face]:
                f0_prev, f1_prev = self._prev[(face, name)]
                f1_new = _plane(F[name], axis, i1)      # eine Zelle innen, bereits n+1
                m = self._m(face, name, f1_new.shape)
                _set_plane(F[name], axis, i0, f1_prev + m*(f1_new - f0_prev))


from common.physics import EPS0, C0                            # noqa: E402
try:
    from common.backend import xp                          # noqa: E402
except Exception:                                          # pragma: no cover
    import numpy as xp


def _cpml_bc_1d(N, npml_lo, npml_hi, dx, dt, m=3, a_max=None, R0=1e-6, kappa_max=1.0,
                lam_m=None):
    """b, c, kappa-Profile an Ganz- (Laenge N) UND Halbknoten (N-1) fuer EINE Achse.
    npml_lo/hi = PML-Zellen am unteren/oberen Ende. Ausserhalb: b=1, c=0, kappa=1.
    kappa = 1 + (kappa_max-1)*Tiefe^m (Koordinatenstreckung). Standard kappa_max = 1:
    gemessen (tests/cpml_compare.py, 10 PML-Zellen) reflektiert kappa 7 bzw. 3 deutlich
    mehr (zu steile Streckung ueber wenige Zellen); CFS (a) hilft bei streifendem Einfall.
    CFS-Term a (complex frequency shifted, Roden-Gedney): a_max = 2*pi*eps0*f0/10
    (f0 = c0/lam). Der fruehere Wert 0.05 S/m war gegen sigma_max ~ 4e5 S/m praktisch
    null -> kein CFS. Mit a ~ 2e3 S/m werden evaneszente und streifend einfallende
    Anteile besser absorbiert, die laufende Welle (omega*eps0 ~ 2e4 S/m) kaum anders."""
    if npml_lo <= 0 and npml_hi <= 0:
        one_e = xp.ones(N, dtype=xp.float32); zero_e = xp.zeros(N, dtype=xp.float32)
        one_h = xp.ones(N-1, dtype=xp.float32); zero_h = xp.zeros(N-1, dtype=xp.float32)
        return (one_e, zero_e, one_e), (one_h, zero_h, one_h)
    if a_max is None:
        a_max = 2.0*np.pi*EPS0*(C0/(lam_m or 850e-9))/10.0
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
        kappa = 1.0 + (kappa_max - 1.0)*depth**m
        a = np.where(depth > 0, a_max*(1.0-depth), 0.0)
        b = np.exp(-(sigma/kappa + a)*dt/EPS0)
        denom = sigma*kappa + kappa*kappa*a
        c = np.zeros_like(sigma); nz = denom > 0
        c[nz] = sigma[nz]*(b[nz]-1.0)/denom[nz]
        return (xp.asarray(b.astype(np.float32)), xp.asarray(c.astype(np.float32)),
                xp.asarray(kappa.astype(np.float32)))

    be, ce, ke = build(np.arange(N, dtype=np.float64))
    bh, ch, kh = build(np.arange(N-1, dtype=np.float64) + 0.5)
    return (be, ce, ke), (bh, ch, kh)


def _npml_for_axis(faces, lo_name, hi_name, npml):
    return (npml if lo_name in faces else 0, npml if hi_name in faces else 0)


class _PsiStrip:
    """psi einer Streckungsrichtung - gespeichert NUR in den PML-Randstreifen.

    Ausserhalb der PML ist c=0, damit bleibt psi dort exakt 0 und traegt nichts
    bei. Statt eines vollen Volumens werden deshalb nur die beiden Randstreifen
    gehalten (Laenge = Zahl der PML-Zellen dieser Achse). Das spart bei npml=10
    und 500x650x200 rund 94% des psi-Speichers (3.1 GB -> ~0.2 GB).

    Die Streifen werden aus dem c-Profil selbst bestimmt (c!=0), damit auch
    geschnittene Profile (z.B. bye[1:-1] fuer die E-Updates) korrekt passen."""

    def __init__(self, shape, axis, b, c, k=None):
        self.axis = axis
        self.parts = []
        N = shape[axis]
        cn = np.abs(np.asarray(to_np_1d(c), dtype=np.float64))
        nlo = 0
        while nlo < N and cn[nlo] > 0.0:
            nlo += 1
        nhi = 0
        while nhi < N and cn[N-1-nhi] > 0.0:
            nhi += 1
        if nlo + nhi >= N:                     # PML fuellt die Achse -> voll halten
            nlo, nhi = N, 0
        shp = lambda n: tuple(n if k == axis else shape[k] for k in range(3))
        bc = lambda v: v.reshape(tuple(-1 if k == axis else 1 for k in range(3)))
        if nlo > 0:
            sl = tuple(slice(0, nlo) if k == axis else slice(None) for k in range(3))
            self.parts.append([sl, bc(b[:nlo]), bc(c[:nlo]),
                               xp.zeros(shp(nlo), dtype=xp.float32),
                               (bc(1.0/k[:nlo] - 1.0) if k is not None else None)])
        if nhi > 0:
            sl = tuple(slice(N-nhi, N) if k == axis else slice(None) for k in range(3))
            self.parts.append([sl, bc(b[N-nhi:]), bc(c[N-nhi:]),
                               xp.zeros(shp(nhi), dtype=xp.float32),
                               (bc(1.0/k[N-nhi:] - 1.0) if k is not None else None)])
        self.active = bool(self.parts)
        self.nbytes = sum(p[3].nbytes for p in self.parts)

    def add_into(self, F, dF, coef, sign):
        """psi = b*psi + c*dF (nur Streifen), dann sign*coef*(psi + (1/kappa-1)*dF)
        auf F addieren. Der Basis-Yee-Schritt hat schon coef*dF addiert; die
        kappa-Korrektur macht daraus coef*dF/kappa (ausserhalb kappa=1 -> 0)."""
        for sl, b, c, p, km1 in self.parts:
            p *= b
            p += c*dF[sl]
            add = p if km1 is None else (p + km1*dF[sl])
            if np.isscalar(coef) or getattr(coef, 'ndim', 0) == 0:
                F[sl] += (sign*coef)*add
            else:
                F[sl] += (sign*coef[sl])*add


def to_np_1d(a):
    """1D-Profil als numpy (fuer die Streifen-Erkennung; klein, egal ob GPU)."""
    try:
        return a.get()                                   # CuPy
    except AttributeError:
        return np.asarray(a)


class CPML_3D:
    """Convolutional PML (Roden-Gedney, CFS: kappa_max=1, a aus der Wellenlaenge) fuer den 3D-Yee-Schritt.

    Fuer jede der 6 Curl-Komponenten werden die zwei Ableitungsterme per
    rekursiver Faltung (psi = b*psi + c*dF) korrigiert. Die psi-Felder liegen
    NUR in den PML-Randstreifen (_PsiStrip) - ausserhalb ist c=0, psi also
    ohnehin 0. Damit ist CPML auch fuer grosse Fenster speichertauglich.
    Aeusserer E-Rand wird als PEC abgeschlossen."""
    def __init__(self, Nx, Ny, Nz, dx, dt, faces, npml=10, **kw):
        nxl, nxh = _npml_for_axis(faces, 'xmin', 'xmax', npml)
        nyl, nyh = _npml_for_axis(faces, 'ymin', 'ymax', npml)
        nzl, nzh = _npml_for_axis(faces, 'zmin', 'zmax', npml)
        self.px = nxl > 0 or nxh > 0
        self.py = nyl > 0 or nyh > 0
        self.pz = nzl > 0 or nzh > 0
        (self.bxe, self.cxe, self.kxe), (self.bxh, self.cxh, self.kxh) = \
            _cpml_bc_1d(Nx, nxl, nxh, dx, dt, **kw)
        (self.bye, self.cye, self.kye), (self.byh, self.cyh, self.kyh) = \
            _cpml_bc_1d(Ny, nyl, nyh, dx, dt, **kw)
        (self.bze, self.cze, self.kze), (self.bzh, self.czh, self.kzh) = \
            _cpml_bc_1d(Nz, nzl, nzh, dx, dt, **kw)
        self.faces = tuple(faces)
        P = _PsiStrip
        # H-Update-psi (an den H-Positionen; Halbknoten-Profile)
        sH_x = (Nx, Ny-1, Nz-1); sH_y = (Nx-1, Ny, Nz-1); sH_z = (Nx-1, Ny-1, Nz)
        self.p_Hx_z = P(sH_x, 2, self.bzh, self.czh, self.kzh)
        self.p_Hx_y = P(sH_x, 1, self.byh, self.cyh, self.kyh)
        self.p_Hy_x = P(sH_y, 0, self.bxh, self.cxh, self.kxh)
        self.p_Hy_z = P(sH_y, 2, self.bzh, self.czh, self.kzh)
        self.p_Hz_y = P(sH_z, 1, self.byh, self.cyh, self.kyh)
        self.p_Hz_x = P(sH_z, 0, self.bxh, self.cxh, self.kxh)
        # E-Update-psi (E-Innenpositionen [1:-1]; Ganzknoten-Profile, geschnitten)
        sE_x = (Nx-1, Ny-2, Nz-2); sE_y = (Nx-2, Ny-1, Nz-2); sE_z = (Nx-2, Ny-2, Nz-1)
        self.p_Ex_y = P(sE_x, 1, self.bye[1:-1], self.cye[1:-1], self.kye[1:-1])
        self.p_Ex_z = P(sE_x, 2, self.bze[1:-1], self.cze[1:-1], self.kze[1:-1])
        self.p_Ey_z = P(sE_y, 2, self.bze[1:-1], self.cze[1:-1], self.kze[1:-1])
        self.p_Ey_x = P(sE_y, 0, self.bxe[1:-1], self.cxe[1:-1], self.kxe[1:-1])
        self.p_Ez_x = P(sE_z, 0, self.bxe[1:-1], self.cxe[1:-1], self.kxe[1:-1])
        self.p_Ez_y = P(sE_z, 1, self.bye[1:-1], self.cye[1:-1], self.kye[1:-1])
        self.psi_bytes = sum(getattr(self, n).nbytes for n in
                             ('p_Hx_z', 'p_Hx_y', 'p_Hy_x', 'p_Hy_z', 'p_Hz_y', 'p_Hz_x',
                              'p_Ex_y', 'p_Ex_z', 'p_Ey_z', 'p_Ey_x', 'p_Ez_x', 'p_Ez_y'))

    def step(self, Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, Ch, da=None):
        """Voller Yee-Schritt mit CPML (H, dann E). Getrennt aufrufbar als step_H/step_E,
        damit z.B. die TF/SF-Einkopplung des Zeitbereichs-Stitch dazwischen greifen kann."""
        self.step_H(Ex, Ey, Ez, Hx, Hy, Hz, Ch)
        self.step_E(Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, da=da)

    def step_H(self, Ex, Ey, Ez, Hx, Hy, Hz, Ch):
        # Basis-Yee-Update ohne psi, danach die psi-Korrekturen NUR in den
        # PML-Streifen aufaddieren (mathematisch identisch zur vollen Variante:
        # F += Ch*((dF1 + psi1) - (dF2 + psi2)) == F += Ch*(dF1-dF2); +Ch*psi1; -Ch*psi2)
        # ---- H-Update (nutzt E^n) ----
        dEy_dz = Ey[:, :, 1:] - Ey[:, :, :-1]      # -> Hx (Halb-z)
        dEz_dy = Ez[:, 1:, :] - Ez[:, :-1, :]      # -> Hx (Halb-y)
        Hx += Ch*(dEy_dz - dEz_dy)
        self.p_Hx_z.add_into(Hx, dEy_dz, Ch, +1)
        self.p_Hx_y.add_into(Hx, dEz_dy, Ch, -1)
        dEz_dx = Ez[1:, :, :] - Ez[:-1, :, :]      # -> Hy (Halb-x)
        dEx_dz = Ex[:, :, 1:] - Ex[:, :, :-1]      # -> Hy (Halb-z)
        Hy += Ch*(dEz_dx - dEx_dz)
        self.p_Hy_x.add_into(Hy, dEz_dx, Ch, +1)
        self.p_Hy_z.add_into(Hy, dEx_dz, Ch, -1)
        dEx_dy = Ex[:, 1:, :] - Ex[:, :-1, :]      # -> Hz (Halb-y)
        dEy_dx = Ey[1:, :, :] - Ey[:-1, :, :]      # -> Hz (Halb-x)
        Hz += Ch*(dEx_dy - dEy_dx)
        self.p_Hz_y.add_into(Hz, dEx_dy, Ch, +1)
        self.p_Hz_x.add_into(Hz, dEy_dx, Ch, -1)

    def step_E(self, Ex, Ey, Ez, Hx, Hy, Hz, ce_x, ce_y, ce_z, da=None):
        if da is not None:                         # Absorption: E <- D E vor dem E-Update
            Ex *= da[0]                            # da = (D_x, D_y, D_z), versetzt
            Ey *= da[1]                            # (fdtd3d_core._stagger_da)
            Ez *= da[2]
        # ---- E-Update (nutzt H^{n+1/2}, Innenbereich) ----
        dHz_dy = Hz[:, 1:, 1:-1] - Hz[:, :-1, 1:-1]   # -> Ex (Ganz-y innen)
        dHy_dz = Hy[:, 1:-1, 1:] - Hy[:, 1:-1, :-1]   # -> Ex (Ganz-z innen)
        Exv = Ex[:, 1:-1, 1:-1]; cev = ce_x[:, 1:-1, 1:-1]
        Exv += cev*(dHz_dy - dHy_dz)
        self.p_Ex_y.add_into(Exv, dHz_dy, cev, +1)
        self.p_Ex_z.add_into(Exv, dHy_dz, cev, -1)
        dHx_dz = Hx[1:-1, :, 1:] - Hx[1:-1, :, :-1]   # -> Ey (Ganz-z innen)
        dHz_dx = Hz[1:, :, 1:-1] - Hz[:-1, :, 1:-1]   # -> Ey (Ganz-x innen)
        Eyv = Ey[1:-1, :, 1:-1]; cev = ce_y[1:-1, :, 1:-1]
        Eyv += cev*(dHx_dz - dHz_dx)
        self.p_Ey_z.add_into(Eyv, dHx_dz, cev, +1)
        self.p_Ey_x.add_into(Eyv, dHz_dx, cev, -1)
        dHy_dx = Hy[1:, 1:-1, :] - Hy[:-1, 1:-1, :]   # -> Ez (Ganz-x innen)
        dHx_dy = Hx[1:-1, 1:, :] - Hx[1:-1, :-1, :]   # -> Ez (Ganz-y innen)
        Ezv = Ez[1:-1, 1:-1, :]; cev = ce_z[1:-1, 1:-1, :]
        Ezv += cev*(dHy_dx - dHx_dy)
        self.p_Ez_x.add_into(Ezv, dHy_dx, cev, +1)
        self.p_Ez_y.add_into(Ezv, dHx_dy, cev, -1)
        # ---- PEC-Abschluss (tangentiales E=0) auf den PML-Aussenflaechen ----
        if 'xmin' in self.faces: Ey[0] = 0; Ez[0] = 0
        if 'xmax' in self.faces: Ey[-1] = 0; Ez[-1] = 0
        if 'ymin' in self.faces: Ex[:, 0] = 0; Ez[:, 0] = 0
        if 'ymax' in self.faces: Ex[:, -1] = 0; Ez[:, -1] = 0
        if 'zmin' in self.faces: Ex[:, :, 0] = 0; Ey[:, :, 0] = 0
        if 'zmax' in self.faces: Ex[:, :, -1] = 0; Ey[:, :, -1] = 0


def make_boundary_3d(kind, dx, dt, faces, shape=None, npml=10, speeds=None, lam_m=None):
    """Rand-Objekt (oder None fuer 'sponge'). kind in {'sponge','mur1','cpml'};
    faces = absorbierende Flaechen. shape=(Nx,Ny,Nz) nur fuer cpml noetig."""
    kind = (kind or 'cpml').lower()
    if kind == 'sponge':
        return None
    if kind == 'mur1':
        return Mur1_3D(dx, dt, faces, speeds=speeds)
    if kind == 'cpml':
        if shape is None:
            raise ValueError('cpml (3D) requires shape=(Nx,Ny,Nz)')
        return CPML_3D(shape[0], shape[1], shape[2], dx, dt, faces, npml=npml, lam_m=lam_m)
    raise ValueError(f"unknown boundary '{kind}' (sponge|cpml|mur1; mur2 removed -> cpml)")
