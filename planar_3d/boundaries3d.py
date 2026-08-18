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


def make_boundary_3d(kind, dx, dt, faces):
    """Rand-Objekt (oder None fuer 'sponge'). kind in {'sponge','mur1','mur2','cpml'};
    faces = absorbierende Flaechen."""
    kind = (kind or 'sponge').lower()
    if kind == 'sponge':
        return None
    if kind == 'mur1':
        return Mur1_3D(dx, dt, faces)
    if kind == 'mur2':
        return Mur2_3D(dx, dt, faces)
    raise ValueError(f"boundary '{kind}' fuer 3D noch nicht verfuegbar (Stufe 3: cpml)")
