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


def make_boundary_3d(kind, dx, dt, faces):
    """Rand-Objekt (oder None fuer 'sponge'/'mur1'-ohne-Objekt-Faelle).
    kind in {'sponge','mur1','mur2','cpml'}. faces = absorbierende Flaechen."""
    kind = (kind or 'sponge').lower()
    if kind in ('sponge', 'mur1'):
        return Mur1_3D(dx, dt, faces) if kind == 'mur1' else None
    raise ValueError(f"boundary '{kind}' fuer 3D noch nicht verfuegbar "
                     f"(Stufe 2/3: mur2, cpml)")
