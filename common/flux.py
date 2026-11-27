"""Exakter CW-Poynting-Fluss durch x-Ebenen aus E UND H (laufende DFT bei f0).

    S_x = 1/2 Re( sum_k  s_k * E_k * conj(H_k) )   ueber den ganzen Querschnitt
          2D TE: -Ez Hy*     2D TM: +Ey Hz*     3D: +Ey Hz* - Ez Hy*

Frueher wurde S_x aus E allein geschaetzt (Faraday, ~Im(E dE*/dx)); fuer TM ist das
mit eps(y) gewichtet nur fuer EINE gefuehrte Mode exakt (Mehrmoden/Streulicht ->
Fehler). Hier wird H direkt mitgeschrieben.

Zeitlage: im Solver laeuft je Schritt H-Update (-> H^{n+1/2}), E-Update (-> E^{n+1}),
dann die Akkumulation mit t = n*dt. Damit E und H dieselbe Phasenreferenz haben, wird
H mit exp(-i w (t - dt/2)) akkumuliert. H (bei i+-1/2) wird auf die E-Ebene gemittelt.
"""
import numpy as np

from common.backend import xp


class PlaneFluxDFT:
    def __init__(self, omega, dt):
        self.omega = float(omega)
        self.dt = float(dt)
        self.acc = {}          # key -> list of [sign, accE, accH]
        self.n = {}

    def reset(self):
        self.acc = {}
        self.n = {}

    def add(self, key, t, pairs):
        """pairs: Folge von (sign, E_plane, H_lo, H_hi); H = (H_lo + H_hi)/2."""
        pe = np.complex64(np.exp(-1j*self.omega*t))
        ph = np.complex64(np.exp(-1j*self.omega*(t - 0.5*self.dt)))
        lst = self.acc.get(key)
        if lst is None:
            lst = [[s, xp.zeros(E.shape, dtype=xp.complex64),
                    xp.zeros(E.shape, dtype=xp.complex64)] for s, E, _, _ in pairs]
            self.acc[key] = lst
            self.n[key] = 0
        for a, (s, E, H0, H1) in zip(lst, pairs):
            a[1] += E.astype(xp.complex64)*pe
            a[2] += (0.5*(H0 + H1)).astype(xp.complex64)*ph
        self.n[key] += 1

    def has(self, key):
        return self.n.get(key, 0) > 0

    def flux(self, key, region=None):
        """Netto-Fluss in +x (Einheiten kuerzen sich in Verhaeltnissen).
        region: optionales Tupel von Slices auf den Querschnitt (2D: (y,), 3D: (y, z)),
        z.B. nur die Kern-Zeilen -> Leistung im Kern."""
        n = self.n.get(key, 0)
        if not n:
            return float('nan')
        sc = (2.0/n)**2
        s = 0.0
        for sign, aE, aH in self.acc[key]:
            if region is not None:
                aE, aH = aE[tuple(region)], aH[tuple(region)]
            s += sign*0.5*float(xp.real(xp.sum(aE*xp.conj(aH), dtype=xp.complex128)))
        return s*sc


SRC_MARGIN = 5          # Zellen zwischen Quellebene und Eingangs-Detektorebene


def guard_in_plane(i_in, src_ix, n_max, margin=SRC_MARGIN):
    """Eingangs-Detektorebene mindestens `margin` Zellen HINTER die Quelle (stromab)
    legen. Liegt sie davor (kurze Gebiete: 10 %/20 % der Laenge < Quellposition),
    misst sie die rueckwaerts abgestrahlte Haelfte -> negativer Fluss, T < 0.
    Rueckgabe: neuer Index (unveraendert, wenn schon weit genug stromab)."""
    lo = int(src_ix) + int(margin)
    if i_in >= lo:
        return int(i_in)
    new = min(lo, int(n_max) - 2)
    print(f'  [Detector] input plane was at/before the source (ix={i_in} <= source {src_ix}) '
          f'-> moved to ix={new}')
    return new


def pairs_2d(pol, il, Ez=None, Hy=None, Ey=None, Hz=None):
    """Feldpaare an der Ebene il (lokaler Index, il >= 1) fuer 2D."""
    if pol == 's':
        return ((-1.0, Ez[il], Hy[il - 1], Hy[il]),)
    return ((+1.0, Ey[il], Hz[il - 1], Hz[il]),)


def pairs_3d(il, Ey, Ez, Hy, Hz):
    """Feldpaare an der Ebene il fuer 3D (volle tangentiale Komponenten)."""
    return ((+1.0, Ey[il], Hz[il - 1], Hz[il]), (-1.0, Ez[il], Hy[il - 1], Hy[il]))


class SideBox:
    """Geschlossene Messbox fuer die Energiebilanz der VCSEL-Laeufe.

    x: von x_a (Ebene hinter der Quelle, 'src_dn') bis x_b (Ausgangsebene 'out'),
    y: j_lo..j_hi, 3D zusaetzlich z: k_lo..k_hi - jeweils knapp VOR dem Absorber.
    Die Seitenflaechen (y unten/oben, z vorne/hinten) messen das seitlich aus dem Gebiet
    laufende Licht (Streuung, Abstrahlung) als Poynting-Fluss nach AUSSEN, exakt aus E und
    H (Yee-Lagen: Ex(i+1/2,j,k) Ey(i,j+1/2,k) Ez(i,j,k+1/2) Hx(i,j+1/2,k+1/2)
    Hy(i+1/2,j,k+1/2) Hz(i+1/2,j+1/2,k); H wird auf die E-Flaeche gemittelt).
        S_y = Ez Hx - Ex Hz,   S_z = Ex Hy - Ey Hx
    Bei Stitch traegt jedes Fenster nur seinen assemblierten x-Bereich bei (eigene Keys
    'side_w<k>_<face>'), sodass nichts doppelt zaehlt."""

    def __init__(self, x_a, x_b, j_lo, j_hi, k_lo=None, k_hi=None):
        self.x_a, self.x_b = int(x_a), int(x_b)
        self.j_lo, self.j_hi = int(j_lo), int(j_hi)
        self.k_lo = None if k_lo is None else int(k_lo)
        self.k_hi = None if k_hi is None else int(k_hi)

    @property
    def region(self):
        """Querschnitts-Region der x-Ebenen (gleiche y/z-Grenzen wie die Seiten)."""
        if self.k_lo is None:
            return (slice(self.j_lo, self.j_hi),)
        return (slice(self.j_lo, self.j_hi), slice(self.k_lo, self.k_hi))

    def _xs(self, glo, ghi, gx0):
        a = max(self.x_a, glo); b = min(self.x_b, ghi if ghi is not None else self.x_b)
        if b <= a:
            return None
        return slice(a - gx0, b - gx0)

    def faces_3d(self, Ex, Ey, Ez, Hx, Hy, Hz, glo=0, ghi=None, gx0=0, tag='w0'):
        """[(key, pairs)] fuer die vier Seitenflaechen (Vorzeichen: nach aussen positiv)."""
        xs = self._xs(glo, ghi, gx0)
        if xs is None:
            return []
        j0, j1, k0, k1 = self.j_lo, self.j_hi, self.k_lo, self.k_hi
        xs1 = slice(xs.start, min(xs.stop, Ex.shape[0]))    # Ex/Hy/Hz: Nx-1 in x
        out = []
        for name, j, s in (('ylo', j0, -1.0), ('yhi', j1, +1.0)):
            out.append((f'side_{tag}_{name}', (
                (+s, Ez[xs, j, k0:k1], Hx[xs, j - 1, k0:k1], Hx[xs, j, k0:k1]),
                (-s, Ex[xs1, j, k0:k1], Hz[xs1, j - 1, k0:k1], Hz[xs1, j, k0:k1]))))
        for name, k, s in (('zlo', k0, -1.0), ('zhi', k1, +1.0)):
            out.append((f'side_{tag}_{name}', (
                (+s, Ex[xs1, j0:j1, k], Hy[xs1, j0:j1, k - 1], Hy[xs1, j0:j1, k]),
                (-s, Ey[xs, j0:j1, k], Hx[xs, j0:j1, k - 1], Hx[xs, j0:j1, k]))))
        return out

    def faces_2d(self, pol, Ez=None, Hx=None, Ex=None, Hz=None, glo=0, ghi=None, gx0=0,
                 tag='w0'):
        """2D: nur die y-Flaechen. TE: S_y = Ez Hx, TM: S_y = -Ex Hz."""
        xs = self._xs(glo, ghi, gx0)
        if xs is None:
            return []
        out = []
        for name, j, s in (('ylo', self.j_lo, -1.0), ('yhi', self.j_hi, +1.0)):
            if pol == 's':
                pr = ((+s, Ez[xs, j], Hx[xs, j - 1], Hx[xs, j]),)
            else:
                xs1 = slice(xs.start, min(xs.stop, Ex.shape[0]))
                pr = ((-s, Ex[xs1, j], Hz[xs1, j - 1], Hz[xs1, j]),)
            out.append((f'side_{tag}_{name}', pr))
        return out

    @staticmethod
    def side_total(flux):
        """Summe aller Seitenfluesse (nach aussen) [Gitter-Einheiten]."""
        return sum(flux.flux(k) for k in flux.acc if k.startswith('side_') and flux.has(k))
