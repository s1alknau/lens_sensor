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

    def flux(self, key):
        """Netto-Fluss in +x (Einheiten kuerzen sich in Verhaeltnissen)."""
        n = self.n.get(key, 0)
        if not n:
            return float('nan')
        sc = (2.0/n)**2
        s = 0.0
        for sign, aE, aH in self.acc[key]:
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
