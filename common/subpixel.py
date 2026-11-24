"""Subpixel-Glaettung fuer Kugel-/Kreis-Einschluesse (Beads) im Yee-Gitter.

Problem (Review F12): Eine harte Maske (Zelle innen/aussen) macht die Bead-Oberflaeche
treppenfoermig und laesst den Abstand Bead-Grenzflaeche auf das Gitter einrasten
(Fehler bis dx/2 -> evaneszente Intensitaet am Bead um exp(2*dx/2/delta) falsch,
bei dx 20 nm ~10 %). Hier bekommt jede Zelle, die die Oberflaeche schneidet, den
VOLUMENANTEIL f der Kugel (Supersampling) und

    eps = (1 - f)*eps_umgebung + f*n_bead^2       (arithmetisches Mittel)

Damit ist die Bead-Position kontinuierlich (Verschieben um 0.1 dx aendert eps
stetig) und die Treppe wird zu einem glatten Uebergang. Das ist die skalare
Variante; die anisotrope (Tensor-)Mittelung nach Farjadpour et al., Opt. Lett. 31,
2972 (2006) waere fuer die Feldkomponente senkrecht zur Oberflaeche noch genauer.
Nur Randzellen werden abgetastet (Abstand der Zellmitte zur Oberflaeche < halbe
Zelldiagonale), alles andere ist exakt 0 oder 1.
"""
import numpy as np


def _offsets(S, h):
    return ((np.arange(S) + 0.5)/S - 0.5)*h


def circle_fraction(xs, ys, xc, yc, r, h, S=8):
    """Flaechenanteil eines Kreises (Mitte xc,yc, Radius r) in den Zellen um die
    Knoten xs (Nx) x ys (Ny), Zellgroesse h. Rueckgabe (Nx, Ny) in [0, 1]."""
    X = xs[:, None] - xc
    Y = ys[None, :] - yc
    d = np.sqrt(X*X + Y*Y)
    half = h*np.sqrt(2.0)/2.0
    f = (d <= r - half).astype(np.float64)
    edge = np.nonzero((d > r - half) & (d < r + half))
    if len(edge[0]):
        o = _offsets(S, h)
        px = X[edge[0], 0][:, None, None] + o[None, :, None]
        py = Y[0, edge[1]][:, None, None] + o[None, None, :]
        f[edge] = np.mean((px*px + py*py) <= r*r, axis=(1, 2))
    return f


def sphere_fraction(xs, ys, zs, xc, yc, zc, r, h, S=6):
    """Volumenanteil einer Kugel in den Zellen um die Knoten xs x ys x zs (Zellgroesse h).
    Rueckgabe (Nx, Ny, Nz) in [0, 1]; Randzellen per S^3-Supersampling."""
    X = xs[:, None, None] - xc
    Y = ys[None, :, None] - yc
    Z = zs[None, None, :] - zc
    d = np.sqrt(X*X + Y*Y + Z*Z)
    half = h*np.sqrt(3.0)/2.0
    f = (d <= r - half).astype(np.float64)
    edge = np.nonzero((d > r - half) & (d < r + half))
    if len(edge[0]):
        o = _offsets(S, h)
        n = len(edge[0])
        hit = np.zeros(n)
        for a in o:                                  # Speicher: n*S^2 statt n*S^3
            px = (X[edge[0], 0, 0] + a)[:, None, None]
            py = Y[0, edge[1], 0][:, None, None] + o[None, :, None]
            pz = Z[0, 0, edge[2]][:, None, None] + o[None, None, :]
            hit += np.sum((px*px + py*py + pz*pz) <= r*r, axis=(1, 2))
        f[edge] = hit/S**3
    return f


def blend(eps, frac, n_incl):
    """eps <- (1-f) eps + f n^2 (in place auf dem uebergebenen Ausschnitt)."""
    m = frac > 0
    eps[m] = (1.0 - frac[m])*eps[m] + frac[m]*(n_incl**2)
    return eps


# ---------------------------------------------------------------------------
# Schichtgrenzen (eben, Normale = y) und beliebige Materialkarten (CAD)
# ---------------------------------------------------------------------------
LAYER_SUBSAMPLES = 16       # Stuetzstellen je Zelle fuer die 1D-Schichtmittelung
CAD_SUBSAMPLES_2D = 4       # je Achse (16 Proben je Zelle)
CAD_SUBSAMPLES_3D = 2       # je Achse (8 Proben je Zelle; Speicher!)


def cell_average_column(ys, dy, eps_of_y, S=LAYER_SUBSAMPLES, harmonic=False, shift=0.0):
    """Mittel von eps(y) ueber die Zelle [y_j+shift-dy/2, y_j+shift+dy/2] (1D, Schichtstapel).
    harmonic=False: arithmetisches Mittel <eps> (Feld PARALLEL zu den Schichten: Ez, Ex);
    harmonic=True:  1/<1/eps>          (Feld SENKRECHT zu den Schichten: Ey).
    Liegen Grenzen zwischen Gitterpunkten, bekommt die Zelle den richtigen Volumenanteil
    statt auf das Gitter zu rasten (z.B. Lipid 30 nm bei dx 20 nm: frueher 20 oder 40 nm)."""
    off = ((np.arange(S) + 0.5)/S - 0.5)*dy + shift
    Y = np.asarray(ys, dtype=np.float64)[:, None] + off[None, :]
    e = np.asarray(eps_of_y(Y.ravel()))
    cplx = np.iscomplexobj(e)                    # Absorption: komplexes eps (n' + i kappa)^2
    e = e.astype(np.complex128 if cplx else np.float64).reshape(Y.shape)
    out = np.complex64 if cplx else np.float32
    if harmonic:
        return (1.0/np.mean(1.0/e, axis=1)).astype(out)
    return np.mean(e, axis=1).astype(out)


def fine_axis(start, n, h, S):
    """Feine Stuetzstellen: S je Zelle, symmetrisch um die Knoten start + i*h."""
    return start + (np.arange(n*S) + 0.5)*(h/S) - 0.5*h


def block_mean(a, S):
    """Mittel ueber S^d-Bloecke (d = a.ndim) -> Volumenanteil-gemitteltes eps."""
    shp = []
    for n in a.shape:
        shp += [n//S, S]
    b = a.reshape(shp)
    return b.mean(axis=tuple(range(1, 2*a.ndim, 2)))
