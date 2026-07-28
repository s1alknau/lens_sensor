"""Gemeinsame physikalische Konstanten und Materialdaten - eine Quelle der Wahrheit.

Zuvor waren die Konstanten (C0/EPS0/MU0/N_AIR), das ``DISPERSION``-Dict und
``n_at()`` in mehreren Solver-Dateien getrennt definiert (und drohten damit
auseinanderzulaufen). Alle Solver beziehen sie jetzt von hier.
"""
import numpy as np

# ---------- Fundamentalkonstanten ----------
C0 = 2.99792458e8            # Vakuum-Lichtgeschwindigkeit [m/s]
EPS0 = 8.8541878128e-12      # elektrische Feldkonstante [F/m]
MU0 = 4.0*np.pi*1e-7         # magnetische Feldkonstante [H/m]
N_AIR = 1.000                # Brechzahl Luft

# ---------- Materialdispersion ----------
# Wellenlaengen-abhaengige Brechzahlen n(lambda[nm]) - lineare Interpolation.
# 850nm = Referenzwerte (Kontinuitaet); 532/940 aus Literatur-Dispersion.
DISPERSION = {
    'pmma':       {532: 1.496, 850: 1.491, 940: 1.490},
    'polystyrol': {532: 1.601, 850: 1.590, 940: 1.588},
    'aqueous':    {532: 1.342, 850: 1.336, 940: 1.335},
    'mucin':      {532: 1.348, 850: 1.342, 940: 1.341},
    'cornea':     {532: 1.382, 850: 1.376, 940: 1.375},
    'lipid':      {532: 1.486, 850: 1.480, 940: 1.479},
}


def n_at(mat, lam_nm):
    """Brechzahl von ``mat`` bei Wellenlaenge ``lam_nm`` (lineare Interpolation)."""
    tbl = DISPERSION[mat]
    xs = sorted(tbl)
    return float(np.interp(lam_nm, xs, [tbl[x] for x in xs]))
