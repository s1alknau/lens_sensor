"""Lokale Sensor-Demo: zeigt das EVANESZENTE Traenenfilm-Signal in EINEM Fenster.

Physik: das evaneszente Feld entsteht dort, wo gefuehrtes Licht die Kern-Unterseite
(Traenenfilm-Grenze) beruehrt. Ein zentral eingekoppelter schmaler VCSEL-Spot
erreicht die Unterseite in einem kurzen Ausschnitt nicht -> kein Signal. Hier
koppeln wir OBERFLAECHENNAH ein (Offset zur Unterseite) mit echtem Traenenfilm,
sodass Feld an y=0 anliegt -> evaneszenter Schwanz in den Tear-Film messbar.

Vergleich: derselbe Lauf mit ZENTRALER Kopplung (offset 0) zeigt ~kein Signal.

GPU-Python:  python tests/sensor_demo.py [d_um L_um dx_nm offset_um]
"""
import os
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, 'planar_beads')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from common.physics import n_at


def run(d_um=250.0, length_um=60.0, dx_nm=100.0, offset_um=None, lam_nm=850.0):
    import fdtd2d_core as f2d
    n_core = n_at('pmma', lam_nm); n_aq = n_at('aqueous', lam_nm)
    lam_um = lam_nm/1000.0
    delta_min = lam_um/(2*np.pi*np.sqrt(max(n_core**2 - n_aq**2, 1e-9)))
    # oberflaechennah: Spot ~ an die Kern-Unterkante (y=0) legen -> offset=-d/2
    off = -d_um/2 if offset_um is None else offset_um
    prev = os.getcwd(); tmp = tempfile.mkdtemp()
    try:
        os.chdir(tmp)
        r = f2d.run_beads(
            wg_mat='pmma', bead_mat='polystyrol', bead_d_um=0.0, dx_nm=dx_nm,
            save_frames=True, n_snapshots=8, wg_thickness_um=d_um, lambda_nm=lam_nm,
            vcsel_waist=3.0, vcsel_offset=off, method='full', place_bead=False,
            length_um=length_um, polarization='s',
            t_aqueous_um=3.5, t_mucin_um=0.5)
        cwm = r.get('cw_frames_memmap'); cwf = r.get('cw_frames') or []
        arr = np.asarray(np.load(cwm, mmap_mode='r')[:len(cwf)])
    finally:
        os.chdir(prev)
    nfr, Nx, Ny = arr.shape
    E = arr[0].astype(np.float64) - 1j*arr[nfr//4 or 1].astype(np.float64)
    dx_um = dx_nm/1000.0
    tear = f2d.TEAR_BUFFER*1e6
    iy0 = int(round(tear/dx_um))               # Kern-Unterkante y=0
    iy_core_hi = int(round((tear + d_um)/dx_um))
    ys = -tear + np.arange(Ny)*dx_um            # y in um (0 = Kern-Unterkante)
    ix = int(0.6*Nx)                            # eingeschwungene Zone
    prof = np.abs(E[ix, :])
    # (1) evaneszente Eindringtiefe: |E| unter y=0 (im Traenenfilm) fitten
    m = (ys <= -2*dx_um) & (ys >= -6*delta_min) & (prof > prof.max()*1e-3)
    if m.sum() >= 4:
        g = np.polyfit(-ys[m], np.log(prof[m]), 1)[0]
        d_evan = 1.0/abs(g) if g else float('nan')
    else:
        d_evan = float('nan')
    # (2) Sensor-K: Intensitaet im Tear-Band / im Kern (am Ausgang)
    I = np.abs(E)**2
    iy_tear_lo = int(round((tear - 3.5)/dx_um))     # 3.5um Aqueous unter dem Kern
    P_core = float(I[ix, iy0:iy_core_hi].sum())
    P_tear = float(I[ix, iy_tear_lo:iy0].sum())
    K = P_tear/P_core if P_core > 0 else 0.0
    print(f'd={d_um}um L={length_um}um dx={dx_nm}nm offset={off:+.0f}um  '
          f'(delta_min theor={delta_min:.3f}um)')
    print(f'  Feld an Kern-Unterkante |E(y=0)|/max = {prof[iy0]/prof.max():.3f}')
    print(f'  evaneszente Eindringtiefe delta = {d_evan:.3f} um')
    print(f'  K = I_tear/I_core = {K:.4e}')
    return d_evan, K


if __name__ == '__main__':
    a = sys.argv
    d = float(a[1]) if len(a) > 1 else 250.0
    L = float(a[2]) if len(a) > 2 else 60.0
    dxn = float(a[3]) if len(a) > 3 else 100.0
    off = float(a[4]) if len(a) > 4 else None
    print('=== oberflaechennahe Kopplung (Offset zur Unterseite) ===')
    run(d, L, dxn, offset_um=off)
    print('=== Vergleich: zentrale Kopplung (offset 0) ===')
    run(d, L, dxn, offset_um=0.0)
