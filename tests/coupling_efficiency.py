"""Kopplungs-Effizienz-Diagnose: welcher Anteil des eingekoppelten VCSEL-Lichts
laeuft als GEFUEHRTE (TIR-)Mode durch die Linse?

Kriterium (Slab/Linse): gefuehrt sind transversale Wellenzahlen mit
    |ky| < k0 * NA,   NA = sqrt(n_core^2 - n_clad^2)   (n_clad = Traenenfilm/Aqueous),
d.h. Strahlwinkel < arcsin(NA/n_core) zur Linsenachse. Wir messen das ueber das
WINKELSPEKTRUM (FFT der komplexen CW-Amplitude entlang y im Kern) an mehreren
x-Querschnitten. Strahlungsanteile (|ky|>k0*NA) lecken ins Cladding -> verloren.

Baseline-Lauf: gerader PMMA-"Linsen"-Waveguide (geflatteter Meridian), Dicke
250 um, Aqueous unten, VCSEL-Gauss-Spot. Ausfuehren mit GPU-Python:
    python tests/coupling_efficiency.py            # Defaults
    python tests/coupling_efficiency.py 250 400 100 2   # d_um L_um dx_nm waist_um
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


def guided_fraction(col, dx_um, k0, NA):
    """Anteil der Leistung im gefuehrten Winkelkegel |ky| < k0*NA (Winkelspektrum
    der komplexen Feldspalte col(y))."""
    n = len(col)
    F = np.fft.fft(col - col.mean())
    ky = 2*np.pi*np.fft.fftfreq(n, d=dx_um)          # rad/um
    P = np.abs(F)**2
    tot = P.sum()
    if tot <= 0:
        return float('nan')
    return float(P[np.abs(ky) < k0*NA].sum()/tot)


def measure(d_um=250.0, length_um=400.0, dx_nm=100.0, waist_um=2.0,
            tilt_deg=0.0, offset_um=0.0, lam_nm=850.0):
    import fdtd2d_core as f2d
    n_core = n_at('pmma', lam_nm)
    n_clad = n_at('aqueous', lam_nm)
    NA = float(np.sqrt(max(n_core**2 - n_clad**2, 1e-9)))
    k0 = 2*np.pi/(lam_nm*1e-3)                         # rad/um
    theta_acc = np.degrees(np.arcsin(NA/n_core))
    print(f'n_core={n_core:.4f} n_clad(aqueous)={n_clad:.4f}  NA={NA:.4f}  '
          f'Akzeptanz-Halbwinkel={theta_acc:.1f} deg (zur Achse)')
    print(f'Lauf: d={d_um} um  L={length_um} um  dx={dx_nm} nm  '
          f'waist={waist_um} um  tilt={tilt_deg} deg  offset={offset_um} um')

    prev = os.getcwd(); tmp = tempfile.mkdtemp()
    try:
        os.chdir(tmp)
        r = f2d.run_beads(
            wg_mat='pmma', bead_mat='polystyrol', bead_d_um=0.0, dx_nm=dx_nm,
            save_frames=True, n_snapshots=8, wg_thickness_um=d_um, lambda_nm=lam_nm,
            vcsel_waist=waist_um, vcsel_tilt=tilt_deg, vcsel_offset=offset_um,
            method='full', place_bead=False, length_um=length_um, polarization='s',
            t_aqueous_um=8.0, t_mucin_um=2.0)
        cwm = r.get('cw_frames_memmap'); cwf = r.get('cw_frames') or []
        arr = np.asarray(np.load(cwm, mmap_mode='r')[:len(cwf)])
    finally:
        os.chdir(prev)

    nfr, Nx, Ny = arr.shape
    E = arr[0].astype(np.float64) - 1j*arr[nfr//4 or 1].astype(np.float64)   # (Nx,Ny)
    dx_um = dx_nm/1000.0
    tear_buf_um = f2d.TEAR_BUFFER*1e6
    iy_lo = int(round(tear_buf_um/dx_um))                 # Kern-Unterkante (y=0)
    iy_hi = int(round((tear_buf_um + d_um)/dx_um))        # Kern-Oberkante (y=d)
    iy_lo = max(0, iy_lo); iy_hi = min(Ny, iy_hi)
    print(f'Grid {Nx}x{Ny}, Kern-Zellen y=[{iy_lo},{iy_hi}] ({iy_hi-iy_lo} Zellen)')

    print('\n x [um] | gefuehrter Anteil eta_guided')
    print(' -------+------------------------------')
    for x_um in (30, 60, 120, 200, 300):
        ix = int(round(x_um/dx_um))
        if not (0 <= ix < Nx):
            continue
        col = E[ix, iy_lo:iy_hi]
        eta = guided_fraction(col, dx_um, k0, NA)
        bar = '#'*int(round(40*eta)) if eta == eta else ''
        print(f'  {x_um:4d}  | {eta:6.3f}  {bar}')
    # Theoretischer Oberwert: Winkelspektrum des reinen Gauss-Quellprofils
    ys = np.arange(iy_hi - iy_lo)*dx_um
    src = np.exp(-((ys - ys.mean())/waist_um)**2).astype(np.complex128)
    print(f'\n(theor. Obergrenze aus Gauss-Quellprofil waist={waist_um}um: '
          f'{guided_fraction(src, dx_um, k0, NA):.3f})')


if __name__ == '__main__':
    a = sys.argv
    measure(d_um=float(a[1]) if len(a) > 1 else 250.0,
            length_um=float(a[2]) if len(a) > 2 else 400.0,
            dx_nm=float(a[3]) if len(a) > 3 else 100.0,
            waist_um=float(a[4]) if len(a) > 4 else 2.0,
            tilt_deg=float(a[5]) if len(a) > 5 else 0.0,
            offset_um=float(a[6]) if len(a) > 6 else 0.0)
