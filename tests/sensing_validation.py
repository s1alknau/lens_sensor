"""Validierung der SENSOR-Physik (Review F2): der dicke 5-um-Slab in crossval.py ist
fast unempfindlich gegen den Traenenfilm (dn_eff/dn_aq ~ 5e-4). Hier ein duenner,
wenig-modiger Kern, bei dem der Film wirklich zaehlt (dn_eff/dn_aq ~ 0.1-0.2):

  (1) n_eff der Grundmode     FDTD (Phase der komplexen CW-Amplitude)  vs analytisch
  (2) dn_eff/dn_aq            zwei Laeufe mit n_aq +- dn                vs analytisch
  (3) Abklinglaenge 1/gamma   Fit von ln|E^(y)| im Aqueous (komplexer Phasor ->
                              keine Stehwellen-/Phasenartefakte)        vs analytisch

Geometrie: Luft / PMMA-Kern d / Aqueous (8 um, praktisch halbunendlich), 850 nm.
Brechzahlen aus common.physics (Sellmeier-Fits). Laeuft auf der GPU (CuPy) oder CPU.

Aufruf:  python tests/sensing_validation.py [d_um] [dx_nm]     (Default 0.5 um, 20 nm)
"""
import os
import sys
import tempfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tests'))
from crossval import slab_neff_analytic                     # noqa: E402
from common.physics import n_at                             # noqa: E402
import planar_beads.fdtd2d_core as f2d                       # noqa: E402

LAM = 850.0
DN = 0.01                     # Variation von n_aq fuer die Ableitung
TOL_NEFF = 0.006              # absolut (Yee-Dispersion bei dx 20 nm ~ +0.003)
TOL_DERIV = 0.05              # relativ
TOL_DECAY = 0.05              # relativ


def _cw_phasor(r):
    """Komplexe CW-Amplitude Ê (Nx, Ny) per DFT ueber alle gespeicherten Phasen."""
    arr = np.asarray(np.load(r['cw_frames_memmap'], mmap_mode='r'), dtype=np.float64)
    n = arr.shape[0]
    return (2.0/n)*np.tensordot(np.exp(-2j*np.pi*np.arange(n)/n), arr, axes=(0, 0))


def run_case(d_um, dx_nm, pol, n_aq, length_um=30.0):
    prev = os.getcwd()
    tmp = tempfile.mkdtemp()
    try:
        os.chdir(tmp)
        r = f2d.run_beads('pmma', 'polystyrol', 0.0, dx_nm=dx_nm, save_frames=True,
                          n_snapshots=8, wg_thickness_um=d_um, lambda_nm=LAM, method='full',
                          place_bead=False, length_um=length_um, polarization=pol,
                          vcsel_waist=max(0.3, 0.6*d_um), t_aqueous_um=8.0, t_mucin_um=0.5,
                          t_lipid_um=0.0, n_aqueous=n_aq, boundary='cpml')
        E = _cw_phasor(r)
        for k in ('frames_memmap', 'cw_frames_memmap'):
            try:
                os.remove(r.get(k) or '')
            except Exception:
                pass
    finally:
        os.chdir(prev)
    dx_um = dx_nm/1000.0
    ys = -f2d.TEAR_BUFFER*1e6 + np.arange(E.shape[1])*dx_um
    i0, i1 = int(0.45*E.shape[0]), int(0.9*E.shape[0])     # weit weg von der Quelle
    # (1) n_eff: Phasensteigung in der Kernmitte, gemittelt ueber den Querschnitt des
    # Kerns (gewichtete Phase: Summe ueber y -> Grundmode dominiert)
    jc = (ys >= 0) & (ys <= d_um)
    line = E[i0:i1][:, jc].sum(axis=1)
    ph = np.unwrap(np.angle(line))
    beta = abs(np.polyfit(np.arange(len(ph))*dx_um, ph, 1)[0])
    neff = beta/(2*np.pi/(LAM*1e-3))
    # (3) Abklinglaenge: Fit von ln|Ê(y)| NUR ueber die erste Abklinglaenge unter der
    # Grenzflaeche (tiefer liegt ein Untergrund aus abgestrahltem Quellenlicht von einigen
    # Prozent, der den Fit verfaelscht), je x-Spalte, dann Median ueber x.
    gamma_ana = 2*np.pi/(LAM*1e-3)*np.sqrt(max(neff**2 - n_aq**2, 1e-12))
    L_ana = 1.0/gamma_ana
    m = (ys <= -2*dx_um) & (ys >= -1.0*L_ana)
    Ls = []
    for ix in range(i0, i1, max(1, (i1 - i0)//40)):
        pr = np.abs(E[ix])
        if m.sum() >= 4 and np.all(pr[m] > 0):
            g = np.polyfit(ys[m], np.log(pr[m]), 1)[0]
            if g > 0:
                Ls.append(1.0/g)
    return neff, (float(np.median(Ls)) if Ls else float('nan'))


def main(d_um=0.5, dx_nm=20.0):
    n_core = n_at('pmma', LAM)
    n_aq0 = n_at('aqueous', LAM)
    print(f'Sensor-Validierung: Kern {d_um} um PMMA (n={n_core:.4f}) / Aqueous n={n_aq0:.4f}, '
          f'{LAM:g} nm, dx {dx_nm:g} nm')
    rows = []
    ok_all = True
    for pol in ('s', 'p'):
        ana0 = slab_neff_analytic(n_core, 1.0, n_aq0, d_um, LAM, pol, 0)
        anap = slab_neff_analytic(n_core, 1.0, n_aq0 + DN, d_um, LAM, pol, 0)
        der_ana = (anap - ana0)/DN
        L_ana = 1.0/(2*np.pi/(LAM*1e-3)*np.sqrt(ana0**2 - n_aq0**2))
        ne0, L0 = run_case(d_um, dx_nm, pol, n_aq0)
        nep, _ = run_case(d_um, dx_nm, pol, n_aq0 + DN)
        der = (nep - ne0)/DN
        okn = abs(ne0 - ana0) < TOL_NEFF
        okd = abs(der - der_ana) < TOL_DERIV*abs(der_ana)
        okL = abs(L0 - L_ana) < TOL_DECAY*L_ana
        ok_all &= okn and okd and okL
        rows.append((pol, ana0, ne0, der_ana, der, L_ana, L0, okn, okd, okL))
    print(f'\n{"Pol":<4}{"n_eff ana":>10}{"FDTD":>9}{"dn/dn_aq ana":>14}{"FDTD":>8}'
          f'{"1/gamma ana":>13}{"FDTD":>8}   ok')
    for pol, a0, n0, da, d, La, L, o1, o2, o3 in rows:
        print(f'{"TE" if pol == "s" else "TM":<4}{a0:>10.4f}{n0:>9.4f}{da:>14.4f}{d:>8.4f}'
              f'{La:>12.3f}u{L:>7.3f}u   {"OK" if o1 else "X"}/{"OK" if o2 else "X"}/'
              f'{"OK" if o3 else "X"}')
    print(f'\nToleranzen: n_eff +-{TOL_NEFF} (Yee-Dispersion), Ableitung +-{100*TOL_DERIV:.0f}%, '
          f'Abklinglaenge +-{100*TOL_DECAY:.0f}%')
    print('GESAMT: ' + ('BESTANDEN' if ok_all else 'NICHT BESTANDEN'))
    return ok_all


if __name__ == '__main__':
    a = sys.argv[1:]
    main(float(a[0]) if a else 0.5, float(a[1]) if len(a) > 1 else 20.0)
