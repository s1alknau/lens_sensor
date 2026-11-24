"""Validierung der ABSORPTION (komplexes n = n' + i kappa) im nativen 2D-Solver.

Fall: duenner PMMA-Kern (0.5 um) auf absorbierendem Aqueous (8 um, praktisch
halbunendlich, kappa_aq), Luft oben, 850 nm - der Sensorfall "Absorptions-
spektroskopie im Traenenfilm": nur das evaneszente Feld sieht den Verlust.

Analytisch: komplexe Grundmode N = n_eff + i n_eff'' aus der Eigenwertgleichung des
asymmetrischen Slabs mit komplexem n_aq (Sekantenverfahren im Komplexen, Start = reelle
Loesung). Die Feldamplitude der Mode klingt wie exp(-k0 n_eff'' x) ab.

FDTD: Lauf mit KAPPA = {'aq': kappa}; Abklingrate aus der Steigung von ln|Ê| der
CW-Amplitude im Kern (Summe ueber den Kernquerschnitt, weit weg von der Quelle), n_eff
aus der Phase. Zusaetzlich die Transmission (Poynting) gegen exp(-2 k0 n_eff'' dx_det).

Aufruf:  python tests/absorption_validation.py [kappa] [dx_nm]   (Default 0.01, 20 nm)
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
D_UM = 0.5
L_UM = 40.0
TOL_IM = 0.05                 # relativ: Abklingrate (n_eff'')
TOL_RE = 0.006                # absolut: n_eff (Yee-Dispersion)


def neff_complex(n_core, n_top, n_bot, d_um, lam_nm, pol):
    """Komplexe Grundmode (m=0) mit komplexem n_bot."""
    k0 = 2*np.pi/(lam_nm*1e-3)

    def disp(N):
        kx = k0*np.sqrt(n_core**2 - N**2 + 0j)
        gt = k0*np.sqrt(N**2 - n_top**2 + 0j)
        gb = k0*np.sqrt(N**2 - n_bot**2 + 0j)
        pt, pb = ((n_core**2/n_top**2)*gt, (n_core**2/n_bot**2)*gb) if pol == 'p' else (gt, gb)
        return kx*d_um - np.arctan(pt/kx) - np.arctan(pb/kx)

    x0 = slab_neff_analytic(n_core, n_top, np.real(n_bot), d_um, lam_nm, pol, 0) + 0j
    x1 = x0 + 1e-4 + 1e-5j
    f0, f1 = disp(x0), disp(x1)
    for _ in range(100):                                       # Sekantenverfahren
        x2 = x1 - f1*(x1 - x0)/(f1 - f0)
        x0, f0, x1, f1 = x1, f1, x2, disp(x2)
        if abs(x1 - x0) < 1e-13:
            break
    return x1


def run_case(pol, kappa, dx_nm):
    prev = os.getcwd(); tmp = tempfile.mkdtemp()
    f2d.KAPPA = {'aq': kappa} if kappa > 0 else {}
    try:
        os.chdir(tmp)
        r = f2d.run_beads('pmma', 'polystyrol', 0.0, dx_nm=dx_nm, save_frames=True,
                          n_snapshots=8, wg_thickness_um=D_UM, lambda_nm=LAM, method='full',
                          place_bead=False, length_um=L_UM, polarization=pol,
                          vcsel_waist=0.3, t_aqueous_um=8.0, t_mucin_um=0.5,
                          t_lipid_um=0.0, boundary='cpml')
        arr = np.asarray(np.load(r['cw_frames_memmap'], mmap_mode='r'), dtype=np.float64)
        n = arr.shape[0]
        E = (2.0/n)*np.tensordot(np.exp(-2j*np.pi*np.arange(n)/n), arr, axes=(0, 0))
        T = float(r['transmission'])
        for k in ('frames_memmap', 'cw_frames_memmap'):
            try:
                os.remove(r.get(k) or '')
            except Exception:
                pass
    finally:
        os.chdir(prev); f2d.KAPPA = {}
    dx_um = dx_nm/1000.0
    ys = -f2d.TEAR_BUFFER*1e6 + np.arange(E.shape[1])*dx_um
    i0, i1 = int(0.35*E.shape[0]), int(0.85*E.shape[0])
    jc = (ys >= 0) & (ys <= D_UM)
    line = E[i0:i1][:, jc].sum(axis=1)
    x = np.arange(len(line))*dx_um
    ph = np.unwrap(np.angle(line))
    k0 = 2*np.pi/(LAM*1e-3)
    n_re = abs(np.polyfit(x, ph, 1)[0])/k0
    n_im = -np.polyfit(x, np.log(np.abs(line)), 1)[0]/k0
    return n_re, n_im, T


def main(kappa=0.01, dx_nm=20.0):
    n_core = n_at('pmma', LAM); n_aq = n_at('aqueous', LAM)
    print(f'Absorptions-Validierung: Kern {D_UM} um PMMA (n={n_core:.4f}) / Aqueous '
          f'n={n_aq:.4f}+{kappa:g}i, {LAM:g} nm, dx {dx_nm:g} nm, L {L_UM:g} um')
    ok_all = True
    rows = []
    for pol in ('s', 'p'):
        N = neff_complex(n_core, 1.0, complex(n_aq, kappa), D_UM, LAM, pol)
        n_re, n_im, T = run_case(pol, kappa, dx_nm)
        okr = abs(n_re - N.real) < TOL_RE
        oki = abs(n_im - N.imag) < TOL_IM*abs(N.imag)
        ok_all &= okr and oki
        # Anteil des Verlusts: n_eff''/kappa ~ Leistungsanteil im Aqueous (x n_aq/n_eff)
        rows.append((pol, N, n_re, n_im, T, okr, oki))
    h_im, h_rat = "n_eff'' ana", "n''/kappa"
    print(f'\n{"Pol":<4}{"n_eff ana":>11}{"FDTD":>9}{h_im:>13}{"FDTD":>11}'
          f'{h_rat:>11}{"T":>9}   ok')
    for pol, N, nr, ni, T, o1, o2 in rows:
        print(f'{"TE" if pol == "s" else "TM":<4}{N.real:>11.5f}{nr:>9.5f}{N.imag:>13.3e}{ni:>11.3e}'
              f'{N.imag/kappa:>11.4f}{T:>9.4f}   {"OK" if o1 else "X"}/{"OK" if o2 else "X"}')
    print(f'\nToleranzen: n_eff +-{TOL_RE}, n_eff\'\' +-{100*TOL_IM:.0f} %')
    print('GESAMT: ' + ('BESTANDEN' if ok_all else 'NICHT BESTANDEN'))
    return ok_all


if __name__ == '__main__':
    a = sys.argv[1:]
    main(float(a[0]) if a else 0.01, float(a[1]) if len(a) > 1 else 20.0)
