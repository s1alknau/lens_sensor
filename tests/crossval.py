"""Physik-Cross-Validierung: geführter Mode-Index n_eff.

Drei unabhaengige Wege fuer DENSELBEN asymmetrischen Slab-Waveguide:
  1. ANALYTISCH  - Dispersionsrelation des asymmetrischen Slabs (Grundwahrheit)
  2. NATIV       - eigener FDTD-Solver: beta aus der Phase des CW-Feldes (FFT in x)
  3. MEEP        - Eigenmode-Solver (MPB), laeuft in WSL

Stimmen 2/3 nicht mit 1 ueberein (ueber der Diskretisierungs-Toleranz), ist das
ein Physik-Bug - genau das, was der Golden-Master-Test NICHT sieht.

Geometrie (wie planar_beads): Luft (oben) / PMMA-Kern d / Aqueous (unten).
"""
import numpy as np

C0 = 2.99792458e8


# ----------------------------------------------------------------------------
# 1) ANALYTISCH: asymmetrischer 3-Schicht-Slab
# ----------------------------------------------------------------------------
def slab_neff_analytic(n_core, n_top, n_bot, d_um, lam_nm, pol='s', m=0):
    """Effektiver Index der m-ten gefuehrten Mode eines asymmetrischen Slabs.
    pol='s' -> TE (Ez/Hx/Hy, E parallel zur Schicht); 'p' -> TM.
    d_um: Kerndicke; Rueckgabe n_eff oder None (keine gefuehrte Mode)."""
    k0 = 2*np.pi/(lam_nm*1e-3)          # 1/um
    d = d_um
    nlo = max(n_top, n_bot)             # n_eff muss in (nlo, n_core) liegen
    if n_core <= nlo:
        return None

    def disp(neff):
        # transversale Wellenzahl im Kern, Abklingraten in den Claddings
        kx = k0*np.sqrt(max(n_core**2 - neff**2, 1e-18))
        gt = k0*np.sqrt(max(neff**2 - n_top**2, 1e-18))
        gb = k0*np.sqrt(max(neff**2 - n_bot**2, 1e-18))
        if pol == 'p':                  # TM: mit Index-Gewichtung
            pt = (n_core**2/n_top**2)*gt
            pb = (n_core**2/n_bot**2)*gb
        else:                           # TE
            pt, pb = gt, gb
        # Eigenwertgleichung: kx*d - atan(pt/kx) - atan(pb/kx) - m*pi = 0
        return kx*d - np.arctan(pt/kx) - np.arctan(pb/kx) - m*np.pi

    # Bisektion im offenen Intervall (nlo, n_core)
    lo, hi = nlo + 1e-9, n_core - 1e-9
    flo, fhi = disp(lo), disp(hi)
    if flo*fhi > 0:
        return None                     # keine Nullstelle -> Mode m existiert nicht
    for _ in range(200):
        mid = 0.5*(lo + hi)
        fm = disp(mid)
        if flo*fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5*(lo + hi)


# ----------------------------------------------------------------------------
# 2) NATIV: beta aus dem CW-Feld (FFT entlang x im Kern)
# ----------------------------------------------------------------------------
def slab_decay_analytic(n_eff, n_clad, lam_nm):
    """Evaneszente 1/e-Abklinglaenge (um) in ein Cladding: L = 1/(k0*sqrt(neff^2-nclad^2))."""
    k0 = 2*np.pi/(lam_nm*1e-3)
    g = k0*np.sqrt(max(n_eff**2 - n_clad**2, 1e-18))
    return 1.0/g


def slab_measure_native(n_core_mat='pmma', d_um=5.0, lam_nm=850.0, pol='s',
                        length_um=60.0, dx_nm=20.0):
    """Laeuft den eigenen 2D-Solver (uniformer WG, kein Bead) und misst aus EINEM Lauf
    (1) n_eff aus der Phase der gefuehrten Mode entlang x und (2) die evaneszente
    Abklinglaenge ins Aqueous (unter dem WG). Rueckgabe (n_eff, L_evan_um)."""
    import os
    import sys
    import tempfile
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(_root, 'planar_beads'))
    import fdtd2d_core as f2d
    prev = os.getcwd()
    tmp = tempfile.mkdtemp()
    try:
        os.chdir(tmp)
        r = f2d.run_beads(
            wg_mat=n_core_mat, bead_mat='polystyrol', bead_d_um=0.0,
            dx_nm=dx_nm, save_frames=True, n_snapshots=8,
            wg_thickness_um=d_um, lambda_nm=lam_nm, method='full',
            place_bead=False, length_um=length_um, polarization=pol,
            t_aqueous_um=8.0, t_mucin_um=2.0)
        # Bevorzugt das eingeschwungene CW-Feld (2 Phasen -> komplexer Phasor).
        ez = None
        cwm = r.get('cw_frames_memmap'); cwf = r.get('cw_frames') or []
        if cwm and os.path.exists(cwm) and len(cwf) >= 2:
            arr = np.asarray(np.load(cwm, mmap_mode='r')[:len(cwf)])
            ez = arr[0].astype(np.float64) - 1j*arr[len(cwf)//4 or 1].astype(np.float64)
        if ez is None:
            frames = r.get('frames') or []
            mm = r.get('frames_memmap')
            ez = (np.asarray(np.load(mm, mmap_mode='r')[len(frames)-1])
                  if (mm and os.path.exists(mm)) else np.asarray(frames[-1]['Ez']))
            ez = ez.astype(np.float64)
    finally:
        os.chdir(prev)
    # ez: (Nx, Ny). y-Index des Kern-Zentrums (Kern y in [0, d]); Geometrie:
    # y = -TEAR_BUFFER + iy*dx  -> Kernmitte y=d/2.
    dx_um = dx_nm/1000.0
    tear_buf_um = f2d.TEAR_BUFFER*1e6
    iy_core = int(round((d_um/2 + tear_buf_um)/dx_um))
    iy_core = min(max(iy_core, 0), ez.shape[1]-1)
    line = ez[:, iy_core]
    i0, i1 = int(0.35*len(line)), int(0.85*len(line))   # eingeschwungener Mittelteil
    seg = line[i0:i1]
    k0 = 2*np.pi/(lam_nm*1e-3)
    if np.iscomplexobj(seg):
        # PRAEZISE: beta = Steigung der entwrappten Phase (linearer Fit)
        ph = np.unwrap(np.angle(seg))
        x = np.arange(len(ph))*dx_um
        beta = abs(np.polyfit(x, ph, 1)[0])          # rad/um
    else:
        # Fallback (reelles Feld): FFT + parabolische Sub-Bin-Interpolation
        s = (seg - seg.mean())*np.hanning(len(seg))
        sp = np.abs(np.fft.rfft(s)); sp[0] = 0.0
        dk = 1.0/(len(s)*dx_um)                       # cycles/um pro Bin
        i = int(np.argmax(sp))
        di = 0.0
        if 0 < i < len(sp)-1:
            den = sp[i-1] - 2*sp[i] + sp[i+1]
            di = 0.5*(sp[i-1] - sp[i+1])/den if den != 0 else 0.0
        beta = 2*np.pi*(i + di)*dk
    neff = beta/k0
    # (2) Evaneszente Abklinglaenge ins Aqueous (y<0): |E(y)| ~ exp(-gamma*|y|).
    ix = (i0 + i1)//2                                # fester x in der Steady-Zone
    prof = np.abs(ez[ix, :]).astype(np.float64)
    ys = -tear_buf_um + np.arange(ez.shape[1])*dx_um
    # Abklinglaenge ins Aqueous ist ~0.2um (n_eff >> n_aq) -> DICHT an der
    # Grenzflaeche fitten, nur wo das Feld ueber dem Rauschen liegt.
    L_ana = slab_decay_analytic(neff, 1.336, lam_nm)   # Aqueous-Cladding
    y_hi = -2*dx_um                                  # knapp unter y=0 (Interface-Zelle weg)
    y_lo = max(-6*L_ana, -1.5)                       # ~6 Abklinglaengen tief
    mask = (ys <= y_hi) & (ys >= y_lo) & (prof > prof.max()*1e-3)
    if mask.sum() >= 4:
        g = np.polyfit(ys[mask], np.log(prof[mask]), 1)[0]   # d ln|E|/dy > 0 im Aqueous
        L_evan = 1.0/abs(g) if g != 0 else float('nan')
    else:
        L_evan = float('nan')
    return neff, L_evan


# ----------------------------------------------------------------------------
# 3) MEEP: Eigenmode-Solver (MPB). Laeuft nur unter Linux/WSL.
# ----------------------------------------------------------------------------
def slab_neff_meep(n_core=1.491, n_top=1.0, n_bot=1.336, d_um=5.0, lam_nm=850.0,
                   pol='s', res=40):
    import meep as mp
    air, tear = 3.0, 8.0
    ly = d_um + air + tear
    y_top = d_um + air; y_bot = -tear; cy = 0.5*(y_top + y_bot)
    cell = mp.Vector3(2.0, ly, 0)
    geom = [
        mp.Block(mp.Vector3(mp.inf, d_um, mp.inf),
                 center=mp.Vector3(0, 0.5*d_um - cy), material=mp.Medium(index=n_core)),
        mp.Block(mp.Vector3(mp.inf, tear, mp.inf),
                 center=mp.Vector3(0, -0.5*tear - cy), material=mp.Medium(index=n_bot)),
    ]
    sim = mp.Simulation(cell_size=cell, resolution=res, geometry=geom,
                        default_material=mp.Medium(index=n_top),
                        boundary_layers=[mp.PML(1.0, direction=mp.Y)])
    sim.init_sim()
    f0 = 1000.0/lam_nm
    parity = mp.ODD_Z if pol == 's' else mp.EVEN_Z
    em = sim.get_eigenmode(
        f0, mp.X, mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, ly, 0)),
        1, mp.Vector3(f0*0.5*(n_core + max(n_top, n_bot)), 0, 0),
        parity=parity)
    return abs(em.k.x)/f0


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'meep':
        # In WSL: python crossval.py meep <d_um> <lam_nm> <pol>
        d_um, lam_nm, pol = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
        nm = slab_neff_meep(1.491, 1.0, 1.336, d_um, lam_nm, pol)
        na = slab_neff_analytic(1.491, 1.0, 1.336, d_um, lam_nm, pol)
        print(f'MEEPRESULT d={d_um} L={lam_nm:.0f} {pol}: meep={nm:.4f} '
              f'analytisch={na:.4f} Delta={nm-na:+.4f}')
        sys.exit(0)
    # PMMA-Kern / Luft oben / Aqueous unten
    N_CORE, N_TOP, N_BOT = 1.491, 1.000, 1.336
    dx = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    cases = [
        dict(d_um=5.0, lam_nm=850.0, pol='s'),
        dict(d_um=5.0, lam_nm=850.0, pol='p'),
        dict(d_um=3.0, lam_nm=532.0, pol='s'),
    ]
    print(f'dx = {dx:g} nm    (n_eff = validierter Kern-Check; Levan = grob/'
          f'experimentell: sehr kurze Abklinglaenge + Substrat-Kontamination)')
    print(f'{"Fall":<22}{"neff_ana":>10}{"neff_nat":>10}{"dNeff":>8}'
          f'{"Levan_ana":>11}{"Levan_nat":>11}{"dL%":>7}')
    for c in cases:
        na = slab_neff_analytic(N_CORE, N_TOP, N_BOT, c['d_um'], c['lam_nm'], c['pol'])
        nn, Ln = slab_measure_native(d_um=c['d_um'], lam_nm=c['lam_nm'],
                                     pol=c['pol'], length_um=40.0, dx_nm=dx)
        La = slab_decay_analytic(na, N_BOT, c['lam_nm'])   # aus Grundwahrheit
        tag = f"d={c['d_um']}um L={c['lam_nm']:.0f} {c['pol']}"
        dL = 100*(Ln - La)/La if (La and Ln == Ln) else float('nan')
        print(f'{tag:<22}{na:>10.4f}{nn:>10.4f}{nn-na:>+8.4f}'
              f'{La:>11.4f}{Ln:>11.4f}{dL:>+7.1f}')
