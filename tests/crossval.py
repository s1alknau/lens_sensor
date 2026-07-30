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
# 2b) NATIV 3D: derselbe Slab im vollen vektoriellen 3D-Solver (z-invariant).
#     Muss dieselbe Slab-Mode/n_eff wie 2D + analytisch liefern.
# ----------------------------------------------------------------------------
def slab_neff_native_3d(d_um=5.0, lam_nm=850.0, pol='s', length_um=30.0,
                        lz_um=4.0, dx_nm=50.0, wg_width_um=None):
    """Nativer 3D-FDTD (uniformer Slab bzw. Kanal wenn wg_width_um), n_eff aus der
    CW-Feldphase entlang x (bei Kern-y, z-Mitte)."""
    import os
    import sys
    import tempfile
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (os.path.join(_root, 'planar_3d'), _root):
        if p not in sys.path:
            sys.path.insert(0, p)
    import fdtd3d_core as f3d
    from common.physics import n_at
    tear = 8.0
    prev = os.getcwd()
    tmp = tempfile.mkdtemp()
    try:
        os.chdir(tmp)
        r = f3d.run_3d(
            label='slab3d', wg_n=n_at('pmma', lam_nm), t_wg_um=d_um,
            air_um=3.0, tear_um=tear, lz_um=lz_um, t_aq_um=tear, t_mu_um=2.0,
            n_aq=n_at('aqueous', lam_nm), n_mu=n_at('mucin', lam_nm),
            n_co=n_at('cornea', lam_nm), lam_nm=lam_nm, lx_um=length_um,
            dx_nm=dx_nm, n_snapshots=8, source_type='cw', polarization=pol,
            wg_width_um=wg_width_um, bead=None)
        cw = r.get('cw_vol')
        cw = np.asarray(cw) if cw is not None else None
    finally:
        os.chdir(prev)
    if cw is None or cw.shape[0] < 2:
        return float('nan')
    nfr = cw.shape[0]
    ez = cw[0].astype(np.float64) - 1j*cw[nfr//4 or 1].astype(np.float64)  # (Nx,Nj,Nk)
    dx_um = dx_nm/1000.0
    iy = int(round((d_um/2 + tear)/dx_um))               # y0 = -tear
    iy = min(max(iy, 0), ez.shape[1]-1)
    iz = ez.shape[2]//2
    line = ez[:, iy, iz]
    i0, i1 = int(0.35*len(line)), int(0.85*len(line))
    ph = np.unwrap(np.angle(line[i0:i1]))
    x = np.arange(len(ph))*dx_um
    beta = abs(np.polyfit(x, ph, 1)[0])
    return beta/(2*np.pi/(lam_nm*1e-3))


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


def slab_neff_meep_fdtd(d_um=5.0, lam_nm=850.0, pol='s', length_um=12.0,
                        lz_um=6.0, res=20, n_core=1.491, n_top=1.0, n_bot=1.336):
    """Meep im ZEITBEREICH (die FDTD-Engine hinter run_meep --dim 3): 3D-Slab,
    EigenModeSource treibt die Mode, n_eff aus der Phase des CW-DFT-Feldes entlang
    x. Schliesst die Luecke 'Meep-3D-FDTD' (mit eigener Numerik-Dispersion)."""
    import meep as mp
    air, tear = 3.0, 6.0
    ly = d_um + air + tear
    y_top = d_um + air; y_bot = -tear; cy = 0.5*(y_top + y_bot)
    dpml = 1.0
    cell = mp.Vector3(length_um, ly, lz_um)
    geom = [
        mp.Block(mp.Vector3(mp.inf, tear, mp.inf), center=mp.Vector3(0, -0.5*tear - cy, 0),
                 material=mp.Medium(index=n_bot)),
        mp.Block(mp.Vector3(mp.inf, d_um, mp.inf), center=mp.Vector3(0, 0.5*d_um - cy, 0),
                 material=mp.Medium(index=n_core)),
    ]
    f0 = 1000.0/lam_nm
    parity = mp.ODD_Z if pol == 's' else mp.EVEN_Z
    comp = mp.Ez if pol == 's' else mp.Ey
    src = mp.EigenModeSource(
        src=mp.ContinuousSource(frequency=f0),
        center=mp.Vector3(-length_um/2 + dpml + 0.3, 0.5*d_um - cy, 0),
        size=mp.Vector3(0, ly, lz_um), eig_band=1, direction=mp.X,
        eig_match_freq=True, eig_parity=parity)
    sim = mp.Simulation(cell_size=cell, resolution=res, geometry=geom, sources=[src],
                        default_material=mp.Medium(index=n_top),
                        boundary_layers=[mp.PML(dpml)], force_complex_fields=True)
    dft = sim.add_dft_fields([comp], f0, 0, 1,
                             center=mp.Vector3(0, 0.5*d_um - cy, 0),
                             size=mp.Vector3(length_um, 0, 0), yee_grid=False)
    sim.run(until=3.0*length_um*n_core + 40)          # bis eingeschwungen
    line = np.asarray(sim.get_dft_array(dft, comp, 0))   # entlang x bei (WG-Mitte, z=0)
    i0, i1 = int(0.35*len(line)), int(0.85*len(line))
    ph = np.unwrap(np.angle(line[i0:i1]))
    x = (np.arange(len(ph))/res)
    beta = abs(np.polyfit(x, ph, 1)[0])
    return beta/(2*np.pi/(lam_nm*1e-3))


def channel_neff_meep(d_um=5.0, w_um=3.0, lam_nm=850.0, n_core=1.491,
                      n_top=1.0, n_bot=1.336, n_side=1.0, res=30):
    """Meep-MPB: n_eff der Fundamentalmode eines RECHTECK-KANALS (Kern endlich in
    y UND z) - die 2D-Querschnittsmode, geloest in einem 3D-Setup. Grundwahrheit
    fuer den nativen 3D-Kanal (analytisch gibt es dafuer keine geschlossene Formel)."""
    import meep as mp
    air, tear = 3.0, 6.0
    ly = d_um + air + tear
    lz = w_um + 6.0                    # Kernbreite + seitliches Cladding/PML
    y_top = d_um + air; y_bot = -tear; cy = 0.5*(y_top + y_bot)
    cell = mp.Vector3(2.0, ly, lz)
    geom = [
        mp.Block(mp.Vector3(mp.inf, tear, mp.inf),        # Aqueous-Substrat (volle Tiefe)
                 center=mp.Vector3(0, -0.5*tear - cy, 0), material=mp.Medium(index=n_bot)),
        mp.Block(mp.Vector3(mp.inf, d_um, w_um),          # PMMA-Kern: endlich in y UND z
                 center=mp.Vector3(0, 0.5*d_um - cy, 0), material=mp.Medium(index=n_core)),
    ]
    sim = mp.Simulation(cell_size=cell, resolution=res, geometry=geom,
                        default_material=mp.Medium(index=n_top),
                        boundary_layers=[mp.PML(1.0, direction=mp.Y),
                                         mp.PML(1.0, direction=mp.Z)])
    sim.init_sim()
    f0 = 1000.0/lam_nm
    em = sim.get_eigenmode(
        f0, mp.X, mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, ly, lz)),
        1, mp.Vector3(f0*0.5*(n_core + max(n_top, n_bot)), 0, 0), parity=mp.NO_PARITY)
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
    if len(sys.argv) > 1 and sys.argv[1] == 'meepfdtd':
        # In WSL: python crossval.py meepfdtd <d_um> <lam_nm> <pol>
        d_um, lam_nm, pol = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
        nf = slab_neff_meep_fdtd(d_um=d_um, lam_nm=lam_nm, pol=pol)
        nmpb = slab_neff_meep(1.491, 1.0, 1.336, d_um, lam_nm, pol)
        print(f'MEEPFDTD d={d_um} L={lam_nm:.0f} {pol}: meep-FDTD n_eff={nf:.4f} '
              f'meep-MPB={nmpb:.4f} Delta={nf-nmpb:+.4f}')
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == 'meepchannel':
        # In WSL: python crossval.py meepchannel <d_um> <w_um> <lam_nm>
        d_um, w_um, lam_nm = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        nm = channel_neff_meep(d_um, w_um, lam_nm)
        print(f'MEEPCHANNEL d={d_um} w={w_um} L={lam_nm:.0f}: meep-MPB n_eff={nm:.4f}')
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == 'natchannel':
        # Windows/GPU: python crossval.py natchannel <d_um> <w_um> <lam_nm> <pol> <dx_nm>
        d_um, w_um = float(sys.argv[2]), float(sys.argv[3])
        lam_nm, pol = float(sys.argv[4]), sys.argv[5]
        dx = float(sys.argv[6]) if len(sys.argv) > 6 else 40.0
        # kompakte Domaene, damit feineres dx auf 4-GB-GPU passt
        n3 = slab_neff_native_3d(d_um=d_um, lam_nm=lam_nm, pol=pol, length_um=20.0,
                                 lz_um=w_um + 4.0, dx_nm=dx, wg_width_um=w_um)
        print(f'NATCHANNEL d={d_um} w={w_um} L={lam_nm:.0f} {pol} dx={dx:g}: '
              f'nativ-3D n_eff={n3:.4f}')
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == '3d':
        # 3D-Slab-Check: analytisch vs nativ-2D vs nativ-3D (gleiches dx).
        dx = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
        print(f'3D-Slab-Check (dx={dx:g}nm)  '
              f'[nativ3D ~ nativ2D = 3D-Solver konsistent zum Slab]')
        print(f'{"Fall":<20}{"analyt":>9}{"nat2D":>9}{"nat3D":>9}{"3D-2D":>8}')
        for c in (dict(d_um=5.0, lam_nm=850.0, pol='s'),
                  dict(d_um=5.0, lam_nm=850.0, pol='p')):
            na = slab_neff_analytic(1.491, 1.0, 1.336, c['d_um'], c['lam_nm'], c['pol'])
            n2, _ = slab_measure_native(d_um=c['d_um'], lam_nm=c['lam_nm'],
                                        pol=c['pol'], length_um=30.0, dx_nm=dx)
            n3 = slab_neff_native_3d(d_um=c['d_um'], lam_nm=c['lam_nm'],
                                     pol=c['pol'], length_um=30.0, dx_nm=dx)
            tag = f"d={c['d_um']} L={c['lam_nm']:.0f} {c['pol']}"
            print(f'{tag:<20}{na:>9.4f}{n2:>9.4f}{n3:>9.4f}{n3-n2:>+8.4f}')
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
