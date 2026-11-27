"""2D wide-angle finite-difference beam propagation method (FD-BPM).

Propagates a CW field along x through a layered cross-section n(y) (contact lens:
air | PMMA core | lipid | aqueous | mucin | cornea) without a grid along x, so it can
cover the whole lens (mm) in seconds to minutes, where FDTD needs days.

Method
------
Slowly varying envelope E(x, y) = u(x, y) exp(i kb x), kb = k0 n_ref. The one-way
Helmholtz equation is

    du/dx = i kb (sqrt(1 + X) - 1) u,     X = (A - kb^2)/kb^2,

    TE (Ez):  A = d2/dy2 + k0^2 eps(y)
    TM (Hz):  A = eps d/dy (1/eps d/dy) + k0^2 eps(y)

sqrt(1 + X) - 1 is replaced by a Pade (m, m) approximant N/D (wide angle; m = 1 is the
classic Hadley 1992 scheme, valid to ~30 deg, m = 2..4 for steeper rays) and the step
is Crank-Nicolson:

    [D - (i s/2) N] u_{n+1} = [D + (i s/2) N] u_n,    s = kb dx.

Both sides are degree-m polynomials in X; they are factored into m tridiagonal factors
(Hadley's multistep method), so one step costs m tridiagonal matvecs + m tridiagonal
solves. The scheme is unitary for real eps (power conserving) and unconditionally
stable.

Grid: non-uniform in y (fine at the tear film, where the evanescent field decays over
~0.2 um, coarse in the 250 um core). eps is averaged over each node's dual cell
(arithmetic; TE and the eps prefactor of TM) and 1/eps over each half interval
(harmonic; TM), the same rule as the FDTD subpixel averaging. A thin layer such as
the 30 nm lipid is therefore represented exactly by its volume share even on a coarse
grid. Open boundaries: PML by complex coordinate stretching of the y spacing.

Curvature: conformal mapping of the bent lens onto a straight one,
eps_eq = eps (1 + u/R)^2, u = y - core centre (exactly as ``--curved`` in the 2D FDTD
and ``kontaktlinse/mode_propagation.py``). Exact for TE; for TM the same scalar mapping
is an approximation (error ~ 1/(k R), small for R = 8.3 mm).
The linear factor (1 + u/R) instead of the exact e^(u/R) (with mapped layer positions)
changes the tear-film power along the 14 mm lens by < 1 % on average and shifts the
zig-zag pattern by ~3-5 um after 7 mm (checked numerically).

Absorption: complex eps (kappa of core / tear layers / cornea, as --kappa-* in the FDTD).
propagate(absorb_regions=...) books the absorbed power per layer from the local
Poynting loss w eps0 Im(eps)|E|^2/2 (absorption_rate), so the tear film's absorption
(ATR) is separated from leakage into the cornea.

Limits: one-way (no back reflection; fine for TIR guiding), 2D (no lateral spreading).
"""
import math

import numpy as np
from scipy.linalg import lapack

LAYER_SUBSAMPLES = 16


# ----------------------------------------------------------------------------- grid
def graded_grid(y_lo, y_hi, fine_zones, dy_fine, dy_coarse, grow=0.04):
    """Non-uniform nodes on [y_lo, y_hi] (um). Inside ``fine_zones`` [(a, b), ...] the
    spacing is dy_fine; it grows linearly with the distance d to the nearest zone,
    h = min(dy_coarse, dy_fine + grow*d) (grow 0.04 -> neighbouring cells differ by
    < 4 %, no spurious reflections from the grading)."""
    zones = [(min(a, b), max(a, b)) for a, b in fine_zones]

    def h(y):
        if not zones:
            return dy_coarse
        d = min(0.0 if a <= y <= b else min(abs(y - a), abs(y - b)) for a, b in zones)
        return min(dy_coarse, dy_fine + grow*d)

    ys = [y_lo]
    y = y_lo
    while y < y_hi:
        y += h(y)
        ys.append(y)
    ys = np.asarray(ys)
    return y_lo + (ys - y_lo)*(y_hi - y_lo)/(ys[-1] - y_lo)   # end exactly on y_hi


def _sample_mean(eps_of_y, a, b, S=LAYER_SUBSAMPLES, harmonic=False):
    """Mean of eps over the intervals [a_j, b_j] (arrays) with S midpoint samples."""
    t = (np.arange(S) + 0.5)/S
    Y = a[:, None] + (b - a)[:, None]*t[None, :]
    e = np.asarray(eps_of_y(Y.ravel()), dtype=np.complex128).reshape(Y.shape)
    if harmonic:
        return np.mean(1.0/e, axis=1)            # <1/eps>
    return np.mean(e, axis=1)


# ------------------------------------------------------------------------ geometry
def lens_eps_function(layers, lam_nm, curved_R_um=None, n_top=1.0, kappa=None):
    """eps(y) [y in um, y = 0 at the core bottom, core [0, t_wg]] from the resolved
    layers dict of run_simulation (same values as the FDTD): air above, core, the stack
    (lipid/aqueous/mucin or --layers) below, then the cornea (layered default or
    constant --n-cornea). curved_R_um: bend radius -> conformal mapping.
    kappa: run_simulation's args.kappa {'wg','lip','aq','mu','co'} (extinction, n + i kappa),
    applied as in the FDTD: lip/aq/mu only to the legacy three-layer stack (a --layers
    stack carries its absorption in the complex n)."""
    from common.layer_stack import stack_eps_profile
    from common.physics import cornea_default
    kap = dict(kappa or {})
    t_wg = float(layers['t_wg_um'])
    n_wg = complex(complex(layers['wg_n']).real, kap.get('wg', complex(layers['wg_n']).imag))
    stack = [dict(L) for L in (layers.get('stack') or [])]
    if not layers.get('stack_custom'):
        for L in stack:
            k = kap.get({'lipid': 'lip', 'aqueous': 'aq', 'mucin': 'mu'}.get(L.get('mat'), ''), 0.0)
            if k > 0:
                L['n'] = complex(complex(L['n']).real, k)
    k_co = kap.get('co', 0.0)
    n_co = layers.get('n_co')
    if n_co is None:
        co = cornea_default(lam_nm)
        eps_sub = lambda depth: (np.asarray(co.at_depth(np.asarray(depth)), dtype=np.complex128)
                                 + 1j*k_co)**2
    else:
        nc = complex(complex(n_co).real, k_co or complex(n_co).imag)
        eps_sub = lambda depth: np.full(np.shape(depth), nc*nc, dtype=np.complex128)

    def eps(y):
        y = np.asarray(y, dtype=np.float64)
        e = np.full(y.shape, complex(n_top)**2, dtype=np.complex128)
        e[(y >= 0.0) & (y < t_wg)] = n_wg*n_wg
        below = y < 0.0
        if np.any(below):
            e[below] = stack_eps_profile(y[below], stack, eps_sub, y_top=0.0)
        if curved_R_um:
            e = e*(1.0 + (y - t_wg/2.0)/float(curved_R_um))**2
        return e
    return eps


def stack_bottom_um(layers):
    from common.layer_stack import stack_total_um
    return -stack_total_um(layers.get('stack') or [])


# -------------------------------------------------------------------------- solver
def _pade_polys(m):
    """Pade (m, m) of sqrt(1+X) - 1 as numpy polynomial coefficient arrays (highest
    power first) N(X), D(X), from s_{k+1} = X/(2 + s_k)."""
    N = np.array([0.0]); D = np.array([1.0])
    for _ in range(2*m):
        N, D = np.polymul([1.0, 0.0], D), np.polyadd(2.0*D, N)
    return np.trim_zeros(N, 'f'), np.trim_zeros(D, 'f')


class BPM2D:
    """Wide-angle FD-BPM on a fixed cross-section.

    y_um     : node positions (non-uniform, um)
    eps_of_y : callable eps(y_um array) -> complex array (geometry, incl. curvature)
    lam_nm   : vacuum wavelength
    pol      : 's' (TE, Ez) or 'p' (TM, Hz)
    n_ref    : reference index (kb = k0 n_ref); best ~ n_core cos(beam angle)
    pml      : (thickness_bottom_um, thickness_top_um, strength); the PML lies inside
               the outermost part of y_um
    pade     : Pade order m (1 = Hadley wide angle)
    """

    def __init__(self, y_um, eps_of_y, lam_nm, pol='s', n_ref=1.48, pml=(2.0, 2.0, 3.0),
                 pade=1):
        y = np.asarray(y_um, dtype=np.float64)
        self.y = y
        self.N = len(y)
        self.lam_um = lam_nm/1000.0
        self.k0 = 2*math.pi/self.lam_um
        self.pol = 'p' if str(pol).lower() in ('p', 'tm') else 's'
        self.n_ref = float(n_ref)
        self.kb = self.k0*self.n_ref
        self.pade = int(pade)
        # --- eps averaging (dual cells / half intervals) ---
        mid = 0.5*(y[:-1] + y[1:])
        a = np.concatenate([[y[0]], mid]); b = np.concatenate([mid, [y[-1]]])
        self.eps = _sample_mean(eps_of_y, a, b)                 # node (dual cell) mean
        self._eps_of_y, self._cell = eps_of_y, (a, b)
        self.inv_eps_half = _sample_mean(eps_of_y, y[:-1], y[1:], harmonic=True)
        # --- PML: complex stretching of the spacings ---
        Lb, Lt, smax = pml
        s_mid = np.ones(len(mid), dtype=np.complex128)
        if Lb > 0:
            d = np.clip((y[0] + Lb - mid)/Lb, 0, None)
            s_mid += 1j*smax*d**2
        if Lt > 0:
            d = np.clip((mid - (y[-1] - Lt))/Lt, 0, None)
            s_mid += 1j*smax*d**2
        self.pml_mask = np.zeros(self.N, dtype=bool)
        if Lb > 0:
            self.pml_mask |= y < y[0] + Lb
        if Lt > 0:
            self.pml_mask |= y > y[-1] - Lt
        h = np.diff(y)*s_mid                                    # complex half spacings
        hl = np.concatenate([[h[0]], h]); hr = np.concatenate([h, [h[-1]]])
        w = 0.5*(hl + hr)                                       # dual widths
        self.w_real = 0.5*(np.concatenate([[y[1]-y[0]], np.diff(y)])
                           + np.concatenate([np.diff(y), [y[-1]-y[-2]]]))
        # --- operator A (tridiagonal) ---
        if self.pol == 'p':
            p = self.inv_eps_half                               # 1/eps at half nodes
            pre = self.eps                                      # eps * d/dy(1/eps d/dy)
        else:
            p = np.ones(self.N - 1, dtype=np.complex128)
            pre = np.ones(self.N, dtype=np.complex128)
        pl = np.concatenate([[p[0]], p]); pr = np.concatenate([p, [p[-1]]])
        self.A_lo = (pre[1:]*p/(w[1:]*h))                       # coefficient of u_{j-1}
        self.A_up = (pre[:-1]*p/(w[:-1]*h))                     # coefficient of u_{j+1}
        self.A_d = -pre*(pr/hr + pl/hl)/w + self.k0**2*self.eps
        # X = (A - kb^2)/kb^2
        kb2 = self.kb**2
        self.X_lo = self.A_lo/kb2; self.X_up = self.A_up/kb2; self.X_d = (self.A_d - kb2)/kb2
        self.Npoly, self.Dpoly = _pade_polys(self.pade)
        self._dx = None

    # -------------------------------------------------------------------- stepping
    def _prepare(self, dx_um):
        """Factor D -/+ (i s/2) N into roots (per step size)."""
        s = self.kb*dx_um
        Nn = np.zeros(len(self.Dpoly), dtype=np.complex128)
        Nn[len(Nn) - len(self.Npoly):] = self.Npoly
        Qm = self.Dpoly - 0.5j*s*Nn                             # left  (implicit)
        Qp = self.Dpoly + 0.5j*s*Nn                             # right (explicit)
        Qm = np.trim_zeros(Qm, 'f'); Qp = np.trim_zeros(Qp, 'f')
        self._rm = np.roots(Qm) if len(Qm) > 1 else np.array([])
        self._rp = np.roots(Qp) if len(Qp) > 1 else np.array([])
        self._c = Qp[0]/Qm[0]
        self._dx = dx_um

    def _matvec(self, u, r):
        """(X - r) u"""
        v = (self.X_d - r)*u
        v[:-1] += self.X_up*u[1:]
        v[1:] += self.X_lo*u[:-1]
        return v

    def _solve(self, v, r):
        """(X - r)^-1 v"""
        _, _, _, x, info = lapack.zgtsv(self.X_lo.copy(), (self.X_d - r).copy(),
                                        self.X_up.copy(), v)
        if info != 0:
            raise RuntimeError(f'zgtsv failed (info={info})')
        return x

    def step(self, u, dx_um):
        if self._dx != dx_um:
            self._prepare(dx_um)
        u = u*self._c
        for k in range(max(len(self._rm), len(self._rp))):
            if k < len(self._rp):
                u = self._matvec(u, self._rp[k])
            if k < len(self._rm):
                u = self._solve(u, self._rm[k])
        return u

    # ---------------------------------------------------------------- diagnostics
    def power_density(self, u):
        """Power per node (TE: |Ez|^2, TM: |Hz|^2/eps), times the dual width (um)."""
        if self.pol == 'p':
            return (np.abs(u)**2*np.real(1.0/self.eps))*self.w_real
        return np.abs(u)**2*self.w_real

    def e_intensity(self, u):
        """|E|^2 per node. TE: |Ez|^2 = |u|^2. TM (u = Hz): Ey = kx Hz/(w eps0 eps) with
        kx ~ kb, Ex = -i dHz/dy/(w eps0 eps) -> |E|^2 = (|u|^2 + |du/dy|^2/kb^2)(n_ref^2/eps)^2,
        scaled so that |E|^2 = |u|^2 in a medium with eps = n_ref^2. This is what the FDTD
        shows as |E|^2 (Ex^2 + Ey^2) and what drives surface sensing."""
        if self.pol != 'p':
            return np.abs(u)**2
        du = np.gradient(u, self.y)
        return ((np.abs(u)**2 + np.abs(du)**2/self.kb**2)
                *(self.n_ref**2/np.abs(self.eps))**2)

    def absorption_weights(self, regions):
        """Per region: Im(eps) averaged over each node's dual cell, counting only the part
        of the cell inside the region (a node on a layer boundary is split exactly)."""
        a, b = self._cell
        out = {}
        for k, (lo, hi) in regions.items():
            f = (lambda yy, lo=lo, hi=hi: np.imag(self._eps_of_y(yy))
                 *((np.asarray(yy) >= lo) & (np.asarray(yy) < hi)) + 0j)
            out[k] = np.real(_sample_mean(f, a, b))*~self.pml_mask
        return out

    def absorption_rate(self, u, weights):
        """Absorbed power per um of propagation in each mask (same units as
        power_density): Poynting, w eps0 Im(eps) |E|^2 / 2 over the flux kx |.|^2/(2 w mu0)
        with kx = kb -> TE: (k0^2/kb) Im(eps)|Ez|^2, TM: (k0^2/kb) Im(eps)|E|^2/n_ref^2 (|E|^2
        as e_intensity). Exact for the BPM's own power loss up to kx = kb."""
        I = self.e_intensity(u)*self.w_real
        c = self.k0**2/self.kb/(self.n_ref**2 if self.pol == 'p' else 1.0)
        return {k: c*float(np.dot(w, I)) for k, w in weights.items()}

    def propagate(self, u0, length_um, dx_um=0.25, rec_dx_um=1.0, regions=None,
                  probes_um=(), x0_um=0.0, save_field=True, field_planes_um=(),
                  progress=True, intensity='H', absorb_regions=None):
        """Propagate u0 over length_um.

        regions : {name: (y_lo, y_hi)} -> power fraction in each region per record
        probes_um : y positions -> |u|^2 per record (interpolated)
        save_field: keep |u|^2 on all nodes per record (float32)
        field_planes_um : x positions where the COMPLEX field is kept (hand-off)
        intensity : 'H' = |u|^2 (TM: |Hz|^2) or 'E' = |E|^2 (e_intensity) for I and the
                    probes; the region powers are always the power flux
        absorb_regions : {name: (y_lo, y_hi)} -> absorbed power per um (A_rate) and
                    accumulated since the start (A_cum) per record, if any eps is lossy
        Returns dict with x, power arrays, probe intensities, I (records x nodes)."""
        n_steps = int(round(length_um/dx_um))
        rec_every = max(1, int(round(rec_dx_um/dx_um)))
        regions = regions or {}
        masks = {k: (self.y >= a) & (self.y < b) for k, (a, b) in regions.items()}
        inside = ~self.pml_mask
        xs, P_tot, P_reg, I_rec, I_probe = [], [], {k: [] for k in masks}, [], []
        planes = sorted(float(p) for p in field_planes_um)
        fields = {}
        u = np.asarray(u0, dtype=np.complex128).copy()
        lossy = bool(absorb_regions) and bool(np.any(np.imag(self.eps)[inside] > 0))
        amasks = self.absorption_weights(absorb_regions) if lossy else {}
        A_rate = {k: [] for k in amasks}; A_cum = {k: [] for k in amasks}
        acc = {k: 0.0 for k in amasks}
        rate = self.absorption_rate(u, amasks) if lossy else {}

        def record(x, u):
            pd = self.power_density(u)
            xs.append(x)
            P_tot.append(float(pd[inside].sum()))
            for k, m in masks.items():
                P_reg[k].append(float(pd[m].sum()))
            I = self.e_intensity(u) if intensity == 'E' else np.abs(u)**2
            for k in amasks:
                A_rate[k].append(rate[k]); A_cum[k].append(acc[k])
            if probes_um:
                I_probe.append(np.interp(probes_um, self.y, I))
            if save_field:
                I_rec.append(I.astype(np.float32))

        record(x0_um, u)
        import time
        t0 = time.time(); last = t0
        for n in range(1, n_steps + 1):
            u = self.step(u, dx_um)
            x = x0_um + n*dx_um
            if lossy:                                           # trapezoid in x
                r_new = self.absorption_rate(u, amasks)
                for k in amasks:
                    acc[k] += 0.5*(rate[k] + r_new[k])*dx_um
                rate = r_new
            while planes and x >= planes[0] - 1e-9:
                fields[planes.pop(0)] = u.copy()
            if n % rec_every == 0 or n == n_steps:
                record(x, u)
            if progress and time.time() - last > 10:
                last = time.time()
                print(f'  [BPM] x = {x:9.1f} / {x0_um + length_um:.1f} um  '
                      f'({100*n/n_steps:5.1f} %, {last - t0:6.0f} s)', flush=True)
        out = dict(x_um=np.asarray(xs), P_total=np.asarray(P_tot),
                   P_region={k: np.asarray(v) for k, v in P_reg.items()},
                   u_end=u, fields=fields, y_um=self.y, eps=self.eps,
                   runtime_s=time.time() - t0)
        if lossy:
            out['A_rate'] = {k: np.asarray(v) for k, v in A_rate.items()}
            out['A_cum'] = {k: np.asarray(v) for k, v in A_cum.items()}
        if probes_um:
            out['I_probe'] = np.asarray(I_probe)
            out['probes_um'] = np.asarray(probes_um)
        if save_field:
            out['I'] = np.asarray(I_rec)
        return out


# --------------------------------------------------------------------------- input
def gaussian_input(y_um, lam_nm, y_c_um, waist_um, tilt_deg=0.0, window=None):
    """VCSEL Gaussian at the coupling facet: waist w0 (1/e^2 intensity radius), tilt in
    air (positive = towards -y, i.e. towards the tear film, same sign as ``--vcsel-tilt``
    in the FDTD). The transverse wavenumber k0 sin(tilt) is conserved across the facet,
    so the field just inside the core has the same profile; refraction follows
    automatically. window=(a, b): keep only a <= y <= b (the facet = core)."""
    k0 = 2*math.pi/(lam_nm/1000.0)
    y = np.asarray(y_um)
    u = np.exp(-((y - y_c_um)/waist_um)**2)*np.exp(-1j*k0*math.sin(math.radians(tilt_deg))
                                                  *(y - y_c_um))
    if window is not None:
        u = u*((y >= window[0]) & (y <= window[1]))
    return u.astype(np.complex128)


def vcsel_facet_input(bpm, y_c_um, waist_um, tilt_deg, distance_um=0.0, air_um=1.0,
                      n_core=1.4835, window=None, dy_um=0.005):
    """VCSEL beam entering the coupling facet, defined exactly as the FDTD source
    (common.vcsel): Gaussian with waist w0 a distance ``distance_um`` before the facet,
    given in the source plane ``air_um`` before the facet (centre y_c_um and tilt in that
    plane; air_um = vcsel.air_path_um(distance, dx), 1 um for distance 0). It is
    propagated through the air gap with the exact angular spectrum and transmitted
    through the facet (air -> core, Fresnel per plane-wave component, TE or TM).

    Returns (u, eta): field just inside the facet with power eta (coupling efficiency =
    1 - facet reflection - Gaussian tails missing the facet window; laser power = 1)."""
    from common import vcsel as V
    lam = bpm.lam_um*1000.0
    k0 = bpm.k0
    y = bpm.y
    yu = np.arange(y[0], y[-1], dy_um)
    amp, ph = V.profile_1d(yu, y_c_um, float(distance_um) - float(air_um), waist_um, lam,
                           1.0, tilt_deg)
    a0 = amp*np.exp(-1j*ph)                     # FDTD phase convention -> exp(+i(kx x - w t))
    ky = 2*math.pi*np.fft.fftfreq(len(yu), dy_um)
    A = np.fft.fft(a0)
    kx1 = np.sqrt((k0**2 - ky**2).astype(np.complex128))
    kx1 = np.where(kx1.imag < 0, -kx1, kx1)
    kx2 = np.sqrt((k0**2*n_core**2 - ky**2).astype(np.complex128))
    prop = np.abs(ky) < k0                      # propagating in air
    A = A*np.exp(1j*kx1*float(air_um))          # air gap (evanescent parts decay)
    if bpm.pol == 'p':                          # Hz: r = (eps2 kx1 - kx2)/(eps2 kx1 + kx2)
        e2 = n_core**2
        r = (e2*kx1 - kx2)/(e2*kx1 + kx2)
        T_pw = 1 - np.abs(r)**2
        t = 1 + r                               # Hz continuous
    else:
        r = (kx1 - kx2)/(kx1 + kx2)
        T_pw = 1 - np.abs(r)**2
        t = 1 + r                               # Ez continuous
    wgt = np.abs(A)**2*kx1.real*prop            # incident power per plane wave
    eta = float((wgt*T_pw).sum()/max(wgt.sum(), 1e-300))
    u = np.fft.ifft(A*t*prop)
    ur = normalize_power(bpm, np.interp(y, yu, u.real) + 1j*np.interp(y, yu, u.imag), eta)
    if window is not None:
        # only the part hitting the facet enters the core (same soft 0.2 um edge as the
        # FDTD --source-core-only); the Gaussian tails outside are lost, not renormalized
        from common.physics import core_source_mask
        ur = ur*core_source_mask(y - window[0], window[1] - window[0])
        eta = float(bpm.power_density(ur)[~bpm.pml_mask].sum())
    return ur, eta


def normalize_power(bpm, u, P=1.0):
    pd = bpm.power_density(u)
    return u*math.sqrt(P/max(float(pd[~bpm.pml_mask].sum()), 1e-300))


# --------------------------------------------------------------------------- z (3D)
# The 2D BPM treats the beam as infinitely wide in z (W per um of z). The real VCSEL beam
# has a waist w0z also in z and spreads sideways in the lens plane. Inside the lens the
# z direction is unguided (free propagation along the shell), so to good approximation
# (paraxial in z: w0z = 3 um -> 3.5 deg in PMMA) the 3D field separates:
#     E3D(x, y, z) = E2D(x, y) * g(x, z),   g = 1D Gaussian beam in z,
# with the z beam propagating along x with kx = kb = k0 n_ref (n_ref = n cos(theta)).
# Flat lens: free propagation. Curved lens (spherical shell, radius R): beams along great
# circles converge (Gaussian curvature 1/R^2, geodesic deviation) -> ABCD of a lens-like
# medium, A = D = cos(x/R), B = R sin(x/R), C = -sin(x/R)/R (refocuses after pi R).
# Not included: the small change of the zig-zag period with the z angle (kx spread
# ~0.2 %) and TE/TM mixing; both need the 3D BPM.

def z_beam_width(x_um, lam_nm, n_med, waist_z_um, distance_um=0.0, R_um=None):
    """1/e^2 radius w_z(x) [um] of the beam in z inside the lens, x from the facet.
    Waist w0z a distance ``distance_um`` before the facet (in air); n_med = n_ref of the
    BPM (kx/k0). R_um: sphere radius (None = flat)."""
    lam = lam_nm/1000.0
    x = np.asarray(x_um, dtype=np.float64)
    zr_air = math.pi*waist_z_um**2/lam
    q_in = n_med*complex(float(distance_um), zr_air)          # facet: q -> n q (paraxial)
    if R_um:
        a = np.cos(x/R_um); b = R_um*np.sin(x/R_um); c = -np.sin(x/R_um)/R_um
        q = (a*q_in + b)/(c*q_in + a)
    else:
        q = q_in + x
    return np.sqrt(-lam/(math.pi*n_med*np.imag(1.0/q)))


def z_peak_factor(w_z_um):
    """On-axis (z = 0) weight of the z profile [1/um]: |g(0)|^2/int |g|^2 dz =
    sqrt(2/pi)/w_z. 3D irradiance-like quantity at z = 0:
        I3D(x, y) [W/um^2] = P_laser [W] * I2D(x, y) [1/um] * z_peak_factor(w_z(x))."""
    return math.sqrt(2.0/math.pi)/np.asarray(w_z_um, dtype=np.float64)
