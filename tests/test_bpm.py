"""Fast checks of the 2D wide-angle BPM (common/bpm.py), ~45 s in total.

1. An eigenmode of a thin layered slab propagates unchanged: |amplitude| = 1 (power
   conserving) and phase = (beta - kb) L (TE and TM).
2. A tilted Gaussian is refracted at the facet by Snell's law (beam centroid angle).
3. On the full lens cross-section the BPM field after 1 mm matches an independent
   guided-mode expansion (kontaktlinse/mode_propagation.solve_modes).
4. The VCSEL facet input transmits a wide tilted beam with the analytic Fresnel
   coefficient (TE and TM), and the centre lands where geometry puts it.
5. The analytic z spreading (z_beam_width, flat and spherical shell) agrees with a
   numerical propagation of the z profile (flat: homogeneous; sphere: the geodesic
   convergence as an index n^2 (1 - z^2/R^2)).
6. Absorption (ATR): an eigenmode of a slab with an absorbing aqueous layer loses power
   at 2 Im(beta) of the complex eigenvalue, and the absorbed power booked per layer
   (absorption_rate / A_cum) accounts for that loss (TE and TM).

    python -m pytest tests/test_bpm.py -q      (or: python tests/test_bpm.py)
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, 'kontaktlinse')):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.bpm import (BPM2D, graded_grid, gaussian_input, normalize_power,  # noqa: E402
                        vcsel_facet_input, z_beam_width)

LAM = 850.0
K0 = 2*math.pi/(LAM/1000.0)


def _slab_eps(t_core, k_aq=0.0):
    def eps(y):
        y = np.asarray(y)
        e = np.full(y.shape, 1.0 + 0j)
        e[(y >= 0) & (y < t_core)] = 1.483**2
        e[(y < 0) & (y >= -0.03)] = 1.4729**2
        e[(y < -0.03) & (y >= -3.53)] = (1.3313 + 1j*k_aq)**2
        e[y < -3.53] = 1.3954**2
        return e
    return eps


def test_eigenmode_phase_and_power():
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    y = graded_grid(-8, 10, [(-4, 0.2), (4.8, 5.2)], 0.005, 0.025)
    for pol in ('s', 'p'):
        b = BPM2D(y, _slab_eps(5.0), LAM, pol=pol, n_ref=1.47, pml=(2, 2, 3), pade=1)
        A = sp.diags([b.A_lo, b.A_d, b.A_up], [-1, 0, 1]).tocsc()
        vals, vecs = sla.eigs(A, k=2, sigma=(K0*1.483)**2)
        o = np.argsort(-vals.real)
        beta, u0 = np.sqrt(vals[o[0]]), vecs[:, o[0]]
        L = 200.0
        u1 = b.propagate(u0, L, dx_um=0.5, rec_dx_um=L, save_field=False, progress=False)['u_end']
        ph = np.vdot(u0, u1)/np.vdot(u0, u0)
        err = np.angle(ph*np.exp(-1j*(beta - b.kb)*L))
        assert abs(abs(ph) - 1.0) < 1e-6, (pol, abs(ph))
        assert abs(err/(K0*L)) < 1e-5, (pol, err)                # delta n_eff < 1e-5


def test_snell_refraction():
    y = graded_grid(-12, 257, [(-4.5, 0.5), (249.5, 250.5)], 0.005, 0.010)

    def eps(yy):
        yy = np.asarray(yy)
        e = np.full(yy.shape, 1.0 + 0j)
        e[(yy >= 0) & (yy < 250)] = 1.4835**2
        e[yy < 0] = 1.3313**2
        return e
    tilt = 20.0
    th = math.asin(math.sin(math.radians(tilt))/1.4835)
    b = BPM2D(y, eps, LAM, pol='s', n_ref=1.4835*math.cos(th), pade=1)
    u0 = normalize_power(b, gaussian_input(y, LAM, 125.0, 3.0, tilt, window=(0, 250)))
    r = b.propagate(u0, 250.0, dx_um=0.1, rec_dx_um=50.0, progress=False)
    m = (y > 0) & (y < 250); w = np.gradient(y)[m]
    c = [(row[m]*y[m]*w).sum()/(row[m]*w).sum() for row in r['I'].astype(float)]
    x = r['x_um']
    slope = (c[1] - c[-1])/(x[-1] - x[1])
    ang = math.degrees(math.atan(slope))
    assert abs(ang - math.degrees(th)) < 0.05, (ang, math.degrees(th))


def test_lens_vs_mode_expansion():
    from mode_propagation import solve_modes
    t = 250.0
    eps = _slab_eps(t)
    y = graded_grid(-12, t + 7, [(-4.5, 0.5), (t - 0.5, t + 0.5)], 0.005, 0.010)
    b = BPM2D(y, eps, LAM, pol='s', n_ref=1.4835, pade=1)
    u0 = normalize_power(b, gaussian_input(y, LAM, t/2, 3.0, 0.0, window=(0, t)))
    L = 1000.0
    uB = b.propagate(u0, L, dx_um=0.25, rec_dx_um=L, save_field=False,
                     progress=False)['u_end']*np.exp(1j*b.kb*L)
    dy = 0.005
    yu = np.arange(-10.0, t + 5.0, dy)
    neff, prof = solve_modes(yu, np.sqrt(eps(yu)).real, LAM/1000.0, 's')
    g = np.interp(yu, y, u0.real) + 1j*np.interp(yu, y, u0.imag)
    a = (prof*g[:, None]).sum(0)*dy
    uM = (prof*(a*np.exp(1j*neff*K0*L))[None, :]).sum(1)
    IB = np.interp(yu, y, np.abs(uB)**2); IM = np.abs(uM)**2
    core = (yu >= 0) & (yu < t)
    assert np.corrcoef(IB[core], IM[core])[0, 1] > 0.999
    eB = np.interp(-0.5, yu, IB)/IB[core].sum(); eM = np.interp(-0.5, yu, IM)/IM[core].sum()
    assert abs(eB/eM - 1) < 0.1, (eB, eM)                       # evanescent field +-10 %


def test_facet_fresnel():
    n = 1.4835
    y = graded_grid(-12, 257, [(-4.5, 0.5), (249.5, 250.5)], 0.005, 0.020)
    for pol in ('s', 'p'):
        b = BPM2D(y, _slab_eps(250.0), LAM, pol=pol, n_ref=n, pade=1)
        for tilt in (0.0, 20.0):
            u, eta = vcsel_facet_input(b, 125.0, 40.0, tilt, 0.0, 1.0, n, window=(0, 250))
            c1 = math.cos(math.radians(tilt)); c2 = math.sqrt(1 - (math.sin(math.radians(tilt))/n)**2)
            r = (c1 - n*c2)/(c1 + n*c2) if pol == 's' else (n*c1 - c2)/(n*c1 + c2)
            assert abs(eta - (1 - r*r)) < 2e-4, (pol, tilt, eta, 1 - r*r)
            assert abs(float(b.power_density(u)[~b.pml_mask].sum()) - eta) < 1e-6
            # centre: defined 1 um before the facet, moves 1 um * tan(tilt) towards -y
            I = np.abs(u)**2
            yc = (I*y).sum()/I.sum()
            assert abs(yc - (125.0 - math.tan(math.radians(tilt)))) < 0.02, (pol, tilt, yc)


def test_z_spreading():
    n, w0, R, L = 1.4435, 3.0, 8300.0, 7000.0
    z = graded_grid(-2500, 2500, [], 0.25, 0.25)
    for Rm in (None, R):
        def eps(zz, Rm=Rm):
            zz = np.asarray(zz)
            return (n**2*(1 - (zz/Rm)**2) if Rm else np.full(zz.shape, n**2)) + 0j
        b = BPM2D(z, eps, LAM, pol='s', n_ref=n, pml=(300, 300, 3.0), pade=1)
        u0 = np.exp(-(z/w0)**2)*np.sqrt(n) + 0j          # waist at the facet, in the medium
        r = b.propagate(u0, L, dx_um=2.0, rec_dx_um=L/2, progress=False)
        for x, I in zip(r['x_um'][1:], r['I'][1:].astype(float)):
            w_num = 2*np.sqrt((I*z**2).sum()/I.sum())     # 1/e^2 radius = 2 sigma
            w_ana = z_beam_width(x, LAM, n, w0, 0.0, Rm)
            assert abs(w_num/w_ana - 1) < 0.01, (Rm, x, w_num, w_ana)
    assert abs(z_beam_width(math.pi*R, LAM, n, w0, 0.0, R) - w0) < 1e-6   # refocus at pi R


def test_absorption_bookkeeping():
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    y = graded_grid(-8, 10, [(-4, 0.2), (4.8, 5.2)], 0.005, 0.025)
    for pol in ('s', 'p'):
        b = BPM2D(y, _slab_eps(5.0, k_aq=0.02), LAM, pol=pol, n_ref=1.47, pml=(2, 2, 3), pade=1)
        A = sp.diags([b.A_lo, b.A_d, b.A_up], [-1, 0, 1]).tocsc()
        vals, vecs = sla.eigs(A, k=2, sigma=(K0*1.483)**2)
        o = np.argsort(-vals.real)
        beta, u0 = np.sqrt(vals[o[0]]), vecs[:, o[0]]
        L = 200.0
        reg = {'lipid': (-0.03, 0.0), 'aqueous': (-3.53, -0.03), 'core': (0.0, 5.0)}
        r = b.propagate(u0, L, dx_um=0.5, rec_dx_um=L, save_field=False, progress=False,
                        absorb_regions=reg)
        P0, P1 = r['P_total']
        loss = 1 - P1/P0
        assert abs(loss/(1 - math.exp(-2*beta.imag*L)) - 1) < 0.01, (pol, loss, beta.imag)
        A_tot = sum(v[-1] for v in r['A_cum'].values())
        assert abs(A_tot/(P0 - P1) - 1) < 0.02, (pol, A_tot, P0 - P1)
        assert r['A_cum']['core'][-1] == 0.0 and r['A_cum']['aqueous'][-1] > 0


if __name__ == '__main__':
    for f in (test_eigenmode_phase_and_power, test_snell_refraction, test_lens_vs_mode_expansion,
              test_facet_fresnel, test_z_spreading, test_absorption_bookkeeping):
        f(); print('OK', f.__name__)
