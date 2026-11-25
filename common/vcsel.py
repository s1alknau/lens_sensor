"""Gemeinsames VCSEL-Quellmodell fuer alle Solver (2D, 3D, Meep).

Physikalisches Bild
-------------------
Der VCSEL sitzt in Luft VOR der Einkoppelflaeche (Stirnflaeche/Facette) des Wellenleiters
und strahlt einen Gauss-Strahl (Grundmode TEM00, M^2 = 1) in Richtung +x ab. Die Taille
w0 (1/e^2-Radius der Intensitaet) liegt am Emitter, im Abstand ``distance_um`` vor der
Facette. Standard ``distance_um = 0``: die Taille - und damit die EBENE Wellenfront -
liegt genau auf der Einkoppelflaeche.

Den Weg durch die Luft, die Fresnel-Reflexion an der Facette und die Kopplung in die
Wellenleitermoden rechnet das FDTD selbst (rigoros). Eingepraegt wird nur das Feld des
Gauss-Strahls in der Quellebene, die immer etwas vor der Facette in Luft liegt:

    z = x_quelle - x_taille  (z < 0: Taille stromab -> konvergente Wellenfront)
    w(z) = w0 sqrt(1 + (z/zR)^2),  zR = pi n w0^2 / lambda,  R(z) = z + zR^2/z
    E(y) ~ sqrt(w0/w) exp(-(y-yc)^2/w^2) sin(w t - k (y-yc)^2/(2R) + k n sin(theta) (y-yc))
    (3D: Faktor w0/w und Gauss in y und z)

Die Quelle strahlt nur VORWAERTS (Huygens-Paar): zwei weiche Quellen in benachbarten
Ebenen ix, ix+1, die zweite mit Amplitude -1 und um k~ dx vorauseilender Phase. Die
rueckwaerts laufenden Anteile loeschen sich aus (exakt fuer die Gitterdispersion k~ in
x-Richtung). Dadurch ist die abgestrahlte Leistung eindeutig messbar:

    P_emit = S_x(Ebene hinter dem Paar) - S_x(Ebene vor dem Paar)

(Poynting-Satz um die Quelle; die zweite Ebene sieht nur das an der Facette reflektierte
Licht.) Mit P_emit wird auf die Laserleistung ``power_mw`` skaliert (FDTD ist linear):
Felder ~ sqrt(P_laser/P_emit).

Grenzen: kohaerenter TEM00-Strahl (M^2 = 1, Einmoden-VCSEL); in 2D ist es eine
Linienquelle (in z unendlich ausgedehnt, keine Divergenz in z).
"""
import math

import numpy as np

from common.backend import xp

C0 = 299792458.0

# Typischer Einmoden-VCSEL bei 850 nm (Standardwerte fuer CLI und GUI)
VCSEL_DEFAULTS = dict(
    lambda_nm=850.0,     # Emissionswellenlaenge
    waist_um=3.0,        # Strahltaille w0 (1/e^2-Radius der Intensitaet) -> ~10 deg Divergenz
    distance_um=0.0,     # Abstand Taille (Emitter) -> Einkoppelflaeche; 0 = Taille auf der Facette
    power_mw=20.0,       # optische Ausgangsleistung
    tilt_deg=0.0,        # Strahlneigung gegen die x-Achse
)

# Mindest-Luftweg zwischen Quellebene und Facette (Platz fuer Huygens-Paar + Messebene)
MIN_AIR_UM = 1.0
MIN_AIR_CELLS = 8


def rayleigh_um(waist_um, lambda_nm, n=1.0):
    """Rayleigh-Laenge zR = pi n w0^2 / lambda [um]."""
    return math.pi*n*waist_um**2/(lambda_nm*1e-3)


def divergence_deg(waist_um, lambda_nm, n=1.0):
    """Voller Divergenzwinkel (1/e^2) im Fernfeld: 2 atan(lambda/(pi n w0)) [deg]."""
    return 2.0*math.degrees(math.atan((lambda_nm*1e-3)/(math.pi*n*waist_um)))


def waist_from_divergence(full_angle_deg, lambda_nm, n=1.0):
    """Umkehrung von divergence_deg: w0 aus dem vollen 1/e^2-Divergenzwinkel [um]."""
    return (lambda_nm*1e-3)/(math.pi*n*math.tan(math.radians(full_angle_deg)/2.0))


def air_path_um(distance_um, dx_um):
    """Luftweg Quellebene -> Facette [um]. Ist der Emitter weiter weg als der Mindestweg,
    sitzt die Quellebene AM Emitter (Taille in der Quellebene); sonst sitzt sie den
    Mindestweg vor der Facette und praegt den (konvergenten) Strahl ein, dessen Taille
    stromab im Abstand distance_um vor der Facette liegt."""
    s_min = max(MIN_AIR_UM, MIN_AIR_CELLS*float(dx_um))
    return max(float(distance_um or 0.0), s_min)


def z_at_source(distance_um, air_um):
    """Ausbreitungskoordinate z der Quellebene relativ zur Taille [um]:
    Taille bei x_facette - distance, Quelle bei x_facette - air -> z = distance - air."""
    return float(distance_um or 0.0) - float(air_um)


def beam_params(z_um, waist_um, lambda_nm, n=1.0):
    """(w(z), R(z)) des Gauss-Strahls [um]; R = inf in der Taille."""
    zr = rayleigh_um(waist_um, lambda_nm, n)
    w = waist_um*math.sqrt(1.0 + (z_um/zr)**2)
    R = math.inf if abs(z_um) < 1e-12 else z_um + zr**2/z_um
    return w, R


def _phase_1d(u_um, R_um, lambda_nm, n, tilt_deg):
    k = 2*math.pi*n/(lambda_nm*1e-3)                         # [1/um]
    ph = k*math.sin(math.radians(tilt_deg))*u_um             # Neigung (wie bisher)
    if math.isfinite(R_um):
        ph = ph - k*u_um**2/(2.0*R_um)                       # Wellenfrontkruemmung
    return ph


def profile_1d(y_um, yc_um, z_um, waist_um, lambda_nm, n=1.0, tilt_deg=0.0):
    """2D (Linienquelle): Amplitude und Phase [rad] auf den Punkten y_um (NumPy)."""
    y = np.asarray(y_um, dtype=np.float64) - yc_um
    w, R = beam_params(z_um, waist_um, lambda_nm, n)
    amp = math.sqrt(waist_um/w)*np.exp(-(y/w)**2)
    return amp, _phase_1d(y, R, lambda_nm, n, tilt_deg)


def profile_2d(y_um, z_um_grid, yc_um, zc_um, zprop_um, waist_y_um, waist_z_um, lambda_nm,
               n=1.0, tilt_deg=0.0):
    """3D (Flaechenquelle): Amplitude/Phase auf dem Gitter (y_um[:,None], z_um_grid[None,:]).
    Astigmatisch erlaubt (eigene Taillen in y und z, gemeinsame Taillenlage); die Neigung
    wirkt in der x-y-Ebene (wie bisher)."""
    y = np.asarray(y_um, dtype=np.float64)[:, None] - yc_um
    zz = np.asarray(z_um_grid, dtype=np.float64)[None, :] - zc_um
    wy, Ry = beam_params(zprop_um, waist_y_um, lambda_nm, n)
    wz, Rz = beam_params(zprop_um, waist_z_um, lambda_nm, n)
    amp = math.sqrt((waist_y_um/wy)*(waist_z_um/wz))*np.exp(-(y/wy)**2 - (zz/wz)**2)
    ph = _phase_1d(y, Ry, lambda_nm, n, tilt_deg) + np.zeros_like(zz)
    if math.isfinite(Rz):
        ph = ph - (2*math.pi*n/(lambda_nm*1e-3))*zz**2/(2.0*Rz)
    return amp, ph


def numerical_kx_dx(lambda_nm, dx_m, dt_m_per_c, n=1.0, tilt_deg=0.0, dims=2):
    """Phasenvorschub k~_x*dx pro Zelle in x fuer das Huygens-Paar, aus der Yee-Dispersion
    entlang der Achse: sin(k~ dx/2) = (n dx/(c dt)) sin(w dt/2); mit Neigung k~ cos(theta).
    dt_m_per_c = c*dt [m]."""
    w_dt_2 = math.pi*dt_m_per_c/(lambda_nm*1e-9)             # w dt / 2
    arg = min(1.0, n*dx_m/dt_m_per_c*math.sin(w_dt_2))
    kdx = 2.0*math.asin(arg)
    return kdx*math.cos(math.radians(tilt_deg))


class HuygensSource:
    """Einseitig abstrahlende weiche Quelle auf den Ebenen ix und ix+1 (siehe Modulkopf).
    amp/phase: Profil in der Quellebene (Form wie F[ix]). Die Amplitude wird durch
    2 sin(k~dx) geteilt, damit die vorwaerts laufende Welle dieselbe Groesse hat wie die
    einer einzelnen weichen Quelle (Felder bleiben in gewohnten Einheiten)."""

    def __init__(self, ix, amp, phase, omega, kx_dx):
        self.ix = int(ix)
        g = 2.0*abs(math.sin(kx_dx)) or 1.0
        self.amp = xp.asarray(np.asarray(amp)/g, dtype=xp.float32)
        self.ph1 = xp.asarray(np.asarray(phase), dtype=xp.float32)
        self.ph2 = xp.asarray(np.asarray(phase) + kx_dx, dtype=xp.float32)
        self.omega = float(omega)
        self.kx_dx = float(kx_dx)

    def inject(self, F, t_phys, envelope, ix_offset=0):
        """Nach dem E-Update aufrufen. ix_offset: globaler -> lokaler Index (Fenster)."""
        i = self.ix - ix_offset
        wt = self.omega*t_phys
        F[i] += envelope*xp.sin(np.float32(wt) + self.ph1)*self.amp
        F[i + 1] -= envelope*xp.sin(np.float32(wt) + self.ph2)*self.amp

    @property
    def plane_up(self):
        """Messebene vor dem Paar (sieht nur Reflexion)."""
        return self.ix - 1

    @property
    def plane_dn(self):
        """Messebene hinter dem Paar (vorwaerts - reflektiert)."""
        return self.ix + 3

    def to_ey(self):
        """Gleiche Quelle auf dem TM-Ey-Gitter (Mittel benachbarter y-Knoten)."""
        s = HuygensSource.__new__(HuygensSource)
        s.ix, s.omega, s.kx_dx = self.ix, self.omega, self.kx_dx
        s.amp = 0.5*(self.amp[:-1] + self.amp[1:])
        s.ph1 = 0.5*(self.ph1[:-1] + self.ph1[1:])
        s.ph2 = 0.5*(self.ph2[:-1] + self.ph2[1:])
        return s


def describe(waist_um, lambda_nm, distance_um, air_um, power_mw, waist_z_um=None):
    """Einzeilige Zusammenfassung fuers Log."""
    z = z_at_source(distance_um, air_um)
    w, R = beam_params(z, waist_um, lambda_nm)
    wz = '' if waist_z_um is None or waist_z_um == waist_um else f'/{waist_z_um:g}'
    return (f'VCSEL {lambda_nm:g} nm: w0={waist_um:g}{wz} um (div. {divergence_deg(waist_um, lambda_nm):.1f} deg '
            f'full 1/e^2, zR={rayleigh_um(waist_um, lambda_nm):.1f} um), waist {distance_um or 0:g} um before facet, '
            f'source plane {air_um:.2f} um before facet (w={w:.2f} um, '
            f'R={"inf" if not math.isfinite(R) else f"{R:.1f} um"}), P={power_mw:g} mW, forward-only (Huygens)')


NPZ_KEYS = ('laser_power_mw', 'p_src_si', 'coupling_efficiency', 'facet_reflection',
            'vcsel_distance_um', 'vcsel_air_um', 'vcsel_waist_um', 'vcsel_waist_z_um',
            'x_facet_um')


def npz_keys(result):
    """VCSEL-Kennwerte eines Ergebnis-Dicts als NPZ-Eintraege (nur vorhandene, endliche)."""
    out = {}
    for k in NPZ_KEYS:
        v = result.get(k)
        try:
            if v is not None and np.isfinite(float(v)):
                out[k] = np.array([float(v)])
        except (TypeError, ValueError):
            pass
    return out
