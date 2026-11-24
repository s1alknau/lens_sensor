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
# n(lambda) als Formeln MIT Quelle (Review F5: die alte Tabelle hatte bei "850 nm"
# Werte aus dem Sichtbaren, z.B. PMMA 1.491 = n bei 587.6 nm). lambda in um.
#
#  pmma       Sultanova, Kasarova, Nikolov, Acta Phys. Pol. A 116, 585 (2009),
#             doi:10.12693/APhysPolA.116.585; Sellmeier 1-Term, 0.437-1.052 um, 20 C.
#  polystyrol ebenda (Sultanova 2009), 0.437-1.052 um.
#  water      Daimon & Masumura, Appl. Opt. 46, 3811 (2007), doi:10.1364/AO.46.003811;
#             Sellmeier 4-Term, 20 C, 0.182-1.129 um.
#  aqueous    Traenenfluessigkeit = Wasser + 0.0035 (Craig et al., Optom. Vis. Sci. 1995,
#             doi:10.1002/j.1538-9235.1995.tb03963.x: Traenen 1.337 im Sichtbaren vs.
#             Wasser 1.3334). ANNAHME: Offset wellenlaengenunabhaengig.
#  cornea     1.376 im Sichtbaren (Gullstrand/Le-Grand-Augenmodell, 589 nm); Dispersion
#             wie Wasser (ANNAHME, ueberwiegend waesseriges Gewebe).
#  mucin      Kein direkter Messwert fuer die Mucinschicht gefunden (Review F5; der alte
#             Wert 1.342 war ohne Quelle). Abgeleitet: Mucus = Traenen + ~0.5-1 % Mucin
#             (Gewicht/Volumen); Brechzahl-Inkrement von (Glyko-)Proteinen dn/dc ~
#             0.18-0.19 mL/g (Zhao, Brown, Schuck, Biophys. J. 100, 2309 (2011),
#             doi:10.1016/j.bpj.2011.03.004) -> dn = 0.01 g/mL * 0.18 mL/g = +0.0018 ueber
#             den Traenen. Obere Abschaetzung (1 %); Dispersion wie Traenen.
#  lipid      Meibum 1.48 im Sichtbaren (Tiffany, Curr. Eye Res. 1986,
#             doi:10.3109/02713688609029242); Dispersion wie PMMA (ANNAHME, organisch).
# Ergebnis bei 850 nm: PMMA 1.4835, PS 1.5764, Wasser 1.3278, Traenen 1.3313,
# Cornea 1.3704, Mucin 1.3331, Lipid 1.4729 (vorher 1.491/1.590/-/1.336/1.376/1.342/1.480).
LAM_VIS_UM = 0.5893                     # Na-D-Linie: Bezugswellenlaenge der Literaturwerte
MUCIN_DN = 0.0018                       # Mucus ueber Traenen: 1 % Mucin x dn/dc 0.18 mL/g


def _sellmeier(lam_um, terms):
    l2 = lam_um*lam_um
    return float(np.sqrt(1.0 + sum(B*l2/(l2 - C) for B, C in terms)))


_SELLMEIER = {
    'pmma':       ((1.1819, 0.011313),),
    'polystyrol': ((1.4435, 0.020216),),
    'water':      ((5.684027565e-1, 5.101829712e-3), (1.726177391e-1, 1.821153936e-2),
                   (2.086189578e-2, 2.620722293e-2), (1.130748688e-1, 1.069792721e1)),
}


def _n_water(lam_um):
    return _sellmeier(lam_um, _SELLMEIER['water'])


def _n_pmma(lam_um):
    return _sellmeier(lam_um, _SELLMEIER['pmma'])


def _shifted(n_vis, base):
    """n(lambda) = n_vis + [base(lambda) - base(lambda_vis)] (Dispersion der Basis)."""
    return lambda l: n_vis + base(l) - base(LAM_VIS_UM)


# Material -> n(lambda_um)
MATERIAL_MODELS = {
    'pmma':       _n_pmma,
    'polystyrol': lambda l: _sellmeier(l, _SELLMEIER['polystyrol']),
    'water':      _n_water,
    'aqueous':    lambda l: _n_water(l) + 0.0035,
    'cornea':     _shifted(1.376, _n_water),
    'mucin':      lambda l: _n_water(l) + 0.0035 + MUCIN_DN,
    'lipid':      _shifted(1.480, _n_pmma),
    # Cornea-Schichten (Patel, Marshall & Fitzke, J. Refract. Surg. 11, 100 (1995):
    # Epithel 1.401, Stroma vorn 1.380 / hinten 1.373; Kammerwasser 1.336 Gullstrand),
    # alle im Sichtbaren gemessen -> Dispersion wie Wasser (ANNAHME).
    'epithelium':    _shifted(1.401, _n_water),
    'stroma':        _shifted(1.3765, _n_water),       # Mittel aus vorn/hinten
    'aqueous_humor': _shifted(1.336, _n_water),
}

# Gueltigkeitsbereich der Fits (um); ausserhalb wird mit Warnung extrapoliert.
_VALID_UM = (0.437, 1.052)

# Kompatibilitaet: DISPERSION (Materialliste/Tabelle) bleibt verfuegbar, jetzt aus den
# Formeln abgeleitet (532/850/940 nm).
DISPERSION = {m: {w: round(f(w/1000.0), 4) for w in (532, 850, 940)}
              for m, f in MATERIAL_MODELS.items()}

_warned = set()


def n_at(mat, lam_nm):
    """Brechzahl von ``mat`` bei Wellenlaenge ``lam_nm`` (Dispersionsformel, s.o.)."""
    lam_um = float(lam_nm)/1000.0
    if not (_VALID_UM[0] <= lam_um <= _VALID_UM[1]) and (mat, lam_nm) not in _warned:
        _warned.add((mat, lam_nm))
        print(f'[physics] Note: {mat} at {lam_nm:g} nm is outside the fit range '
              f'{_VALID_UM[0]*1000:.0f}-{_VALID_UM[1]*1000:.0f} nm (extrapolated).')
    return float(MATERIAL_MODELS[mat](lam_um))


N_TEAR_SCENARIO_REF = 1.336            # 'normal' in den Szenario-Tabellen


def tear_index_at(n_tear_vis, lam_nm):
    """Traenen-Brechzahl eines Szenarios bei lam_nm. Die Szenariowerte (1.336 normal,
    1.340 hyperosmolar) sind Werte im Sichtbaren; gezaehlt wird ihr Abstand zum normalen
    Traenenfilm (1.336), aufgesetzt auf das Dispersionsmodell 'aqueous'."""
    return float(n_at('aqueous', lam_nm) + (n_tear_vis - N_TEAR_SCENARIO_REF))



# ---------- Cornea als Schichtung (statt halbunendlich mit einem n) ----------
# Tiefe d ab der Grenze Mucin/Cornea (um), zentrale Cornea ~545 um (Werte im Sichtbaren):
CORNEA_LAYERS_UM = dict(epithelium=50.0, bowman=12.0, stroma_end=525.0, endothelium_end=545.0)
_N_EPI, _N_STROMA_ANT, _N_STROMA_POST, _N_HUMOR = 1.401, 1.380, 1.373, 1.336


class CorneaProfile(float):
    """Brechzahl der Cornea als Funktion der Tiefe (Review-Folge: Cornea nicht als ein
    halbunendliches n). Der float-Wert ist das Epithel (die Schicht direkt unter dem
    Mucin) - so funktionieren Ausgaben/Tabellen unveraendert; die Builder fragen
    at_depth() ab. Schichten: Epithel 0-50 um (1.401), Bowman bis 62 um (1.380),
    Stroma bis 525 um (linear 1.380 -> 1.373), Descemet/Endothel bis 545 um (1.373),
    darunter Kammerwasser (1.336). Alle Werte + Wasser-Dispersion auf lambda."""
    def __new__(cls, lam_nm):
        obj = float.__new__(cls, n_at('epithelium', lam_nm))
        obj.lam_nm = float(lam_nm)
        return obj

    def __reduce__(self):
        return (CorneaProfile, (self.lam_nm,))

    def at_depth(self, d_um):
        d = np.asarray(d_um, dtype=np.float64)
        L = CORNEA_LAYERS_UM
        b0 = L['epithelium']; b1 = b0 + L['bowman']
        n = np.full(d.shape, _N_HUMOR)
        n = np.where(d < L['endothelium_end'], _N_STROMA_POST, n)
        st = (d >= b1) & (d < L['stroma_end'])
        frac = np.clip((d - b1)/(L['stroma_end'] - b1), 0.0, 1.0)
        n = np.where(st, _N_STROMA_ANT + (_N_STROMA_POST - _N_STROMA_ANT)*frac, n)
        n = np.where((d >= b0) & (d < b1), _N_STROMA_ANT, n)
        n = np.where(d < b0, _N_EPI, n)
        lam_um = self.lam_nm/1000.0
        return n + _n_water(lam_um) - _n_water(LAM_VIS_UM)


def cornea_default(lam_nm):
    """Standard-Cornea (geschichtet) bei lam_nm; ein explizites --n-cornea ersetzt sie
    durch ein konstantes n."""
    return CorneaProfile(lam_nm)


def cornea_full_depth_um(margin_um=5.0):
    """Tiefe ab der Grenze Mucin/Cornea bis ins Kammerwasser (Cornea-Rueckseite +
    margin), fuer --cornea-full."""
    return CORNEA_LAYERS_UM['endothelium_end'] + float(margin_um)


# ---------- Absorption: komplexe Brechzahl n = n' + i*kappa ----------
# Modell: bei der Betriebswellenlaenge lambda0 wird
#     eps = (n' + i kappa)^2 = (n'^2 - kappa^2) + i * 2 n' kappa
# als reelles eps' = n'^2 - kappa^2 plus Leitfaehigkeit sigma = omega0 eps0 * 2 n' kappa
# umgesetzt (FDTD: J = sigma E). Bei lambda0 EXAKT; fuer breitbandige Pulse ist der
# Verlust ~1/omega statt konstant (Naeherung). Intensitaet klingt im Volumenmaterial wie
# exp(-alpha z) ab, alpha = 4 pi kappa / lambda0. kappa > n' (Metalle) braucht ein
# Drude-Modell und wird abgelehnt.
LOSS_KEYS = ('wg', 'lip', 'aq', 'mu', 'co', 'bead')


def parse_n(s):
    """Brechzahl aus Text: '1.52' -> 1.52, '1.33+0.001i' / '1.33+1e-3j' -> complex.
    Imaginaerteil = Extinktionskoeffizient kappa (>= 0, absorbierend)."""
    if isinstance(s, (int, float, complex)):
        v = s
    else:
        t = str(s).strip().lower().replace(' ', '').replace('i', 'j')
        if t.startswith('n='):
            t = t[2:]
        v = complex(t) if 'j' in t else float(t)
    if isinstance(v, complex):
        if v.imag < 0:
            raise ValueError(f'kappa < 0 ({s}) would mean gain - convention: n = n\' + i*kappa, kappa >= 0')
        if v.imag >= v.real:
            raise ValueError(f'kappa >= n\' ({s}): metallic, requires a Drude model (not supported)')
        if v.imag == 0:
            return float(v.real)
    return v


def split_n(v):
    """(n', kappa) aus reeller oder komplexer Brechzahl (None bleibt None)."""
    if v is None:
        return None, 0.0
    if isinstance(v, complex):
        return float(v.real), float(v.imag)
    return v, 0.0


def fmt_n(v, k=0.0, prec=4):
    """Brechzahl fuer Ausgaben, mit Absorption als 'n+ik'."""
    n, kk = split_n(v)
    kk = kk or k
    return f'{n:.{prec}f}' + (f'+{kk:.3g}i' if kk else '')


def loss_sigma(eps_imag, lam_m):
    """Leitfaehigkeit [S/m] zu Im(eps_r) bei lambda0: sigma = omega0 eps0 eps''."""
    return 2*np.pi*C0/lam_m*EPS0*np.asarray(eps_imag)


def alpha_intensity(kappa, lam_m):
    """Intensitaets-Absorptionskoeffizient im Volumen [1/m]: 4 pi kappa / lambda."""
    return 4*np.pi*kappa/lam_m


# ---------------------------------------------------------------------------
# Quelle nur im Kern (--source-core-only)
# ---------------------------------------------------------------------------
SRC_CORE_RAMP_UM = 0.2      # weiche Kante (cos^2) ausserhalb des Kerns: harter Schnitt
                            # beugt selbst Licht in steile Winkel


def core_source_mask(y_um, t_wg_um, ramp_um=SRC_CORE_RAMP_UM, z_um=None, wg_width_um=None):
    """Gewicht der Quelle, wenn sie nur durch den Kern eintreten soll: 1 im Kern
    (0 <= y <= t_wg; im 3D-Kanal zusaetzlich |z| <= w/2), cos^2-Rampe ueber ramp_um
    nach aussen, sonst 0. Ohne Maske strahlt der Gauss-Ausläufer der Quelle direkt in
    Luft und Traenenfilm (Untergrund unter dem evaneszenten Feld)."""
    def ramp(d):
        d = np.asarray(d, dtype=np.float64)
        return np.where(d <= 0.0, 1.0,
                        np.where(d < ramp_um, np.cos(0.5*np.pi*d/max(ramp_um, 1e-9))**2, 0.0))
    y = np.asarray(y_um, dtype=np.float64)
    m = ramp(np.maximum(-y, y - float(t_wg_um)))
    if z_um is not None and wg_width_um:
        z = np.asarray(z_um, dtype=np.float64)
        m = m*ramp(np.abs(z) - 0.5*float(wg_width_um))
    return m
