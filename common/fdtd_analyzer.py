r"""Interaktives Dashboard fuer Sliding-FDTD-Analysen.

Alle Analysen in einem Fenster, per Radio-Button umschaltbar:
  - Field 2D     : Wellenfeld als Heatmap, mit Schicht-Outlines
  - Field log    : log10|Ez| (evaneszent sichtbar)
  - Line Cut     : Vertikaler Schnitt bei x-Cursor, linear + log
  - Evanescent   : Exp-Decay-Fit unterhalb Lens-Bot
  - Time Trend   : max|Ez| pro Frame ueber alle Slides
  - Compare      : Bei mehreren NPZs: Vergleichs-Bars

Slider unten:
  - Frame: Frame-Index waehlen
  - x [um]: x-Position fuer Line-Cut/Evanescent

Bei mehreren geladenen NPZs:
  - "<" / ">" Buttons oben links zum Szenario-Wechsel

VERWENDUNG:
  python fdtd_analyzer.py
        -> oeffnet Datei-Dialog (springt in ./results)
  python fdtd_analyzer.py results\sliding_Gesund_frames.npz
  python fdtd_analyzer.py results\sliding_*_frames.npz
"""
import argparse
import glob
import os
import sys
# Rechenbibliothek (OpenBLAS) einfaedig: der Analyzer braucht keine parallelen BLAS-Threads,
# und OpenBLAS reserviert je Thread Puffer - bei knappem Windows-Commit-Speicher bricht es
# sonst mit "Memory allocation still failed ... giving up" den ganzen Prozess STILL ab.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider, Button
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib.patches import Polygon as MPoly, Circle, Rectangle
try:                                        # als Paket (common.fdtd_analyzer) ...
    from common.layer_stack import stack_from_npz, stack_bounds, stack_total_um
except ImportError:                         # ... oder als Skript aus common/
    from layer_stack import stack_from_npz, stack_bounds, stack_total_um

# Geometrie-Parameter
T_LENS = 250e-6
R_BEND = 8.3e-3
D_LENS = 14e-3
D1_S_CENTER = 2.0e-3
D3_S_CENTER = 1.8e-3
D1_H = 50e-6
D3_LEN = 50e-6
D3_QUERSCHN = 200e-6
N_PMMA = 1.491
N_POLYSTYR = 1.59
N_LIPID = 1.480
N_MUCIN = 1.342
N_CORNEA = 1.376

# Demo (planar) Geometrie: PS-WG bei y=0..5 um
PS_TOP = 5.0    # um
PS_BOT = 0.0    # um

MODES = ['Field 2D', 'Field log', 'Time Avg', 'Line Cut', 'Evanescent',
         'Resonance', 'Layer Power', 'Modes', 'Sensor',
         'Time Trend', 'Compare', 'Diff (A-B)', 'Scattered', 'Spectrum']

# Waehlbare Farbskalen (Knopf/Taste): sequentiell fuer Betraege/Intensitaeten,
# divergierend fuer vorzeichenbehaftete Felder (E, Differenz). KONTRAST saettigt die
# Skala (vmax * Faktor) -> schwache Streuanteile werden sichtbar.
CMAPS_SEQ = ['inferno', 'magma', 'viridis', 'cividis', 'turbo', 'hot', 'gray']
CMAPS_DIV = ['RdBu_r', 'seismic', 'coolwarm', 'PuOr_r', 'PiYG', 'bwr', 'twilight_shifted']
CONTRAST = [1.0, 0.3, 0.1, 0.03, 0.01]


def find_neff_peaks(neff, power, lo, hi, frac=0.06, min_dneff=0.004):
    """Peaks im n_eff-Spektrum im Band [lo,hi] -> Liste (n_eff, power),
    absteigend sortierbar. Portiert aus analyze_modes.py."""
    m = (neff >= lo) & (neff <= hi)
    nx, px = neff[m], power[m]
    if len(px) == 0 or px.max() <= 0:
        return []
    thr = frac*px.max()
    raw = [(nx[i], px[i]) for i in range(1, len(px)-1)
           if px[i] > px[i-1] and px[i] >= px[i+1] and px[i] >= thr]
    raw.sort()
    merged = []
    for n, p in raw:
        if merged and n - merged[-1][0] < min_dneff:
            if p > merged[-1][1]:
                merged[-1] = (n, p)
        else:
            merged.append((n, p))
    return merged


def get_detector_polygons():
    x_d1_um = (D1_S_CENTER - D_LENS/2)*1e6
    sag_d1 = R_BEND - np.sqrt(R_BEND**2 - (x_d1_um*1e-6)**2)
    y_mid_d1_um = -sag_d1*1e6
    slope = -(x_d1_um*1e-6)/np.sqrt(R_BEND**2 - (x_d1_um*1e-6)**2)
    theta = np.arctan(slope)
    ct, st = np.cos(theta), np.sin(theta)
    L_2D = 100.0
    H_um = D1_H*1e6
    max_y_shift = L_2D/2*abs(st) + H_um/2*abs(ct)
    d_norm = max(max_y_shift/abs(ct), 50)
    def mk(xc, yc):
        return [(xc + ct*u - st*v, yc + st*u + ct*v)
                for u, v in [(-L_2D/2, -H_um/2), (+L_2D/2, -H_um/2),
                             (+L_2D/2, +H_um/2), (-L_2D/2, +H_um/2)]]
    polys = {'D1': mk(x_d1_um + st*d_norm, y_mid_d1_um - ct*d_norm),
             'D2': mk(x_d1_um - st*d_norm, y_mid_d1_um + ct*d_norm)}
    x_d3_um = (D3_S_CENTER - D_LENS/2)*1e6
    sag_d3 = R_BEND - np.sqrt(R_BEND**2 - (x_d3_um*1e-6)**2)
    y_mid_d3_um = -sag_d3*1e6
    slope_d3 = -(x_d3_um*1e-6)/np.sqrt(R_BEND**2 - (x_d3_um*1e-6)**2)
    th3 = np.arctan(slope_d3)
    ct3, st3 = np.cos(th3), np.sin(th3)
    L3 = D3_LEN*1e6
    Q3 = D3_QUERSCHN*1e6
    polys['D3'] = [
        (x_d3_um + ct3*(-L3/2) - st3*(-Q3/2), y_mid_d3_um + st3*(-L3/2) + ct3*(-Q3/2)),
        (x_d3_um + ct3*(+L3/2) - st3*(-Q3/2), y_mid_d3_um + st3*(+L3/2) + ct3*(-Q3/2)),
        (x_d3_um + ct3*(+L3/2) - st3*(+Q3/2), y_mid_d3_um + st3*(+L3/2) + ct3*(+Q3/2)),
        (x_d3_um + ct3*(-L3/2) - st3*(+Q3/2), y_mid_d3_um + st3*(-L3/2) + ct3*(+Q3/2)),
    ]
    return polys


def compute_frame_amps(ezs):
    """Berechnet max|Ez| pro Frame, frame-by-frame (memory-effizient).
    Vermeidet die volle Allokation eines np.abs(ezs)-Arrays."""
    amps = np.empty(len(ezs), dtype=np.float32)
    for i in range(len(ezs)):
        amps[i] = float(np.max(np.abs(ezs[i])))
    return amps


MAX_DISPLAY_PIX = 500    # max Pixel pro Achse fuer imshow (Display-Downsampling)
                         # Kleiner = schnellere Redraws beim Slider-Scrubben
DEBOUNCE_MS = 250        # Wartezeit nach Slider-Event bevor gerendert wird


def downsample_for_display(ez, max_pix=MAX_DISPLAY_PIX):
    """Schneidet ein 2D-Array auf max_pix x max_pix runter (stride-slicing).
    Rohdaten bleiben unveraendert, nur fuer die Visualisierung."""
    Nx, Ny = ez.shape
    step_x = max(1, (Nx + max_pix - 1) // max_pix)
    step_y = max(1, (Ny + max_pix - 1) // max_pix)
    if step_x == 1 and step_y == 1:
        return ez
    return ez[::step_x, ::step_y]


_NPZ_STATS_CACHE = {}   # Path -> (scenario, max, mean) gecached um wiederholtes NPZ-Laden zu vermeiden


def compute_npz_stats(path):
    """Max und Mean |Ez| ueber ein ganzes NPZ-File, gecached."""
    if path in _NPZ_STATS_CACHE:
        return _NPZ_STATS_CACHE[path]
    d = np.load(path)
    ezs = d['Ez']
    scenario = str(d['scenario'])
    mx = 0.0
    sum_abs = 0.0
    n_pix = 0
    for i in range(len(ezs)):
        a = np.abs(ezs[i])
        mx = max(mx, float(a.max()))
        sum_abs += float(a.sum())
        n_pix += a.size
    result = (scenario, mx, (sum_abs / n_pix if n_pix else 0.0))
    _NPZ_STATS_CACHE[path] = result
    return result


# Leistungs-Skalierung (Eingabe 'P [mW]'): FDTD ist linear -> Felder ~ sqrt(P), Intensitaeten ~ P
POWER = {'mw': None, 'depth_um': 4.0}    # global fuer alle Dateien (Analyzer + Compare + Referenz)


def power_scale(p_si, unit, mw_file=None):
    """Feld-Skalenfaktor s = sqrt(P_soll / P_sim) fuer eine Datei. p_si = vom Solver exakt
    berechnete Leistung: bei VCSEL-Laeufen die vom Laser ABGESTRAHLTE Leistung (p_src_si),
    bei aelteren Dateien die eingekoppelte (Fluss durch die Eingangsebene). 3D in W, 2D in
    W/m (pro Meter Tiefe in z -> mal angenommene Tiefe POWER['depth_um']).
    P_soll: Eingabe 'Power [mW]', leer -> Laserleistung des Laufs (mw_file).
    Rueckgabe (s, P_sim_W) bzw. (1.0, None), wenn keine Skalierung (aus / Datei ohne Leistung)."""
    mw = POWER.get('mw') or mw_file
    if not mw or p_si is None or not np.isfinite(p_si) or p_si <= 0:
        return 1.0, None
    p_w = float(p_si)*(float(POWER['depth_um'])*1e-6 if str(unit) == 'W/m' else 1.0)
    return float(np.sqrt(mw*1e-3/p_w)), p_w


from matplotlib.widgets import TextBox as _MplTextBox


class FastTextBox(_MplTextBox):
    """TextBox that only applies its value on Enter and stays light:
    - matplotlib's TextBox redraws the WHOLE figure on every keystroke, on hover and
      (via stop_typing) on every click anywhere in the window - with a heavy field
      plot that makes typing and clicking sluggish. Here only the box itself is
      redrawn (blit of its axes).
    - Leaving the box without Enter does NOT submit; `on_cancel` (optional callable)
      can restore the shown value instead."""
    on_cancel = None

    def _fast_draw(self):
        canvas = self.ax.get_figure(root=True).canvas
        try:
            self.ax.draw(canvas.get_renderer())
            canvas.blit(self.ax.bbox)
        except Exception:
            canvas.draw_idle()

    def _rendercursor(self):
        fig = self.ax.get_figure(root=True)
        canvas = fig.canvas
        if fig._get_renderer() is None:
            return super()._rendercursor()
        had = 'draw' in canvas.__dict__
        old = canvas.__dict__.get('draw')
        canvas.draw = self._fast_draw          # the parent ends with canvas.draw()
        try:
            super()._rendercursor()
        finally:
            if had:
                canvas.draw = old
            else:
                canvas.__dict__.pop('draw', None)

    def stop_typing(self):
        if not self.capturekeystrokes:
            return                             # not typing: no redraw on foreign clicks
        if self._on_stop_typing is not None:
            self._on_stop_typing()
            self._on_stop_typing = None
        self.capturekeystrokes = False
        self.cursor.set_visible(False)
        self._fast_draw()
        if self.on_cancel is not None:         # Enter only: no submit when leaving
            try:
                self.on_cancel()
            except Exception:
                pass

    def _motion(self, event):
        if self.ignore(event):
            return
        from matplotlib import colors as _mcolors
        c = self.hovercolor if self.ax.contains(event)[0] else self.color
        if not _mcolors.same_color(c, self.ax.get_facecolor()):
            self.ax.set_facecolor(c)
            self._fast_draw()


def _npz_stream_frames(path, name):
    """Liefert die Frames (erste Achse) des Arrays `name` aus einer (auch komprimierten)
    npz einzeln, ohne das ganze Array in den Speicher zu laden (C-Reihenfolge)."""
    import zipfile
    from numpy.lib import format as _fmt
    with zipfile.ZipFile(path) as z:
        with z.open(name + '.npy') as f:
            ver = _fmt.read_magic(f)
            rd = _fmt.read_array_header_1_0 if ver == (1, 0) else _fmt.read_array_header_2_0
            shape, fortran, dtype = rd(f)
            if fortran:
                raise ValueError(f'{name}: Fortran order not supported')
            nb = int(np.prod(shape[1:]))*dtype.itemsize
            for _ in range(shape[0]):
                buf = f.read(nb)
                if len(buf) != nb:
                    raise ValueError(f'{name}: file incomplete')
                yield np.frombuffer(buf, dtype=dtype).reshape(shape[1:])


def _npz_power(d):
    """(P_sim [SI], Einheit, Laserleistung des Laufs [mW] oder None). VCSEL-Laeufe:
    abgestrahlte Laserleistung (p_src_si) -> Skalierung auf die LASER-Leistung; aeltere
    Dateien: eingekoppelte Leistung (p_in_si), ohne Standard-Leistung."""
    files = set(d.files)
    u = str(np.asarray(d['p_in_unit']).flat[0]) if 'p_in_unit' in files else None

    def _f(k):
        try:
            v = float(np.asarray(d[k]).flat[0])
            return v if np.isfinite(v) and v > 0 else None
        except Exception:
            return None
    ps = _f('p_src_si') if 'p_src_si' in files else None
    if ps is not None:
        mw = _f('laser_power_mw') if 'laser_power_mw' in files else None
        return ps, u, mw
    p = float(np.asarray(d['p_in_si']).flat[0]) if 'p_in_si' in files else None
    return p, u, None


def _sibling_polarization(path):
    """Polarisation fuer Dateien ohne 'polarization' (3D-Volumen vor dem 24.11.): aus der
    zugehoerigen Schnittdatei <stem>_frames.npz desselben Laufs, sonst 's' (alte Dateien)."""
    for suf in ('_vol3d.npz',):
        if str(path).endswith(suf):
            sib = str(path)[:-len(suf)] + '_frames.npz'
            if os.path.exists(sib):
                try:
                    with np.load(sib) as d2:
                        if 'polarization' in d2.files:
                            return str(np.asarray(d2['polarization']).flat[0]).lower()
                except Exception:
                    pass
    return 's'


_COMPARE_CACHE = {}


def compare_stats(path, comp, cw):
    """(Beschriftung, max, mean, Hinweis) von |E_comp| einer Datei fuer den Compare-Modus.
    comp: 'Ez'/'Ex'/'Ey'/'|E|'; cw=True -> eingeschwungene Einhuellende (knotenfrei),
    sonst |E| ueber alle einlaufenden Momentbilder. max/mean = None, wenn die Komponente
    bzw. Ansicht in der Datei nicht gespeichert ist (Hinweis-Text statt Balken).
    Speicherlayout: 'Ez' = Momentbilder der HAUPTkomponente (TE: Ez, TM: Ey), ausser in
    3D-Vektor-Volumen ('vector'): dort Ez/Ex/Ey je eigener Schluessel. 'Ez_cw' = CW-Phasen
    der Hauptkomponente, 'Ex_cw_amp' = komplexe Ex-Amplitude (2D-TM), 'Iavg' = <|E|^2>
    (alle Komponenten, wenn 'iavg_full')."""
    key = (path, os.path.getmtime(path), comp, bool(cw), POWER.get('mw'), POWER.get('depth_um'))
    if key in _COMPARE_CACHE:
        return _COMPARE_CACHE[key]
    d = np.load(path)
    files = set(d.files)
    scen = str(d['scenario']) if 'scenario' in files else os.path.basename(path)
    pol = (str(np.asarray(d['polarization']).flat[0]).lower()
           if 'polarization' in files else _sibling_polarization(path))
    prim = 'Ey' if pol in ('p', 'tm') else 'Ez'
    vector = 'vector' in files and 'Ex' in files and 'Ey' in files
    label = f'{scen} [{"TM" if prim == "Ey" else "TE"}]\n{os.path.basename(os.path.dirname(path))}'

    def env(phases):                                 # CW-Phasen -> Einhuellende |Ê|
        a = np.asarray(phases, dtype=np.float32)
        return np.sqrt(2.0*np.mean(a.astype(np.float64)**2, axis=0))

    arr, note, _sc = None, '', 1.0
    try:
        if comp == '|E|':
            if 'Iavg' in files and 'iavg_full' in files:
                arr = np.sqrt(2.0*np.maximum(np.asarray(d['Iavg'], dtype=np.float64), 0.0))
                if not cw:
                    note = 'time avg.'
            elif vector and not cw:
                arr = np.sqrt(sum(np.asarray(d[c], dtype=np.float32)**2 for c in ('Ez', 'Ex', 'Ey')))
            else:
                note = '|E| n/v'
        elif vector and not cw and comp in ('Ez', 'Ex', 'Ey'):
            arr = np.abs(np.asarray(d[comp], dtype=np.float32))
        elif comp == prim:
            if cw:
                arr = env(d['Ez_cw']) if 'Ez_cw' in files else None
                note = '' if arr is not None else 'no CW'
            else:
                arr = np.abs(np.asarray(d['Ez'], dtype=np.float32)) if not vector \
                    else np.abs(np.asarray(d[prim], dtype=np.float32))
        elif comp == 'Ex' and 'Ex_cw_amp' in files:
            arr = np.abs(np.asarray(d['Ex_cw_amp']))
            if not cw:
                note = 'CW only'
        else:
            note = f'{comp} n/v'
        _sc, _pw = power_scale(*_npz_power(d))
        if (POWER.get('mw') or _npz_power(d)[2]) and _pw is None:
            note = (note + ' ' if note else '') + 'no P'
    finally:
        d.close()
    if arr is not None and _sc != 1.0:
        arr = arr*_sc                             # Felder in V/m bei P_soll
    if arr is None:
        res = (label, None, None, note or f'{comp} n/v')
    else:
        res = (label, float(np.max(arr)), float(np.mean(arr)), note)
    _COMPARE_CACHE[key] = res
    return res


class Analyzer:
    def __init__(self, paths):
        self.paths = paths
        self.cur_path = 0
        self.load()                    # setzt u.a. self.planar (Geometrie-Erkennung)
        self.frame_idx = int(np.argmax(self.frame_amps))
        # Seitenverhaeltnis der Bild-Modi: 'auto' (gestreckt, WG gut sichtbar) oder
        # 'equal' (masstabsgetreu, Kugel rund). Umschaltbar per Button.
        self.aspect_mode = 'auto' if self.planar else 'equal'
        if self.planar:
            self.x_um = float((self.meta[0, 0] + self.meta[0, 1])/2)
        else:
            self.x_um = -5000.0
        self.mode = 'Field 2D'
        # Evanescent-Fit-Fenster (fit_top/fit_depth, in um rel. WG-Unterkante) wird
        # in load() -> _set_evan_defaults() delta-abhaengig gesetzt (Tiefe ~3*delta_min,
        # Start ~1-2 Zellen unter der Grenzflaeche). Fallback, falls load() es nicht
        # gesetzt hat:
        if not hasattr(self, 'fit_top'):
            self.fit_top = -0.05
        if not hasattr(self, 'fit_depth'):
            self.fit_depth = 1.5
        self.polys = get_detector_polygons()
        self.build_ui()
        self._update_power_hint()
        # Debounce-Timer: Slider-Events triggern keinen sofortigen Redraw,
        # sondern starten/resetten diesen Timer. Erst wenn DEBOUNCE_MS keine
        # weiteren Slider-Events kommen, wird gerendert -> kein Freeze.
        self._redraw_timer = self.fig.canvas.new_timer(interval=DEBOUNCE_MS)
        self._redraw_timer.single_shot = True
        self._redraw_timer.add_callback(self._safe_redraw)
        self.redraw()

    def load(self):
        import gc
        path = self.paths[self.cur_path]
        print(f'  Loading {os.path.basename(path)} ...', end=' ', flush=True)
        d = np.load(path)
        # WICHTIG: bei compressed NPZ ist d['Ez'] lazy - jeder Index-Zugriff
        # dekomprimiert neu. Wir forcen komplette RAM-Ladung mit np.asarray.
        _Ez = np.asarray(d['Ez'])
        # 3D-Volumen (_vol3d.npz) = 4D-Array (nframes, Nx, Ny, Nz).
        # 2D-Frames (_frames.npz) = 3D-Array (nframes, Nx, Ny).
        self.is3d = (_Ez.ndim == 4)
        self.scenario = str(d['scenario']) if 'scenario' in d.files else 'FDTD'
        self.layers = np.asarray(d['layers']) if 'layers' in d.files \
                      else np.array([0.030e-6, 3.5e-6, 0.5e-6, 1.336])
        # Material-Namen + Brechzahlen (5 Schichten: lens, lipid, aq, mucin, cornea)
        if 'material_names' in d.files:
            self.mat_names = [str(s) for s in np.asarray(d['material_names'])]
        else:
            self.mat_names = ['PMMA', 'Lipid', 'Aqueous', 'Mucin', 'Cornea']
        if 'material_indices' in d.files:
            self.mat_indices = np.asarray(d['material_indices']).tolist()
        else:
            self.mat_indices = [N_PMMA, N_LIPID, float(self.layers[3]),
                                N_MUCIN, N_CORNEA]
        self._build_stack(d)
        # Flat/Curved Flag (Full-Domain Flattened)
        self.flat_mode = ('flattened' in d.files and
                          float(np.asarray(d['flattened']).flat[0]) > 0)
        # Tatsaechliche WG-Kerndicke (um) fuer die Overlay-Oberkante; Default PS_TOP.
        # (Frueher fest 5um -> falsch, wenn der Lauf eine andere WG-Dicke nutzte.)
        self.wg_top_um = (float(np.asarray(d['t_wg_um']).flat[0])
                          if 't_wg_um' in d.files else PS_TOP)
        # Slab-Koordinaten: alle Dateien des einheitlichen Runners (native 2D/3D,
        # Meep) speichern t_wg_um und legen den WG-Kern GERADE auf y=0..t_wg - auch
        # die (geflattete bzw. konform abgebildete) Kontaktlinse. Nur die alten
        # kontaktlinse/-Skripte nutzen Linsen-Koordinaten (Sag bzw. +-T/2).
        self.slab_coords = 't_wg_um' in d.files
        # Iavg enthaelt alle E-Komponenten (<|E|^2>)? Aeltere Dateien: nur die
        # Hauptkomponente -> dann NICHT als volle Intensitaet verwenden/beschriften.
        self.iavg_full = 'iavg_full' in d.files
        # Freie CAD-Geometrie: Regionskarte (welcher Koerper liegt in welcher Zelle)
        # statt Schichtstapel. 2D-Datei: (Nx,Ny); 3D-Volumen: (Nx,Ny,Nz).
        self.is_cad = ('geometry' in d.files
                       and str(np.asarray(d['geometry']).flat[0]) == 'cad')
        self.cad_region = (np.asarray(d['cad_region']) if (self.is_cad and 'cad_region' in d.files)
                           else None)
        self.cad_core = int(np.asarray(d['cad_core']).flat[0]) if self.is_cad else 0
        self.cad_tear = int(np.asarray(d['cad_tear']).flat[0]) if self.is_cad else 0
        self.cad_file = str(np.asarray(d['cad_file']).flat[0]) if self.is_cad else ''
        # Fourier-Stitch (Puls): Transmission je Frequenz + Gewicht des Quellspektrums
        self.spec = ((np.asarray(d['spec_lam_nm']), np.asarray(d['spec_T']),
                      np.asarray(d['spec_weight'])) if 'spec_lam_nm' in d.files else None)
        self.reg2d = self.cad_region if (self.cad_region is not None
                                         and self.cad_region.ndim == 2) else None
        # Bead-Info (nur planar_beads-Dateien): Position/Groesse fuer Marker
        # Alte Dateien: Bead stand auf Platz 2 der Namensliste (statt der Lipidschicht).
        self._legacy_bead_slot = (not self.stack_custom and len(self.mat_names) > 1
                                  and self.mat_names[1] == 'Bead')
        if 'bead_diameter_um' in d.files:
            self.bead_d = float(np.asarray(d['bead_diameter_um']).flat[0])
        else:
            self.bead_d = float(self.layers[1])*1e6 if self._legacy_bead_slot else 0.0
        self.is_bead = self.bead_d > 0 or self._legacy_bead_slot
        self.bead_material = (str(np.asarray(d['bead_material']).flat[0])
                              if 'bead_material' in d.files else '')
        # Bead-Brechzahl: eigener Eintrag (neu) oder Platz 2 (alt)
        self.bead_n = (float(np.asarray(d['bead_n']).flat[0]) if 'bead_n' in d.files
                       else (float(self.mat_indices[1]) if self._legacy_bead_slot else float('nan')))
        self.bead_x = (float(np.asarray(d['bead_x_um']).flat[0])
                       if 'bead_x_um' in d.files else 500.0)
        # Bead-Zentrum in y/z (fuer 3D-Schnittkreis in jeder Ebene). Defaults:
        # y=-r (aufliegend), z=0 (Domainmitte) - passend zu aelteren Dateien.
        self.bead_y = (float(np.asarray(d['bead_y_um']).flat[0])
                       if 'bead_y_um' in d.files else -self.bead_d/2.0)
        self.bead_z = (float(np.asarray(d['bead_z_um']).flat[0])
                       if 'bead_z_um' in d.files else 0.0)
        # Einkoppelabstand / WG-Eintrittsfacette (x): Luftweg vor dem WG-Start.
        self.x_wg_start = (float(np.asarray(d['x_wg_start_um']).flat[0])
                           if 'x_wg_start_um' in d.files else 0.0)
        self.x_src = (float(np.asarray(d['x_src_um']).flat[0])
                      if 'x_src_um' in d.files else 0.0)
        self.end_facet = (float(np.asarray(d['end_facet_um']).flat[0])
                          if 'end_facet_um' in d.files else 0.0)
        # WG-Endfacette in x (2D-Beads): WG-Kern nur zwischen x_wg_start..x_wg_end.
        self.x_wg_end = (float(np.asarray(d['x_wg_end_um']).flat[0])
                         if 'x_wg_end_um' in d.files else None)
        # Wellenlaenge (fuer Modenanalyse n_eff=k_x/k0). Default 850 nm (alte Dateien).
        self.lam_nm = (float(np.asarray(d['lam_nm']).flat[0])
                       if 'lam_nm' in d.files else 850.0)
        # Polarisation (fuer Beschriftung): s/TE -> gespeichertes Feld ist Ez,
        # p/TM -> gespeichertes (Haupt-)Feld ist Ey. Alte Dateien = s.
        _polv = (str(np.asarray(d['polarization']).flat[0]).lower()
                 if 'polarization' in d.files else _sibling_polarization(path))
        self.pol = 'p' if _polv in ('p', 'tm') else 's'
        # Primaerkomponente = gespeichertes Skalarfeld; Label je nach Polarisation
        self.prim = 'Ey' if self.pol == 'p' else 'Ez'
        self.comp = getattr(self, '_comp_wish', None) or self.prim   # gewaehlte Komponente behalten (s. _sync)
        self.pol_tag = 'TM/p' if self.pol == 'p' else 'TE/s'
        self.ex_amp2d = None
        # eingekoppelte Leistung (SI) -> Felder bei gewuenschter Leistung in V/m
        self.p_in_si, self.p_in_unit, self.p_laser_mw = _npz_power(d)
        self._escale, self._p_sim_w = power_scale(self.p_in_si, self.p_in_unit, self.p_laser_mw)
        _sc = self._escale

        def _S(a, k=1):                         # Feld (k=1) bzw. Intensitaet (k=2) skalieren
            if a is None or _sc == 1.0:
                return a
            dt = np.complex64 if np.iscomplexobj(a) else np.float32   # float16 liefe ueber
            return np.asarray(a, dtype=dt)*dt(_sc**k)
        if self.is3d:
            # Volles Volumen: (nf, Nx=Laenge, Ny=Querschnitt, Nz=Tiefe)
            self.vol = _Ez
            self.dx_um = float(np.asarray(d['dx_um']).flat[0]) if 'dx_um' in d.files else 0.05
            self.x0_um = float(np.asarray(d['x0_um']).flat[0]) if 'x0_um' in d.files else 0.0
            self.y0_um = float(np.asarray(d['y0_um']).flat[0]) if 'y0_um' in d.files else 0.0
            self.z0_um = float(np.asarray(d['z0_um']).flat[0]) if 'z0_um' in d.files else 0.0
            self.t_wg_um = float(np.asarray(d['t_wg_um']).flat[0]) if 't_wg_um' in d.files else 0.0
            self._steps = (np.asarray(d['steps']) if 'steps' in d.files
                           else np.arange(len(_Ez)))
            self._times = (np.asarray(d['times']) if 'times' in d.files
                           else np.zeros(len(_Ez)))
            # physikalischer Zeitschritt (die gespeicherten 'times' sind beim 3D-Lauf
            # Rechenzeiten in s, nicht die Simulationszeit -> t = Schritt * dt)
            self._dt_s = float(np.asarray(d['dt']).flat[0]) if 'dt' in d.files else None
            # Vektor-Modus: Ex/Ey zusaetzlich vorhanden -> Komponenten-Auswahl
            self.is_vector = ('Ex' in d.files and 'Ey' in d.files)
            self.vol_ex = np.asarray(d['Ex']) if self.is_vector else None
            self.vol_ey = np.asarray(d['Ey']) if self.is_vector else None
            # Ansichten: einlaufend (transient) / eingeschwungen (CW-Volumen)
            _ez_cw_vol = np.asarray(d['Ez_cw']) if 'Ez_cw' in d.files else None
            _cwonly = bool(int(np.asarray(d['cw_only']).flat[0])) if 'cw_only' in d.files else False
            _hascw = bool(int(np.asarray(d['has_cw']).flat[0])) if 'has_cw' in d.files else False
            _vprim = str(np.asarray(d['view_primary']).flat[0]) if 'view_primary' in d.files else 'transient'
            self.comp = getattr(self, '_comp_wish', None) or self.prim   # gewaehlte Komponente behalten (s. _sync)
            self.plane = 'xy'                       # Laenge-Querschnitt (Standard)
            self.slice_idx = self.vol.shape[3]//2   # Tiefe-Mitte
            # Vom Solver EXAKT akkumuliertes Zeitmittel <E^2> (jeder Zeitschritt der
            # 2. Lauf-Haelfte) -> fuer 'Int-Mittel' viel besser als das Mittel ueber
            # die wenigen Snapshots (die oft phasengleich sind -> lambda/2-Artefakt).
            self.iavg_vol = np.asarray(d['Iavg']) if 'Iavg' in d.files else None
            self.iavg2d = None
            d.close()
            if _sc != 1.0:
                _Ez = _S(_Ez); _ez_cw_vol = _S(_ez_cw_vol)
                self.vol = _Ez; self.vol_ex = _S(self.vol_ex); self.vol_ey = _S(self.vol_ey)
                self.iavg_vol = _S(self.iavg_vol, 2)
            self._build_views_3d(_Ez, self._steps, self._times, _ez_cw_vol,
                                 _cwonly, _hascw, _vprim)
            self._apply_slice()                     # setzt self.ezs, self.meta, frame_amps
        else:
            self.vol = None
            self.is_vector = False
            self.ezs = _Ez
            self.meta = np.asarray(d['meta'])
            # Ansichten: einlaufend (transient) / eingeschwungen (CW). Flags + 2. Satz
            _ez_cw = np.asarray(d['Ez_cw']) if 'Ez_cw' in d.files else None
            _meta_cw = np.asarray(d['meta_cw']) if 'meta_cw' in d.files else None
            _cwonly = bool(int(np.asarray(d['cw_only']).flat[0])) if 'cw_only' in d.files else False
            _hastr = bool(int(np.asarray(d['has_transient']).flat[0])) if 'has_transient' in d.files else True
            _hascw = bool(int(np.asarray(d['has_cw']).flat[0])) if 'has_cw' in d.files else False
            _vprim = str(np.asarray(d['view_primary']).flat[0]) if 'view_primary' in d.files else 'transient'
            # exakt akkumuliertes Zeitmittel (falls vom Solver mitgespeichert)
            self.iavg_vol = None
            self.iavg2d = np.asarray(d['Iavg']).astype(np.float64) if 'Iavg' in d.files else None
            # TM: komplexe CW-Amplitude von Ex (tangential) -> Ex/Ey/|E| umschaltbar
            self.ex_amp2d = np.asarray(d['Ex_cw_amp']) if 'Ex_cw_amp' in d.files else None
            d.close()
            if _sc != 1.0:
                _Ez = _S(_Ez); _ez_cw = _S(_ez_cw)
                self.ezs = _Ez; self.iavg2d = _S(self.iavg2d, 2); self.ex_amp2d = _S(self.ex_amp2d)
            self._build_views_2d(_Ez, self.meta, _ez_cw, _meta_cw, _cwonly, _hastr, _hascw, _vprim)
            self.frame_amps = compute_frame_amps(self.ezs)
        self.path_label = os.path.basename(path)
        # Geometrie: 'planar' = gerader WG-Kern bei y=PS_BOT..wg_top_um. Gilt fuer
        # alle Slab-Koordinaten-Dateien (inkl. Linse aus dem Runner) und fuer alte
        # Planar-Demos (y-Range ~20um). Sonst alte Linsen-Datei (Sag-Koordinaten).
        # Pro Datei neu (nicht nur im __init__), damit Datei-Wechsel stimmt.
        y_range_um = float(self.meta[0, 3] - self.meta[0, 2])
        # CAD: gerade Achsen, Grenzflaechen kommen aus der Regionskarte (-> 'planar'-Zweige
        # fuer Zoom/Seitenverhaeltnis; alle Geometrie-Abfragen laufen ueber die Regionen)
        self.planar = self.slab_coords or y_range_um < 100.0 or self.is_cad
        self._set_evan_defaults()      # delta-abhaengiges Fit-Fenster fuer Evanescent
        _kind = '3D volume' if self.is3d else '2D frames'
        print(f'{len(self.ezs)} Frames {self.ezs.shape[1]}x{self.ezs.shape[2]} '
              f'[{_kind}] - computing statistics ...', flush=True)
        self._sync_3d_controls()   # Schnitt-Regler an neue Datei anpassen (2D<->3D)
        gc.collect()

    # ------- Schichtstapel unter dem Kern (frei konfigurierbar) -------
    # Alte Dateien: feste Slots Lipid/Aqueous/Mucin mit ihren bisherigen Farben/Stilen.
    _LEGACY_STYLE = {'lipid': ('#88AAFF', ':', 0.18, '.3f'),
                     'aqueous': ('#56C4FF', '--', 0.15, '.2f'),
                     'mucin': ('#D88AFF', '--', 0.15, '.2f')}
    STACK_COLORS = ('#88AAFF', '#56C4FF', '#D88AFF', '#7CFC00', '#40E0D0', '#FFB6C1',
                    '#F4A460', '#B0C4DE', '#E0E070', '#9AA0FF')
    SUB_COLOR = '#553030'
    CORE_COLOR = '#FFD75E'

    def _build_stack(self, d):
        """self.stack (Schichten vom Kern abwaerts, Dicke > 0), self.stack_report (fuer
        Integrale/Balken: alte Dateien behalten ALLE drei Slots, auch Dicke 0),
        Kern-/Substrat-Name und -Index. Neue Dateien: stack_* Schluessel und
        material_names/indices = [Kern, *Stapel, Substrat]; alte: aus 'layers'."""
        names = list(self.mat_names)
        idx = list(self.mat_indices)
        st = stack_from_npz(d)
        self.stack_custom = st is not None
        if st is None:
            dn = ['PMMA', 'Lipid', 'Aqueous', 'Mucin', 'Cornea']
            dv = [N_PMMA, N_LIPID, float(self.layers[3]), N_MUCIN, N_CORNEA]
            names = names + dn[len(names):]
            idx = idx + dv[len(idx):]
            t3 = [float(self.layers[k])*1e6 for k in range(3)]
            slots = ('lipid', 'aqueous', 'mucin')
            full = [dict(name=str(names[1 + k]), t_um=t3[k], n=complex(float(idx[1 + k]), 0.0),
                         kappa=0.0, mat=slots[k], slot=slots[k]) for k in range(3)]
            # Gezeichnet wird wie bisher: Lipid nur ab 1 nm, Aqueous/Mucin IMMER (auch mit
            # Dicke 0 - alte Plots zeigten dann die Linie + Legende "0.00um").
            # (legacy_stack wuerde Null-Schichten verwerfen -> andere Legenden.)
            st = [dict(L) for L in full if not (L['slot'] == 'lipid' and L['t_um'] <= 1e-3)]
            for L in full + st:
                c, ls, a, f = self._LEGACY_STYLE[L['slot']]
                L.update(color=c, ls=ls, alpha=a, tfmt=f)
            self.stack_report = full
            self.core_name, self.core_n = str(names[0]), float(idx[0])
            self.sub_name, self.sub_n = str(names[4]), float(idx[4])
        else:
            for k, L in enumerate(st):
                L.update(slot=None, color=self.STACK_COLORS[k % len(self.STACK_COLORS)],
                         ls='--', alpha=0.15, tfmt='.3f' if L['t_um'] < 0.1 else '.2f')
            self.stack_report = st
            self.core_name = str(names[0]) if names else 'Core'
            self.core_n = float(np.real(idx[0])) if idx else N_POLYSTYR
            self.sub_name = str(names[-1]) if len(names) >= 2 else 'Substrate'
            self.sub_n = float(np.real(idx[-1])) if len(idx) >= 2 else N_CORNEA
        self.stack = st

    def _stack_bounds(self, y_top=0.0):
        """[(name, y_top, y_bot, n), ...] der Stapel-Schichten ab y_top (Kern-Unterkante)
        nach unten."""
        return stack_bounds(getattr(self, 'stack', []), y_top)

    def _stack_total(self):
        """Gesamtdicke des Stapels (um); darunter beginnt das Substrat."""
        return stack_total_um(getattr(self, 'stack', []))

    @staticmethod
    def _n_str(n):
        n = complex(n)
        return f'{n.real:.3f}' + (f'+{n.imag:.1e}i' if n.imag > 0 else '')

    def _layer_label(self, L):
        """'Name 3.50um (n=1.336)' (Dickenformat wie bisher je Slot)."""
        return f'{L["name"]} {L["t_um"]:{L.get("tfmt", ".2f")}}um (n={self._n_str(L["n"])})'

    def _clad_n(self, legacy=None):
        """Brechzahl direkt unter dem Kern (Mantel fuer delta_min/Moden/Evanescent):
        stack[0] bzw. das Substrat bei leerem Stapel. Alte Dateien: ``legacy``
        (bisheriger Wert, i.d.R. Aqueous), damit sich nichts aendert."""
        if not getattr(self, 'stack_custom', False) and legacy is not None:
            return float(legacy)
        # Thin films (< 0.1 um, e.g. the 30 nm lipid layer) are much thinner than the
        # decay length; the evanescent tail lives in the first THICK layer below them.
        # (Using a 30 nm lipid with n~1.47 would give delta_min ~1.3 um, far too long.)
        for L in getattr(self, 'stack', []) or []:
            if L['t_um'] >= 0.1:
                return float(complex(L['n']).real)
        return float(getattr(self, 'sub_n', 1.0))

    def _tear_band(self):
        """(Tiefe um, Beschriftung) des 'Tear'-Bands fuer die Sensor-Metrik: alte
        Dateien wie bisher die Aqueous-Dicke; neue Dateien der ganze Stapel."""
        if not getattr(self, 'stack_custom', False):
            try:
                t = float(self.layers[1])*1e6
            except Exception:
                t = 2.0
            return max(0.5, min(t, 8.0)), (self.mat_names + [''] * 3)[2]
        tot = self._stack_total()
        lab = ' + '.join(L['name'] for L in self.stack) if self.stack else \
            f'{self.sub_name} (no layers, 0.5um substrate band)'
        return (tot if tot > 0 else 0.5), lab

    # ------- Ansichten: einlaufend (transient) / eingeschwungen (CW) -------
    def _make_illustrative_transient(self, cw_ezs, cw_meta, M=24):
        """Aus dem eingeschwungenen CW-Satz eine ILLUSTRATIVE einlaufende Front
        erzeugen (weiche Stufe x<Front laeuft von links ein). Kein echter
        Transient - nur zur Veranschaulichung, klar so benannt."""
        nfr, Nx, Ny = cw_ezs.shape
        x0 = float(cw_meta[0, 0]); x1 = float(cw_meta[0, 1]); span = max(x1 - x0, 1e-9)
        xs = np.linspace(x0, x1, Nx)
        frames = np.empty((M, Nx, Ny), dtype=np.float32); meta = []
        w = 6.0/span
        for j in range(M):
            front = x0 + span*(j + 1)/M
            mask = 1.0/(1.0 + np.exp((xs - front)*w))     # 1 links, 0 rechts der Front
            frames[j] = (cw_ezs[j % nfr]*mask[:, None]).astype(np.float32)
            meta.append((x0, x1, float(cw_meta[0, 2]), float(cw_meta[0, 3]), 0, j, j))
        return frames, np.array(meta, dtype=np.float64)

    def _make_illustrative_transient_3d(self, cw_vol, M=24):
        """3D-Analogon: einlaufende Front (weiche Stufe in x) aus dem CW-Volumen."""
        nfr, Nx, Ny, Nz = cw_vol.shape
        x0 = self.x0_um; span = max(Nx*self.dx_um, 1e-9)
        xs = x0 + np.arange(Nx)*self.dx_um
        w = 6.0/span
        out = np.empty((M, Nx, Ny, Nz), dtype=np.float32)
        for j in range(M):
            front = x0 + span*(j + 1)/M
            mask = (1.0/(1.0 + np.exp((xs - front)*w))).astype(np.float32)
            out[j] = (np.asarray(cw_vol[j % nfr]).astype(np.float32)*mask[:, None, None])
        steps = np.arange(M); times = np.arange(M, dtype=np.float64)
        return out, steps, times

    def _build_views_2d(self, ez, meta, ez_cw, meta_cw, cw_only, has_tr, has_cw, primary):
        self._views = {}; self._view_names = []
        if cw_only:
            self._views['Steady state'] = ('2d', ez, meta)
            tr, tm = self._make_illustrative_transient(ez, meta)
            self._views['Transient (illustrative)'] = ('2d', tr, tm)
            self._view_names = ['Steady state', 'Transient (illustrative)']
            cur = 'Steady state'
        else:
            self._views['Transient'] = ('2d', ez, meta); self._view_names = ['Transient']
            if has_cw and ez_cw is not None:
                self._views['Steady state'] = ('2d', ez_cw, meta_cw)
                self._view_names.append('Steady state')
            cur = 'Steady state' if (primary == 'cw' and 'Steady state' in self._views) else 'Transient'
        self._view_cur = cur
        _, self.ezs, self.meta = self._views[cur]

    def _build_views_3d(self, vol, steps, times, cw_vol, cw_only, has_cw, primary):
        self._views = {}; self._view_names = []
        f0 = 2.99792458e8/(getattr(self, 'lam_nm', 850.0)*1e-9)
        if cw_only:
            self._views['Steady state'] = ('3d', vol, np.arange(len(vol)),
                                              np.array([(k/len(vol))/f0 for k in range(len(vol))]))
            tv, ts, tt = self._make_illustrative_transient_3d(vol)
            self._views['Transient (illustrative)'] = ('3d', tv, ts, tt)
            self._view_names = ['Steady state', 'Transient (illustrative)']
            cur = 'Steady state'
        else:
            self._views['Transient'] = ('3d', vol, steps, times)
            self._view_names = ['Transient']
            if has_cw and cw_vol is not None:
                ncw = len(cw_vol)
                self._views['Steady state'] = ('3d', cw_vol, np.arange(ncw),
                                                  np.array([(k/ncw)/f0 for k in range(ncw)]))
                self._view_names.append('Steady state')
            cur = 'Steady state' if (primary == 'cw' and 'Steady state' in self._views) else 'Transient'
        self._view_cur = cur
        _, self.vol, self._steps, self._times = self._views[cur]

    def _apply_view(self, name):
        if name not in getattr(self, '_views', {}):
            return
        self._view_cur = name
        payload = self._views[name]
        if payload[0] == '3d':
            _, self.vol, self._steps, self._times = payload
            self.slice_idx = min(getattr(self, 'slice_idx', 0), self.vol.shape[3] - 1)
            self._apply_slice()                       # setzt self.ezs + self.meta neu
            self.frame_amps = compute_frame_amps(self.ezs)
        else:
            _, self.ezs, self.meta = payload
            self.frame_amps = compute_frame_amps(self.ezs)
        self.frame_idx = min(getattr(self, 'frame_idx', 0), len(self.ezs) - 1)
        try:
            self.s_frame.valmax = len(self.ezs) - 1
            self.s_frame.ax.set_xlim(0, len(self.ezs) - 1)
            self.s_frame.set_val(self.frame_idx)
        except Exception:
            pass
        if hasattr(self, 'b_view'):
            self.b_view.label.set_text(f'View: {name}')
        try:
            self.redraw()
        except Exception:
            self._safe_redraw()

    def on_view_toggle(self, event=None):
        names = getattr(self, '_view_names', [])
        if len(names) < 2:
            # Nur EINE Ansicht in der Datei -> klar sagen warum (statt stumm nichts
            # zu tun). Haeufigster Fall: der Lauf hat keinen CW-Satz gespeichert
            # (has_cw=0 / kein Ez_cw), z.B. abgebrochene oder wiederhergestellte Laeufe.
            print(f'[View] Only "{getattr(self, "_view_cur", "-")}" is available in this file '
                  f'(no second frame set saved) -> nothing to '
                  f'toggle. A file with a CW set is required (Ez_cw/has_cw=1).')
            if hasattr(self, 'b_view'):
                self.b_view.label.set_text(f'View: {self._view_cur} (only one)')
                self._safe_redraw()
            return
        i = (names.index(self._view_cur) + 1) % len(names)
        self._busy(f'Switching to {names[i]} ...')
        self._apply_view(names[i])

    # ------- 3D-Volumen: Schnittebenen-Logik -------
    PLANE_LABELS = {'xy': ('Length x (um)', 'Cross-section y (um)'),
                    'xz': ('Length x (um)', 'Depth z (um)'),
                    'yz': ('Cross-section y (um)', 'Depth z (um)')}

    def _slice_max_idx(self, plane=None):
        """Groesster gueltiger Schnitt-Index fuer die (fixe) 3. Achse."""
        plane = plane or self.plane
        _, Nx, Ny, Nz = self.vol.shape
        return {'xy': Nz, 'xz': Ny, 'yz': Nx}[plane] - 1

    def _default_slice_idx(self, plane=None):
        """Sinnvolle Startposition der Schnittebene je Ansicht, damit beim
        Ebenenwechsel NICHT ein alter Index in eine irrelevante Hoehe faellt:
          xy -> schneidet Tiefe z: Mitte (z=0, Kanalmitte)
          xz -> schneidet Querschnitt y: WG-KERNMITTE (y=t_wg/2)
          yz -> schneidet Laenge x: Domainmitte."""
        plane = plane or self.plane
        _, Nx, Ny, Nz = self.vol.shape
        maxidx = self._slice_max_idx(plane)
        if plane == 'xz':
            t_wg = getattr(self, 't_wg_um', 0.0) or 0.0
            if t_wg > 0:
                idx = int(round((t_wg/2.0 - self.y0_um)/self.dx_um))
                return int(max(0, min(idx, maxidx)))
            return Ny//2
        elif plane == 'yz':
            return Nx//2
        return Nz//2   # xy

    def _apply_slice(self):
        """Erzeugt aus dem Volumen den aktuellen 2D-Schnitt (alle Frames) und
        baut ein passendes meta (Extents fuer die beiden Anzeige-Achsen).
        xy = Laenge-Querschnitt (fix Tiefe), xz = Laenge-Tiefe (fix Querschnitt),
        yz = Querschnitt-Tiefe (fix Laenge)."""
        v = self.vol
        nf, Nx, Ny, Nz = v.shape
        dx = self.dx_um
        self.slice_idx = int(max(0, min(self.slice_idx, self._slice_max_idx())))
        i = self.slice_idx

        def _slc(vol):
            if self.plane == 'xy':
                return np.ascontiguousarray(vol[:, :, :, i])
            elif self.plane == 'xz':
                return np.ascontiguousarray(vol[:, :, i, :])
            else:  # 'yz'
                return np.ascontiguousarray(vol[:, i, :, :])

        # exakt akkumuliertes Zeitmittel <E^2> (3D, ohne Frame-Achse) mitschneiden
        _iv = getattr(self, 'iavg_vol', None)
        if _iv is not None and getattr(_iv, 'ndim', 0) == 3:
            try:
                if self.plane == 'xy':
                    self.iavg2d = np.asarray(_iv[:, :, min(i, _iv.shape[2]-1)], dtype=np.float64)
                elif self.plane == 'xz':
                    self.iavg2d = np.asarray(_iv[:, min(i, _iv.shape[1]-1), :], dtype=np.float64)
                else:  # yz
                    self.iavg2d = np.asarray(_iv[min(i, _iv.shape[0]-1), :, :], dtype=np.float64)
            except Exception:
                self.iavg2d = None

        comp = getattr(self, 'comp', 'Ez')
        # Feldbild folgt der Komponente nur im Vektor-Modus; sonst zeigt es die
        # gespeicherte Hauptkomponente (die Wahl wirkt dann auf die 1D-Profile).
        self._is_magnitude = (comp == '|E|' and getattr(self, 'is_vector', False))
        if comp == '|E|' and getattr(self, 'is_vector', False):
            ez = _slc(self.vol).astype(np.float32)
            ex = _slc(self.vol_ex).astype(np.float32)
            ey = _slc(self.vol_ey).astype(np.float32)
            self.ezs = np.sqrt(ex*ex + ey*ey + ez*ez)
        else:
            src = {'Ez': self.vol, 'Ex': getattr(self, 'vol_ex', None),
                   'Ey': getattr(self, 'vol_ey', None)}.get(comp, self.vol)
            if src is None:
                src = self.vol
            # float32-Cast: Volumen liegt oft als float16 vor (Speicher), aber
            # np.linalg (Evanescent-/Resonance-Fits) kann kein float16. Der
            # Schnitt ist klein -> Cast kostet kaum RAM, Volumen bleibt float16.
            self.ezs = _slc(src).astype(np.float32)

        if self.plane == 'xy':
            a0, a1 = self.x0_um, self.x0_um + Nx*dx
            b0, b1 = self.y0_um, self.y0_um + Ny*dx
        elif self.plane == 'xz':
            a0, a1 = self.x0_um, self.x0_um + Nx*dx
            b0, b1 = self.z0_um, self.z0_um + Nz*dx
        else:  # 'yz'
            a0, a1 = self.y0_um, self.y0_um + Ny*dx
            b0, b1 = self.z0_um, self.z0_um + Nz*dx
        st, tm = self._steps, self._times
        self.meta = np.array(
            [(a0, a1, b0, b1, 0,
              float(tm[k]) if k < len(tm) else 0.0,
              int(st[k]) if k < len(st) else 0) for k in range(nf)],
            dtype=np.float64)
        # Overlays (Schichtlinien) nur sinnvoll in der xy-Ebene (Querschnitt=y);
        # CAD-Konturen kommen aus der Regionskarte und passen in JEDER Ebene.
        self._overlay_ok = (self.plane == 'xy') or getattr(self, 'is_cad', False)
        _cr = getattr(self, 'cad_region', None)
        if _cr is not None and _cr.ndim == 3:
            ii = min(i, _cr.shape[{'xy': 2, 'xz': 1, 'yz': 0}[self.plane]] - 1)
            self.reg2d = (_cr[:, :, ii] if self.plane == 'xy' else
                          _cr[:, ii, :] if self.plane == 'xz' else _cr[ii, :, :])
        self.frame_amps = compute_frame_amps(self.ezs)

    def _slice_pos_um(self):
        """Physikalische Position der aktuellen Schnittebene (fixe Achse)."""
        dx = self.dx_um
        if self.plane == 'xy':
            return self.z0_um + self.slice_idx*dx, 'Depth z'
        elif self.plane == 'xz':
            return self.y0_um + self.slice_idx*dx, 'Cross-section y'
        else:
            return self.x0_um + self.slice_idx*dx, 'Length x'

    # Slider-Beschriftung je Ebene (physikalische Achse, nicht Roh-Index)
    _SLICE_AXIS_LABEL = {'xy': 'Depth z', 'xz': 'Cross-section y', 'yz': 'Length x'}

    def _update_slice_valtext(self):
        """Zeigt am Slice-Slider die PHYSIKALISCHE Position (um bzw. mm) statt des
        Roh-Index -> intuitiver. Wird nach jeder Aenderung aufgerufen."""
        try:
            pos, _ = self._slice_pos_um()
            if abs(pos) >= 1000.0:
                txt = f'{pos/1000.0:.3f} mm'
            else:
                txt = f'{pos:.2f} um'
            self.s_slice.valtext.set_text(txt)
        except Exception:
            pass

    def on_plane(self, label):
        if getattr(self, '_syncing', False):
            return                               # programmatischer Sync -> nicht feuern
        if not getattr(self, 'is3d', False) or getattr(self, 'vol', None) is None:
            return                               # ausgeblendet (2D-Datei): Klicks ignorieren
        self.plane = label.split()[0]            # 'xy'/'xz'/'yz' aus Label
        self._busy(f'Slicing plane {self.plane} ...')
        maxidx = self._slice_max_idx()
        # Beim Ebenenwechsel auf eine SINNVOLLE Schnittposition springen
        # (WG-Kernmitte / Kanalmitte), statt den alten Index zu uebernehmen.
        self.slice_idx = min(self._default_slice_idx(), maxidx)
        try:
            self.s_slice.valmin = 0
            self.s_slice.valmax = maxidx
            self.s_slice.ax.set_xlim(0, maxidx)
            self.s_slice.label.set_text(self._SLICE_AXIS_LABEL[self.plane])
            # Handle/Wert auf gueltigen Bereich setzen (sonst zeigt der Slider
            # einen alten, zu grossen Wert an -> valmax-Zicken von matplotlib)
            self.s_slice.set_val(self.slice_idx)
            self._update_slice_valtext()
        except Exception:
            pass
        self._apply_slice()
        self._update_x_slider_range()   # x-Cursor an neue horizontale Achse anpassen
        self._safe_redraw()

    def on_slice(self, val):
        if getattr(self, '_syncing', False):
            return                               # programmatischer Sync -> nicht feuern
        if not getattr(self, 'is3d', False) or getattr(self, 'vol', None) is None:
            return                               # ausgeblendet (2D-Datei): Klicks ignorieren
        new = int(val)
        if new == self.slice_idx:
            return
        self.slice_idx = new
        self._update_slice_valtext()
        self._apply_slice()
        self._schedule_redraw()

    def _disp_comp(self):
        """Komponente, die self.ezs (Feldbild/Frames) TATSAECHLICH enthaelt: im Vektor-
        Modus die gewaehlte, sonst die gespeicherte Hauptkomponente. Fuer Beschriftungen
        der Modi, die self.ezs zeigen (nur 'Evanescent' wertet die Wahl selbst aus)."""
        c = getattr(self, 'comp', self.prim)
        return c if (getattr(self, 'is_vector', False) or c == self.prim) else self.prim

    def _comp_options(self):
        """Fuer die aktuelle Datei sinnvolle Feldkomponenten. Vektor-Volumen: alle;
        sonst die gespeicherte Hauptkomponente + |E| (aus dem Zeitmittel <|E|^2> aller
        Komponenten) + bei TM-2D Ex (komplexe CW-Amplitude, Ex_cw_amp)."""
        if getattr(self, 'is3d', False) and getattr(self, 'is_vector', False):
            return ['Ez', 'Ex', 'Ey', '|E|']
        opts = [self.prim]
        if getattr(self, 'ex_amp2d', None) is not None and self.prim != 'Ex':
            opts.append('Ex')
        if getattr(self, 'iavg_full', False) and (self.pol == 'p' or getattr(self, 'is3d', False)):
            opts.append('|E|')                   # TE-2D: |E| == |Ez| -> keine eigene Option
        return opts

    def on_comp(self, label):
        if getattr(self, '_syncing', False):
            return                               # programmatischer Sync -> nicht feuern
        if label not in self._comp_options():
            print(f'  [Field] {label} is not stored in this file '
                  f'(available: {", ".join(self._comp_options())}) -> {self.prim}')
            self._comp_msg = (f'{label} is not stored in this file '
                              f'(available: {", ".join(self._comp_options())})')
            label = self.prim
            self._syncing = True
            try:
                self.radio_comp.set_active(['Ez', 'Ex', 'Ey', '|E|'].index(label))
            finally:
                self._syncing = False
        self.comp = label                        # 'Ez'/'Ex'/'Ey'/'|E|'
        self._comp_wish = label                  # beim Dateiwechsel beibehalten, falls vorhanden
        if getattr(self, 'is3d', False):
            self._busy(f'Switching to {label} ...')
            self._apply_slice()
        self._safe_redraw()

    def _sync_3d_controls(self):
        """Blendet die 3D-Schnitt-Regler (Ebene / Slice / Feld) passend zur
        AKTUELL geladenen Datei ein bzw. aus und stellt Slice-Bereich + aktive
        Ebene/Komponente ein. Wird am Ende jedes load() (und einmal in build_ui)
        aufgerufen - so tauchen die Regler auch auf, wenn man erst eine 2D-Datei
        und danach ein 3D-Volumen laedt (und verschwinden beim Rueckwechsel)."""
        if not hasattr(self, 's_slice'):
            return                               # UI noch nicht gebaut (1. load im __init__)
        is3d = getattr(self, 'is3d', False)
        isvec = is3d and getattr(self, 'is_vector', False)
        for ax in (self.ax_plane, self.ax_slice):
            ax.set_visible(is3d)
        # Feld-Auswahl sichtbar, sobald es mehr als eine Komponente gibt (auch 2D-TM)
        _opts = self._comp_options()
        self.ax_comp.set_visible(len(_opts) > 1)
        # Ausgeblendete Widgets nehmen in matplotlib trotzdem Klicks an -> abschalten
        for _w, _on in ((getattr(self, 'radio_plane', None), is3d),
                        (getattr(self, 's_slice', None), is3d),
                        (getattr(self, 'radio_comp', None), len(_opts) > 1)):
            if _w is not None:
                _w.active = bool(_on)
        # nicht gespeicherte Komponenten ausgrauen (statt still zurueckzuspringen)
        for _lbl in getattr(self, 'radio_comp', None).labels if hasattr(self, 'radio_comp') else []:
            _lbl.set_color('#dddddd' if _lbl.get_text() in _opts else '#555555')
        if getattr(self, 'comp', self.prim) not in _opts:
            self.comp = self.prim
        if not is3d:
            self._syncing = True
            try:
                self.radio_comp.set_active(['Ez', 'Ex', 'Ey', '|E|'].index(self.comp))
            except Exception:
                pass
            finally:
                self._syncing = False
            self.fig.canvas.draw_idle()
            return
        self._syncing = True                     # Widget-Updates nicht als User-Event werten
        try:
            # Ebenen-Radio auf aktuelle Ebene (load setzt plane='xy')
            try:
                self.radio_plane.set_active({'xy': 0, 'xz': 1, 'yz': 2}[self.plane])
            except Exception:
                pass
            # Slice-Slider-Bereich + Position an dieses Volumen anpassen
            smax = self._slice_max_idx()
            self.slice_idx = int(min(getattr(self, 'slice_idx', smax // 2), smax))
            try:
                self.s_slice.valmin = 0
                self.s_slice.valmax = smax
                self.s_slice.ax.set_xlim(0, smax)
                self.s_slice.label.set_text(self._SLICE_AXIS_LABEL[self.plane])
                self.s_slice.set_val(self.slice_idx)
                self._update_slice_valtext()
            except Exception:
                pass
            # Feldkomponente auf die aktive (gueltige) Komponente stellen
            if len(_opts) > 1:
                _cl = ['Ez', 'Ex', 'Ey', '|E|']
                try:
                    self.radio_comp.set_active(_cl.index(getattr(self, 'comp', self.prim)))
                except Exception:
                    pass
        finally:
            self._syncing = False
        self.fig.canvas.draw_idle()

    def build_ui(self):
        self.fig = plt.figure(figsize=(14, 8), facecolor='#0a0a0a')
        gs = self.fig.add_gridspec(
            nrows=6, ncols=8,
            width_ratios=[1.2, 1, 1, 1, 1, 1, 1, 0.2],
            height_ratios=[12, 0.35, 0.35, 0.35, 0.35, 0.35],
            hspace=0.6, wspace=0.3,
            left=0.05, right=0.97, top=0.94, bottom=0.06)
        self.ax = self.fig.add_subplot(gs[0, 1:])
        self.ax.set_facecolor('#0a0a0a')

        # Linke Steuerspalte: Mode (oben), darunter Ebene + Feld-Komponente (3D)
        ax_radio = plt.axes([0.008, 0.50, 0.108, 0.44], facecolor='#1a1a1a')
        ax_radio.set_title('Mode', color='#ffffff', fontsize=10)
        self.radio = RadioButtons(ax_radio, MODES, active=0, activecolor='#FF8800')
        for lbl in self.radio.labels:
            lbl.set_color('#dddddd')
            lbl.set_fontsize(9)
        self.radio.on_clicked(self.on_mode)

        ax_fr = self.fig.add_subplot(gs[1, 1:])
        ax_fr.set_facecolor('#1a1a1a')
        self.s_frame = Slider(ax_fr, 'Frame', 0, len(self.ezs)-1,
                              valinit=self.frame_idx, valstep=1,
                              color='#FF8800', track_color='#444444')
        self.s_frame.label.set_color('#dddddd')
        self.s_frame.valtext.set_color('#dddddd')
        self.s_frame.on_changed(self.on_frame)

        ax_x = self.fig.add_subplot(gs[3, 1:])
        ax_x.set_facecolor('#1a1a1a')
        # x-Slider deckt GENAU den tatsaechlich berechneten x-Bereich der Daten ab
        # (aus meta), nicht mehr eine feste +-100um-Skala. Schrittweite adaptiv.
        x_slider_min, x_slider_max, x_step = self._x_slider_range()
        self.x_um = max(x_slider_min, min(x_slider_max, self.x_um))
        self.s_x = Slider(ax_x, 'x [um]\n(Cut/Cursor)', x_slider_min, x_slider_max,
                          valinit=self.x_um, valstep=x_step,
                          color='#56C4FF', track_color='#444444')
        self.s_x.label.set_color('#dddddd')
        self.s_x.valtext.set_color('#dddddd')
        self.s_x.on_changed(self.on_x)

        # Fit-Top: Range -5..+1 um (auch oberhalb PS-Bot zum Vergleich)
        # Schrittweite 0.01 um = 10 nm
        ax_ft = self.fig.add_subplot(gs[4, 1:])
        ax_ft.set_facecolor('#1a1a1a')
        self.s_fit_top = Slider(ax_ft, 'Fit Top [um]\n(rel. PS-Bot)',
                                 -5.0, 1.0, valinit=self.fit_top, valstep=0.01,
                                 color='#FF4400', track_color='#444444')
        self.s_fit_top.label.set_color('#dddddd')
        self.s_fit_top.valtext.set_color('#dddddd')
        self.s_fit_top.on_changed(self.on_fit_top)

        # Fit-Depth: 0.05 .. 5 um, Schrittweite 0.01 um (10 nm)
        ax_fd = self.fig.add_subplot(gs[5, 1:])
        ax_fd.set_facecolor('#1a1a1a')
        self.s_fit_depth = Slider(ax_fd, 'Fit Depth [um]',
                                   0.05, 5.0, valinit=self.fit_depth, valstep=0.01,
                                   color='#FF4400', track_color='#444444')
        self.s_fit_depth.label.set_color('#dddddd')
        self.s_fit_depth.valtext.set_color('#dddddd')
        self.s_fit_depth.on_changed(self.on_fit_depth)

        # Datei-Navigation (<, >) + Laden IMMER verfuegbar, damit man auch zur
        # Laufzeit weitere Dateien fuer Compare/Diff dazuladen kann.
        ax_prev = plt.axes([0.005, 0.965, 0.026, 0.028])
        ax_next = plt.axes([0.033, 0.965, 0.026, 0.028])
        ax_load = plt.axes([0.061, 0.965, 0.065, 0.028])
        self.b_prev = Button(ax_prev, '<', color='#1a1a1a', hovercolor='#333333')
        self.b_next = Button(ax_next, '>', color='#1a1a1a', hovercolor='#333333')
        self.b_load = Button(ax_load, 'Load+', color='#12301a', hovercolor='#1f5030')
        for _b in (self.b_prev, self.b_next, self.b_load):
            _b.label.set_color('#dddddd'); _b.label.set_fontsize(8)
        self.b_prev.on_clicked(lambda e: self.switch_path(-1))
        self.b_next.on_clicked(lambda e: self.switch_path(+1))
        self.b_load.on_clicked(self.on_load)
        # Tastenkuerzel: l=Laden, a=aspect, ,/.=vor/zurueck
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # 3D-Volumen: Ebenen-Auswahl (links, unter Mode) + Positions-Slider
        # (als vollwertiger Slider direkt UNTER dem Frame-Slider, gs-Zeile 2).
        # 3D-Schnitt-Steuerung (Ebene / Slice / Feldkomponente) IMMER anlegen -
        # auch wenn die ERSTE Datei 2D ist. Sonst fehlen diese Regler dauerhaft,
        # sobald man spaeter ein 3D-Volumen dazulaedt (das war die Regression).
        # Die Sichtbarkeit wird pro geladener Datei via _sync_3d_controls() gesetzt.
        if not hasattr(self, 'plane'):
            self.plane = 'xy'
        if not hasattr(self, 'slice_idx'):
            self.slice_idx = 0
        self.ax_plane = plt.axes([0.008, 0.33, 0.108, 0.14], facecolor='#1a1a1a')
        self.ax_plane.set_title('Plane', color='#ffffff', fontsize=8)
        self.radio_plane = RadioButtons(
            self.ax_plane, ['xy length/cross-sec.', 'xz length/depth',
                            'yz cross-sec./depth'],
            active={'xy': 0, 'xz': 1, 'yz': 2}.get(self.plane, 0),
            activecolor='#39FF14')
        for lbl in self.radio_plane.labels:
            lbl.set_color('#dddddd'); lbl.set_fontsize(7)
        self.radio_plane.on_clicked(self.on_plane)
        self.ax_slice = self.fig.add_subplot(gs[2, 1:], facecolor='#1a1a1a')
        _smax = self._slice_max_idx() if getattr(self, 'is3d', False) else 1
        self.s_slice = Slider(self.ax_slice, self._SLICE_AXIS_LABEL[self.plane], 0,
                              _smax, valinit=min(self.slice_idx, _smax),
                              valstep=1, color='#39FF14', track_color='#444444')
        self.s_slice.label.set_color('#dddddd')
        self.s_slice.valtext.set_color('#dddddd')
        self.s_slice.on_changed(self.on_slice)
        # Feldkomponenten-Auswahl Ez/Ex/Ey/|E| (unter Ebene) - immer anlegen,
        # sichtbar nur bei Vektor-Volumen (Ex+Ey vorhanden).
        self.ax_comp = plt.axes([0.008, 0.19, 0.108, 0.12], facecolor='#1a1a1a')
        self.ax_comp.set_title('Field', color='#ffffff', fontsize=8)
        _cl = ['Ez', 'Ex', 'Ey', '|E|']
        self.radio_comp = RadioButtons(
            self.ax_comp, _cl,
            active=(_cl.index(self.prim) if getattr(self, 'prim', 'Ez') in _cl else 0),
            activecolor='#FF8800')
        for lbl in self.radio_comp.labels:
            lbl.set_color('#dddddd'); lbl.set_fontsize(8)
        self.radio_comp.on_clicked(self.on_comp)
        self._sync_3d_controls()   # Sichtbarkeit + Slice-Bereich fuer erste Datei

        # Farbskalen + Kontrast (Tasten c / v / k)
        self._i_seq = getattr(self, '_i_seq', 0); self._i_div = getattr(self, '_i_div', 0)
        self._i_con = getattr(self, '_i_con', 0)
        _bs = []
        for _y, _txt, _cb in ((0.140, self._lbl_seq(), self.on_cmap_seq),
                              (0.107, self._lbl_div(), self.on_cmap_div),
                              (0.074, self._lbl_con(), self.on_contrast)):
            _a = plt.axes([0.008, _y, 0.108, 0.027], facecolor='#1a1a1a')
            _b = Button(_a, _txt, color='#1a1a1a', hovercolor='#333333')
            _b.label.set_color('#dddddd'); _b.label.set_fontsize(7.5)
            _b.on_clicked(_cb); _bs.append(_b)
        self.b_cseq, self.b_cdiv, self.b_con = _bs

        # Bead-Position im Evaneszenz-Profil zuschalten (Taste 'b')
        self.show_bead_evan = getattr(self, 'show_bead_evan', False)
        ax_bd = plt.axes([0.008, 0.472, 0.108, 0.025], facecolor='#2a2010')
        self.b_bead = Button(ax_bd, self._lbl_bead(), color='#2a2010', hovercolor='#4a3818')
        self.b_bead.label.set_color('#ffd27f'); self.b_bead.label.set_fontsize(7.5)
        self.b_bead.on_clicked(self.on_bead_toggle)

        # Seitenverhaeltnis-Umschalter auto <-> equal (neben Laden+, Taste 'a')
        ax_asp = plt.axes([0.132, 0.965, 0.075, 0.028], facecolor='#1a1a1a')
        self.b_aspect = Button(ax_asp, f'aspect: {self.aspect_mode}',
                               color='#1a1a1a', hovercolor='#333333')
        self.b_aspect.label.set_color('#dddddd')
        self.b_aspect.label.set_fontsize(8)
        self.b_aspect.on_clicked(self.on_aspect)

        # Ansicht-Umschalter: einlaufend (transient) <-> eingeschwungen (CW).
        # Nur sinnvoll, wenn mind. 2 Ansichten vorhanden (sonst inaktiv).
        ax_view = plt.axes([0.008, 0.02, 0.115, 0.03], facecolor='#22331a')
        self.b_view = Button(ax_view, f'View: {getattr(self, "_view_cur", "-")}',
                             color='#22331a', hovercolor='#33501a')
        self.b_view.label.set_color('#bfe0a0'); self.b_view.label.set_fontsize(8)
        self.b_view.on_clicked(self.on_view_toggle)

        # Drehbare 3D-Ansicht (nur bei Volumen-Dateien sinnvoll, Taste '3')
        ax_v3d = plt.axes([0.212, 0.965, 0.085, 0.028], facecolor='#1a2233')
        self.b_view3d = Button(ax_v3d, '3D view', color='#1a2233',
                               hovercolor='#2a3a55')
        self.b_view3d.label.set_color('#9ecbff')
        self.b_view3d.label.set_fontsize(8)
        self.b_view3d.on_clicked(self.on_view3d)

        # Reset: geladene Datensaetze VERWERFEN und neu waehlen (Taste 'r').
        # (Laden+ haengt an, Reset ersetzt -> sauberes neues A/B-Paar.)
        # Leistungs-Skalierung: P [mW] (leer = relative Einheiten) und - nur 2D - angenommene
        # Tiefe in z [um] (2D rechnet pro Meter Tiefe). Enter uebernimmt.
        ax_pw = plt.axes([0.30, 0.008, 0.07, 0.03], facecolor='#1a1a1a')
        self.tb_power = FastTextBox(ax_pw, 'Power [mW] ', initial=(f'{POWER["mw"]:g}' if POWER.get('mw') else ''),
                                color='#1a1a1a', hovercolor='#333333')
        ax_dz = plt.axes([0.50, 0.008, 0.06, 0.03], facecolor='#1a1a1a')
        self.tb_depth = FastTextBox(ax_dz, 'Depth z (2D only) [um] ', initial=f'{POWER["depth_um"]:g}',
                                color='#1a1a1a', hovercolor='#333333')
        for _tb in (self.tb_power, self.tb_depth):
            # matplotlib 3.11: TextBox._resize laeuft durch einen Dekorator, der event.inaxes
            # liest - ein ResizeEvent hat das nicht -> Traceback bei jeder Fenster-
            # groessenaenderung. Den Resize-Callback (zuletzt verbunden) ersetzen.
            try:
                self.fig.canvas.mpl_disconnect(_tb._cids.pop())
                self.fig.canvas.mpl_connect('resize_event', lambda _e, t=_tb: t.stop_typing())
            except Exception:
                pass
            _tb.label.set_color('#dddddd'); _tb.label.set_fontsize(8)
            _tb.text_disp.set_color('#ffd27f'); _tb.text_disp.set_fontsize(8)
        self.tb_power.on_submit(self.on_power)
        self.tb_depth.on_submit(self.on_depth)

        def _revert(tb, txt):
            """Left the box without Enter -> show the value that is in effect."""
            tb.eventson = False
            try:
                tb.set_val(txt())
                tb.cursor.set_visible(False)
            finally:
                tb.eventson = True
        self.tb_power.on_cancel = lambda: _revert(
            self.tb_power, lambda: f'{POWER["mw"]:g}' if POWER.get('mw') else '')
        self.tb_depth.on_cancel = lambda: _revert(
            self.tb_depth, lambda: f'{POWER["depth_um"]:g}')
        # Input boxes right of the x / fit sliders (type an exact value + Enter);
        # they replace the sliders' value text and follow the sliders.
        self._slider_inputs = {}
        for _sl, _fmt in ((self.s_x, '{:.2f}'), (self.s_fit_top, '{:.2f}'),
                          (self.s_fit_depth, '{:.2f}')):
            self._add_slider_input(_sl, _fmt)
        self._bead_hint = self.fig.text(0.86, 0.016, '', color='#39FF14', fontsize=7.5)
        self._update_bead_hint()
        self._power_hint = self.fig.text(0.57, 0.016, '', color='#aaaaaa', fontsize=7.5)

        ax_reset = plt.axes([0.302, 0.965, 0.070, 0.028], facecolor='#301616')
        self.b_reset = Button(ax_reset, 'Reset', color='#301616',
                              hovercolor='#5a2626')
        self.b_reset.label.set_color('#ffb3b3')
        self.b_reset.label.set_fontsize(8)
        self.b_reset.on_clicked(self.on_reset)

    # ---- Leistungs-Skalierung ----
    def _apply_power(self):
        """Neue Leistung/Tiefe: aktuelle Datei neu laden (Arrays skaliert), Referenz-
        Cache leeren (Diff/Streufeld nutzen die Leistung der Referenzdatei), neu zeichnen."""
        fi = self.frame_idx
        self._ref_cache = {}
        self.load()
        self.frame_idx = min(fi, len(self.ezs) - 1)
        self._update_power_hint()
        self._safe_redraw()

    def _update_power_hint(self):
        h = getattr(self, '_power_hint', None)
        if h is None:
            return
        _mwf = getattr(self, 'p_laser_mw', None)
        if not POWER.get('mw') and not _mwf:
            t = 'empty = relative units'
        elif getattr(self, '_p_sim_w', None):
            t = (f'laser {POWER.get("mw") or _mwf:g} mW' + ('' if POWER.get('mw') else ' (from run)')
                 + f': P_sim = {self._p_sim_w:.3g} W -> fields x{self._escale:.3g} (V/m), '
                 f'intensities x{self._escale**2:.3g}')
        else:
            t = 'this file has no power info (older run) -> unscaled'
        h.set_text(t)

    def on_power(self, text):
        t = str(text).strip().replace(',', '.')
        try:
            POWER['mw'] = float(t) if t else None
            if POWER['mw'] is not None and POWER['mw'] <= 0:
                POWER['mw'] = None
        except ValueError:
            print(f'  [Power] "{text}" is not a number')
            return
        self._apply_power()

    def on_depth(self, text):
        try:
            v = float(str(text).strip().replace(',', '.'))
            if v > 0:
                POWER['depth_um'] = v
        except ValueError:
            print(f'  [Power] depth "{text}" is not a number')
            return
        if POWER.get('mw'):
            self._apply_power()

    # ---- Bead im Evaneszenz-Profil ----
    def _lbl_bead(self):
        return f'Bead in profile: {"on" if getattr(self, "show_bead_evan", False) else "off"}'

    def on_bead_toggle(self, _evt=None):
        self.show_bead_evan = not getattr(self, 'show_bead_evan', False)
        try:
            self.b_bead.label.set_text(self._lbl_bead())
        except Exception:
            pass
        if self.mode == 'Evanescent':
            self._safe_redraw()
        else:
            self.fig.canvas.draw_idle()

    def _bead_evan_marks(self, y_bot_wg):
        """Bead im Evaneszenz-Plot (Abszisse = y): seine y-Ausdehnung an der aktuellen
        x-Position (bzw. am Schnitt durch die Kugel) als Band, Mitte gestrichelt, und
        in der Legende der Abstand Bead-Oberkante <-> WG-Unterseite sowie, ob der
        x-Cursor durch den Bead geht (dann ist das Profil KEIN reines Abklingen)."""
        if not getattr(self, 'is_bead', False) or getattr(self, 'bead_d', 0.0) <= 0:
            return
        r = self.bead_d/2.0
        yc = getattr(self, 'bead_y', -r)
        xc = getattr(self, 'bead_x', 0.0)
        # 3D: Kugelschnitt in der aktuellen z-Ebene (xy); 2D/Schnittdatei: volle Kugel
        rr = r
        if getattr(self, 'is3d', False) and getattr(self, 'plane', 'xy') == 'xy':
            dz = self._slice_pos_um()[0] - getattr(self, 'bead_z', 0.0)
            rr = float(np.sqrt(max(r*r - dz*dz, 0.0)))
            if rr <= 0:
                self.ax.axvline(yc, color='#FFB000', ls=':', lw=1.0, alpha=0.6,
                                label=f'Bead (not in this z plane, |dz|={abs(dz):.2f} um)')
                return
        dx_c = self.x_um - xc
        # Ausdehnung in y an der Cursor-Position x (Sehne der Kugel/des Kreises)
        half = float(np.sqrt(max(rr*rr - dx_c*dx_c, 0.0)))
        gap = y_bot_wg - (yc + rr)
        self.ax.axvspan(yc - rr, yc + rr, color='#FFB000', alpha=0.10, zorder=0)
        if half > 0:
            self.ax.axvspan(yc - half, yc + half, color='#FFB000', alpha=0.25, zorder=0)
            where = f'x cursor CUTS the bead ({2*half:.2f} um chord) -> profile through the sphere'
        else:
            where = f'x cursor {abs(dx_c):.2f} um beside the bead (x_bead={xc:.1f} um)'
        self.ax.axvline(yc, color='#FFB000', ls='--', lw=1.0, alpha=0.8,
                        label=(f'Bead d={2*r:.2f} um, y={yc - rr:.2f}..{yc + rr:.2f} um, '
                               f'gap to WG {max(gap, 0.0)*1000:.0f} nm\n{where}'))

    # ---- Farbskalen / Kontrast ----
    def _cm_seq(self):
        return CMAPS_SEQ[getattr(self, '_i_seq', 0) % len(CMAPS_SEQ)]

    def _cm_div(self):
        return CMAPS_DIV[getattr(self, '_i_div', 0) % len(CMAPS_DIV)]

    def _sat(self):
        return CONTRAST[getattr(self, '_i_con', 0) % len(CONTRAST)]

    def _lbl_seq(self):
        return f'Color |E|: {self._cm_seq()}'

    def _lbl_div(self):
        return f'Color +/-: {self._cm_div()}'

    def _lbl_con(self):
        return f'Contrast: {1/self._sat():g}x'

    def _cycle(self, attr, btn, lbl_fn):
        setattr(self, attr, getattr(self, attr, 0) + 1)
        try:
            btn.label.set_text(lbl_fn())
        except Exception:
            pass
        self._safe_redraw()

    def on_cmap_seq(self, event=None):
        self._cycle('_i_seq', getattr(self, 'b_cseq', None), self._lbl_seq)

    def on_cmap_div(self, event=None):
        self._cycle('_i_div', getattr(self, 'b_cdiv', None), self._lbl_div)

    def on_contrast(self, event=None):
        self._cycle('_i_con', getattr(self, 'b_con', None), self._lbl_con)

    def on_aspect(self, event):
        self.aspect_mode = 'equal' if self.aspect_mode == 'auto' else 'auto'
        try:
            self.b_aspect.label.set_text(f'aspect: {self.aspect_mode}')
        except Exception:
            pass
        self._safe_redraw()

    def on_load(self, event=None):
        """Oeffnet einen Datei-Dialog und fuegt weitere NPZ zur Sitzung hinzu
        (fuer Compare/Diff). Danach ist die zuletzt geladene die 'aktuelle'."""
        try:
            new = pick_files_dialog(self._tk_parent())
        except Exception as e:
            print(f'[Load] file dialog not available ({e}). '
                  f'Pass the files as arguments at startup instead.')
            return
        added = 0
        for p in new:
            if p and os.path.exists(p) and p not in self.paths:
                self.paths.append(p)
                added += 1
        if not added:
            print('[Load] nothing new added.')
            return
        print(f'[Load] {added} file(s) added -> {len(self.paths)} loaded.')
        self._ref_cache = {}                 # Diff-Referenz-Cache invalidieren
        self.cur_path = len(self.paths) - 1  # zur neu geladenen wechseln
        self.load()
        self.frame_idx = min(self.frame_idx, len(self.ezs)-1)
        try:
            self.s_frame.valmax = len(self.ezs)-1
            self.s_frame.ax.set_xlim(0, len(self.ezs)-1)
            self.s_frame.set_val(self.frame_idx)
        except Exception:
            pass
        self._update_x_slider_range()
        self.redraw()

    def _tk_parent(self):
        """Tk-Toplevel des Analyzer-Fensters (TkAgg) fuer Dialoge, sonst None."""
        try:
            return self.fig.canvas.get_tk_widget().winfo_toplevel()
        except Exception:
            return None

    def on_reset(self, event=None):
        """Verwirft ALLE geladenen Datensaetze und laedt eine neue Auswahl
        (ersetzt statt anzuhaengen) - fuer ein sauberes neues A/B-Paar."""
        try:
            new = pick_files_dialog(self._tk_parent())
        except Exception as e:
            print(f'[Reset] file dialog not available ({e}).')
            return
        new = [p for p in new if p and os.path.exists(p)]
        if not new:
            print('[Reset] no files selected - nothing changed.')
            return
        self.paths = new
        self.cur_path = 0
        self._ref_cache = {}
        self.load()
        self.frame_idx = min(self.frame_idx, len(self.ezs)-1)
        try:
            self.s_frame.valmax = len(self.ezs)-1
            self.s_frame.ax.set_xlim(0, len(self.ezs)-1)
            self.s_frame.set_val(self.frame_idx)
        except Exception:
            pass
        self._update_x_slider_range()
        self.redraw()
        print(f'[Reset] {len(self.paths)} file(s) reloaded: '
              f'{[os.path.basename(p) for p in self.paths]}')

    def _typing(self):
        """True while one of the input boxes has the keyboard (then no shortcuts:
        ',' '.' '3' ... are text, not 'previous file' / 'next file' / '3D view')."""
        boxes = [getattr(self, 'tb_power', None), getattr(self, 'tb_depth', None)]
        boxes += list(getattr(self, '_slider_inputs', {}).values())
        return any(b is not None and b.capturekeystrokes for b in boxes)

    def _on_key(self, event):
        if self._typing():
            return
        k = getattr(event, 'key', None)
        if k == 'l':
            self.on_load()
        elif k == 'r':
            self.on_reset()
        elif k == 'a':
            self.on_aspect(None)
        elif k in ('.', 'right'):
            self.switch_path(+1)
        elif k in (',', 'left'):
            self.switch_path(-1)
        elif k == '3':
            self.on_view3d()
        elif k == 'c':
            self.on_cmap_seq()
        elif k == 'v':
            self.on_cmap_div()
        elif k == 'k':
            self.on_contrast()
        elif k == 'b':
            self.on_bead_toggle()

    @staticmethod
    def _wire_box3d(ax, x0, x1, y0, y1, z0, z1, color):
        import numpy as _np
        p = _np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                       [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                     (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            ax.plot(*zip(p[a], p[b]), color=color, lw=1.1, alpha=0.9)

    @staticmethod
    def _wire_sphere3d(ax, cx, cy, cz, r, color):
        import numpy as _np
        u = _np.linspace(0, 2*_np.pi, 18); v = _np.linspace(0, _np.pi, 10)
        x = cx + r*_np.outer(_np.cos(u), _np.sin(v))
        y = cy + r*_np.outer(_np.sin(u), _np.sin(v))
        z = cz + r*_np.outer(_np.ones_like(u), _np.cos(v))
        ax.plot_wireframe(x, y, z, color=color, lw=0.8, alpha=0.9)

    def _layer_planes3d(self, ax, x0, x1, z0, z1, y_lo=None, y_hi=None):
        """Zeichnet die Material-Schicht-Grenzen (WG-Top/Bot, Aqueous, Mucin,
        Cornea) als transparente horizontale Ebenen im 3D-Raum zur Orientierung."""
        try:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except Exception:
            return
        t_wg = getattr(self, 't_wg_um', 0.0) or 5.0
        cn, sn = self.core_name, self.sub_name
        levels = [(t_wg, '#FFD75E', f'{cn}-Top'),
                  (0.0, '#FFD75E', f'{cn}-Bot' + ('' if self.stack else f' -> {sn}'))]
        nst = len(self.stack)
        for k, ((nm, _yt, yb, _n), L) in enumerate(zip(self._stack_bounds(0.0), self.stack)):
            if k == nst - 1:
                lab = f'{nm} -> {sn}'
            elif L.get('slot') == 'lipid':
                lab = nm
            elif L.get('slot') == 'aqueous':
                lab = f'{nm} ({L["t_um"]:.1f}um)'
            else:
                lab = f'{nm} ({L["t_um"]:.2g}um)'
            levels.append((yb, L['color'], lab))
        for yv, col, lab in levels:
            if (y_lo is not None and yv < y_lo - 1e-9) or (y_hi is not None and yv > y_hi + 1e-9):
                continue                        # ausserhalb des dargestellten Bereichs
            quad = [[(x0, yv, z0), (x1, yv, z0), (x1, yv, z1), (x0, yv, z1)]]
            pc = Poly3DCollection(quad, alpha=0.10)
            pc.set_facecolor(col); pc.set_edgecolor(col)
            ax.add_collection3d(pc)
            ax.text(x1, yv, z1, ' '+lab, color=col, fontsize=7)

    def on_view3d(self, event=None):
        """Oeffnet ein SEPARATES, mit der Maus drehbares 3D-Fenster mit dem Feld-Volumen
        (Punktwolke ueber einer LOGARITHMISCHEN Schwelle in dB unter dem Maximum, log.
        Farbskala) + WG-Box + Schichten + Bead. Im Fenster: Schwelle [dB], Frame-Regler
        (zeitliche Propagation bzw. CW-Phase), Differenz A-B (diese Datei minus die
        naechste geladene = Bead - Referenz), Ausschnitt 'um den Bead' in voller
        Aufloesung und feste Blickrichtungen (Iso / x-y / x-z / y-z).
        Speicher: es wird nur das ausgeduennte bzw. ausgeschnittene Teilvolumen gehalten;
        die Referenz wird Frame fuer Frame aus der npz gestreamt (nie ganz geladen)."""
        if not getattr(self, 'is3d', False):
            msg = ('3D view only for volume files (_vol3d.npz).\n'
                   'The loaded file is a slice/2D file (_frames.npz):\n'
                   'Load+ (key l) -> select the matching _vol3d.npz.')
            print('[3D view] ' + msg.replace('\n', ' '))
            self.ax.text(0.5, 0.5, msg, color='#FFAA55', ha='center', va='center', fontsize=10,
                         transform=self.ax.transAxes, zorder=30,
                         bbox=dict(facecolor='#1a1a1a', edgecolor='#FFAA55', alpha=0.9))
            self.fig.canvas.draw_idle()
            return
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception as e:
            print(f'[3D view] 3D backend not available: {e}')
            return
        from matplotlib.widgets import Slider as _Slider, Button as _Button
        cw_view = str(getattr(self, '_view_cur', '')).startswith('Steady state')
        comp = self._disp_comp()
        if cw_view or not getattr(self, 'is_vector', False):
            comp = self.prim                       # CW-Volumen = nur Hauptkomponente
        nf = int(self.vol.shape[0])
        Nx, Ny, Nz = self.vol.shape[1:]
        dx = self.dx_um
        t_wg = getattr(self, 't_wg_um', 0.0) or 5.0
        st = {'fi': int(min(self.frame_idx, nf - 1)), 'db': -15.0, 'diff': False, 'crop': False,
              'view': None, 'view_set': 'Iso',
              'lim': None,      # Zoom: (xa, xb, ya, yb, za, zb) in um, bleibt bei Slidern erhalten
              'zreg': None,     # 'Fein rastern': Index-Bereich aus dem Zoom
              'scat': False}    # scattered field |E_s| from the CW phase frames
        has_bead = bool(getattr(self, 'is_bead', False) and self.bead_d > 0)

        def region(crop):
            """Index-Bereich + Schrittweite des dargestellten Teilvolumens."""
            if st['zreg'] is not None:
                i0, i1, j0, j1, k0, k1 = st['zreg']
            elif crop and has_bead:
                r = self.bead_d/2.0
                byc = getattr(self, 'bead_y', -r)
                bzc = getattr(self, 'bead_z', 0.0)

                def _i(u, o, n):
                    return int(np.clip(round((u - o)/dx), 0, n))
                i0, i1 = _i(self.bead_x - 2.5, self.x0_um, Nx), _i(self.bead_x + 2.5, self.x0_um, Nx)
                # y: 2.5 um unter den Bead bis ueber die GANZE Kernhoehe (frueher endete der
                # Ausschnitt 2 um ueber dem Bead -> Feld im Kern wirkte 'abgeschnitten')
                j0, j1 = _i(byc - 2.5, self.y0_um, Ny), _i(t_wg + 0.3, self.y0_um, Ny)
                k0, k1 = _i(bzc - 2.5, self.z0_um, Nz), _i(bzc + 2.5, self.z0_um, Nz)
            else:
                i0, i1, j0, j1, k0, k1 = 0, Nx, 0, Ny, 0, Nz
            n = max(1, (i1 - i0)*(j1 - j0)*(k1 - k0))
            s = max(1, int(np.ceil((n/1.5e6)**(1/3))))
            return (i0, i1, j0, j1, k0, k1, s)

        def _sl(rg):
            i0, i1, j0, j1, k0, k1, s = rg
            return (slice(i0, i1, s), slice(j0, j1, s), slice(k0, k1, s))

        def own_sub(fi, rg):
            """Teilvolumen dieser Datei (aktuelle Ansicht/Komponente), float32."""
            sl = (fi,) + _sl(rg)
            if comp == '|E|' and getattr(self, 'is_vector', False):
                return np.sqrt(self.vol[sl].astype(np.float32)**2 + self.vol_ex[sl].astype(np.float32)**2
                               + self.vol_ey[sl].astype(np.float32)**2), False
            src = None
            if getattr(self, 'is_vector', False) and not cw_view:
                src = {'Ex': getattr(self, 'vol_ex', None), 'Ey': getattr(self, 'vol_ey', None)}.get(comp)
            return np.array((src if src is not None else self.vol)[sl], dtype=np.float32), True

        _subs = {}       # (diff, crop) -> Teilvolumina je Frame (klein)

        def ref_subs(rg, force_cw=False):
            """Referenz-Teilvolumina aller Frames, Frame fuer Frame aus der npz gestreamt
            (die ganze Referenz waeren hier 2x 760 MB -> MemoryError)."""
            rpath = self.paths[(self.cur_path + 1) % len(self.paths)]
            with np.load(rpath) as d:
                _sc, _ = power_scale(*_npz_power(d))
                files = set(d.files)
            if cw_view or force_cw:
                names = ['Ez_cw']
            elif comp in ('Ex', 'Ey') and getattr(self, 'is_vector', False):
                names = [comp]                      # wie own_sub: vol_ex / vol_ey
            elif comp == '|E|':
                names = ['Ez', 'Ex', 'Ey']
            else:
                names = ['Ez']
            miss = [n for n in names if n not in files]
            if miss:
                raise ValueError(f'Reference lacks {miss} (view/component missing)')
            acc = None
            for n in names:
                parts = []
                for arr in _npz_stream_frames(rpath, n):
                    if arr.ndim != 3:
                        raise ValueError('Reference is not a volume file (_vol3d.npz)')
                    if arr.shape != (Nx, Ny, Nz):
                        raise ValueError(f'Shapes do not match: {(Nx, Ny, Nz)} vs reference {arr.shape}')
                    parts.append(np.array(arr[_sl(rg)], dtype=np.float32)*np.float32(_sc))
                    del arr
                if len(names) == 1:
                    acc = parts
                elif acc is None:
                    acc = [p**2 for p in parts]
                else:
                    acc = [a + p**2 for a, p in zip(acc, parts)]
            if len(names) > 1:
                acc = [np.sqrt(a) for a in acc]
            return acc

        def scat_subs(rg):
            """|E_s| = |E^(bead) - E^(ref)| from the steady-state CW phase frames of both
            files (complex amplitude by DFT over the phases), on the shown sub-volume."""
            if len(self.paths) < 2:
                raise ValueError('Scattered field needs a 2nd file (reference)')
            vcw = getattr(self, '_views', {}).get('Steady state')
            if not vcw:
                raise ValueError('This file has no steady-state CW volume (Ez_cw)')
            cw = vcw[1]
            own = np.stack([np.array(cw[(f,) + _sl(rg)], dtype=np.float32)
                            for f in range(cw.shape[0])])
            ref = np.stack(ref_subs(rg, force_cw=True))
            if ref.shape[1:] != own.shape[1:]:
                raise ValueError(f'Shapes do not match: {own.shape[1:]} vs reference {ref.shape[1:]}')
            Es = np.abs(self._phasor(own) - self._phasor(ref)).astype(np.float32)
            return [Es]

        def subs():
            key = (st['diff'], st['crop'], st['zreg'], st['scat'])
            if key not in _subs and st['scat']:
                rg = region(st['crop'])
                vols = scat_subs(rg)
                _subs.clear()
                _subs[key] = (rg, vols, False, float(vols[0].max()) or 1.0)
            if key not in _subs:
                rg = region(st['crop'])
                own = [own_sub(f, rg) for f in range(nf)]
                vols = [o[0] for o in own]
                signed = own[0][1]
                if st['diff']:
                    ref = ref_subs(rg)
                    vols = [v - ref[min(f, len(ref) - 1)] for f, v in enumerate(vols)]
                    signed = True
                vmax = max(float(np.abs(v).max()) for v in vols) or 1.0
                _subs.clear()                       # nur einen Satz halten
                _subs[key] = (rg, vols, signed, vmax)
            return _subs[key]

        _old = getattr(self, '_v3d_fig', None)
        if _old is not None:
            try:
                plt.close(_old)
            except Exception:
                pass
        fig = plt.figure(figsize=(10, 8.5), facecolor='#0a0a0a')
        fig.canvas.mpl_connect('key_press_event', self._on_key)
        ax = fig.add_axes([0.13, 0.12, 0.86, 0.78], projection='3d', facecolor='#0a0a0a')
        # 2D axes for the axis views (x-y / x-z / y-z): projection image instead of a point
        # cloud -> no grid dots, no moire, sharp at any zoom
        ax2 = fig.add_axes([0.25, 0.20, 0.72, 0.67], facecolor='#0a0a0a')
        ax2.set_visible(False)
        VIEWS = {'Iso': (25, -60, 'persp'), 'x-y': (90, -90, 'ortho'),
                 'x-z': (0, -90, 'ortho'), 'y-z': (0, 0, 'ortho')}
        DEPTH = {'x-y': 2, 'x-z': 1, 'y-z': 0}      # viewing axis of each axis view
        AXLAB = ['Length x (um)', 'Cross-section y (um)', 'Depth z (um)']
        _jit = np.random.default_rng(12345)

        def _full_lim(rg):
            i0, i1, j0, j1, k0, k1, _s = rg
            return (self.x0_um + i0*dx, self.x0_um + i1*dx, self.y0_um + j0*dx,
                    self.y0_um + j1*dx, self.z0_um + k0*dx, self.z0_um + k1*dx)

        def _mode2d():
            return st['view_set'] in DEPTH

        def _colors(vals, lvl, vmax, signed, alpha_max):
            """Colour by log strength frac in [0, 1] (0 = threshold, 1 = maximum): weak =
            dark + transparent, strong = bright + opaque. Sign: + red->yellow,
            - blue->cyan (no white centre as with RdBu)."""
            av = np.abs(vals).astype(np.float64)
            rng = np.log10(max(vmax/lvl, 1.0000001))
            frac = np.clip(np.log10(np.maximum(av, lvl)/lvl)/rng, 0.0, 1.0)
            rgba = np.zeros(av.shape + (4,))
            if signed:
                lo_p, hi_p = np.array([0.45, 0.06, 0.06]), np.array([1.0, 0.92, 0.35])
                lo_n, hi_n = np.array([0.06, 0.12, 0.50]), np.array([0.45, 0.95, 1.0])
                f3 = frac[..., None]
                pos = (vals >= 0)[..., None]
                rgba[..., :3] = np.where(pos, lo_p + (hi_p - lo_p)*f3, lo_n + (hi_n - lo_n)*f3)
            else:
                _cm = self._cm_seq()
                _cm = plt.get_cmap(_cm) if isinstance(_cm, str) else _cm
                rgba[:] = _cm(0.2 + 0.8*frac)
            rgba[..., 3] = alpha_max*(0.25 + 0.75*frac)
            return rgba, frac

        def _layer_levels():
            cn, sn = self.core_name, self.sub_name
            lv = [(t_wg, '#FFD75E', f'{cn} top'),
                  (0.0, '#FFD75E', f'{cn} bottom' + ('' if self.stack else f' -> {sn}'))]
            nst = len(self.stack)
            for k, ((nm, _yt, yb, _n), L) in enumerate(zip(self._stack_bounds(0.0), self.stack)):
                lv.append((yb, L['color'], f'{nm} -> {sn}' if k == nst - 1 else nm))
            return lv

        def _draw2d(sub, rg, lvl, vmax, signed):
            """Signed max-|value| projection along the viewing axis inside the zoom box."""
            i0, i1, j0, j1, k0, k1, s = rg
            d = DEPTH[st['view_set']]
            lim = st['lim']
            org = (self.x0_um + i0*dx, self.y0_um + j0*dx, self.z0_um + k0*dx)
            sl = []
            for n_ in range(3):                     # index range of the zoom box
                lo = int(np.clip(np.floor((lim[2*n_] - org[n_])/(s*dx)), 0, sub.shape[n_]))
                hi = int(np.clip(np.ceil((lim[2*n_ + 1] - org[n_])/(s*dx)) + 1, lo + 1,
                                 sub.shape[n_]))
                sl.append(slice(lo, hi))
            box = sub[tuple(sl)]
            if box.size == 0:
                return 0
            idx = np.argmax(np.abs(box), axis=d)
            proj = np.take_along_axis(box, np.expand_dims(idx, d), axis=d).squeeze(d)
            h, v = [n_ for n_ in range(3) if n_ != d]   # horizontal, vertical axis
            img = proj.T                            # rows = vertical axis
            rgba, _f = _colors(img, lvl, vmax, signed, 1.0)
            rgba[..., 3] = np.where(np.abs(img) >= lvl, 1.0, 0.0)
            ext = [org[h] + sl[h].start*s*dx - 0.5*s*dx, org[h] + (sl[h].stop - 1)*s*dx + 0.5*s*dx,
                   org[v] + sl[v].start*s*dx - 0.5*s*dx, org[v] + (sl[v].stop - 1)*s*dx + 0.5*s*dx]
            ax2.clear()
            ax2.set_facecolor('#0a0a0a')
            ax2.imshow(rgba, origin='lower', extent=ext, interpolation='bilinear',
                       aspect='equal', zorder=1)
            # overlays: layer / waveguide lines (y), bead outline
            if 1 in (h, v):
                for yv, col, lab in _layer_levels():
                    if lim[2] - 1e-9 <= yv <= lim[3] + 1e-9:
                        if v == 1:
                            ax2.axhline(yv, color=col, lw=0.8, alpha=0.7, zorder=2)
                            ax2.text(lim[2*h + 1], yv, ' ' + lab, color=col, fontsize=7,
                                     va='center', ha='left', clip_on=False)
                        else:
                            ax2.axvline(yv, color=col, lw=0.8, alpha=0.7, zorder=2)
                            ax2.text(yv, lim[2*v], ' ' + lab, color=col, fontsize=7, rotation=90,
                                     va='bottom', ha='right', clip_on=True, zorder=4)
            if has_bead:
                bc = (self.bead_x, getattr(self, 'bead_y', -self.bead_d/2), getattr(self, 'bead_z', 0.0))
                ax2.add_patch(Circle((bc[h], bc[v]), self.bead_d/2, fill=False, ec='#39FF14',
                                     lw=1.2, zorder=3))
            ax2.set_xlim(lim[2*h], lim[2*h + 1])
            ax2.set_ylim(lim[2*v], lim[2*v + 1])
            ax2.set_xlabel(AXLAB[h], color='#ccc')
            ax2.set_ylabel(AXLAB[v], color='#ccc')
            ax2.tick_params(colors='#aaa')
            for sp in ax2.spines.values():
                sp.set_color('#444444')
            return int(np.count_nonzero(np.abs(img) >= lvl))

        def _draw3d(sub, rg, lvl, vmax, signed, el, az):
            i0, i1, j0, j1, k0, k1, s = rg
            xa, xb, ya, yb, za, zb = st['lim']
            ii, jj, kk = np.where(np.abs(sub) >= lvl)
            xs = self.x0_um + (i0 + ii*s)*dx
            ys = self.y0_um + (j0 + jj*s)*dx
            zs = self.z0_um + (k0 + kk*s)*dx
            vals = sub[ii, jj, kk]
            m = (xs >= xa) & (xs <= xb) & (ys >= ya) & (ys <= yb) & (zs >= za) & (zs <= zb)
            xs, ys, zs, vals = xs[m], ys[m], zs[m], vals[m]   # mplot3d does not clip
            npts = int(xs.size)
            if npts > 120000:                       # thin out evenly
                sel = np.linspace(0, npts - 1, 120000).astype(int)
                xs, ys, zs, vals = xs[sel], ys[sel], zs[sel], vals[sel]
            # random shift by up to half a grid cell: breaks the regular lattice -> no moire
            hw = 0.5*s*dx
            xs = xs + _jit.uniform(-hw, hw, xs.size)
            ys = ys + _jit.uniform(-hw, hw, ys.size)
            zs = zs + _jit.uniform(-hw, hw, zs.size)
            ax.clear()
            ax.set_facecolor('#0a0a0a')
            for _pa in (ax.xaxis, ax.yaxis, ax.zaxis):   # dark panes instead of grey
                _pa.set_pane_color((0.09, 0.09, 0.10, 1.0))
                _pa._axinfo['grid']['color'] = (0.35, 0.35, 0.35, 0.6)
            ax.view_init(elev=el, azim=az)
            amax = float(np.clip(0.5*np.sqrt(20000.0/max(xs.size, 1)), 0.12, 0.9))
            rgba, frac = _colors(vals, lvl, vmax, signed, amax)
            span = max(xb - xa, yb - ya, zb - za)
            psize = float(np.clip(15.0*(s*dx/span*100.0)**2, 3.0, 80.0))
            order = np.argsort(frac)                # strong points drawn last (on top)
            ax.scatter(xs[order], ys[order], zs[order], c=rgba[order], s=psize,
                       linewidths=0, depthshade=False)
            if max(0.0, ya) < min(t_wg, yb):
                self._wire_box3d(ax, xa, xb, max(0.0, ya), min(t_wg, yb), za, zb, '#FFD75E')
            self._layer_planes3d(ax, xa, xb, za, zb, ya, yb)
            if has_bead:
                bx, by = self.bead_x, getattr(self, 'bead_y', -self.bead_d/2)
                bz = getattr(self, 'bead_z', 0.0)
                if xa <= bx <= xb and ya <= by <= yb and za <= bz <= zb:
                    self._wire_sphere3d(ax, bx, by, bz, self.bead_d/2, '#39FF14')
            ax.set_xlim(xa, xb)
            ax.set_ylim(ya, yb)
            ax.set_zlim(za, zb)
            for a_, l_ in zip((ax.xaxis, ax.yaxis, ax.zaxis), AXLAB):
                a_.set_label_text(l_, color='#ccc')
            ax.tick_params(colors='#aaa')
            ax.set_proj_type('persp')
            try:
                ax.set_box_aspect((xb - xa, yb - ya, zb - za), zoom=0.95)
            except Exception:
                pass
            return npts

        def _render(_=None):
            el, az = ax.elev, ax.azim               # keep mouse rotation
            if st['view'] is not None:              # view button: apply once
                st['view_set'] = st['view']
                st['view'] = None
                if st['view_set'] == 'Iso':
                    el, az = VIEWS['Iso'][0], VIEWS['Iso'][1]
            m2 = _mode2d()
            tgt = ax2 if m2 else ax
            if (st['diff'], st['crop'], st['zreg'], st['scat']) not in _subs:
                _busy3d.set_text(('Computing scattered field (streaming reference) ...'
                                             if st['scat'] else 'Streaming reference ...'
                                             if st['diff'] else 'Loading volume ...'))
                _busy3d.set_visible(True)
                try:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                except Exception:
                    pass
            ax.set_visible(not m2)
            ax2.set_visible(m2)
            try:
                rg, vols, signed, vmax = subs()
            except Exception as e:
                _busy3d.set_visible(False)
                tgt.clear()
                tgt.set_title(f'Error: {e}', color='#FF5555', fontsize=10)
                fig.canvas.draw_idle()
                return
            _busy3d.set_visible(False)
            s = rg[6]
            full = _full_lim(rg)
            if st['lim'] is None:
                st['lim'] = full
            sub = vols[min(st['fi'], len(vols) - 1)]
            lvl = vmax*10**(st['db']/20.0)
            if m2:
                npts = _draw2d(sub, rg, lvl, vmax, signed)
                what_pts = (f'{npts} pixels, max-|field| projection along '
                            f'{"xyz"[DEPTH[st["view_set"]]]}{" (sign kept)" if signed else ""}')
            else:
                npts = _draw3d(sub, rg, lvl, vmax, signed, el, az)
                what_pts = f'{npts} points'
            if cw_view:
                _times = getattr(self, '_times', [])
                _t = float(_times[st['fi']]) if st['fi'] < len(_times) else 0.0
            else:
                _stp = getattr(self, '_steps', [])
                _dts = getattr(self, '_dt_s', None)
                _t = (float(_stp[st['fi']])*_dts) if (_dts and st['fi'] < len(_stp)) else 0.0
            _ref = os.path.basename(self.paths[(self.cur_path + 1) % len(self.paths)])
            _what = ('Scattered field |E_bead - E_ref| (phase-free; frame slider has no effect)'
                     if st['scat'] else
                     f'Difference {comp}: this file - {_ref}' if st['diff']
                     else f'{self.scenario} - {comp}')
            _zm = ' (zoomed)' if (st['lim'] != full and st['zreg'] is None) else ''
            _rg = ('zoom, fine grid' if st['zreg'] is not None else
                   'region around the bead' if st['crop'] else 'whole volume')
            _when = ('CW amplitude over all phase frames' if st['scat'] else
                     f'Frame {st["fi"] + 1}/{nf}  t={_t*1e12:.3f} ps  '
                     f'({"steady state" if cw_view else "transient"})')
            tgt.set_title(f'{_what}\n{_when}  '
                          f'|field| >= {st["db"]:.0f} dB = {lvl:.3g}\n{what_pts}, '
                          f'grid {s*dx*1e3:.0f} nm, {_rg}{_zm}',
                          color='#fff', fontsize=9)
            _legend.set_text(('red/yellow = positive, blue/cyan = negative; ' if signed else '')
                             + 'bright = strong, dark = close to the threshold.  '
                             + ('Mouse wheel = zoom, toolbar zoom/pan work too'
                                if m2 else 'Mouse wheel = zoom, right drag = zoom, middle drag = pan'))
            fig.canvas.draw_idle()

        _legend = fig.text(0.5, 0.115, '', color='#aaaaaa', fontsize=8, ha='center')
        _busy3d = fig.text(0.995, 0.995, '', ha='right', va='top', fontsize=10, color='#111111',
                           zorder=100, visible=False,
                           bbox=dict(facecolor='#FFD75E', edgecolor='none', pad=4, alpha=0.95))

        def _set_lim(lim):
            """Clamp the zoom box to the data region and redraw."""
            rg = region(st['crop'])
            f = _full_lim(rg)
            out = []
            for n_ in range(3):
                lo, hi = lim[2*n_], lim[2*n_ + 1]
                flo, fhi = f[2*n_], f[2*n_ + 1]
                w = min(hi - lo, fhi - flo)
                w = max(w, 4*dx)
                lo = min(max(lo, flo), fhi - w)
                out += [lo, lo + w]
            st['lim'] = tuple(out)
            _render()

        def on_scroll(ev):
            """Mouse wheel: zoom about the mouse position (2D) or the image centre (3D).
            In an axis view the viewing axis keeps its full depth."""
            if ev.inaxes not in (ax, ax2) or st['lim'] is None:
                return
            k = 0.8 if ev.button == 'up' else 1.25
            lim = list(st['lim'])
            if _mode2d():
                d = DEPTH[st['view_set']]
                h, v = [n_ for n_ in range(3) if n_ != d]
                for n_, c in ((h, ev.xdata), (v, ev.ydata)):
                    lo, hi = lim[2*n_], lim[2*n_ + 1]
                    c = 0.5*(lo + hi) if c is None else c
                    lim[2*n_], lim[2*n_ + 1] = c - (c - lo)*k, c + (hi - c)*k
            else:
                for n_ in range(3):
                    c = 0.5*(lim[2*n_] + lim[2*n_ + 1])
                    hh = 0.5*(lim[2*n_ + 1] - lim[2*n_])*k
                    lim[2*n_], lim[2*n_ + 1] = c - hh, c + hh
            _set_lim(lim)

        def on_release(ev):
            """After a mouse/toolbar zoom or pan, take over the new limits."""
            if ev.inaxes is ax2 and _mode2d():
                d = DEPTH[st['view_set']]
                h, v = [n_ for n_ in range(3) if n_ != d]
                lim = list(st['lim'])
                lim[2*h], lim[2*h + 1] = ax2.get_xlim()
                lim[2*v], lim[2*v + 1] = ax2.get_ylim()
                if not np.allclose(lim, st['lim'], atol=1e-6):
                    _set_lim(lim)
                return
            if ev.inaxes is not ax:
                return
            cur = tuple(ax.get_xlim()) + tuple(ax.get_ylim()) + tuple(ax.get_zlim())
            if st['lim'] is not None and not np.allclose(cur, st['lim'], atol=1e-6):
                _set_lim(cur)
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        fig.canvas.mpl_connect('button_release_event', on_release)

        a_db = fig.add_axes([0.18, 0.075, 0.46, 0.025], facecolor='#1a1a1a')
        s_db = _Slider(a_db, 'Threshold [dB]', -60.0, 0.0, valinit=st['db'], valstep=1.0,
                       color='#39FF14', track_color='#444444')
        a_fr = fig.add_axes([0.18, 0.035, 0.46, 0.025], facecolor='#1a1a1a')
        s_fr = _Slider(a_fr, 'Frame (time)', 0, max(nf - 1, 1), valinit=st['fi'], valstep=1,
                       color='#FF8800', track_color='#444444')
        for sl_ in (s_db, s_fr):
            sl_.label.set_color('#dddddd')
            sl_.valtext.set_color('#dddddd')

        def on_db(v):
            st['db'] = float(v)
            _render()

        def on_fr(v):
            st['fi'] = int(min(max(int(v), 0), nf - 1))
            _render()
        s_db.on_changed(on_db)
        s_fr.on_changed(on_fr)
        a_df = fig.add_axes([0.72, 0.07, 0.25, 0.035], facecolor='#1a1a1a')
        b_df = _Button(a_df, 'Difference A-B: off', color='#1a1a1a', hovercolor='#333333')
        a_cr = fig.add_axes([0.72, 0.025, 0.25, 0.035], facecolor='#1a1a1a')
        b_cr = _Button(a_cr, 'Region: whole volume', color='#1a1a1a', hovercolor='#333333')
        btns = [b_df, b_cr]
        # feste Blickrichtungen (orthographisch fuer die Achsen-Ansichten)
        _vlab = {'Iso': 'Iso (3D)', 'x-y': 'x-y (top view, along z)',
                 'x-z': 'x-z (along y)', 'y-z': 'y-z (along x)'}
        for n, key in enumerate(VIEWS):
            a_v = fig.add_axes([0.01, 0.90 - n*0.045, 0.20, 0.038], facecolor='#1a1a1a')
            b_v = _Button(a_v, _vlab[key], color='#1a1a1a', hovercolor='#333333')

            def _on_v(_e, key=key):
                st['view'] = key
                _render()
            b_v.on_clicked(_on_v)
            btns.append(b_v)
        a_zr = fig.add_axes([0.01, 0.90 - 4*0.045 - 0.02, 0.20, 0.038], facecolor='#1a1a1a')
        b_zr = _Button(a_zr, 'Reset zoom', color='#1a1a1a', hovercolor='#333333')
        a_zf = fig.add_axes([0.01, 0.90 - 5*0.045 - 0.02, 0.20, 0.038], facecolor='#1a1a1a')
        b_zf = _Button(a_zf, 'Zoom: fine grid', color='#1a1a1a', hovercolor='#333333')

        def on_zr(_e):
            st['zreg'] = None
            st['lim'] = None
            b_zf.label.set_text('Zoom: fine grid')
            _render()

        def on_zf(_e):
            """Zoom-Bereich in voller Aufloesung neu einlesen (Diff: Referenz neu streamen)."""
            if st['zreg'] is not None:
                st['zreg'] = None                   # zurueck zum groben Raster, Zoom bleibt
                b_zf.label.set_text('Zoom: fine grid')
                _render()
                return
            xa, xb, ya, yb, za, zb = st['lim'] or _full_lim(region(st['crop']))

            def _i(u, o, n):
                return int(np.clip(np.floor((u - o)/dx), 0, n))
            reg = (_i(xa, self.x0_um, Nx), min(Nx, _i(xb, self.x0_um, Nx) + 1),
                   _i(ya, self.y0_um, Ny), min(Ny, _i(yb, self.y0_um, Ny) + 1),
                   _i(za, self.z0_um, Nz), min(Nz, _i(zb, self.z0_um, Nz) + 1))
            st['zreg'] = reg
            b_zf.label.set_text('Zoom fine grid: on (loading ...)')
            fig.canvas.draw()
            _render()
            b_zf.label.set_text('Zoom fine grid: on')
            fig.canvas.draw_idle()
        b_zr.on_clicked(on_zr)
        b_zf.on_clicked(on_zf)
        btns += [b_zr, b_zf]
        a_sc = fig.add_axes([0.01, 0.90 - 6*0.045 - 0.04, 0.20, 0.038], facecolor='#1a1a1a')
        b_sc = _Button(a_sc, 'Scattered |E_s|: off', color='#1a1a1a', hovercolor='#333333')

        def on_sc(_e):
            if len(self.paths) < 2:
                b_sc.label.set_text('Scattered: load a 2nd file')
                fig.canvas.draw_idle()
                return
            st['scat'] = not st['scat']
            _render()
            b_sc.label.set_text(f'Scattered |E_s|: {"on" if st["scat"] else "off"}')
            fig.canvas.draw_idle()
        b_sc.on_clicked(on_sc)
        btns.append(b_sc)
        for b in btns:
            b.label.set_color('#dddddd')
            b.label.set_fontsize(8)

        def on_df(_e):
            if len(self.paths) < 2:
                b_df.label.set_text('Difference: load a 2nd file')
                fig.canvas.draw_idle()
                return
            st['diff'] = not st['diff']
            if st['diff']:
                b_df.label.set_text('Difference A-B: loading reference ...')
                fig.canvas.draw()                   # Hinweis sofort zeigen (Streamen dauert)
            _render()
            b_df.label.set_text(f'Difference A-B: {"on" if st["diff"] else "off"}')
            fig.canvas.draw_idle()

        def on_cr(_e):
            st['crop'] = not st['crop']
            st['zreg'] = None                       # neuer Bereich -> Zoom zuruecksetzen
            st['lim'] = None
            b_zf.label.set_text('Zoom: fine grid')
            b_cr.label.set_text('Region: around the bead' if st['crop'] else 'Region: whole volume')
            _render()
        b_df.on_clicked(on_df)
        b_cr.on_clicked(on_cr)
        st['view'] = 'Iso'
        self._v3d_widgets = [s_db, s_fr] + btns     # Referenzen halten (sonst GC -> Regler tot)
        self._v3d_slider = s_db
        self._v3d_fig = fig
        self._v3d_state = st
        self._v3d_render = _render
        _render()
        print('[3D view] opened - threshold [dB], frame, difference, region, views.')
        try:
            fig.show()
        except Exception:
            plt.show()

    def _style_axes(self):
        """Konsistenter Dark-Mode Look fuer alle Plots."""
        for sp in self.ax.spines.values():
            sp.set_color('#444444')
        self.ax.tick_params(colors='#cccccc')

    def _busy(self, msg='Loading ...'):
        """Small 'busy' badge (top right) + wait cursor, drawn immediately, so slow
        switches (file, view, plane, component, heavy modes) show that work is going on.
        redraw() hides it again."""
        try:
            t = getattr(self, '_busy_txt', None)
            if t is None or t.figure is not self.fig:
                t = self._busy_txt = self.fig.text(
                    0.995, 0.995, '', ha='right', va='top', fontsize=10, color='#111111',
                    zorder=100, bbox=dict(facecolor='#FFD75E', edgecolor='none', pad=4, alpha=0.95))
            t.set_text(msg)
            t.set_visible(True)
            try:
                from matplotlib.backend_tools import Cursors
                self.fig.canvas.set_cursor(Cursors.WAIT)
            except Exception:
                pass
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def _busy_done(self):
        t = getattr(self, '_busy_txt', None)
        if t is not None:
            t.set_visible(False)
        try:
            from matplotlib.backend_tools import Cursors
            self.fig.canvas.set_cursor(Cursors.POINTER)
        except Exception:
            pass

    def _safe_redraw(self):
        """Verhindert dass Slider-Events sich gegenseitig ueberholen.
        Wenn ein Redraw laeuft, wird das naechste Event uebersprungen
        (nicht aufgestaut) - der finale Slider-Wert wird in jedem Fall
        gezeichnet weil on_frame/on_x den self.frame_idx schon gesetzt haben."""
        if getattr(self, '_redrawing', False):
            return
        self._redrawing = True
        try:
            self.redraw()
            # Gib Tk-EventLoop kurz Zeit, anstehende Events zu verarbeiten
            try:
                self.fig.canvas.flush_events()
            except Exception:
                pass
        finally:
            self._redrawing = False

    def _schedule_redraw(self):
        """Resettet den Debounce-Timer - Redraw erst nach 150 ms Ruhe."""
        try:
            self._redraw_timer.stop()
            self._redraw_timer.start()
        except Exception:
            # Falls Timer noch nicht initialisiert: direkt rendern
            self._safe_redraw()

    def on_mode(self, label):
        # Stop pending Slider-Redraws bevor wir das Layout wechseln
        try:
            self._redraw_timer.stop()
        except Exception:
            pass
        self.mode = label
        if label in ('Compare', 'Diff (A-B)', 'Scattered', 'Spectrum', 'Modes', 'Resonance'):
            self._busy(f'Computing {label} ...')
        # Mode-Wechsel: sofort rendern (kein Scrubbing)
        self._safe_redraw()

    def on_frame(self, val):
        new_idx = int(val)
        if new_idx == self.frame_idx:
            return
        self.frame_idx = new_idx
        self._schedule_redraw()

    def _x_slider_range(self):
        """Regelbereich + Schrittweite des x-Cursors aus dem AKTUELL berechneten
        x-Bereich der Daten (meta), mit 2%% Rand. Schritt adaptiv (fein bei
        kleinen Domaenen, groeber bei der grossen Linse)."""
        x_min = float(np.min(self.meta[:, 0]))
        x_max = float(np.max(self.meta[:, 1]))
        span = max(x_max - x_min, 1e-6)
        pad = 0.02*span
        step = max(0.01, round(span/2000.0, 2))
        return x_min - pad, x_max + pad, step

    def _update_x_slider_range(self):
        """Passt den x-Slider an den aktuellen x-Bereich an (z.B. nach Ebenen-/
        Datei-Wechsel), damit der Cursor nie ausserhalb der Daten liegt."""
        if not hasattr(self, 's_x'):
            return
        lo, hi, step = self._x_slider_range()
        self.s_x.valmin = lo
        self.s_x.valmax = hi
        self.s_x.valstep = step
        self.s_x.ax.set_xlim(lo, hi)
        self.x_um = max(lo, min(hi, self.x_um))
        try:
            self.s_x.set_val(self.x_um)
        except Exception:
            pass

    def _add_slider_input(self, slider, fmt):
        """Put a small TextBox right of a slider (in place of its value text): typing a
        number + Enter sets the slider (clamped to its range); moving the slider
        updates the box. Comma as decimal separator is accepted."""
        pos = slider.ax.get_position()
        w_box, gap = 0.045, 0.006
        slider.ax.set_position([pos.x0, pos.y0, pos.width - w_box - gap, pos.height])
        h = min(pos.height, 0.032)
        ax_in = self.fig.add_axes([pos.x0 + pos.width - w_box, pos.y0 + (pos.height - h)/2,
                                   w_box, h], facecolor='#1a1a1a')
        tb = FastTextBox(ax_in, '', initial=fmt.format(slider.val), color='#1a1a1a',
                     hovercolor='#333333')
        try:                                     # mpl 3.11 resize-event bug (see tb_power)
            self.fig.canvas.mpl_disconnect(tb._cids.pop())
            self.fig.canvas.mpl_connect('resize_event', lambda _e, t=tb: t.stop_typing())
        except Exception:
            pass
        tb.text_disp.set_color('#7fd8ff')
        tb.text_disp.set_fontsize(8)
        slider.valtext.set_visible(False)
        state = {'sync': False}

        def _show(v):
            if tb.capturekeystrokes:
                return
            state['sync'] = True
            try:
                tb.set_val(fmt.format(v))
                tb.cursor.set_visible(False)
            finally:
                state['sync'] = False

        def _submit(text):
            if state['sync']:
                return
            try:
                v = float(str(text).replace(',', '.'))
            except ValueError:
                _show(slider.val)
                return
            v = max(slider.valmin, min(slider.valmax, v))
            tb.stop_typing()                     # Enter: done typing, then apply
            slider.set_val(v)
            _show(v)
        tb.on_submit(_submit)
        tb.on_cancel = lambda: _show(slider.val)   # left without Enter -> show old value
        slider.on_changed(_show)
        self._slider_inputs[id(slider)] = tb
        return tb

    def _update_bead_hint(self):
        """Bead extent in x next to the x input box (where the bead starts/ends)."""
        t = getattr(self, '_bead_hint', None)
        if t is None:
            return
        if getattr(self, 'is_bead', False) and getattr(self, 'bead_d', 0) > 0:
            r = self.bead_d/2.0
            t.set_text(f'bead: x {self.bead_x - r:.2f}..{self.bead_x + r:.2f} um')
        else:
            t.set_text('')

    def on_x(self, val):
        new_x = float(val)
        if abs(new_x - self.x_um) < 0.005:   # 5 nm Mindest-Bewegung
            return
        self.x_um = new_x
        if self.mode in ('Line Cut', 'Evanescent', 'Field 2D', 'Field log'):
            self._schedule_redraw()

    def on_fit_top(self, val):
        self.fit_top = float(val)
        if self.mode == 'Evanescent':
            self._schedule_redraw()

    def on_fit_depth(self, val):
        self.fit_depth = float(val)
        if self.mode == 'Evanescent':
            self._schedule_redraw()

    def switch_path(self, delta):
        self.cur_path = (self.cur_path + delta) % len(self.paths)
        self._busy(f'Loading {os.path.basename(self.paths[self.cur_path])} ...')
        self.load()
        self.frame_idx = min(self.frame_idx, len(self.ezs)-1)
        self.s_frame.valmax = len(self.ezs)-1
        self.s_frame.ax.set_xlim(0, len(self.ezs)-1)
        self.s_frame.set_val(self.frame_idx)
        self._update_x_slider_range()   # x-Cursor an neuen Daten-x-Bereich anpassen
        self.redraw()

    def redraw(self):
        try:
            self._redraw_inner()
        finally:
            self._busy_done()
            self._update_bead_hint()
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                pass

    def _redraw_inner(self):
        # Sichere Achsen-Bereinigung: clear + alte Colorbars/Legenden entfernen
        _cb = getattr(self, '_scatter_cb', None)
        if _cb is not None:
            try:
                _cb.ax.remove()
            except Exception:
                pass
            self._scatter_cb = None
        try:
            self.ax.clear()
            self.ax.set_facecolor('#0a0a0a')
            # Reset Axes-Scale (log-Skala von vorherigem Mode bleibt sonst kleben)
            self.ax.set_xscale('linear')
            self.ax.set_yscale('linear')
            # WICHTIG: aspect='equal' von Field 2D zuruecksetzen.
            # Sonst staucht matplotlib Line-Cut / Evanescent / Time-Trend Plots
            # vertikal extrem zusammen.
            self.ax.set_aspect('auto')
        except Exception:
            pass
        # 1D-Schnitt-Modi setzen die xy-Konvention voraus (x=Propagation,
        # y=Schichtstapel). In 3D-Ebenen xz/yz sind sie nicht definiert -> Hinweis.
        # Modi, die den y-Schichtstapel auf der Vertikalen brauchen -> nur xy.
        _1d = ('Line Cut', 'Evanescent', 'Resonance', 'Layer Power', 'Sensor')
        _needs_x = ('Modes',)   # brauchen x auf der Horizontalen -> xy oder xz
        _bad = (getattr(self, 'is3d', False) and getattr(self, 'plane', 'xy') != 'xy'
                and self.mode in _1d) or \
               (getattr(self, 'is3d', False) and getattr(self, 'plane', 'xy') == 'yz'
                and self.mode in _needs_x)
        if _bad:
            _ok_planes = "'xy'" if self.mode in _1d else "'xy' or 'xz'"
            self.ax.text(0.5, 0.5,
                         f"'{self.mode}' is only meaningful in plane {_ok_planes}\n"
                         f"(propagation x horizontal / layer stack y vertical).\n\n"
                         f"Current plane: {self.plane}.  ->  switch the plane,\n"
                         f"or use an image mode (Field 2D/log, Time Avg).",
                         color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=11,
                         bbox=dict(facecolor='#1a1a1a', edgecolor='#FFAA55', alpha=0.85))
            self._style_axes()
            self._set_header(f'{self.scenario}  |  Mode: {self.mode} '
                             f'(plane {self.plane})')
            self.fig.canvas.draw_idle()
            return
        try:
            if self.mode == 'Field 2D':
                self._draw_field(log=False)
            elif self.mode == 'Field log':
                self._draw_field(log=True)
            elif self.mode == 'Time Avg':
                self._draw_intavg()
            elif self.mode == 'Line Cut':
                self._draw_line_cut()
            elif self.mode == 'Evanescent':
                self._draw_evanescent()
            elif self.mode == 'Resonance':
                self._draw_resonance()
            elif self.mode == 'Layer Power':
                self._draw_layer_power()
            elif self.mode == 'Modes':
                self._draw_modes()
            elif self.mode == 'Sensor':
                self._draw_sensor()
            elif self.mode == 'Time Trend':
                self._draw_time_trend()
            elif self.mode == 'Compare':
                self._draw_compare()
            elif self.mode == 'Diff (A-B)':
                self._draw_diff()
            elif self.mode == 'Scattered':
                self._draw_scatter()
            elif self.mode == 'Spectrum':
                self._draw_spectrum()
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                self.ax.text(0.5, 0.5, f'Error: {e}', color='#FF5555',
                             ha='center', va='center', transform=self.ax.transAxes)
            except Exception:
                pass
        _msg = getattr(self, '_comp_msg', None)
        if _msg:                                  # einmaliger Hinweis (Feld nicht vorhanden)
            self.ax.text(0.5, 0.02, _msg, color='#FFAA55', ha='center', va='bottom', fontsize=9,
                         transform=self.ax.transAxes, zorder=20,
                         bbox=dict(facecolor='#1a1a1a', edgecolor='#FFAA55', alpha=0.85))
            self._comp_msg = None
        info = f'[{self.cur_path+1}/{len(self.paths)}] ' if len(self.paths) > 1 else ''
        self._set_header(f'{info}{self.scenario}  |  Mode: {self.mode}')
        self.fig.canvas.draw_idle()

    def _set_header(self, text):
        """Titel linksbuendig RECHTS neben der oberen Button-Leiste platzieren
        (Buttons enden bei x~0.37), damit er nicht mit Bedienelementen ueberlappt."""
        _pt = getattr(self, 'pol_tag', None)
        if _pt:
            text = f'{text}   [{_pt}]'
        if POWER.get('mw'):
            if getattr(self, '_p_sim_w', None):
                _d = f', depth {POWER["depth_um"]:g} um' if str(self.p_in_unit) == 'W/m' else ''
                text = f'{text}   [P = {POWER["mw"]:g} mW{_d} -> E in V/m]'
            else:
                text = f'{text}   [P = {POWER["mw"]:g} mW: file has no power info -> unscaled]'
        self.fig.suptitle(text, color='#ffffff', fontsize=9,
                          x=0.40, y=0.978, ha='left')

    # ---- Hilfen fuer Moden-/Sensor-Analyse (2D + 3D-Ebene) ----
    def _iavg_ok(self):
        """Vom Solver akkumuliertes Zeitmittel <|E|^2> fuer den aktuellen 2D-Schnitt,
        falls passend zur Frame-Geometrie (sonst None). Enthaelt ALLE E-Komponenten
        (TM: Ex+Ey, 3D: Ex+Ey+Ez) -> physikalisch richtige Intensitaet, waehrend die
        Frames nur die Hauptkomponente zeigen."""
        iv = getattr(self, 'iavg2d', None)
        if not getattr(self, 'iavg_full', False):
            return None                      # alte Datei: Iavg nur Hauptkomponente
        if iv is None or np.shape(iv) != np.shape(self.ezs[0]):
            return None
        if not np.allclose(self.meta[:, :4], self.meta[0, :4]):
            return None                      # Stitch-Fenster mit wechselndem x
        return np.asarray(iv, dtype=np.float64)

    # ---- CAD: Regionskarte statt Schichtstapel ----
    CAD_COLORS = ('#FFD75E', '#56C4FF', '#D88AFF', '#7CFC00', '#FF8C69', '#88AAFF',
                  '#FFB6C1', '#40E0D0', '#F4A460', '#B0C4DE')

    def _cad_name(self, idx):
        return 'Ambient' if idx <= 0 else str(self.mat_names[idx - 1])

    def _cad_n(self, idx):
        if idx <= 0:
            return 1.0
        try:
            return float(self.mat_indices[idx - 1])
        except Exception:
            return float('nan')

    def _eps_slice(self, shape):
        """Relative Permittivitaet n^2 auf dem Anzeigeraster (Nx, Nv) des aktuellen
        Schnitts - fuer die TM-Gewichtung von Sx. CAD: aus der Regionskarte; Schicht-
        stapel: aus den Schichtdicken (nur 2D bzw. 3D-xy-Ebene; Bead lokal ignoriert)."""
        try:
            if getattr(self, 'is_cad', False):
                r = self._reg_slice()
                if r is None or r.shape != tuple(shape):
                    return None
                eps = np.ones(shape)
                for g in np.unique(r):
                    eps[r == g] = self._cad_n(int(g))**2
                return eps
            if getattr(self, 'is3d', False) and getattr(self, 'plane', 'xy') != 'xy':
                return None
            y0, y1 = self.meta[0, 2], self.meta[0, 3]
            Nv = shape[1]
            yc = y0 + (np.arange(Nv) + 0.5)*(y1 - y0)/Nv
            t_wg = self._t_wg_eff()
            nv = np.full(Nv, float(self.sub_n))                      # Substrat
            for _nm, _yt, yb, n_ in reversed(self._stack_bounds(0.0)):
                nv[yc >= yb] = complex(n_).real
            nv[yc >= 0.0] = float(self.core_n)
            nv[yc > t_wg] = 1.0                                       # Luft
            return np.broadcast_to(nv**2, shape)
        except Exception:
            return None

    def _cad_color(self, idx):
        return '#777777' if idx <= 0 else self.CAD_COLORS[(idx - 1) % len(self.CAD_COLORS)]

    def _reg_slice(self):
        """Regionskarte passend zur Form der angezeigten Frames (Zuschnitt/Rand bei
        +-1 Zelle Yee-Versatz), oder None."""
        r = getattr(self, 'reg2d', None)
        if r is None:
            return None
        nx, nv = self.ezs.shape[1], self.ezs.shape[2]
        out = np.zeros((nx, nv), dtype=np.int8)
        a, b = min(nx, r.shape[0]), min(nv, r.shape[1])
        out[:a, :b] = r[:a, :b]
        if b < nv:
            out[:, b:] = out[:, b-1:b]
        if a < nx:
            out[a:] = out[a-1:a]
        return out

    def _cad_segments(self, x_um, i=None):
        """Zusammenhaengende Abschnitte entlang der Vertikalen bei x:
        [(y_lo, y_hi, region), ...] von unten nach oben."""
        r = self._reg_slice()
        if r is None:
            return []
        i = self.frame_idx if i is None else i
        x0, x1, v0, v1 = self.meta[min(i, len(self.meta)-1), :4]
        nx, nv = r.shape
        ix = int(np.clip(round((x_um - x0)/max(x1 - x0, 1e-12)*nx), 0, nx - 1))
        col = r[ix]
        dv = (v1 - v0)/nv
        segs, j0 = [], 0
        for j in range(1, nv + 1):
            if j == nv or col[j] != col[j0]:
                segs.append((v0 + j0*dv, v0 + j*dv, int(col[j0])))
                j0 = j
        return segs

    def _cad_below(self, x_um):
        """Abschnitt direkt unter dem Kern bei x (Region, y_lo, y_hi) oder None."""
        segs = self._cad_segments(x_um)
        for k, (lo, hi, reg) in enumerate(segs):
            if reg == self.cad_core and k > 0:
                b = segs[k - 1]
                return b[2], b[0], b[1]
        return None

    def _wg_y_bounds(self, x_um):
        """(y_unterkante, y_oberkante) des WG-Kerns in um bei Laengsposition x_um:
        planar/Slab-Koordinaten -> PS_BOT..wg_top_um; alte geflattete Linse ->
        +-T_LENS/2; alte gekruemmte Linse -> Sag-Kurve; CAD -> Kern-Region in der
        Spalte bei x (groesster zusammenhaengender Abschnitt)."""
        if getattr(self, 'is_cad', False):
            core = [s for s in self._cad_segments(x_um) if s[2] == self.cad_core]
            if core:
                lo, hi, _ = max(core, key=lambda s: s[1] - s[0])
                return lo, hi
            return float(self.meta[0, 2]), float(self.meta[0, 3])
        if self.planar:
            return PS_BOT, self.wg_top_um
        half = T_LENS*1e6/2
        if self.flat_mode:
            return -half, half
        x_m = max(-R_BEND + 1e-9, min(R_BEND - 1e-9, x_um*1e-6))
        y_mid = -(R_BEND - np.sqrt(R_BEND**2 - x_m**2))*1e6
        return y_mid - half, y_mid + half

    def _t_wg_eff(self):
        """WG-Kerndicke in um (3D: aus Datei; 2D-Planar: PS_TOP-PS_BOT)."""
        return getattr(self, 't_wg_um', 0.0) or getattr(self, 'wg_top_um', PS_TOP - PS_BOT)

    def _delta_min(self):
        """Theoretische MINIMALE evaneszente Eindringtiefe delta_min (um) =
        lam/(2 pi sqrt(n_core^2 - n_clad^2)) (grazing incidence)."""
        try:
            n_hi = float(self.mat_indices[0])
            # Medium direkt unter dem Kern: stack[0] (bzw. Substrat); alte Dateien Aqueous
            n_lo = self._clad_n(legacy=self.mat_indices[2] if len(self.mat_indices) > 2 else None)
            if getattr(self, 'is_cad', False):           # Kern gegen das Medium darunter
                n_hi = self._cad_n(self.cad_core)
                _b = self._cad_below(getattr(self, 'x_um', 0.0)) if hasattr(self, 'ezs') else None
                n_lo = self._cad_n(_b[0]) if _b else 1.0
            lam_um = float(self.lam_nm)/1000.0
            return lam_um/(2*np.pi*max((n_hi**2 - n_lo**2)**0.5, 1e-6))
        except Exception:
            return 0.2

    def _set_evan_defaults(self):
        """Fit-Fenster fuer den Evanescent-Modus an die tatsaechliche Physik
        koppeln: Tiefe ~3*delta_min (statt fixer 1.5 um, die bei delta~0.2 um fast
        nur Rauschen fittet), Start ~1-2 Zellen UNTER der Grenzflaeche (weg vom
        Interface-Sprung, wichtig fuer TM/Ey)."""
        dmin = self._delta_min()
        self._delta_min_um = dmin
        dx = 0.02
        try:
            if not getattr(self, 'is3d', False) and self.ezs.ndim == 3:
                y0, y1 = float(self.meta[0, 2]), float(self.meta[0, 3])
                dx = abs(y1 - y0)/max(self.ezs.shape[2], 1)
        except Exception:
            pass
        self.fit_depth = round(min(5.0, max(0.3, 3.0*dmin)), 2)
        self.fit_top = -round(min(4.9, max(0.05, 1.5*dx)), 3)
        for attr, val in (('s_fit_depth', self.fit_depth), ('s_fit_top', self.fit_top)):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.set_val(val)
                except Exception:
                    pass

    def _cw_phase_stack(self):
        """Eingeschwungener Phasen-Satz (nph, Nx, Nv) EINES x-Fensters + dessen
        meta-Zeile, oder None. Nur wenn ein CW-Satz existiert und alle Frames
        dasselbe x-Fenster teilen (voller Lauf) - dann ist die zeitgemittelte
        Amplitude/Intensitaet knotenfrei und robust. Bei Stitch (unterschiedliche
        Fenster) -> None (Aufrufer faellt auf Momentaufnahme zurueck)."""
        v = getattr(self, '_views', {}).get('Steady state') if hasattr(self, '_views') else None
        if not v:
            return None
        if v[0] == '3d':
            # 3D: CW-Volumen in der aktuellen xy-Ebene schneiden (nur dort sind
            # Laenge x / Schichtstapel y die Anzeigeachsen); Extents = self.meta.
            if getattr(self, 'plane', 'xy') != 'xy' or np.ndim(v[1]) != 4:
                return None
            k = int(min(self.slice_idx, np.shape(v[1])[3] - 1))
            return np.asarray(v[1][:, :, :, k], dtype=np.float32), self.meta[0]
        arr = np.asarray(v[1])
        meta = np.asarray(v[2])
        if arr.ndim != 3 or arr.shape[0] < 2 or meta.shape[0] < 1:
            return None
        if not np.allclose(meta[:, :4], meta[0, :4]):
            return None
        return arr, meta[0]

    def _evan_envelope(self, comp, i, ix, Nx, Ny):
        """Knotenfreie CW-Einhuellende |Ê_comp|(y) an der x-Spalte ix des Bildes i, oder
        (None, ''). Quellen:
          Hauptkomponente (Ez TE / Ey TM): eingeschwungener Phasensatz
          Ex (TM-2D):  gespeicherte komplexe Amplitude Ex_cw_amp
          |E|:         sqrt(2<|E|^2>) aus dem Solver-Zeitmittel (alle Komponenten),
                       sonst sqrt(|Ê_y|^2 + |Ê_x|^2) aus beiden Amplituden."""
        def prim_env():
            ps = self._cw_phase_stack()
            if ps is None:
                return None
            arr, m0 = ps
            if arr.shape[1] != Nx or arr.shape[2] != Ny or not np.allclose(m0[:4], self.meta[i, :4]):
                return None
            e = np.sqrt(2.0*np.mean(arr[:, ix, :].astype(np.float64)**2, axis=0))
            return e if float(e.max()) > 0 else None

        def ex_env():
            a = getattr(self, 'ex_amp2d', None)
            if a is None or getattr(self, 'is3d', False) or a.shape != (Nx, Ny):
                return None
            if not np.allclose(self.meta[:, :4], self.meta[0, :4]):
                return None                      # Fenster-Frames mit wechselndem x
            return np.abs(a[ix]).astype(np.float64)

        if comp == '|E|':
            iv = self._iavg_ok()
            if iv is not None and iv.shape == (Nx, Ny):
                return (np.sqrt(2.0*np.maximum(iv[ix], 0.0)),
                        'sqrt(2<|E|^2>), time average of all components')
            # Iavg liegt auf dem Yee-Gitter der Hauptkomponente, die Momentbilder im
            # Vektor-Modus zellzentriert (eine Zelle weniger je Achse) -> Spalte auf die
            # gleichen physikalischen Koordinaten interpolieren statt |E| zu verweigern.
            ivr = getattr(self, 'iavg2d', None)
            if (ivr is not None and getattr(self, 'iavg_full', False) and np.ndim(ivr) == 2
                    and abs(ivr.shape[0] - Nx) <= 1 and abs(ivr.shape[1] - Ny) <= 1
                    and np.allclose(self.meta[:, :4], self.meta[0, :4])):
                jx = int(round(ix*(ivr.shape[0] - 1)/max(Nx - 1, 1)))
                col = np.asarray(ivr[jx], dtype=np.float64)
                if col.shape[0] != Ny:
                    col = np.interp(np.linspace(0, 1, Ny), np.linspace(0, 1, col.shape[0]), col)
                return (np.sqrt(2.0*np.maximum(col, 0.0)),
                        'sqrt(2<|E|^2>), time average of all components')
            pe, xe = prim_env(), ex_env()
            if pe is not None and xe is not None:
                return np.sqrt(pe**2 + xe**2), f'sqrt(|{self.prim}|^2+|Ex|^2), CW amplitudes'
            return None, ''
        if comp == 'Ex' and not getattr(self, 'is3d', False):
            e = ex_env()
            return (e, 'CW amplitude |Ex| (tangential, continuous at interfaces)') if e is not None \
                else (None, '')
        if comp == self.prim:
            e = prim_env()
            if e is not None:
                return e, ('CW amplitude (node-free)' if self.prim != 'Ey' else
                           'CW amplitude |Ey| (normal, jumps by eps1/eps2 at interfaces)')
        return None, ''

    def _v_index(self, val, v0, v1, Nv):
        """Index auf der VERTIKALEN Anzeigeachse (y bzw. z) fuer Wert val."""
        if v1 == v0:
            return 0
        return int(round((val - v0)/(v1 - v0)*Nv))

    def _x_index(self, val, x0, x1, Nx):
        """Index auf der HORIZONTALEN Achse (Laenge x) fuer Wert val."""
        if x1 == x0:
            return 0
        return int(round((val - x0)/(x1 - x0)*Nx))

    def _ez_signed(self):
        """Vorzeichenbehafteter Ez-Schnitt (nf, Nx, Nv) fuer die aktuelle Ebene.
        Fuer die Moden-FFT noetig (|E| ist gleichgerichtet -> falsche k_x). In 2D
        ist self.ezs bereits das signierte Ez."""
        if not getattr(self, 'is3d', False):
            return self.ezs
        v = self.vol
        i = self.slice_idx
        if self.plane == 'xy':
            return np.ascontiguousarray(v[:, :, :, i]).astype(np.float32)
        elif self.plane == 'xz':
            return np.ascontiguousarray(v[:, :, i, :]).astype(np.float32)
        else:
            return np.ascontiguousarray(v[:, i, :, :]).astype(np.float32)

    def _draw_modes(self):
        """Modenspektrum: FFT des Feldes entlang x in einem Kernstreifen ->
        Leistung ueber n_eff=k_x/k0. Peaks im gefuehrten Band = gefuehrte Moden.
        Funktioniert in 2D und in den 3D-Ebenen mit x-Achse (xy, xz)."""
        fi = min(self.frame_idx, len(self.ezs)-1)
        # FFT braucht das VORZEICHENBEHAFTETE Feld. |E| ist gleichgerichtet und
        # wuerde die Ortsfrequenzen (Peak bei 2*beta statt beta) verfaelschen ->
        # bei |E| intern auf signiertes Ez zurueckgreifen.
        _src = self._ez_signed() if self._disp_comp() == '|E|' else self.ezs
        fi = min(fi, len(_src)-1)
        ez = _src[fi].astype(np.float64)              # (Nx, Nv)
        x0, x1, y0, y1 = self.meta[fi, :4]
        Nx, Nv = ez.shape
        if Nx < 8 or (x1 - x0) <= 0:
            self.ax.text(0.5, 0.5, 'too few x points for FFT', color='#FFAA55',
                         ha='center', va='center', transform=self.ax.transAxes)
            self._style_axes()
            self._set_header(f'{self.scenario}  |  Mode: Modes')
            return
        dxx = (x1 - x0)*1e-6/Nx
        if (not getattr(self, 'is3d', False)) or self.plane == 'xy':
            t_wg = self._t_wg_eff()
            v_lo = self._v_index(0.0, y0, y1, Nv)
            v_hi = self._v_index(t_wg, y0, y1, Nv)
            vlabel = f'core y=0..{t_wg:g}um'
        else:                                          # xz: vertikal = Tiefe z
            v_lo, v_hi = int(Nv*0.35), int(Nv*0.65)
            vlabel = 'depth center'
        v_lo = max(0, min(v_lo, Nv-1))
        v_hi = max(v_lo+1, min(v_hi, Nv))
        strip = ez[:, v_lo:v_hi]
        win = np.hanning(Nx)[:, None]
        F = np.fft.rfft(strip*win, axis=0)
        freq = np.fft.rfftfreq(Nx, d=dxx)
        k0 = 2*np.pi/(getattr(self, 'lam_nm', 850.0)*1e-9)
        neff = 2*np.pi*freq/k0
        power = (np.abs(F)**2).sum(axis=1)
        n_core = float(self.mat_indices[0])
        n_aq = self._clad_n(legacy=self.mat_indices[2] if len(self.mat_indices) > 2 else None)
        lo = max(1.0, n_aq) + 0.002
        hi = n_core
        guided = find_neff_peaks(neff, power, lo, hi)
        leaky = find_neff_peaks(neff, power, 1.0, lo)
        m = (neff >= 0.9) & (neff <= n_core + 0.08)
        pmax = float(power[m].max()) if power[m].size and power[m].max() > 0 else 1.0
        self.ax.plot(neff[m], power[m]/pmax, color='#4da6ff', lw=1.0)
        self.ax.axvspan(lo, hi, color='#2e8b57', alpha=0.12,
                        label=f'guided ({lo:.3f}-{hi:.3f})')
        for n, _ in guided:
            self.ax.axvline(n, color='#ff5b5b', ls='--', lw=0.8)
            self.ax.text(n, 1.02, f'{n:.3f}', color='#ff8a8a', fontsize=7,
                         rotation=90, va='bottom', ha='center')
        self.ax.set_xlim(0.9, n_core + 0.08)
        self.ax.set_ylim(0, 1.18)
        self.ax.set_xlabel('n_eff = k_x / k0', color='#cccccc', fontsize=9)
        self.ax.set_ylabel('spectral power (norm.)', color='#cccccc', fontsize=9)
        _pl = f' | plane {self.plane}' if getattr(self, 'is3d', False) else ''
        L_um = (x1 - x0)
        dneff = (self.lam_nm/1000.0)/L_um if L_um > 0 else 9.9   # = lam0/L
        self.ax.set_title(
            f'Mode spectrum ({vlabel}, frame {fi+1}, {self.lam_nm:g}nm) - '
            f'{len(guided)} guided modes' + (f', {len(leaky)} leaky' if leaky else '')
            + _pl, color='#ffffff', fontsize=10)
        # Aufloesungshinweis: n_eff-Rasterung = lam0/L. Bei kurzer Strecke zu grob.
        band = max(hi - lo, 1e-6)
        if dneff > band/5.0:
            self.ax.text(0.5, 0.90,
                         f'Coarse resolution: dn_eff={dneff:.3f} (grid ~ lam0/L). '
                         f'For clear modes, simulate a longer x range (>=50 um).',
                         color='#FFAA55', ha='center', va='top', fontsize=8,
                         transform=self.ax.transAxes,
                         bbox=dict(facecolor='#1a1a1a', edgecolor='#FFAA55', alpha=0.7))
        self._style_axes()
        leg = self.ax.legend(loc='upper right', fontsize=8, framealpha=0.6,
                             facecolor='#1a1a1a')
        for t in leg.get_texts():
            t.set_color('#dddddd')

    def _draw_sensor(self):
        """Sensor-Metrik: zeitgemittelte Intensitaet <E^2> in Kern / Tear-Film /
        Luft an einem Detektorfenster nahe dem Ausgang. K = I_tear/I_guided ist der
        evaneszente Ueberlapp (= Sensor-Signal). Zusaetzlich, falls die komplexe
        CW-Amplitude vorliegt, der gerichtete Poynting-Fluss Sx ~ Im(Ê dÊ*/dx) als
        ECHTE gefuehrte Leistung (-> Poynting-Transmission). Bevorzugt den
        eingeschwungenen Phasensatz (knotenfrei); sonst die letzten Transient-Frames."""
        x0, x1, y0, y1 = self.meta[0, :4]
        t_wg = self._t_wg_eff()
        # Tear-Band aus dem Schichtstapel (alte Dateien: Aqueous; neue: ganzer Stapel).
        tear_depth, tear_lab = self._tear_band()

        # Datenquelle: eingeschwungener Phasensatz (zeitgemittelt) bevorzugt.
        ps = self._cw_phase_stack()
        Ehat = None
        if ps is not None and np.allclose(ps[1][:4], self.meta[0, :4]):
            arr = ps[0].astype(np.float64)              # (nph, Nx, Nv), eine Periode
            intens = np.mean(arr**2, axis=0)            # <E^2> zeitgemittelt
            # Komplexe Amplitude per DFT ueber ALLE Phasen: Frame k = Re(Ê e^{iωt_k})
            # mit t_k = k/N*T -> Ê = (2/N) Σ E_k e^{-2πik/N}. Exakt fuer jedes N>=3
            # (frueher E_0 - i*E_{N/4}: nur korrekt fuer N durch 4 teilbar).
            nph = arr.shape[0]
            if nph >= 3:
                w = np.exp(-2j*np.pi*np.arange(nph)/nph)
                Ehat = (2.0/nph)*np.tensordot(w, arr, axes=(0, 0))
            src = f'CW amplitude ({arr.shape[0]} phases, time-averaged)'
        else:
            N = min(6, len(self.ezs))
            intens = np.mean(self.ezs[-N:].astype(np.float64)**2, axis=0)
            src = f'{N} transient frames (no CW set available)'
        # Intensitaet bevorzugt aus dem Solver-Zeitmittel <|E|^2> (alle Komponenten);
        # Ê bleibt fuer den Poynting-Fluss aus dem CW-Satz der Hauptkomponente.
        _iv = self._iavg_ok()
        if _iv is not None and _iv.shape == intens.shape:
            intens = _iv
            src += '; intensity = solver <|E|^2> (all components)'
        Nx, Nv = intens.shape

        def xr(frac):
            c = x0 + frac*(x1 - x0)
            w = 0.10*(x1 - x0)
            lo = self._x_index(c - w/2, x0, x1, Nx)
            hi = self._x_index(c + w/2, x0, x1, Nx)
            return max(0, lo), max(lo+1, min(hi, Nx))

        dl, dh = xr(0.8)
        il, ih = xr(0.2)
        iy_wg_lo = max(0, self._v_index(0.0, y0, y1, Nv))
        iy_wg_hi = min(Nv, self._v_index(t_wg, y0, y1, Nv))
        iy_air_hi = min(Nv, self._v_index(y1, y0, y1, Nv))
        iy_tear_lo = max(0, self._v_index(-tear_depth, y0, y1, Nv))

        def bandI(field, cl, ch, ylo, yhi):
            if yhi <= ylo or ch <= cl:
                return 0.0
            return float(np.mean(np.sum(field[cl:ch, ylo:yhi], axis=1)))

        _rs = self._reg_slice() if getattr(self, 'is_cad', False) else None
        if _rs is not None and _rs.shape == intens.shape and (_rs == self.cad_core).any():
            # CAD: Baender = Regionen (Kern / Film / Umgebung), Fenster entlang des Kerns
            m_core = _rs == self.cad_core
            cols = np.nonzero(m_core.any(axis=1))[0]
            ca, cb = int(cols.min()), int(cols.max()) + 1
            xs_ax = np.linspace(x0, x1, Nx)

            def cr(frac):
                c = ca + frac*(cb - ca); w = max(2.0, 0.10*(cb - ca))
                lo = int(round(c - w/2)); return max(0, lo), max(lo + 1, min(int(round(c + w/2)), Nx))
            dl, dh = cr(0.8); il, ih = cr(0.2)
            _xc = float(xs_ax[(dl + dh)//2])
            _below = self._cad_below(_xc)
            tear_reg = self.cad_tear or (_below[0] if _below else -1)
            m_tear = _rs == tear_reg
            m_air = _rs == 0

            def bandM(field, cl, ch, mask):
                if ch <= cl:
                    return 0.0
                return float(np.mean(np.sum(np.where(mask[cl:ch], field[cl:ch], 0.0), axis=1)))
            P_guid = bandM(intens, dl, dh, m_core)
            P_guid_in = bandM(intens, il, ih, m_core)
            P_tear = bandM(intens, dl, dh, m_tear)
            P_air = bandM(intens, dl, dh, m_air)
            band_core = band_in = m_core
            _cad_lbl = (f'core "{self._cad_name(self.cad_core)}"   film "{self._cad_name(tear_reg)}"'
                        f'{"" if self.cad_tear else " (automatic: directly below the core)"}   '
                        f'ambient')
        else:
            _cad_lbl = None
            P_guid = bandI(intens, dl, dh, iy_wg_lo, iy_wg_hi)
            P_guid_in = bandI(intens, il, ih, iy_wg_lo, iy_wg_hi)
            P_tear = bandI(intens, dl, dh, iy_tear_lo, iy_wg_lo)
            P_air = bandI(intens, dl, dh, iy_wg_hi, iy_air_hi)
        K = P_tear/P_guid if P_guid > 0 else 0.0
        atten = P_guid/P_guid_in if P_guid_in > 0 else 0.0

        # Gerichteter Poynting-Fluss (Konstante egal fuer das Verhaeltnis).
        # Der evaneszente Ausläufer klingt in y ab, transportiert seine Energie aber in x
        # mit -> Sx ueber den GANZEN Querschnitt (Kern + Film + Luft) integrieren.
        #   TE (Ez):  Sx = -1/2 Re(Ez Hy*), Hy = (1/iwmu) dEz/dx  ->  Sx ~ Im(Ê dÊ*/dx)  (exakt)
        #   TM (Ey):  Sx = 1/2 Re(Ey Hz*), Hz ~ (w eps/beta) Ey   ->  Sx ~ eps(y) Im(Ê dÊ*/dx)
        #             (Naeherung gefuehrter Mode; beta kuerzt sich in allen Verhaeltnissen)
        trans_S = trans_S_core = eta_film = eta_core = eta_air = float('nan')
        sx_note = ''
        if Ehat is not None:
            Sx = np.imag(Ehat*np.conj(np.gradient(Ehat, axis=0)))
            if getattr(self, 'pol', 's') == 'p':
                eps = self._eps_slice(Sx.shape)
                if eps is not None:
                    Sx = Sx*eps
                    sx_note = ' (TM: eps(y)-weighted)'
                else:
                    sx_note = ' (TM: no eps weighting, approximation)'
            if _cad_lbl is not None and Sx.shape == band_core.shape:
                S_det = bandM(Sx, dl, dh, band_core)
                S_in = bandM(Sx, il, ih, band_in)
                S_film = bandM(Sx, dl, dh, m_tear)
                S_air = bandM(Sx, dl, dh, m_air)
            else:
                S_det = bandI(Sx, dl, dh, iy_wg_lo, iy_wg_hi)
                S_in = bandI(Sx, il, ih, iy_wg_lo, iy_wg_hi)
                S_film = bandI(Sx, dl, dh, iy_tear_lo, iy_wg_lo)
                S_air = bandI(Sx, dl, dh, iy_wg_hi, iy_air_hi)
            S_det_all = bandI(Sx, dl, dh, 0, Nv)
            S_in_all = bandI(Sx, il, ih, 0, Nv)
            trans_S_core = S_det/S_in if abs(S_in) > 1e-30 else float('nan')
            trans_S = S_det_all/S_in_all if abs(S_in_all) > 1e-30 else float('nan')
            if abs(S_det_all) > 1e-30:
                eta_core, eta_film, eta_air = (S_det/S_det_all, S_film/S_det_all,
                                               S_air/S_det_all)

        xd = x0 + (dl + dh)/2*(x1 - x0)/Nx
        xin = x0 + (il + ih)/2*(x1 - x0)/Nx

        self.ax.axis('off')
        lines = [
            f'SENSOR METRIC   {self.scenario}',
            f'Source: {src}',
            '',
            f'Detector x = {xd:6.2f} um     Input x = {xin:6.2f} um   (10% of length each)',
            (_cad_lbl if _cad_lbl is not None else
             f'Core y = 0..{t_wg:g} um   Tear y = -{tear_depth:g}..0 um ({tear_lab})   '
             f'Air y = {t_wg:g}..{y1:.1f} um'),
            '',
            f'I_guided (core,  detector) :  {P_guid:.3e}',
            f'I_tear   (tear,  detector) :  {P_tear:.3e}',
            f'I_air    (air,   detector) :  {P_air:.3e}',
            '',
            f'K = I_tear / I_guided      :  {K:.4e}   (evanescent overlap = sensor signal)',
            f'Transmission (intensity)   :  {atten:.3f}   (I_det / I_in, core)',
        ]
        if np.isfinite(trans_S):
            lines += [
                f'Transmission (Poynting Sx) :  {trans_S:.3f}   '
                f'(directed power, full cross-section){sx_note}',
                f'   of which core only      :  {trans_S_core:.3f}',
                '',
                'Power fraction at detector (Sx flux, film = carried evanescently):',
                f'   eta_core = {eta_core:.4f}   eta_film = {eta_film:.4e}   eta_air = {eta_air:.4e}',
            ]
        self.ax.text(0.03, 0.97, '\n'.join(lines), color='#e6e6e6', fontsize=10.5,
                     family='monospace', va='top', ha='left',
                     transform=self.ax.transAxes)
        # kleine Balken (auf max normiert) fuer die drei Intensitaeten
        mx = max(P_guid, P_tear, P_air, 1e-30)
        for i, (v, l, c) in enumerate([(P_guid, 'Core', '#FFD75E'),
                                       (P_tear, 'Tear', '#56C4FF'),
                                       (P_air, 'Air', '#999999')]):
            yb = 0.10 - i*0.045
            w = 0.55*(v/mx)
            self.ax.add_patch(Rectangle((0.03, yb), w, 0.035, transform=self.ax.transAxes,
                                        facecolor=c, edgecolor='none'))
            self.ax.text(0.03 + w + 0.01, yb + 0.025, l, color='#bbbbbb',
                         fontsize=8, va='center', transform=self.ax.transAxes)
        _pl = f'  (plane {self.plane})' if getattr(self, 'is3d', False) else ''
        self.ax.set_title(f'Sensor evaluation{_pl}', color='#ffffff', fontsize=10)

    def _cad_contours(self, ax):
        """CAD: Konturen aller Koerper aus der Regionskarte (jede Ebene), Legende mit n."""
        r = self._reg_slice()
        if r is None:
            return
        a0, a1, b0, b1 = self.meta[min(self.frame_idx, len(self.meta)-1), :4]
        nx, nv = r.shape
        xs = a0 + (np.arange(nx) + 0.5)*(a1 - a0)/nx
        vs = b0 + (np.arange(nv) + 0.5)*(b1 - b0)/nv
        step = max(1, max(nx, nv)//1500)                 # Kontur auf Anzeigegroesse
        for reg in np.unique(r):
            if reg <= 0:
                continue
            m = (r[::step, ::step] == reg).astype(np.float32)
            ax.contour(xs[::step], vs[::step], m.T, levels=[0.5], colors=[self._cad_color(reg)],
                       linewidths=0.9, alpha=0.9)
            ax.plot([], [], color=self._cad_color(reg), lw=1.2,
                    label=f'{self._cad_name(reg)} (n={self._cad_n(reg):.3f})'
                          + ('  [core]' if reg == self.cad_core else '')
                          + ('  [film]' if reg == self.cad_tear else ''))

    def _layer_lines(self, ax, x_start, x_end, y_start):
        if getattr(self, 'is_cad', False):
            self._cad_contours(ax)
            return
        # Material-Namen/Indizes aus dem Schichtstapel (Kern, Schichten, Substrat)
        n_lens, i_lens = self.core_name, self.core_n
        n_cor, i_cor = self.sub_name, self.sub_n
        t_tot = self._stack_total()
        if self.planar or self.flat_mode:
            # Planar/Flattened: gerade WG bei y=0..PS_TOP (planar) bzw. y_rel=+-T/2 (flat)
            if self.planar:
                y_wg_top = self.wg_top_um
                y_wg_bot = PS_BOT
            else:
                # Flattened Lens: y_rel=0 = Linsenmitte, body bei +-T_LENS/2
                y_wg_top = T_LENS*1e6/2
                y_wg_bot = -T_LENS*1e6/2
            y_mu_bot = y_wg_bot - t_tot             # Stapel-Unterkante = Substrat-Oberkante
            # WG-Kern NUR zwischen den Facetten zeichnen (nicht ueber die ganze
            # Breite) + vertikale Grenzlinien an Eintritt und Ende, wenn die
            # Facetten-x bekannt sind (2D-Beads). Substrat-Linien laufen weiter,
            # da das Wasser real ueberall unter dem Slab liegt.
            _xe = getattr(self, 'x_wg_end', None)
            _xs = getattr(self, 'x_wg_start', 0.0) or 0.0
            if _xe:
                _a = max(x_start, _xs)
                _b = min(x_end, _xe)
                if _b > _a:
                    ax.plot([_a, _b], [y_wg_top, y_wg_top], color='#FFD75E', lw=1.0,
                            alpha=0.9, label=f'{n_lens} (n={i_lens:.3f})')
                    ax.plot([_a, _b], [y_wg_bot, y_wg_bot], color='#FFD75E', lw=1.0, alpha=0.9)
                for _xf in (_xs, _xe):
                    if x_start <= _xf <= x_end:
                        ax.plot([_xf, _xf], [y_wg_bot, y_wg_top], color='#FFD75E', lw=1.3)
                        ax.axvline(_xf, color='#FFD75E', lw=0.7, ls=':', alpha=0.5)
            else:
                ax.axhline(y_wg_top, color='#FFD75E', lw=1.0, alpha=0.9,
                           label=f'{n_lens} (n={i_lens:.3f})')
                ax.axhline(y_wg_bot, color='#FFD75E', lw=1.0, alpha=0.9)
            for (_nm, _yt, yb, _n), L in zip(self._stack_bounds(y_wg_bot), self.stack):
                if L.get('slot') == 'lipid' and L['t_um'] <= 1e-3:
                    continue                        # wie bisher: Lipid ~0 nicht zeichnen
                ax.axhline(yb, color=L['color'], ls=L['ls'], lw=0.9,
                           label=self._layer_label(L))
            ax.axhspan(y_start, y_mu_bot, color=self.SUB_COLOR, alpha=0.18,
                       label=f'{n_cor} (n={i_cor:.3f})')
        else:
            # Lens: gekruemmte Geometrie
            xs_out = np.linspace(x_start, x_end, 200)
            xs_clip = np.clip(xs_out, -R_BEND*1e6 + 1, R_BEND*1e6 - 1)
            sag = (R_BEND - np.sqrt(R_BEND**2 - (xs_clip*1e-6)**2))*1e6
            y_mid_o = -sag
            y_lens_top = y_mid_o + T_LENS*1e6/2
            y_lens_bot = y_mid_o - T_LENS*1e6/2
            y_mu_bot = y_lens_bot - t_tot
            ax.plot(xs_out, y_lens_top, '-', color='#FFD75E', lw=1.0, alpha=0.9,
                    label=f'{n_lens} (n={i_lens:.3f})')
            ax.plot(xs_out, y_lens_bot, '-', color='#FFD75E', lw=1.0, alpha=0.9)
            # alte Linsen-Dateien: alle drei Slots (wie bisher, auch Lipid 0)
            for (_nm, _yt, yb, _n), L in zip(stack_bounds(self.stack_report, 0.0),
                                             self.stack_report):
                ax.plot(xs_out, y_lens_bot + yb, L['ls'], color=L['color'], lw=0.9,
                        label=self._layer_label(L))
            ax.fill_between(xs_out, y_start, y_mu_bot, color=self.SUB_COLOR, alpha=0.18,
                            label=f'{n_cor} (n={i_cor:.3f})')

        # Einkoppelabstand + WG-Eintrittsfacette (x liegt auf der Horizontalen).
        self._wg_entrance_lines(ax, x_start, x_end)

    def _wg_entrance_lines(self, ax, x_start, x_end):
        """Zeichnet WG-Eintrittsfacette (Einkoppelabstand) und End-Facette als
        vertikale Linien in x. Nur wo x auf der Horizontalachse liegt
        (xy-Ebene, xz-Ebene, 2D-Laengsschnitt)."""
        xws = getattr(self, 'x_wg_start', 0.0)
        if xws and xws > 0 and x_start <= xws <= x_end:
            xsrc = getattr(self, 'x_src', 0.0)
            gap = xws - xsrc
            if gap > 0 and x_start <= xsrc <= x_end:
                ax.axvspan(xsrc, xws, color='#00E5FF', alpha=0.10,
                           label=f'Coupling gap {gap:.2f}um (air)')
                ax.axvline(xsrc, color='#00E5FF', lw=0.8, ls=':', alpha=0.7)
            ax.axvline(xws, color='#00E5FF', lw=1.3, ls='-.', alpha=0.95,
                       label=f'WG entry x={xws:.2f}um')
        ef = getattr(self, 'end_facet', 0.0)
        if ef and ef > 0:
            x_ef = x_end - ef
            if x_start <= x_ef <= x_end:
                ax.axvline(x_ef, color='#00E5FF', lw=1.3, ls='-.', alpha=0.8,
                           label=f'WG end x={x_ef:.2f}um')

    def _bead_cut(self, ax):
        """Zeichnet den Bead-SCHNITTKREIS in der aktuellen Ebene (3D). Der Radius
        ist der an der Schnittposition tatsaechliche: rc = sqrt(r^2 - d_perp^2),
        wobei d_perp der Abstand der Schnittebene zum Bead-Zentrum entlang der
        fixen Achse ist. Trifft die Ebene den Bead nicht (|d_perp|>r), wird nichts
        gezeichnet."""
        r = getattr(self, 'bead_d', 0.0)/2.0
        if r <= 0:
            return
        xc = getattr(self, 'bead_x', 0.0)
        yc = getattr(self, 'bead_y', -r)
        zc = getattr(self, 'bead_z', 0.0)
        pos = self._slice_pos_um()[0]        # Position der fixen Achse (um)
        if self.plane == 'xy':               # Anzeige x-y, fix z
            cx, cy, d_perp = xc, yc, pos - zc
        elif self.plane == 'xz':             # Anzeige x-z, fix y
            cx, cy, d_perp = xc, zc, pos - yc
        else:                                # yz: Anzeige y-z, fix x
            cx, cy, d_perp = yc, zc, pos - xc
        if abs(d_perp) > r:
            return                           # Ebene trifft den Bead nicht
        rc = float(np.sqrt(max(r*r - d_perp*d_perp, 0.0)))
        ax.add_patch(Circle((cx, cy), rc, fill=False, ec='#FFD400',
                            lw=1.6, zorder=6))
        ax.plot([cx], [cy], marker='+', color='#FFD400', ms=7, mew=1.4, zorder=7)
        ax.text(cx, cy + rc, f' bead cross-section d={2*rc:.2f}um', color='#FFD400',
                fontsize=8, ha='center', va='bottom', zorder=7)

    def _layer_marks(self, y_top, y_bot):
        """Datenabhaengige Schicht-Markierungen [(y, farbe, label)] aus den
        TATSAECHLICH berechneten Material-Namen/Indizes. Ueberspringt Schichten
        mit Dicke ~0 (z.B. Lipid beim Bead-Setup)."""
        if getattr(self, 'is_cad', False):
            # CAD: jede Grenzflaeche entlang der Spalte am x-Cursor, "unten | oben"
            segs = self._cad_segments(self.x_um)
            return [(a[1], self._cad_color(b[2]),
                     f'{self._cad_name(a[2])} | {self._cad_name(b[2])}')
                    for a, b in zip(segs[:-1], segs[1:])]
        cn, sn = self.core_name, self.sub_name
        marks = [(y_top, '#FFD75E', f'{cn}-Top (n={self.core_n:.3f})'),
                 (y_bot, '#FFD75E', f'{cn} bottom (y=0)'
                  + ('' if self.stack else f' -> {sn} (n={self.sub_n:.3f}) below'))]
        nst = len(self.stack)
        for k, ((_nm, _yt, yb, _n), L) in enumerate(zip(self._stack_bounds(y_bot), self.stack)):
            if L.get('slot') == 'lipid' and L['t_um'] <= 1e-3:
                continue                            # wie bisher: Lipid ~0 ueberspringen
            marks.append((yb, L['color'], self._layer_label(L)
                          + (f' -> {sn} below' if k == nst - 1 else '')))
        return marks

    def _draw_intavg(self):
        """Zeit-gemittelte Intensitaet <E^2>.

        BEVORZUGT das vom Solver EXAKT akkumulierte Iavg (jeder Zeitschritt der
        2. Lauf-Haelfte) -> echte Envelope. Nur als Fallback wird ueber die
        gespeicherten Snapshots gemittelt: die liegen bei full-domain oft fast
        phasengleich (Snapshot-Intervall ~ Vielfaches der Periode), dann bleibt
        der cos^2-Traeger stehen und man sieht Streifen mit lambda/2 - ein
        Sampling-Artefakt, keine Physik. In diesem Fall wird gewarnt."""
        nf = len(self.ezs)
        k = max(1, nf // 2)
        _iv = getattr(self, 'iavg2d', None)
        self._intavg_src = 'solver'
        self._intavg_warn = ''
        if _iv is not None and np.shape(_iv) == np.shape(self.ezs[0]):
            Iavg = np.asarray(_iv, dtype=np.float64)
        else:
            Iavg = np.mean(self.ezs[-k:].astype(np.float64)**2, axis=0)
            self._intavg_src = 'snapshots'
            # Phasengleichheit der gemittelten Frames pruefen -> Artefakt-Warnung
            try:
                fr = self.ezs[-k:].astype(np.float64).reshape(k, -1)
                if k >= 2:
                    ref = fr[-1]; sd = ref.std()
                    cs = [abs(np.corrcoef(fr[j], ref)[0, 1]) for j in range(k-1)
                          if fr[j].std() > 1e-12 and sd > 1e-12]
                    if cs and np.mean(cs) > 0.85:
                        self._intavg_warn = ('  [!] frames nearly in phase -> '
                                             'lambda/2 stripes are a sampling artifact')
            except Exception:
                pass
        x_start, x_end, y_start, y_end = self.meta[0, :4]
        Ishow = downsample_for_display(Iavg)
        vmax = float(Ishow.max())
        if vmax <= 0:
            vmax = 1.0
        vmin = vmax * 1e-4
        Ishow = np.maximum(Ishow, vmin * 1e-2)
        aspect = self.aspect_mode
        self.ax.imshow(Ishow.T, extent=[x_start, x_end, y_start, y_end],
                       origin='lower', cmap=self._cm_seq(),
                       norm=LogNorm(vmin=vmin, vmax=max(vmax*self._sat(), vmin*10)), aspect=aspect,
                       interpolation='nearest')
        if getattr(self, '_overlay_ok', True):
            self._layer_lines(self.ax, x_start, x_end, y_start)
            if (not getattr(self, 'is3d', False)) and getattr(self, 'is_bead', False) \
                    and self.bead_d > 0 and x_start <= self.bead_x <= x_end:
                self.ax.axvline(self.bead_x, color='#39FF14', lw=1.0, ls='--', alpha=0.85)
                self.ax.text(self.bead_x, y_start, f' Bead d={self.bead_d:g}um',
                             color='#39FF14', fontsize=8, ha='center', va='bottom')
        elif getattr(self, 'is3d', False) and getattr(self, 'plane', 'xy') == 'xz':
            self._wg_entrance_lines(self.ax, x_start, x_end)   # x auch in xz horizontal
        if getattr(self, 'is3d', False) and getattr(self, 'is_bead', False) and self.bead_d > 0:
            self._bead_cut(self.ax)          # Bead-Schnittkreis in JEDER Ebene
        self.ax.set_xlim(x_start, x_end)
        t_tot = self._stack_total()
        if getattr(self, 'is3d', False):
            self.ax.set_ylim(y_start, y_end)
            xl, yl = self.PLANE_LABELS[self.plane]
            self.ax.set_xlabel(xl, color='#cccccc', fontsize=9)
            self.ax.set_ylabel(yl, color='#cccccc', fontsize=9)
        elif self.planar and not getattr(self, 'is_cad', False):
            self.ax.set_ylim(max(y_start, -t_tot-2), min(y_end, self.wg_top_um+2))
        self._style_axes()
        _pl = f' | plane {self.plane}' if getattr(self, 'is3d', False) else ''
        _src = ('solver time average, exact' if getattr(self, '_intavg_src', '') == 'solver'
                else f'mean over last {k} frames{getattr(self, "_intavg_warn", "")}')
        _q = ('|E|' if getattr(self, '_intavg_src', '') == 'solver'
              and getattr(self, 'iavg_full', False) else self.prim)
        self.ax.set_title(f'Time-averaged intensity <{_q}^2> ({_src}, log){_pl}',
                          color='#ffffff', fontsize=10)
        if getattr(self, '_overlay_ok', True):
            leg = self.ax.legend(loc='upper left', fontsize=7, framealpha=0.6, facecolor='#1a1a1a')
            for t in leg.get_texts():
                t.set_color('#dddddd')

    def _draw_field(self, log=False):
        i = self.frame_idx
        x_start, x_end, y_start, y_end = self.meta[i, :4]
        ez = self.ezs[i]
        vmax_g = float(np.max(np.abs(ez)))
        if vmax_g < 1e-30:
            vmax_g = 1.0
        # Display-Downsampling: matplotlib kann bei 6250x1500 das RGBA-Bild
        # nicht allokieren. Wir slicen fuer die Anzeige (Daten bleiben unberuehrt).
        ez_show = downsample_for_display(ez)
        # Planar: aspect='auto' (sonst extrem schmaler Streifen)
        aspect = self.aspect_mode
        if getattr(self, '_is_magnitude', False):
            # |E| >= 0 -> sequentielle Colormap, kein +/- (Betrag)
            if log:
                vmin = max(vmax_g*1e-4, 1e-30)
                norm = LogNorm(vmin=vmin, vmax=max(vmax_g, vmin*10))
                self.ax.imshow(np.maximum(ez_show.T, 1e-30),
                               extent=[x_start, x_end, y_start, y_end],
                               origin='lower', cmap=self._cm_seq(), norm=norm,
                               aspect=aspect, interpolation='nearest')
            else:
                self.ax.imshow(ez_show.T, extent=[x_start, x_end, y_start, y_end],
                               origin='lower', cmap=self._cm_seq(), vmin=0, vmax=vmax_g*self._sat(),
                               aspect=aspect, interpolation='nearest')
        elif log:
            lt = max(vmax_g*1e-4, 1e-30)
            _vg = vmax_g*self._sat()
            norm = SymLogNorm(linthresh=lt, vmin=-_vg, vmax=_vg)
            self.ax.imshow(ez_show.T, extent=[x_start, x_end, y_start, y_end],
                           origin='lower', cmap=self._cm_div(), norm=norm,
                           aspect=aspect, interpolation='nearest')
        else:
            vm = vmax_g*0.5*self._sat()
            self.ax.imshow(ez_show.T, extent=[x_start, x_end, y_start, y_end],
                           origin='lower', cmap=self._cm_div(), vmin=-vm, vmax=vm,
                           aspect=aspect, interpolation='nearest')
        if getattr(self, '_overlay_ok', True):
            self._layer_lines(self.ax, x_start, x_end, y_start)
            if (not getattr(self, 'is3d', False)) and getattr(self, 'is_bead', False) \
                    and self.bead_d > 0 and x_start <= self.bead_x <= x_end:
                _r = self.bead_d/2.0
                self.ax.axvline(self.bead_x, color='#FFD400', lw=1.0, ls='--', alpha=0.85)
                self.ax.add_patch(Circle((self.bead_x, getattr(self, 'bead_y', -_r)), _r, fill=False, ec='#FFD400', lw=1.6, zorder=6))
                self.ax.text(self.bead_x, y_start, f' Bead d={self.bead_d:g}um', color='#FFD400',
                             fontsize=8, ha='center', va='bottom')
        elif getattr(self, 'is3d', False) and getattr(self, 'plane', 'xy') == 'xz':
            self._wg_entrance_lines(self.ax, x_start, x_end)   # x auch in xz horizontal
        if getattr(self, 'is3d', False) and getattr(self, 'is_bead', False) and self.bead_d > 0:
            self._bead_cut(self.ax)          # Bead-Schnittkreis in JEDER Ebene
        if not self.planar:
            for name, ec in [('D1', '#FF8800'), ('D2', '#00AAFF'), ('D3', '#55FF55')]:
                if any(x_start < p[0] < x_end for p in self.polys[name]):
                    self.ax.add_patch(MPoly(self.polys[name], closed=True,
                                            fill=False, ec=ec, lw=1.5))
        self.ax.axvline(self.x_um, color='#FFFFFF', lw=0.5, ls=':', alpha=0.5)
        self.ax.set_xlim(x_start, x_end)
        # Auto-Zoom y-Range auf interessante Region (WG/Linse + Tear-Stack)
        t_tot = self._stack_total()
        if getattr(self, 'is3d', False) or getattr(self, 'is_cad', False):
            # 3D-Schnitt bzw. CAD: volle Extents zeigen (Achsen je nach Ebene, s.u.)
            self.ax.set_ylim(y_start, y_end)
        elif self.planar:
            y_tear_bot = -t_tot - 2
            y_top_view = self.wg_top_um + 2
            self.ax.set_ylim(max(y_start, y_tear_bot), min(y_end, y_top_view))
        else:
            # Lens: y_mid(x) folgt der Kruemmung -> Linsen-Band wandert in y
            # innerhalb eines Slides (besonders an den Lens-Raendern bei x=+-7mm).
            # Sample y_mid an MEHREREN x-Positionen im Window und nimm min/max.
            xs_sample = np.linspace(x_start, x_end, 25)
            xs_clip = np.clip(xs_sample*1e-6, -R_BEND + 1e-9, R_BEND - 1e-9)
            sag_sample = R_BEND - np.sqrt(R_BEND**2 - xs_clip**2)
            y_mid_sample_um = -sag_sample*1e6
            y_mid_top = float(y_mid_sample_um.max())
            y_mid_bot = float(y_mid_sample_um.min())
            half_thick = T_LENS*1e6/2
            tear_total = t_tot
            # Zeige: hoechster Lens-Top + 30um Luft  bis  tiefster Cornea + 50um
            zoom_top = y_mid_top + half_thick + 30
            zoom_bot = y_mid_bot - half_thick - tear_total - 50
            # Innerhalb des Window-Bereichs clippen
            zoom_top = min(zoom_top, y_end)
            zoom_bot = max(zoom_bot, y_start)
            self.ax.set_ylim(zoom_bot, zoom_top)
        self._style_axes()
        info = (f"Frame {i+1}/{len(self.ezs)}  "
                f"Slide {int(self.meta[i,4])+1}")
        if self.meta.shape[1] >= 6:
            tvals = np.abs(np.asarray(self.meta[:, 5], dtype=np.float64))
            tmax = float(np.nanmax(tvals)) if tvals.size else 0.0
            if 0 < tmax < 1e-9:
                # gespeicherte Zeiten sind PHYSIKALISCH (Sekunden, sub-ps) -> Stitch/CW:
                # in fs anzeigen + Phase relativ zur optischen Periode.
                t_fs = float(self.meta[i, 5])*1e15
                T_fs = ((getattr(self, 'lam_nm', 850.0)*1e-9)/2.99792458e8)*1e15
                ph = (360.0*t_fs/T_fs) if T_fs > 0 else 0.0
                info += f"  t = {t_fs:.3f} fs   (CW phase {ph:.0f}° of 360°)"
            elif self.meta.shape[1] >= 7:
                # transiente Zeitschritte -> grobe Abschaetzung aus step
                step = int(self.meta[i, 6])
                dt_est_fs = 0.06 if self.planar else 0.18
                t_ps = step * dt_est_fs / 1000
                info += f"  step {step}  t = {t_ps:.3f} ps"
        if getattr(self, 'is3d', False):
            xl, yl = self.PLANE_LABELS[self.plane]
            self.ax.set_xlabel(xl, color='#cccccc', fontsize=9)
            self.ax.set_ylabel(yl, color='#cccccc', fontsize=9)
            pos, fixname = self._slice_pos_um()
            _cn = self._disp_comp()
            _cl = f'|E|=sqrt(Ex2+Ey2+Ez2)' if _cn == '|E|' else f'{_cn} component'
            info = (f"{_cl} | plane {self.plane} | "
                    f"{fixname}={pos:.2f} um (Idx {self.slice_idx})  -  " + info)
        self.ax.set_title(info, color='#ffffff', fontsize=10)
        if getattr(self, '_overlay_ok', True):
            leg = self.ax.legend(loc='upper left', fontsize=7, framealpha=0.6,
                                 facecolor='#1a1a1a')
            for t in leg.get_texts():
                t.set_color('#dddddd')

    def _find_frame_with_x(self, x_um):
        """Sucht den Frame mit hoechstem |Ez|, der x_um im Window-Bereich hat."""
        best_i = -1
        best_amp = -1.0
        for k in range(len(self.ezs)):
            xs0, xe0 = self.meta[k, 0], self.meta[k, 1]
            if xs0 <= x_um <= xe0:
                a = self.frame_amps[k]
                if a > best_amp:
                    best_amp = a
                    best_i = k
        return best_i

    def _sample_normal_cut(self, i, x_um, d_min_um=-50.0, d_max_um=80.0,
                            n_samples=400):
        """Samplet das Ez-Feld entlang der Linsen-NORMALEN bei x_um.
        d = Distanz von der Lens-Bot-Grenze, positiv = ins Tear-Film hinein.
        Fuer planare Geometrie ist die Normale (0, -1), also vertikal.
        Fuer gekruemmte Linse rotiert die Normale mit x_um.

        Returns: (d_array_um, ez_along_normal, x_bot_um, y_bot_um, n_x, n_y)
        """
        x_start, x_end, y_start, y_end = self.meta[i, :4]
        Nx, Ny = self.ezs[i].shape
        if self.planar or self.flat_mode:
            # Planar: Normale ist (0, -1), Cut ist vertikal
            y_lens_bot = self._wg_y_bounds(x_um)[0]
            n_x, n_y = 0.0, -1.0
            x_bot_um = x_um
        else:
            # Lens: Normale folgt der Kruemmung
            x_m = x_um*1e-6
            x_m_clip = max(-R_BEND + 1e-9, min(R_BEND - 1e-9, x_m))
            sag = R_BEND - np.sqrt(R_BEND**2 - x_m_clip**2)
            y_center = -sag*1e6
            y_lens_bot = y_center - T_LENS*1e6/2
            # Normale (zeigt INS Tear, also weg vom Krummungs-Zentrum-Vektor)
            n_x = -x_m_clip / R_BEND
            n_y = -np.sqrt(R_BEND**2 - x_m_clip**2) / R_BEND
            x_bot_um = x_um
        # Sample-Punkte entlang der Normalen (in um)
        # d < 0: ins Linsenmaterial hinein (Air-side oben)
        # d > 0: in den Tear-Film hinein
        d_arr = np.linspace(d_min_um, d_max_um, n_samples)
        sample_x = x_bot_um + d_arr * n_x   # negative d zieht in -n_x Richtung
        sample_y = y_lens_bot + d_arr * n_y
        # Konvertiere zu Array-Indizes
        ix_float = (sample_x - x_start) / (x_end - x_start) * (Nx - 1)
        iy_float = (sample_y - y_start) / (y_end - y_start) * (Ny - 1)
        # Bilinear-Interpolation
        try:
            from scipy.ndimage import map_coordinates
            ez_along = map_coordinates(self.ezs[i],
                                       np.array([ix_float, iy_float]),
                                       order=1, mode='constant', cval=0.0)
        except ImportError:
            # Fallback: Nearest Neighbor
            ix = np.clip(np.round(ix_float).astype(int), 0, Nx-1)
            iy = np.clip(np.round(iy_float).astype(int), 0, Ny-1)
            ez_along = self.ezs[i][ix, iy]
        return d_arr, ez_along, x_bot_um, y_lens_bot, n_x, n_y

    def _draw_line_cut(self):
        i = self.frame_idx
        x_start, x_end, y_start, y_end = self.meta[i, :4]
        if not (x_start <= self.x_um <= x_end):
            # KEIN Auto-Sprung mehr - Frame-Slider bleibt stehen wo Nutzer ihn hat.
            # Stattdessen: Hinweis-Text mit Empfehlung welcher Frame x abdeckt.
            j = self._find_frame_with_x(self.x_um)
            if j >= 0:
                msg = (f'x = {self.x_um:.0f} um NOT in current frame {i+1}\n'
                       f'(Window x = [{x_start:.0f}, {x_end:.0f}] um).\n'
                       f'-> Set the frame slider to frame {j+1}\n'
                       f'   OR choose an x within the current window.')
            else:
                msg = (f'x = {self.x_um:.0f} um is not in any slide window.\n'
                       f'Current frame {i+1}: x = [{x_start:.0f}, {x_end:.0f}] um')
            self.ax.text(0.5, 0.5, msg, color='#FFAA55',
                         ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=10,
                         bbox=dict(facecolor='#1a1a1a', edgecolor='#FFAA55',
                                   alpha=0.85))
            self._style_axes()
            return
        Nx, Ny = self.ezs[i].shape
        ix = int(round((self.x_um - x_start)/(x_end - x_start)*Nx))
        ix = max(0, min(Nx-1, ix))
        ez_cut = self.ezs[i, ix, :]
        y_um = np.linspace(y_start, y_end, Ny)
        # Auto-Zoom: y-Range auf WG-Region + Tear-Region
        y_lens_bot, y_lens_top = self._wg_y_bounds(self.x_um)
        y_mu_bot = y_lens_bot - self._stack_total()     # Stapel-Unterkante
        # Bei planar Demo enger zoomen (Schichten sind nur ~5 um) als bei Lens
        if self.planar:
            zoom_top = y_lens_top + 2
            zoom_bot = y_mu_bot - 2
            top_label = 'WG-Top'
            bot_label = 'WG-Bot'
        else:
            zoom_top = y_lens_top + 30
            zoom_bot = y_mu_bot - 30
            top_label = 'Lens-Top'
            bot_label = 'Lens-Bot'
        # Maske fuer Y-Auto-Skalierung
        mask = (y_um >= zoom_bot) & (y_um <= zoom_top)
        self.ax.plot(y_um, ez_cut, 'cyan', lw=1.0, label=self.prim)
        if mask.any():
            ez_zoom = ez_cut[mask]
            vmax = float(np.max(np.abs(ez_zoom)))
            if vmax > 1e-30:
                self.ax.set_ylim(-vmax*1.1, vmax*1.1)
        self.ax.set_xlim(zoom_bot, zoom_top)
        self.ax.set_xlabel('y (um)  — zoom on lens + tear region', color='#cccccc')
        self.ax.set_ylabel(self._disp_comp(), color='#cccccc')
        self.ax.grid(True, alpha=0.2)
        for y_line, color, label in self._layer_marks(y_lens_top, y_lens_bot):
            self.ax.axvline(y_line, color=color, ls='--', lw=1, alpha=0.7,
                            label=label)
        self._style_axes()
        self.ax.set_title(
            f'Line cut at x={self.x_um:.0f} um  -  frame {i+1}/{len(self.ezs)}  '
            f'Slide {int(self.meta[i,4])+1}', color='#ffffff', fontsize=10)
        leg = self.ax.legend(loc='upper right', fontsize=7, facecolor='#1a1a1a')
        for t in leg.get_texts():
            t.set_color('#dddddd')

    def _draw_evanescent(self):
        if not (self.planar or self.flat_mode) and not (-7000 < self.x_um < 7000):
            # nur alte gekruemmte Linse (Sag-Koordinaten, x um den Apex)
            self.ax.text(0.5, 0.5, 'x outside lens',
                         color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes)
            return
        y_lens_bot_um, y_lens_top_um = self._wg_y_bounds(self.x_um)
        # Anzuzeigendes Frame folgt dem FRAME-SLIDER; faellt auf ein x-abdeckendes
        # Frame zurueck, falls das aktuelle Frame x / die WG-Unterkante nicht enthaelt
        # (fuer Stitch-Daten, wo Frames unterschiedliche x-Fenster abdecken).
        covering = [k for k in range(len(self.ezs))
                    if self.meta[k, 0] <= self.x_um <= self.meta[k, 1]
                    and self.meta[k, 2] < y_lens_bot_um < self.meta[k, 3]]
        if not covering:
            self.ax.text(0.5, 0.5,
                         f'No frame with x={self.x_um:.0f} and WG bottom edge in range',
                         color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes)
            return
        i = (self.frame_idx if self.frame_idx in covering
             else min(covering, key=lambda k: abs(k - self.frame_idx)))
        x_start, x_end, y_start, y_end = self.meta[i, :4]
        Nx, Ny = self.ezs[i].shape
        ix = int(round((self.x_um - x_start)/(x_end - x_start)*Nx))
        ix = min(max(ix, 0), Nx - 1)
        y_um = np.linspace(y_start, y_end, Ny)
        abs_ez = np.abs(self.ezs[i, ix, :].astype(np.float64))   # Momentaufnahme (folgt Slider)
        max_amp = float(abs_ez.max())

        # Robuste Fit-Groesse: knotenfreie CW-Amplituden-Einhuellende |Ê| aus dem
        # eingeschwungenen Phasensatz desselben x-Fensters (|Ê| = sqrt(2<E^2>)),
        # sonst die vom Slider gewaehlte Momentaufnahme.
        # Komponente waehlbar (Feld-Radio): TE Ez (tangential); TM Ey (NORMAL zur
        # Grenzflaeche, springt dort um eps1/eps2), Ex (tangential, stetig) oder |E|
        # (alle Komponenten). Fit und Kurve folgen der Wahl, nicht nur der Beschriftung.
        _comp = getattr(self, 'comp', self.prim)
        if _comp not in self._comp_options():
            _comp = self.prim
        _snap_comp = self._disp_comp()               # was self.ezs (Momentbild) enthaelt
        self._evan_fit_src = f'snapshot (frame {i+1})'
        fit_amp = abs_ez if _snap_comp == _comp else None
        _env, _src = self._evan_envelope(_comp, i, ix, Nx, Ny)
        if _env is not None:
            fit_amp = _env
            self._evan_fit_src = _src
        if fit_amp is None:                          # gewaehlte Komponente hier nicht greifbar
            fit_amp = abs_ez
            self._evan_fit_src = (f'snapshot {_snap_comp} (frame {i+1}; '
                                  f'{_comp} not available for this image)')
            _comp = _snap_comp

        # Auto-Zoom auf Lens/WG + Tear-Region
        _cad_b = self._cad_below(self.x_um) if getattr(self, 'is_cad', False) else None
        if getattr(self, 'is_cad', False):
            # CAD: Bereich unter dem Kern = Region direkt darunter (z.B. der Film)
            y_aq_bot_z = y_mu_bot_z = (_cad_b[1] if _cad_b else y_lens_bot_um - 5.0)
        else:
            _tt = self._stack_total()
            if _tt <= 0:                    # kein Stapel: Substrat direkt unter dem Kern
                _tt = max(1.0, 3.0*getattr(self, '_delta_min_um', 0.3))
            y_mu_bot_z = y_lens_bot_um - _tt
        if self.planar:
            # WG-Top an der tatsaechlichen Kerndicke (nicht fest 5um)
            _top = y_lens_top_um if getattr(self, 'is_cad', False) else self.wg_top_um
            zoom_top = _top + 2
            zoom_bot = y_mu_bot_z - 2
            top_marker_y = _top
            top_marker_label = 'WG-Top'
        else:
            zoom_top = y_lens_top_um + 30
            zoom_bot = y_mu_bot_z - 30
            top_marker_y = y_lens_top_um
            top_marker_label = 'Lens-Top'

        # Daten ueberhaupt aussagekraeftig?
        if max_amp < 1e-15:
            self.ax.text(0.5, 0.5,
                         f'NPZ empty / Mur-absorbed\n'
                         f'max|{self.prim}| at x={self.x_um:.0f} um = {max_amp:.1e}\n\n'
                         f'No evanescent fit possible.\n'
                         f'-> multi-snapshot run required.',
                         color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=10,
                         bbox=dict(facecolor='#330000', edgecolor='#FFAA55', alpha=0.7))
            self._style_axes()
            return

        # NaN/Negativ-Schutz fuer semilogy (sonst matplotlib UserWarning)
        floor = max(max_amp*1e-6, 1e-30)
        ez_safe = np.maximum(np.nan_to_num(abs_ez, nan=floor, posinf=floor,
                                            neginf=floor), floor)
        flab = _comp if _comp == '|E|' else f'|{_comp}|'
        if fit_amp is not abs_ez:
            # Hauptkurve = knotenfreie CW-Amplitude |Ê| (dieselbe Groesse, die gefittet
            # wird). Das Momentbild geht durch Nulldurchgaenge des Feldes (scharfe
            # Dips im log-Plot, keine Amplitudenminima) -> nur blass im Hintergrund,
            # und nur, wenn es dieselbe Komponente ist.
            if _snap_comp == _comp:
                self.ax.semilogy(y_um, ez_safe, color='#557777', lw=0.8, alpha=0.6,
                                 label=f'{flab} snapshot (frame {i+1}, zero crossings)')
            abs_ez = np.asarray(fit_amp, dtype=np.float64)
            max_amp = float(abs_ez.max())
            floor = max(max_amp*1e-6, 1e-30)
            self.ax.semilogy(y_um, np.maximum(abs_ez, floor), 'cyan', lw=1.4,
                             label=f"|{_comp.strip('|')}| CW amplitude (envelope over one period)")
        else:
            self.ax.semilogy(y_um, ez_safe, 'cyan', lw=1.2, label=flab)
        self.ax.set_xlim(zoom_bot, zoom_top)
        # Besseres y-Range: an Tear-Film-Bereich orientieren (nicht globalem Max).
        # Tear-Bereich liegt zwischen y_lens_bot und y_mu_bot.
        tear_mask = (y_um <= y_lens_bot_um) & (y_um >= y_mu_bot_z)
        if tear_mask.any() and abs_ez[tear_mask].max() > max_amp*1e-8:
            tear_max = float(abs_ez[tear_mask].max())
            tear_min = float(np.maximum(abs_ez[tear_mask], max_amp*1e-8).min())
            # Untere Grenze am Tear-Film orientiert (Decay sichtbar), obere am
            # GLOBALEN Max -> Kern-Peak wird nicht abgeschnitten (passt ins Bild).
            y_lo = max(tear_min/3, max_amp*1e-8)
            y_hi = max_amp*2.0
        else:
            y_lo = max_amp*1e-5
            y_hi = max_amp*2
        self.ax.set_ylim(y_lo, y_hi)
        # Erzwinge Log-Ticks an jeder Dekade + Minor-Ticks (sonst zeigt mpl bei
        # breitem flachem Plot nur 2 Ticks gesamt).
        from matplotlib.ticker import LogLocator, NullFormatter
        self.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
        self.ax.yaxis.set_minor_locator(LogLocator(base=10,
                                                    subs=[2.0, 3.0, 5.0, 7.0],
                                                    numticks=60))
        self.ax.yaxis.set_minor_formatter(NullFormatter())
        self.ax.grid(True, which='major', alpha=0.3)
        self.ax.grid(True, which='minor', alpha=0.1, ls=':')

        # Fit-Bereich aus User-Slidern (oder Default)
        # self.fit_top: 0..-3 (in um, relativ zu PS-Bot, negativ = unter PS-Bot)
        # self.fit_depth: 0.1..3 um Fit-Tiefe
        y_fit_top = y_lens_bot_um + self.fit_top   # fit_top ist negativ
        y_fit_bot = y_lens_bot_um + self.fit_top - self.fit_depth
        mask = (y_um <= y_fit_top) & (y_um >= y_fit_bot)
        # Material-Konstanten fuer Theorie-Vergleich: WG-Index UND Wellenlaenge
        # aus dem TATSAECHLICHEN Lauf (frueher fest 850nm + Polystyrol 1.59 -> bei
        # z.B. PMMA/532nm falsches d_min und falscher Winkel).
        _mi = getattr(self, 'mat_indices', None)
        n_high = float(_mi[0]) if _mi else (N_POLYSTYR if self.planar else N_PMMA)
        n_low = self._clad_n(legacy=self.layers[3])   # Medium unter dem Kern (alt: n_aqueous)
        if getattr(self, 'is_cad', False):              # Kern gegen die Region darunter
            n_high = self._cad_n(self.cad_core)
            n_low = self._cad_n(_cad_b[0]) if _cad_b else 1.0
        lam_um = getattr(self, 'lam_nm', 850.0)/1000.0
        # Theoretische min Decay (grazing incidence)
        d_min_um = lam_um / (2*np.pi*np.sqrt(max(n_high**2 - n_low**2, 1e-10)))

        if mask.sum() >= 5 and fit_amp[mask].max() > fit_amp.max()*1e-5:
            y_fit = y_lens_bot_um - y_um[mask]   # Distanz unter PS-Bot
            log_a = np.log(np.maximum(fit_amp[mask], 1e-30))
            sort_idx = np.argsort(y_fit)
            y_fit_s = y_fit[sort_idx]
            log_a_s = log_a[sort_idx]
            if len(y_fit_s) >= 3:
                slope, intercept = np.polyfit(y_fit_s, log_a_s, 1)
                A0 = float(np.exp(intercept))
                # R^2 berechnen
                log_a_fit = slope*y_fit_s + intercept
                ss_res = float(np.sum((log_a_s - log_a_fit)**2))
                ss_tot = float(np.sum((log_a_s - log_a_s.mean())**2))
                r2 = 1.0 - ss_res/ss_tot if ss_tot > 1e-30 else 0.0
                # Decay-Status-Diagnose
                if slope >= 0:
                    decay_um = float('nan')
                    decay_label = 'NaN [slope>=0, wave grows]'
                    angle_label = 'NaN [no decay]'
                else:
                    decay_um = -1.0/slope
                    decay_label = f'{decay_um:.3f} um'
                    if decay_um < d_min_um*0.5:
                        angle_label = 'NaN [d < d_min, unphysical]'
                    else:
                        k = lam_um / (2*np.pi*decay_um)
                        sin2_th = (n_low**2 + k**2) / n_high**2
                        if sin2_th > 1:
                            angle_label = f'NaN [sin²θ>1: d too small]'
                        elif sin2_th < (n_low/n_high)**2:
                            # Unter dem kritischen Winkel
                            theta = float(np.degrees(np.arcsin(np.sqrt(max(sin2_th, 0)))))
                            angle_label = f'{theta:.1f}° [<θ_c=Sub-Critical]'
                        else:
                            theta = float(np.degrees(np.arcsin(np.sqrt(sin2_th))))
                            theta_c = float(np.degrees(np.arcsin(n_low/n_high)))
                            status = 'TIR' if theta > theta_c else 'critical'
                            angle_label = f'{theta:.1f}° [{status}, θ_c={theta_c:.1f}°]'
                # Plot
                y_plot = np.linspace(y_fit_s.min(), y_fit_s.max(), 30)
                self.ax.semilogy(y_lens_bot_um - y_plot,
                                 np.exp(intercept + slope*y_plot),
                                 'r-', lw=2.0, alpha=0.95,
                                 label=(f'Penetration depth δ = {decay_label}\n'
                                        f'(theor. minimum δ_min = {d_min_um:.3f} um)\n'
                                        f'R²={r2:.3f}   θ ≈ {angle_label} [ray model]\n'
                                        f'Fit: {self._evan_fit_src}'))
                # Fit-Bereich markieren
                self.ax.axvspan(y_lens_bot_um - y_fit_s.max(),
                                y_lens_bot_um - y_fit_s.min(),
                                color='#FF4400', alpha=0.15, zorder=0)
        for y_line, color, label in self._layer_marks(top_marker_y, y_lens_bot_um):
            self.ax.axvline(y_line, color=color, ls='--', lw=1.2, alpha=0.8,
                            label=label)
        if getattr(self, 'show_bead_evan', False):       # zuschaltbar (Knopf/Taste 'b')
            self._bead_evan_marks(y_lens_bot_um)
        self.ax.set_xlabel('y (um) — zoom on lens + tear region',
                           color='#cccccc')
        self.ax.set_ylabel(f'{flab} (log)', color='#cccccc')
        self.ax.set_title(f'Evanescent decay at x={self.x_um:.0f} um  '
                          f'(Frame {i+1}, Slide {int(self.meta[i,4])+1})',
                          color='#ffffff', fontsize=10)
        self.ax.grid(True, alpha=0.2, which='both')
        leg = self.ax.legend(loc='upper left', fontsize=6.5, framealpha=0.65,
                             facecolor='#1a1a1a', labelspacing=0.3)
        for t in leg.get_texts():
            t.set_color('#dddddd')
        self._style_axes()

    def _draw_resonance(self):
        """|Ez|^2 Profil durch alle Schichten mit Schicht-Annotationen.
        Zeigt resonante Effekte (Standing Waves, Frustrated TIR) durch alle Materialien."""
        i = self.frame_idx
        x_start, x_end, y_start, y_end = self.meta[i, :4]
        if not (x_start <= self.x_um <= x_end):
            j = self._find_frame_with_x(self.x_um)
            if j >= 0:
                msg = f'x={self.x_um:.0f}um not in frame {i+1}. -> frame {j+1}'
            else:
                msg = f'x={self.x_um:.0f}um not in any frame'
            self.ax.text(0.5, 0.5, msg, color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=11)
            self._style_axes()
            return
        Nx, Ny = self.ezs[i].shape
        ix = int(round((self.x_um - x_start)/(x_end - x_start)*Nx))
        ix = max(0, min(Nx-1, ix))
        ez_col = self.ezs[i, ix, :]
        intensity = ez_col*ez_col  # Momentaufnahme der Hauptkomponente
        _iv = self._iavg_ok()
        _res_avg = _iv is not None and not getattr(self, '_is_magnitude', False)
        if _res_avg:                 # zeitgemittelt, alle Komponenten (knotenfrei)
            intensity = _iv[ix, :]
        y_um = np.linspace(y_start, y_end, Ny)
        # Schicht-Grenzen
        y_wg_bot, y_wg_top = self._wg_y_bounds(self.x_um)
        if not np.any(np.asarray(intensity) > 0):       # z.B. erstes Bild vor dem Puls
            self.ax.text(0.5, 0.5, f'No field at x={self.x_um:.1f} um in this image\n'
                         f'(move the frame slider or use "Steady state")', color='#FFAA55',
                         ha='center', va='center', transform=self.ax.transAxes)
            self._style_axes()
            return
        # Plot
        intensity_safe = np.maximum(intensity, 1e-30)
        _comp = self._disp_comp()
        _fl = _comp if _comp == '|E|' else f'|{_comp}|'
        self.ax.semilogy(y_um, intensity_safe, 'cyan', lw=1.0,
                         label=('<|E|^2> time average (all components)' if _res_avg
                                else f'{_fl}^2 (snapshot)'))
        ymax_y = float(intensity_safe.max())*5
        ymin_y = max(float(intensity_safe.min()), ymax_y*1e-10)
        if getattr(self, 'is_cad', False):
            # CAD: Abschnitte der Spalte am x-Cursor faerben + Integral je Koerper
            dy = float(y_um[1] - y_um[0])
            sums = {}
            for lo, hi, reg in self._cad_segments(self.x_um):
                self.ax.axvspan(lo, hi, color=self._cad_color(reg), alpha=0.13 if reg else 0.05)
                m = (y_um >= lo) & (y_um < hi)
                sums[reg] = sums.get(reg, 0.0) + (float(intensity[m].sum())*dy if m.any() else 0.0)
            annot = 'Integrated per body (column):\n' + '\n'.join(
                f'  {self._cad_name(r)}: {v:.2e}' for r, v in sorted(sums.items()))
            self.ax.text(0.02, 0.98, annot, color='#dddddd', va='top', ha='left',
                         transform=self.ax.transAxes, fontsize=8,
                         bbox=dict(facecolor='#1a1a1a', edgecolor='#444', alpha=0.85))
            for reg in sorted(sums):
                self.ax.plot([], [], color=self._cad_color(reg), lw=6, alpha=0.5,
                             label=self._cad_name(reg))
            self.ax.set_xlim(y_start, y_end)
        else:
            self._resonance_layers(y_um, intensity, y_start, y_wg_bot, y_wg_top)
        self.ax.set_ylim(ymin_y, ymax_y)
        from matplotlib.ticker import LogLocator, NullFormatter
        self.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
        self.ax.yaxis.set_minor_locator(LogLocator(base=10,
                                                    subs=[2.0, 5.0],
                                                    numticks=30))
        self.ax.yaxis.set_minor_formatter(NullFormatter())
        self.ax.set_xlabel('y (um)', color='#cccccc')
        self.ax.set_ylabel('<|E|^2> (log)' if _res_avg else f'{_fl}^2 (log)', color='#cccccc')
        self.ax.set_title(f'Resonance profile at x={self.x_um:.0f}um  '
                          f'(Frame {self.frame_idx+1}, Slide {int(self.meta[self.frame_idx,4])+1})',
                          color='#ffffff', fontsize=10)
        self.ax.grid(True, alpha=0.25, which='both')
        leg = self.ax.legend(loc='upper right', fontsize=7, facecolor='#1a1a1a')
        for t in leg.get_texts():
            t.set_color('#dddddd')
        self._style_axes()

    def _resonance_layers(self, y_um, intensity, y_start, y_wg_bot, y_wg_top):
        """Schichtstapel: Bereiche faerben (ein Band je Schicht), Integral je Schicht, Zoom."""
        self.ax.axvspan(y_wg_bot, y_wg_top, color=self.CORE_COLOR, alpha=0.10,
                        label=f'{self.core_name}')
        rep = self.stack_report
        bnds = stack_bounds(rep, y_wg_bot)
        y_mu_bot = y_wg_bot - stack_total_um(rep)          # Substrat-Oberkante
        for (nm, yt, yb, _n), L in zip(bnds, rep):
            if L['t_um'] > 0:
                self.ax.axvspan(yb, yt, color=L['color'], alpha=L.get('alpha', 0.15),
                                label=f'{nm}')
        self.ax.axvspan(y_start, y_mu_bot, color=self.SUB_COLOR, alpha=0.18,
                        label=f'{self.sub_name}')
        # Schicht-integrierte Intensitaeten als Annotation
        dy = float(y_um[1] - y_um[0])
        def integ(y_lo, y_hi):
            m = (y_um >= y_lo) & (y_um <= y_hi)
            return float(intensity[m].sum())*dy if m.any() else 0.0
        rows = [(self.core_name, integ(y_wg_bot, y_wg_top))]
        rows += [(nm, integ(yb, yt) if L['t_um'] > 0 else 0) for (nm, yt, yb, _n), L in zip(bnds, rep)]
        rows.append((self.sub_name, integ(y_start, y_mu_bot)))
        annot = (f'Integrated |{self.prim}|^2 per layer:\n'
                 + '\n'.join(f'  {nm}: {v:.2e}' for nm, v in rows))
        self.ax.text(0.02, 0.98, annot, color='#dddddd', va='top', ha='left',
                     transform=self.ax.transAxes, fontsize=8,
                     bbox=dict(facecolor='#1a1a1a', edgecolor='#444', alpha=0.85))
        # Zoom auf interessante Region
        zoom_bot = y_mu_bot - 1
        zoom_top = y_wg_top + 1
        self.ax.set_xlim(zoom_bot, zoom_top)

    def _draw_layer_power(self):
        """Bar-Chart: integrierte |Ez|^2 pro Schicht."""
        # Balken: Kern, je Stapel-Schicht einer (alte Dateien: alle 3 Slots), Substrat
        rep = self.stack_report
        layer_names = [self.core_name] + [L['name'] for L in rep] + [self.sub_name]
        layer_colors = [self.CORE_COLOR] + [L['color'] for L in rep] + [self.SUB_COLOR]
        i = -1
        x_start, x_end, y_start, y_end = self.meta[i, :4]
        Nx, Ny = self.ezs[i].shape
        y_um = np.linspace(y_start, y_end, Ny)
        intensity = self.ezs[i]*self.ezs[i]
        _iv = self._iavg_ok()
        _lp_avg = _iv is not None and not getattr(self, '_is_magnitude', False)
        if _lp_avg:                  # Zeitmittel <|E|^2>, alle Komponenten
            intensity = _iv
        if getattr(self, 'is_cad', False) and self._reg_slice() is not None:
            # CAD: Integral der Intensitaet ueber jeden Koerper (ganzer Schnitt)
            r = self._reg_slice()
            regs = [int(v) for v in np.unique(r)]
            vals = [float(intensity[r == g].sum()) for g in regs]
            safe = np.maximum(vals, 1e-30)
            xp_ = np.arange(len(regs))
            bars = self.ax.bar(xp_, safe, color=[self._cad_color(g) for g in regs],
                               edgecolor='#888', lw=0.7)
            for bar, v in zip(bars, vals):
                self.ax.text(bar.get_x() + bar.get_width()/2, max(v, 1e-30)*1.5, f'{v:.2e}',
                             color='#dddddd', ha='center', va='bottom', fontsize=8)
            self.ax.set_yscale('log')
            self.ax.set_ylim(float(safe.max())*1e-7, float(safe.max())*30)
            self.ax.set_xticks(xp_)
            self.ax.set_xticklabels([self._cad_name(g) + (' [core]' if g == self.cad_core else '')
                                     for g in regs], color='#cccccc', rotation=15)
            self.ax.set_ylabel('Integrated <|E|^2> (time avg., log)' if _lp_avg else
                               f'Integrated |{self._disp_comp()}|^2 (log)',
                               color='#cccccc')
            _pl = f'  (plane {self.plane})' if getattr(self, 'is3d', False) else ''
            self.ax.set_title(f'Power per body - {self.cad_file}{_pl}', color='#ffffff', fontsize=10)
            self.ax.grid(True, alpha=0.25, axis='y', which='both')
            self._style_axes()
            return
        y_wg_bot, y_wg_top = self._wg_y_bounds((x_start + x_end)/2)
        bnds = stack_bounds(rep, y_wg_bot)
        y_mu_bot = y_wg_bot - stack_total_um(rep)          # Substrat-Oberkante
        def integ_layer(y_lo, y_hi):
            m = (y_um >= y_lo) & (y_um <= y_hi)
            if not m.any():
                return 0.0
            return float(intensity[:, m].sum())
        I_lens = integ_layer(y_wg_bot, y_wg_top)
        I_lay = [integ_layer(yb, yt) if L['t_um'] > 0 else 0
                 for (_nm, yt, yb, _n), L in zip(bnds, rep)]
        # Bead: LOKALES Integral ueber die Bead-Scheibe (x=Bead_x +- r, y 0..-d).
        # Alte Dateien (Slot 1 hiess 'Bead'): ersetzt den Slot; neue Dateien:
        # eigener zusaetzlicher Balken, die Schichtnamen bleiben unveraendert.
        I_bead = None
        if getattr(self, 'is_bead', False) and self.bead_d > 0:
            _r = self.bead_d/2.0
            _xu = np.linspace(x_start, x_end, Nx)
            _xm = (_xu >= self.bead_x - _r) & (_xu <= self.bead_x + _r)
            _yc = getattr(self, 'bead_y', -_r)          # Bead-Mitte (auch verschoben, --bead-y-um)
            _ym = (y_um >= _yc - _r) & (y_um <= _yc + _r)
            I_bead = float(intensity[np.ix_(_xm, _ym)].sum()) if (_xm.any() and _ym.any()) else 0.0
            if getattr(self, '_legacy_bead_slot', False) and I_lay:
                I_lay[0], I_bead = I_bead, None
        I_cor = integ_layer(y_start, y_mu_bot)
        I_all = [I_lens] + I_lay + [I_cor]
        if I_bead is not None:
            I_all.append(I_bead)
            _bm = getattr(self, 'bead_material', '')
            layer_names.append(f'Bead ({_bm}, local)' if _bm else 'Bead (local)')
            layer_colors.append('#FF7F50')
        x_pos = np.arange(len(layer_names))
        powers_safe = np.maximum(I_all, 1e-30)
        bars = self.ax.bar(x_pos, powers_safe, color=layer_colors,
                            edgecolor='#888', lw=0.7)
        for bar, val in zip(bars, I_all):
            self.ax.text(bar.get_x() + bar.get_width()/2,
                         max(val, 1e-30)*1.5,
                         f'{val:.2e}', color='#dddddd',
                         ha='center', va='bottom', fontsize=8)
        self.ax.set_yscale('log')
        # Kopf-Freiraum, damit die Wert-Labels ueber den Balken ins Bild passen.
        _mx = float(powers_safe.max())
        self.ax.set_ylim(_mx*1e-7, _mx*30)
        self.ax.set_xticks(x_pos)
        if len(layer_names) > 6:             # viele Schichten: schraeg + kleiner
            self.ax.set_xticklabels(layer_names, color='#cccccc', rotation=12, ha='right',
                                    fontsize=8)
        else:
            self.ax.set_xticklabels(layer_names, color='#cccccc')
        self.ax.set_ylabel('Integrated <|E|^2> (time avg., log)' if _lp_avg else
                           f'Integrated |{self._disp_comp()}|^2 (log)',
                           color='#cccccc')
        _bt = '  (bead = local integral around x=%.0fum)' % getattr(self,'bead_x',500.0) if getattr(self,'is_bead',False) else ''
        self.ax.set_title(f'Layer Power - {self.scenario}{_bt}',
                          color='#ffffff', fontsize=10)
        self.ax.grid(True, alpha=0.25, axis='y', which='both')
        self._style_axes()

    def _draw_time_trend(self):
        amps = self.frame_amps   # gecached
        slides = self.meta[:, 4].astype(int)
        # x_start pro Frame (in um) -> Sliding-Window-Position
        x_starts_um = self.meta[:, 0]
        # Schutz gegen leere/Null/NaN Daten (alte NPZs ohne Multi-Frame)
        amps_clean = np.nan_to_num(amps, nan=1e-30, posinf=1e-30, neginf=1e-30)
        amps_safe = np.maximum(amps_clean, 1e-30)
        max_amp = float(np.nanmax(amps_clean)) if len(amps_clean) else 0.0
        use_log = max_amp > 1e-15   # nur log wenn signifikante Werte

        # --- Slide-Hintergrundbaender (abwechselnd faerben) ---
        plot_fn_amps = amps_safe  # umbenennen fuer Klarheit
        per_sl_arr = amps_clean
        n_sl = int(slides.max()) + 1
        # finde Frame-Range jedes Slides
        slide_ranges = []
        for s in range(n_sl):
            idx = np.where(slides == s)[0]
            if len(idx):
                slide_ranges.append((s, idx[0] - 0.5, idx[-1] + 0.5))
        # alternierende Baender
        for k, (s, x0, x1) in enumerate(slide_ranges):
            col = '#1a2a3a' if k % 2 == 0 else '#0a1520'
            self.ax.axvspan(x0, x1, color=col, alpha=0.6, zorder=0)
        # Slide-Nummern + x_start in um am oberen Rand
        ax_top = self.ax.secondary_xaxis('top')
        slide_centers = [(x0 + x1)/2 for _, x0, x1 in slide_ranges]
        slide_labels = [f'Slide {s}\nx={x_starts_um[slides == s][0]:.0f}um'
                        for s, _, _ in slide_ranges]
        ax_top.set_xticks(slide_centers)
        ax_top.set_xticklabels(slide_labels, color='#FFD75E', fontsize=8)
        ax_top.tick_params(axis='x', colors='#FFD75E', length=4)
        for sp in ax_top.spines.values():
            sp.set_color('#444444')

        plot_fn = self.ax.semilogy if use_log else self.ax.plot
        plot_fn(np.arange(len(amps_clean)), amps_safe, 'o-',
                color='#FF8800', lw=1, ms=4, label='max |Ez| per frame', zorder=3)
        per_sl = np.array([amps_clean[slides == s].mean() if (slides == s).any() else 1e-30
                           for s in range(n_sl)])
        per_sl_safe = np.maximum(np.nan_to_num(per_sl, nan=1e-30), 1e-30)
        slide_x = [np.where(slides == s)[0].mean() for s in range(n_sl)]
        plot_fn(slide_x, per_sl_safe, 's-', color='#56C4FF', lw=2, ms=6,
                label='mean per slide', zorder=4)
        # Slide-Grenz-Linien (vertikal, gestrichelt)
        for _, x0, x1 in slide_ranges[:-1]:
            self.ax.axvline(x1, color='#FFD75E', lw=0.6, ls='--', alpha=0.5, zorder=1)
        self.ax.axvline(self.frame_idx, color='#ffffff', lw=1.2, ls=':', alpha=0.8,
                        label=f'Frame {self.frame_idx+1}', zorder=5)
        if not use_log:
            self.ax.text(0.5, 0.95,
                         f'WARNING: all |Ez| < 1e-15 (NPZ empty or Mur-absorbed)\n'
                         f'-> multi-snapshot run recommended (new sliding_window_fdtd.py)',
                         color='#FFAA55', ha='center', va='top',
                         transform=self.ax.transAxes, fontsize=9,
                         bbox=dict(facecolor='#330000', edgecolor='#FFAA55', alpha=0.7))
        self.ax.set_xlabel(f'Frame index  (total: {len(amps)} frames in {n_sl} slide windows)',
                           color='#cccccc')
        self.ax.set_ylabel(f'max |{self._disp_comp()}|'
                           + (' (log)' if use_log else ''), color='#cccccc')
        self.ax.set_title('Time evolution of the wave amplitude '
                          '(bands = slide windows, top: x start position)',
                          color='#ffffff', fontsize=10)
        self.ax.grid(True, alpha=0.2, which='both')
        leg = self.ax.legend(loc='best', fontsize=8, facecolor='#1a1a1a')
        for t in leg.get_texts():
            t.set_color('#dddddd')
        self._style_axes()

    def _draw_spectrum(self):
        """Fourier-Stitch: Transmission T(lambda) je Frequenz (Kern-Ebene 90 % / 10 %),
        darunter das normierte Quellspektrum |S| als Gewicht."""
        sp = getattr(self, 'spec', None)
        if sp is None:
            self.ax.text(0.5, 0.5, 'No spectrum in this file.\n'
                         'Produced by the pulse stitch: --source-type pulse --method stitch '
                         '(--pulse-stitch fourier).', color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=10)
            self._style_axes()
            return
        lam, T, wgt = sp
        o = np.argsort(lam)
        self.ax.fill_between(lam[o], 0, wgt[o]*max(T.max(), 1e-9), color='#555555', alpha=0.35,
                             label='Source spectrum |S| (normalized)')
        self.ax.plot(lam[o], T[o], 'o-', color='#56C4FF', lw=1.4, ms=4, label='Transmission T(lambda)')
        Tw = float(np.sum(wgt**2*T)/max(np.sum(wgt**2), 1e-30))
        self.ax.axhline(Tw, color='#FF8800', ls='--', lw=1, label=f'spectrally weighted {Tw:.4f}')
        self.ax.set_xlabel('Wavelength (nm)', color='#cccccc')
        self.ax.set_ylabel('T = P_out / P_in (core)', color='#cccccc')
        self.ax.set_title(f'Spectrum from the Fourier stitch ({len(lam)} frequencies)',
                          color='#ffffff', fontsize=10)
        self.ax.grid(True, alpha=0.25)
        leg = self.ax.legend(loc='best', fontsize=8, facecolor='#1a1a1a')
        for t in leg.get_texts():
            t.set_color('#dddddd')
        self._style_axes()

    def _draw_compare(self):
        if len(self.paths) < 2:
            self.ax.text(0.5, 0.5,
                         'Several _frames.npz files required\n'
                         '(e.g. python fdtd_analyzer.py results\\sliding_*_frames.npz)',
                         color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=11)
            self._style_axes()
            return
        # Komponente (Feld-Radio) und Ansicht (einlaufend/eingeschwungen) gelten fuer
        # ALLE Dateien. Frueher: immer das rohe 'Ez'-Array, Beschriftung fest '|Ez|'.
        comp = getattr(self, 'comp', self.prim)
        cw = str(getattr(self, '_view_cur', '')).startswith('Steady state')
        names, maxs, means, notes = [], [], [], []
        for p in self.paths:
            label, mx, mean, note = compare_stats(p, comp, cw)
            names.append(label); maxs.append(mx); means.append(mean); notes.append(note)
        x_pos = np.arange(len(names))
        width = 0.38
        ok = np.array([m is not None for m in maxs])
        maxs_safe = np.array([max(m, 1e-30) if m is not None else np.nan for m in maxs])
        means_safe = np.array([max(m, 1e-30) if m is not None else np.nan for m in means])
        flab = comp if comp == '|E|' else f'|{comp}|'
        self.ax.bar(x_pos[ok] - width/2, maxs_safe[ok], width,
                    color='#FF8800', edgecolor='#888', lw=0.7, label=f'max {flab}')
        self.ax.bar(x_pos[ok] + width/2, means_safe[ok], width,
                    color='#56C4FF', edgecolor='#888', lw=0.7, label=f'mean {flab}')
        self.ax.set_yscale('log')
        top = np.nanmax(maxs_safe) if ok.any() else 1.0
        for k in range(len(names)):
            if notes[k]:                             # Komponente/Ansicht hier nicht vorhanden
                self.ax.text(x_pos[k], top*0.05 if ok.any() else 0.5, notes[k],
                             color='#FFAA55', ha='center', va='center', fontsize=8,
                             rotation=90 if ok[k] else 0)
        # Beschriftung IM Plot (unter den Balken waere sie zu lang und liefe in den
        # Frame-Regler): Nummer als Tick, Name/Polarisation/Ordner senkrecht im Balkenfeld
        self.ax.set_xticks(x_pos)
        self.ax.set_xticklabels([str(k + 1) for k in range(len(names))], color='#cccccc')
        y_lo = self.ax.get_ylim()[0]
        for k, nm in enumerate(names):
            self.ax.text(x_pos[k] - width/2, y_lo*1.15, nm.replace('\n', '  |  '), rotation=90,
                         color='#ffffff', fontsize=7, ha='center', va='bottom', zorder=5,
                         bbox=dict(facecolor='#000000', edgecolor='none', alpha=0.55, pad=1.5))
        self.ax.set_ylabel(f'{flab} (log)', color='#cccccc')
        self.ax.set_title(f'Comparison of the loaded files: max/mean {flab} '
                          f'({"steady state, envelope" if cw else "transient, all snapshots"})',
                          color='#ffffff', fontsize=10)
        self.ax.grid(True, alpha=0.25, axis='y', which='both')
        leg = self.ax.legend(loc='best', fontsize=8, facecolor='#1a1a1a')
        for t in leg.get_texts():
            t.set_color('#dddddd')
        self._style_axes()

    # ------- Diff-Modus: aktuelle Datei minus Referenz -------
    def _load_ref_data(self, path):
        """Laedt (gecached) die Referenz-Datei fuer den Diff-Modus."""
        cache = getattr(self, '_ref_cache', None)
        if cache is None:
            cache = self._ref_cache = {}
        key_p = (path, POWER.get('mw'), POWER.get('depth_um'))
        if key_p in cache:
            return cache[key_p]
        d = np.load(path)
        _sc, _ = power_scale(*_npz_power(d))       # eigene eingekoppelte Leistung der Referenz

        def _S(a):
            if a is None or _sc == 1.0:
                return a
            return np.asarray(a, dtype=np.float32)*np.float32(_sc)
        ez = _S(np.asarray(d['Ez']))
        refd = {'is3d': (ez.ndim == 4), 'ez': ez,
                'ez_cw': _S(np.asarray(d['Ez_cw'])) if 'Ez_cw' in d.files else None,
                'meta_cw': np.asarray(d['meta_cw']) if 'meta_cw' in d.files else None,
                'ex': _S(np.asarray(d['Ex'])) if 'Ex' in d.files else None,
                'ey': _S(np.asarray(d['Ey'])) if 'Ey' in d.files else None}
        d.close()
        cache[key_p] = refd
        return refd

    def _ref_slice(self, refd, frame):
        """2D-Schnitt der Referenz passend zu Ebene/Position/Komponente/Frame."""
        comp = self._disp_comp()
        # Gleiche Ansicht wie die aktuelle Datei: eingeschwungen (CW-Phasen) gegen
        # CW-Phasen der Referenz, einlaufend gegen einlaufend. (Frueher wurde immer der
        # transiente Referenz-Frame abgezogen -> Unsinn in der CW-Ansicht.)
        use_cw = (str(getattr(self, '_view_cur', '')).startswith('Steady state')
                  and refd.get('ez_cw') is not None)
        if use_cw:
            cw = refd['ez_cw']
            if cw.ndim == 3:                               # 2D: (N, Nx, Ny)
                return np.asarray(cw[min(frame, cw.shape[0] - 1)], dtype=np.float32)
            if cw.ndim == 4:                               # 3D-Volumen: (N, Nx, Ny, Nz)
                i = self.slice_idx; f = min(frame, cw.shape[0] - 1)
                sl = (cw[f, :, :, i] if self.plane == 'xy' else
                      cw[f, :, i, :] if self.plane == 'xz' else cw[f, i, :, :])
                return np.asarray(sl, dtype=np.float32)
        if refd['is3d']:
            i = self.slice_idx
            if comp == '|E|' and refd['ex'] is not None:
                vol = np.sqrt(refd['ez'].astype(np.float32)**2
                              + refd['ex'].astype(np.float32)**2
                              + refd['ey'].astype(np.float32)**2)
            elif comp == 'Ex' and refd['ex'] is not None:
                vol = refd['ex']
            elif comp == 'Ey' and refd['ey'] is not None:
                vol = refd['ey']
            else:
                vol = refd['ez']
            if self.plane == 'xy':
                sl = vol[frame, :, :, i]
            elif self.plane == 'xz':
                sl = vol[frame, :, i, :]
            else:
                sl = vol[frame, i, :, :]
            return np.asarray(sl, dtype=np.float32)
        return np.asarray(refd['ez'][frame], dtype=np.float32)

    @staticmethod
    def _phasor(arr):
        """Komplexe CW-Amplitude per DFT ueber die N Phasen-Frames (N, Nx, Nv)."""
        a = np.asarray(arr, dtype=np.float64)
        n = a.shape[0]
        return (2.0/n)*np.tensordot(np.exp(-2j*np.pi*np.arange(n)/n), a, axes=(0, 0))

    def _draw_scatter(self):
        """Streufeld |E_s| = |Ê(Bead) - Ê(Referenz)| aus den eingeschwungenen CW-Saetzen,
        in dB relativ zum Maximum des Referenzfelds. Phasenfrei (keine Wellenfronten-
        Streifen wie im Momentbild-Diff) -> zeigt, wohin der Bead streut."""
        msg = None
        if len(self.paths) < 2:
            msg = 'Scattered field needs 2 files: bead + reference (same run without bead).'
        if msg is None and getattr(self, 'is3d', False):
            self._draw_scatter_3d()
            return
        v = getattr(self, '_views', {}).get('Steady state') if msg is None else None
        if msg is None and not v:
            msg = 'Scattered field: current file has no steady-state CW set.'
        if msg is None:
            refd = self._load_ref_data(self.paths[(self.cur_path + 1) % len(self.paths)])
            if refd.get('ez_cw') is None:
                msg = 'Scattered field: reference file has no CW set (Ez_cw).'
        if msg is None:
            Ea = self._phasor(v[1]); Eb = self._phasor(refd['ez_cw'])
            if Ea.shape != Eb.shape:
                msg = f'Shapes do not match: {Ea.shape} vs reference {Eb.shape}'
        if msg is not None:
            self.ax.text(0.5, 0.5, msg, color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=11)
            self._style_axes(); return
        Es = np.abs(Ea - Eb)
        ref_max = max(float(np.abs(Eb).max()), 1e-30)
        db = 20*np.log10(np.maximum(Es, ref_max*1e-8)/ref_max)
        x_start, x_end, y_start, y_end = np.asarray(v[2])[0, :4]
        top = 20*np.log10(max(float(Es.max()), 1e-30)/ref_max)
        top = top + 20*np.log10(self._sat())            # Kontrast: Skala nach unten
        im = self.ax.imshow(downsample_for_display(db).T,
                            extent=[x_start, x_end, y_start, y_end], origin='lower',
                            cmap=self._cm_seq(), vmin=top - 60, vmax=top,
                            aspect=self.aspect_mode, interpolation='nearest')
        cax = self.ax.inset_axes([0.975, 0.05, 0.012, 0.9])  # im Bild, stiehlt keinen Platz
        cb = self.fig.colorbar(im, cax=cax)
        cax.yaxis.set_ticks_position('left'); cax.yaxis.set_label_position('left')
        cb.set_label('|E_s| / max|E_ref|  [dB]', color='#cccccc')
        cb.ax.tick_params(colors='#cccccc')
        self._scatter_cb = cb
        if getattr(self, '_overlay_ok', True):
            self._layer_lines(self.ax, x_start, x_end, y_start)
        self.ax.set_title(f'Scattered field |E_bead - E_ref| (CW, 60 dB scale)  -  '
                          f'{os.path.basename(self.paths[self.cur_path])}',
                          color='#ffffff', fontsize=10)
        self._style_axes()

    def _scatter_slice_3d(self, cw_vol, plane, i):
        """Slice of every CW phase frame of a volume (N, Nx, Ny, Nz) -> (N, Na, Nb)."""
        if plane == 'xy':
            return np.asarray(cw_vol[:, :, :, i], dtype=np.float32)
        if plane == 'xz':
            return np.asarray(cw_vol[:, :, i, :], dtype=np.float32)
        return np.asarray(cw_vol[:, i, :, :], dtype=np.float32)

    def _ref_cw_slice_3d(self, path, plane, i):
        """CW phase frames of the reference, sliced like the current plane. Streamed
        frame by frame from the npz (the whole reference would be ~0.8 GB per array)
        and cached per (file, plane, index, power)."""
        cache = getattr(self, '_scat_ref_cache', None)
        if cache is None:
            cache = self._scat_ref_cache = {}
        key = (path, plane, int(i), POWER.get('mw'), POWER.get('depth_um'))
        if key in cache:
            return cache[key]
        with np.load(path) as d:
            if 'Ez_cw' not in d.files:
                return None
            _sc, _ = power_scale(*_npz_power(d))
        out = []
        for arr in _npz_stream_frames(path, 'Ez_cw'):
            if arr.ndim != 3:
                raise ValueError('Reference is not a volume file (_vol3d.npz)')
            sl = (arr[:, :, i] if plane == 'xy' else arr[:, i, :] if plane == 'xz' else arr[i, :, :])
            out.append(np.array(sl, dtype=np.float32)*np.float32(_sc))
            del arr
        res = np.stack(out)
        if len(cache) > 6:
            cache.clear()
        cache[key] = res
        return res

    def _draw_scatter_3d(self):
        """Scattered field |E_s| = |E^(bead) - E^(ref)| in the current 3D slice, from the
        steady-state CW phase frames (complex amplitude by DFT over the phases), in dB
        relative to max|E_ref| in that slice. Only the slice is processed -> a few MB."""
        msg = None
        v = getattr(self, '_views', {}).get('Steady state')
        if not v:
            msg = 'Scattered field: current file has no steady-state CW volume (Ez_cw).'
        Ea = Eb = None
        if msg is None:
            try:
                i = self.slice_idx
                rpath = self.paths[(self.cur_path + 1) % len(self.paths)]
                rb = self._ref_cw_slice_3d(rpath, self.plane, i)
                if rb is None:
                    msg = 'Scattered field: reference file has no CW volume (Ez_cw).'
                else:
                    ra = self._scatter_slice_3d(v[1], self.plane, i)
                    if ra.shape[1:] != rb.shape[1:]:
                        msg = f'Shapes do not match: {ra.shape[1:]} vs reference {rb.shape[1:]}'
                    else:
                        Ea = self._phasor(ra)
                        Eb = self._phasor(rb)
            except Exception as e:
                msg = f'Scattered field: {e}'
        if msg is not None:
            self.ax.text(0.5, 0.5, msg, color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=11)
            self._style_axes()
            return
        Es = np.abs(Ea - Eb)
        ref_max = max(float(np.abs(Eb).max()), 1e-30)
        db = 20*np.log10(np.maximum(Es, ref_max*1e-8)/ref_max)
        nf, Nx, Ny, Nz = v[1].shape
        dx = self.dx_um
        if self.plane == 'xy':
            ext = [self.x0_um, self.x0_um + Nx*dx, self.y0_um, self.y0_um + Ny*dx]
        elif self.plane == 'xz':
            ext = [self.x0_um, self.x0_um + Nx*dx, self.z0_um, self.z0_um + Nz*dx]
        else:
            ext = [self.y0_um, self.y0_um + Ny*dx, self.z0_um, self.z0_um + Nz*dx]
        top = 20*np.log10(max(float(Es.max()), 1e-30)/ref_max)
        top = top + 20*np.log10(self._sat())
        im = self.ax.imshow(downsample_for_display(db).T, extent=ext, origin='lower',
                            cmap=self._cm_seq(), vmin=top - 60, vmax=top,
                            aspect=self.aspect_mode, interpolation='nearest')
        cax = self.ax.inset_axes([0.975, 0.05, 0.012, 0.9])
        cb = self.fig.colorbar(im, cax=cax)
        cax.yaxis.set_ticks_position('left')
        cax.yaxis.set_label_position('left')
        cb.set_label('|E_s| / max|E_ref|  [dB]', color='#cccccc')
        cb.ax.tick_params(colors='#cccccc')
        self._scatter_cb = cb
        if self.plane == 'xy' and getattr(self, '_overlay_ok', True):
            self._layer_lines(self.ax, ext[0], ext[1], ext[2])
        if getattr(self, 'is_bead', False) and self.bead_d > 0:
            bc = {'x': self.bead_x, 'y': getattr(self, 'bead_y', -self.bead_d/2),
                  'z': getattr(self, 'bead_z', 0.0)}
            self.ax.add_patch(Circle((bc[self.plane[0]], bc[self.plane[1]]), self.bead_d/2,
                                     fill=False, ec='#39FF14', lw=1.0))
        pos = {'xy': ('z', self.z0_um), 'xz': ('y', self.y0_um), 'yz': ('x', self.x0_um)}[self.plane]
        self.ax.set_title(f'Scattered field |E_bead - E_ref| (CW, 60 dB scale), plane {self.plane} at '
                          f'{pos[0]}={pos[1] + self.slice_idx*dx:.2f} um  -  '
                          f'{os.path.basename(self.paths[self.cur_path])}',
                          color='#ffffff', fontsize=10)
        self._style_axes()

    def _draw_diff(self):
        if len(self.paths) < 2:
            self.ax.text(0.5, 0.5,
                         'Diff needs 2 loaded files (bead + reference):\n'
                         'python fdtd_analyzer.py results\\beads3d_*_vol3d.npz',
                         color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=11)
            self._style_axes(); return
        ref_path = self.paths[(self.cur_path + 1) % len(self.paths)]
        refd = self._load_ref_data(ref_path)
        i = self.frame_idx
        cur = np.asarray(self.ezs[i], dtype=np.float32)
        try:
            ref = self._ref_slice(refd, i)
        except Exception as e:
            self.ax.text(0.5, 0.5, f'Reference slice error: {e}', color='#FF5555',
                         ha='center', va='center', transform=self.ax.transAxes)
            self._style_axes(); return
        if ref.shape != cur.shape:
            self.ax.text(0.5, 0.5,
                         f'Shapes do not match: current {cur.shape} vs ref {ref.shape}\n'
                         '(same file type required: both _vol3d OR both _frames)',
                         color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=10)
            self._style_axes(); return
        diff = cur - ref
        x_start, x_end, y_start, y_end = self.meta[i, :4]
        vmax = float(np.max(np.abs(diff)))
        if vmax < 1e-30:
            vmax = 1.0
        ds = downsample_for_display(diff)
        aspect = self.aspect_mode
        vmax = vmax*self._sat()
        self.ax.imshow(ds.T, extent=[x_start, x_end, y_start, y_end],
                       origin='lower', cmap=self._cm_div(), vmin=-vmax, vmax=vmax,
                       aspect=aspect, interpolation='nearest')
        if getattr(self, '_overlay_ok', True):
            self._layer_lines(self.ax, x_start, x_end, y_start)
            if getattr(self, 'is_bead', False) and self.bead_d > 0 and x_start <= self.bead_x <= x_end:
                _r = self.bead_d/2.0
                self.ax.add_patch(Circle((self.bead_x, getattr(self, 'bead_y', -_r)), _r, fill=False,
                                         ec='#39FF14', lw=1.6, zorder=6))
                self.ax.axvline(self.bead_x, color='#39FF14', lw=0.8, ls='--', alpha=0.7)
        if getattr(self, 'is3d', False):
            self.ax.set_ylim(y_start, y_end)
            xl, yl = self.PLANE_LABELS[self.plane]
            self.ax.set_xlabel(xl, color='#cccccc', fontsize=9)
            self.ax.set_ylabel(yl, color='#cccccc', fontsize=9)
        self._style_axes()
        cn = self._disp_comp()
        # einzeilig: zweizeilige Titel ueberlappten die Kopfzeile (Dateiname steht dort)
        self.ax.set_title(f'Diff {cn} = A - B  (B = {os.path.basename(ref_path)})   '
                          f'Frame {i+1}   max|Delta|={vmax:.3e}', color='#ffffff', fontsize=8)
        if getattr(self, '_overlay_ok', True):
            leg = self.ax.legend(loc='upper left', fontsize=7, framealpha=0.6,
                                 facecolor='#1a1a1a')
            for t in leg.get_texts():
                t.set_color('#dddddd')


def pick_files_dialog(parent=None):
    """Datei-Auswahl. parent: das Tk-Fenster des laufenden Analyzers -> der Dialog nutzt
    DESSEN Tcl-Interpreter. Frueher wurde immer ein zweites tk.Tk() erzeugt und wieder
    zerstoert; zwei Tk-Interpreter im selben Prozess sind unter Windows instabil (der
    Analyzer konnte bei Laden+/Reset wortlos beendet werden)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return []
    initial = os.path.abspath('results') if os.path.isdir('results') else os.getcwd()
    kw = dict(title='Select FDTD NPZ (_vol3d.npz = 3D volume, _frames.npz = 2D)',
              initialdir=initial,
              filetypes=[('All NPZ (volume + frames)', '*.npz'),
                         ('3D volume only', '*_vol3d.npz'),
                         ('2D frames only', '*_frames.npz'),
                         ('All files', '*.*')])
    if parent is not None:
        return list(filedialog.askopenfilenames(parent=parent, **kw))
    root = tk.Tk()                                   # nur beim Start (noch kein Fenster)
    root.withdraw()
    files = filedialog.askopenfilenames(**kw)
    root.destroy()
    return list(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz', nargs='*')
    args = ap.parse_args()
    paths = []
    if args.npz:
        for inp in args.npz:
            ms = glob.glob(inp)
            paths.extend(ms if ms else [inp])
        paths = [p for p in paths if os.path.exists(p)]
    else:
        print('No path given - opening file dialog...')
        paths = pick_files_dialog()
    if not paths:
        print('No files selected - aborting.')
        sys.exit(1)
    print(f'Loading {len(paths)} files:')
    for p in paths:
        print('  ', p)
    _an = Analyzer(paths)          # Referenz halten (Callbacks allein halten es sonst nur ueber die Figur)
    plt.show()


if __name__ == '__main__':
    main()
