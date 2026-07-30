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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider, Button
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib.patches import Polygon as MPoly, Circle, Rectangle

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

MODES = ['Field 2D', 'Field log', 'Int-Mittel', 'Line Cut', 'Evanescent',
         'Resonance', 'Layer Power', 'Moden', 'Sensor',
         'Time Trend', 'Compare', 'Diff (A-B)']


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


class Analyzer:
    def __init__(self, paths):
        self.paths = paths
        self.cur_path = 0
        self.load()
        self.frame_idx = int(np.argmax(self.frame_amps))
        # Auto-Erkennung Planar vs Lens:
        # Planar-Demo hat y-Range ca. 20um (5um WG + Buffer), Lens >100um.
        # Frueher checkten wir x_start>=0, aber Linse hat auch positive x_start
        # bei VCSEL-nahen Slides -> false positive. Y-Range ist eindeutiger.
        y_range_um = float(self.meta[0, 3] - self.meta[0, 2])
        self.planar = y_range_um < 100.0
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
        print(f'  Lade {os.path.basename(path)} ...', end=' ', flush=True)
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
        # Flat/Curved Flag (Full-Domain Flattened)
        self.flat_mode = ('flattened' in d.files and
                          float(np.asarray(d['flattened']).flat[0]) > 0)
        # Tatsaechliche WG-Kerndicke (um) fuer die Overlay-Oberkante; Default PS_TOP.
        # (Frueher fest 5um -> falsch, wenn der Lauf eine andere WG-Dicke nutzte.)
        self.wg_top_um = (float(np.asarray(d['t_wg_um']).flat[0])
                          if 't_wg_um' in d.files else PS_TOP)
        # Bead-Info (nur planar_beads-Dateien): Position/Groesse fuer Marker
        self.is_bead = ('Bead' in self.mat_names)
        if 'bead_diameter_um' in d.files:
            self.bead_d = float(np.asarray(d['bead_diameter_um']).flat[0])
        else:
            self.bead_d = float(self.layers[1])*1e6 if self.is_bead else 0.0
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
                 if 'polarization' in d.files else 's')
        self.pol = 'p' if _polv in ('p', 'tm') else 's'
        # Primaerkomponente = gespeichertes Skalarfeld; Label je nach Polarisation
        self.prim = 'Ey' if self.pol == 'p' else 'Ez'
        self.pol_tag = 'TM/p' if self.pol == 'p' else 'TE/s'
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
            # Vektor-Modus: Ex/Ey zusaetzlich vorhanden -> Komponenten-Auswahl
            self.is_vector = ('Ex' in d.files and 'Ey' in d.files)
            self.vol_ex = np.asarray(d['Ex']) if self.is_vector else None
            self.vol_ey = np.asarray(d['Ey']) if self.is_vector else None
            # Ansichten: einlaufend (transient) / eingeschwungen (CW-Volumen)
            _ez_cw_vol = np.asarray(d['Ez_cw']) if 'Ez_cw' in d.files else None
            _cwonly = bool(int(np.asarray(d['cw_only']).flat[0])) if 'cw_only' in d.files else False
            _hascw = bool(int(np.asarray(d['has_cw']).flat[0])) if 'has_cw' in d.files else False
            _vprim = str(np.asarray(d['view_primary']).flat[0]) if 'view_primary' in d.files else 'transient'
            self.comp = self.prim                   # aktive Komponente (Pol-abhaengig)
            self.plane = 'xy'                       # Laenge-Querschnitt (Standard)
            self.slice_idx = self.vol.shape[3]//2   # Tiefe-Mitte
            d.close()
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
            d.close()
            self._build_views_2d(_Ez, self.meta, _ez_cw, _meta_cw, _cwonly, _hastr, _hascw, _vprim)
            self.frame_amps = compute_frame_amps(self.ezs)
        self.path_label = os.path.basename(path)
        self._set_evan_defaults()      # delta-abhaengiges Fit-Fenster fuer Evanescent
        _kind = '3D-Volumen' if self.is3d else '2D-Frames'
        print(f'{len(self.ezs)} Frames {self.ezs.shape[1]}x{self.ezs.shape[2]} '
              f'[{_kind}] - berechne Statistiken ...', flush=True)
        gc.collect()

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
            self._views['Eingeschwungen'] = ('2d', ez, meta)
            tr, tm = self._make_illustrative_transient(ez, meta)
            self._views['Einlaufend (illustrativ)'] = ('2d', tr, tm)
            self._view_names = ['Eingeschwungen', 'Einlaufend (illustrativ)']
            cur = 'Eingeschwungen'
        else:
            self._views['Einlaufend'] = ('2d', ez, meta); self._view_names = ['Einlaufend']
            if has_cw and ez_cw is not None:
                self._views['Eingeschwungen'] = ('2d', ez_cw, meta_cw)
                self._view_names.append('Eingeschwungen')
            cur = 'Eingeschwungen' if (primary == 'cw' and 'Eingeschwungen' in self._views) else 'Einlaufend'
        self._view_cur = cur
        _, self.ezs, self.meta = self._views[cur]

    def _build_views_3d(self, vol, steps, times, cw_vol, cw_only, has_cw, primary):
        self._views = {}; self._view_names = []
        f0 = 2.99792458e8/(getattr(self, 'lam_nm', 850.0)*1e-9)
        if cw_only:
            self._views['Eingeschwungen'] = ('3d', vol, np.arange(len(vol)),
                                              np.array([(k/len(vol))/f0 for k in range(len(vol))]))
            tv, ts, tt = self._make_illustrative_transient_3d(vol)
            self._views['Einlaufend (illustrativ)'] = ('3d', tv, ts, tt)
            self._view_names = ['Eingeschwungen', 'Einlaufend (illustrativ)']
            cur = 'Eingeschwungen'
        else:
            self._views['Einlaufend'] = ('3d', vol, steps, times)
            self._view_names = ['Einlaufend']
            if has_cw and cw_vol is not None:
                ncw = len(cw_vol)
                self._views['Eingeschwungen'] = ('3d', cw_vol, np.arange(ncw),
                                                  np.array([(k/ncw)/f0 for k in range(ncw)]))
                self._view_names.append('Eingeschwungen')
            cur = 'Eingeschwungen' if (primary == 'cw' and 'Eingeschwungen' in self._views) else 'Einlaufend'
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
            self.b_view.label.set_text(f'Ansicht: {name}')
        try:
            self.redraw()
        except Exception:
            self._safe_redraw()

    def on_view_toggle(self, event=None):
        names = getattr(self, '_view_names', [])
        if len(names) < 2:
            return
        i = (names.index(self._view_cur) + 1) % len(names)
        self._apply_view(names[i])

    # ------- 3D-Volumen: Schnittebenen-Logik -------
    PLANE_LABELS = {'xy': ('Laenge x (um)', 'Querschnitt y (um)'),
                    'xz': ('Laenge x (um)', 'Tiefe z (um)'),
                    'yz': ('Querschnitt y (um)', 'Tiefe z (um)')}

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

        comp = getattr(self, 'comp', 'Ez')
        self._is_magnitude = (comp == '|E|')
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
        # Overlays (Schichtlinien) nur sinnvoll in der xy-Ebene (Querschnitt=y)
        self._overlay_ok = (self.plane == 'xy')
        self.frame_amps = compute_frame_amps(self.ezs)

    def _slice_pos_um(self):
        """Physikalische Position der aktuellen Schnittebene (fixe Achse)."""
        dx = self.dx_um
        if self.plane == 'xy':
            return self.z0_um + self.slice_idx*dx, 'Tiefe z'
        elif self.plane == 'xz':
            return self.y0_um + self.slice_idx*dx, 'Querschnitt y'
        else:
            return self.x0_um + self.slice_idx*dx, 'Laenge x'

    # Slider-Beschriftung je Ebene (physikalische Achse, nicht Roh-Index)
    _SLICE_AXIS_LABEL = {'xy': 'Tiefe z', 'xz': 'Querschnitt y', 'yz': 'Laenge x'}

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
        self.plane = label.split()[0]            # 'xy'/'xz'/'yz' aus Label
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
        new = int(val)
        if new == self.slice_idx:
            return
        self.slice_idx = new
        self._update_slice_valtext()
        self._apply_slice()
        self._schedule_redraw()

    def on_comp(self, label):
        self.comp = label                        # 'Ez'/'Ex'/'Ey'/'|E|'
        self._apply_slice()
        self._safe_redraw()

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
        self.b_load = Button(ax_load, 'Laden+', color='#12301a', hovercolor='#1f5030')
        for _b in (self.b_prev, self.b_next, self.b_load):
            _b.label.set_color('#dddddd'); _b.label.set_fontsize(8)
        self.b_prev.on_clicked(lambda e: self.switch_path(-1))
        self.b_next.on_clicked(lambda e: self.switch_path(+1))
        self.b_load.on_clicked(self.on_load)
        # Tastenkuerzel: l=Laden, a=aspect, ,/.=vor/zurueck
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # 3D-Volumen: Ebenen-Auswahl (links, unter Mode) + Positions-Slider
        # (als vollwertiger Slider direkt UNTER dem Frame-Slider, gs-Zeile 2).
        if getattr(self, 'is3d', False):
            ax_plane = plt.axes([0.008, 0.33, 0.108, 0.14], facecolor='#1a1a1a')
            ax_plane.set_title('Ebene', color='#ffffff', fontsize=8)
            self.radio_plane = RadioButtons(
                ax_plane, ['xy Laenge-Querschnitt', 'xz Laenge-Tiefe',
                           'yz Querschnitt-Tiefe'],
                active=0, activecolor='#39FF14')
            for lbl in self.radio_plane.labels:
                lbl.set_color('#dddddd'); lbl.set_fontsize(7)
            self.radio_plane.on_clicked(self.on_plane)
            ax_slice = self.fig.add_subplot(gs[2, 1:], facecolor='#1a1a1a')
            self.s_slice = Slider(ax_slice, self._SLICE_AXIS_LABEL[self.plane], 0,
                                  self._slice_max_idx(), valinit=self.slice_idx,
                                  valstep=1, color='#39FF14', track_color='#444444')
            self.s_slice.label.set_color('#dddddd')
            self.s_slice.valtext.set_color('#dddddd')
            self.s_slice.on_changed(self.on_slice)
            self._update_slice_valtext()   # Startanzeige in um statt Roh-Index

            # Vektor-Daten: Feldkomponenten-Auswahl Ez/Ex/Ey/|E| (unter Ebene)
            if getattr(self, 'is_vector', False):
                ax_comp = plt.axes([0.008, 0.19, 0.108, 0.12], facecolor='#1a1a1a')
                ax_comp.set_title('Feld', color='#ffffff', fontsize=8)
                _cl = ['Ez', 'Ex', 'Ey', '|E|']
                self.radio_comp = RadioButtons(ax_comp, _cl,
                                               active=(_cl.index(self.prim)
                                                       if self.prim in _cl else 0),
                                               activecolor='#FF8800')
                for lbl in self.radio_comp.labels:
                    lbl.set_color('#dddddd'); lbl.set_fontsize(8)
                self.radio_comp.on_clicked(self.on_comp)

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
        self.b_view = Button(ax_view, f'Ansicht: {getattr(self, "_view_cur", "-")}',
                             color='#22331a', hovercolor='#33501a')
        self.b_view.label.set_color('#bfe0a0'); self.b_view.label.set_fontsize(8)
        self.b_view.on_clicked(self.on_view_toggle)

        # Drehbare 3D-Ansicht (nur bei Volumen-Dateien sinnvoll, Taste '3')
        ax_v3d = plt.axes([0.212, 0.965, 0.085, 0.028], facecolor='#1a2233')
        self.b_view3d = Button(ax_v3d, '3D-Ansicht', color='#1a2233',
                               hovercolor='#2a3a55')
        self.b_view3d.label.set_color('#9ecbff')
        self.b_view3d.label.set_fontsize(8)
        self.b_view3d.on_clicked(self.on_view3d)

        # Reset: geladene Datensaetze VERWERFEN und neu waehlen (Taste 'r').
        # (Laden+ haengt an, Reset ersetzt -> sauberes neues A/B-Paar.)
        ax_reset = plt.axes([0.302, 0.965, 0.070, 0.028], facecolor='#301616')
        self.b_reset = Button(ax_reset, 'Reset', color='#301616',
                              hovercolor='#5a2626')
        self.b_reset.label.set_color('#ffb3b3')
        self.b_reset.label.set_fontsize(8)
        self.b_reset.on_clicked(self.on_reset)

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
            new = pick_files_dialog()
        except Exception as e:
            print(f'[Laden] Dialog nicht verfuegbar ({e}). '
                  f'Dateien stattdessen beim Start als Argumente uebergeben.')
            return
        added = 0
        for p in new:
            if p and os.path.exists(p) and p not in self.paths:
                self.paths.append(p)
                added += 1
        if not added:
            print('[Laden] nichts Neues hinzugefuegt.')
            return
        print(f'[Laden] {added} Datei(en) dazu -> {len(self.paths)} geladen.')
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

    def on_reset(self, event=None):
        """Verwirft ALLE geladenen Datensaetze und laedt eine neue Auswahl
        (ersetzt statt anzuhaengen) - fuer ein sauberes neues A/B-Paar."""
        try:
            new = pick_files_dialog()
        except Exception as e:
            print(f'[Reset] Dialog nicht verfuegbar ({e}).')
            return
        new = [p for p in new if p and os.path.exists(p)]
        if not new:
            print('[Reset] Keine Dateien gewaehlt - nichts geaendert.')
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
        print(f'[Reset] {len(self.paths)} Datei(en) neu geladen: '
              f'{[os.path.basename(p) for p in self.paths]}')

    def _on_key(self, event):
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

    def _layer_planes3d(self, ax, x0, x1, z0, z1):
        """Zeichnet die Material-Schicht-Grenzen (WG-Top/Bot, Aqueous, Mucin,
        Cornea) als transparente horizontale Ebenen im 3D-Raum zur Orientierung."""
        try:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except Exception:
            return
        t_lip = self.layers[0]*1e6
        t_aq = self.layers[1]*1e6
        t_mu = self.layers[2]*1e6
        t_wg = getattr(self, 't_wg_um', 0.0) or 5.0
        nm = getattr(self, 'mat_names', ['WG', 'Lipid', 'Aqueous', 'Mucin', 'Cornea'])
        levels = [(t_wg, '#FFD75E', f'{nm[0]}-Top'),
                  (0.0, '#FFD75E', f'{nm[0]}-Bot')]
        y = 0.0
        if t_lip > 1e-3:
            y -= t_lip; levels.append((y, '#88AAFF', nm[1]))
        y -= t_aq; levels.append((y, '#56C4FF', f'{nm[2]} ({t_aq:.1f}um)'))
        y -= t_mu; levels.append((y, '#D88AFF', f'{nm[3]} -> {nm[4]}'))
        for yv, col, lab in levels:
            quad = [[(x0, yv, z0), (x1, yv, z0), (x1, yv, z1), (x0, yv, z1)]]
            pc = Poly3DCollection(quad, alpha=0.10)
            pc.set_facecolor(col); pc.set_edgecolor(col)
            ax.add_collection3d(pc)
            ax.text(x1, yv, z1, ' '+lab, color=col, fontsize=7)

    def on_view3d(self, event=None):
        """Oeffnet ein SEPARATES, mit der Maus drehbares 3D-Fenster mit dem
        aktuellen Feld-Volumen (Punktwolke ueber Schwelle) + WG-Box + Bead."""
        if not getattr(self, 'is3d', False):
            print('[3D-Ansicht] Nur fuer Volumen-Dateien (_vol3d.npz) - '
                  'aktuell sind 2D-Frames geladen.')
            return
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception as e:
            print(f'[3D-Ansicht] 3D-Backend nicht verfuegbar: {e}')
            return
        from matplotlib.widgets import Slider as _Slider
        fi = self.frame_idx
        comp = getattr(self, 'comp', 'Ez')
        vez = self.vol[fi].astype(np.float32)
        if comp == '|E|' and getattr(self, 'is_vector', False):
            vol = np.sqrt(vez**2 + self.vol_ex[fi].astype(np.float32)**2
                          + self.vol_ey[fi].astype(np.float32)**2); signed = False
        elif comp == 'Ex' and getattr(self, 'is_vector', False):
            vol = self.vol_ex[fi].astype(np.float32); signed = True
        elif comp == 'Ey' and getattr(self, 'is_vector', False):
            vol = self.vol_ey[fi].astype(np.float32); signed = True
        else:
            vol = vez; signed = True
            comp = 'Ez' if getattr(self, 'is_vector', False) else self.prim
        Nx, Ny, Nz = vol.shape
        dx = self.dx_um
        vmax = float(np.abs(vol).max()) or 1.0
        stride = max(1, int(round((Nx*Ny*Nz/1.5e6)**(1/3))))
        sub = vol[::stride, ::stride, ::stride]
        absub = np.abs(sub)
        lx = self.x0_um + Nx*dx
        t_wg = getattr(self, 't_wg_um', 0.0) or 5.0
        fig = plt.figure(figsize=(10, 8), facecolor='#0a0a0a')
        ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a0a')

        def _render(frac):
            lvl = max(float(frac), 1e-4)*vmax
            ii, jj, kk = np.where(absub >= lvl)
            if ii.size > 120000:        # gleichmaessig ausduennen -> behaelt AUCH
                sel = np.linspace(0, ii.size-1, 120000).astype(int)  # schwache Bereiche
                ii, jj, kk = ii[sel], jj[sel], kk[sel]
            xs = self.x0_um + ii*stride*dx
            ys = self.y0_um + jj*stride*dx
            zs = self.z0_um + kk*stride*dx
            vals = sub[ii, jj, kk]
            ax.clear()
            ax.set_facecolor('#0a0a0a')
            sc = dict(c=vals, cmap=('RdBu_r' if signed else 'inferno'),
                      s=6, alpha=0.35, linewidths=0)
            if signed:
                sc.update(vmin=-vmax*0.6, vmax=vmax*0.6)
            ax.scatter(xs, ys, zs, **sc)
            self._wire_box3d(ax, self.x0_um, lx, 0.0, t_wg,
                             self.z0_um, self.z0_um + Nz*dx, '#FFD75E')
            self._layer_planes3d(ax, self.x0_um, lx,
                                 self.z0_um, self.z0_um + Nz*dx)
            if getattr(self, 'is_bead', False) and self.bead_d > 0:
                self._wire_sphere3d(ax, self.bead_x, -self.bead_d/2, 0.0,
                                    self.bead_d/2, '#39FF14')
            ax.set_xlabel('Laenge x (um)', color='#ccc')
            ax.set_ylabel('Querschnitt y (um)', color='#ccc')
            ax.set_zlabel('Tiefe z (um)', color='#ccc')
            ax.tick_params(colors='#aaa')
            try:
                ax.set_box_aspect((Nx*dx, Ny*dx, Nz*dx))
            except Exception:
                pass
            ax.set_title(f'{self.scenario} - {comp} - Frame {fi+1}\n'
                         f'(drehbar; |Feld|>={lvl:.3g}, {int(ii.size)} Punkte)',
                         color='#fff', fontsize=10)
            fig.canvas.draw_idle()

        _render(0.30)
        ax_thr = fig.add_axes([0.25, 0.02, 0.5, 0.03], facecolor='#1a1a1a')
        self._v3d_slider = _Slider(ax_thr, 'Schwelle (x max)', 0.02, 0.80,
                                   valinit=0.30, valstep=0.01,
                                   color='#39FF14', track_color='#444444')
        self._v3d_slider.label.set_color('#dddddd')
        self._v3d_slider.valtext.set_color('#dddddd')
        self._v3d_slider.on_changed(_render)
        self._v3d_fig = fig      # Referenz halten (sonst GC)
        print('[3D-Ansicht] geoeffnet - Schwelle unten im Fenster regelbar, '
              'mit Maus drehbar.')
        try:
            fig.show()
        except Exception:
            plt.show()

    def _style_axes(self):
        """Konsistenter Dark-Mode Look fuer alle Plots."""
        for sp in self.ax.spines.values():
            sp.set_color('#444444')
        self.ax.tick_params(colors='#cccccc')

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
        self.load()
        self.frame_idx = min(self.frame_idx, len(self.ezs)-1)
        self.s_frame.valmax = len(self.ezs)-1
        self.s_frame.ax.set_xlim(0, len(self.ezs)-1)
        self.s_frame.set_val(self.frame_idx)
        self._update_x_slider_range()   # x-Cursor an neuen Daten-x-Bereich anpassen
        self.redraw()

    def redraw(self):
        # Sichere Achsen-Bereinigung: clear + alte Colorbars/Legenden entfernen
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
        _needs_x = ('Moden',)   # brauchen x auf der Horizontalen -> xy oder xz
        _bad = (getattr(self, 'is3d', False) and getattr(self, 'plane', 'xy') != 'xy'
                and self.mode in _1d) or \
               (getattr(self, 'is3d', False) and getattr(self, 'plane', 'xy') == 'yz'
                and self.mode in _needs_x)
        if _bad:
            _ok_planes = "'xy'" if self.mode in _1d else "'xy' oder 'xz'"
            self.ax.text(0.5, 0.5,
                         f"'{self.mode}' ist nur in Ebene {_ok_planes} sinnvoll\n"
                         f"(Propagation x horizontal / Schichtstapel y vertikal).\n\n"
                         f"Aktuelle Ebene: {self.plane}.  ->  Ebene passend stellen,\n"
                         f"oder einen Bild-Modus (Field 2D/log, Int-Mittel) nutzen.",
                         color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=11,
                         bbox=dict(facecolor='#1a1a1a', edgecolor='#FFAA55', alpha=0.85))
            self._style_axes()
            self._set_header(f'{self.scenario}  |  Mode: {self.mode} '
                             f'(Ebene {self.plane})')
            self.fig.canvas.draw_idle()
            return
        try:
            if self.mode == 'Field 2D':
                self._draw_field(log=False)
            elif self.mode == 'Field log':
                self._draw_field(log=True)
            elif self.mode == 'Int-Mittel':
                self._draw_intavg()
            elif self.mode == 'Line Cut':
                self._draw_line_cut()
            elif self.mode == 'Evanescent':
                self._draw_evanescent()
            elif self.mode == 'Resonance':
                self._draw_resonance()
            elif self.mode == 'Layer Power':
                self._draw_layer_power()
            elif self.mode == 'Moden':
                self._draw_modes()
            elif self.mode == 'Sensor':
                self._draw_sensor()
            elif self.mode == 'Time Trend':
                self._draw_time_trend()
            elif self.mode == 'Compare':
                self._draw_compare()
            elif self.mode == 'Diff (A-B)':
                self._draw_diff()
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                self.ax.text(0.5, 0.5, f'Fehler: {e}', color='#FF5555',
                             ha='center', va='center', transform=self.ax.transAxes)
            except Exception:
                pass
        info = f'[{self.cur_path+1}/{len(self.paths)}] ' if len(self.paths) > 1 else ''
        self._set_header(f'{info}{self.scenario}  |  Mode: {self.mode}')
        self.fig.canvas.draw_idle()

    def _set_header(self, text):
        """Titel linksbuendig RECHTS neben der oberen Button-Leiste platzieren
        (Buttons enden bei x~0.37), damit er nicht mit Bedienelementen ueberlappt."""
        _pt = getattr(self, 'pol_tag', None)
        if _pt:
            text = f'{text}   [{_pt}]'
        self.fig.suptitle(text, color='#ffffff', fontsize=9,
                          x=0.40, y=0.978, ha='left')

    # ---- Hilfen fuer Moden-/Sensor-Analyse (2D + 3D-Ebene) ----
    def _t_wg_eff(self):
        """WG-Kerndicke in um (3D: aus Datei; 2D-Planar: PS_TOP-PS_BOT)."""
        return getattr(self, 't_wg_um', 0.0) or getattr(self, 'wg_top_um', PS_TOP - PS_BOT)

    def _delta_min(self):
        """Theoretische MINIMALE evaneszente Eindringtiefe delta_min (um) =
        lam/(2 pi sqrt(n_core^2 - n_clad^2)) (grazing incidence)."""
        try:
            n_hi = float(self.mat_indices[0])
            n_lo = float(self.mat_indices[2])
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
        v = getattr(self, '_views', {}).get('Eingeschwungen') if hasattr(self, '_views') else None
        if not v or v[0] != '2d':
            return None
        arr = np.asarray(v[1])
        meta = np.asarray(v[2])
        if arr.ndim != 3 or arr.shape[0] < 2 or meta.shape[0] < 1:
            return None
        if not np.allclose(meta[:, :4], meta[0, :4]):
            return None
        return arr, meta[0]

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
        _src = self._ez_signed() if getattr(self, 'comp', 'Ez') == '|E|' else self.ezs
        fi = min(fi, len(_src)-1)
        ez = _src[fi].astype(np.float64)              # (Nx, Nv)
        x0, x1, y0, y1 = self.meta[fi, :4]
        Nx, Nv = ez.shape
        if Nx < 8 or (x1 - x0) <= 0:
            self.ax.text(0.5, 0.5, 'zu wenig x-Punkte fuer FFT', color='#FFAA55',
                         ha='center', va='center', transform=self.ax.transAxes)
            self._style_axes()
            self._set_header(f'{self.scenario}  |  Mode: Moden')
            return
        dxx = (x1 - x0)*1e-6/Nx
        if (not getattr(self, 'is3d', False)) or self.plane == 'xy':
            t_wg = self._t_wg_eff()
            v_lo = self._v_index(0.0, y0, y1, Nv)
            v_hi = self._v_index(t_wg, y0, y1, Nv)
            vlabel = f'Kern y=0..{t_wg:g}um'
        else:                                          # xz: vertikal = Tiefe z
            v_lo, v_hi = int(Nv*0.35), int(Nv*0.65)
            vlabel = 'Tiefen-Mitte'
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
        n_aq = float(self.mat_indices[2])
        lo = max(1.0, n_aq) + 0.002
        hi = n_core
        guided = find_neff_peaks(neff, power, lo, hi)
        leaky = find_neff_peaks(neff, power, 1.0, lo)
        m = (neff >= 0.9) & (neff <= n_core + 0.08)
        pmax = float(power[m].max()) if power[m].size and power[m].max() > 0 else 1.0
        self.ax.plot(neff[m], power[m]/pmax, color='#4da6ff', lw=1.0)
        self.ax.axvspan(lo, hi, color='#2e8b57', alpha=0.12,
                        label=f'gefuehrt ({lo:.3f}-{hi:.3f})')
        for n, _ in guided:
            self.ax.axvline(n, color='#ff5b5b', ls='--', lw=0.8)
            self.ax.text(n, 1.02, f'{n:.3f}', color='#ff8a8a', fontsize=7,
                         rotation=90, va='bottom', ha='center')
        self.ax.set_xlim(0.9, n_core + 0.08)
        self.ax.set_ylim(0, 1.18)
        self.ax.set_xlabel('n_eff = k_x / k0', color='#cccccc', fontsize=9)
        self.ax.set_ylabel('spektrale Leistung (norm.)', color='#cccccc', fontsize=9)
        _pl = f' | Ebene {self.plane}' if getattr(self, 'is3d', False) else ''
        L_um = (x1 - x0)
        dneff = (self.lam_nm/1000.0)/L_um if L_um > 0 else 9.9   # = lam0/L
        self.ax.set_title(
            f'Modenspektrum ({vlabel}, Frame {fi+1}, {self.lam_nm:g}nm) - '
            f'{len(guided)} gefuehrte Moden' + (f', {len(leaky)} leaky' if leaky else '')
            + _pl, color='#ffffff', fontsize=10)
        # Aufloesungshinweis: n_eff-Rasterung = lam0/L. Bei kurzer Strecke zu grob.
        band = max(hi - lo, 1e-6)
        if dneff > band/5.0:
            self.ax.text(0.5, 0.90,
                         f'Aufloesung grob: dn_eff={dneff:.3f} (Raster ~ lam0/L). '
                         f'Fuer klare Moden laengere x-Strecke (>=50 um) rechnen.',
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
        # Tear-Band aus dem Schichtstapel (Aqueous unter dem Kern), nicht hartkodiert.
        try:
            t_aq_um = float(self.layers[1])*1e6
        except Exception:
            t_aq_um = 2.0
        tear_depth = max(0.5, min(t_aq_um, 8.0))

        # Datenquelle: eingeschwungener Phasensatz (zeitgemittelt) bevorzugt.
        ps = self._cw_phase_stack()
        Ehat = None
        if ps is not None and np.allclose(ps[1][:4], self.meta[0, :4]):
            arr = ps[0].astype(np.float64)              # (nph, Nx, Nv), eine Periode
            intens = np.mean(arr**2, axis=0)            # <E^2> zeitgemittelt
            q = arr.shape[0]//4 or 1
            Ehat = arr[0] - 1j*arr[q]                   # komplexe Amplitude
            src = f'CW-Amplitude ({arr.shape[0]} Phasen, zeitgemittelt)'
        else:
            N = min(6, len(self.ezs))
            intens = np.mean(self.ezs[-N:].astype(np.float64)**2, axis=0)
            src = f'{N} Transient-Frames (kein CW-Satz vorhanden)'
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

        P_guid = bandI(intens, dl, dh, iy_wg_lo, iy_wg_hi)
        P_guid_in = bandI(intens, il, ih, iy_wg_lo, iy_wg_hi)
        P_tear = bandI(intens, dl, dh, iy_tear_lo, iy_wg_lo)
        P_air = bandI(intens, dl, dh, iy_wg_hi, iy_air_hi)
        K = P_tear/P_guid if P_guid > 0 else 0.0
        atten = P_guid/P_guid_in if P_guid_in > 0 else 0.0

        # Gerichteter Poynting-Fluss (Konstante egal fuer das Verhaeltnis).
        trans_S = float('nan')
        if Ehat is not None:
            Sx = np.imag(Ehat*np.conj(np.gradient(Ehat, axis=0)))
            S_det = bandI(Sx, dl, dh, iy_wg_lo, iy_wg_hi)
            S_in = bandI(Sx, il, ih, iy_wg_lo, iy_wg_hi)
            trans_S = S_det/S_in if abs(S_in) > 1e-30 else float('nan')

        xd = x0 + 0.8*(x1 - x0)
        xin = x0 + 0.2*(x1 - x0)

        self.ax.axis('off')
        lines = [
            f'SENSOR-METRIK   {self.scenario}',
            f'Quelle: {src}',
            '',
            f'Detektor x = {xd:6.2f} um     Eingang x = {xin:6.2f} um   (je 10% Laenge)',
            f'Kern y = 0..{t_wg:g} um   Tear y = -{tear_depth:g}..0 um (Aqueous)   '
            f'Luft y = {t_wg:g}..{y1:.1f} um',
            '',
            f'I_guided (Kern,  Detektor) :  {P_guid:.3e}',
            f'I_tear   (Tear,  Detektor) :  {P_tear:.3e}',
            f'I_air    (Luft,  Detektor) :  {P_air:.3e}',
            '',
            f'K = I_tear / I_guided      :  {K:.4e}   (evaneszenter Ueberlapp = Sensor-Signal)',
            f'Transmission (Intensitaet) :  {atten:.3f}   (I_Det / I_Eingang, Kern)',
        ]
        if np.isfinite(trans_S):
            lines.append(f'Transmission (Poynting Sx) :  {trans_S:.3f}   '
                         f'(echte gerichtete Leistung)')
        self.ax.text(0.03, 0.97, '\n'.join(lines), color='#e6e6e6', fontsize=10.5,
                     family='monospace', va='top', ha='left',
                     transform=self.ax.transAxes)
        # kleine Balken (auf max normiert) fuer die drei Intensitaeten
        mx = max(P_guid, P_tear, P_air, 1e-30)
        for i, (v, l, c) in enumerate([(P_guid, 'Kern', '#FFD75E'),
                                       (P_tear, 'Tear', '#56C4FF'),
                                       (P_air, 'Luft', '#999999')]):
            yb = 0.28 - i*0.08
            w = 0.55*(v/mx)
            self.ax.add_patch(Rectangle((0.03, yb), w, 0.05, transform=self.ax.transAxes,
                                        facecolor=c, edgecolor='none'))
            self.ax.text(0.03 + w + 0.01, yb + 0.025, l, color='#bbbbbb',
                         fontsize=8, va='center', transform=self.ax.transAxes)
        _pl = f'  (Ebene {self.plane})' if getattr(self, 'is3d', False) else ''
        self.ax.set_title(f'Sensor-Auswertung{_pl}', color='#ffffff', fontsize=10)

    def _layer_lines(self, ax, x_start, x_end, y_start):
        t_lip = self.layers[0]*1e6
        t_aq = self.layers[1]*1e6
        t_mu = self.layers[2]*1e6
        # Material-Namen (aus NPZ wenn vorhanden, sonst Defaults)
        n_lens, n_lip, n_aq, n_mu, n_cor = (
            self.mat_names[0], self.mat_names[1], self.mat_names[2],
            self.mat_names[3], self.mat_names[4])
        i_lens, i_lip, i_aq, i_mu, i_cor = (
            self.mat_indices[0], self.mat_indices[1], self.mat_indices[2],
            self.mat_indices[3], self.mat_indices[4])
        if self.planar or self.flat_mode:
            # Planar/Flattened: gerade WG bei y=0..PS_TOP (planar) bzw. y_rel=+-T/2 (flat)
            if self.planar:
                y_wg_top = self.wg_top_um
                y_wg_bot = PS_BOT
            else:
                # Flattened Lens: y_rel=0 = Linsenmitte, body bei +-T_LENS/2
                y_wg_top = T_LENS*1e6/2
                y_wg_bot = -T_LENS*1e6/2
            y_lipid_bot = y_wg_bot - t_lip
            y_aq_bot = y_lipid_bot - t_aq
            y_mu_bot = y_aq_bot - t_mu
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
            if t_lip > 1e-3:
                ax.axhline(y_lipid_bot, color='#88AAFF', ls=':', lw=0.9,
                           label=f'{n_lip} {t_lip:.3f}um (n={i_lip:.3f})')
            ax.axhline(y_aq_bot, color='#56C4FF', ls='--', lw=0.9,
                       label=f'{n_aq} {t_aq:.2f}um (n={i_aq:.3f})')
            ax.axhline(y_mu_bot, color='#D88AFF', ls='--', lw=0.9,
                       label=f'{n_mu} {t_mu:.2f}um (n={i_mu:.3f})')
            ax.axhspan(y_start, y_mu_bot, color='#553030', alpha=0.18,
                       label=f'{n_cor} (n={i_cor:.3f})')
        else:
            # Lens: gekruemmte Geometrie
            xs_out = np.linspace(x_start, x_end, 200)
            xs_clip = np.clip(xs_out, -R_BEND*1e6 + 1, R_BEND*1e6 - 1)
            sag = (R_BEND - np.sqrt(R_BEND**2 - (xs_clip*1e-6)**2))*1e6
            y_mid_o = -sag
            y_lens_top = y_mid_o + T_LENS*1e6/2
            y_lens_bot = y_mid_o - T_LENS*1e6/2
            y_lipid_bot = y_lens_bot - t_lip
            y_aq_bot = y_lipid_bot - t_aq
            y_mu_bot = y_aq_bot - t_mu
            ax.plot(xs_out, y_lens_top, '-', color='#FFD75E', lw=1.0, alpha=0.9,
                    label=f'{n_lens} (n={i_lens:.3f})')
            ax.plot(xs_out, y_lens_bot, '-', color='#FFD75E', lw=1.0, alpha=0.9)
            ax.plot(xs_out, y_lipid_bot, ':', color='#88AAFF', lw=0.9,
                    label=f'{n_lip} {t_lip:.3f}um (n={i_lip:.3f})')
            ax.plot(xs_out, y_aq_bot, '--', color='#56C4FF', lw=0.9,
                    label=f'{n_aq} {t_aq:.2f}um (n={i_aq:.3f})')
            ax.plot(xs_out, y_mu_bot, '--', color='#D88AFF', lw=0.9,
                    label=f'{n_mu} {t_mu:.2f}um (n={i_mu:.3f})')
            ax.fill_between(xs_out, y_start, y_mu_bot, color='#553030', alpha=0.18,
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
                           label=f'Einkoppelabstand {gap:.2f}um (Luft)')
                ax.axvline(xsrc, color='#00E5FF', lw=0.8, ls=':', alpha=0.7)
            ax.axvline(xws, color='#00E5FF', lw=1.3, ls='-.', alpha=0.95,
                       label=f'WG-Eintritt x={xws:.2f}um')
        ef = getattr(self, 'end_facet', 0.0)
        if ef and ef > 0:
            x_ef = x_end - ef
            if x_start <= x_ef <= x_end:
                ax.axvline(x_ef, color='#00E5FF', lw=1.3, ls='-.', alpha=0.8,
                           label=f'WG-Ende x={x_ef:.2f}um')

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
        ax.text(cx, cy + rc, f' Bead-Schnitt d={2*rc:.2f}um', color='#FFD400',
                fontsize=8, ha='center', va='bottom', zorder=7)

    def _layer_marks(self, y_top, y_bot):
        """Datenabhaengige Schicht-Markierungen [(y, farbe, label)] aus den
        TATSAECHLICH berechneten Material-Namen/Indizes. Ueberspringt Schichten
        mit Dicke ~0 (z.B. Lipid beim Bead-Setup)."""
        t_lip = self.layers[0]*1e6; t_aq = self.layers[1]*1e6; t_mu = self.layers[2]*1e6
        nm = self.mat_names; ix = self.mat_indices
        marks = [(y_top, '#FFD75E', f'{nm[0]}-Top (n={ix[0]:.3f})'),
                 (y_bot, '#FFD75E', f'{nm[0]}-Unterseite (y=0)')]
        y = y_bot
        if t_lip > 1e-3:
            y -= t_lip
            marks.append((y, '#88AAFF', f'{nm[1]} {t_lip:.3f}um (n={ix[1]:.3f})'))
        y -= t_aq
        marks.append((y, '#56C4FF', f'{nm[2]} {t_aq:.2f}um (n={ix[2]:.3f})'))
        y -= t_mu
        marks.append((y, '#D88AFF', f'{nm[3]} {t_mu:.2f}um (n={ix[3]:.3f}) -> {nm[4]} darunter'))
        return marks

    def _draw_intavg(self):
        """Zeit-gemittelte Intensitaet <Ez^2> ueber die Steady-State-Frames
        (letzte Haelfte) -> Envelope ohne Traeger-Streifen (log-Skala)."""
        nf = len(self.ezs)
        k = max(1, nf // 2)
        Iavg = np.mean(self.ezs[-k:].astype(np.float64)**2, axis=0)
        x_start, x_end, y_start, y_end = self.meta[0, :4]
        Ishow = downsample_for_display(Iavg)
        vmax = float(Ishow.max())
        if vmax <= 0:
            vmax = 1.0
        vmin = vmax * 1e-4
        Ishow = np.maximum(Ishow, vmin * 1e-2)
        aspect = self.aspect_mode
        self.ax.imshow(Ishow.T, extent=[x_start, x_end, y_start, y_end],
                       origin='lower', cmap='inferno',
                       norm=LogNorm(vmin=vmin, vmax=vmax), aspect=aspect,
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
        t_lip = self.layers[0]*1e6; t_aq = self.layers[1]*1e6; t_mu = self.layers[2]*1e6
        if getattr(self, 'is3d', False):
            self.ax.set_ylim(y_start, y_end)
            xl, yl = self.PLANE_LABELS[self.plane]
            self.ax.set_xlabel(xl, color='#cccccc', fontsize=9)
            self.ax.set_ylabel(yl, color='#cccccc', fontsize=9)
        elif self.planar:
            self.ax.set_ylim(max(y_start, -(t_lip+t_aq+t_mu)-2), min(y_end, self.wg_top_um+2))
        self._style_axes()
        _pl = f' | Ebene {self.plane}' if getattr(self, 'is3d', False) else ''
        self.ax.set_title(f'Zeit-gemittelte Intensitaet <{self.prim}^2> (letzte {k} Frames, log){_pl}',
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
                               origin='lower', cmap='inferno', norm=norm,
                               aspect=aspect, interpolation='nearest')
            else:
                self.ax.imshow(ez_show.T, extent=[x_start, x_end, y_start, y_end],
                               origin='lower', cmap='inferno', vmin=0, vmax=vmax_g,
                               aspect=aspect, interpolation='nearest')
        elif log:
            lt = max(vmax_g*1e-4, 1e-30)
            norm = SymLogNorm(linthresh=lt, vmin=-vmax_g, vmax=vmax_g)
            self.ax.imshow(ez_show.T, extent=[x_start, x_end, y_start, y_end],
                           origin='lower', cmap='RdBu_r', norm=norm,
                           aspect=aspect, interpolation='nearest')
        else:
            vm = vmax_g*0.5
            self.ax.imshow(ez_show.T, extent=[x_start, x_end, y_start, y_end],
                           origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm,
                           aspect=aspect, interpolation='nearest')
        if getattr(self, '_overlay_ok', True):
            self._layer_lines(self.ax, x_start, x_end, y_start)
            if (not getattr(self, 'is3d', False)) and getattr(self, 'is_bead', False) \
                    and self.bead_d > 0 and x_start <= self.bead_x <= x_end:
                _r = self.bead_d/2.0
                self.ax.axvline(self.bead_x, color='#FFD400', lw=1.0, ls='--', alpha=0.85)
                self.ax.add_patch(Circle((self.bead_x, -_r), _r, fill=False, ec='#FFD400', lw=1.6, zorder=6))
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
        t_lip = self.layers[0]*1e6
        t_aq = self.layers[1]*1e6
        t_mu = self.layers[2]*1e6
        if getattr(self, 'is3d', False):
            # 3D-Schnitt: volle Extents zeigen (Achsen je nach Ebene, s.u.)
            self.ax.set_ylim(y_start, y_end)
        elif self.planar:
            y_tear_bot = -(t_lip + t_aq + t_mu) - 2
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
            tear_total = t_lip + t_aq + t_mu
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
                info += f"  t = {t_fs:.3f} fs   (CW-Phase {ph:.0f}° von 360°)"
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
            _cn = getattr(self, 'comp', 'Ez')
            _cl = f'|E|=sqrt(Ex2+Ey2+Ez2)' if _cn == '|E|' else f'{_cn}-Komponente'
            info = (f"{_cl} | Ebene {self.plane} | "
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
            y_lens_bot = PS_BOT if self.planar else 0.0
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
                msg = (f'x = {self.x_um:.0f} um NICHT in aktuellem Frame {i+1}\n'
                       f'(Window x = [{x_start:.0f}, {x_end:.0f}] um).\n'
                       f'-> Setze Frame-Slider auf Frame {j+1}\n'
                       f'   ODER waehle ein x im aktuellen Window-Bereich.')
            else:
                msg = (f'x = {self.x_um:.0f} um liegt in keinem Slide-Window.\n'
                       f'Aktueller Frame {i+1}: x = [{x_start:.0f}, {x_end:.0f}] um')
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
        if self.planar:
            y_lens_top = self.wg_top_um
            y_lens_bot = PS_BOT
        else:
            sag = R_BEND - np.sqrt(R_BEND**2 - (self.x_um*1e-6)**2)
            y_lens_mid = -sag*1e6
            y_lens_top = y_lens_mid + T_LENS*1e6/2
            y_lens_bot = y_lens_mid - T_LENS*1e6/2
        y_lipid_bot = y_lens_bot - self.layers[0]*1e6
        y_aq_bot = y_lipid_bot - self.layers[1]*1e6
        y_mu_bot = y_aq_bot - self.layers[2]*1e6
        # Bei planar Demo enger zoomen (Schichten sind nur ~5 um) als bei Lens
        if self.planar:
            zoom_top = y_lens_top + 2
            zoom_bot = y_mu_bot - 2
            top_label = 'Polystyrol-Top'
            bot_label = 'Polystyrol-Bot'
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
        self.ax.set_xlabel('y (um)  — Zoom auf Lens + Tear-Region', color='#cccccc')
        self.ax.set_ylabel('Ez', color='#cccccc')
        self.ax.grid(True, alpha=0.2)
        for y_line, color, label in self._layer_marks(y_lens_top, y_lens_bot):
            self.ax.axvline(y_line, color=color, ls='--', lw=1, alpha=0.7,
                            label=label)
        self._style_axes()
        self.ax.set_title(
            f'Linien-Cut bei x={self.x_um:.0f} um  -  Frame {i+1}/{len(self.ezs)}  '
            f'Slide {int(self.meta[i,4])+1}', color='#ffffff', fontsize=10)
        leg = self.ax.legend(loc='upper right', fontsize=7, facecolor='#1a1a1a')
        for t in leg.get_texts():
            t.set_color('#dddddd')

    def _draw_evanescent(self):
        if self.planar:
            # Demo: PS-WG-Bottom fest bei y=0
            y_lens_bot_um = PS_BOT
        else:
            if not (-7000 < self.x_um < 7000):
                self.ax.text(0.5, 0.5, 'x ausserhalb Lens',
                             color='#FFAA55', ha='center', va='center',
                             transform=self.ax.transAxes)
                return
            sag = R_BEND - np.sqrt(R_BEND**2 - (self.x_um*1e-6)**2)
            y_lens_bot_um = (-sag - T_LENS/2)*1e6
        # Anzuzeigendes Frame folgt dem FRAME-SLIDER; faellt auf ein x-abdeckendes
        # Frame zurueck, falls das aktuelle Frame x / die WG-Unterkante nicht enthaelt
        # (fuer Stitch-Daten, wo Frames unterschiedliche x-Fenster abdecken).
        covering = [k for k in range(len(self.ezs))
                    if self.meta[k, 0] <= self.x_um <= self.meta[k, 1]
                    and self.meta[k, 2] < y_lens_bot_um < self.meta[k, 3]]
        if not covering:
            self.ax.text(0.5, 0.5,
                         f'Kein Frame mit x={self.x_um:.0f} und WG-Unterkante im Bereich',
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
        fit_amp = abs_ez
        self._evan_fit_src = f'Momentaufnahme (Frame {i+1})'
        _ps = self._cw_phase_stack()
        if _ps is not None:
            _arr, _m0 = _ps
            if _arr.shape[1] == Nx and np.allclose(_m0[:4], self.meta[i, :4]):
                _env = np.sqrt(2.0*np.mean(_arr[:, ix, :].astype(np.float64)**2, axis=0))
                if float(_env.max()) > 0:
                    fit_amp = _env
                    self._evan_fit_src = 'CW-Amplitude (knotenfrei)'

        # Auto-Zoom auf Lens/WG + Tear-Region
        y_aq_bot_z = y_lens_bot_um - self.layers[0]*1e6 - self.layers[1]*1e6
        y_mu_bot_z = y_aq_bot_z - self.layers[2]*1e6
        if self.planar:
            # WG-Top an der tatsaechlichen Kerndicke (nicht fest 5um)
            zoom_top = self.wg_top_um + 2
            zoom_bot = y_mu_bot_z - 2
            top_marker_y = self.wg_top_um
            top_marker_label = 'WG-Top'
        else:
            zoom_top = y_lens_bot_um + T_LENS*1e6 + 30
            zoom_bot = y_mu_bot_z - 30
            top_marker_y = y_lens_bot_um + T_LENS*1e6
            top_marker_label = 'Lens-Top'

        # Daten ueberhaupt aussagekraeftig?
        if max_amp < 1e-15:
            self.ax.text(0.5, 0.5,
                         f'NPZ leer / Mur-absorbiert\n'
                         f'max|{self.prim}| an x={self.x_um:.0f} um = {max_amp:.1e}\n\n'
                         f'Kein Evanescent-Fit moeglich.\n'
                         f'-> Multi-Snapshot-Lauf erforderlich.',
                         color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=10,
                         bbox=dict(facecolor='#330000', edgecolor='#FFAA55', alpha=0.7))
            self._style_axes()
            return

        # NaN/Negativ-Schutz fuer semilogy (sonst matplotlib UserWarning)
        floor = max(max_amp*1e-6, 1e-30)
        ez_safe = np.maximum(np.nan_to_num(abs_ez, nan=floor, posinf=floor,
                                            neginf=floor), floor)
        _comp = getattr(self, 'comp', 'Ez')
        flab = _comp if _comp == '|E|' else f'|{_comp}|'
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
        n_low = float(self.layers[3])   # n_aqueous
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
                    decay_label = 'NaN [slope>=0, Welle waechst]'
                    angle_label = 'NaN [kein Decay]'
                else:
                    decay_um = -1.0/slope
                    decay_label = f'{decay_um:.3f} um'
                    if decay_um < d_min_um*0.5:
                        angle_label = 'NaN [d < d_min, unphysikalisch]'
                    else:
                        k = lam_um / (2*np.pi*decay_um)
                        sin2_th = (n_low**2 + k**2) / n_high**2
                        if sin2_th > 1:
                            angle_label = f'NaN [sin²θ>1: d zu klein]'
                        elif sin2_th < (n_low/n_high)**2:
                            # Unter dem kritischen Winkel
                            theta = float(np.degrees(np.arcsin(np.sqrt(max(sin2_th, 0)))))
                            angle_label = f'{theta:.1f}° [<θ_c=Sub-Critical]'
                        else:
                            theta = float(np.degrees(np.arcsin(np.sqrt(sin2_th))))
                            theta_c = float(np.degrees(np.arcsin(n_low/n_high)))
                            status = 'TIR' if theta > theta_c else 'kritisch'
                            angle_label = f'{theta:.1f}° [{status}, θ_c={theta_c:.1f}°]'
                # Plot
                y_plot = np.linspace(y_fit_s.min(), y_fit_s.max(), 30)
                self.ax.semilogy(y_lens_bot_um - y_plot,
                                 np.exp(intercept + slope*y_plot),
                                 'r-', lw=2.0, alpha=0.95,
                                 label=(f'Eindringtiefe δ = {decay_label}\n'
                                        f'(theor. Minimum δ_min = {d_min_um:.3f} um)\n'
                                        f'R²={r2:.3f}   θ ≈ {angle_label} [Strahlmodell]\n'
                                        f'Fit: {self._evan_fit_src}'))
                # Fit-Bereich markieren
                self.ax.axvspan(y_lens_bot_um - y_fit_s.max(),
                                y_lens_bot_um - y_fit_s.min(),
                                color='#FF4400', alpha=0.15, zorder=0)
        for y_line, color, label in self._layer_marks(top_marker_y, y_lens_bot_um):
            self.ax.axvline(y_line, color=color, ls='--', lw=1.2, alpha=0.8,
                            label=label)
        self.ax.set_xlabel('y (um) — Zoom auf Lens + Tear-Region',
                           color='#cccccc')
        self.ax.set_ylabel(f'{flab} (log)', color='#cccccc')
        self.ax.set_title(f'Evanescent-Decay bei x={self.x_um:.0f} um  '
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
                msg = f'x={self.x_um:.0f}um nicht in Frame {i+1}. -> Frame {j+1}'
            else:
                msg = f'x={self.x_um:.0f}um in keinem Frame'
            self.ax.text(0.5, 0.5, msg, color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=11)
            self._style_axes()
            return
        Nx, Ny = self.ezs[i].shape
        ix = int(round((self.x_um - x_start)/(x_end - x_start)*Nx))
        ix = max(0, min(Nx-1, ix))
        ez_col = self.ezs[i, ix, :]
        intensity = ez_col*ez_col  # |Ez|^2 = Intensitaet
        y_um = np.linspace(y_start, y_end, Ny)
        # Schicht-Grenzen
        t_lip = self.layers[0]*1e6
        t_aq = self.layers[1]*1e6
        t_mu = self.layers[2]*1e6
        if self.planar:
            y_wg_top = PS_TOP
            y_wg_bot = PS_BOT
        else:
            sag = R_BEND - np.sqrt(R_BEND**2 - (self.x_um*1e-6)**2)
            y_mid = -sag*1e6
            y_wg_top = y_mid + T_LENS*1e6/2
            y_wg_bot = y_mid - T_LENS*1e6/2
        y_lipid_bot = y_wg_bot - t_lip
        y_aq_bot = y_lipid_bot - t_aq
        y_mu_bot = y_aq_bot - t_mu
        # Plot
        intensity_safe = np.maximum(intensity, 1e-30)
        _comp = getattr(self, 'comp', 'Ez')
        _fl = _comp if _comp == '|E|' else f'|{_comp}|'
        self.ax.semilogy(y_um, intensity_safe, 'cyan', lw=1.0,
                         label=f'{_fl}^2 (Intensitaet)')
        # Schicht-Bereiche faerben
        ymax_y = float(intensity_safe.max())*5
        ymin_y = max(float(intensity_safe.min()), ymax_y*1e-10)
        self.ax.axvspan(y_wg_bot, y_wg_top, color='#FFD75E', alpha=0.10,
                        label=f'{self.mat_names[0]}')
        if t_lip > 0:
            self.ax.axvspan(y_lipid_bot, y_wg_bot, color='#88AAFF', alpha=0.18,
                            label=f'{self.mat_names[1]}')
        if t_aq > 0:
            self.ax.axvspan(y_aq_bot, y_lipid_bot, color='#56C4FF', alpha=0.15,
                            label=f'{self.mat_names[2]}')
        if t_mu > 0:
            self.ax.axvspan(y_mu_bot, y_aq_bot, color='#D88AFF', alpha=0.15,
                            label=f'{self.mat_names[3]}')
        self.ax.axvspan(y_start, y_mu_bot, color='#553030', alpha=0.18,
                        label=f'{self.mat_names[4]}')
        # Schicht-integrierte Intensitaeten als Annotation
        dy = float(y_um[1] - y_um[0])
        def integ(y_lo, y_hi):
            m = (y_um >= y_lo) & (y_um <= y_hi)
            return float(intensity[m].sum())*dy if m.any() else 0.0
        I_lens = integ(y_wg_bot, y_wg_top)
        I_lip = integ(y_lipid_bot, y_wg_bot) if t_lip > 0 else 0
        I_aq = integ(y_aq_bot, y_lipid_bot) if t_aq > 0 else 0
        I_mu = integ(y_mu_bot, y_aq_bot) if t_mu > 0 else 0
        I_cor = integ(y_start, y_mu_bot)
        annot = (f'Integriert |{self.prim}|^2 pro Schicht:\n'
                 f'  {self.mat_names[0]}: {I_lens:.2e}\n'
                 f'  {self.mat_names[1]}: {I_lip:.2e}\n'
                 f'  {self.mat_names[2]}: {I_aq:.2e}\n'
                 f'  {self.mat_names[3]}: {I_mu:.2e}\n'
                 f'  {self.mat_names[4]}: {I_cor:.2e}')
        self.ax.text(0.02, 0.98, annot, color='#dddddd', va='top', ha='left',
                     transform=self.ax.transAxes, fontsize=8,
                     bbox=dict(facecolor='#1a1a1a', edgecolor='#444', alpha=0.85))
        # Zoom auf interessante Region
        zoom_bot = y_mu_bot - 1
        zoom_top = y_wg_top + 1
        self.ax.set_xlim(zoom_bot, zoom_top)
        self.ax.set_ylim(ymin_y, ymax_y)
        from matplotlib.ticker import LogLocator, NullFormatter
        self.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
        self.ax.yaxis.set_minor_locator(LogLocator(base=10,
                                                                                      subs=[2.0, 5.0],
                                                    numticks=30))
        self.ax.yaxis.set_minor_formatter(NullFormatter())
        self.ax.set_xlabel('y (um)', color='#cccccc')
        self.ax.set_ylabel(f'{_fl}^2 (log)', color='#cccccc')
        self.ax.set_title(f'Resonance-Profil bei x={self.x_um:.0f}um  '
                          f'(Frame {self.frame_idx+1}, Slide {int(self.meta[self.frame_idx,4])+1})',
                          color='#ffffff', fontsize=10)
        self.ax.grid(True, alpha=0.25, which='both')
        leg = self.ax.legend(loc='upper right', fontsize=7, facecolor='#1a1a1a')
        for t in leg.get_texts():
            t.set_color('#dddddd')
        self._style_axes()

    def _draw_layer_power(self):
        """Bar-Chart: integrierte |Ez|^2 pro Schicht."""
        layer_names = self.mat_names if hasattr(self, 'mat_names') else \
                      ['Lens', 'Lipid', 'Aqueous', 'Mucin', 'Cornea']
        layer_colors = ['#FFD75E', '#88AAFF', '#56C4FF', '#D88AFF', '#553030']
        i = -1
        x_start, x_end, y_start, y_end = self.meta[i, :4]
        Nx, Ny = self.ezs[i].shape
        y_um = np.linspace(y_start, y_end, Ny)
        intensity = self.ezs[i]*self.ezs[i]
        t_lip = self.layers[0]*1e6
        t_aq = self.layers[1]*1e6
        t_mu = self.layers[2]*1e6
        if self.planar or self.flat_mode:
            y_wg_top, y_wg_bot = PS_TOP, PS_BOT
        else:
            x_c = (x_start + x_end)/2
            x_m = max(-R_BEND + 1e-9, min(R_BEND - 1e-9, x_c*1e-6))
            sag = R_BEND - np.sqrt(R_BEND**2 - x_m**2)
            y_mid = -sag*1e6
            y_wg_top = y_mid + T_LENS*1e6/2
            y_wg_bot = y_mid - T_LENS*1e6/2
        y_lipid_bot = y_wg_bot - t_lip
        y_aq_bot = y_lipid_bot - t_aq
        y_mu_bot = y_aq_bot - t_mu
        def integ_layer(y_lo, y_hi):
            m = (y_um >= y_lo) & (y_um <= y_hi)
            if not m.any():
                return 0.0
            return float(intensity[:, m].sum())
        I_lens = integ_layer(y_wg_bot, y_wg_top)
        I_lip = integ_layer(y_lipid_bot, y_wg_bot) if t_lip > 0 else 0
        # Bead-Dateien: 'Bead'-Balken = LOKALES Integral ueber die Bead-Scheibe
        # (x=Bead_x +- r, y 0..-d) statt des leeren Lipid-Slots.
        if getattr(self, 'is_bead', False) and self.bead_d > 0:
            _r = self.bead_d/2.0
            _xu = np.linspace(x_start, x_end, Nx)
            _xm = (_xu >= self.bead_x - _r) & (_xu <= self.bead_x + _r)
            _ym = (y_um >= -self.bead_d) & (y_um <= 0.0)
            I_lip = float(intensity[np.ix_(_xm, _ym)].sum()) if (_xm.any() and _ym.any()) else 0.0
        I_aq = integ_layer(y_aq_bot, y_lipid_bot) if t_aq > 0 else 0
        I_mu = integ_layer(y_mu_bot, y_aq_bot) if t_mu > 0 else 0
        I_cor = integ_layer(y_start, y_mu_bot)
        I_all = [I_lens, I_lip, I_aq, I_mu, I_cor]
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
        self.ax.set_xticklabels(layer_names, color='#cccccc')
        self.ax.set_ylabel('Integrierte |Ez|^2 (log)', color='#cccccc')
        _bt = '  (Bead = lokales Integral um x=%.0fum)' % getattr(self,'bead_x',500.0) if getattr(self,'is_bead',False) else ''
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
                color='#FF8800', lw=1, ms=4, label='max |Ez| pro Frame', zorder=3)
        per_sl = np.array([amps_clean[slides == s].mean() if (slides == s).any() else 1e-30
                           for s in range(n_sl)])
        per_sl_safe = np.maximum(np.nan_to_num(per_sl, nan=1e-30), 1e-30)
        slide_x = [np.where(slides == s)[0].mean() for s in range(n_sl)]
        plot_fn(slide_x, per_sl_safe, 's-', color='#56C4FF', lw=2, ms=6,
                label='Mittel pro Slide', zorder=4)
        # Slide-Grenz-Linien (vertikal, gestrichelt)
        for _, x0, x1 in slide_ranges[:-1]:
            self.ax.axvline(x1, color='#FFD75E', lw=0.6, ls='--', alpha=0.5, zorder=1)
        self.ax.axvline(self.frame_idx, color='#ffffff', lw=1.2, ls=':', alpha=0.8,
                        label=f'Frame {self.frame_idx+1}', zorder=5)
        if not use_log:
            self.ax.text(0.5, 0.95,
                         f'WARNUNG: alle |Ez| < 1e-15 (NPZ leer oder Mur-absorbiert)\n'
                         f'-> Multi-Snapshot-Lauf empfohlen (neue sliding_window_fdtd.py)',
                         color='#FFAA55', ha='center', va='top',
                         transform=self.ax.transAxes, fontsize=9,
                         bbox=dict(facecolor='#330000', edgecolor='#FFAA55', alpha=0.7))
        self.ax.set_xlabel(f'Frame-Index  (gesamt: {len(amps)} Frames in {n_sl} Slide-Fenstern)',
                           color='#cccccc')
        self.ax.set_ylabel('max |Ez|' + (' (log)' if use_log else ''), color='#cccccc')
        self.ax.set_title('Zeitliche Entwicklung der Wellen-Amplitude '
                          '(Bands = Slide-Fenster, oben: x-Startposition)',
                          color='#ffffff', fontsize=10)
        self.ax.grid(True, alpha=0.2, which='both')
        leg = self.ax.legend(loc='best', fontsize=8, facecolor='#1a1a1a')
        for t in leg.get_texts():
            t.set_color('#dddddd')
        self._style_axes()

    def _draw_compare(self):
        if len(self.paths) < 2:
            self.ax.text(0.5, 0.5,
                         'Mehrere _frames.npz-Dateien noetig\n'
                         '(z.B. python fdtd_analyzer.py results\\sliding_*_frames.npz)',
                         color='#FFAA55', ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=11)
            self._style_axes()
            return
        names = []
        maxs = []
        means = []
        for p in self.paths:
            scenario, mx, mean = compute_npz_stats(p)
            names.append(scenario)
            maxs.append(mx)
            means.append(mean)
        x_pos = np.arange(len(names))
        width = 0.38
        maxs_safe = np.maximum(maxs, 1e-30)
        means_safe = np.maximum(means, 1e-30)
        self.ax.bar(x_pos - width/2, maxs_safe, width,
                    color='#FF8800', edgecolor='#888', lw=0.7, label='max |Ez|')
        self.ax.bar(x_pos + width/2, means_safe, width,
                    color='#56C4FF', edgecolor='#888', lw=0.7, label='mean |Ez|')
        self.ax.set_yscale('log')
        self.ax.set_xticks(x_pos)
        self.ax.set_xticklabels(names, color='#cccccc', rotation=20, ha='right')
        self.ax.set_ylabel('|Ez| (log)', color='#cccccc')
        self.ax.set_title('Vergleich der geladenen Szenarien (max/mean |Ez|)',
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
        if path in cache:
            return cache[path]
        d = np.load(path)
        ez = np.asarray(d['Ez'])
        refd = {'is3d': (ez.ndim == 4), 'ez': ez,
                'ex': np.asarray(d['Ex']) if 'Ex' in d.files else None,
                'ey': np.asarray(d['Ey']) if 'Ey' in d.files else None}
        d.close()
        cache[path] = refd
        return refd

    def _ref_slice(self, refd, frame):
        """2D-Schnitt der Referenz passend zu Ebene/Position/Komponente/Frame."""
        comp = getattr(self, 'comp', 'Ez')
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

    def _draw_diff(self):
        if len(self.paths) < 2:
            self.ax.text(0.5, 0.5,
                         'Diff braucht 2 geladene Dateien (Bead + Referenz):\n'
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
            self.ax.text(0.5, 0.5, f'Referenz-Slice-Fehler: {e}', color='#FF5555',
                         ha='center', va='center', transform=self.ax.transAxes)
            self._style_axes(); return
        if ref.shape != cur.shape:
            self.ax.text(0.5, 0.5,
                         f'Formen passen nicht: aktuell {cur.shape} vs Ref {ref.shape}\n'
                         '(gleicher Dateityp noetig: beide _vol3d ODER beide _frames)',
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
        self.ax.imshow(ds.T, extent=[x_start, x_end, y_start, y_end],
                       origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       aspect=aspect, interpolation='nearest')
        if getattr(self, '_overlay_ok', True):
            self._layer_lines(self.ax, x_start, x_end, y_start)
            if getattr(self, 'is_bead', False) and self.bead_d > 0 and x_start <= self.bead_x <= x_end:
                _r = self.bead_d/2.0
                self.ax.add_patch(Circle((self.bead_x, -_r), _r, fill=False,
                                         ec='#39FF14', lw=1.6, zorder=6))
                self.ax.axvline(self.bead_x, color='#39FF14', lw=0.8, ls='--', alpha=0.7)
        if getattr(self, 'is3d', False):
            self.ax.set_ylim(y_start, y_end)
            xl, yl = self.PLANE_LABELS[self.plane]
            self.ax.set_xlabel(xl, color='#cccccc', fontsize=9)
            self.ax.set_ylabel(yl, color='#cccccc', fontsize=9)
        self._style_axes()
        cn = getattr(self, 'comp', 'Ez')
        self.ax.set_title(f'Diff {cn}:  {self.path_label}  MINUS  {os.path.basename(ref_path)}\n'
                          f'Frame {i+1}   max|Delta|={vmax:.3e}  '
                          f'(Streufeld am Bead)', color='#ffffff', fontsize=9)
        if getattr(self, '_overlay_ok', True):
            leg = self.ax.legend(loc='upper left', fontsize=7, framealpha=0.6,
                                 facecolor='#1a1a1a')
            for t in leg.get_texts():
                t.set_color('#dddddd')


def pick_files_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return []
    root = tk.Tk()
    root.withdraw()
    initial = os.path.abspath('results') if os.path.isdir('results') else os.getcwd()
    files = filedialog.askopenfilenames(
        title='FDTD NPZ auswaehlen (_vol3d.npz = 3D-Volumen, _frames.npz = 2D)',
        initialdir=initial,
        filetypes=[('Alle NPZ (Volumen + Frames)', '*.npz'),
                   ('Nur 3D-Volumen', '*_vol3d.npz'),
                   ('Nur 2D-Frames', '*_frames.npz'),
                   ('Alle Dateien', '*.*')])
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
        print('Kein Pfad angegeben - oeffne Datei-Auswahl...')
        paths = pick_files_dialog()
    if not paths:
        print('Keine Dateien ausgewaehlt - Abbruch.')
        sys.exit(1)
    print(f'Lade {len(paths)} Dateien:')
    for p in paths:
        print('  ', p)
    Analyzer(paths)
    plt.show()


if __name__ == '__main__':
    main()
