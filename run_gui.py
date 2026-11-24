"""Grafischer Einstieg (tkinter) fuer run_simulation.py.

Das GUI erzeugt seine Eingabefelder AUTOMATISCH aus run_simulation.build_parser()
- jede aktuelle und kuenftige CLI-Option ist damit ohne doppelte Pflege abgedeckt.
Aus den ausgefuellten Feldern wird der Argument-Vektor gebaut und run_simulation.py
als Subprozess gestartet; dessen Ausgabe laeuft live ins Log-Fenster.

Start:  python run_gui.py     (oder run_GUI.bat unter Windows)

Die eigentliche Rechnung passiert im Subprozess - ein Absturz/OOM des Solvers
laesst das GUI am Leben. Die Geometrie-/Methoden-Logik (z.B. grosse Linse -> nur
Stitch) steckt in run_simulation.py; die entsprechenden Hinweise erscheinen im Log.
"""
import os
import sys
import glob
import math
import queue
import threading
import subprocess

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import run_simulation as rs
from common.physics import n_at                      # noqa: E402  (Default-Brechzahlen fuer Hinweise)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Zweites Werkzeug: Moden-Propagation (Kontaktlinse). Liegt in kontaktlinse/ und
# nutzt "from config import ...", daher dessen Ordner auf den Pfad legen und dann
# importieren. Faellt es aus, bleibt das GUI ein reines FDTD-Frontend.
_KL_DIR = os.path.join(_REPO_ROOT, 'kontaktlinse')
if _KL_DIR not in sys.path:
    sys.path.insert(0, _KL_DIR)
try:
    import mode_propagation as mp
except Exception:
    mp = None

# Kurzhinweise fuer Felder ohne festen Default: diese werden erst in
# run_simulation.resolve_args aus Geometrie/Dimension bzw. Materialdispersion
# aufgeloest. Als grauer Platzhalter im Feld sichtbar (nicht mitgesendet).
_AUTO_HINTS = {
    'resolution_nm': 'auto: 20 nm (2D) / 50 nm (3D)',
    'pulse_tmax_fs': 'auto: pulse + transit time (Fourier stitch)',
    'pulse_nfreq': 'auto: from bandwidth and time window',
    'length_um': 'auto: planar 1000 / lens 14000 (2D), planar 40 / lens 60 (3D) um',
    'window_um': 'auto: 350 (2D) / 20 (3D)',
    'slide_um': 'auto: 150 (2D) / 12 (3D)',
    'boundary': 'empty = cpml (2D and 3D)',
    'tear': 'auto: lipid+aqueous+mucin + ~1.5um cornea + absorber',
    # Material-Overrides: Default = Brechzahl @850 nm (leer lassen -> dieser Wert)
    # Default-Brechzahlen bei 850 nm aus common.physics (Sellmeier-Fits mit Quellen)
    'wg_n': f"auto: PMMA {n_at('pmma', 850):.4f} @850nm (dispersion model)",
    'bead_n': f"auto: polystyrene {n_at('polystyrol', 850):.4f} @850nm",
    'n_aqueous': f"auto: tears {n_at('aqueous', 850):.4f} @850nm",
    'n_mucin': f"auto: {n_at('mucin', 850):.4f} @850nm (tears + 1% mucin, estimate)",
    'n_cornea': f"auto: {n_at('cornea', 850):.4f} @850nm",
    'n_lipid': f"auto: {n_at('lipid', 850):.4f} @850nm",
    'kappa_wg': 'empty = 0 (no absorption)', 'kappa_lipid': 'empty = 0 (no absorption)',
    'kappa_aqueous': 'empty = 0 (no absorption)', 'kappa_mucin': 'empty = 0 (no absorption)',
    'kappa_cornea': 'empty = 0 (no absorption)', 'kappa_bead': 'empty = 0 (no absorption)',
    'bead_x': 'auto: center (length/2)', 'bead_y_um': 'auto: at WG bottom edge',
    'wg_width': 'empty = slab (infinitely wide)',
    # Moden-Propagation
    'L_mm': 'auto: from lens geometry (~12 mm)',
    't_lipid_nm': 'empty = scenario value (Gesund=30)',
    'scenario': 'empty = all scenarios',
    'layer_names': 'auto: default layer names (lens: tear-film layers / otherwise generic)',
}


def _auto_hint(dest):
    return _AUTO_HINTS.get(dest, 'auto (solver default)')


# Optionen, die im GUI NICHT als Feld erscheinen (bleiben per CLI nutzbar).
# no_bead: intern fuer den automatischen Referenzlauf; im GUI genuegt der eine
# Haken 'no_reference' (Referenzlauf abschalten) - Default rechnet beides.
_GUI_HIDE = {'no_bead', 'layer_names', 'meep_cad_eps'}   # layer_names: eigene Namensfelder je Schicht

# Auswahlwerte, die im GUI NICHT angeboten werden (bleiben per CLI nutzbar):
# mur1 ist eine Experten-Option (nur Grundmode/senkrechter Einfall exakt).
_GUI_CHOICE_HIDE = {'boundary': {'mur1'}}

# Anzeige-Titel der argparse-Gruppen (Englisch). Bereits englische Titel bleiben
# unveraendert (_GROUP_EN.get(title, title)).
_GROUP_EN = {
    # run_simulation.py
    'Fall-Auswahl': 'Case', 'Grundeinstellungen': 'Basic settings',
    'Material & Schichten': 'Material & layers',
    'CAD-Geometrie (--geometry cad)': 'CAD geometry (--geometry cad)',
    'Bead (nur planar)': 'Bead (planar only)', 'Quelle (VCSEL)': 'Source (VCSEL)',
    '3D-spezifisch': '3D specific', 'Stitch / Meep': 'Stitch / Meep',
    'Ablauf': 'Run control', 'Kalibrierung & Vorschau': 'Calibration & preview',
    # kontaktlinse/mode_propagation.py
    'Szenario & Physik': 'Scenario & physics', 'Einkopplung (VCSEL)': 'Coupling (VCSEL)',
    'Traenenfilm & ATR': 'Tear film & ATR', 'Numerik': 'Numerics',
}


# Klartext-Beschriftungen (mit Einheit) statt roher CLI-Flags. Das rohe Flag und
# der Hilfetext erscheinen im Tooltip. Nicht gelistete Flags werden aus dem
# Flag-Namen abgeleitet (_pretty).
_LABELS = {
    # Fall-Auswahl
    'engine': 'Compute engine', 'geometry': 'Geometry', 'dim': 'Dimension',
    'method': 'Method', 'scenario': 'Dry-eye scenario (lens only)',
    # Grundeinstellungen
    'length_um': 'Propagation length (um)', 'resolution_nm': 'Grid resolution dx (nm)',
    'lambda_nm': 'Wavelength lambda (nm)',
    'polarization': 'Polarization: s=TE (Ez) / p=TM (Ey)',
    'boundary': 'Absorbing boundary (empty = cpml; 3D also sponge)',
    'source_type': 'Source time profile', 'snapshots': 'Saved frames (count)',
    # Material & Schichten
    'layer_names': 'Layer names (core,L1,L2,L3,substrate)',
    'cornea_full': 'Full cornea thickness (to aqueous humour; expensive)',
    'wg_material': 'Waveguide material', 'wg_thickness_um': 'Waveguide thickness (um)',
    'wg_n': 'Waveguide refractive index (override)', 't_aqueous': 'Aqueous thickness (um)',
    't_mucin': 'Mucin thickness (um)', 't_lipid': 'Lipid thickness (um)',
    'n_aqueous': 'Aqueous refractive index (override)', 'n_mucin': 'Mucin refractive index (override)',
    'n_cornea': 'Cornea refractive index (override)', 'n_lipid': 'Lipid refractive index (override)',
    # Bead
    'bead_material': 'Bead material', 'bead_diameter': 'Bead diameter (um)',
    'bead_x': 'Bead position x (um)', 'bead_y_um': 'Bead position y (um)',
    'bead_n': 'Bead refractive index (override)', 'no_bead': 'Without bead (reference run)',
    'kappa_bead': 'Bead absorption kappa (n = n + i*kappa)',
    'no_reference': 'Do NOT compute reference (bead run only)',
    # Quelle (VCSEL)
    'vcsel_waist': 'Beam waist y (um)', 'vcsel_waist_z': 'Beam waist z (um, 3D only)',
    'vcsel_tilt': 'Beam tilt (deg)', 'source_core_only': 'Source only in the core (no direct light into the tear film)', 'vcsel_offset': 'Spot offset y (um)',
    'vcsel_offset_z': 'Spot offset z (um, 3D only)',
    'input_gap': 'Coupling gap laser->WG (um)',
    # 3D-spezifisch
    'lz_um': 'TOTAL domain depth z (+/-lz/2 around z=0)',
    'air': 'Air above WG (um, 3D only)',
    'tear': 'Tear film+cornea below WG (um, 3D only)',
    'wg_width': 'Waveguide channel width in z (empty=slab/infinite)',
    'wg_clad_n': 'Lateral cladding n (3D only)',
    'steps_factor': 'Time-step factor (3D full only)',
    'vol_dtype': '3D volume storage format', 'pec_faces': 'PEC mirror faces (3D full only)',
    'end_facet': 'Air zone at WG end (um, 3D full only)',
    'save_vector': 'Save Ex,Ey,Ez (3D full only)',
    # Stitch / Meep
    'window_um': 'Stitch window width (um)', 'slide_um': 'Stitch step size (um)',
    'meep_modes': 'Meep: guided modes in handoff', 'meep_handoff': 'Meep: handoff method',
    'meep_np': 'Meep: MPI processes (CPU cores)',
    # Ablauf
    'allow_large': 'Bypass RAM/cell-count guard', 'no_save': 'Do not save results',
    # Kalibrierung & Vorschau
    'dry_run': 'Show configuration only (dry run)',
    'check_resources': 'Check VRAM/RAM before run', 'calibrate': 'Calibrate: time N steps only',
    # Moden-Propagation (Kontaktlinse)
    'r_bend_mm': 'Bend radius R (mm; 0=straight, lens=8.3)', 'offset_um': 'VCSEL offset y (um)',
    'waist_um': 'VCSEL waist (um)', 'couple': 'Coupling method',
    'offset_scan': 'Offset scan (um list, e.g. -120,-60,0)',
    't_lipid_nm': 'Lipid thickness (nm; empty=scenario)', 'alpha_tear_permm': 'Tear absorption (1/mm)',
    'dy_nm': 'Transverse resolution dy (nm)', 'L_mm': 'Propagation length (mm; empty=auto)',
    'validate': 'Validate eigensolver (and exit)',
}


def _pretty(dest, flag):
    """Klartext-Beschriftung fuer ein Feld (mit Fallback aus dem Flag-Namen)."""
    if dest in _LABELS:
        return _LABELS[dest]
    return flag.lstrip('-').replace('-', ' ').capitalize()


# ---------------------------------------------------------------------------
# Reine Logik (ohne tkinter -> headless testbar): Werte-Dict -> argv-Liste.
# ---------------------------------------------------------------------------
def collect_argv(parser, values, force_dry_run=False):
    """Baut die Argumentliste fuer run_simulation.py aus {dest: wert}.
    - Flags (store_true): nur wenn wert True -> "--flag".
    - Choice/Value: leer ODER == Default -> weglassen (Default gilt), sonst
      "--flag" "wert".
    force_dry_run haengt --dry-run an, falls nicht ohnehin gesetzt."""
    argv = []
    for act in parser._actions:
        if not act.option_strings or act.dest == 'help':
            continue
        flag = act.option_strings[0]
        val = values.get(act.dest)
        if act.nargs == 0:                      # store_true / Flag
            if bool(val):
                argv.append(flag)
            continue
        s = '' if val is None else str(val).strip()
        if s == '':
            continue                            # leer -> Default des Solvers
        if act.default is not None and s == str(act.default):
            continue                            # unveraendert -> Default weglassen
        argv += [flag, s]
    if force_dry_run and '--dry-run' not in argv:
        argv.append('--dry-run')
    return argv


# ---------------------------------------------------------------------------
# Mini-Tooltip fuer die Hilfetexte der Optionen.
# ---------------------------------------------------------------------------
class _Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind('<Enter>', self._show)
        widget.bind('<Leave>', self._hide)

    def _show(self, _e=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f'+{x}+{y}')
        tk.Label(self.tip, text=self.text, justify='left', wraplength=380,
                 background='#ffffe0', relief='solid', borderwidth=1,
                 font=('Segoe UI', 8)).pack()

    def _hide(self, _e=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class SimGUI:
    def __init__(self, root):
        self.root = root
        # Werkzeuge: key -> (Label, Parser, Skript-relpfad). FDTD immer; Moden nur
        # wenn importierbar.
        self.tools = {'fdtd': ('FDTD simulation', rs.build_parser(), 'run_simulation.py')}
        if mp is not None:
            self.tools['modes'] = ('Mode propagation (contact lens)', mp.build_parser(),
                                   os.path.join('kontaktlinse', 'mode_propagation.py'))
        self.tool = 'fdtd'
        self.parser = self.tools['fdtd'][1]
        self.script = self.tools['fdtd'][2]
        self.vars = {}          # dest -> tk Variable
        self.widgets = {}       # dest -> Eingabe-Widget (fuer Enable/Disable)
        self.labels = {}        # dest -> Label-Widget
        self.placeholders = {}  # dest -> Platzhaltertext (fuer None-Defaults)
        self.proc = None
        self.q = queue.Queue()
        root.title('Lens sensor - launcher')
        root.geometry('880x860')
        self._build()
        self.root.after(100, self._drain_log)

    # ---- UI-Aufbau ----
    def _build(self):
        self._init_style()
        top = ttk.Frame(self.root, padding=(12, 10, 12, 6), style='Head.TFrame')
        top.pack(fill='x')
        self.title_lbl = ttk.Label(top, text='Start FDTD simulation', style='Title.TLabel')
        self.title_lbl.pack(anchor='w')
        ttk.Label(top, text='Grouped by section; the most important choices are at the top. '
                  'Grey field = automatic default. Hover over a label to see the '
                  'CLI flag and help.',
                  style='Sub.TLabel', wraplength=840, justify='left').pack(anchor='w',
                                                                          pady=(2, 0))

        # Werkzeug-Auswahl (nur wenn Moden-Propagation verfuegbar).
        if len(self.tools) > 1:
            self.tool_var = tk.StringVar(value=self.tool)
            tf = ttk.Frame(self.root, padding=(12, 6, 12, 0))
            tf.pack(fill='x')
            ttk.Label(tf, text='Tool:', style='Sub.TLabel').pack(side='left')
            for key, (label, _p, _s) in self.tools.items():
                ttk.Radiobutton(tf, text=label, value=key, variable=self.tool_var,
                                command=self._on_tool_change).pack(side='left', padx=(8, 0))

        # Uebersichts-Skizze der gewaehlten Geometrie (nur FDTD).
        self.skf = ttk.LabelFrame(self.root, text='  Overview: selected geometry  ',
                                  style='Section.TLabelframe', padding=(6, 4))
        self.skf.pack(fill='x', padx=10, pady=(6, 0))
        self.sketch = tk.Canvas(self.skf, height=165, highlightthickness=0, background='white')
        self.sketch.pack(fill='x', expand=True)

        # Scrollbarer Formularbereich (Container bleibt; Inhalt je Werkzeug neu).
        self.mid = ttk.Frame(self.root)
        self.mid.pack(fill='both', expand=True, padx=10, pady=(6, 0))
        canvas = tk.Canvas(self.mid, highlightthickness=0, background=self._bg)
        vsb = ttk.Scrollbar(self.mid, orient='vertical', command=canvas.yview)
        self.form = ttk.Frame(canvas, padding=(0, 0, 8, 0))
        self.form.bind('<Configure>',
                       lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        win = canvas.create_window((0, 0), window=self.form, anchor='nw')
        # Formularbreite an die Canvas-Breite koppeln, damit die LabelFrames fuellen.
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        canvas.bind_all('<MouseWheel>',
                        lambda e: canvas.yview_scroll(int(-e.delta/120), 'units'))
        self.sketch.bind('<Configure>', self._draw_sketch)
        self.sketch.bind('<Button-1>', self._on_sketch_click)   # CAD: Flaeche offen/reflektierend

        self._build_buttons_and_log()
        self._populate_form()

    def _on_tool_change(self):
        key = self.tool_var.get()
        if key == self.tool:
            return
        self.tool = key
        self.parser = self.tools[key][1]
        self.script = self.tools[key][2]
        self._populate_form()

    def _populate_form(self):
        """Formular fuer das aktive Werkzeug (self.parser) neu aufbauen."""
        for w in self.form.winfo_children():
            w.destroy()
        self.vars.clear(); self.widgets.clear(); self.labels.clear(); self.placeholders.clear()
        self._name_slots = {}
        for grp in self.parser._action_groups:
            acts = [a for a in grp._group_actions
                    if a.option_strings and a.dest != 'help' and a.dest not in _GUI_HIDE]
            if not acts:
                continue
            sec = ttk.LabelFrame(self.form, text='  ' + _GROUP_EN.get(grp.title or '', grp.title or '') + '  ',
                                 style='Section.TLabelframe', padding=(12, 8, 12, 10))
            sec.pack(fill='x', expand=True, pady=(0, 10), padx=2)
            sec.columnconfigure(1, weight=1)
            for row, act in enumerate(acts):
                self._add_field(sec, row, act)

        if self.tool == 'fdtd' and self._name_slots:
            # Interner Wert fuer --layer-names (kein eigenes Feld): aus den Namensfeldern
            self.vars['layer_names'] = tk.StringVar(value='')
            for i in self._name_slots:
                self._set_name_ph(i)
        if self.tool == 'fdtd':
            # Bedingte Sichtbarkeit: Szenario nur bei Geometrie == lens aktiv.
            if 'geometry' in self.vars and 'scenario' in self.widgets:
                self.vars['geometry'].trace_add('write', self._apply_conditional)
                self._apply_conditional()
            # Feldbeschriftungen der Schichten folgen den Schichtnamen.
            for dest in ('geometry', 'layer_names'):
                v = self.vars.get(dest)
                if v is not None:
                    v.trace_add('write', self._relabel_layers)
            self._relabel_layers()
            # Skizze bei Aenderung relevanter Felder neu zeichnen.
            for dest in ('geometry', 'dim', 'wg_material', 'wg_thickness_um', 'bead_diameter',
                         'bead_x', 'bead_y_um', 'no_bead', 'vcsel_offset', 'vcsel_tilt',
                         'input_gap', 'length_um', 'wg_width', 'lz_um', 'scenario',
                         'cad', 'cad_plane_z', 'cad_scale', 'lambda_nm', 'layer_names'):
                v = self.vars.get(dest)
                if v is not None:
                    try:
                        v.trace_add('write', self._draw_sketch)
                    except Exception:
                        pass

        # Titel, Skizze und Dry-Run-Button je Werkzeug.
        self.title_lbl.configure(
            text='Start mode propagation (contact lens)' if self.tool == 'modes'
            else 'Start FDTD simulation')
        if self.tool == 'fdtd':
            if not self.skf.winfo_ismapped():
                self.skf.pack(fill='x', padx=10, pady=(6, 0), before=self.mid)
            self._draw_sketch()
        else:
            self.skf.pack_forget()
        if hasattr(self, 'btn_dry'):                 # Moden-Solver hat kein --dry-run
            self.btn_dry.configure(state='disabled' if self.tool == 'modes' else 'normal')

    def _resolved_geometry(self):
        """Aufgeloeste Geometrie ('lens'/'planar') aus dem Freitext, oder None."""
        key = str(self.vars['geometry'].get()).strip().lower()
        return rs._GEOM_ALIASES.get(key)

    def _apply_conditional(self, *_):
        """Felder je nach Geometrie aktivieren/ausgrauen."""
        geom = self._resolved_geometry()
        self._set_enabled('scenario', geom == 'lens')
        for dest in ('cad', 'cad_scale', 'cad_axis', 'cad_materials', 'cad_core',
                     'cad_tear', 'cad_background', 'cad_plane_z'):
            self._set_enabled(dest, geom == 'cad')
        # Schichtstapel/Bead gibt es bei CAD nicht (Materialien kommen aus den Koerpern)
        for dest in ('wg_material', 'wg_thickness_um', 'wg_n', 't_aqueous', 't_mucin',
                     't_lipid', 'n_aqueous', 'n_mucin', 'n_cornea', 'n_lipid',
                     'bead_material', 'bead_diameter', 'bead_x', 'bead_y_um', 'bead_n',
                     'kappa_wg', 'kappa_lipid', 'kappa_aqueous', 'kappa_mucin',
                     'kappa_cornea', 'kappa_bead',
                     'no_reference', 'length_um', 'curved', 'layer_names'):
            if dest in self.widgets:
                self._set_enabled(dest, geom != 'cad')
        for slot in getattr(self, '_name_slots', {}).values():
            slot['ent'].configure(state='normal' if geom != 'cad' else 'disabled')

    # Schichtfelder -> (Index in der Namensliste, Beschriftungsmuster)
    _LAYER_FIELDS = {
        'wg_material': (0, '{}: material'), 'wg_thickness_um': (0, '{}: thickness (um)'),
        'wg_n': (0, '{}: refractive index (override)'),
        't_lipid': (1, '{}: thickness (um)'), 'n_lipid': (1, '{}: refractive index (override)'),
        't_aqueous': (2, '{}: thickness (um)'), 'n_aqueous': (2, '{}: refractive index (override)'),
        't_mucin': (3, '{}: thickness (um)'), 'n_mucin': (3, '{}: refractive index (override)'),
        'n_cornea': (4, '{}: refractive index (override)'),
        'kappa_wg': (0, '{}: absorption kappa'), 'kappa_lipid': (1, '{}: absorption kappa'),
        'kappa_aqueous': (2, '{}: absorption kappa'), 'kappa_mucin': (3, '{}: absorption kappa'),
        'kappa_cornea': (4, '{}: absorption kappa'),
    }

# Namensfeld je Schicht: hinter der Dicke (Kern, Schicht 1-3) bzw. der Substrat-Brechzahl
    _NAME_SLOT = {'wg_thickness_um': 0, 't_lipid': 1, 't_aqueous': 2, 't_mucin': 3,
                  'n_cornea': 4}

    def _default_layer_names(self):
        geom = self._resolved_geometry() if 'geometry' in self.vars else None
        return list(rs.LENS_LAYER_NAMES if geom == 'lens' else rs.GENERIC_LAYER_NAMES)

    def _add_name_slot(self, parent, row, i):
        fr = ttk.Frame(parent)
        fr.grid(row=row, column=2, sticky='w', padx=(10, 0))
        ttk.Label(fr, text='Name').pack(side='left', padx=(0, 4))
        var = tk.StringVar(value='')
        ent = ttk.Entry(fr, textvariable=var, width=16)
        ent.pack(side='left')
        _Tooltip(ent, 'Display name of this layer (GUI, sketch, analyzer legends).\n'
                      'Empty = default name (lens: tear-film names, otherwise generic).')
        slot = {'var': var, 'ent': ent, 'is_ph': False}
        self._name_slots[i] = slot

        def on_in(_e=None):
            if slot['is_ph']:
                slot['is_ph'] = False
                var.set(''); ent.configure(foreground='black')

        def on_out(_e=None):
            if var.get().strip() == '':
                self._set_name_ph(i)
        ent.bind('<FocusIn>', on_in)
        ent.bind('<FocusOut>', on_out)
        var.trace_add('write', lambda *_: self._compose_layer_names())

    def _set_name_ph(self, i):
        """Grauer Platzhalter = Standardname der Schicht (zaehlt als 'nicht gesetzt')."""
        slot = self._name_slots[i]
        slot['is_ph'] = True
        ph = self._default_layer_names()[i]
        if slot['var'].get() != ph:
            slot['var'].set(ph)
        slot['ent'].configure(foreground='grey')

    def _compose_layer_names(self):
        """Namensfelder -> internes --layer-names (leere Eintraege = Standardname)."""
        v = self.vars.get('layer_names')
        if v is None:
            return
        parts = ['' if (i not in self._name_slots or self._name_slots[i]['is_ph'])
                 else self._name_slots[i]['var'].get().strip().replace(',', ' ')
                 for i in range(5)]
        txt = ','.join(parts) if any(parts) else ''
        if v.get() != txt:
            v.set(txt)

    def _layer_names_now(self):
        """Schichtnamen wie run_simulation._layer_names (Linse vs. neutral + Feld)."""
        geom = self._resolved_geometry()
        base = list(rs.LENS_LAYER_NAMES if geom == 'lens' else rs.GENERIC_LAYER_NAMES)
        v = self.vars.get('layer_names')
        txt = v.get().strip() if v is not None else ''
        if txt and txt != self.placeholders.get('layer_names'):
            for i, t in enumerate(txt.split(',')[:5]):
                if t.strip():
                    base[i] = t.strip()
        return base

    def _relabel_layers(self, *_):
        """Beschriftung der Schicht-Dicken/Brechzahlen nach den aktuellen Schichtnamen."""
        for i, slot in getattr(self, '_name_slots', {}).items():
            if slot['is_ph']:
                self._set_name_ph(i)
        try:
            names = self._layer_names_now()
        except Exception:
            return
        for dest, (i, pat) in self._LAYER_FIELDS.items():
            lbl = self.labels.get(dest)
            if lbl is not None:
                try:
                    lbl.configure(text=pat.format(names[i]))
                except Exception:
                    pass

    # ---- CAD: Datei waehlen, Koerper/Materialien zuweisen, Vorschau ----------
    def _cad_path(self):
        v = self.vars.get('cad')
        p = v.get().strip() if v is not None else ''
        return p if p and os.path.exists(p) else None

    def _cad_scale(self):
        v = self.vars.get('cad_scale')
        s = v.get().strip() if v is not None else ''
        if not s or s == self.placeholders.get('cad_scale'):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def _cad_field(self, dest):
        """Text eines CAD-Feldes (--cad-axis/-core/-tear/-background) oder None, wenn leer
        bzw. Platzhalter. Die Vorschau muss dieselbe Szene bauen wie der Lauf."""
        v = self.vars.get(dest)
        s = str(v.get()).strip() if v is not None else ''
        if not s or s == self.placeholders.get(dest):
            return None
        return s

    def _cad_overrides(self):
        return dict(scale_um=self._cad_scale(), axis=self._cad_field('cad_axis'),
                    core=self._cad_field('cad_core'), tear=self._cad_field('cad_tear'),
                    background=self._cad_field('cad_background'))

    def _cad_browse(self):
        from tkinter import filedialog
        from common.cad_geometry import SUPPORTED_EXT
        pats = ' '.join('*' + e for e in SUPPORTED_EXT)
        p = filedialog.askopenfilename(
            title='Select CAD geometry', initialdir=os.path.join(_REPO_ROOT, 'cad_examples'),
            filetypes=[('CAD (SolidCraft, STEP, Inventor, STL ...)', pats), ('All files', '*.*')])
        if p:
            self.vars['cad'].set(p)
            if 'geometry' in self.vars:
                self.vars['geometry'].set('cad')
            self._cad_scene_cache = None
            self._cad_dialog()

    def _load_cad_scene(self):
        """CadScene der gewaehlten Datei (gecacht; Konvertierung kann bei Inventor dauern)."""
        from common.cad_geometry import CadScene
        p = self._cad_path()
        if not p:
            return None
        ov = self._cad_overrides()          # wie run_simulation._cad_setup (Achse/Kern/Film/...)
        key = (p, os.path.getmtime(p), tuple(sorted(ov.items())))
        cache = getattr(self, '_cad_scene_cache', None)
        if cache and cache[0] == key:
            return cache[1]
        self.root.config(cursor='watch'); self.root.update_idletasks()
        try:
            sc = CadScene.from_config(p, **ov)
        finally:
            self.root.config(cursor='')
        self._cad_scene_cache = (key, sc)
        return sc

    def _cad_dialog(self):
        """Koerperliste mit Material-Dropdown, Kern/Film, Massstab -> <datei>.lens.json."""
        from common.cad_geometry import MATERIALS
        try:
            sc = self._load_cad_scene()
        except Exception as e:
            messagebox.showerror('CAD', f'File not readable:\n{e}')
            return
        if sc is None:
            messagebox.showinfo('CAD', 'Select a CAD file first.')
            return
        dlg = tk.Toplevel(self.root)
        dlg.title(f'Bodies & materials - {os.path.basename(sc.path)}')
        dlg.transient(self.root)
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill='both', expand=True)
        ttk.Label(frm, text='Display name is free text (analyzer/legend). Material per body: list, '
                  '"n=1.52", absorbing "n=1.52+0.002i" or "aqueous+0.001i" (n + i*kappa).\n'
                  'Where bodies overlap, the smaller one wins.',
                  foreground='#555').grid(
            row=0, column=0, columnspan=6, sticky='w', pady=(0, 8))
        for c, h in enumerate(('Body (CAD)', 'Size x/y/z (um)', 'Volume (um3)', 'Material', 'Display name')):
            ttk.Label(frm, text=h, font=('Segoe UI', 9, 'bold')).grid(row=1, column=c, sticky='w', padx=4)
        mats = {}
        lbls = {}
        for r, b in enumerate(sc.body_order()[::-1], start=2):
            lo, hi = sc.bbox(b)
            ttk.Label(frm, text=b).grid(row=r, column=0, sticky='w', padx=4)
            ttk.Label(frm, text=' x '.join(f'{v:.2f}' for v in (hi - lo))).grid(row=r, column=1, sticky='w', padx=4)
            ttk.Label(frm, text=f'{sc.volumes[b]:.3g}').grid(row=r, column=2, sticky='w', padx=4)
            v = tk.StringVar(value=str(sc.materials.get(b, '')))
            ttk.Combobox(frm, textvariable=v, values=list(MATERIALS), width=14).grid(
                row=r, column=3, sticky='w', padx=4, pady=2)
            mats[b] = v
            vl = tk.StringVar(value=sc.labels.get(b, ''))
            ttk.Entry(frm, textvariable=vl, width=18).grid(row=r, column=4, sticky='w', padx=4, pady=2)
            lbls[b] = vl
        r += 1
        names = ['(automatic)'] + list(sc.bodies)
        ttk.Separator(frm).grid(row=r, column=0, columnspan=5, sticky='ew', pady=8); r += 1
        v_core = tk.StringVar(value=sc.core or '(automatic)')
        v_tear = tk.StringVar(value=sc.tear or '(automatic)')
        v_bg = tk.StringVar(value=str(sc.background))
        v_scale = tk.StringVar(value=f'{sc.scale_um:g}')
        for lbl, var, vals in (('Core (waveguide)', v_core, names),
                               ('Film (sensor evaluation)', v_tear, names),
                               ('Background', v_bg, list(MATERIALS)),
                               ('Scale (um per CAD mm)', v_scale, ['1000', '1'])):
            ttk.Label(frm, text=lbl).grid(row=r, column=0, columnspan=2, sticky='w', padx=4)
            ttk.Combobox(frm, textvariable=var, values=vals, width=18).grid(
                row=r, column=2, columnspan=2, sticky='w', padx=4, pady=2)
            r += 1

        def save():
            try:
                for b, v in mats.items():
                    s = v.get().strip()
                    if s:
                        sc._n(s, 850.0)               # pruefen (wirft bei Tippfehler)
                    sc.materials[b] = s
                sc.labels = {b: vl.get().strip() for b, vl in lbls.items() if vl.get().strip()}
                sc.core = None if v_core.get().startswith('(') else v_core.get()
                sc.tear = None if v_tear.get().startswith('(') else v_tear.get()
                sc.background = v_bg.get().strip() or 'air'
                new_scale = float(v_scale.get())
            except Exception as e:
                messagebox.showerror('CAD', str(e), parent=dlg)
                return
            if abs(new_scale - sc.scale_um) > 1e-12:     # Massstab neu -> Szene neu laden
                sc.scale_um = new_scale
                sc.save_config()
                self._cad_scene_cache = None
                if 'cad_scale' in self.vars:
                    self.vars['cad_scale'].set(f'{new_scale:g}')
            else:
                sc.save_config()
            self._append(f'[CAD] Assignment saved: {sc.config_path(sc.path)}\n')
            dlg.destroy()
            self._draw_sketch()
        btns = ttk.Frame(frm)
        btns.grid(row=r, column=0, columnspan=5, sticky='e', pady=(10, 0))
        ttk.Button(btns, text='Cancel', command=dlg.destroy).pack(side='right', padx=4)
        ttk.Button(btns, text='Save', command=save).pack(side='right')

    def _set_enabled(self, dest, on):
        w = self.widgets.get(dest)
        lbl = self.labels.get(dest)
        if isinstance(w, ttk.Frame):                     # zusammengesetztes Feld (CAD-Datei)
            for ch in w.winfo_children():
                try:
                    ch.configure(state='normal' if on else 'disabled')
                except Exception:
                    pass
            w = None
        if w is not None:
            if isinstance(w, ttk.Combobox):
                w.configure(state='readonly' if on else 'disabled')
            else:
                w.configure(state='normal' if on else 'disabled')
        if lbl is not None:
            lbl.configure(foreground='#333333' if on else '#a0a0a0')

    # ---- Uebersichts-Skizze -------------------------------------------------
    def _sketch_val(self, dest, default):
        """Numerischer Feldwert; Platzhalter/leer/ungueltig -> default."""
        v = self.vars.get(dest)
        if v is None:
            return default
        s = v.get()
        if s == self.placeholders.get(dest) or str(s).strip() == '':
            return default
        try:
            return float(s)
        except Exception:
            return default

    def _wave(self, c, x0, x1, yc, amp, cycles=6, color='#c58a00'):
        n = 60
        pts = []
        for k in range(n + 1):
            x = x0 + (x1 - x0)*k/n
            pts += [x, yc - amp*math.sin(2*math.pi*cycles*k/n)]
        c.create_line(*pts, fill=color, width=1)

    def _draw_sketch(self, *_):
        if getattr(self, 'tool', 'fdtd') != 'fdtd':      # Skizze nur fuer FDTD
            return
        c = getattr(self, 'sketch', None)
        if c is None:
            return
        c.delete('all')
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 50:
            W = 820
        if H < 40:
            H = 165
        geom = self._resolved_geometry() or 'planar'
        try:
            dim = int(float(self.vars['dim'].get()))
        except Exception:
            dim = 3
        xL, xR = 74, W - 150            # links Quelle, rechts Legende
        yT, yB = 20, H - 12
        if geom == 'cad':
            self._sketch_cad(c, xL, xR, yT, yB, W)
        elif geom == 'lens':
            self._sketch_lens(c, xL, xR, yT, yB)
            # Propagation: Rand (VCSEL links) -> Apex, links->rechts (wie planar).
            c.create_line(xL, 10, xL + 44, 10, fill='#14618c', width=2, arrow='last')
            c.create_text(xL + 48, 10, anchor='w',
                          text='x: rim -> apex (propagation, as in planar)',
                          fill='#14618c', font=('Segoe UI', 7))
        else:
            self._sketch_planar(c, xL, xR, yT, yB)
            c.create_line(xL, 10, xL + 44, 10, fill='#14618c', width=2, arrow='last')
            c.create_text(xL + 48, 10, anchor='w', text='x (propagation)',
                          fill='#14618c', font=('Segoe UI', 7))
        # 3D-Badge
        if dim == 3:
            lz = self._sketch_val('lz_um', 8.0)
            ww = self.vars.get('wg_width')
            is_chan = bool(ww is not None and ww.get().strip()
                           and ww.get() != self.placeholders.get('wg_width'))
            c.create_text(W - 6, H - 4, anchor='se',
                          text=f'3D · z depth {lz:g} µm · {"channel" if is_chan else "slab"}',
                          fill='#14618c', font=('Segoe UI', 7, 'bold'))
        else:
            c.create_text(W - 6, H - 4, anchor='se', text='2D (z invariant)',
                          fill='#14618c', font=('Segoe UI', 7, 'bold'))

    def _sketch_planar(self, c, xL, xR, yT, yB):
        Ht = yB - yT
        _nm = self._layer_names_now()
        bands = [('Air (n=1.0)', '#eaf4ff', 0.17),
                 ('WG core', '#ffd75e', 0.17),
                 (_nm[1], '#fff2cc', 0.07),
                 (_nm[2], '#bfe3ff', 0.27),
                 (_nm[3], '#e2d1f4', 0.11),
                 (_nm[4], '#f8c9c9', 0.21)]
        y = yT
        ycore = core_h = 0
        wg_mat = (self.vars['wg_material'].get() if 'wg_material' in self.vars else 'pmma')
        for name, col, frac in bands:
            h = frac*Ht
            c.create_rectangle(xL, y, xR, y + h, fill=col, outline='#cfd8e0')
            c.create_text(xR + 6, y + h/2, anchor='w', text=name, fill='#444',
                          font=('Segoe UI', 7))
            if name == 'WG core':
                ycore, core_h = y, h
                d = self._sketch_val('wg_thickness_um', 5.0)
                c.create_text((xL + xR)/2, y + h/2,
                              text=f'{wg_mat.upper()}  d={d:g} um', fill='#5a4600',
                              font=('Segoe UI', 8, 'bold'))
            y += h
        ycm = ycore + core_h/2
        # gefuehrte Welle im Kern
        self._wave(c, xL + 4, xR - 4, ycm, core_h*0.30)
        # Quelle (VCSEL) links, mit Offset/Neigung
        off = self._sketch_val('vcsel_offset', 0.0)
        tilt = max(-45.0, min(45.0, self._sketch_val('vcsel_tilt', 0.0)))
        gap = self._sketch_val('input_gap', 0.0)
        ysrc = ycm - off*3.0
        dy = math.tan(math.radians(tilt))*(xL - 12)
        c.create_line(10, ysrc - dy, xL - 2, ysrc, fill='#e23', width=3, arrow='last')
        c.create_text(10, ysrc - 12, anchor='w', text='VCSEL', fill='#e23',
                      font=('Segoe UI', 7, 'bold'))
        if gap > 0:
            c.create_line(xL, yT, xL, yB, fill='#e23', dash=(2, 2))
            c.create_text(xL + 3, yB - 6, anchor='w', text=f'gap {gap:g}um',
                          fill='#e23', font=('Segoe UI', 6))
        # evaneszenter Pfeil in den Traenenfilm
        xa = xL + 0.62*(xR - xL)
        c.create_line(xa, ycore + core_h, xa, ycore + core_h + 0.16*Ht,
                      fill='#0a7', width=2, arrow='last', dash=(3, 2))
        c.create_text(xa + 4, ycore + core_h + 0.09*Ht, anchor='w', text='evanescent',
                      fill='#0a7', font=('Segoe UI', 6))
        # Bead (nur wenn nicht --no-bead und Durchmesser > 0)
        nb = self.vars.get('no_bead')
        if not (nb is not None and bool(nb.get())):
            bd = self._sketch_val('bead_diameter', 0.5)
            if bd > 0:
                length = self._sketch_val('length_um', 0.0)
                bx = self._sketch_val('bead_x', float('nan'))
                fx = 0.5 if (length <= 0 or math.isnan(bx)) else max(0.05, min(0.95, bx/length))
                by = self._sketch_val('bead_y_um', float('nan'))
                r = max(4, min(14, bd*6))
                # Default (bead_y leer): Oberkante an der WG-Unterkante (y=0), also
                # direkt am Waveguide anliegend -> Mittelpunkt r unter der Kernkante.
                cyb = (ycore + core_h + r) if math.isnan(by) else (ycm - by*3.0)
                cxb = xL + fx*(xR - xL)
                c.create_oval(cxb - r, cyb - r, cxb + r, cyb + r,
                              fill='#ff7043', outline='#a33')
                c.create_text(cxb, cyb - r - 6, text=f'Bead {bd:g}um', fill='#a33',
                              font=('Segoe UI', 6))

    def _sketch_lens(self, c, xL, xR, yT, yB):
        """Kontaktlinse = gekruemmter Schalen-Waveguide KONSTANTER Dicke (250 um)
        entlang der Augenkruemmung (R~8.3 mm); Traenenfilm zwischen Linse und Cornea.
        Kruemmung ist fuer die Skizze uebertrieben (der reale Sim-Ausschnitt ist fast
        flach)."""
        Ht = yB - yT
        n = 56
        cur = 0.22*Ht                       # uebertriebene Woelbung (Dome nach oben)
        # Kanten-Basislinien (an den Raendern; Mitte woelbt sich um 'cur' nach oben).
        # Traenenfilm ZWISCHEN Linse und Cornea, in seinen drei Schichten (von der
        # Linse abwaerts): Lipid | waessrig (Aqueous) | Mucin.
        e_lens_top = yT + 0.22*Ht
        e_lens_bot = yT + 0.44*Ht           # -> Linsendicke konstant ~250 um
        e_lip_bot = yT + 0.49*Ht            # Lipidschicht (an der Linse)
        e_aq_bot = yT + 0.60*Ht             # waessrige Schicht (dickste)
        e_tear_bot = yT + 0.66*Ht           # Mucinschicht (an der Cornea)
        e_cornea_bot = yB

        def surf(edge):
            return [(xL + (xR - xL)*k/n, edge - cur*(1 - (2*(k/n) - 1)**2))
                    for k in range(n + 1)]

        def band(et, eb, fill, name, col='#333', fs=7):
            poly = surf(et) + surf(eb)[::-1]
            c.create_polygon(*[v for xy in poly for v in xy], fill=fill,
                             outline='#cfd8e0', smooth=True)
            if name:
                c.create_text(xR + 6, (et + eb)/2, anchor='w', text=name, fill=col,
                              font=('Segoe UI', fs))

        c.create_text(xR + 6, yT + 0.10*Ht, anchor='w', text='Air (outside)',
                      fill='#6a8', font=('Segoe UI', 7))
        band(e_tear_bot, e_cornea_bot, '#c9e8c9', 'Cornea (eye)')                  # gruen
        band(e_aq_bot, e_tear_bot, '#e2d1f4', 'Mucin', col='#6a4a8a', fs=6)         # violett
        band(e_lip_bot, e_aq_bot, '#bfe3ff', 'Tear film aqueous',
             col='#1a6', fs=6)                                                      # blau
        band(e_lens_bot, e_lip_bot, '#fff2cc', 'Lipid', col='#8a6d1a', fs=6)       # gelblich
        band(e_lens_top, e_lens_bot, '#ffe08a', '')                                # Linse gelb
        xc = (xL + xR)/2
        c.create_text(xc, (e_lens_top + e_lens_bot)/2 - cur,
                      text='Contact lens PMMA  d=250 um  R=8.3 mm', fill='#5a4600',
                      font=('Segoe UI', 8, 'bold'))

        # TIR-Zickzack im Glas (Fuehrung durch Totalreflexion, vom rechten Rand)
        bounces = 8

        def top_y(t):
            return e_lens_top - cur*(1 - (2*t - 1)**2) + 3

        def bot_y(t):
            return e_lens_bot - cur*(1 - (2*t - 1)**2) - 3
        zz = []
        for j in range(bounces + 1):
            t = j/bounces
            zz += [xL + (xR - xL)*t, bot_y(t) if j % 2 else top_y(t)]
        c.create_line(*zz, fill='#d33', width=1.4)
        # evaneszente Pfeile an den unteren TIR-Punkten in den Traenenfilm
        for j in range(1, bounces, 2):
            t = j/bounces
            x = xL + (xR - xL)*t
            c.create_line(x, bot_y(t), x, bot_y(t) + 0.05*Ht, fill='#0a7', width=1.4,
                          arrow='last', dash=(2, 2))
        c.create_text(xc, e_lip_bot - cur + 0.02*Ht, text='evanescent -> tear film (lipid)',
                      fill='#0a7', font=('Segoe UI', 6))

        # TANGENTIALE Einkopplung: der Strahl laeuft entlang der LINSEN-TANGENTE am
        # linken Rand (Auflagen-/Fuehrungsflaeche), NICHT horizontal. Facette steht
        # senkrecht zur Tangente ueber die Linsendicke.
        y_rim = (e_lens_top + e_lens_bot)/2
        half = (e_lens_bot - e_lens_top)/2
        tvx, tvy = (xR - xL), -4.0*cur                 # Tangente der Mittellinie bei t=0
        tn = math.hypot(tvx, tvy); tvx, tvy = tvx/tn, tvy/tn
        px, py = -tvy, tvx                             # Normale (ueber die Linsendicke)
        off = self._sketch_val('vcsel_offset', 0.0)
        tilt = max(-45.0, min(45.0, self._sketch_val('vcsel_tilt', 0.0)))
        ox = max(-half, min(half, off*3.0))            # Eintritt ueber die Dicke verschoben
        ex_, ey_ = xL + px*ox, y_rim + py*ox
        # Eintrittsfacette senkrecht zur Tangente (magenta), ueber die Linsendicke
        c.create_line(ex_ - px*half, ey_ - py*half, ex_ + px*half, ey_ + py*half,
                      fill='magenta', width=2)
        # gestrichelte Tangente entlang der Fuehrungsflaeche (zeigt "tangential")
        c.create_line(ex_, ey_, ex_ + tvx*0.20*(xR - xL), ey_ + tvy*0.20*(xR - xL),
                      fill='#e23', width=1, dash=(3, 2))
        # Strahl entlang Tangente (+ optionaler Tilt), von links-unten in die Facette
        ang = math.atan2(tvy, tvx) + math.radians(tilt)
        Ls = 50
        x0, y0 = ex_ - math.cos(ang)*Ls, ey_ - math.sin(ang)*Ls
        c.create_rectangle(x0 - 11, y0 - 6, x0 + 1, y0 + 6, fill='#FF3333', outline='black')
        c.create_line(x0, y0, ex_, ey_, fill='#e23', width=3, arrow='last')
        c.create_text(x0 - 13, y0, anchor='e', text='VCSEL', fill='#e23',
                      font=('Segoe UI', 7, 'bold'))
        c.create_text(ex_ + tvx*0.10*(xR - xL) + 4, ey_ + tvy*0.10*(xR - xL) + 8,
                      anchor='w', text='tangentially coupled', fill='#e23',
                      font=('Segoe UI', 6))
        # Szenario (falls aktiv)
        sc = self.vars.get('scenario')
        w = self.widgets.get('scenario')
        if sc is not None and w is not None and str(w.cget('state')) != 'disabled':
            c.create_text(xc, e_lens_top - cur - 9, text=f'Scenario: {sc.get()}',
                          fill='#14618c', font=('Segoe UI', 7, 'bold'))

    _CAD_COLORS = ('#e8c35a', '#7cc3e8', '#c79ae0', '#9fd86b', '#f0a080', '#a0b4f0',
                   '#f4b6c8', '#6fd6cc', '#e0b080', '#b8c8dc')

    def _sketch_cad(self, c, xL, xR, yT, yB, W):
        """CAD-Vorschau: 2D-Schnitt (Regionen) in der Ebene z=--cad-plane-z, dazu die
        automatisch ausgerichtete Quelle (Pfeil auf die Eintrittsflaeche)."""
        import numpy as np
        from common.cad_geometry import CadRun
        try:
            sc = self._load_cad_scene()
        except Exception as e:
            c.create_text(xL, (yT + yB)//2, anchor='w', fill='#b00',
                          text=f'CAD file not readable: {e}', font=('Segoe UI', 8))
            return
        if sc is None:
            c.create_text(xL, (yT + yB)//2, anchor='w', fill='#555', font=('Segoe UI', 9),
                          text='Select a CAD file (CAD field, "File..." button)')
            return
        lam = self._sketch_val('lambda_nm', 850.0)
        missing = sc.unassigned()
        saved = dict(sc.materials)
        for b in missing:                               # Vorschau auch ohne Zuordnung
            sc.materials[b] = 'n=1.5'
        try:
            pz = self._sketch_val('cad_plane_z', None)
            run = CadRun(sc, 2, 0.05, lam, plane_z_um=pz)
            nx, ny = max(50, int(xR - xL)), max(30, int(yB - yT))
            xs = np.linspace(0, run.lx_um, nx)
            ys = np.linspace(run.y0_um, run.y0_um + run.ly_um, ny)
            _, reg = sc.eps_region(xs, ys, np.array([run.z0_um]), lam)
            src = run.source(0.3)
        except Exception as e:
            c.create_text(xL, (yT + yB)//2, anchor='w', fill='#b00',
                          text=f'CAD preview: {e}', font=('Segoe UI', 8))
            return
        finally:
            sc.materials.clear(); sc.materials.update(saved)
        reg = reg[:, :, 0]
        pal = ['#f4f6f8'] + [self._CAD_COLORS[i % len(self._CAD_COLORS)] for i in range(len(sc.bodies))]
        img = tk.PhotoImage(width=nx, height=ny)
        rows = []
        for j in range(ny - 1, -1, -1):                 # oben = grosses y
            rows.append('{' + ' '.join(pal[v] for v in reg[:, j]) + '}')
        img.put(' '.join(rows))
        self._sketch_img = img                          # Referenz halten (sonst weg)
        c.create_image(xL, yT, image=img, anchor='nw')
        c.create_rectangle(xL, yT, xL + nx, yT + ny, outline='#999')

        def cx(x):
            return xL + x/run.lx_um*nx

        def cy(y):
            return yT + (run.y0_um + run.ly_um - y)/run.ly_um*ny
        f = run.facet
        fc, d = f['center'], f['dir']
        L = 0.12*run.lx_um
        c.create_line(cx(fc[0] - L*d[0]), cy(fc[1] - L*d[1]), cx(fc[0]), cy(fc[1]),
                      fill='#e23', width=2, arrow='last')
        c.create_text(xL - 6, cy(fc[1] - L*d[1]), anchor='e', text='VCSEL', fill='#e23',
                      font=('Segoe UI', 8, 'bold'))
        # Flaechen fuer den Mausklick (offen <-> reflektierend); offene rot markieren
        faces = [q for q in sc.planar_faces()
                 if q['axis'] in (0, 1) and q['lo'][2] - 1e-6 <= run.z0_um <= q['hi'][2] + 1e-6]
        for q in faces:
            if not sc.is_open(q):
                continue
            if q['axis'] == 0:                          # Normale x -> senkrechte Linie
                X = q['center'][0]
                c.create_line(cx(X), cy(q['lo'][1]), cx(X), cy(q['hi'][1]), fill='#d00', width=4)
            else:                                       # Normale y -> waagerechte Linie
                Y = q['center'][1]
                c.create_line(cx(q['lo'][0]), cy(Y), cx(q['hi'][0]), cy(Y), fill='#d00', width=4)
        self._cad_pick = dict(scene=sc, run=run, faces=faces, xL=xL, yT=yT, nx=nx, ny=ny)
        c.create_text(xL + nx, yT - 3, anchor='se', font=('Segoe UI', 7), fill='#a00',
                      text='Click a face: open (red, runs into the absorber) <-> reflecting')
        ty = yT
        for i, b in enumerate(sc.region_names(), start=1):
            tag = ' [core]' if b == run.core else (' [film]' if b == sc.tear else '')
            m = saved.get(b) or '?'
            c.create_rectangle(xR + 10, ty + 2, xR + 20, ty + 12, fill=pal[i], outline='#666')
            c.create_text(xR + 24, ty + 7, anchor='w', font=('Segoe UI', 7),
                          text=f'{b}: {m}{tag}', fill='#b00' if b in missing else '#222')
            ty += 14
        c.create_text(xL, yB + 1, anchor='sw', font=('Segoe UI', 7), fill='#555',
                      text=f'Slice z={run.z0_um:.2f} um   Box {run.lx_um:.1f} x {run.ly_um:.1f} um   '
                           f'Source: tilt {src["tilt_deg"]:.1f} deg'
                           + ('   MISSING material: ' + ', '.join(missing) if missing else ''))

    def _on_sketch_click(self, ev):
        """CAD-Vorschau: Flaeche unter dem Mausklick offen <-> reflektierend schalten."""
        import numpy as np
        pk = getattr(self, '_cad_pick', None)
        if pk is None or self._resolved_geometry() != 'cad':
            return
        run, sc = pk['run'], pk['scene']
        if not (pk['xL'] <= ev.x <= pk['xL'] + pk['nx'] and pk['yT'] <= ev.y <= pk['yT'] + pk['ny']):
            return
        x = (ev.x - pk['xL'])/pk['nx']*run.lx_um
        y = run.y0_um + run.ly_um - (ev.y - pk['yT'])/pk['ny']*run.ly_um
        sx, sy = run.lx_um/pk['nx'], run.ly_um/pk['ny']          # um pro Pixel
        best, bd = None, 1e9
        for q in pk['faces']:
            ax = q['axis']
            if ax == 0:                                 # Abstand in Pixeln zur Flaechenlinie
                d = abs(x - q['center'][0])/sx
                inside = q['lo'][1] - 3*sy <= y <= q['hi'][1] + 3*sy
            else:
                d = abs(y - q['center'][1])/sy
                inside = q['lo'][0] - 3*sx <= x <= q['hi'][0] + 3*sx
            if inside and d < bd:
                best, bd = q, d
        if best is None or bd > 8:
            self._append('[CAD] No face near the click (click on a body edge).\n')
            return
        ent = run.facet
        if sc.is_entrance(best, ent):
            messagebox.showinfo('CAD', 'The entrance face of the core stays closed - '
                                'the source sits in front of it.')
            return
        now_open = sc.toggle_open(best)
        sc.save_config()
        n = best['normal']
        self._append(f'[CAD] {best["body"]}: face with normal '
                     f'{"+-"[0 if n[best["axis"]] > 0 else 1]}{"xyz"[best["axis"]]} at '
                     f'{"xyz"[best["axis"]]}={best["center"][best["axis"]]:.2f} um -> '
                     f'{"OPEN (runs into the absorber)" if now_open else "reflecting"}\n')
        self._draw_sketch()

    def _add_field(self, parent, row, act):
        """Eine Zeile: Klartext-Label (Tooltip: Flag+Hilfe) + passendes Widget."""
        flag = act.option_strings[0]
        lbl = ttk.Label(parent, text=_pretty(act.dest, flag), anchor='w')
        lbl.grid(row=row, column=0, sticky='w', padx=(2, 10), pady=3)
        tip = (flag + '\n' + act.help) if act.help else flag
        _Tooltip(lbl, tip)
        if act.dest == 'geometry':                      # Auswahl, aber frei tippbar (Aliase)
            var = tk.StringVar(value=str(act.default))
            w = ttk.Combobox(parent, textvariable=var, values=['planar', 'lens', 'cad'])
            w.grid(row=row, column=1, sticky='ew')
        elif act.dest == 'cad':                         # Datei + Materialdialog
            var = tk.StringVar(value='')
            w = ttk.Frame(parent)
            w.grid(row=row, column=1, sticky='ew')
            w.columnconfigure(0, weight=1)
            ent = ttk.Entry(w, textvariable=var)
            ent.grid(row=0, column=0, sticky='ew')
            ttk.Button(w, text='File...', width=8,
                       command=self._cad_browse).grid(row=0, column=1, padx=(4, 0))
            ttk.Button(w, text='Bodies & materials...',
                       command=self._cad_dialog).grid(row=0, column=2, padx=(4, 0))
            self._cad_entry = ent
        elif act.nargs == 0:                            # Flag -> Checkbox
            var = tk.BooleanVar(value=bool(act.default))
            w = ttk.Checkbutton(parent, variable=var)
            w.grid(row=row, column=1, sticky='w')
        elif act.choices:                               # Auswahl -> Combobox
            var = tk.StringVar(value='' if act.default is None else str(act.default))
            _vals = [str(c) for c in act.choices if str(c) not in _GUI_CHOICE_HIDE.get(act.dest, ())]
            if act.default is None:                     # leer = Default des Solvers
                _vals = [''] + _vals
            w = ttk.Combobox(parent, textvariable=var, state='readonly', values=_vals)
            w.grid(row=row, column=1, sticky='ew')
        elif act.default is None:                       # Freitext ohne festen Default
            ph = _auto_hint(act.dest)
            var = tk.StringVar(value=ph)
            w = ttk.Entry(parent, textvariable=var, foreground='grey')
            w.grid(row=row, column=1, sticky='ew')
            self.placeholders[act.dest] = ph
            self._bind_placeholder(w, var, ph)
        else:                                           # Freitext mit Default
            var = tk.StringVar(value=str(act.default))
            w = ttk.Entry(parent, textvariable=var)
            w.grid(row=row, column=1, sticky='ew')
        self.vars[act.dest] = var
        self.widgets[act.dest] = w
        self.labels[act.dest] = lbl
        if self.tool == 'fdtd' and act.dest in self._NAME_SLOT:
            self._add_name_slot(parent, row, self._NAME_SLOT[act.dest])

    def _init_style(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        self._bg = '#f4f6f8'
        self.root.configure(background=self._bg)
        style.configure('.', background=self._bg)
        style.configure('Head.TFrame', background='#eaeff3')
        style.configure('Title.TLabel', font=('Segoe UI', 13, 'bold'),
                        background='#eaeff3', foreground='#14618c')
        style.configure('Sub.TLabel', font=('Segoe UI', 9), background='#eaeff3',
                        foreground='#555')
        style.configure('Section.TLabelframe', background=self._bg, relief='solid',
                        borderwidth=1)
        style.configure('Section.TLabelframe.Label', font=('Segoe UI', 10, 'bold'),
                        foreground='#14618c', background=self._bg)
        style.configure('TLabel', background=self._bg)
        style.configure('TCheckbutton', background=self._bg)
        style.configure('TButton', font=('Segoe UI', 9))

    def _build_buttons_and_log(self):
        btns = ttk.Frame(self.root, padding=(10, 8))
        btns.pack(fill='x')
        self.btn_dry = ttk.Button(btns, text='Check configuration (dry run)',
                                  command=self.on_dry_run)
        self.btn_dry.pack(side='left')
        self.btn_run = ttk.Button(btns, text='Start simulation', command=self.on_run)
        self.btn_run.pack(side='left', padx=6)
        self.btn_stop = ttk.Button(btns, text='Stop', command=self.on_stop,
                                   state='disabled')
        self.btn_stop.pack(side='left')
        ttk.Button(btns, text='Open analyzer', command=self.on_analyzer).pack(
            side='left', padx=(16, 0))
        ttk.Button(btns, text='Clear log', command=self._clear_log).pack(side='right')

        self.log = scrolledtext.ScrolledText(self.root, height=12, wrap='word',
                                             font=('Consolas', 9))
        self.log.pack(fill='both', expand=False, padx=10, pady=(0, 10))

    def _bind_placeholder(self, ent, var, ph):
        """Grauer Platzhalter: bei Fokus leeren, bei leerem Verlassen zurueck."""
        def on_in(_e=None):
            if var.get() == ph:
                var.set(''); ent.configure(foreground='black')
        def on_out(_e=None):
            if var.get().strip() == '':
                var.set(ph); ent.configure(foreground='grey')
        ent.bind('<FocusIn>', on_in)
        ent.bind('<FocusOut>', on_out)

    # ---- Aktionen ----
    def _values(self):
        out = {}
        for dest, v in self.vars.items():
            w = self.widgets.get(dest)
            if isinstance(w, ttk.Frame):                 # zusammengesetztes Feld -> sein Eingabefeld
                w = next((ch for ch in w.winfo_children() if isinstance(ch, ttk.Entry)), None)
            if w is not None and str(w.cget('state')) == 'disabled':
                out[dest] = ''          # ausgegrautes Feld -> Default gilt, nicht senden
                continue
            val = v.get()
            # Platzhaltertext == "nichts eingegeben" -> leer (Default gilt).
            out[dest] = '' if val == self.placeholders.get(dest) else val
        return out

    def on_dry_run(self):
        self._launch(collect_argv(self.parser, self._values(), force_dry_run=True))

    def on_run(self):
        self._launch(collect_argv(self.parser, self._values()))

    def on_analyzer(self):
        """Startet den FDTD-Analyzer (eigenes Fenster). Oeffnet das NEUESTE
        Ergebnis in results/ direkt; sonst nur den Datei-Dialog des Analyzers.
        3D-Lauf: zuerst das volle Volumen (_vol3d.npz -> Ebenen xy/xz/yz + Slice),
        die z-Mittelebene (_frames.npz) als zweite Datei (Pfeiltasten wechseln)."""
        analyzer = os.path.join(_REPO_ROOT, 'common', 'fdtd_analyzer.py')
        cmd = [sys.executable, analyzer]
        frames = sorted(glob.glob(os.path.join(_REPO_ROOT, 'results', '*_frames.npz')),
                        key=os.path.getmtime)
        files = []
        if frames:
            vol = frames[-1][:-len('_frames.npz')] + '_vol3d.npz'
            files = ([vol] if os.path.exists(vol) else []) + [frames[-1]]
        cmd += files
        try:
            subprocess.Popen(cmd, cwd=_REPO_ROOT)   # nicht-blockierend, eigenes Fenster
            self._append('[Analyzer started'
                         + (': ' + ' + '.join(os.path.basename(f) for f in files)
                            if files else ' (file dialog)') + ']\n')
        except Exception as e:
            self._append(f'[ERROR] Analyzer start: {e}\n')

    def _launch(self, argv):
        if self.proc is not None:
            messagebox.showinfo('Already running', 'A computation is already running.')
            return
        cmd = [sys.executable, '-u', os.path.join(_REPO_ROOT, self.script)] + argv
        self._append('$ python ' + self.script.replace('\\', '/') + ' ' + ' '.join(argv) + '\n')
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=_REPO_ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception as e:
            self._append(f'[ERROR] Start failed: {e}\n')
            self.proc = None
            return
        self.btn_run['state'] = 'disabled'
        self.btn_dry['state'] = 'disabled'
        self.btn_stop['state'] = 'normal'
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        for line in proc.stdout:
            self.q.put(line)
        proc.wait()
        self.q.put(('__done__', proc.returncode))

    def on_stop(self):
        if self.proc is not None:
            self.proc.terminate()
            self._append('[aborted]\n')

    def _drain_log(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item and item[0] == '__done__':
                    self._append(f'[done, exit code {item[1]}]\n')
                    self.proc = None
                    self.btn_run['state'] = 'normal'
                    self.btn_dry['state'] = 'normal'
                    self.btn_stop['state'] = 'disabled'
                else:
                    self._append(item)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log)

    def _append(self, text):
        self.log.insert('end', text)
        self.log.see('end')

    def _clear_log(self):
        self.log.delete('1.0', 'end')


def main():
    root = tk.Tk()
    SimGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
