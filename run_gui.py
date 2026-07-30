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

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Kurzhinweise fuer Felder ohne festen Default: diese werden erst in
# run_simulation.resolve_args aus Geometrie/Dimension bzw. Materialdispersion
# aufgeloest. Als grauer Platzhalter im Feld sichtbar (nicht mitgesendet).
_AUTO_HINTS = {
    'resolution_nm': 'auto: 20 nm (2D) / 50 nm (3D)',
    'length_um': 'auto: je Geometrie & Dimension',
    'window_um': 'auto: 350 (2D) / 20 (3D)',
    'slide_um': 'auto: 150 (2D) / 12 (3D)',
}


def _auto_hint(dest):
    return _AUTO_HINTS.get(dest, 'auto (Default des Solvers)')


# Klartext-Beschriftungen (mit Einheit) statt roher CLI-Flags. Das rohe Flag und
# der Hilfetext erscheinen im Tooltip. Nicht gelistete Flags werden aus dem
# Flag-Namen abgeleitet (_pretty).
_LABELS = {
    # Fall-Auswahl
    'engine': 'Rechen-Engine', 'geometry': 'Geometrie', 'dim': 'Dimension',
    'method': 'Methode', 'scenario': 'Trockenauge-Szenario (nur Linse)',
    # Grundeinstellungen
    'length_um': 'Propagationslaenge (um)', 'resolution_nm': 'Gitteraufloesung dx (nm)',
    'lambda_nm': 'Wellenlaenge lambda (nm)', 'polarization': 'Polarisation',
    'source_type': 'Quell-Zeitverlauf', 'snapshots': 'Gespeicherte Frames (Anzahl)',
    # Material & Schichten
    'wg_material': 'Waveguide-Material', 'wg_thickness_um': 'Waveguide-Dicke (um)',
    'wg_n': 'Waveguide-Brechzahl (override)', 't_aqueous': 'Aqueous-Dicke (um)',
    't_mucin': 'Mucin-Dicke (um)', 't_lipid': 'Lipid-Dicke (um)',
    'n_aqueous': 'Aqueous-Brechzahl (override)', 'n_mucin': 'Mucin-Brechzahl (override)',
    'n_cornea': 'Cornea-Brechzahl (override)', 'n_lipid': 'Lipid-Brechzahl (override)',
    # Bead
    'bead_material': 'Bead-Material', 'bead_diameter': 'Bead-Durchmesser (um)',
    'bead_x': 'Bead-Position x (um)', 'bead_y_um': 'Bead-Position y (um)',
    'bead_n': 'Bead-Brechzahl (override)', 'no_bead': 'Ohne Bead (Referenzlauf)',
    # Quelle (VCSEL)
    'vcsel_waist': 'Strahltaille y (um)', 'vcsel_waist_z': 'Strahltaille z (um, nur 3D)',
    'vcsel_tilt': 'Strahlneigung (Grad)', 'vcsel_offset': 'Spot-Versatz y (um)',
    'vcsel_offset_z': 'Spot-Versatz z (um, nur 3D)',
    'input_gap': 'Einkoppelabstand Laser->WG (um)',
    # 3D-spezifisch
    'lz_um': 'Domaenen-Tiefe z (um, nur 3D)', 'air': 'Luft ueber WG (um, nur 3D)',
    'tear': 'Traenenfilm+Cornea unter WG (um, nur 3D)',
    'wg_width': 'Kanalbreite z (um; leer=Slab, nur 3D)',
    'wg_clad_n': 'Seitliches Cladding n (nur 3D)',
    'steps_factor': 'Zeitschritt-Faktor (nur 3D full)',
    'vol_dtype': 'Speicherformat 3D-Volumen', 'pec_faces': 'PEC-Spiegelflaechen (nur 3D full)',
    'end_facet': 'Luftzone am WG-Ende (um, nur 3D full)',
    'save_vector': 'Ex,Ey,Ez speichern (nur 3D full)',
    # Stitch / Meep
    'window_um': 'Stitch-Fensterbreite (um)', 'slide_um': 'Stitch-Schrittweite (um)',
    'meep_modes': 'Meep: gefuehrte Moden im Handoff', 'meep_handoff': 'Meep: Handoff-Verfahren',
    'meep_np': 'Meep: MPI-Prozesse (CPU-Kerne)',
    # Ablauf
    'allow_large': 'RAM-/Zellzahl-Schutz umgehen', 'no_save': 'Ergebnisse nicht speichern',
    # Kalibrierung & Vorschau
    'dry_run': 'Nur Konfiguration zeigen (Dry-Run)',
    'check_resources': 'VRAM/RAM vor Lauf pruefen', 'calibrate': 'Kalibrieren: nur N Steps messen',
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
        self.parser = rs.build_parser()
        self.vars = {}          # dest -> tk Variable
        self.widgets = {}       # dest -> Eingabe-Widget (fuer Enable/Disable)
        self.labels = {}        # dest -> Label-Widget
        self.placeholders = {}  # dest -> Platzhaltertext (fuer None-Defaults)
        self.proc = None
        self.q = queue.Queue()
        root.title('Lens-Sensor FDTD - Einstieg')
        root.geometry('880x860')
        self._build()
        self.root.after(100, self._drain_log)

    # ---- UI-Aufbau ----
    def _build(self):
        self._init_style()
        top = ttk.Frame(self.root, padding=(12, 10, 12, 6), style='Head.TFrame')
        top.pack(fill='x')
        ttk.Label(top, text='FDTD-Simulation starten', style='Title.TLabel').pack(anchor='w')
        ttk.Label(top, text='Nach Sektionen gruppiert; oben die wichtigsten Auswahlen. '
                  'Graues Feld = automatischer Default. Maus ueber die Beschriftung zeigt '
                  'CLI-Flag und Hilfe. "Konfiguration pruefen" zeigt die aufgeloeste Config, '
                  'ohne zu rechnen.',
                  style='Sub.TLabel', wraplength=840, justify='left').pack(anchor='w',
                                                                          pady=(2, 0))

        # Uebersichts-Skizze der gewaehlten Geometrie (Querschnitt x-y).
        skf = ttk.LabelFrame(self.root, text='  Uebersicht: gewaehlte Geometrie  ',
                             style='Section.TLabelframe', padding=(6, 4))
        skf.pack(fill='x', padx=10, pady=(6, 0))
        self.sketch = tk.Canvas(skf, height=165, highlightthickness=0, background='white')
        self.sketch.pack(fill='x', expand=True)

        # Scrollbarer Formularbereich
        mid = ttk.Frame(self.root)
        mid.pack(fill='both', expand=True, padx=10, pady=(6, 0))
        canvas = tk.Canvas(mid, highlightthickness=0, background=self._bg)
        vsb = ttk.Scrollbar(mid, orient='vertical', command=canvas.yview)
        form = ttk.Frame(canvas, padding=(0, 0, 8, 0))
        form.bind('<Configure>',
                  lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        win = canvas.create_window((0, 0), window=form, anchor='nw')
        # Formularbreite an die Canvas-Breite koppeln, damit die LabelFrames fuellen.
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        canvas.bind_all('<MouseWheel>',
                        lambda e: canvas.yview_scroll(int(-e.delta/120), 'units'))

        # Je argparse-Gruppe eine gerahmte, betitelte Sektion (LabelFrame).
        for grp in self.parser._action_groups:
            acts = [a for a in grp._group_actions
                    if a.option_strings and a.dest != 'help']
            if not acts:
                continue
            sec = ttk.LabelFrame(form, text='  ' + (grp.title or '') + '  ',
                                 style='Section.TLabelframe', padding=(12, 8, 12, 10))
            sec.pack(fill='x', expand=True, pady=(0, 10), padx=2)
            sec.columnconfigure(1, weight=1)
            for row, act in enumerate(acts):
                self._add_field(sec, row, act)

        # Bedingte Sichtbarkeit: Szenario nur bei Geometrie == lens aktiv.
        if 'geometry' in self.vars and 'scenario' in self.widgets:
            self.vars['geometry'].trace_add('write', self._apply_conditional)
            self._apply_conditional()

        # Uebersichts-Skizze bei Aenderung relevanter Felder neu zeichnen.
        for dest in ('geometry', 'dim', 'wg_material', 'wg_thickness_um', 'bead_diameter',
                     'bead_x', 'bead_y_um', 'no_bead', 'vcsel_offset', 'vcsel_tilt',
                     'input_gap', 'length_um', 'wg_width', 'lz_um', 'scenario'):
            v = self.vars.get(dest)
            if v is not None:
                try:
                    v.trace_add('write', self._draw_sketch)
                except Exception:
                    pass
        self.sketch.bind('<Configure>', self._draw_sketch)

        self._build_buttons_and_log()

    def _resolved_geometry(self):
        """Aufgeloeste Geometrie ('lens'/'planar') aus dem Freitext, oder None."""
        key = str(self.vars['geometry'].get()).strip().lower()
        return rs._GEOM_ALIASES.get(key)

    def _apply_conditional(self, *_):
        """Felder je nach Geometrie aktivieren/ausgrauen."""
        self._set_enabled('scenario', self._resolved_geometry() == 'lens')

    def _set_enabled(self, dest, on):
        w = self.widgets.get(dest)
        lbl = self.labels.get(dest)
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
        if geom == 'lens':
            self._sketch_lens(c, xL, xR, yT, yB)
        else:
            self._sketch_planar(c, xL, xR, yT, yB)
        # Propagationsrichtung
        c.create_line(xL, 10, xL + 44, 10, fill='#14618c', width=2, arrow='last')
        c.create_text(xL + 48, 10, anchor='w', text='x (Ausbreitung)',
                      fill='#14618c', font=('Segoe UI', 7))
        # 3D-Badge
        if dim == 3:
            lz = self._sketch_val('lz_um', 8.0)
            ww = self.vars.get('wg_width')
            is_chan = bool(ww is not None and ww.get().strip()
                           and ww.get() != self.placeholders.get('wg_width'))
            c.create_text(W - 6, H - 4, anchor='se',
                          text=f'3D · z-Tiefe {lz:g} µm · {"Kanal" if is_chan else "Slab"}',
                          fill='#14618c', font=('Segoe UI', 7, 'bold'))
        else:
            c.create_text(W - 6, H - 4, anchor='se', text='2D (z invariant)',
                          fill='#14618c', font=('Segoe UI', 7, 'bold'))

    def _sketch_planar(self, c, xL, xR, yT, yB):
        Ht = yB - yT
        bands = [('Luft (n=1.0)', '#eaf4ff', 0.17),
                 ('Lipid', '#fff2cc', 0.07),
                 ('WG-Kern', '#ffd75e', 0.17),
                 ('Traenenfilm (Aqueous)', '#bfe3ff', 0.27),
                 ('Mucin', '#e2d1f4', 0.11),
                 ('Cornea', '#f8c9c9', 0.21)]
        y = yT
        ycore = core_h = 0
        wg_mat = (self.vars['wg_material'].get() if 'wg_material' in self.vars else 'pmma')
        for name, col, frac in bands:
            h = frac*Ht
            c.create_rectangle(xL, y, xR, y + h, fill=col, outline='#cfd8e0')
            c.create_text(xR + 6, y + h/2, anchor='w', text=name, fill='#444',
                          font=('Segoe UI', 7))
            if name == 'WG-Kern':
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
        c.create_text(xa + 4, ycore + core_h + 0.09*Ht, anchor='w', text='evaneszent',
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
                cyb = (ycore + core_h + 0.05*Ht) if math.isnan(by) else (ycm - by*3.0)
                r = max(4, min(14, bd*6))
                cxb = xL + fx*(xR - xL)
                c.create_oval(cxb - r, cyb - r, cxb + r, cyb + r,
                              fill='#ff7043', outline='#a33')
                c.create_text(cxb, cyb - r - 6, text=f'Bead {bd:g}um', fill='#a33',
                              font=('Segoe UI', 6))

    def _sketch_lens(self, c, xL, xR, yT, yB):
        Ht = yB - yT
        base = yB - 0.34*Ht          # Unterkante Linse / Oberkante Traenenfilm
        # untere Schichten
        for name, col, y0, y1 in (('Traenenfilm', '#bfe3ff', base, yB - 0.18*Ht),
                                  ('Cornea', '#f8c9c9', yB - 0.18*Ht, yB)):
            c.create_rectangle(xL, y0, xR, y1, fill=col, outline='#cfd8e0')
            c.create_text(xR + 6, (y0 + y1)/2, anchor='w', text=name, fill='#444',
                          font=('Segoe UI', 7))
        # plan-konvexer Linsenkoerper
        top = yT + 0.14*Ht
        n = 48
        pts = []
        for k in range(n + 1):
            t = k/n
            x = xL + (xR - xL)*t
            y = base - (base - top)*(1 - (2*t - 1)**2)
            pts += [x, y]
        pts += [xR, base, xL, base]
        c.create_polygon(*pts, fill='#ffe08a', outline='#c9a53a', smooth=True)
        c.create_text((xL + xR)/2, (top + base)/2, text='Kontaktlinse (PMMA) ~250 um',
                      fill='#5a4600', font=('Segoe UI', 8, 'bold'))
        # Szenario (falls aktiv)
        sc = self.vars.get('scenario')
        w = self.widgets.get('scenario')
        if sc is not None and w is not None and str(w.cget('state')) != 'disabled':
            c.create_text((xL + xR)/2, top - 6, text=f'Szenario: {sc.get()}',
                          fill='#14618c', font=('Segoe UI', 7, 'bold'))
        ymid = (top + base)/2
        self._wave(c, xL + 6, xR - 6, ymid, 6)
        # Quelle
        off = self._sketch_val('vcsel_offset', 0.0)
        c.create_line(10, ymid - off*3.0, xL - 2, ymid, fill='#e23', width=3, arrow='last')
        c.create_text(10, ymid - 12, anchor='w', text='VCSEL', fill='#e23',
                      font=('Segoe UI', 7, 'bold'))
        # evaneszent in den Traenenfilm
        xa = xL + 0.6*(xR - xL)
        c.create_line(xa, base, xa, base + 0.10*Ht, fill='#0a7', width=2,
                      arrow='last', dash=(3, 2))
        c.create_text(xa + 4, base + 0.06*Ht, anchor='w', text='evaneszent',
                      fill='#0a7', font=('Segoe UI', 6))

    def _add_field(self, parent, row, act):
        """Eine Zeile: Klartext-Label (Tooltip: Flag+Hilfe) + passendes Widget."""
        flag = act.option_strings[0]
        lbl = ttk.Label(parent, text=_pretty(act.dest, flag), anchor='w')
        lbl.grid(row=row, column=0, sticky='w', padx=(2, 10), pady=3)
        tip = (flag + '\n' + act.help) if act.help else flag
        _Tooltip(lbl, tip)
        if act.nargs == 0:                              # Flag -> Checkbox
            var = tk.BooleanVar(value=bool(act.default))
            w = ttk.Checkbutton(parent, variable=var)
            w.grid(row=row, column=1, sticky='w')
        elif act.choices:                               # Auswahl -> Combobox
            var = tk.StringVar(value='' if act.default is None else str(act.default))
            w = ttk.Combobox(parent, textvariable=var, state='readonly',
                             values=[str(c) for c in act.choices])
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
        self.btn_dry = ttk.Button(btns, text='Konfiguration pruefen (dry-run)',
                                  command=self.on_dry_run)
        self.btn_dry.pack(side='left')
        self.btn_run = ttk.Button(btns, text='Simulation starten', command=self.on_run)
        self.btn_run.pack(side='left', padx=6)
        self.btn_stop = ttk.Button(btns, text='Stop', command=self.on_stop,
                                   state='disabled')
        self.btn_stop.pack(side='left')
        ttk.Button(btns, text='Analyzer oeffnen', command=self.on_analyzer).pack(
            side='left', padx=(16, 0))
        ttk.Button(btns, text='Log leeren', command=self._clear_log).pack(side='right')

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
        Ergebnis in results/ direkt; sonst nur den Datei-Dialog des Analyzers."""
        analyzer = os.path.join(_REPO_ROOT, 'common', 'fdtd_analyzer.py')
        cmd = [sys.executable, analyzer]
        frames = sorted(glob.glob(os.path.join(_REPO_ROOT, 'results', '*_frames.npz')),
                        key=os.path.getmtime)
        if frames:
            cmd.append(frames[-1])
        try:
            subprocess.Popen(cmd, cwd=_REPO_ROOT)   # nicht-blockierend, eigenes Fenster
            self._append('[Analyzer gestartet'
                         + (f': {os.path.basename(frames[-1])}' if frames
                            else ' (Datei-Dialog)') + ']\n')
        except Exception as e:
            self._append(f'[FEHLER] Analyzer-Start: {e}\n')

    def _launch(self, argv):
        if self.proc is not None:
            messagebox.showinfo('Laeuft bereits', 'Es laeuft bereits eine Rechnung.')
            return
        cmd = [sys.executable, '-u', os.path.join(_REPO_ROOT, 'run_simulation.py')] + argv
        self._append('$ python run_simulation.py ' + ' '.join(argv) + '\n')
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=_REPO_ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception as e:
            self._append(f'[FEHLER] Start fehlgeschlagen: {e}\n')
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
            self._append('[abgebrochen]\n')

    def _drain_log(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item and item[0] == '__done__':
                    self._append(f'[fertig, Exit-Code {item[1]}]\n')
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
