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
        self.placeholders = {}  # dest -> Platzhaltertext (fuer None-Defaults)
        self.proc = None
        self.q = queue.Queue()
        root.title('Lens-Sensor FDTD - Einstieg')
        root.geometry('880x720')
        self._build()
        self.root.after(100, self._drain_log)

    # ---- UI-Aufbau ----
    def _build(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill='x')
        ttk.Label(top, text='FDTD-Simulation starten',
                  font=('Segoe UI', 12, 'bold')).pack(anchor='w')
        ttk.Label(top, text='Nach Sektionen gruppiert. Oben die wichtigsten '
                  'Auswahlen (Engine/Geometrie/Dimension/Methode/Material). '
                  'Leeres Feld = Default. Maus ueber den Namen = Hilfe. '
                  'Zuerst "Konfiguration pruefen (dry-run)" zeigt die aufgeloeste Config.',
                  foreground='#555', wraplength=820, justify='left').pack(anchor='w')

        # Scrollbarer Formularbereich
        mid = ttk.Frame(self.root)
        mid.pack(fill='both', expand=True, padx=8)
        canvas = tk.Canvas(mid, highlightthickness=0)
        vsb = ttk.Scrollbar(mid, orient='vertical', command=canvas.yview)
        form = ttk.Frame(canvas)
        form.bind('<Configure>',
                  lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=form, anchor='nw')
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        canvas.bind_all('<MouseWheel>',
                        lambda e: canvas.yview_scroll(int(-e.delta/120), 'units'))

        # Formular aus den argparse-GRUPPEN: je Gruppe eine Sektion mit Ueberschrift.
        row = 0
        for grp in self.parser._action_groups:
            acts = [a for a in grp._group_actions
                    if a.option_strings and a.dest != 'help']
            if not acts:
                continue
            hdr = ttk.Label(form, text='  ' + (grp.title or ''),
                            font=('Segoe UI', 10, 'bold'), foreground='#14618c')
            hdr.grid(row=row, column=0, columnspan=2, sticky='w', pady=(12, 3))
            row += 1
            for act in acts:
                name = act.option_strings[0]
                lbl = ttk.Label(form, text=name, width=20, anchor='w')
                lbl.grid(row=row, column=0, sticky='w', padx=(16, 6), pady=1)
                if act.help:
                    _Tooltip(lbl, act.help)
                if act.nargs == 0:                          # Flag -> Checkbox
                    var = tk.BooleanVar(value=bool(act.default))
                    ttk.Checkbutton(form, variable=var).grid(row=row, column=1, sticky='w')
                elif act.choices:                           # Auswahl -> Combobox
                    var = tk.StringVar(value='' if act.default is None else str(act.default))
                    ttk.Combobox(form, textvariable=var, width=28, state='readonly',
                                 values=[str(c) for c in act.choices]).grid(
                                     row=row, column=1, sticky='w')
                elif act.default is None:                   # Freitext ohne festen Default
                    ph = _auto_hint(act.dest)               # grauer Platzhalter
                    var = tk.StringVar(value=ph)
                    ent = ttk.Entry(form, textvariable=var, width=30, foreground='grey')
                    ent.grid(row=row, column=1, sticky='w')
                    self.placeholders[act.dest] = ph
                    self._bind_placeholder(ent, var, ph)
                else:                                       # Freitext mit Default
                    var = tk.StringVar(value=str(act.default))
                    ttk.Entry(form, textvariable=var, width=30).grid(
                        row=row, column=1, sticky='w')
                self.vars[act.dest] = var
                row += 1

        # Buttons
        btns = ttk.Frame(self.root, padding=8)
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

        # Log
        self.log = scrolledtext.ScrolledText(self.root, height=14, wrap='word',
                                             font=('Consolas', 9))
        self.log.pack(fill='both', expand=False, padx=8, pady=(0, 8))

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
        # Platzhaltertext == "nichts eingegeben" -> leer (Default gilt).
        return {dest: ('' if v.get() == self.placeholders.get(dest) else v.get())
                for dest, v in self.vars.items()}

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
