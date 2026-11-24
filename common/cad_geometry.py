"""Freie CAD-Geometrie fuer die FDTD-Solver.

Kette:  CAD-Datei --(Konverter)--> ein STL pro Koerper --(Voxelisierer)--> eps(x,y[,z])
        + Regionskarte (welcher Koerper liegt in welcher Zelle) + automatische Quelle.

Formate
  .stl                        direkt (ein Koerper)
  .sc .step .stp .igs .iges   ueber die SolidCraft-Kommandozeile (ein STL je Koerper)
  .3mf .obj .glb              ueber SolidCraft (Mesh-Koerper)
  .ipt .iam                   ueber Autodesk Inventor (COM) -> STEP -> SolidCraft

Koordinaten
  CAD:        X, Y, Z in mm (Z = oben)
  Simulation: x = Ausbreitung, y = Schichtstapel (oben), z = Tiefe, in um.
  Standard-Achsenzuordnung 'xzy':  x_sim = X,  y_sim = Z,  z_sim = Y.
  scale_um = um pro CAD-mm (1000 = massstabsgetreu; 1 = "1 mm im CAD = 1 um").

Materialien
  Pro Koerper ein Material aus der Dispersionstabelle (common.physics.DISPERSION)
  oder eine feste Brechzahl. Die Zuordnung liegt in '<cad-datei>.lens.json'
  (die GUI schreibt sie). Ueberlappen sich Koerper, gewinnt der KLEINERE (ein Bead
  im Film), solange keine 'priority' gesetzt ist.

Voxelisierung
  Strahlen-Paritaet entlang x: fuer jede (y, z)-Zellmitte werden die Schnittpunkte
  mit den Dreiecken bestimmt; eine Zelle ist innen, wenn links von ihr eine
  ungerade Zahl von Schnitten liegt. Die Schnittpunkte haengen nur vom (y, z)-
  Gitter ab und werden gecacht -> jedes Stitch-Fenster kostet nur noch ein
  searchsorted + kumulative Paritaet.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from common.physics import DISPERSION, n_at, parse_n  # noqa: E402

MESH_EXT = ('.stl',)
SOLIDCRAFT_EXT = ('.sc', '.solidcraft', '.step', '.stp', '.igs', '.iges', '.3mf', '.obj', '.glb')
INVENTOR_EXT = ('.ipt', '.iam')
SUPPORTED_EXT = MESH_EXT + SOLIDCRAFT_EXT + INVENTOR_EXT
MATERIALS = tuple(sorted(DISPERSION)) + ('air',)


# ---------------------------------------------------------------------------
# STL lesen
# ---------------------------------------------------------------------------
def read_stl(path):
    """STL (binaer oder ASCII) -> Dreiecke (n, 3, 3) float64 in Dateieinheiten."""
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) >= 84:
        n = int(np.frombuffer(data[80:84], dtype='<u4')[0])
        if 84 + 50*n == len(data):                       # binaer (Groesse passt exakt)
            rec = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
            arr = np.frombuffer(data, dtype=rec, count=n, offset=84)
            return arr['v'].astype(np.float64)
    verts = []
    for line in data.decode('ascii', errors='ignore').splitlines():
        s = line.strip()
        if s.startswith('vertex'):
            verts.append([float(t) for t in s.split()[1:4]])
    v = np.asarray(verts, dtype=np.float64)
    if len(v) == 0 or len(v) % 3:
        raise ValueError(f'STL not readable or empty: {path}')
    return v.reshape(-1, 3, 3)


# ---------------------------------------------------------------------------
# Konverter: CAD-Datei -> {Koerpername: STL-Pfad}
# ---------------------------------------------------------------------------
def _solidcraft_cli():
    env = os.environ.get('LENS_SOLIDCRAFT_CLI')
    cands = [env] if env else []
    cands += [shutil.which('solidcraft-cli') or '',
              r'C:\Program Files\SolidCraft\solidcraft-cli.exe']
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise RuntimeError('SolidCraft command line (solidcraft-cli) not found. '
                       'Set the path via the environment variable LENS_SOLIDCRAFT_CLI.')


def _cache_dir(src):
    st = os.stat(src)
    key = hashlib.sha1(f'{os.path.abspath(src)}|{st.st_size}|{st.st_mtime_ns}'.encode()).hexdigest()[:16]
    d = os.path.join(tempfile.gettempdir(), 'lens_sensor_cad', key)
    os.makedirs(d, exist_ok=True)
    return d


def _safe(name):
    return ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in name) or 'body'


def _solidcraft_bodies(src, outdir, quality='fine'):
    """Alle Koerper einer von SolidCraft lesbaren Datei als je ein STL exportieren."""
    cli = _solidcraft_cli()
    info_json = os.path.join(outdir, 'bodies.json')
    r = subprocess.run([cli, 'eval', '--in', src, '--json-out', info_json],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(info_json):
        raise RuntimeError(f'SolidCraft could not read {os.path.basename(src)}:\n{r.stderr[-2000:]}')
    with open(info_json, encoding='utf-8') as f:
        info = json.load(f)
    names = [b.get('name') for b in info.get('bodies', []) if b.get('closed', True)]
    names = [n for n in names if n]
    if not names:
        raise RuntimeError(f'No closed bodies in {os.path.basename(src)}.')
    out, cmds = {}, []
    for nm in names:
        p = os.path.join(outdir, _safe(nm) + '.stl')
        out[nm] = p
        cmds.append({'command': 'file.export', 'params': {'path': p, 'bodies': [nm]}})
    script = os.path.join(outdir, 'export.json')
    with open(script, 'w', encoding='utf-8') as f:
        json.dump({'commands': cmds}, f)
    r = subprocess.run([cli, 'run', script, '--in', src, '--quality', quality, '--quiet'],
                       capture_output=True, text=True)
    missing = [n for n, p in out.items() if not os.path.exists(p)]
    if r.returncode != 0 or missing:
        raise RuntimeError(f'SolidCraft export failed ({missing}):\n{r.stderr[-2000:]}')
    return out


def _inventor_to_step(src, out_step):
    """Inventor-Datei (.ipt/.iam) per COM als STEP speichern. Startet Inventor
    unsichtbar, falls es nicht laeuft, und beendet es danach wieder."""
    try:
        import pythoncom
        import win32com.client
    except ImportError as e:
        raise RuntimeError('Inventor import requires pywin32 (pip install pywin32).') from e
    pythoncom.CoInitialize()
    started = False
    try:
        try:
            app = win32com.client.GetActiveObject('Inventor.Application')
        except Exception:
            app = win32com.client.Dispatch('Inventor.Application')
            started = True
            app.Visible = False
        app.SilentOperation = True
        doc = app.Documents.Open(os.path.abspath(src), False)
        try:
            doc.SaveAs(os.path.abspath(out_step), True)      # True = Kopie; Format aus der Endung
        finally:
            doc.Close(True)
        if started:
            app.Quit()
    finally:
        pythoncom.CoUninitialize()
    if not os.path.exists(out_step):
        raise RuntimeError(f'Inventor did not write a STEP file: {out_step}')
    return out_step


def convert_to_stl(src, quality='fine'):
    """CAD-Datei -> {Koerpername: STL-Pfad} (gecacht, solange sich die Datei nicht aendert)."""
    src = os.path.abspath(src)
    ext = os.path.splitext(src)[1].lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f'Format {ext} not supported. Allowed: {", ".join(SUPPORTED_EXT)}')
    if ext in MESH_EXT:
        return {os.path.splitext(os.path.basename(src))[0]: src}
    outdir = _cache_dir(src)
    index = os.path.join(outdir, 'index.json')
    if os.path.exists(index):
        with open(index, encoding='utf-8') as f:
            cached = json.load(f)
        if all(os.path.exists(p) for p in cached.values()):
            return cached
    if ext in INVENTOR_EXT:
        step = _inventor_to_step(src, os.path.join(outdir, 'inventor.step'))
        out = _solidcraft_bodies(step, outdir, quality)
    else:
        out = _solidcraft_bodies(src, outdir, quality)
    with open(index, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    return out


# ---------------------------------------------------------------------------
# Geometrie-Helfer
# ---------------------------------------------------------------------------
def _signed_volume(tris):
    return float(np.einsum('ij,ij->i', tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum()/6.0)


def _ray_hits(tris, ys, zs):
    """Schnittpunkte der Strahlen parallel zu x durch (ys[j], zs[k]) mit den
    Dreiecken. Rueckgabe (J, K, X). ys/zs sind leicht verschoben, damit kein
    Strahl exakt eine Kante oder Ecke trifft (sonst doppelte Zaehlung)."""
    ys = np.asarray(ys, dtype=np.float64); zs = np.asarray(zs, dtype=np.float64)
    J, K, X = [], [], []
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    ymin = tris[:, :, 1].min(1); ymax = tris[:, :, 1].max(1)
    zmin = tris[:, :, 2].min(1); zmax = tris[:, :, 2].max(1)
    j0 = np.searchsorted(ys, ymin); j1 = np.searchsorted(ys, ymax, side='right')
    k0 = np.searchsorted(zs, zmin); k1 = np.searchsorted(zs, zmax, side='right')
    # Projektion auf (y, z): Determinante; ~0 = Dreieck parallel zu x -> kein Schnitt
    e1 = v1 - v0; e2 = v2 - v0
    det = e1[:, 1]*e2[:, 2] - e1[:, 2]*e2[:, 1]
    ok = (np.abs(det) > 1e-18) & (j1 > j0) & (k1 > k0)
    for t in np.nonzero(ok)[0]:
        jj = np.arange(j0[t], j1[t]); kk = np.arange(k0[t], k1[t])
        Y, Z = np.meshgrid(ys[jj], zs[kk], indexing='ij')
        py = Y - v0[t, 1]; pz = Z - v0[t, 2]
        u = (py*e2[t, 2] - pz*e2[t, 1])/det[t]
        v = (e1[t, 1]*pz - e1[t, 2]*py)/det[t]
        m = (u >= 0) & (v >= 0) & (u + v <= 1)
        if not m.any():
            continue
        x = v0[t, 0] + u[m]*e1[t, 0] + v[m]*e2[t, 0]
        jg, kg = np.meshgrid(jj, kk, indexing='ij')
        J.append(jg[m]); K.append(kg[m]); X.append(x)
    if not J:
        e = np.zeros(0, dtype=np.int64)
        return e, e, np.zeros(0)
    return np.concatenate(J), np.concatenate(K), np.concatenate(X)


def _occupancy(hits, xs, ny, nz):
    """Paritaet entlang x: bool (Nx, Ny, Nz), innen fuer die Zellmitten xs."""
    J, K, X = hits
    nx = len(xs)
    tog = np.zeros((nx + 1, ny, nz), dtype=np.uint8)
    if len(X):
        i = np.searchsorted(xs, X)                  # Zellen ab i liegen rechts vom Schnitt
        np.add.at(tog, (i, J, K), 1)
    return (np.cumsum(tog[:nx], axis=0, dtype=np.uint8) & 1).astype(bool)


# ---------------------------------------------------------------------------
# Szene
# ---------------------------------------------------------------------------
class CadScene:
    """CAD-Koerper in Simulationskoordinaten (um) mit Materialzuordnung."""

    AXES = {'xzy': (0, 2, 1), 'xyz': (0, 1, 2)}

    def __init__(self, path, scale_um=1000.0, axis='xzy', materials=None,
                 background='air', priority=None, core=None, tear=None, quality='fine',
                 open_faces=None, labels=None):
        self.path = os.path.abspath(path)
        self.scale_um = float(scale_um)
        self.axis = axis
        self.background = background
        self.materials = dict(materials or {})
        self.priority = dict(priority or {})
        self.core = core
        self.tear = tear
        # Offene Flaechen: [{'body', 'axis' (0/1/2), 'sign' (+1/-1), 'pos' (um, Sim.-Koord.
        # ohne Verschiebung)}] -> Querschnitt wird ueber die Flaeche hinaus bis zum Rand
        # fortgesetzt (keine Facette, keine Reflexion; das Bauteil 'laeuft weiter').
        # Jede Flaeche merkt sich Massstab/Achsen-Zuordnung, mit denen 'pos'/'axis' gelten;
        # weichen sie ab (--cad-scale/--cad-axis oder GUI geaendert), wird umgerechnet
        # statt eine veraltete Ebene (falscher Ort/falsche Achse) zu extrudieren.
        self.open_faces = [self._convert_face(f) for f in (open_faces or [])]
        # Anzeigenamen je Koerper (frei vergebbar; Schluessel bleibt der CAD-Name)
        self.labels = {k: str(v) for k, v in dict(labels or {}).items() if str(v).strip()}
        stls = convert_to_stl(self.path, quality=quality)
        perm = self.AXES[axis]
        self.bodies = {}
        for name, p in stls.items():
            t = read_stl(p)[:, :, perm]*self.scale_um
            if _signed_volume(t) < 0:                      # nach aussen orientieren
                t = t[:, ::-1, :]
            self.bodies[name] = t
        self.volumes = {n: abs(_signed_volume(t)) for n, t in self.bodies.items()}
        self.shift = np.zeros(3)                           # Verschiebung in die Rechenbox
        self._hit_cache = {}

    def _convert_face(self, f):
        """Offene Flaeche auf den aktuellen Massstab/die aktuelle Achsen-Zuordnung bringen
        (sim-Achse ax <-> CAD-Achse AXES[axis][ax]; pos skaliert mit scale_um)."""
        f = dict(f)
        ra = f.get('ref_axis', self.axis)
        rs = float(f.get('ref_scale', self.scale_um) or self.scale_um)
        if ra != self.axis and ra in self.AXES:
            f['axis'] = self.AXES[self.axis].index(self.AXES[ra][int(f['axis'])])
        if rs > 0 and abs(rs - self.scale_um) > 1e-12:
            f['pos'] = float(f['pos'])*self.scale_um/rs
        f['ref_axis'] = self.axis
        f['ref_scale'] = self.scale_um
        return f

    # ---- Konfiguration (GUI <-> Datei) ----
    @staticmethod
    def config_path(path):
        return os.path.abspath(path) + '.lens.json'

    @classmethod
    def from_config(cls, path, **override):
        cfg = {}
        cp = cls.config_path(path)
        if os.path.exists(cp):
            with open(cp, encoding='utf-8') as f:
                cfg = json.load(f)
        # Flaechen ohne Bezug (aeltere Dateien) gelten im Massstab/der Achse der DATEI -
        # vor dem Ueberschreiben durch --cad-scale/--cad-axis festhalten
        _fs, _fa = cfg.get('scale_um', 1000.0), cfg.get('axis', 'xzy')
        if cfg.get('open_faces'):
            cfg['open_faces'] = [dict(f, ref_scale=f.get('ref_scale', _fs),
                                      ref_axis=f.get('ref_axis', _fa)) for f in cfg['open_faces']]
        cfg.update({k: v for k, v in override.items() if v is not None})
        keys = ('scale_um', 'axis', 'materials', 'background', 'priority', 'core', 'tear',
                'quality', 'open_faces', 'labels')
        return cls(path, **{k: cfg[k] for k in keys if k in cfg})

    def save_config(self):
        cfg = dict(scale_um=self.scale_um, axis=self.axis, materials=self.materials,
                   background=self.background, priority=self.priority,
                   core=self.core, tear=self.tear, open_faces=self.open_faces,
                   labels=self.labels)
        with open(self.config_path(self.path), 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return self.config_path(self.path)

    # ---- Materialien ----
    @staticmethod
    def _nc(spec, lam_nm):
        """Brechzahl, ggf. KOMPLEX (Absorption): 'pmma', 'n=1.52', '1.52',
        'n=1.50+0.002i' (feste komplexe Zahl) oder 'aqueous+0.001i' (Dispersionsmaterial
        plus Extinktion kappa)."""
        if spec is None:
            return None
        if isinstance(spec, (int, float, complex)):
            return parse_n(spec)
        s = str(spec).strip().lower().replace(' ', '')
        if s in ('air', 'luft'):
            return 1.0
        if s.startswith('n='):
            return parse_n(s[2:])
        try:
            return parse_n(s)
        except ValueError:
            pass
        kappa = 0.0
        if '+' in s and s[-1] in 'ij':                 # 'material+kappa i'
            s, kk = s.split('+', 1)
            kappa = float(kk[:-1])
        if s not in DISPERSION:
            raise ValueError(f'Unknown material "{spec}". Allowed: {", ".join(MATERIALS)}, '
                             f'n=<number>, n=<n>+<kappa>i or <material>+<kappa>i')
        n = n_at(s, lam_nm)
        return parse_n(complex(n, kappa)) if kappa else n

    @staticmethod
    def _n(spec, lam_nm):
        """Realteil der Brechzahl (Quelle, Kern-Wahl, Ausgaben)."""
        v = CadScene._nc(spec, lam_nm)
        return v.real if isinstance(v, complex) else v

    def n_of(self, body, lam_nm):
        return self._n(self.materials.get(body), lam_nm)

    def nc_of(self, body, lam_nm):
        return self._nc(self.materials.get(body), lam_nm)

    def unassigned(self):
        return [b for b in self.bodies if self.materials.get(b) in (None, '')]

    def body_order(self):
        """Malreihenfolge: hohe 'priority' zuletzt, sonst grosse Koerper zuerst."""
        return sorted(self.bodies, key=lambda b: (self.priority.get(b, 0), -self.volumes[b]))

    def core_body(self, lam_nm):
        """Kern = explizit gesetzt, sonst der hoechstbrechende Koerper unter denen, die
        mindestens die halbe Laenge (x) der Szene ueberspannen (ein Bead oder Prisma
        mit hohem n ist kein Wellenleiter)."""
        if self.core in self.bodies:
            return self.core
        if self.core:                       # gesetzt, aber unbekannt (Tippfehler) -> nicht still raten
            raise ValueError(f'--cad-core "{self.core}": body does not exist. '
                             f'Available: {", ".join(self.bodies)}')
        lo, hi = self.bbox()
        span = hi[0] - lo[0]
        longs = [b for b in self.bodies
                 if (self.bbox(b)[1][0] - self.bbox(b)[0][0]) >= 0.5*span] or list(self.bodies)
        return max(longs, key=lambda b: (self.n_of(b, lam_nm) or 0.0, self.volumes[b]))

    # ---- Geometrie ----
    def bbox(self, body=None):
        names = [body] if body else list(self.bodies)
        pts = np.concatenate([self.bodies[b].reshape(-1, 3) for b in names]) + self.shift
        return pts.min(0), pts.max(0)

    def set_shift(self, shift):
        self.shift = np.asarray(shift, dtype=np.float64)
        self._hit_cache.clear()

    def _hits(self, body, ys, zs):
        key = (body, len(ys), float(ys[0]), float(ys[-1]), len(zs), float(zs[0]), float(zs[-1]))
        if key not in self._hit_cache:
            t = self.bodies[body] + self.shift
            self._hit_cache[key] = _ray_hits(t, ys, zs)
        return self._hit_cache[key]

    def eps_region(self, xs, ys, zs, lam_nm):
        """eps (Nx,Ny,Nz) float32 und Regionskarte (Nx,Ny,Nz) int8 an den Zellmitten
        (xs, ys, zs in um, Simulationskoordinaten). Region 0 = Hintergrund, sonst
        1 + Index in self.region_names()."""
        xs = np.asarray(xs, dtype=np.float64)
        # minimale Verschiebung, damit kein Strahl exakt eine Kante trifft
        h = (ys[1] - ys[0]) if len(ys) > 1 else 1e-3
        ys_q = np.asarray(ys, dtype=np.float64) + 1.37e-7*h
        zs_q = np.asarray(zs, dtype=np.float64) + 2.71e-7*h
        nb = self._nc(self.background, lam_nm) or 1.0
        lossy = isinstance(nb, complex) or any(isinstance(self.nc_of(b, lam_nm), complex)
                                               for b in self.bodies)
        # Absorption: komplexes eps (n' + i kappa)^2; der Solver trennt eps' / sigma
        eps = np.full((len(xs), len(ys), len(zs)), nb*nb,
                      dtype=np.complex64 if lossy else np.float32)
        reg = np.zeros(eps.shape, dtype=np.int8)
        names = self.region_names()
        for b in self.body_order():
            n = self.nc_of(b, lam_nm)
            if n is None:
                raise ValueError(f'Body "{b}" has no material. Assign one in the GUI or in '
                                 f'{os.path.basename(self.config_path(self.path))}.')
            occ = _occupancy(self._hits(b, ys_q, zs_q), xs, len(ys), len(zs))
            eps[occ] = n*n
            reg[occ] = names.index(b) + 1
            # offene Flaechen dieses Koerpers: letzten Querschnitt bis zum Rand fortsetzen
            for f in self.open_faces:
                if f.get('body') != b:
                    continue
                ext = self._extrusion(b, f, xs, ys_q, zs_q)
                if ext is not None:
                    eps[ext] = n*n
                    reg[ext] = names.index(b) + 1
        return eps, reg

    def region_names(self):
        return list(self.bodies)

    def display_name(self, body):
        """Vom Nutzer vergebener Anzeigename (Default: Name aus der CAD-Datei)."""
        return self.labels.get(body) or body

    # ---- automatische Quelle ----
    def _extrusion(self, body, f, xs, ys_q, zs_q):
        """Maske der Zellen jenseits einer offenen Flaeche, die der fortgesetzte
        Querschnitt (Schnitt knapp innerhalb der Flaeche) belegt."""
        ax, sg = int(f['axis']), int(f['sign'])
        p0 = float(f['pos']) + float(self.shift[ax])
        h = (ys_q[1] - ys_q[0]) if len(ys_q) > 1 else 1e-3
        inner = p0 - sg*0.5*h                           # knapp innerhalb der Flaeche
        if ax == 0:
            sec = _occupancy(self._hits(body, ys_q, zs_q), np.array([inner]), len(ys_q), len(zs_q))
            beyond = (sg*(xs - p0) > -1e-6*h)[:, None, None]    # inkl. Zellen AUF der Ebene
            return beyond & sec                          # (Nx,Ny,Nz) per Broadcast
        if ax == 1:
            sec = _occupancy(_ray_hits(self.bodies[body] + self.shift, np.array([inner]), zs_q),
                             xs, 1, len(zs_q))           # (Nx,1,Nz)
            beyond = (sg*(np.asarray(ys_q) - p0) > -1e-6*h)[None, :, None]
            return beyond & sec
        sec = _occupancy(_ray_hits(self.bodies[body] + self.shift, ys_q, np.array([inner])),
                         xs, len(ys_q), 1)               # (Nx,Ny,1)
        beyond = (sg*(np.asarray(zs_q) - p0) > -1e-6*h)[None, None, :]
        return beyond & sec

    def planar_faces(self, min_area_frac=1e-3, axis_tol_deg=25.0):
        """Ebene Flaechen aller Koerper: [{'body', 'normal', 'center', 'area', 'axis',
        'sign', 'pos', 'lo', 'hi'}] (Sim.-Koordinaten mit Verschiebung). 'axis' nur bei
        ungefaehr achsparalleler Normale (sonst None -> nicht oeffenbar)."""
        out = []
        cos_t = np.cos(np.deg2rad(axis_tol_deg))
        for b, t0 in self.bodies.items():
            t = t0 + self.shift
            nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            a2 = np.linalg.norm(nrm, axis=1)
            ok = a2 > 0
            un = np.zeros_like(nrm); un[ok] = nrm[ok]/a2[ok, None]
            cen = t.mean(1)
            lo, hi = self.bbox(b)
            sc = max(hi - lo)
            groups = {}
            for i in np.nonzero(ok)[0]:
                key = tuple(np.round(un[i], 2)) + (round(float(np.dot(un[i], cen[i]))/sc, 3),)
                groups.setdefault(key, []).append(i)
            tot = a2.sum()/2
            for ix in groups.values():
                a = a2[ix]/2
                if a.sum() < min_area_frac*tot:
                    continue
                c = (cen[ix]*a[:, None]).sum(0)/a.sum()
                nv = (un[ix]*a[:, None]).sum(0); nv /= np.linalg.norm(nv)
                k = int(np.argmax(np.abs(nv)))
                axis = k if abs(nv[k]) >= cos_t else None
                pts = t[ix].reshape(-1, 3)
                out.append(dict(body=b, normal=nv, center=c, area=float(a.sum()),
                                axis=axis, sign=(int(np.sign(nv[k])) if axis is not None else 0),
                                pos=(float(c[k] - self.shift[k]) if axis is not None else None),
                                lo=pts.min(0), hi=pts.max(0)))
        return out

    def open_by_spec(self, spec, lam_nm=850.0):
        """'Kern:+x,Film:+x' -> diese Flaechen oeffnen (je Koerper die aeusserste Flaeche
        mit dieser Normalenrichtung). 'Koerper:-x' usw.; Achsen x/y/z (Sim.-Koordinaten)."""
        faces = self.planar_faces()
        ent = self.entrance_facet(lam_nm)
        for item in [t.strip() for t in str(spec).split(',') if t.strip()]:
            body, _, d = item.rpartition(':')
            if body not in self.bodies or len(d) != 2 or d[0] not in '+-' or d[1] not in 'xyz':
                raise ValueError(f'--cad-open: "{item}" invalid (format body:+x; is the body known?)')
            ax, sg = 'xyz'.index(d[1]), (1 if d[0] == '+' else -1)
            cand = [f for f in faces if f['body'] == body and f['axis'] == ax and f['sign'] == sg]
            if not cand:
                raise ValueError(f'--cad-open: body "{body}" has no planar face with normal {d}.')
            f = max(cand, key=lambda q: sg*q['pos'])
            if self.is_entrance(f, ent):
                raise ValueError('The entry face of the core cannot be open (the source sits there).')
            if not self.is_open(f):
                self.toggle_open(f)

    def is_entrance(self, face, ent=None, lam_nm=850.0):
        """Ist diese ebene Flaeche die Eintrittsflaeche des Kerns? Vergleich ueber Koerper,
        Achse, Richtung und Lage (Toleranz ~Rundung der Flaechengruppierung) statt ueber
        die Mittelpunkte: planar_faces und entrance_facet gruppieren unterschiedlich fein,
        die flaechengewichteten Zentren stimmen dann nicht auf 1e-6 ueberein."""
        ent = ent or self.entrance_facet(lam_nm)
        if face.get('body') != ent['core'] or face.get('axis') is None:
            return False
        nv = np.asarray(ent['normal'], dtype=np.float64)
        k = int(np.argmax(np.abs(nv)))
        if int(face['axis']) != k or int(face['sign']) != int(np.sign(nv[k])):
            return False
        lo, hi = self.bbox(ent['core'])
        tol = max(2e-3*float(np.max(hi - lo)), 1e-6)
        epos = float(ent['center'][k] - self.shift[k])
        return abs(float(face['pos']) - epos) <= tol

    def is_open(self, face):
        return any(f.get('body') == face['body'] and f.get('axis') == face['axis']
                   and f.get('sign') == face['sign'] and abs(f.get('pos', 1e9) - face['pos']) < 1e-6
                   for f in self.open_faces)

    def toggle_open(self, face):
        """Flaeche offen/reflektierend umschalten; True = jetzt offen."""
        if face.get('axis') is None:
            raise ValueError('Only approximately axis-parallel faces can be opened.')
        if self.is_open(face):
            self.open_faces = [f for f in self.open_faces
                               if not (f.get('body') == face['body'] and f.get('axis') == face['axis']
                                       and f.get('sign') == face['sign']
                                       and abs(f.get('pos', 1e9) - face['pos']) < 1e-6)]
            return False
        self.open_faces.append(dict(body=face['body'], axis=int(face['axis']),
                                    sign=int(face['sign']), pos=float(face['pos']),
                                    ref_scale=self.scale_um, ref_axis=self.axis))
        return True

    def entrance_facet(self, lam_nm, max_tilt_deg=60.0):
        """Groesste ebene Flaeche des Kern-Koerpers, die zur Quellseite (-x) zeigt,
        im vorderen Viertel des Kerns. Rueckgabe dict(center, normal, dir, area,
        bbox_min, bbox_max, core)."""
        core = self.core_body(lam_nm)
        t = self.bodies[core] + self.shift
        nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        area2 = np.linalg.norm(nrm, axis=1)
        good = area2 > 0
        un = np.zeros_like(nrm); un[good] = nrm[good]/area2[good, None]
        cen = t.mean(1)
        lo, hi = self.bbox(core)
        cos_lim = np.cos(np.deg2rad(max_tilt_deg))
        cand = good & (-un[:, 0] >= cos_lim) & (cen[:, 0] <= lo[0] + 0.25*(hi[0] - lo[0]))
        if not cand.any():
            raise ValueError(f'No face facing the source side (-x) found on core "{core}".')
        # in Ebenen gruppieren: gleiche Normale + gleicher Abstand
        d = np.einsum('ij,ij->i', un, cen)
        scale = max(hi - lo)
        keys = {}
        for i in np.nonzero(cand)[0]:
            k = tuple(np.round(un[i], 3)) + (round(d[i]/scale, 4),)
            keys.setdefault(k, []).append(i)
        best = max(keys.values(), key=lambda ix: area2[ix].sum())
        a = area2[best]/2.0
        c = (cen[best]*a[:, None]).sum(0)/a.sum()
        n = (un[best]*a[:, None]).sum(0); n /= np.linalg.norm(n)
        pts = t[best].reshape(-1, 3)
        return dict(core=core, center=c, normal=n, dir=-n, area=float(a.sum()),
                    bbox_min=pts.min(0), bbox_max=pts.max(0))

    def summary(self, lam_nm=850.0):
        lines = []
        for b in self.body_order():
            lo, hi = self.bbox(b)
            n = self.n_of(b, lam_nm)
            nc = self.nc_of(b, lam_nm)
            _k = f'+{nc.imag:.3g}i' if isinstance(nc, complex) else ''
            _d = b if b not in self.labels else f'{b} "{self.labels[b]}"'
            lines.append(f'  {_d:<20} {str(self.materials.get(b, "-")):<12} '
                         f'n={("%.3f" % n) if n else "  ?  "}{_k}  '
                         f'x {lo[0]:8.2f}..{hi[0]:8.2f}  y {lo[1]:7.2f}..{hi[1]:7.2f}  '
                         f'z {lo[2]:7.2f}..{hi[2]:7.2f} um')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Rechenbox + Quelle + Detektoren fuer einen Lauf
# ---------------------------------------------------------------------------
class CadRun:
    """Legt aus einer CadScene die Rechenbox fuer einen 2D- oder 3D-Lauf fest.

    Die Szene wird so verschoben, dass die Box bei x = 0 beginnt (die Solver
    zaehlen x ab 0); y und z bleiben in CAD-Koordinaten. Quelle: Gauss-Spot vor der
    Eintrittsflaeche des Kerns, Strahl entlang deren Innennormale. Detektoren: Kern-
    Querschnitt bei 10 % und 90 % seiner Laenge."""

    def __init__(self, scene, dim, dx_um, lam_nm, plane_z_um=None,
                 margin_front_um=2.0, margin_back_um=1.0, margin_y_um=1.5,
                 margin_z_um=1.0, absorber_um=0.0):
        self.scene = scene
        self.dim = int(dim)
        self.dx_um = float(dx_um)
        self.lam_nm = float(lam_nm)
        scene.set_shift(np.zeros(3))
        lo, hi = scene.bbox()
        a = float(absorber_um)
        x0 = lo[0] - margin_front_um - a
        scene.set_shift([-x0, 0.0, 0.0])
        lo, hi = scene.bbox()
        self.lx_um = hi[0] + margin_back_um + a
        self.y0_um = lo[1] - margin_y_um - a
        self.ly_um = (hi[1] + margin_y_um + a) - self.y0_um
        if self.dim == 3:
            self.z0_um = lo[2] - margin_z_um - a
            self.lz_um = (hi[2] + margin_z_um + a) - self.z0_um
        else:
            core = scene.core_body(lam_nm)
            clo, chi = scene.bbox(core)
            self.z0_um = 0.5*(clo[2] + chi[2]) if plane_z_um is None else float(plane_z_um)
            self.lz_um = 0.0
        self.x_shift_um = -x0
        self.facet = scene.entrance_facet(lam_nm)
        self.core = self.facet['core']
        self.core_index = scene.region_names().index(self.core) + 1
        if scene.tear and scene.tear not in scene.bodies:
            raise ValueError(f'--cad-tear "{scene.tear}": body does not exist. '
                             f'Available: {", ".join(scene.bodies)}')
        tear = scene.tear if scene.tear in scene.bodies else None
        self.tear_index = (scene.region_names().index(tear) + 1) if tear else 0

    # ---- Gitter ----
    def axes(self, x_start_um, nx, ny, nz=1):
        dx = self.dx_um
        xs = x_start_um + (np.arange(nx) + 0.0)*dx
        ys = self.y0_um + np.arange(ny)*dx
        zs = (self.z0_um + np.arange(nz)*dx) if self.dim == 3 else np.array([self.z0_um])
        return xs, ys, zs

    # Subpixel-Glaettung der Koerperoberflaechen: S^d Stichproben je Zelle, eps = Zellmittel
    # (Volumenanteil, arithmetisch). Region = Stichprobe nahe der Zellmitte. S=1 -> hart.
    smooth_2d = 4
    smooth_3d = 2

    def eps2d(self, x_start_um, nx, ny, smooth=None):
        from common.subpixel import fine_axis, block_mean
        S = self.smooth_2d if smooth is None else int(smooth)
        if S <= 1:
            xs, ys, zs = self.axes(x_start_um, nx, ny)
            eps, reg = self.scene.eps_region(xs, ys, zs, self.lam_nm)
            return eps[:, :, 0], reg[:, :, 0]
        xs = fine_axis(x_start_um, nx, self.dx_um, S)
        ys = fine_axis(self.y0_um, ny, self.dx_um, S)
        eps, reg = self.scene.eps_region(xs, ys, np.array([self.z0_um]), self.lam_nm)
        k = S//2
        return (block_mean(eps[:, :, 0], S).astype(np.complex64 if np.iscomplexobj(eps) else np.float32),
                np.ascontiguousarray(reg[k::S, k::S, 0]))

    def eps2d_ey(self, x_start_um, nx, ny, smooth=None):
        """TM (p): eps an den Ey-Positionen (i, j+1/2), shape (nx, ny-1). Ey steht auf
        Grenzflaechen normal zu y senkrecht -> HARMONISCH ueber die Zellhoehe [y_j, y_j+1]
        mitteln, arithmetisch in x (wie der eingebaute Schichtstapel). eps2d (arithmetisch
        an den Knoten) setzt Grenzflaechen fuer Ey sonst mit einem Fehler 1. Ordnung."""
        from common.subpixel import fine_axis
        S = max(2, self.smooth_2d if smooth is None else int(smooth))
        dx = self.dx_um
        xs = fine_axis(x_start_um, nx, dx, S)
        ys = fine_axis(self.y0_um + 0.5*dx, ny - 1, dx, S)
        eps, _ = self.scene.eps_region(xs, ys, np.array([self.z0_um]), self.lam_nm)
        e = np.maximum(np.real(eps[:, :, 0]).astype(np.float64), 1e-6).reshape(nx, S, ny - 1, S)
        e_y = 1.0/np.mean(1.0/e, axis=3)                  # harmonisch in y
        return np.mean(e_y, axis=1).astype(np.float32)    # arithmetisch in x

    def eps3d(self, x_start_um, nx, ny, nz, smooth=None):
        from common.subpixel import fine_axis, block_mean
        S = self.smooth_3d if smooth is None else int(smooth)
        if S <= 1:
            xs, ys, zs = self.axes(x_start_um, nx, ny, nz)
            return self.scene.eps_region(xs, ys, zs, self.lam_nm)
        xs = fine_axis(x_start_um, nx, self.dx_um, S)
        ys = fine_axis(self.y0_um, ny, self.dx_um, S)
        zs = fine_axis(self.z0_um, nz, self.dx_um, S)
        eps, reg = self.scene.eps_region(xs, ys, zs, self.lam_nm)
        k = S//2
        return (block_mean(eps, S).astype(np.complex64 if np.iscomplexobj(eps) else np.float32),
                np.ascontiguousarray(reg[k::S, k::S, k::S]))

    # ---- Quelle ----
    def source(self, x_src_um):
        """Quellzentrum (y, z) an der Quellebene x_src und Neigung in der x-y-Ebene,
        so dass der Strahl die Facettenmitte entlang der Innennormale trifft.
        tilt_deg folgt der Solver-Konvention (Phase +k0 sin(t) y -> Strahl nach -y)."""
        c = self.facet['center']; d = self.facet['dir']
        th_xy = np.arctan2(d[1], d[0])
        th_xz = np.arctan2(d[2], d[0])
        back = (c[0] - x_src_um)/max(d[0], 1e-9)            # Laufweg Quelle -> Facette
        y_src = c[1] - back*d[1]
        z_src = (c[2] - back*d[2]) if self.dim == 3 else self.z0_um
        return dict(y_um=float(y_src), z_um=float(z_src),
                    tilt_deg=float(-np.degrees(th_xy)),
                    tilt_z_deg=float(np.degrees(th_xz)),
                    facet_center=c.tolist(), facet_normal=self.facet['normal'].tolist())

    def core_k(self, nz):
        """3D: z-Index der Kernmitte (fuer den Regionsschnitt der Detektoren). Die Box
        umfasst die ganze Szene in z; ein Kern abseits der Szenenmitte (oder schmaler als
        ein versetzter Film) wird von der z-Mittelebene verfehlt."""
        if self.dim != 3:
            return 0
        clo, chi = self.scene.bbox(self.core)
        k = int(round((0.5*(clo[2] + chi[2]) - self.z0_um)/self.dx_um))
        return min(max(k, 0), int(nz) - 1)

    # ---- Detektoren ----
    def detectors(self, region_xy, x_start_um=0.0, frac=(0.1, 0.9), width_frac=0.1):
        """Detektor-Fenster am Kern: (ix0, ix1, iy0, iy1) fuer Eingang und Ausgang
        aus der Regionskarte (Nx, Ny) eines 2D-Schnitts bzw. (3D) des Schnitts durch
        die Kernmitte (core_k). Liegt in der Mittelspalte kein Kern (S-Kurve, anderer
        Koerper darueber), werden Nachbarspalten im Fenster abgesucht; die y-Zeilen sind
        der zusammenhaengende Kernabschnitt nahe der Kernmitte (nicht min..max ueber
        Luecken). Ohne Treffer: Warnung + Kern-bbox in y statt stiller Vollhoehe."""
        lo, hi = self.scene.bbox(self.core)
        L = hi[0] - lo[0]
        dx = self.dx_um
        ny = region_xy.shape[1]
        jc_exp = 0.5*(lo[1] + hi[1])                       # erwartete Kernmitte (y, um)
        out = []
        for f in frac:
            xc = lo[0] + f*L
            hw = max(2, int(round(0.5*width_frac*L/dx)))
            c = int(round((xc - x_start_um)/dx))
            ix0, ix1 = max(0, c - hw), min(region_xy.shape[0], c + hw)
            rows = np.zeros(0, dtype=int)
            for dc in sorted(range(-hw, hw + 1), key=abs):  # Mitte zuerst, dann Nachbarn
                cc = min(max(c + dc, 0), region_xy.shape[0] - 1)
                rows = np.nonzero(region_xy[cc] == self.core_index)[0]
                if len(rows):
                    break
            if len(rows):
                # zusammenhaengende Abschnitte -> der Kernmitte naechste
                runs = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
                jexp = (jc_exp - self.y0_um)/dx
                run = min(runs, key=lambda r: abs(0.5*(r[0] + r[-1]) - jexp))
                iy0, iy1 = int(run[0]), int(run[-1]) + 1
            else:
                iy0 = max(0, int(np.floor((lo[1] - self.y0_um)/dx)))
                iy1 = min(ny, int(np.ceil((hi[1] - self.y0_um)/dx)) + 1)
                print(f'  [WARN] CAD detector at {100*f:.0f} % of the core length: core '
                      f'"{self.core}" not in the cut -> y from the core bbox ({iy0}..{iy1})')
            out.append((ix0, ix1, iy0, iy1))
        return out

    def meta(self):
        """Fuer die NPZ (Analyzer): Namen, Brechzahlen, Kern/Film, Quelle, Verschiebung."""
        names = self.scene.region_names()
        return dict(cad_file=os.path.basename(self.scene.path),
                    cad_names=[self.scene.display_name(b) for b in names],
                    cad_bodies=names,
                    cad_n=[self.scene.n_of(b, self.lam_nm) for b in names],
                    cad_materials=[str(self.scene.materials.get(b, '')) for b in names],
                    cad_core=self.core_index, cad_tear=self.tear_index,
                    cad_x_shift_um=self.x_shift_um, cad_plane_z_um=self.z0_um,
                    cad_scale_um=self.scene.scale_um)
