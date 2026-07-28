# Meep-Integration — Machbarkeits- & Mapping-Plan

Status: **Planungsdokument (noch kein Code).** Ziel ist, dieselben Simulationen
(planarer Waveguide + Kontaktlinse, 2D und 3D) zusätzlich mit der
[Meep](https://meep.readthedocs.io)-FDTD-Toolbox rechnen zu können — als
**unabhängige Cross-Validierung** des eigenen NumPy/CuPy-Solvers und als
alternativer, feature-reicher Rechenweg. Ergebnisse sollen im **bestehenden
Analyzer-Format** landen, damit `common/fdtd_analyzer.py` sie direkt öffnet.

---

## 1. Ausführungsumgebung (der kritische Punkt)

Meep ist **nicht** nativ unter Windows lauffähig:

- Kein PyPI-Paket (`pip install meep`/`pymeep` schlägt fehl).
- conda-forge hat **kein `pymeep` für win-64** (nur linux-64 / osx).

**Lösung: WSL2** (auf diesem Rechner bereits installiert, Version 2). Meep läuft
in einem Linux-conda-Env innerhalb WSL2; das Repo ist von dort unter
`/mnt/c/Users/AdminAlex/Documents/GitHub/lens_sensor` erreichbar. Ergebnisse in
`results/` sind damit auch für den **Windows-Analyzer** sichtbar.

**GPU-Hinweis:** Meep ist **CPU/MPI**-basiert. Die CuPy-GPU-Beschleunigung des
eigenen Solvers greift bei Meep nicht — Parallelisierung läuft über MPI-Prozesse
(`mpirun -np N python run_meep.py …`).

### WSL2-Setup (einmalig)

```bash
# in WSL2 (Ubuntu):
#  1) Miniforge/conda installieren (falls nicht vorhanden)
#  2) Meep-Env anlegen (conda-forge liefert pymeep + parallele Variante)
conda create -n meep -c conda-forge pymeep pymeep-extras
#  parallele (MPI-)Variante alternativ:
#     conda create -n meep -c conda-forge pymeep=*=mpi_mpich_* pymeep-extras
conda activate meep
python -c "import meep; print(meep.__version__)"   # Funktionstest
cd /mnt/c/Users/AdminAlex/Documents/GitHub/lens_sensor
```

> **Einschränkung:** Diese Umgebung kann in der Windows-Sitzung nicht getestet
> werden. Verifikation der Meep-Läufe erfolgt in WSL2 (gemeinsam, Schritt für
> Schritt).

---

## 2. Vorgeschlagene Architektur

Neuer Ordner `meep/` (parallel zu `planar_beads/`, `planar_3d/`):

| Datei | Rolle |
|---|---|
| `meep/run_meep.py`     | Einstieg, spiegelt die CLI von `run_simulation.py` (`--geometry`, `--dim`, `--scenario`, `--length-um`, …). Baut daraus eine Meep-Simulation. |
| `meep/meep_build.py`   | Übersetzt Geometrie/Quelle/Monitore/Ränder in Meep-Objekte (Kernstück des Mappings). |
| `meep/meep_to_npz.py`  | Schreibt Meep-Felder in **analyzer-kompatible** `*_frames.npz` / `*_vol3d.npz`. |
| `docs/meep_setup_wsl.md` | Ausführliche WSL2-Installationsanleitung. |

**Wiederverwendung:** `common/physics.py` (n(λ), Konstanten) und die
`LENS_SCENARIOS` aus `run_simulation.py` werden importiert — die
Material-/Szenario-Definitionen bleiben also **eine Quelle der Wahrheit** für
beide Engines.

`run_meep.py` soll dieselben Default-/Alias-Regeln nutzen wie `run_simulation.py`
(Geometrie-Aliasse, Szenario-Layer, „große Linse"-Hinweis), damit ein Lauf
1:1 vergleichbar parametrisiert werden kann.

---

## 3. Das Mapping (Solver → Meep)

### 3.1 Einheiten & Auflösung

Der eigene Solver rechnet in **SI** (dx in nm, dt aus CFL). Meep rechnet in
**natürlichen Einheiten** mit einer charakteristischen Länge `a` und c=1.

Empfehlung: **`a = 1 µm`.** Dann:

| Größe | Solver (SI) | Meep |
|---|---|---|
| Länge x µm | `lx_um` | `lx_um` (dimensionslos, in Einheiten von a) |
| Auflösung | `dx_nm` | `resolution = 1000 / dx_nm` (Pixel pro µm) |
| Wellenlänge | `lam_nm` | `λ = lam_nm/1000`, Frequenz `f = 1/λ` |
| Zeit | `dt` (s) | Meep wählt dt selbst (Courant); Laufzeit in Einheiten a/c |
| Brechzahl n | `n_at(mat, λ)` | `mp.Medium(index=n)` bzw. `epsilon=n**2` |

### 3.2 Koordinaten

Solver: `x`=Propagation, `y`=Schichtstapel, `z`=Tiefe (3D). Meep-Zellen
(`mp.Vector3`) übernehmen dieselbe Zuordnung. In **2D** ist `z` die invariante
Richtung (Meep-Zelle mit `size.z = 0`).

### 3.3 Materialien / Geometrie

Schichtstapel (Air / Waveguide / Lipid / Aqueous / Mucin / Cornea) → Liste von
`mp.Block`:

- Jede Schicht ein `mp.Block` mit `size=(lx, dicke, mp.inf|lz)`, `center` auf der
  y-Mitte der Schicht, `material=mp.Medium(index=n)`.
- 2D: `size.z = mp.inf` (invariant). 3D: endliche `lz`, ggf. WG-Kanalbreite
  `wg_width` als schmaleres Block in z.
- **Bead**: 2D → `mp.Cylinder(radius=r, height=mp.inf, axis=z, center=…)`
  (entspricht dem „unendlichen Zylinder" der 2D-Demo); 3D → `mp.Sphere(radius=r,
  center=…)`. Position wie im Solver (in Aqueous, Default aufliegend an WG-Unterseite).
- **Referenzlauf ohne Bead** (`_ref`) = Geometrie ohne das Bead-Objekt → für die
  Transmissions-Normierung (siehe 3.6).

### 3.4 Quelle

| Solver | Meep |
|---|---|
| CW (`source_type='cw'`) | `mp.ContinuousSource(frequency=f)` |
| Puls (`source_type='pulse'`) | `mp.GaussianSource(frequency=f, fwidth=…)` |
| Gauss-Spot (Taille y[,z]) | `mp.Source(..., amp_func=<Gauss-Profil>)` an der Quellebene |
| **Empfohlen** | `mp.EigenModeSource` — regt sauber die **geführte Mode** an (numerisch stabiler als ein roher Gauss-Spot) |
| Neigung `vcsel_tilt` | `mp.EigenModeSource(direction=…)` bzw. Phasenrampe im `amp_func` |

**Polarisation — Achtung, Konventionsfalle:** „s (TE, Ez, E entlang Tiefe z)"
entspricht in Meep-2D der Feldkomponente **`mp.Ez`** (aus der Ebene). „p (TM,
E in der x-y-Ebene)" entspricht **`mp.Ey`/`mp.Ex`** (in der Ebene). Meeps eigene
TE/TM-Bezeichnung ist gegenläufig zur hier verwendeten — deshalb wird auf die
**Komponente** (Ez vs. Ey) abgebildet, nicht auf die Labels. Muss gegen eine
bekannte Mode validiert werden.

### 3.5 Ränder

Solver-**Sponge**-Absorber → Meep-**PML**: `boundary_layers=[mp.PML(dpml)]` auf
allen offenen Rändern; optional `mp.Absorber` statt PML bei schwierigen Moden.
Solver-`pec_faces` → `mp.Metal`-Randbedingung auf der jeweiligen Fläche.
`end_facet` (Material→Luft) → einfach durch die Geometrie (kein WG-Block am Ende).

### 3.6 Methoden — wichtig

- **full** → Meeps normaler Full-Domain-Lauf. Direktes Gegenstück.
- **sliding / stitch** → **entfällt.** Das sind Speicher-Tricks des eigenen
  Solvers. Meep rechnet lange Domänen nativ (nötigenfalls per MPI verteilt) und
  braucht keine Gebiets-Zerlegung. `run_meep.py` bildet `stitch`/`sliding`
  daher auf einen **full**-Lauf ab (mit Hinweis im Log).

### 3.7 Transmission / Sensor-Kennzahl

Solver: zeitintegriertes `|E|²` über WG-Querschnittsboxen bei `det_in`/`det_out`.
Meep (rigoroser): **Flux-Monitore**

```
in_flux  = sim.add_flux(f, 0, 1, mp.FluxRegion(center=det_in_plane,  size=WG-Querschnitt))
out_flux = sim.add_flux(f, 0, 1, mp.FluxRegion(center=det_out_plane, size=WG-Querschnitt))
```

- **Normierungslauf** ohne Bead (`_ref`) liefert `in0/out0`.
- **Lauf mit Bead** liefert `in1/out1`.
- Transmission `T = get_fluxes(out_flux) / <Normierung>` (analog zu eurem
  bead/ref-Vergleich).

> Erwartung: Meep-Transmission ist **nicht bit-identisch** zur Box-Integration
> des eigenen Solvers (andere Definition, PML statt Sponge). Genau das macht den
> Cross-Check wertvoll: Übereinstimmung im **Trend/Betrag** validiert beide.

### 3.8 Feld-Snapshots

- **Transient**: `sim.run(mp.at_every(t_snap, capture), until=…)`, wobei
  `capture` per `sim.get_array(component=mp.Ez, center=…, size=…)` einen Frame
  zieht → Stapel `(n_frames, Nx, Ny)`.
- **Eingeschwungen (CW)**: `sim.add_dft_fields([mp.Ez], f, 0, 1, where=…)`,
  nach Steady-State die komplexe Amplitude `sim.get_dft_array(...)` holen und
  **Phasen über eine Periode** rekonstruieren → entspricht eurem `Ez_cw`.

---

## 4. Analyzer-kompatibles NPZ (Zielformat)

`meep_to_npz.py` muss exakt die Schlüssel schreiben, die
`common/fdtd_analyzer.py` erwartet (wie von `save_results`/`save_results_3d`
erzeugt).

### 4.1 2D — `<name>_frames.npz`

| Key | Inhalt | Meep-Quelle |
|---|---|---|
| `Ez` | `(n_fr, Nx, Ny)` Feld-Frames | `get_array` je Snapshot |
| `meta` | je Frame `(x_start, x_end, y_start, y_end, slide_i, time, step)` | aus Zellgeometrie + Snapshot-Zeit |
| `scenario` | Label | CLI |
| `layers` | `(t_lip, t_aq, t_mu, n_aq)` in m | aus Szenario/Params |
| `material_names`, `material_indices` | Overlay-Legende | Params |
| `bead_x_um`, `bead_diameter_um` | Bead-Overlay | Params |
| `t_wg_um`, `x_wg_start_um`, `x_wg_end_um` | WG-Overlay | Params |
| `lam_nm`, `polarization` | Metadaten | CLI |
| `Ez_cw`, `meta_cw` | CW-Satz | DFT-Feld |
| `has_transient`, `has_cw`, `cw_only`, `view_primary` | Ansichts-Flags | gesetzt |

### 4.2 3D — `<name>_frames.npz` (z-Mittelebene) + `<name>_vol3d.npz`

- `_frames.npz`: wie 2D, aber `Ez` = **z-Mittelebene** des Volumens (der 2D-Viewer).
- `_vol3d.npz`: volles `Ez`-Volumen `(n_fr, Nx, Ny, Nz)`, `Iavg`, `dx_um`,
  `x0_um/y0_um/z0_um`, `layers`, `material_*`, `t_wg_um`, `wg_width_um`, `bead_*`
  — die Schlüssel, die der 3D-Viewer (`_build_views_3d`, `on_view3d`) für Achsen
  und Overlays liest.

Die Achsen-/Nullpunkt-Konventionen (`y0_um = -tear_um`, `z0_um = -lz/2`) müssen
übernommen werden, damit Overlays (Schichtlinien, Bead-Kreis/Kugel) korrekt sitzen.

---

## 5. Umfang-spezifische Punkte (planar + Linse, 2D + 3D)

| Fall | Aufwand | Anmerkung |
|---|---|---|
| **planar 2D** | klein | Idealer erster Validierungsfall (dünner WG + Bead) |
| **planar 3D** | mittel | echte Kugel via `mp.Sphere`, WG-Kanal via z-begrenztem Block |
| **lens 2D** | klein–mittel | dicker 250-µm-Slab + Trockenauge-Schichten; volle 14 mm in 2D machbar |
| **lens 3D** | **groß / begrenzt** | 250-µm-Querschnitt in 3D ist auch für Meep sehr teuer; **kein Stitch** → nur kurze Ausschnitte realistisch (physikalische Grenze, keine Tool-Grenze) |

Die geflattete Linsen-Näherung (flacher Schichtstapel, Krümmung ~3 % vernachlässigt)
bleibt identisch — Meep bekommt denselben flachen Stapel wie der eigene Solver.

---

## 6. Validierungsstrategie

1. **Kleiner planar-2D-Fall** in beiden Engines mit denselben Parametern
   (kurze Länge, grobe Auflösung).
2. Vergleich: (a) Transmission (Trend/Betrag), (b) Feldmuster/Modenprofil
   qualitativ, (c) evaneszente Abklingtiefe.
3. Erwartung: **nah, nicht bit-identisch** (PML vs. Sponge, Flux vs. Box,
   Subpixel-Averaging). Systematische Abweichungen dokumentieren.
4. Erst nach bestandenem 2D-Cross-Check die 3D-/Linsen-Fälle nachziehen.

---

## 7. Risiken & offene Punkte

- **Polarisations-Konvention** (Meep-2D TE/TM ↔ Ez/Ey) — Hauptfehlerquelle,
  zwingend gegen eine bekannte Mode prüfen.
- **Quell-Normierung & Subpixel-Averaging** in Meep können Amplituden
  systematisch verschieben — für Transmission über Normierungslauf abfangen.
- **Einheiten-Fallen** (a=1 µm konsequent; f=1/λ, nicht 2π/λ).
- **Keine GPU** — Laufzeit über MPI-Kerne skalieren; große Fälle sind CPU-teuer.
- **Nicht in Windows testbar** — Verifikation nur in WSL2.
- **NPZ-Treue**: fehlende/fehlbenannte Keys brechen die Analyzer-Overlays →
  1:1 gegen `save_results`/`save_results_3d` abgleichen.

---

## 8. Empfohlene Reihenfolge (auch bei „vollem" Zielumfang)

1. WSL2-Meep-Env aufsetzen + Funktionstest (`docs/meep_setup_wsl.md`).
2. `meep/meep_build.py` + `run_meep.py` für **planar 2D**, Flux-Transmission.
3. `meep_to_npz.py` (2D) → im vorhandenen Analyzer öffnen.
4. **Cross-Check planar 2D** gegen `run_simulation.py`.
5. planar 3D → 3D-NPZ/Volumen.
6. lens 2D (voller 14-mm-Slab).
7. lens 3D (kurzer Ausschnitt, Grenzen dokumentieren).

> Trotz „vollem" Zielumfang wird inkrementell gebaut: jeder Schritt ist einzeln
> gegen den eigenen Solver validierbar, bevor der nächste dazukommt.
