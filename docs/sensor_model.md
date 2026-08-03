# Kontaktlinsen-Wellenleiter-Sensor — Modell, Methoden & Nutzung

> Diese Doku beschreibt **Aufbau, Physik, Rechenwege und Bedienung** des
> Kontaktlinsen-Sensors. **Konkrete Messergebnisse** (Szenario-Diskriminierung,
> Kennzahlen je Trockenauge-Fall) sind hier bewusst **nicht** enthalten — diese
> erzeugt man durch Ausführen der unten genannten Werkzeuge.

---

## 1. Sensor-Prinzip

Eine gekrümmte **PMMA-Kontaktlinse** (Dicke *t* = 250 µm, Krümmungsradius
*R* = 8.3 mm, Durchmesser *D* = 14 mm, n ≈ 1.491 @ 850 nm) wirkt als
**Wellenleiter**. Ein **VCSEL (850 nm)** wird an der **Stirnfläche am Rand**
*butt-gekoppelt* (End-Fire); das Licht läuft per **Totalreflexion (TIR)** entlang
des Bogens zum Apex bzw. zu den Detektoren.

An jeder TIR-Reflexion an der **Unterseite** reicht das **evaneszente Feld**
(~0.2 µm) in den **Tränenfilm**. Dessen optische Eigenschaften — vor allem die
**Lipid-Schicht** (direkt unter dem Kern) und der **Aqueous-Index** (Osmolarität)
— modulieren, wie viel geführte Leistung erhalten bleibt bzw. ausgekoppelt wird.
Der Sensor ist also im Kern ein **ATR-Sensor** (*attenuated total reflection*).

**Detektor-System** (Definitionen in `kontaktlinse/config.py`, Skizze in
`kontaktlinse/plot_geometry.py`):

| Detektor | Rolle | Lage |
|---|---|---|
| **D1** | Tear-Detektor | tangential an der Linse, Tränenfilm-seitig |
| **D2** | Luft-Referenz | tangential, Luft-seitig (Gegenstück zu D1) |
| **D3** | End-Fire / TIR-Licht | am Linsenende, fängt das durchgelaufene TIR-Licht |

D1/D2 sind je 500 × 500 × 50 µm (Dicke **50 µm** = `D1_DICKE`, tangential rotiert),
D3 ist 200 × 50 µm (tangential). Die Detektoren werden **tangential zur lokalen
Linsenfläche** rotiert platziert.

---

## 2. Physikalische Grundlagen

- **Geführte Moden:** Ausbreitungskonstante β = n_eff · k₀, mit
  n_clad < n_eff < n_core. k₀ = 2π/λ.
- **Evaneszente Eindringtiefe** (Amplitude, 1/e):
  δ = 1/γ = λ / (2π·√(n_eff² − n_clad²)). Für PMMA-Kern / Aqueous-Cladding liegt
  δ theoretisch bei ~0.2 µm; ins Luft-Cladding ist δ noch kleiner (~0.06 µm).
  Die **Intensitäts**-Eindringtiefe ist δ/2.
- **Akzeptanzwinkel:** Geführt ist Licht, das an **beiden** Grenzflächen
  totalreflektiert. Die engere Grenze ist der Tränenfilm; kritischer Winkel
  θ_c = arcsin(n_clad/n_core), maximaler Strahlwinkel zur Linsenachse
  ≈ 90° − θ_c (Größenordnung ±26°). VCSEL-Divergenz (~5–10° in PMMA) passt
  komfortabel hinein → hohe geführte Kopplung, **sofern entlang der Tangente
  eingekoppelt** wird.
- **ATR / Netto-Leistung:** Ein **verlustfreies** evaneszentes Feld trägt
  **keine Netto-Leistung** in den Tränenfilm (senkrechter Poynting-Fluss = 0).
  Ein reales Gerät „misst" das evaneszente Feld daher nur über eine
  **Wechselwirkung** — Absorption (→ ATR-Dämpfung), Streuung oder Fluoreszenz.
- **Modaler Verlust (ATR):** α_m = α_tear · Γ_m, mit dem **Confinement-Faktor**
  Γ_m = Anteil der Modenleistung im absorbierenden Tränenfilm. Das überlebende
  TIR-Licht ist Σ|a_m|²·exp(−α_m·L).

---

## 3. Zwei Rechenwege

Beide Wege bleiben im Repo erhalten und ergänzen sich.

### 3a. Brute-Force-FDTD (Voll-Feld)

- **`kontaktlinse/full_domain_fdtd.py`** — **flattened coordinates**
  (y_rel = y − y_mid(x)): die gekrümmte Linse wird zu einem **geraden** planaren
  Wellenleiter. Vorteile: ganze 14-mm-Linse in **einem** Lauf; die vertikale
  Quellspalte ist im flattened Bild **automatisch orthogonale Fläche + tangentiale
  Richtung** → **korrekte Einkopplung by construction**; dünne y-Ausdehnung
  (speichereffizient). Nachteil: die ~3 % Krümmung (t/R) wird vernachlässigt; die
  Anzeige zeigt die Linse begradigt.
- **`kontaktlinse/sliding_window_fdtd.py`** — **globale gekrümmte** Koordinaten
  mit **Co-Moving-Fenster** (nur ein Fenster im Speicher). Achtung: In globaler
  Krümmung liegt die Linse **diagonal** im Fenster; eine **vertikale** Quellspalte
  ist am steilen Rand (~57° Tangente) **fehlangepasst** → Einkopplungs-Kollaps.
  Für den gekrümmten Solver ist eine **tangenten-orientierte** Quelle nötig
  (orthogonale Fläche, Richtung entlang der Tangente).

**Grundsätzliche Grenze:** Über die **mm-lange** Strecke akkumuliert die
**numerische Dispersion**; adäquate Auflösung über die gesamte Länge ist nicht
bezahlbar (siehe §5). Brute-Force-FDTD ist daher ideal für **lokale** Details
(Einkopplung, Detektor, Streuer), nicht für die volle Länge in Absolut-Genauigkeit.

### 3b. Native Moden-Propagation (`kontaktlinse/mode_propagation.py`)

Auflösungsrobuster, schneller Weg — analog zum Meep-Mode-Handoff, aber nativ und
**ohne FDTD-Gitter entlang der Länge**:

1. **Transversale Eigenmoden** des Schicht-Slabs (Luft \| PMMA-Kern \| Lipid \|
   Aqueous \| Mucin \| Cornea) via **tridiagonalem 1D-Eigenlöser**
   (`scipy.linalg.eigh_tridiagonal`), fein in y (löst das evaneszente Feld auf).
   Liefert exakte β_m/n_eff + Profile φ_m(y).
2. **Einkopplung** → Moden-Amplituden a_m: entweder **analytischer Gauss-Overlap**
   oder **lokales FDTD am Rand** (`--couple fdtd`, reale Butt-Coupling-Physik inkl.
   Fresnel/Nahfeld) mit anschließender **Projektion** auf die Moden.
3. **Analytische Propagation** über den Bogen: a_m → a_m·exp(iβ_m·L). Keine
   akkumulierende Gitter-Dispersion, keine Rechnung Zelle-für-Zelle über die Länge.
4. **Sensor-Auswertung:**
   - *Direkte Bestimmung* des evaneszenten Feldes im Tränenfilm (Modell):
     Intensität an der Oberfläche, integriertes Feld im Aqueous, Eindringtiefe δ.
   - *ATR-Dämpfung* (real messbar): modaler Verlust α_m = α_tear·Γ_m → überlebendes
     TIR-Licht (→ D3).

**Validierung:** `--validate` prüft den Eigenlöser gegen die **analytische
asymmetrische Slab-Formel** (`tests/crossval.py`) an einem dünnen Few-Mode-Slab
(Übereinstimmung im Rahmen der Rundung).

---

## 4. Koordinaten & Einkopplung

- **Flattened vs. curved:** Flattened rechnet die Krümmung heraus (Linse gerade),
  curved behält die reale Geometrie. Für **einen** VCSEL ist die 2D-Meridian-Ebene
  die **exakte** Reduktion (der Strahl bleibt in der Ebene); die Krümmung ist ein
  ~3 %-Effekt (t/R).
- **Tangentiale Einkopplung (Pflicht bei curved):** Die Einkoppelfläche ist
  **orthogonal** zur Linse (Querschnitt), die Einkoppel**richtung** liegt
  **tangential** zur Linsenkrümmung. Eine vertikale Quellspalte auf einer geneigten
  Stirnfläche koppelt fehl → Verlust. Im flattened Bild ist die Tangente = +x,
  daher ist die vertikale Spalte dort korrekt.

---

## 5. Machbarkeit & Grenzen (wichtig)

- **Auflösung vs. Wellenlänge:** λ im PMMA ≈ 570 nm. 300 nm/Zelle ≈ 1.9 Zellen/λ
  (an der Nyquist-Grenze) → grob; für gute Genauigkeit ≥ ~10–20 Zellen/λ.
- **Aufwand 2D-FDTD ∝ (1/dx)³** (zwei Ortsdimensionen × Zeitschritte via Courant).
  Eine Verfeinerung von 300 nm auf 10 nm ist grob **~27 000×** mehr Rechenzeit.
  → **10 nm über die ganze 14-mm-Linse ist nicht machbar.** Feine Auflösung ist nur
  **lokal** (kleiner Ausschnitt) sinnvoll.
- **Akkumulierende Dispersion:** Der Phasenfehler summiert sich linear mit der Zahl
  der Wellenlängen (~10⁴ über mm). Deshalb liefert grobe FDTD über die volle Länge
  keine belastbaren **Absolut**-Werte; **Verhältnisse** (Referenz-normiert) kürzen
  systematische Fehler teilweise. Die **Moden-Propagation** umgeht das Problem
  (exaktes β, keine Gitter-Propagation).
- **3D:** Die 250-µm-Dicke ist der **Speicher**-Engpass; volle-Linse-3D ist
  unmöglich. Meep (CPU/MPI) eignet sich für **lokale** 3D-Ausschnitte; auf einer
  kleinen Box **seriell / wenige Ränge** rechnen (viele MPI-Ränge vervielfachen den
  Speicher durch Overhead + Full-Field-Gather → OOM-Risiko). Siehe
  `docs/meep_setup_wsl.md`.

---

## 6. Nutzung (CLI)

**Native Moden-Propagation** (schnell, auflösungsrobust):
```bash
# aus dem Verzeichnis kontaktlinse/  (importiert config.py)
python mode_propagation.py                      # alle Szenarien (Default dy=10nm)
python mode_propagation.py --validate           # Eigenloeser gegen Analytik pruefen
python mode_propagation.py --couple fdtd         # reale Rand-Einkopplung (lokales FDTD)
python mode_propagation.py --offset-scan="-120,-60,0,60,120"   # Kopplung vs Offset
# Parameter: --scenario, --lambda-nm, --dy-nm, --L-mm, --offset-um, --waist-um,
#            --alpha-tear-permm (Traenenfilm-Absorption fuer die ATR-Daempfung)
```

**Brute-Force-FDTD:**
```bash
python full_domain_fdtd.py --scenario Gesund --gpu          # ganze Linse, flattened
python sliding_window_fdtd.py --scenario Gesund --gpu       # Co-Moving-Fenster
# Parameter u.a.: --resolution (nm), --window-w/--window-h/--slide (um), --t-* / --n-*
```

**Auswertung** der FDTD-Ergebnisse im Analyzer:
```bash
python common/fdtd_analyzer.py results/<datei>_frames.npz
```

> Das Ausführen erzeugt die jeweiligen **Kennzahlen** (n_eff, δ, evan_ratio,
> ATR-Dämpfung, D1/D2/D3, R_tear). Diese Zahlen sind in dieser Doku bewusst nicht
> gelistet.

---

## 7. Detektor-Geometrie (Referenz)

Maßgeblich sind `kontaktlinse/config.py` (Positionen/Größen: `D1_S_CENTER`,
`D1_DICKE=50 µm`, `D3_LEN`, `D3_QUERSCHN`, `T_BUF`, …) und die Illustration in
`kontaktlinse/plot_geometry.py` (Panel A: Übersicht; Panel B: lokaler Zoom des
D1/D2-Doppelstacks + D3; Panel C: VCSEL-Butt-Coupling an der Stirnfläche).
Die Detektor-Platzierung in den Solvern rotiert tangential zur lokalen
Linsenfläche und stimmt mit Panel B überein.

---

## 8. Bekannte Einschränkungen / offene Punkte

- **Moden-Propagation:** aktuell 2D-Meridian (für **einen** VCSEL exakt); die
  **azimutale** Strahl-Aufweitung erfordert ein lokales 3D. Für einen dicken,
  hoch-multimodigen Kern hängt das Sensor-Signal von der **Moden-Verteilung** ab,
  die wiederum von der **Einkopplung** bestimmt wird (→ `--couple fdtd`,
  `--offset-scan`).
- **Tränenfilm-Absorption:** `--alpha-tear-permm` ist ein **repräsentativer**
  Parameter; für kalibrierte ATR-Signale müsste die wellenlängenabhängige
  Absorption/Streuung des Analyten hinterlegt werden.
- **Gekrümmter sliding-Solver:** benötigt noch die tangenten-orientierte Quelle;
  bis dahin `full_domain_fdtd.py` (flattened) für korrekte Einkopplung nutzen.
- **Detektor-Signale aus grober FDTD** sind relativ (Referenz-normiert) zu lesen,
  nicht als kalibrierte Absolutwerte (siehe §5).
