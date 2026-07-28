# Ein WSL2-Env für alles (Solver + CuPy-GPU + Meep)

Ziel: **eine** Umgebung, in der sowohl der eigene NumPy/CuPy-Solver als auch
Meep laufen. Da Meep keinen Windows-Build hat, liegt dieses Env in **WSL2**
(Linux). Das Windows-Repo ist von dort unter `/mnt/c/...` erreichbar; die GPU
(RTX A500) wird via WSL-CUDA durchgereicht; GUIs laufen über WSLg.

> Distribution: **Ubuntu 24.04 LTS**. Das Setup läuft **als root** in der Distro
> — damit entfällt die interaktive Benutzer/Passwort-Anlage (OOBE). Ein normaler
> Linux-User kann später mit `adduser` angelegt werden, ist aber nicht nötig.

---

## 0. Voraussetzungen (bereits erfüllt auf diesem Rechner)

- WSL2 aktiv (Version 2.4.13, eigener Kernel) inkl. **WSLg** (GUI-Support).
- NVIDIA-Treiber 595.95 auf Windows (unterstützt CUDA-on-WSL).

## 1. Ubuntu 24.04 registrieren (ohne interaktiven Erststart)

```powershell
wsl --install -d Ubuntu-24.04 --no-launch
```

Prüfen:

```powershell
wsl -l -v          # Ubuntu-24.04 sollte gelistet sein (Version 2)
```

## 2. Basis-Setup als root (Miniforge)

```bash
wsl -d Ubuntu-24.04 -u root bash -lc '
  apt-get update && apt-get install -y wget ca-certificates bzip2 && \
  wget -qO /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
  bash /tmp/miniforge.sh -b -p /opt/conda && \
  /opt/conda/bin/conda --version
'
```

## 3. Unified-Env `lens_sensor` (Meep) + MKL-Soname-Fix

Meep ist das primäre Ziel und wird zuerst installiert (CPU-Deps inklusive):

```bash
wsl -d Ubuntu-24.04 -u root bash -lc '
  /opt/conda/bin/conda create -y -n lens_sensor -c conda-forge \
    python=3.11 pymeep pymeep-extras \
    numpy scipy matplotlib imageio scikit-image psutil packaging
'
```

**Wichtig — MKL-Soname-Fix:** pymeep 1.30 ist gegen `libmkl_rt.so.2` gebaut, der
conda-Solve zieht aber das brandneue `mkl 2026` (nur `libmkl_rt.so.3`) →
`ImportError: libmkl_rt.so.2`. Lösung: MKL auf 2024.x pinnen (liefert `.so.2`)
und den Pin dauerhaft im Env verankern:

```bash
wsl -d Ubuntu-24.04 -u root bash -lc '
  /opt/conda/bin/conda install -y -n lens_sensor -c conda-forge "mkl=2024.*" && \
  echo "mkl 2024.*" > /opt/conda/envs/lens_sensor/conda-meta/pinned && \
  /opt/conda/bin/conda run -n lens_sensor python -c "import meep; print(\"meep\", meep.__version__)"
'
```

> Der Env hat dadurch **numpy 1.26.4** (pymeep 1.30 verlangt numpy<2). Das ist
> für den Fixpunkt der nächsten Schritte wichtig.

## 4. CuPy-GPU im selben Env (WSL-CUDA)

In WSL liefert der Windows-Treiber nur `libcuda` (Device-API) — die **CUDA-Runtime
+ Header** fehlen und müssen aus conda-forge dazu. Weil der Env numpy 1.26 hat,
wird bewusst **CuPy 13.x** (nicht 14.x, das numpy≥2.0 bräuchte) installiert, damit
numpy und damit Meep unangetastet bleiben.

```bash
wsl -d Ubuntu-24.04 -u root bash -lc '
  # (a) CuPy 13.x (numpy-1.26-kompatibel, aendert numpy NICHT)
  /opt/conda/bin/conda run -n lens_sensor pip install "cupy-cuda12x>=13,<14"
  # (b) CUDA-12-Runtime + Header (nvrtc, cudart, math-Libs, dev-Header)
  /opt/conda/bin/conda install -y -n lens_sensor -c conda-forge "cuda-version=12.*" \
    cuda-nvrtc cuda-cudart libcublas libcufft libcurand libcusolver libcusparse \
    cuda-cudart-dev cuda-cccl
  # (c) conda-forge legt CUDA-Header unter targets/.../include -> nach include/ verlinken
  ln -sf /opt/conda/envs/lens_sensor/targets/x86_64-linux/include/* \
         /opt/conda/envs/lens_sensor/include/
'
```

**CUDA-Pfade dauerhaft** (sonst findet CuPy nvrtc/Header nicht) — Activation-Hook:

```bash
wsl -d Ubuntu-24.04 -u root bash -lc '
  mkdir -p /opt/conda/envs/lens_sensor/etc/conda/activate.d && \
  printf "export CUDA_PATH=/opt/conda/envs/lens_sensor\nexport LD_LIBRARY_PATH=/opt/conda/envs/lens_sensor/lib:\$LD_LIBRARY_PATH\n" \
    > /opt/conda/envs/lens_sensor/etc/conda/activate.d/cuda_paths.sh
'
```

## 5. Verifikation

```bash
wsl -d Ubuntu-24.04 -u root bash -lc '
  echo "--- GPU in WSL ---"; nvidia-smi --query-gpu=name --format=csv,noheader; \
  cd /mnt/c/Users/AdminAlex/Documents/GitHub/lens_sensor && \
  /opt/conda/bin/conda run -n lens_sensor python -c \
    "import meep, cupy as cp; \
     print(\"meep\", meep.__version__, \"+ cupy\", cupy.__version__); \
     print(\"GPU-Test sum:\", float(cp.sum(cp.arange(5,dtype=cp.float32)**2).get()), \"(erwartet 30.0)\")"
'
```

Erwartete Ausgabe: `meep 1.30.0 + cupy 13.6.0` und `GPU-Test sum: 30.0`.

## 6. Projekt aus WSL starten

```bash
wsl -d Ubuntu-24.04 -u root bash -lc '
  cd /mnt/c/Users/AdminAlex/Documents/GitHub/lens_sensor && \
  /opt/conda/bin/conda run -n lens_sensor python run_simulation.py \
    --geometry planar --dim 3 --method full --length-um 6 --dry-run
'
```

Der eigene Solver läuft damit in WSL (GPU, falls CuPy installiert), Ergebnisse
landen in `results/` — sichtbar auch für den Windows-Analyzer.

## 7. GUI (tkinter) via WSLg

```bash
wsl -d Ubuntu-24.04 -u root bash -lc '
  cd /mnt/c/Users/AdminAlex/Documents/GitHub/lens_sensor && \
  /opt/conda/bin/conda run -n lens_sensor python run_gui.py
'
```

WSLg zeigt das Fenster direkt unter Windows an (kein X-Server nötig).

## Hinweise

- **Root-Setup** ist für ein Dev-Env in WSL üblich und unkritisch. Wer einen
  normalen User will: `adduser <name>` + conda-Init für diesen User.
- **Performance:** Zugriff auf `/mnt/c` ist langsamer als das Linux-Dateisystem.
  Für große Läufe ggf. das Repo nach `~/lens_sensor` klonen und dort rechnen,
  Ergebnisse zurückkopieren. Für Meep-MPI: `mpirun -np N ...`.
- **Meep = CPU/MPI**, keine GPU. CuPy beschleunigt nur den eigenen Solver.
- Ein komfortabler Alias (in `~/.bashrc` der Distro):
  `alias ls-run='cd /mnt/c/Users/AdminAlex/Documents/GitHub/lens_sensor && conda run -n lens_sensor'`
