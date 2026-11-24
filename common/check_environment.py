"""Diagnose-Skript: prueft alle Abhaengigkeiten und GPU-Verfuegbarkeit.

Ausfuehren mit:  python common/check_environment.py   (vom Repo-Root)
"""
import sys
import importlib

REQUIRED = {
    'numpy':      'pip install numpy',
    'matplotlib': 'pip install matplotlib',
    'imageio':    'pip install imageio',
    'cupy':       'pip install "cupy-cuda12x>=13"',
}

PYTHON_VERSION_MIN = (3, 10)


def ok(msg):
    print(f'  [OK]   {msg}')

def fail(msg):
    print(f'  [FAIL] {msg}')


print('='*60)
print('Lens-Sensor: environment diagnostics')
print('='*60)

# Python-Version
print('\n--- Python ---')
v = sys.version_info
if v >= PYTHON_VERSION_MIN:
    ok(f'Python {v.major}.{v.minor}.{v.micro}')
else:
    fail(f'Python {v.major}.{v.minor} - at least {PYTHON_VERSION_MIN} required')

# Pakete
print('\n--- Python packages ---')
missing = []
versions = {}
for pkg, install_cmd in REQUIRED.items():
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, '__version__', '?')
        versions[pkg] = ver
        ok(f'{pkg} {ver}')
    except ImportError as e:
        fail(f'{pkg} NOT installed -> {install_cmd}')
        missing.append(pkg)

# GPU (via CuPy)
print('\n--- GPU ---')
if 'cupy' in versions:
    try:
        import cupy as cp
        n_dev = cp.cuda.runtime.getDeviceCount()
        ok(f'{n_dev} GPU(s) detected')
        for i in range(n_dev):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props['name'].decode()
            mem_free, mem_total = cp.cuda.runtime.memGetInfo()
            ok(f'GPU {i}: {name}')
            ok(f'         VRAM total: {mem_total//(1024**3)} GB')
            ok(f'         VRAM free:  {mem_free//(1024**3)} GB')
            ok(f'         Compute Capability: {props["major"]}.{props["minor"]}')
        # Mini-Test: array berechnen
        a = cp.array([1, 2, 3, 4], dtype=cp.float32)
        b = cp.sum(a*a).get()
        if abs(b - 30.0) < 1e-3:
            ok(f'GPU test (sum of squares): {b}')
        else:
            fail(f'GPU test wrong: {b} (expected 30)')
    except Exception as e:
        fail(f'GPU test failed: {e}')
else:
    fail('CuPy missing - GPU cannot be tested')

# Lokale Files (relativ zum Repo-Root, unabhaengig vom Arbeitsverzeichnis)
print('\n--- Lens-Sensor Files ---')
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
needed_files = ['run_simulation.py', 'run_gui.py',
                'common/fdtd_analyzer.py', 'planar_beads/fdtd2d_core.py',
                'planar_3d/fdtd3d_core.py']
for f in needed_files:
    if os.path.exists(os.path.join(_root, f)):
        ok(f'{f} present')
    else:
        fail(f'{f} NOT present')

# Output-Verzeichnis
_results = os.path.join(_root, 'results')
if not os.path.exists(_results):
    os.makedirs(_results)
    ok('results/ directory created')
else:
    ok('results/ directory present')

# Speicher
print('\n--- Memory estimate ---')
total_ram_mb = None
try:
    import psutil
    total_ram_mb = psutil.virtual_memory().total // (1024**2)
    ok(f'RAM total: {total_ram_mb // 1024} GB')
except ImportError:
    print('  [INFO] psutil not installed, RAM check skipped')
    print('         pip install psutil (optional)')

# Zusammenfassung
print('\n' + '='*60)
if missing:
    print(f'ERROR: {len(missing)} package(s) missing!')
    for pkg in missing:
        print(f'  -> {REQUIRED[pkg]}')
    sys.exit(1)
else:
    print('ALL CHECKS PASSED — ready for simulation!')
    print('\nNext steps:')
    print('  GUI:  python run_gui.py')
    print('  CLI:  python run_simulation.py --geometry planar --dim 2 --method full')
    print('        python run_simulation.py --geometry lens   --dim 3 --scenario Gesund')
    print('        python run_simulation.py --help          (all options)')
    sys.exit(0)
