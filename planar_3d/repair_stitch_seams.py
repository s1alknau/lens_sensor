"""Repariert Stitch-Naht-Artefakte (weisse Streifen) in bereits gerechneten
3D-Ergebnissen NACHTRAEGLICH - ohne Neurechnung.

Die weissen Streifen entstanden, weil beim Assemblieren die (gedaempfte)
Sponge-Zone jedes Fensters mitkopiert wurde -> schmale gedaempfte Spalten an
den Fenstergrenzen. Dieses Skript erkennt diese Spalten (Energie-Einbruch ueber
die GANZE Querschnittshoehe) und ersetzt sie durch lineare Interpolation aus den
sauberen Nachbarspalten. KOSMETISCHE Reparatur: an den (duennen) Nahtspalten war
das Feld gedaempft, die exakte Information ist dort verloren - die Interpolation
ist eine sehr gute Naeherung fuer Darstellung und Auswertung abseits der Spalten.

Aufruf (im Ordner planar_3d):
    python repair_stitch_seams.py            # alle *stitch*_vol3d.npz in results\
    python repair_stitch_seams.py --lowram   # speicherschonend (Memmap-Streaming)
Erzeugt je Datei eine *_repariert.npz-Kopie (Original bleibt erhalten).
"""
import numpy as np, os, sys, glob, zipfile, shutil


def detect_seams(frames_npz):
    """Naht-Spalten aus der (kleinen) z-Mittelebenen-Slice erkennen."""
    d = np.load(frames_npz, allow_pickle=True)
    Ez = np.asarray(d['Ez']).astype(np.float32); d.close()
    nfr, Nx, Ny = Ez.shape
    e = (Ez**2).sum(axis=(0, 2))              # Energie pro x-Spalte (ueber Frames + y)
    med = np.median(e[e > 0]) if np.any(e > 0) else 0.0
    low = e < 0.2*med
    edge = max(3, Nx//40)
    low[:edge] = False; low[-edge:] = False   # echte Domaenenraender aussparen
    runs = []; i = 0
    while i < Nx:
        if low[i]:
            j = i
            while j < Nx and low[j]:
                j += 1
            if (j - i) <= 40:                 # nur schmale Streifen (Sponge-Breite)
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def inpaint(arr, xaxis, runs):
    """Naht-Spalten entlang xaxis durch lineare Interpolation ersetzen."""
    Nx = arr.shape[xaxis]
    for a, b in runs:
        la = max(0, a - 1); rb = min(Nx - 1, b + 1); span = (rb - la) or 1

        def sl(k):
            return tuple(k if ax == xaxis else slice(None) for ax in range(arr.ndim))
        L = np.asarray(arr[sl(la)]).astype(np.float32)
        Rr = np.asarray(arr[sl(rb)]).astype(np.float32)
        for c in range(a, b + 1):
            w = (c - la)/span
            arr[sl(c)] = ((1 - w)*L + w*Rr).astype(arr.dtype)


def repair_frames(p, runs):
    d = np.load(p, allow_pickle=True); kw = {k: np.asarray(d[k]) for k in d.files}; d.close()
    for key, ax in (('Ez', 1), ('Ez_cw', 1)):
        if key in kw:
            a = kw[key].astype(kw[key].dtype); inpaint(a, ax, runs); kw[key] = a
    out = p.replace('.npz', '_repariert.npz')
    np.savez_compressed(out, **kw); print('  Slice ->', os.path.basename(out))


def repair_vol(p, runs, lowram=False):
    out = p.replace('.npz', '_repariert.npz')
    if not lowram:
        try:
            d = np.load(p, allow_pickle=True); kw = {k: np.asarray(d[k]) for k in d.files}; d.close()
            for key, ax in (('Ez', 1), ('Ez_cw', 1), ('Iavg', 0)):
                if key in kw:
                    a = kw[key]; inpaint(a, ax, runs); kw[key] = a
            np.savez_compressed(out, **kw); print('  Volumen ->', os.path.basename(out)); return
        except MemoryError:
            print('  [RAM knapp -> Streaming-Modus]')
    # Streaming: grosse Arrays per Memmap reparieren, kleine Metadaten kopieren
    zin = zipfile.ZipFile(p); big = {'Ez.npy': 1, 'Ez_cw.npy': 1, 'Iavg.npy': 0}; tmps = {}
    tdir = os.path.dirname(os.path.abspath(out))
    for name, ax in big.items():
        if name not in zin.namelist():
            continue
        tp = os.path.join(tdir, '_tmp_' + name)
        with zin.open(name) as s, open(tp, 'wb') as o:
            shutil.copyfileobj(s, o, 1 << 24)
        arr = np.lib.format.open_memmap(tp, mode='r+'); inpaint(arr, ax, runs); arr.flush(); del arr
        tmps[name] = tp
    zo = zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED)
    for name in zin.namelist():
        if name in tmps:
            continue
        zo.writestr(name, zin.read(name))
    for name, tp in tmps.items():
        zo.write(tp, name); os.remove(tp)
    zo.close(); zin.close(); print('  Volumen ->', os.path.basename(out), '(unkomprimiert)')


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    resdir = os.path.join(here, 'results')
    lowram = '--lowram' in sys.argv
    vols = sorted(glob.glob(os.path.join(resdir, '*stitch_vol3d.npz')))
    if not vols:
        print('Keine *stitch_vol3d.npz in', resdir); sys.exit(0)
    for v in vols:
        if '_repariert' in v:
            continue
        fr = v.replace('_vol3d.npz', '_frames.npz')
        print(os.path.basename(v))
        if not os.path.exists(fr):
            print('  [WARN] zugehoerige _frames.npz fehlt -> uebersprungen'); continue
        runs = detect_seams(fr)
        print('  Nahtstellen (Spalten-Runs):', runs)
        repair_frames(fr, runs)
        repair_vol(v, runs, lowram=lowram)
    print('FERTIG')
