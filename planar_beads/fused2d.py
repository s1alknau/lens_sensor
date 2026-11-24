"""Fusionierte CUDA-Kernel (CuPy) fuer die 2D-CPML-Yee-Updates (Review: Performance).

Die NumPy/CuPy-Ausdrucksform eines CPML-Updates erzeugt ~20 Temporaer-Arrays und
ebenso viele Speicher-Durchlaeufe pro Zeitschritt (FDTD ist speicherbandbreiten-
begrenzt). Hier ist jedes Update EIN Kernel: jedes Feldelement wird einmal gelesen
und geschrieben. Mathematisch identisch zu boundaries.CPML_TE/CPML_TM (gleiche
Reihenfolge der Operationen, float32); bei NumPy-Backend nicht aktiv.

Indexkonventionen (C-Order, Ny = Knotenzahl in y):
  TE: Ez (Nx,Ny), Hx (Nx,Ny-1), Hy (Nx-1,Ny)
  TM: Hz (Nx-1,Ny-1), Ex (Nx-1,Ny), Ey (Nx,Ny-1)
"""
import numpy as np

try:
    import cupy as cp
    _OK = True
except Exception:                                    # pragma: no cover
    cp = None
    _OK = False

ENABLED = True       # global abschaltbar (Vergleichsmessungen)

_K = {}


def available(*arrays):
    """Fusion nur fuer CuPy-Arrays, C-zusammenhaengend, float32."""
    if not (ENABLED and _OK):
        return False
    for a in arrays:
        if not isinstance(a, cp.ndarray) or a.dtype != cp.float32 or not a.flags.c_contiguous:
            return False
    return True


def _k(name):
    if name in _K:
        return _K[name]
    EK = cp.ElementwiseKernel
    if name == 'te_hx':
        k = EK('raw float32 Ez, raw float32 b, raw float32 c, raw float32 kap, float32 Ch, int32 Ny',
               'raw float32 Hx, raw float32 psi',
               '''int a = i / (Ny - 1); int j = i - a*(Ny - 1);
                  float d = Ez[a*Ny + j + 1] - Ez[a*Ny + j];
                  float p = b[j]*psi[i] + c[j]*d; psi[i] = p;
                  Hx[i] -= Ch*(d/kap[j] + p);''', 'cpml_te_hx')
    elif name == 'te_hy':
        k = EK('raw float32 Ez, raw float32 b, raw float32 c, raw float32 kap, float32 Ch, int32 Ny',
               'raw float32 Hy, raw float32 psi',
               '''int a = i / Ny; int j = i - a*Ny;
                  float d = Ez[(a + 1)*Ny + j] - Ez[a*Ny + j];
                  float p = b[a]*psi[i] + c[a]*d; psi[i] = p;
                  Hy[i] += Ch*(d/kap[a] + p);''', 'cpml_te_hy')
    elif name == 'te_ez':
        k = EK('raw float32 Hx, raw float32 Hy, raw float32 CeE, raw float32 CeH, '
               'raw float32 bex, raw float32 cex, raw float32 kex, '
               'raw float32 bey, raw float32 cey, raw float32 key, int32 Ny',
               'raw float32 Ez, raw float32 px, raw float32 py',
               '''int a = i / (Ny - 2); int j = i - a*(Ny - 2);
                  int ga = a + 1; int gb = j + 1;
                  float dHy = Hy[ga*Ny + gb] - Hy[a*Ny + gb];
                  float dHx = Hx[ga*(Ny - 1) + gb] - Hx[ga*(Ny - 1) + j];
                  float qx = bex[ga]*px[i] + cex[ga]*dHy; px[i] = qx;
                  float qy = bey[gb]*py[i] + cey[gb]*dHx; py[i] = qy;
                  int g = ga*Ny + gb;
                  Ez[g] = CeE[g]*Ez[g] + CeH[g]*(dHy/kex[ga] + qx - dHx/key[gb] - qy);''',
               'cpml_te_ez')
    elif name == 'tm_hz':
        k = EK('raw float32 Ex, raw float32 Ey, raw float32 bhx, raw float32 chx, raw float32 khx, '
               'raw float32 bhy, raw float32 chy, raw float32 khy, float32 Ch, int32 Ny',
               'raw float32 Hz, raw float32 pzx, raw float32 pzy',
               '''int a = i / (Ny - 1); int j = i - a*(Ny - 1);
                  float dEyx = Ey[(a + 1)*(Ny - 1) + j] - Ey[a*(Ny - 1) + j];
                  float dExy = Ex[a*Ny + j + 1] - Ex[a*Ny + j];
                  float qx = bhx[a]*pzx[i] + chx[a]*dEyx; pzx[i] = qx;
                  float qy = bhy[j]*pzy[i] + chy[j]*dExy; pzy[i] = qy;
                  Hz[i] -= Ch*((dEyx/khx[a] + qx) - (dExy/khy[j] + qy));''', 'cpml_tm_hz')
    elif name == 'tm_ex':
        k = EK('raw float32 Hz, raw float32 Ca, raw float32 b, raw float32 c, raw float32 kap, int32 Ny',
               'raw float32 Ex, raw float32 psi',
               '''int a = i / (Ny - 2); int j = i - a*(Ny - 2);
                  float d = Hz[a*(Ny - 1) + j + 1] - Hz[a*(Ny - 1) + j];
                  float p = b[j + 1]*psi[i] + c[j + 1]*d; psi[i] = p;
                  int g = a*Ny + j + 1;
                  Ex[g] += Ca[g]*(d/kap[j + 1] + p);''', 'cpml_tm_ex')
    elif name == 'tm_ey':
        k = EK('raw float32 Hz, raw float32 Ca, raw float32 b, raw float32 c, raw float32 kap, int32 Ny',
               'raw float32 Ey, raw float32 psi',
               '''int a = i / (Ny - 1); int j = i - a*(Ny - 1);
                  float d = Hz[(a + 1)*(Ny - 1) + j] - Hz[a*(Ny - 1) + j];
                  float p = b[a + 1]*psi[i] + c[a + 1]*d; psi[i] = p;
                  int g = (a + 1)*(Ny - 1) + j;
                  Ey[g] += -Ca[g]*(d/kap[a + 1] + p);''', 'cpml_tm_ey')
    else:
        raise KeyError(name)
    _K[name] = k
    return k


def _i(n):
    return np.int32(n)


def te_update_H(p, Ez, Hx, Hy, Ch):
    Ny = Ez.shape[1]; Ch = np.float32(Ch)
    _k('te_hx')(Ez, p.bhy, p.chy, p.khy, Ch, _i(Ny), Hx, p.psi_Hxy, size=Hx.size)
    _k('te_hy')(Ez, p.bhx, p.chx, p.khx, Ch, _i(Ny), Hy, p.psi_Hyx, size=Hy.size)


def te_update_E(p, Ez, Hx, Hy, Ce_E, Ce_H):
    Nx, Ny = Ez.shape
    _k('te_ez')(Hx, Hy, Ce_E, Ce_H, p.bex, p.cex, p.kex, p.bey, p.cey, p.key, _i(Ny),
                Ez, p.psi_Ezx, p.psi_Ezy, size=(Nx - 2)*(Ny - 2))


def tm_update_H(p, Ex, Ey, Hz, Ch):
    Ny = Ex.shape[1]
    _k('tm_hz')(Ex, Ey, p.bhx, p.chx, p.khx, p.bhy, p.chy, p.khy, np.float32(Ch), _i(Ny),
                Hz, p.psi_Hzx, p.psi_Hzy, size=Hz.size)


def tm_update_E(p, Ex, Ey, Hz, Ca_ex, Ca_ey):
    Ny = Ex.shape[1]; Nx = Ey.shape[0]
    _k('tm_ex')(Hz, Ca_ex, p.bey, p.cey, p.key, _i(Ny), Ex, p.psi_Exy, size=(Nx - 1)*(Ny - 2))
    _k('tm_ey')(Hz, Ca_ey, p.bex, p.cex, p.kex, _i(Ny), Ey, p.psi_Eyx, size=(Nx - 2)*(Ny - 1))
