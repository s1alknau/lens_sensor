"""Automatic stitch window planning (used when --window-um / --slide-um are not given).

Rules
-----
1. Size from memory: the window is the largest that fits the memory budget
   (3D: free GPU memory, ~26 float32 per cell (+5 with absorption) as in
   ``check_resources_3d``; 2D: a fixed cell budget, the 2D grid is cheap). If the whole
   domain fits, ONE window = the full domain (a full run with the stitch machinery).
2. Never larger than the domain (the old 2D default of 350 um was longer than short
   waveguides -> reflection at the waveguide end inside the window).
3. Overlap >= max(MIN_OVERLAP_UM, 40 % of the window): the hand-off and the absorber
   of the previous window must lie in the overlap.
4. Seams away from features: a window boundary / overlap region must not contain a
   feature (bead, facet, end facet, CAD body edges) within FEATURE_MARGIN_UM. The slide
   is reduced in small steps until no seam hits a feature (window count may grow).

The solvers keep their uniform layout (window w starts at min(w*slide, L - window)),
so only (window_um, slide_um) are chosen here.
"""
import math

MIN_OVERLAP_UM = 4.0
OVERLAP_FRAC = 0.3
FEATURE_MARGIN_UM = 1.0
BYTES_PER_CELL_3D = 26*4
BYTES_PER_CELL_3D_LOSSY = 31*4
BYTES_PER_CELL_2D = 48       # 2D: fields, coefficients, CPML, DFT accumulators


def free_gpu_bytes():
    try:
        import cupy as cp
        free, _total = cp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return None


def max_window_um(dim, dx_um, cross_cells, lossy=False, budget_frac=0.6):
    """Largest window length [um] that fits the memory budget."""
    free = free_gpu_bytes()
    if free is None:                                 # CPU: be conservative (4 GB)
        free = 4e9
    if dim == 2:
        bpc = BYTES_PER_CELL_2D
    else:
        bpc = BYTES_PER_CELL_3D_LOSSY if lossy else BYTES_PER_CELL_3D
    return budget_frac*free/(bpc*max(cross_cells, 1))*dx_um


def seams(length_um, window_um, slide_um):
    """Overlap regions [start, end] (um) where window w>0 takes over from w-1."""
    if window_um >= length_um:
        return []
    n_win = int(math.ceil((length_um - window_um)/slide_um)) + 1
    out = []
    for w in range(1, n_win):
        x0 = min(w*slide_um, length_um - window_um)
        out.append((x0, x0 + (window_um - slide_um)))
    return out


def _conflicts(sm, features, margin):
    return [(f, s) for s in sm for f in features
            if s[0] - margin <= f <= s[1] + margin]


def plan(length_um, dx_um, dim, cross_cells, features=(), lossy=False,
         window_um=None, slide_um=None, wmax_um=None):
    """-> dict(window_um, slide_um, n_win, seams, conflicts, reason)."""
    L = float(length_um)
    feats = sorted(float(f) for f in features if f is not None)
    if window_um is not None and slide_um is not None:       # user choice: only report
        w = min(float(window_um), L)
        s = min(float(slide_um), w)
        sm = seams(L, w, s)
        return dict(window_um=w, slide_um=s, n_win=len(sm) + 1, seams=sm,
                    conflicts=_conflicts(sm, feats, FEATURE_MARGIN_UM), reason='user')
    wmax = wmax_um if wmax_um is not None else max_window_um(dim, dx_um, cross_cells, lossy)
    if wmax >= L:
        return dict(window_um=L, slide_um=L, n_win=1, seams=[], conflicts=[],
                    reason=f'whole domain fits in memory (max window ~{wmax:.0f} um)')
    w = max(2*MIN_OVERLAP_UM, math.floor(wmax/dx_um)*dx_um)
    ov = max(MIN_OVERLAP_UM, OVERLAP_FRAC*w)
    s0 = max(dx_um, w - ov)
    best = None
    n_try = 400                                         # slide candidates s0 ... 0.5 s0
    for i in range(n_try + 1):
        s = max(dx_um, math.floor((s0*(1.0 - 0.5*i/n_try))/dx_um)*dx_um)
        sm = seams(L, w, s)
        c = _conflicts(sm, feats, FEATURE_MARGIN_UM)
        if best is None or len(c) < len(best[2]):
            best = (s, sm, c)
        if not c:
            break
    s, sm, c = best
    return dict(window_um=w, slide_um=s, n_win=len(sm) + 1, seams=sm, conflicts=c,
                reason=f'memory limit (max window ~{wmax:.0f} um), overlap {w - s:.1f} um'
                       + ('' if not c else f'; {len(c)} seam(s) still near a feature'))


def describe(p):
    lines = [f'  Stitch plan: {p["n_win"]} window(s) of {p["window_um"]:.1f} um, slide '
             f'{p["slide_um"]:.1f} um ({p["reason"]})']
    if p['seams']:
        sm = p['seams']
        txt = ', '.join(f'{a:.1f}-{b:.1f}' for a, b in (sm if len(sm) <= 8 else sm[:3] + sm[-2:]))
        lines.append(f'  Seams (overlap regions, um): {txt}' + (f'  ({len(sm)} in total)' if len(sm) > 8 else ''))
    for f, s in p['conflicts']:
        lines.append(f'  [WARN] feature at x={f:.2f} um lies in/near seam {s[0]:.1f}-{s[1]:.1f} um')
    return '\n'.join(lines)
