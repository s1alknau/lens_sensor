"""Freely configurable layer stack below the waveguide core (planar geometry).

A stack is a list of layers ordered from the core DOWNWARD (layer 0 touches the core
bottom at y = 0). Each layer is a dict

    {'name': str, 't_um': float (> 0), 'n': complex, 'kappa': float, 'mat': str|None}

(``mat`` = material name when n was given as a material, else None.)

with n = n_real + i*kappa (kappa >= 0 = absorption). Below the last layer comes the
semi-infinite substrate (--n-cornea / layered cornea profile, unchanged); above the
core the air (3D --air) - both are NOT part of the stack.

CLI form (``--layers``):  ``Name:thickness_um:n[,Name:thickness_um:n ...]``

  * n is a number (``1.333``), a complex number (``1.333+0.0001j``) or a material
    name known to ``common.physics.n_at`` (``aqueous``, ``mucin``, ``lipid``, ``pmma``,
    ``polystyrol`` ...), evaluated at the run's wavelength.
  * ``none`` (or an empty string) = no layers: the core sits directly on the substrate.
  * Example: ``--layers "Lipid:0.03:lipid,Aqueous:3.5:aqueous,Mucin:0.5:mucin"`` is the
    default physiological tear film.

Legacy runs (no ``--layers``) use the three fixed layers lipid / aqueous / mucin
(``legacy_stack``); zero-thickness layers are dropped.

Saved files carry the stack as ``stack_names`` / ``stack_t_um`` / ``stack_n_re`` /
``stack_n_im`` (``stack_to_npz`` / ``stack_from_npz``); the old 4-value ``layers`` key is
still written (``legacy_layers_tuple``) so older analyzers keep working.
"""
import numpy as np

LEGACY_NAMES = ('Lipid', 'Aqueous', 'Mucin')


def _parse_n(tok, lam_nm):
    """-> (n complex, material name or None). A material name keeps its dispersion
    (``mat``), so pulse/spectrum runs can evaluate it at other wavelengths."""
    tok = tok.strip()
    try:
        v = complex(tok.replace(' ', ''))
        return complex(v.real, abs(v.imag)), None
    except ValueError:
        from common.physics import n_at
        return complex(n_at(tok.lower(), lam_nm), 0.0), tok.lower()


def parse_layer_stack(spec, lam_nm):
    """``--layers`` string -> stack (list of layer dicts). Raises ValueError with a
    readable message on bad input."""
    if spec is None:
        return None
    s = str(spec).strip()
    if s.lower() in ('', 'none', '-'):
        return []
    stack = []
    for i, part in enumerate(s.split(',')):
        f = [p.strip() for p in part.split(':')]
        if len(f) != 3:
            raise ValueError(f'--layers entry {i + 1} "{part}": expected Name:thickness_um:n')
        name, t, n = f
        try:
            t_um = float(t)
        except ValueError:
            raise ValueError(f'--layers entry {i + 1} "{part}": thickness "{t}" is not a number')
        if t_um <= 0:
            continue                                   # zero thickness = layer omitted
        try:
            nc, mat = _parse_n(n, lam_nm)
        except Exception:
            raise ValueError(f'--layers entry {i + 1} "{part}": n "{n}" is neither a number '
                             f'nor a known material')
        stack.append(dict(name=name or f'Layer {i + 1}', t_um=t_um, n=nc, kappa=nc.imag,
                          mat=mat))
    return stack


def legacy_stack(t_lip_um, n_lip, t_aq_um, n_aq, t_mu_um, n_mu, names=None):
    """Old fixed three-layer tear film -> stack (zero-thickness layers dropped)."""
    names = list(names) if names else list(LEGACY_NAMES)
    out = []
    for nm, t, n, mat in zip(names, (t_lip_um, t_aq_um, t_mu_um), (n_lip, n_aq, n_mu),
                             ('lipid', 'aqueous', 'mucin')):
        if t is None or float(t) <= 1e-9:
            continue
        nc = complex(n) if n is not None else complex(1.333, 0.0)
        out.append(dict(name=str(nm), t_um=float(t), n=nc, kappa=nc.imag, mat=mat))
    return out


def stack_total_um(stack):
    return float(sum(L['t_um'] for L in stack or []))


def stack_bounds(stack, y_top=0.0):
    """[(name, y_top_um, y_bot_um, n_complex), ...] walking down from ``y_top``."""
    out, y = [], float(y_top)
    for L in stack or []:
        out.append((L['name'], y, y - L['t_um'], complex(L['n'])))
        y -= L['t_um']
    return out


def stack_eps_profile(y_um, stack, eps_sub, y_top=0.0):
    """Relative permittivity (complex) below the core for positions ``y_um`` (< y_top):
    layer eps inside the stack, ``eps_sub(depth_below_stack_um)`` (array -> array) below
    it. Positions >= y_top are returned as NaN (core/air are handled by the caller)."""
    y = np.asarray(y_um, dtype=np.float64)
    e = np.full(y.shape, np.nan, dtype=np.complex128)
    y_bot = y_top - stack_total_um(stack)
    for _nm, yt, yb, n in stack_bounds(stack, y_top):
        e[(y < yt) & (y >= yb)] = n*n
    m = y < y_bot
    if np.any(m):
        e[m] = np.asarray(eps_sub(y_bot - y[m]), dtype=np.complex128)
    return e


def stack_to_npz(stack):
    st = list(stack or [])
    return dict(stack_names=np.array([L['name'] for L in st] or [''], dtype=str)[:len(st)],
                stack_t_um=np.array([L['t_um'] for L in st], dtype=np.float64),
                stack_n_re=np.array([complex(L['n']).real for L in st], dtype=np.float64),
                stack_n_im=np.array([complex(L['n']).imag for L in st], dtype=np.float64))


def stack_from_npz(d):
    """Stack from a loaded npz (``d.files``); None if the file has no stack keys."""
    files = set(getattr(d, 'files', d.keys() if hasattr(d, 'keys') else []))
    if 'stack_t_um' not in files:
        return None
    t = np.atleast_1d(np.asarray(d['stack_t_um'], dtype=np.float64))
    names = [str(s) for s in np.atleast_1d(np.asarray(d['stack_names']))] if 'stack_names' in files else []
    nre = np.atleast_1d(np.asarray(d['stack_n_re'], dtype=np.float64)) if 'stack_n_re' in files else np.ones_like(t)
    nim = np.atleast_1d(np.asarray(d['stack_n_im'], dtype=np.float64)) if 'stack_n_im' in files else np.zeros_like(t)
    out = []
    for i in range(len(t)):
        n = complex(nre[i], nim[i])
        out.append(dict(name=names[i] if i < len(names) else f'Layer {i + 1}',
                        t_um=float(t[i]), n=n, kappa=n.imag))
    return out


def legacy_layers_tuple(stack, n_default=1.333):
    """Old 4-value ``layers`` key (t1_m, t2_m, t3_m, n2) from the first three layers,
    for analyzers that do not know the stack keys (best effort)."""
    st = list(stack or [])
    t = [st[i]['t_um']*1e-6 if i < len(st) else 0.0 for i in range(3)]
    n2 = complex(st[1]['n']).real if len(st) > 1 else (complex(st[0]['n']).real if st else n_default)
    return (t[0], t[1], t[2], n2)


def describe(stack):
    """One-line human-readable summary for console output."""
    if not stack:
        return 'no layers (core directly on the substrate)'
    parts = []
    for L in stack:
        n = complex(L['n'])
        ns = f'{n.real:.4f}' + (f'+{n.imag:.2e}i' if n.imag else '')
        parts.append(f'{L["name"]} {L["t_um"]:g} um (n={ns})')
    return ' | '.join(parts)
