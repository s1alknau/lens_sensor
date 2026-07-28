"""Zentrale Backend-Auswahl: CuPy (NVIDIA-GPU) mit automatischem NumPy-Fallback.

Alle Solver importieren ``xp``, ``GPU_AVAILABLE``, ``to_np`` (und bei Bedarf
``cp``) von hier - statt den identischen try/except-Block je Datei zu wiederholen.
Das Verhalten ist gleich wie zuvor; ``cp`` ist ``None``, wenn keine GPU vorhanden.
"""
import numpy as np

try:
    import cupy as cp
    xp = cp
    GPU_AVAILABLE = True
    print('[Backend] CuPy detected - using NVIDIA GPU')
    # Pinned-Memory-Allokator abschalten: umgeht "cudaErrorAlreadyMapped", das
    # bei grossen Host->Device-Transfers (v.a. nach abgebrochenen Laeufen) auftritt.
    # Kostet minimal Transfer-Speed, ist aber robust. Transfers sind hier selten.
    try:
        cp.cuda.set_pinned_memory_allocator(None)
    except Exception:
        pass
except ImportError:
    cp = None
    xp = np
    GPU_AVAILABLE = False
    print('[Backend] CuPy not available - fallback to NumPy (CPU)')


def to_np(a):
    """Bringt ein Array garantiert nach NumPy (CuPy -> Host, sonst durchgereicht)."""
    return a.get() if (GPU_AVAILABLE and hasattr(a, 'get')) else np.asarray(a)
