"""Charakterisierungs-Test (Golden Master) fuer die FDTD-Solver.

Zweck: ein SICHERHEITSNETZ fuer Refactorings. Der Test laeuft je einen winzigen
2D- und 3D-Lauf auf der CPU (deterministisch, wenige Sekunden) und vergleicht
reduzierte Kennzahlen (Transmission, Leistungen, Feld-Summen/Extremwerte) mit
fest hinterlegten Referenzwerten. Aendert ein Refactor die Physik, weicht eine
dieser Zahlen ab und der Test schlaegt fehl.

Die Referenzwerte wurden auf CPU (NumPy) erfasst. Sie beschreiben das VERHALTEN,
nicht die physikalische "Richtigkeit" - genau das braucht ein Golden Master.

Ausfuehren:
    python tests/test_characterization.py      # Standalone (druckt PASS/FAIL)
    pytest tests/test_characterization.py       # oder via pytest
"""
import os
import sys
import shutil
import tempfile
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("planar_3d", "planar_beads"):
    _p = os.path.join(_REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Toleranz: die Arithmetik ist auf derselben Maschine bit-deterministisch; die
# Extraktionen in Tier 1/2 aendern die Rechenreihenfolge nicht. rtol laesst nur
# vernachlaessigbare Plattform-/Version-Unterschiede zu, faengt aber jede echte
# Verhaltensaenderung.
RTOL = 1e-6

# ---- Referenzwerte (auf CPU/NumPy erfasst) --------------------------------
# Neu erfasst (2026-11-27) nach: rigoroser VCSEL-Quelle vor der Facette, 3D full mit
# Schichtstapel vor der Facette wie Stitch/2D; davor: exakter E/H-Poynting-Transmission, Eingangsebene stromab der
# Quelle, versetzter eps-/Daempfungs-Mittelung in 3D und CPML als 3D-Standard.
GOLD_3D = dict(
    transmission=0.9435572692291655,
    P_in=1.6187998931632774e-16,
    P_out=1.2167653597534091e-16,
    Iavg=dict(shape=[40, 40, 19], sum=5642.517313088292,
              absmax=4.259119510650635, mean=0.18560912214106223),
    vols=dict(shape=[2, 40, 40, 19], sum=609.2458105086328,
              absmax=2.801161289215088, mean=0.010020490304418301),
)
GOLD_2D = dict(
    transmission=0.8439352444232754,
    P_in=5.402917919781816e-10,
    P_out=6.301416977012967e-10,
    frames=dict(shape=[2, 80, 160], sum=-43.32355685241299,
                absmax=3.02363657951355, mean=-0.0016923264395473825),
)
# 3D-Stitch (Gebiets-Zerlegung). Die CW-Frames sind SIGNIERTE Phasen, deren
# Volumen-Summe nahe Null liegt (Ausloeschung) -> fuer 'vols' nur shape+absmax
# pinnen (robust), nicht sum/mean.
GOLD_3D_STITCH = dict(
    transmission=0.8222802982518829,
    P_in=4.0199279785156264e-20,
    P_out=3.551814651489259e-20,
    Iavg=dict(shape=[40, 40, 19], sum=5543.204919499725,
              absmax=4.195054531097412, mean=0.18234226708880674),
    vols_shape=[2, 40, 40, 19],
    vols_absmax=2.483750104904175,
)


def _reduce(a):
    a = np.asarray(a, dtype=np.float64)
    return dict(shape=list(a.shape), sum=float(np.sum(a)),
                absmax=float(np.max(np.abs(a))), mean=float(np.mean(a)))


def _run_3d_case():
    import fdtd3d_core as f3d
    r = f3d.run_3d(
        label="golden3d", wg_n=1.491,
        lx_um=4.0, air_um=1.0, tear_um=2.0, lz_um=2.0,
        dx_nm=100.0, t_wg_um=1.0, t_aq_um=1.0, t_mu_um=0.5,
        lam_nm=850.0, n_snapshots=2, steps_factor=1.0,
        vcsel_waist_um=1.0, vcsel_waist_z_um=1.0,
        bead=dict(x_um=2.0, d_um=0.5, n=1.59), n_sponge=8)
    return dict(transmission=r["transmission"], P_in=r["P_in"],
                P_out=r["P_out"], Iavg=_reduce(r["Iavg"]),
                vols=_reduce(np.asarray(r["vols_array"])))


def _run_3d_stitch_case():
    import fdtd3d_core as f3d
    r = f3d.run_3d_stitched(
        label="goldenstitch", wg_n=1.491, window_w_um=2.0, slide_um=1.2,
        lx_um=4.0, air_um=1.0, tear_um=2.0, lz_um=2.0, dx_nm=100.0,
        t_wg_um=1.0, t_aq_um=1.0, t_mu_um=0.5, lam_nm=850.0, n_snapshots=2,
        vcsel_waist_um=1.0, vcsel_waist_z_um=1.0,
        bead=dict(x_um=2.0, d_um=0.5, n=1.59), n_sponge=8)
    vols = np.asarray(r["vols_array"])
    return dict(transmission=r["transmission"], P_in=r["P_in"],
                P_out=r["P_out"], Iavg=_reduce(r["Iavg"]),
                vols_shape=list(vols.shape),
                vols_absmax=float(np.max(np.abs(vols.astype(np.float64)))))


def _run_2d_case():
    import fdtd2d_core as d2
    r = d2.run_beads(
        wg_mat="pmma", bead_mat="polystyrol", bead_d_um=0.5,
        dx_nm=100.0, save_frames=True, n_snapshots=2,
        wg_thickness_um=1.0, lambda_nm=850.0, method="full",
        length_um=8.0, bead_x_um=4.0)
    frames = r.get("frames") or []
    mm = r.get("frames_memmap")
    if mm and os.path.exists(mm):
        ez = np.asarray(np.load(mm, mmap_mode="r")[:len(frames)])
    else:
        ez = np.zeros(0)
    return dict(transmission=r["transmission"], P_in=r["P_in"],
                P_out=r["P_out"], frames=_reduce(ez))


def _assert_scalar(name, got, want):
    np.testing.assert_allclose(
        got, want, rtol=RTOL,
        err_msg=f"{name}: got {got!r} != golden {want!r}")


def _assert_reduce(name, got, want):
    assert got["shape"] == want["shape"], \
        f"{name}.shape: got {got['shape']} != golden {want['shape']}"
    for key in ("sum", "absmax", "mean"):
        _assert_scalar(f"{name}.{key}", got[key], want[key])


def _check_all():
    """Fuehrt beide Laeufe in einem temporaeren cwd aus und vergleicht alles."""
    prev_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="lens_golden_")
    try:
        os.chdir(tmp)
        got3 = _run_3d_case()
        got3s = _run_3d_stitch_case()
        got2 = _run_2d_case()
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    _assert_scalar("3D.transmission", got3["transmission"], GOLD_3D["transmission"])
    _assert_scalar("3D.P_in", got3["P_in"], GOLD_3D["P_in"])
    _assert_scalar("3D.P_out", got3["P_out"], GOLD_3D["P_out"])
    _assert_reduce("3D.Iavg", got3["Iavg"], GOLD_3D["Iavg"])
    _assert_reduce("3D.vols", got3["vols"], GOLD_3D["vols"])

    _assert_scalar("3Dstitch.transmission", got3s["transmission"], GOLD_3D_STITCH["transmission"])
    _assert_scalar("3Dstitch.P_in", got3s["P_in"], GOLD_3D_STITCH["P_in"])
    _assert_scalar("3Dstitch.P_out", got3s["P_out"], GOLD_3D_STITCH["P_out"])
    _assert_reduce("3Dstitch.Iavg", got3s["Iavg"], GOLD_3D_STITCH["Iavg"])
    assert got3s["vols_shape"] == GOLD_3D_STITCH["vols_shape"], \
        f"3Dstitch.vols.shape: got {got3s['vols_shape']} != golden {GOLD_3D_STITCH['vols_shape']}"
    _assert_scalar("3Dstitch.vols.absmax", got3s["vols_absmax"], GOLD_3D_STITCH["vols_absmax"])

    _assert_scalar("2D.transmission", got2["transmission"], GOLD_2D["transmission"])
    _assert_scalar("2D.P_in", got2["P_in"], GOLD_2D["P_in"])
    _assert_scalar("2D.P_out", got2["P_out"], GOLD_2D["P_out"])
    _assert_reduce("2D.frames", got2["frames"], GOLD_2D["frames"])


def test_characterization_3d_and_2d():
    """pytest-Einstieg: prueft, dass 2D- und 3D-Solver unveraendertes Verhalten
    zeigen (Golden Master)."""
    _check_all()


if __name__ == "__main__":
    try:
        _check_all()
    except AssertionError as e:
        print("\n[FAIL] Charakterisierung abgewichen:\n ", e)
        sys.exit(1)
    print("\n[PASS] Charakterisierung unveraendert (2D full + 3D full + 3D stitch).")
    sys.exit(0)
