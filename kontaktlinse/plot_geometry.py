"""Geometrie-Schema (übersichtlich, keine überlappenden Texte)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch
from config import *

os.makedirs('results', exist_ok=True)


# ------- Hilfsfunktionen -------
def _label_box(ax, x, y, text, color='black', fontsize=8, ha='center', va='center'):
    """Saubere Label-Box mit weißem Hintergrund."""
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, lw=0.8, alpha=0.9))


def _det_rect_tangential(x0_um, side, L_um, h_um, buf_um, d_norm_minimum=50):
    """Erzeugt Buffer + Detektor-Polygon tangential zur Lens an x0."""
    x0_m = x0_um*1e-6
    sag = R_BEND - np.sqrt(R_BEND**2 - x0_m**2)
    y_mid_um = -sag*1e6
    slope = -x0_m/np.sqrt(R_BEND**2 - x0_m**2)
    theta = np.arctan(slope)
    ct, st = np.cos(theta), np.sin(theta)
    max_y_shift = L_um/2*abs(st) + h_um/2*abs(ct)
    d_norm = max(max_y_shift/abs(ct), d_norm_minimum)
    if side == 'lower':
        x_c, y_c = x0_um + st*d_norm, y_mid_um - ct*d_norm
    else:
        x_c, y_c = x0_um - st*d_norm, y_mid_um + ct*d_norm
    def rot(u, v):
        return (x_c + ct*u - st*v, y_c + st*u + ct*v)
    det = [rot(-L_um/2, -h_um/2), rot(L_um/2, -h_um/2),
           rot(L_um/2, h_um/2), rot(-L_um/2, h_um/2)]
    if side == 'lower':
        buf = [rot(-L_um/2, -h_um/2 - buf_um), rot(L_um/2, -h_um/2 - buf_um),
               rot(L_um/2, -h_um/2), rot(-L_um/2, -h_um/2)]
    else:
        buf = [rot(-L_um/2, h_um/2), rot(L_um/2, h_um/2),
               rot(L_um/2, h_um/2 + buf_um), rot(-L_um/2, h_um/2 + buf_um)]
    return buf, det, (x_c, y_c), np.degrees(theta)


def _d3_rect(x0_um, L_um, Q_um):
    """D3 rotiert tangential, schmale Seite (L) entlang Lens-Tangente."""
    x0_m = x0_um*1e-6
    sag = R_BEND - np.sqrt(R_BEND**2 - x0_m**2)
    y_mid_um = -sag*1e6
    slope = -x0_m/np.sqrt(R_BEND**2 - x0_m**2)
    theta = np.arctan(slope)
    ct, st = np.cos(theta), np.sin(theta)
    def rot(u, v):
        return (x0_um + ct*u - st*v, y_mid_um + st*u + ct*v)
    poly = [rot(-L_um/2, -Q_um/2), rot(L_um/2, -Q_um/2),
            rot(L_um/2, Q_um/2), rot(-L_um/2, Q_um/2)]
    return poly, (x0_um, y_mid_um), np.degrees(theta)


# =====================================================================
# Panel A: Sensor-Übersicht (ganze Linse)
# =====================================================================
def panel_overview(ax):
    ax.set_aspect('equal')
    xs_um = np.linspace(-7000, 7000, 800)
    sag_um = (R_BEND - np.sqrt(R_BEND**2 - (xs_um*1e-6)**2))*1e6
    y_mid_um = -sag_um
    y_top_um = y_mid_um + T_LENS*1e6/2
    y_bot_um = y_mid_um - T_LENS*1e6/2

    # Layer-Hintergrund
    t_lip, t_aq, t_mu = 0.030, 3.5, 0.5
    ax.fill_between(xs_um, np.full_like(xs_um, -4500),
                    y_bot_um - t_lip - t_aq - t_mu,
                    color='#5FA85F', alpha=0.35)
    ax.fill_between(xs_um, y_bot_um - t_lip - t_aq - t_mu,
                    y_bot_um, color='#56C4FF', alpha=0.6)
    ax.fill_between(xs_um, y_bot_um, y_top_um, color='#FFD75E', alpha=0.55,
                    edgecolor='black', lw=1.0)

    # Detektoren
    x_d1_um = (D1_S_CENTER - D_LENS/2)*1e6
    L_um, H_um, BUF_um = 100.0, D1_DICKE*1e6, T_BUF*1e6
    buf1, det1, c1, th1 = _det_rect_tangential(x_d1_um, 'lower', L_um, H_um, BUF_um)
    buf2, det2, c2, th2 = _det_rect_tangential(x_d1_um, 'upper', L_um, H_um, BUF_um)
    ax.add_patch(Polygon(det1, fc='#363636', ec='#FF6600', lw=1.5))
    ax.add_patch(Polygon(det2, fc='#363636', ec='#0066FF', lw=1.5))

    x_d3_um = (D3_S_CENTER - D_LENS/2)*1e6
    d3_poly, c3, th3 = _d3_rect(x_d3_um, D3_LEN*1e6, D3_QUERSCHN*1e6)
    ax.add_patch(Polygon(d3_poly, fc='#363636', ec='#00CC44', lw=1.5))

    # Stirnflaeche + VCSEL rechts
    y_edge = float(y_mid_um[-1])
    ax.plot([7000, 7000], [y_edge - T_LENS*1e6/2, y_edge + T_LENS*1e6/2],
            '-', color='magenta', lw=2.5)
    ax.add_patch(Rectangle((7050, y_edge - 100), 250, 200,
                            fc='#FF3333', ec='black', lw=1))

    # TIR-Zigzag (von rechts nach links)
    xs_z = np.linspace(6800, x_d3_um + 100, 22)
    ys_z = np.zeros_like(xs_z)
    for i, x in enumerate(xs_z):
        sg = R_BEND - np.sqrt(R_BEND**2 - (x*1e-6)**2)
        ys_z[i] = -sg*1e6 + (-1)**i * (T_LENS*1e6/2 - 5)
    ax.plot(xs_z, ys_z, '-', color='red', lw=0.7, alpha=0.6)

    # KLARE LABELS (außerhalb der Lens, in weißen Boxen)
    _label_box(ax, 0, 250, 'LUFT', color='steelblue', fontsize=10)
    ax.text(0, -50, 'PMMA-Linse  n=1.491  t=250 µm  R=8.3 mm  D=14 mm',
            ha='center', fontsize=9, fontweight='bold')
    _label_box(ax, 0, -3300, 'Cornea', color='#005500', fontsize=10)
    _label_box(ax, -3500, -4200, 'Tränenfilm (3-Schicht: Lipid 30 nm + Aqueous 3.5 µm + Mucin 0.5 µm)',
               color='#003366', fontsize=8)

    # Detektor-Labels mit Pfeilen
    ax.annotate('D1 (Tear-Detektor)\n500×500×50 µm', xy=c1, xytext=(-3000, -3000),
                fontsize=8, color='#cc4400', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cc4400', lw=0.8),
                arrowprops=dict(arrowstyle='->', color='#cc4400', lw=0.8))
    ax.annotate('D2 (Air-Referenz)\n500×500×50 µm', xy=c2, xytext=(-3000, 200),
                fontsize=8, color='#0044cc', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#0044cc', lw=0.8),
                arrowprops=dict(arrowstyle='->', color='#0044cc', lw=0.8))
    ax.annotate('D3 (End-Fire, TIR-Licht)\n200×200×50 µm', xy=c3, xytext=(-6500, -3000),
                fontsize=8, color='#006622', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#006622', lw=0.8),
                arrowprops=dict(arrowstyle='->', color='#006622', lw=0.8))
    ax.annotate('VCSEL 850 nm\n(an Stirnfläche)', xy=(7000, y_edge), xytext=(5500, -3000),
                fontsize=8, color='#cc0000', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cc0000', lw=0.8),
                arrowprops=dict(arrowstyle='->', color='#cc0000', lw=0.8))

    ax.set_xlim(-9000, 9000)
    ax.set_ylim(-4700, 700)
    ax.set_xlabel('x [µm]   (Apex bei 0, Lichtpropagation: +x → −x)', fontsize=10)
    ax.set_ylabel('y [µm]', fontsize=10)
    ax.set_title('A) Sensor-Übersicht: PMMA-Kontaktlinse 250 µm + 3-Detektor-System + VCSEL Butt-Coupling',
                  fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


# =====================================================================
# Panel B: Lokaler Zoom (D1/D2/D3)
# =====================================================================
def panel_zoom(ax):
    ax.set_aspect('equal')
    x_center = (D1_S_CENTER - D_LENS/2)*1e6
    half_w, half_h = 800, 500
    xs_um = np.linspace(x_center - half_w, x_center + half_w, 400)
    sag_um = (R_BEND - np.sqrt(R_BEND**2 - (xs_um*1e-6)**2))*1e6
    y_mid_um = -sag_um
    y_top_um = y_mid_um + T_LENS*1e6/2
    y_bot_um = y_mid_um - T_LENS*1e6/2

    # Layer
    ax.fill_between(xs_um, y_mid_um.min() - half_h, y_bot_um - 4.0,
                    color='#5FA85F', alpha=0.35)
    ax.fill_between(xs_um, y_bot_um - 4.0, y_bot_um, color='#56C4FF', alpha=0.6)
    ax.fill_between(xs_um, y_bot_um, y_top_um, color='#FFD75E', alpha=0.55,
                    edgecolor='black', lw=0.8)

    # Detektoren
    L_um, H_um, BUF_um = 100.0, D1_DICKE*1e6, T_BUF*1e6
    buf1, det1, c1, th1 = _det_rect_tangential(x_center, 'lower', L_um, H_um, BUF_um)
    buf2, det2, c2, th2 = _det_rect_tangential(x_center, 'upper', L_um, H_um, BUF_um)
    ax.add_patch(Polygon(det1, fc='#363636', ec='#FF6600', lw=2.0))
    ax.add_patch(Polygon(det2, fc='#363636', ec='#0066FF', lw=2.0))

    x_d3_um = (D3_S_CENTER - D_LENS/2)*1e6
    d3_poly, c3, th3 = _d3_rect(x_d3_um, D3_LEN*1e6, D3_QUERSCHN*1e6)
    ax.add_patch(Polygon(d3_poly, fc='#363636', ec='#00CC44', lw=2.0))

    # TIR-Zigzag von rechts
    xs_z = np.linspace(x_center + half_w*0.9, x_d3_um, 6)
    ys_z = []
    for i, x in enumerate(xs_z):
        sg = R_BEND - np.sqrt(R_BEND**2 - (x*1e-6)**2)
        ys_z.append(-sg*1e6 + (-1)**i * (T_LENS*1e6/2 - 10))
    ax.plot(xs_z, ys_z, '-', color='red', lw=1.2, alpha=0.7)
    ax.annotate('TIR', xy=(xs_z[2], ys_z[2]), xytext=(xs_z[2]+150, ys_z[2]+80),
                fontsize=9, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1))

    # Labels in weißen Boxen
    ax.annotate(f'D1 — Tear\n{th1:+.0f}°', xy=c1, xytext=(c1[0]+250, c1[1]-300),
                fontsize=9, color='#cc4400', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cc4400'),
                arrowprops=dict(arrowstyle='->', color='#cc4400'))
    ax.annotate(f'D2 — Ref.\n{th2:+.0f}°', xy=c2, xytext=(c2[0]+250, c2[1]+300),
                fontsize=9, color='#0044cc', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#0044cc'),
                arrowprops=dict(arrowstyle='->', color='#0044cc'))
    ax.annotate(f'D3 — End-Fire\n{th3:+.0f}°', xy=c3, xytext=(c3[0]-300, c3[1]+250),
                fontsize=9, color='#006622', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#006622'),
                arrowprops=dict(arrowstyle='->', color='#006622'))

    _label_box(ax, x_center + 300, y_mid_um.mean() + 350, 'LUFT', color='steelblue', fontsize=9)
    _label_box(ax, x_center - 300, y_mid_um.mean() - 350, 'CORNEA', color='#005500', fontsize=9)
    _label_box(ax, x_center, y_mid_um.mean(), 'PMMA', color='#995500', fontsize=9)

    ax.set_xlim(x_center - half_w, x_center + half_w)
    ax.set_ylim(y_mid_um.mean() - half_h, y_mid_um.mean() + half_h*0.5)
    ax.set_xlabel('x [µm]', fontsize=10)
    ax.set_ylabel('y [µm]', fontsize=10)
    ax.set_title('B) Lokaler Zoom: D1/D2 als tangentialer Doppelstack + D3 (End-Fire)',
                  fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


# =====================================================================
# Panel C: Butt-Coupling (VCSEL direkt an Stirnfläche)
# =====================================================================
def panel_butt_coupling(ax):
    ax.set_xlim(-400, 250)
    ax.set_ylim(-180, 180)
    ax.set_aspect('equal')

    # PMMA-Linse links
    ax.add_patch(Rectangle((-400, -125), 400, 250, fc='#FFD75E', alpha=0.5,
                            ec='black', lw=1.5))
    # Stirnflaeche
    ax.plot([0, 0], [-125, 125], '-', color='magenta', lw=3)

    # VCSEL-Schichtstapel direkt an Stirnflaeche
    ax.add_patch(Rectangle((0, -60), 5, 120, fc='#FFCCAA', ec='black'))     # Top-Bragg
    ax.add_patch(Rectangle((5, -40), 3, 80, fc='#FF6600', ec='black'))      # Aktive Zone
    ax.add_patch(Rectangle((8, -60), 5, 120, fc='#FFCCAA', ec='black'))     # Bottom-Bragg
    ax.add_patch(Rectangle((13, -75), 60, 150, fc='#888888', ec='black'))   # GaAs-Substrat

    # Divergenz-Kegel (VCSEL emittiert nach links in PMMA)
    th = np.linspace(-0.15, 0.15, 12)
    for t in th:
        ax.plot([6, -400], [0, t*450], '-', color='red', lw=0.4, alpha=0.6)
    ax.annotate('', xy=(-100, 0), xytext=(6, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))

    # Labels in sauberen Boxen
    _label_box(ax, -200, 100, 'PMMA-Linse\n(Lens-Rim)', color='black', fontsize=9)
    _label_box(ax, 0, 155, 'Stirnfläche\n(x = 0)', color='magenta', fontsize=8)
    _label_box(ax, 43, 110, 'VCSEL 850 nm\n(Butt-Coupling)', color='darkred', fontsize=9)
    _label_box(ax, -200, -40, 'Divergente Mode\n(~10° in PMMA)', color='red', fontsize=8)

    # Bragg/Aktive Annotationen
    ax.annotate('Aktive Zone\n(3 µm)', xy=(6.5, 0), xytext=(120, 80),
                fontsize=7, color='#cc6600', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cc6600', lw=0.5),
                arrowprops=dict(arrowstyle='->', color='#cc6600', lw=0.6))
    ax.annotate('Bragg-Mirrors', xy=(10, -30), xytext=(120, -80),
                fontsize=7, color='#996600', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#996600', lw=0.5),
                arrowprops=dict(arrowstyle='->', color='#996600', lw=0.6))

    # Fresnel-Hinweis
    ax.text(-200, -150,
            'Fresnel-Verlust an GaAs/PMMA-Grenze:  R = ((3.5−1.491)/(3.5+1.491))² ≈ 16 %\n'
            'Mit Index-Match-Layer (n≈1.7, 50 nm): R < 2 %',
            ha='center', fontsize=7.5, fontweight='bold', color='#666666',
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFFFCC', ec='gray', lw=0.6))

    ax.set_xlabel('x [µm]   (Stirnfläche bei x=0)', fontsize=10)
    ax.set_ylabel('y [µm]', fontsize=10)
    ax.set_title('C) VCSEL Butt-Coupling: direktes Andocken an Lens-Stirnfläche (kein Mikrolinsen-Spalt)',
                  fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


def make_plot():
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.3, 0.9], hspace=0.55)
    panel_overview(fig.add_subplot(gs[0]))
    panel_zoom(fig.add_subplot(gs[1]))
    panel_butt_coupling(fig.add_subplot(gs[2]))
    plt.suptitle('Kontaktlinsen-Wellenleiter-Sensor: '
                  'PMMA 250 µm, R=8.3 mm, D=14 mm, λ=850 nm, VCSEL-Butt-Coupling',
                  fontsize=13, fontweight='bold', y=0.995)
    plt.subplots_adjust(left=0.06, right=0.96, top=0.96, bottom=0.04)
    plt.savefig('results/01_geometry.png', dpi=120)
    plt.close()
    print('Saved: results/01_geometry.png')


if __name__ == '__main__':
    make_plot()
