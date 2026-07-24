"""Auswertung der Sliding-Window-FDTD-Ergebnisse: 8 Tear-Film-Szenarien.

Liest results/sliding_<scenario>.pkl Dateien und erstellt:
  - 04_sliding_sensor_results.png mit Sensor-Diskriminierung
  - sliding_summary.txt mit Tabellen und Empfindlichkeiten
"""
import os, pickle
import numpy as np
import matplotlib.pyplot as plt

SCENARIOS = [
    ('Gesund',       0.030, 3.5, 0.5, 1.336),
    ('DED',          0.010, 1.5, 0.3, 1.336),
    ('Frisch',       0.050, 6.0, 1.0, 1.336),
    ('Hyperosmolar', 0.020, 2.0, 0.4, 1.340),
    ('MGD',          0.005, 3.5, 0.5, 1.336),
    ('Mucin-Mangel', 0.030, 3.5, 0.05, 1.336),
    ('Mucin-reich',  0.030, 3.5, 2.0, 1.336),
    ('Lipid-reich',  0.100, 3.5, 0.5, 1.336),
]


def load_all(results_dir='results'):
    data = {}
    for nm, *_ in SCENARIOS:
        path = f'{results_dir}/sliding_{nm}.pkl'
        if not os.path.exists(path):
            print(f'[Warning] missing: {path}')
            continue
        with open(path, 'rb') as f:
            data[nm] = pickle.load(f)
    return data


def make_plot(data, out_path='results/04_sliding_sensor_results.png'):
    names = [n for n, *_ in SCENARIOS if n in data]
    n = len(names)
    if n == 0:
        print('[Error] Keine Daten vorhanden!')
        return
    xp = np.arange(n)
    P_D1 = np.array([data[k]['P_D1'] for k in names])
    P_D2 = np.array([data[k]['P_D2'] for k in names])
    P_D3 = np.array([data[k]['P_D3'] for k in names])
    R = np.array([data[k]['R_tear'] for k in names])
    ref = R[0]
    rel = (R - ref)/ref*100

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # A) Sensor-Diskriminierung R_tear
    ax = axes[0, 0]
    colors = ['gray' if i == 0 else ('#cc3333' if rel[i] < 0 else '#33aa33')
               for i in range(n)]
    ax.bar(xp, rel, color=colors, alpha=0.85, edgecolor='black')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(xp); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Δ R_tear vs. Gesund [%]')
    ax.set_title('A) Sensor-Diskriminierung R_tear = P_D1/P_D2', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rel):
        off = max(abs(v)*0.1, 0.3)*(1 if v >= 0 else -1)
        ax.text(i, v + off, f'{v:+.2f}%', ha='center', fontsize=8, fontweight='bold')

    # B) D1 absolute Leistung
    ax = axes[0, 1]
    ax.bar(xp, P_D1, color='#FF6600', alpha=0.85, edgecolor='black', label='D1')
    ax.bar(xp, P_D2, color='#0066FF', alpha=0.55, edgecolor='black', label='D2',
           bottom=0)
    ax.set_xticks(xp); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Detektor-Leistung')
    ax.set_title('B) D1 (Tear) und D2 (Air) Absolutleistung', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # C) D3 (End-Fire, TIR-Throughput)
    ax = axes[0, 2]
    ax.bar(xp, P_D3/P_D3[0], color='#00CC44', alpha=0.85, edgecolor='black')
    ax.axhline(1, color='k', lw=0.5)
    ax.set_xticks(xp); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('P_D3 / P_D3(Gesund)')
    ax.set_title('C) End-Fire-Signal D3 (totalreflektiertes Licht)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(P_D3/P_D3[0]):
        ax.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=7.5, fontweight='bold')

    # D) Sensitivitäten als Tabelle
    ax = axes[1, 0]
    ax.axis('off')
    # Parameter-Änderungen vs. Gesund
    layers_dict = {nm: lay for nm, *lay in SCENARIOS}
    cells = [['Szenario', 'Lipid', 'Aq', 'Mucin', 'n_aq', 'R_tear', 'ΔR%']]
    for i, k in enumerate(names):
        lay = layers_dict[k]
        cells.append([k, f'{lay[0]:.3f}', f'{lay[1]:.1f}', f'{lay[2]:.2f}',
                      f'{lay[3]:.3f}', f'{R[i]:.4f}', f'{rel[i]:+.2f}'])
    table = ax.table(cellText=cells, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1, 1.5)
    for j in range(7):
        table[(0, j)].set_facecolor('#cccccc')
        table[(0, j)].set_text_props(weight='bold')
    ax.set_title('D) Ergebnis-Tabelle', fontweight='bold')

    # E) Empfindlichkeiten dR_tear / dParameter
    ax = axes[1, 1]
    sens = {}
    if 'DED' in data:
        sens['Aqueous-Dicke'] = (R[names.index('DED')] - ref)/(1.5 - 3.5)
    if 'Lipid-reich' in data:
        sens['Lipid-Dicke'] = (R[names.index('Lipid-reich')] - ref)/(0.100 - 0.030)
    if 'Mucin-reich' in data:
        sens['Mucin-Dicke'] = (R[names.index('Mucin-reich')] - ref)/(2.0 - 0.5)
    if 'Mucin-Mangel' in data:
        sens['Mucin-Verlust'] = (R[names.index('Mucin-Mangel')] - ref)/(0.05 - 0.5)
    if 'Hyperosmolar' in data:
        sens['n_aq (Osmol.)'] = (R[names.index('Hyperosmolar')] - ref)/(1.340 - 1.336)
    keys = list(sens.keys())
    vals = [sens[k] for k in keys]
    yp = np.arange(len(keys))
    cols = ['#cc3333' if v < 0 else '#33aa33' for v in vals]
    ax.barh(yp, vals, color=cols, alpha=0.85, edgecolor='black')
    ax.axvline(0, color='k', lw=0.5)
    ax.set_yticks(yp); ax.set_yticklabels(keys, fontsize=9)
    ax.set_xlabel('dR_tear / dParameter')
    ax.set_title('E) Empfindlichkeiten (linear)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(vals):
        ax.text(v, i, f'  {v:+.3e}', va='center', fontsize=8, fontweight='bold')

    # F) Time per scenario
    ax = axes[1, 2]
    times = [data[k].get('total_time', 0) for k in names]
    ax.bar(xp, times, color='#446699', alpha=0.85, edgecolor='black')
    ax.set_xticks(xp); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Sim-Zeit [s]')
    ax.set_title('F) FDTD-Wandzeit pro Szenario', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(times):
        ax.text(i, v + max(times)*0.02, f'{v/60:.1f} min', ha='center',
                fontsize=8, fontweight='bold')

    plt.suptitle('Sliding-Window-FDTD: 8 Tränenfilm-Szenarien — '
                  'Sensor-Charakterisierung (volle 14 mm PMMA-Linse)',
                  fontsize=13, fontweight='bold')
    plt.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.10,
                         wspace=0.32, hspace=0.45)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f'Saved: {out_path}')


def write_summary(data, out_path='results/sliding_summary.txt'):
    names = [n for n, *_ in SCENARIOS if n in data]
    layers_dict = {nm: lay for nm, *lay in SCENARIOS}
    with open(out_path, 'w') as f:
        f.write('Sliding-Window-FDTD: 8 Tränenfilm-Szenarien (volle 14 mm Linse)\n')
        f.write('='*90 + '\n\n')
        f.write(f'{"Szenario":<14} {"Lipid":>7} {"Aq":>6} {"Mucin":>6} {"n_aq":>7} '
                 f'{"P_D1":>11} {"P_D2":>11} {"P_D3":>11} {"R_tear":>8} {"ΔR%":>8}\n')
        f.write('-'*90 + '\n')
        ref = data[names[0]]['R_tear']
        for k in names:
            r = data[k]
            lay = layers_dict[k]
            dR = (r['R_tear'] - ref)/ref*100
            f.write(f'{k:<14} {lay[0]:>7.3f} {lay[1]:>6.1f} {lay[2]:>6.2f} {lay[3]:>7.3f} '
                     f'{r["P_D1"]:>11.3e} {r["P_D2"]:>11.3e} {r["P_D3"]:>11.3e} '
                     f'{r["R_tear"]:>8.4f} {dR:>+8.2f}\n')

        # Empfindlichkeiten
        f.write('\n--- Empfindlichkeiten dR_tear/dParameter ---\n')
        R = {k: data[k]['R_tear'] for k in names}
        if 'DED' in data:
            v = (R['DED'] - R['Gesund'])/(1.5 - 3.5)
            f.write(f'  d(R_tear)/d(Aqueous-Dicke) = {v:+.4e} /µm\n')
        if 'Lipid-reich' in data:
            v = (R['Lipid-reich'] - R['Gesund'])/(0.100 - 0.030)
            f.write(f'  d(R_tear)/d(Lipid-Dicke)   = {v:+.4e} /µm\n')
        if 'Mucin-reich' in data:
            v = (R['Mucin-reich'] - R['Gesund'])/(2.0 - 0.5)
            f.write(f'  d(R_tear)/d(Mucin-Dicke)   = {v:+.4e} /µm\n')
        if 'Hyperosmolar' in data:
            v = (R['Hyperosmolar'] - R['Gesund'])/(1.340 - 1.336)
            f.write(f'  d(R_tear)/d(n_aq Osmol.)   = {v:+.4e} /RIU\n')
        # Sim-Zeiten
        f.write('\n--- Sim-Zeiten ---\n')
        total = sum(data[k].get('total_time', 0) for k in names)
        for k in names:
            t = data[k].get('total_time', 0)
            f.write(f'  {k:<14} {t/60:>7.1f} min\n')
        f.write(f'  {"Total":<14} {total/60:>7.1f} min ({total/3600:.1f} h)\n')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    data = load_all()
    if data:
        make_plot(data)
        write_summary(data)
    else:
        print('Keine Daten gefunden! Bitte erst sliding_window_fdtd.py '
              'fuer alle 8 Szenarien ausfuehren.')
