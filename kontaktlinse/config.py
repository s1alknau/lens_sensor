"""Zentrale Konfiguration des Kontaktlinsen-Sensors."""
import numpy as np

LAM   = 850e-9
k0    = 2*np.pi/LAM
C0    = 2.99792458e8
EPS0  = 8.8541878128e-12
MU0   = 4*np.pi*1e-7

N_PMMA   = 1.491
N_AIR    = 1.000
N_LIPID  = 1.480
N_MUCIN  = 1.342
N_CORNEA = 1.376
N_INGAAS = 3.5 + 0.05j
N_BUFFER = 1.42

T_LENS = 250e-6
R_BEND = 8.3e-3
D_LENS = 14e-3

D1_LEN   = 500e-6
D1_DICKE = 50e-6
D2_LEN   = 500e-6
D2_DICKE = 50e-6
D3_LEN   = 50e-6         # tangential zur Lens-Flaeche
D3_QUERSCHN = 160e-6     # normal zur Lens-Flaeche (passt trotz Kruemmung)
T_BUF    = 20e-6

D1_S_CENTER = 2.0e-3     # x = -5000 um (Peripherie)
D2_S_CENTER = 2.0e-3
D3_S_CENTER = 1.8e-3     # x = -5200 um (links von D1/D2, vor Lichteinfall)

D1_S_LO = D1_S_CENTER - D1_LEN/2
D1_S_HI = D1_S_CENTER + D1_LEN/2
D2_S_LO = D2_S_CENTER - D2_LEN/2
D2_S_HI = D2_S_CENTER + D2_LEN/2
D3_S_LO = D3_S_CENTER - D3_LEN/2
D3_S_HI = D3_S_CENTER + D3_LEN/2

D1_XI_LO = -T_LENS/2 + T_BUF
D1_XI_HI = D1_XI_LO + D1_DICKE
D2_XI_HI = +T_LENS/2 - T_BUF
D2_XI_LO = D2_XI_HI - D2_DICKE
D3_XI_LO = -D3_QUERSCHN/2
D3_XI_HI = +D3_QUERSCHN/2

SCENARIOS = [
    ('Gesund',         0.030, 3.5, 0.5, 1.336),
    ('DED',            0.010, 1.5, 0.3, 1.336),
    ('Frisch',         0.050, 6.0, 1.0, 1.336),
    ('Hyperosmolar',   0.020, 2.0, 0.4, 1.340),
    ('MGD',            0.005, 3.5, 0.5, 1.336),
    ('Mucin-Mangel',   0.030, 3.5, 0.05,1.336),
    ('Mucin-reich',    0.030, 3.5, 2.0, 1.336),
    ('Lipid-reich',    0.100, 3.5, 0.5, 1.336),
]
