"""Der Verweilzeit-Generator (erfunden, keine reale Datenquelle): Kopie der Formel aus `verweilzeit-demo` (`vwz_generator.py`), kopiert statt importiert.

Container haben Art (0 Import, 1 Export, 2 Umschlag, 3 Leer), Kunde (0 bis 11), Zoll (nur bei Import), Wochenende, Vorlauf in Tagen (nur bei Export und
Umschlag) und Gewicht (reines Rauschen, Wirkung null). Die Verweilzeit d ist lognormal: ln d ~ N(mu, s^2), die bedingte Mitte mu und die Streuung s stehen als
Formel fest. Das ist die bekannte Funktion mu der Linie: jede Wahrheit der Demo (wahre Wahrscheinlichkeit der Klasse 1 an einem geänderten Punkt) wird daraus gerechnet.

Was gegenüber dem Original geändert ist: (1) die Kundeneffekte stehen als Zahlen da (das Original zieht sie mit numpy `default_rng(12345).normal(0, 0.30, 12)`);
(2) der Zufall kommt aus `kfk_rng.py` (SplitMix64, auf jeder Plattform bitgleich) statt aus numpy; deshalb sind die gezogenen Container nicht dieselben wie in
`verweilzeit-demo`, die Formel für mu und s aber schon. Der Kopie-Test (`tests/test_generator.py`) hält die Formel gegen Werte, die mit dem Original gerechnet wurden."""

import math

import numpy as np

from kfk_rng import SplitMix64

ART_P = (0.35, 0.30, 0.25, 0.10)            # Anteile der Containerarten
K_KUNDEN = 12                                # Kunden mit je einem Effekt auf die log-Verweilzeit
U_SCALE = (1.0, 0.3, 0.3, 1.0)               # Kundeneffekt je Art (bei Export und Umschlag ein Drittel)
S_BASE = (0.55, 0.25, 0.35, 0.90)            # Streuung der log-Verweilzeit je Art (heteroskedastisch)
P_ZOLL = 0.25                                # Zollhalt bei Import
# Kundeneffekte des Originals (numpy default_rng(12345).normal(0, 0.30, 12)), als Zahlen übernommen
UK = np.array([-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629,
               -0.2222653956256827, -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569,
               0.7042228963136555, 0.2905490717255771])
HISTORY_OFFSET = 10_000_000                  # Zufallsstrom-Bereich der Trainingshistorien
N_FEATURES = 6                               # art, kunde, zoll, wochenende, vorlauf, gewicht
NAMES = ("art", "kunde", "zoll", "wochenende", "vorlauf", "gewicht")


def log_median_and_s(art, kunde, zoll, we, vorlauf):
    """Bedingte Verteilung der log-Verweilzeit: N(mu, s^2). Vektorisiert; mu ist die bekannte Funktion der Linie."""
    art = np.asarray(art)
    kunde = np.asarray(kunde)
    u = UK[kunde] * np.take(U_SCALE, art)
    med = np.select(
        [art == 0, art == 1, art == 2],
        [3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vorlauf, 0.3 + 0.8 * vorlauf],
        default=7.0 * np.exp(0.30 * we),
    )
    mu = np.log(med) + u
    s = np.take(S_BASE, art)
    return mu, s


def draw_history(n: int, seed: int):
    """n Container mit Merkmalen X (n, 6) und log-Verweilzeit y (n,): eine unabhängige Historie ohne Blockbeschränkung, bitgleich zu jedem Zufalls-Seed."""
    rf = SplitMix64(HISTORY_OFFSET + 2 * seed)
    rz = SplitMix64(HISTORY_OFFSET + 2 * seed + 1)
    X = np.zeros((n, N_FEATURES))
    for i in range(n):
        art = rf.choice(ART_P)
        kunde = rf.below(K_KUNDEN)
        zoll = 1.0 if (rf.random() < P_ZOLL and art == 0) else 0.0
        tday = rf.random() * 7000.0
        we = 1.0 if (int(tday) % 7) >= 5 else 0.0
        u = rf.random()
        vorlauf = 1.0 + 9.0 * u if art == 1 else (0.5 + 5.5 * u if art == 2 else 0.0)
        gewicht = rf.normal()
        X[i] = (art, kunde, zoll, we, vorlauf, gewicht)
    mu, s = log_median_and_s(X[:, 0].astype(int), X[:, 1].astype(int), X[:, 2], X[:, 3], X[:, 4])
    z = np.array([rz.normal() for _ in range(n)])
    return X, mu + s * z


def p_longer(mu, s, threshold_days: float) -> np.ndarray:
    """Wahre Wahrscheinlichkeit, dass die Verweilzeit länger als threshold_days ist (P(ln d > ln Schwelle)); Normalverteilung über math.erfc."""
    z = (math.log(threshold_days) - np.asarray(mu)) / np.asarray(s)
    return 0.5 * np.vectorize(math.erfc)(z / math.sqrt(2.0))


def truth_longer(mu, threshold_days: float) -> np.ndarray:
    """Wahrheit je Funktional (Eingriffs-Lesart): die wahre Wahrscheinlichkeit der Klasse 1 liegt am Punkt bei mindestens 0,5, genau dann, wenn mu >= ln(Schwelle)."""
    return np.asarray(mu) >= math.log(threshold_days)
