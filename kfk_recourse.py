"""Der Weg: die billigste erlaubte Änderung einer Instanz, nach der ein Modell „kein Langsteher“ sagt (Klasse 0), auf einem festen Raster.

Raster der Kandidaten (Art 4 x Kunde 12 x Zoll 2 x Wochenende 2 x Vorlauf 21 x Gewicht 13 = 52 416), davon 13 104 strukturell zulässig (Zoll nur bei Import; Vorlauf 0 bei
Import und Leer, mindestens 1 Tag bei Export, mindestens 0,5 Tage bei Umschlag). Gerechnet wird nur über die 13 104 zulässigen **Kandidaten**; die Reihenfolge der Kandidaten ist
fest (Art, Kunde, Zoll, Wochenende, Vorlauf, Gewicht aufsteigend), und bei gleichen Kosten gewinnt der **kleinste Kandidaten-Index** (Gleichstandsregel des Weges).

Kosten sind von dieser Demo **gesetzt**: 1 je geänderte Kategorie oder Binärvariable (Art, Kunde, Zoll, Wochenende), Vorlauf |Delta| geteilt durch die Standardabweichung der von 0
verschiedenen Vorlaufzeiten im Training, Gewicht |Delta|. **Politik** = Liste der veränderbaren Merkmale. **Sicherheitsabstand** m: verlangt wird Logit <= -m (nicht nur <= 0).
Drei Rechenwege: Aufzählung über das Raster (Hauptverfahren), ganzzahliges Programm (`scipy.optimize.milp`, nur für das lineare Modell) und Blattquader (nur für Bäume).
"""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

import kfk_generator as G
from kfk_tree import split_matrix

EPS = 1e-6
NA, NK, NV, NG = 4, 12, 21, 13
FEATURES = ("Art", "Kunde", "Zoll", "Wochenende", "Vorlauf", "Gewicht")
# Reihenfolge der Blöcke: art, kunde, zoll, wochenende, vorlauf, gewicht
POLICIES = {
    "A": (1, 1, 1, 1, 1, 1),
    "B": (0, 0, 1, 0, 1, 0),
    "C": (0, 0, 0, 0, 1, 0),
}
POLICY_NAMES = {"A": "A: alles veränderbar", "B": "B: Vorlauf und Zoll", "C": "C: nur Vorlauf"}
MARGINS = (0.0, 0.5, 1.0)

_G = np.meshgrid(np.arange(NA), np.arange(NK), np.arange(2), np.arange(2), np.arange(NV), np.arange(NG), indexing="ij")
_CA, _CK, _CZ, _CW, _CV, _CG = [g.ravel() for g in _G]
_STRUCT = (_CZ <= (_CA == 0)) & np.where(np.isin(_CA, (0, 3)), _CV == 0, True) & ~((_CA == 1) & (_CV < 2)) & ~((_CA == 2) & (_CV < 1))
N_GRID = len(_CA)
CA, CK, CZ, CW, CV, CG = [a[_STRUCT] for a in (_CA, _CK, _CZ, _CW, _CV, _CG)]   # die zulässigen Kandidaten
N_CAND = len(CA)
LEVELS = np.column_stack([CA, CK, CZ, CW, CV, CG])
# Rohwerte der Kandidaten wie die Trainingsmerkmale: Vorlauf in Tagen, Gewicht in Einheiten
XC = np.column_stack([CA, CK, CZ, CW, CV * 0.5, (CG - 6) * 0.5]).astype(float)
MU_C, S_C = G.log_median_and_s(CA, CK, CZ.astype(float), CW.astype(float), CV * 0.5)


def design(X):
    """Entwurfsmatrix der Modelle: Art (4 Spalten), Kunde (12 Spalten), Zoll, Wochenende, Vorlauf, Gewicht = 20 Spalten."""
    n = len(X)
    D = np.zeros((n, 20))
    D[np.arange(n), X[:, 0].astype(int)] = 1
    D[np.arange(n), 4 + X[:, 1].astype(int)] = 1
    D[:, 16], D[:, 17], D[:, 18], D[:, 19] = X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    return D


DC = design(XC)


def to_raster(X):
    """Instanzen und Trainingsdaten auf das Raster legen: Vorlauf und Gewicht in 0,5-Schritten, Gewicht in [-3, 3]."""
    X = np.array(X, dtype=float, copy=True)
    X[:, 4] = np.round(X[:, 4] * 2) / 2
    X[:, 5] = np.clip(np.round(X[:, 5] * 2) / 2, -3, 3)
    return X


def levels(x):
    """Ebenen-Tupel (Art, Kunde, Zoll, Wochenende, Vorlauf-Stufe, Gewicht-Stufe) einer Instanz auf dem Raster."""
    return (int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(round(x[4] * 2)), int(round(x[5] * 2)) + 6)


def index_of(lev):
    """Kandidaten-Index einer Ebenen-Tupel (der Kandidat muss zulässig sein)."""
    hit = np.flatnonzero((LEVELS == np.array(lev)).all(axis=1))
    if len(hit) != 1:
        raise ValueError(f"kein zulässiger Kandidat: {lev}")
    return int(hit[0])


def sd_vorlauf(Xr):
    """Standardabweichung der von 0 verschiedenen Vorlaufzeiten des (gerasterten) Trainings: Maßstab der Vorlaufkosten."""
    v = Xr[:, 4]
    return float(v[v > 0].std())


def cost_cands(lev, sd_v):
    a0, k0, z0, w0, v0, g0 = lev
    return (CA != a0) * 1.0 + (CK != k0) + (CZ != z0) + (CW != w0) + np.abs(CV - v0) * 0.5 / sd_v + np.abs(CG - g0) * 0.5


def allowed(lev, act):
    """Zulässige Kandidaten: nur die in `act` freigegebenen Merkmale dürfen von der Instanz abweichen."""
    ok = np.ones(N_CAND, dtype=bool)
    for flag, arr, v in zip(act, (CA, CK, CZ, CW, CV, CG), lev):
        if not flag:
            ok &= arr == v
    return ok


def cheapest(lev, act, margin, scores, sd_v):
    """Hauptverfahren: Aufzählung über die Kandidaten. Liefert (Kosten, Kandidaten-Index) oder None (kein Weg). `scores`: Logit je Kandidat."""
    ok = allowed(lev, act) & (scores <= -margin - EPS)
    if not ok.any():
        return None
    c = np.where(ok, cost_cands(lev, sd_v), np.inf)
    i = int(np.argmin(c))
    return float(c[i]), i


def near_candidates(lev, act, margin, scores, sd_v, cost_min, tol=0.10):
    """Alle erlaubten Kandidaten mit Kosten innerhalb von 10 % über den Mindestkosten (Indizes)."""
    ok = allowed(lev, act) & (scores <= -margin - EPS)
    c = np.where(ok, cost_cands(lev, sd_v), np.inf)
    return np.flatnonzero(c <= (1 + tol) * cost_min + 1e-9)


def change_signature(lev, idx):
    """Bitmaske der geänderten Merkmale je Kandidat (Mengen geänderter Merkmale; zwei Kandidaten mit gleicher Menge sind derselbe „Weg“)."""
    a0, k0, z0, w0, v0, g0 = lev
    return ((CA[idx] != a0) * 1 + (CK[idx] != k0) * 2 + (CZ[idx] != z0) * 4 + (CW[idx] != w0) * 8 + (CV[idx] != v0) * 16 + (CG[idx] != g0) * 32)


def changed_features(lev, i):
    sig = int(change_signature(lev, np.array([i]))[0])
    return [FEATURES[j] for j in range(6) if sig >> j & 1]


def number_of_ways(lev, idx):
    """Zahl verschiedener Mengen geänderter Merkmale unter den Kandidaten idx."""
    return int(len(np.unique(change_signature(lev, idx))))


def truth_valid(i, threshold_days):
    """Wahrheit je Funktional (Eingriffs-Lesart): am Kandidaten i liegt die wahre Wahrscheinlichkeit der Klasse 1 unter 0,5, das heißt mu < ln(Schwelle)."""
    return not bool(G.truth_longer(MU_C[i], threshold_days))


def milp_way(lev, act, margin, w, b, sd_v):
    """Ganzzahliges Programm (HiGHS) für das lineare Modell: alle Variablen binär (Art 4, Kunde 12, Zoll, Wochenende, Vorlauf-Stufe 21, Gewicht-Stufe 13).
    Liefert (Kosten, Ebenen-Tupel) oder None. Zweiter Rechenweg gegen `cheapest`."""
    a0, k0, z0, w0, v0, g0 = lev
    n = 52
    iv = 18 + np.arange(NV)
    ig = 39 + np.arange(NG)
    c = np.zeros(n)
    c[a0] -= 1
    c[4 + k0] -= 1
    c[16] += 1 if z0 == 0 else -1
    c[17] += 1 if w0 == 0 else -1
    c[iv] = np.abs(np.arange(NV) - v0) * 0.5 / sd_v
    c[ig] = np.abs(np.arange(NG) - g0) * 0.5
    const = 1 + 1 + z0 + w0
    lb = np.zeros(n)
    ub = np.ones(n)
    fix = {0: [(a, 1 if a == a0 else 0) for a in range(4)], 1: [(4 + k, 1 if k == k0 else 0) for k in range(12)], 2: [(16, z0)], 3: [(17, w0)],
           4: [(18 + s, 1 if s == v0 else 0) for s in range(NV)], 5: [(39 + s, 1 if s == g0 else 0) for s in range(NG)]}
    for blk, flag in enumerate(act):
        if not flag:
            for j, v in fix[blk]:
                lb[j] = ub[j] = v
    rows, lo, hi = [], [], []

    def add(row, low, high):
        rows.append(row)
        lo.append(low)
        hi.append(high)

    r = np.zeros(n)
    r[0:16] = w[0:16]
    r[16], r[17] = w[16], w[17]
    r[iv] = w[18] * 0.5 * np.arange(NV)
    r[ig] = w[19] * 0.5 * (np.arange(NG) - 6)
    add(r, -np.inf, -margin - EPS - b)
    for sl in (slice(0, 4), slice(4, 16), iv, ig):
        r = np.zeros(n)
        r[sl] = 1
        add(r, 1, 1)
    def structure(on, off, low, high):
        """Zeile: Summe der Binärvariablen `on` minus die Variablen `off` zwischen low und high."""
        row = np.zeros(n)
        row[on] = 1
        for j in off:
            row[j] = -1
        add(row, low, high)

    structure(iv[1:], (1, 2), -np.inf, 0)      # Vorlauf > 0 nur bei Export und Umschlag
    structure(iv[2:], (1,), 0, np.inf)         # Export: Vorlauf mindestens 1 Tag
    structure(iv[1:], (2,), 0, np.inf)         # Umschlag: Vorlauf mindestens 0,5 Tage
    structure([16], (0,), -np.inf, 0)          # Zoll nur bei Import
    res = milp(c, constraints=LinearConstraint(np.array(rows), np.array(lo, dtype=float), np.array(hi, dtype=float)), integrality=np.ones(n, dtype=int), bounds=Bounds(lb, ub))
    if res.status != 0:
        return None
    x = res.x
    sol = (int(np.argmax(x[0:4])), int(np.argmax(x[4:16])), int(round(x[16])), int(round(x[17])), int(np.argmax(x[iv])), int(np.argmax(x[ig])))
    return float(res.fun + const), sol


def box_way(lev, act, margin, boxes, sd_v):
    """Weg über Blattquader (Bäume), ohne die Vorhersage des Baums zu rufen: `boxes` = Liste (Zugehörigkeit je Kandidat als bool-Feld, Blatt-Logit).
    Je Blatt mit Logit <= -margin - EPS die billigste erlaubte Zeile im Quader; das Minimum über alle Blätter. Liefert (Kosten, Kandidaten-Index) oder None."""
    ok0 = allowed(lev, act)
    c = cost_cands(lev, sd_v)
    best = None
    for member, logit in boxes:
        if logit > -margin - EPS:
            continue
        m = ok0 & member
        if m.any():
            cc = np.where(m, c, np.inf)
            i = int(np.argmin(cc))
            if best is None or cc[i] < best[0]:
                best = (float(cc[i]), i)
    return best


def tree_cut_matrix(spec):
    """Schnittmatrix der Kandidaten für den exakt optimalen Baum (Kandidaten x Schnitte)."""
    return split_matrix(XC, spec)
