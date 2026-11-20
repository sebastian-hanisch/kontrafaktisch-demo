"""Handbeispiele des Plans (Teile 1 bis 4), exakt in Brüchen: ein kleines lineares Modell auf einem Raster mit 84 Kandidaten, das ganzzahlige Programm als zweiter Rechenweg,
Neu-Training und ein kleiner Baum. Teil 5 (geschlossene Form) steht in test_oracle_kontrafaktisch.py, Teil 6 (exakter Baum) in test_tree.py."""

from fractions import Fraction as F

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

EPS = F(1, 1000000)
SD_V = F(2)                                         # Vorlauf-Schritt von |dv| Tagen kostet |dv| / 2, jede Binärgröße 1
V_RASTER = [F(k, 2) for k in range(21)]             # Vorlauf 0, 0,5, ..., 10 Tage
B1, CV1, CZ1, CW1 = F(-11, 5), F(3, 5), F(1), F(1, 2)    # Logit = -11/5 + 3/5 v + z + w/2
X0 = (F(6), 1, 1)                                   # Instanz: Vorlauf 6, Zoll ja, Wochenende ja


def logit1(v, z, w, koef=(B1, CV1, CZ1, CW1)):
    b, cv, cz, cw = koef
    return b + cv * v + cz * z + cw * w


def kosten(x0, v, z, w):
    return abs(v - x0[0]) / SD_V + (z != x0[1]) + (w != x0[2])


def aufzaehlen(x0, erlaubt, margin, score):
    out = []
    for v in V_RASTER:
        for z in (0, 1):
            for w in (0, 1):
                if ("v" not in erlaubt and v != x0[0]) or ("z" not in erlaubt and z != x0[1]) or ("w" not in erlaubt and w != x0[2]):
                    continue
                if score(v, z, w) <= -margin - EPS:
                    out.append((kosten(x0, v, z, w), (v, z, w)))
    return sorted(out)


def test_teil1_cheapest_way_by_hand_and_by_enumeration():
    assert logit1(*X0) == F(29, 10)                                          # Klasse 1
    assert len(V_RASTER) * 4 == 84
    alle = aufzaehlen(X0, {"v", "z", "w"}, F(0), logit1)
    assert alle[0] == (F(5, 2), (F(1), 1, 1))                               # nur Vorlauf: 3/5 dv >= 29/10 -> dv = 5, Kosten 5/2
    assert logit1(F(1), 1, 1) == F(-1, 10)
    assert [c for c in alle if c[1][1] == 0 and c[1][2] == 1][0] == (F(11, 4), (F(5, 2), 0, 1))     # Zoll abschalten und v = 5/2
    nahe = [c for c in alle if c[0] <= F(11, 10) * alle[0][0]]
    mengen = {tuple(i for i, (a, b) in enumerate(zip(c, X0)) if a != b) for _, c in nahe}
    assert len(nahe) == 3 and mengen == {(0,), (0, 1)}                       # drei Rasterpunkte, zwei Wege
    assert aufzaehlen(X0, {"z", "w"}, F(0), logit1) == []                    # nur Zoll und Wochenende: kleinstes Logit 29/10 - 3/2 = 7/5 > 0
    a_mg = aufzaehlen(X0, {"v", "z", "w"}, F(1, 2), logit1)
    assert a_mg[0] == (F(3), (F(0), 1, 1)) and a_mg[0][0] / alle[0][0] == F(6, 5)          # Sicherheitsabstand 1/2: 20 % teurer
    assert aufzaehlen(X0, {"v", "z", "w"}, F(3), logit1) == []               # kleinstes mögliches Logit -11/5


def milp_rueckgriff(x0, erlaubt, margin):
    b, cv, cz, cw = float(B1), float(CV1), float(CZ1), float(CW1)
    n = 2 + 21
    c = np.zeros(n)
    c[0] = 1 if x0[1] == 0 else -1
    c[1] = 1 if x0[2] == 0 else -1
    v0 = int(x0[0] * 2)
    c[2:] = [abs(k - v0) * 0.5 / float(SD_V) for k in range(21)]
    const = float(x0[1] + x0[2])
    lb, ub = np.zeros(n), np.ones(n)
    if "z" not in erlaubt:
        lb[0] = ub[0] = x0[1]
    if "w" not in erlaubt:
        lb[1] = ub[1] = x0[2]
    if "v" not in erlaubt:
        lb[2:] = ub[2:] = [1.0 if k == v0 else 0.0 for k in range(21)]
    zeile = np.zeros(n)
    zeile[0], zeile[1] = cz, cw
    zeile[2:] = [cv * k / 2 for k in range(21)]
    eins = np.zeros(n)
    eins[2:] = 1
    cons = [LinearConstraint(zeile[None, :], -np.inf, -float(margin) - float(EPS) - b), LinearConstraint(eins[None, :], 1, 1)]
    res = milp(c, constraints=cons, integrality=np.ones(n, dtype=int), bounds=Bounds(lb, ub))
    return None if res.status != 0 else float(res.fun) + const


def test_teil2_integer_program_equals_enumeration():
    for erl in ({"v", "z", "w"}, {"v"}, {"z", "w"}, {"v", "z"}):
        for margin in (F(0), F(1, 2), F(1), F(2)):
            e = aufzaehlen(X0, erl, margin, logit1)
            m = milp_rueckgriff(X0, erl, margin)
            assert (e == []) == (m is None), (erl, margin)
            if e:
                assert abs(float(e[0][0]) - m) < 1e-9, (erl, margin)


def test_teil3_new_training_invalidates_the_cheapest_way_but_not_the_one_with_margin():
    koef2 = (F(-2), F(11, 20), F(1), F(1, 2))                               # neu trainiert: -2 + 0,55 v + z + w/2

    def logit2(v, z, w):
        return logit1(v, z, w, koef2)

    alle = aufzaehlen(X0, {"v", "z", "w"}, F(0), logit1)
    a_mg = aufzaehlen(X0, {"v", "z", "w"}, F(1, 2), logit1)
    assert logit2(F(1), 1, 1) == F(1, 20) and logit2(*alle[0][1]) > 0        # ungültig
    assert logit2(*a_mg[0][1]) == F(-1, 2)                                  # gültig
    ungueltig, gesamt = {0: 0, 1: 0}, 0
    for v0 in [F(k, 2) for k in range(8, 21)]:
        x0 = (v0, 1, 1)
        if logit1(*x0) <= 0:
            continue
        gesamt += 1
        for i, mgn in enumerate((F(0), F(1, 2))):
            e = aufzaehlen(x0, {"v", "z", "w"}, mgn, logit1)
            if e and logit2(*e[0][1]) > -EPS:
                ungueltig[i] += 1
    assert gesamt == 13 and ungueltig == {0: 13, 1: 0}                      # ein Beispiel, kein Messwert der Linie


def test_teil4_small_tree_exact_over_leaf_boxes_and_the_edge_point_trap():
    def baum(v, z, w):
        if v <= F(7, 2):
            return 0
        return 0 if z == 0 else 1

    assert baum(*X0) == 1
    aufz = sorted((kosten(X0, v, z, w), (v, z, w)) for v in V_RASTER for z in (0, 1) for w in (0, 1) if baum(v, z, w) == 0)
    assert aufz[0][0] == 1                                                  # Zoll abschalten
    assert min(F(5, 4), F(1)) == aufz[0][0]                                 # Blattquader: v <= 7/2 kostet (6 - 7/2)/2 = 5/4, Zoll 0 kostet 1
    assert sorted(c for c, _ in aufz)[1] == F(5, 4)
    assert sum(1 for c, _ in aufz if c <= F(11, 10) * aufz[0][0]) == 1
    assert baum(F(7, 2), 1, 1) == 0 and baum(F(7, 2) + EPS, 1, 1) == 1      # Randpunkt-Falle: auf dem Rand gilt Klasse 0, knapp darüber Klasse 1
