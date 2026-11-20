"""Eingefrorene Werte des Baum-Codes: Beim Bau von `duenner-baum-demo` wird `kfk_tree.py` kopiert; dort prüft der Kopie-Test gegen genau diese Werte (und die Dateien sind gleich).
Zufall und Raster sind plattformfest (SplitMix64, ganzzahlige Fehler), die Werte gelten also auf jeder Plattform exakt."""

import numpy as np

import kfk_fragility as F
import kfk_generator as G
import kfk_recourse as R
import kfk_tree as T

KEY_KLEINSTER = (0, 15, 18, -1, -1, 16, -1, -1, 3, -1, 29, -1, -1)
KEY_GROESSTER = (20, 15, 7, -1, -1, 16, -1, -1, 6, 29, -1, -1, 28, -1, -1)
KEY_REG = (0, 15, 18, -1, -1, 16, -1, -1, 25, 3, -1, -1, 31, -1, -1)
KEY_150_KLEINSTER = (0, 18, 10, -1, -1, -1, 1, 28, -1, -1, 2, -1, -1)
KEY_150_GROESSTER = (0, 18, 11, -1, -1, -1, 28, 20, -1, -1, 6, -1, -1)


def _problem(mode):
    X, y = G.draw_history(1000, 1)
    Xr = R.to_raster(X)
    return T.Problem(Xr, F.labels(y).astype(float) if mode == "cls" else y, T.raster_spec(), mode)


def test_frozen_classification_tree_zufalls_seed_1():
    pb = _problem("cls")
    e, tree = pb.best(np.arange(1000), 3)
    assert e == 220.0 and T.n_leaves(tree) == 7 and T.tree_key(tree) == KEY_KLEINSTER
    e2, tree2 = pb.best(np.arange(1000), 3, "groesster")
    assert e2 == 220.0 and T.n_leaves(tree2) == 8 and T.tree_key(tree2) == KEY_GROESSTER      # gleicher Fehler, anderer Baum: Gleichstand


def test_frozen_regression_tree_zufalls_seed_1():
    pb = _problem("reg")
    e, tree = pb.best(np.arange(1000), 3)
    assert abs(e - 300.15447855566055) < 1e-6 and T.n_leaves(tree) == 8 and T.tree_key(tree) == KEY_REG


def test_frozen_tie_count_on_150_rows():
    pb = _problem("cls")
    idx = np.arange(150)
    assert pb.best_count(idx, 3) == (26.0, 44)
    assert T.tree_key(pb.best(idx, 3, "kleinster")[1]) == KEY_150_KLEINSTER
    assert T.tree_key(pb.best(idx, 3, "groesster")[1]) == KEY_150_GROESSTER


def test_frozen_handbeispiel_counts():
    ZEILEN = [(a, b, (b if a <= 1 else 1 - b)) for a in range(4) for b in (0, 1)]
    M = np.array([[a <= 0, a <= 1, a <= 2, b <= 0] for a, b, _ in ZEILEN])
    p = T.Problem.from_matrix(M, np.array([k for _, _, k in ZEILEN]), "cls", minleaf=1)
    idx = np.arange(8)
    assert [len(p.all_trees(idx, d)) for d in (1, 2)] == [5, 42]
    assert p.best_count(idx, 1) == (4.0, 5) and p.best_count(idx, 2) == (0.0, 2)
