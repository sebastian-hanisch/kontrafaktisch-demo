"""Exakt optimaler Baum: Dynamische Programmierung gegen Aufzählung aller Bäume (anderer Rechenweg), Gleichstandsregel, Blattquader.
Handbeispiel Teil 6 des Plans (Brüche, Ganzzahlen): 5 Bäume bis Tiefe 1, 42 bis Tiefe 2, 2 optimale Bäume mit Fehler 0."""

import numpy as np
import pytest

import kfk_fragility as F
import kfk_generator as G
import kfk_recourse as R
import kfk_tree as T

# Datensatz des Handbeispiels: x1 in {0..3}, x2 in {0, 1}; Klasse = x2 für x1 <= 1, sonst 1 - x2; vier Schnitte (x1 <= 0, x1 <= 1, x1 <= 2, x2 <= 0)
ZEILEN = [(a, b, (b if a <= 1 else 1 - b)) for a in range(4) for b in (0, 1)]


def toy():
    M = np.array([[a <= 0, a <= 1, a <= 2, b <= 0] for a, b, _ in ZEILEN])
    return T.Problem.from_matrix(M, np.array([k for _, _, k in ZEILEN]), "cls", minleaf=1), np.arange(8)


def test_handbeispiel_tree_counts_and_optimum():
    p, idx = toy()
    assert len(p.all_trees(idx, 1)) == 5 and p.best_count(idx, 1) == (4.0, 5)         # Blatt und vier Schnitte, alle mit 4 Fehlern von 8
    assert len(p.all_trees(idx, 2)) == 1 + 8 + 9 + 8 + 16 == 42                         # 2*4, 3*3, 4*2, 4*4 Teilbaumpaare
    assert p.best_count(idx, 2) == (0.0, 2)
    opt = [t for t in p.all_trees(idx, 2) if p.tree_error(t) == 0]
    assert len(opt) == 2 and {t[1] for t in opt} == {1, 3}                              # Wurzeln x1 <= 1 und x2 <= 0


def test_handbeispiel_two_optimal_trees_are_one_function_on_all_rows():
    p, idx = toy()
    opt = [t for t in p.all_trees(idx, 2) if p.tree_error(t) == 0]
    assert all(np.array_equal(T.predict(opt[0], p.M), T.predict(t, p.M)) for t in opt)


def test_handbeispiel_rule_picks_smallest_and_largest_index():
    p, idx = toy()
    e, tk = p.best(idx, 2, "kleinster")
    e2, tg = p.best(idx, 2, "groesster")
    assert e == e2 == 0.0
    assert tk[1] == 1 and tg[1] == 3                                                    # Wurzel x1 <= 1 (Index 1) vor x2 <= 0 (Index 3)


def test_handbeispiel_way_through_leaf_boxes_equals_enumeration():
    """Instanz (3, 0) ist Klasse 1; Kosten |dx1| / 2 + |dx2|; billigster Weg Kosten 1 mit zwei gleich billigen Wegen (1, 0) und (3, 1)."""
    p, idx = toy()
    tree = p.best(idx, 2)[1]
    raster = [(a, b) for a in range(4) for b in (0, 1)]
    Mc = np.array([[a <= 0, a <= 1, a <= 2, b <= 0] for a, b in raster])
    assert T.predict(tree, Mc)[raster.index((3, 0))] == 1

    def cost(q):
        return abs(q[0] - 3) / 2 + abs(q[1] - 0)

    by_predict = sorted((cost(q), q) for q, v in zip(raster, T.predict(tree, Mc)) if v == 0)
    boxes = T.leaf_boxes(tree, Mc, Mc.shape[1])
    by_box = sorted((cost(q), q) for cond, val in boxes if val == 0 for q, inside in zip(raster, T.in_box(Mc, cond)) if inside)
    assert by_predict == by_box
    assert by_box[0][0] == 1 and {q for c, q in by_box if c == 1} == {(1, 0), (3, 1)}


SPEC_KLEIN = [(0, "menge", (0,)), (0, "menge", (0, 1)), (2, "le", 0.5), (3, "le", 0.5), (4, "le", 2.25), (4, "le", 5.25)]


def _small_problem(seed, mode, rows=90, minleaf=5):
    X, y = G.draw_history(rows, 900 + seed)
    lab = F.labels(y).astype(float) if mode == "cls" else y
    return T.Problem(R.to_raster(X), lab, SPEC_KLEIN, mode, minleaf=minleaf), np.arange(rows)


@pytest.mark.parametrize("mode", ["cls", "reg"])
@pytest.mark.parametrize("depth", [2, 3])
def test_dp_equals_enumeration_of_all_trees(mode, depth):
    """Orakel: kleinster Fehler, Zahl der Bäume mit diesem Fehler und die Wahl nach beiden Gleichstandsregeln gegen die Aufzählung aller Bäume (Fehler aus der Vorhersage gerechnet)."""
    for seed in range(5 if depth == 3 else 8):
        p, idx = _small_problem(seed, mode)
        trees = p.all_trees(idx, depth)
        errs = np.array([p.tree_error(t) for t in trees])
        mn = errs.min()
        opt = [t for t, e in zip(trees, errs) if e <= mn + p.tol]
        e_s, t_s = p.best(idx, depth, "kleinster")
        e_g, t_g = p.best(idx, depth, "groesster")
        assert e_s == pytest.approx(mn, abs=1e-6) and e_g == pytest.approx(mn, abs=1e-6)
        n, c = p.best_count(idx, depth)
        assert n == pytest.approx(mn, abs=1e-6) and c == len(opt)
        assert T.tree_key(t_s) == min(T.tree_key(t) for t in opt)
        assert T.tree_key(t_g) == max(T.tree_key(t) for t in opt)


def test_ties_are_common_and_the_two_rules_can_differ():
    """Gleichstände sind der Normalfall (Klassifikation, ganzzahlige Fehler); die Regeln wählen verschiedene Bäume mit gleichem Fehler."""
    verschieden = 0
    for seed in range(6):
        p, idx = _small_problem(seed, "cls", rows=120)
        n, c = p.best_count(idx, 2)
        if c > 1:
            verschieden += int(T.tree_key(p.best(idx, 2, "kleinster")[1]) != T.tree_key(p.best(idx, 2, "groesster")[1]))
    assert verschieden >= 1


def test_leaf_ids_boxes_and_prediction_agree_on_real_data(train):
    spec = T.raster_spec()
    assert len(spec) == 47
    pb = T.Problem(train["Xr"], train["lab"].astype(float), spec, "cls")
    e, tree = pb.best(np.arange(1000), 3)
    assert 3 <= T.n_leaves(tree) <= 8
    assert pb.tree_error(tree) == e                                           # Fehler aus der Vorhersage gleich Fehler der DP
    ids, nl = T.leaf_ids(tree, pb.M)
    assert nl == T.n_leaves(tree) and set(ids) == set(range(nl))
    Mc = R.tree_cut_matrix(spec)
    boxes = T.leaf_boxes(tree, Mc, Mc.shape[1])
    inside = np.array([T.in_box(Mc, cond) for cond, _ in boxes])
    assert np.all(inside.sum(axis=0) == 1)                                    # jede Zeile liegt in genau einem Quader
    assert np.array_equal(np.argmax(inside, axis=0), T.leaf_ids(tree, Mc)[0])


def test_minimum_leaf_size_is_respected():
    p, idx = _small_problem(1, "cls", rows=90, minleaf=20)
    _, tree = p.best(idx, 3)
    ids, nl = T.leaf_ids(tree, p.M)
    assert min(np.bincount(ids)) >= 20 or nl == 1


def test_depth_zero_and_too_small_set_are_a_single_leaf():
    p, idx = _small_problem(2, "cls", rows=8, minleaf=5)
    assert p.best(idx, 3)[1][0] == "blatt"                                    # kein Schnitt lässt beidseitig 5 Zeilen übrig
    p, idx = _small_problem(2, "cls", rows=60)
    assert p.best(idx, 0)[1][0] == "blatt"
