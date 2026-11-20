"""Orakel (anderer Rechenweg als die Demo): der Weg durch Aufzählung gegen (a) eine unabhängige Schleife über das rohe Raster aus Python-Zahlen, (b) das ganzzahlige Programm,
(c) Blattquader bei Bäumen; dazu der Kern des Baums (DP gegen Aufzählung, in `test_tree.py`) und die geschlossene Form der Ungültigkeit gegen Integration und Simulation."""

import itertools
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

import kfk_evaluation as EV
import kfk_fragility as F
import kfk_recourse as R
from kfk_rng import SplitMix64


def brute_python(lev, act, margin, w, b, sd_v):
    """Unabhängige Aufzählung: Schleife über alle 52 416 Rasterpunkte, Strukturregeln und Kosten neu geschrieben, Logit als Summe der Koeffizienten. Liefert die kleinsten Kosten oder None."""
    a0, k0, z0, w0, v0, g0 = lev
    best = None
    for a, k, z, we, v, g in itertools.product(range(4), range(12), range(2), range(2), range(21), range(13)):
        if z == 1 and a != 0:
            continue
        if a in (0, 3) and v != 0:
            continue
        if a == 1 and v < 2:
            continue
        if a == 2 and v < 1:
            continue
        if (a != a0 and not act[0]) or (k != k0 and not act[1]) or (z != z0 and not act[2]) or (we != w0 and not act[3]) or (v != v0 and not act[4]) or (g != g0 and not act[5]):
            continue
        logit = b + w[a] + w[4 + k] + w[16] * z + w[17] * we + w[18] * (v / 2) + w[19] * ((g - 6) / 2)
        if logit > -margin - 1e-6:
            continue
        cost = (a != a0) + (k != k0) + (z != z0) + (we != w0) + abs(v / 2 - v0 / 2) / sd_v + abs((g - 6) / 2 - (g0 - 6) / 2)
        if best is None or cost < best:
            best = cost
    return best


def _lr_cases(models, train, n_inst=3):
    m = models["reg"]
    w, b = m.linear_weights()
    return m, w, b, F.instances_for(m, train["Xt"], n_inst)


@pytest.mark.parametrize("pol,mg", [("A", 0.0), ("A", 1.0), ("B", 0.5), ("C", 0.0)])
def test_enumeration_equals_independent_python_loop(models, train, pol, mg):
    m, w, b, levs = _lr_cases(models, train, 2)
    sc = m.cand_scores()
    for lev in levs:
        got = R.cheapest(lev, R.POLICIES[pol], mg, sc, train["sd_v"])
        ref = brute_python(lev, R.POLICIES[pol], mg, w, b, train["sd_v"])
        assert (got is None) == (ref is None)
        if got:
            assert got[0] == pytest.approx(ref, abs=1e-9)


def test_integer_program_equals_enumeration_on_random_models(train):
    """Zufällige lineare Modelle (feste Zufalls-Seeds), zufällige Instanzen aus den Testzeilen, alle Politiken und Abstände: 100 Fälle."""
    rng = SplitMix64(2026)
    ok = tot = 0
    for case in range(4):
        w = np.array([rng.normal() * 0.8 for _ in range(20)])
        b = rng.normal() * 0.5
        sc = R.DC @ w + b
        for row in range(3):
            lev = R.levels(train["Xt"][case * 3 + row])
            for pol, act in R.POLICIES.items():
                for mg in R.MARGINS:
                    bf = R.cheapest(lev, act, mg, sc, train["sd_v"])
                    mp = R.milp_way(lev, act, mg, w, b, train["sd_v"])
                    tot += 1
                    ok += int((bf is None) == (mp is None) and (bf is None or abs(bf[0] - mp[0]) < 1e-6))
    assert tot == 108 and ok == tot


@pytest.mark.parametrize("key", ["baum", "opt"])
def test_leaf_boxes_equal_enumeration_for_trees(models, train, key):
    m = models[key]
    sc = m.cand_scores()
    boxes = m.boxes()
    tot = 0
    for lev in F.instances_for(m, train["Xt"], 5):
        for pol, act in R.POLICIES.items():
            for mg in R.MARGINS:
                bf = R.cheapest(lev, act, mg, sc, train["sd_v"])
                bx = R.box_way(lev, act, mg, boxes, train["sd_v"])
                assert (bf is None) == (bx is None)
                if bf:
                    assert bf[0] == pytest.approx(bx[0], abs=1e-9)
                tot += 1
    assert tot == 45


def test_tree_scores_equal_leaf_fraction_lookup(models, train):
    """Das Logit des Baums an den Kandidaten ist der Blattanteil: aus den Quadern (ohne Modellaufruf) gerechnet gleich cand_scores."""
    for key in ("baum", "opt"):
        m = models[key]
        from_boxes = np.full(R.N_CAND, np.nan)
        for member, lg in m.boxes():
            from_boxes[member] = lg
        assert np.allclose(from_boxes, m.cand_scores(), atol=1e-12)


def test_closed_form_of_invalidity_equals_integration_and_simulation():
    """P = (s/a)(t Phi(-t) + phi(0) - phi(t)), t = a/s: gegen numerische Integration und eine Simulation im Band (4,5 Standardfehler); Handwert 0,2464 für a = 0,3, s = 0,2."""
    a, s = 0.3, 0.2
    p = EV.p_invalid_model(a, s)
    assert p == pytest.approx(EV.p_invalid_numeric(a, s), abs=1e-12)
    assert round(p, 4) == 0.2464
    num, _ = quad(lambda x: norm.cdf(-x / s), 0, a)
    assert p == pytest.approx(num / a, abs=1e-12)
    rng = SplitMix64(77)
    n = 40000
    hits = sum(1 for _ in range(n) if s * rng.normal() > a * rng.random())
    sim = hits / n
    assert abs(sim - p) < 4.5 * math.sqrt(p * (1 - p) / n)
    assert abs(EV.p_invalid_model(1e-6, s) - 0.5) < 1e-5                      # kein Spielraum: die Hälfte
    assert EV.p_invalid_model(3.0, s) < EV.p_invalid_model(0.3, s) < EV.p_invalid_model(0.03, s) < 0.5
