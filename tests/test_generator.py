"""Kopie-Test: Zufall (SplitMix64) und Generator-Formel gegen eingefrorene Werte; die Formelwerte wurden mit dem Original `verweilzeit-demo/vwz_generator.py` gerechnet."""

import math

import numpy as np
import pytest

import kfk_generator as G
from kfk_rng import SplitMix64


def test_splitmix64_frozen_values():
    r = SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [16294208416658607535, 7960286522194355700, 487617019471545679]   # bekannte Referenzfolge
    r = SplitMix64(12345)
    assert r.random() == pytest.approx(0.1330796686614273, abs=1e-15)
    assert r.normal() == pytest.approx(1.3016520405386593, abs=1e-12)


# (art, kunde, zoll, wochenende, vorlauf) -> (mu, s) aus verweilzeit-demo/vwz_generator.py (log_median_and_s)
ORIGINAL = [
    ((0, 0, 0, 0, 0.0), (0.8256154575589787, 0.55)),
    ((0, 3, 1, 1, 0.0), (2.1750109980150487, 0.55)),
    ((1, 5, 0, 0, 4.5), (1.420590659772188, 0.25)),
    ((2, 11, 0, 1, 2.0), (0.729018607690068, 0.35)),
    ((3, 7, 0, 1, 0.0), (2.4405779897132254, 0.9)),
    ((1, 0, 0, 0, 1.0), (0.2434193031515662, 0.25)),
    ((2, 2, 0, 0, 0.5), (-0.43503450035505015, 0.35)),
]


@pytest.mark.parametrize("args,expected", ORIGINAL)
def test_formula_equals_the_original(args, expected):
    mu, s = G.log_median_and_s(np.array(args[0]), args[1], args[2], args[3], args[4])
    assert float(mu) == pytest.approx(expected[0], rel=1e-12) and float(s) == pytest.approx(expected[1], rel=1e-12)


def test_customer_effects_are_the_numbers_of_the_original():
    assert len(G.UK) == 12 and G.UK[0] == -0.42714751093638936 and G.UK[11] == 0.2905490717255771
    assert abs(float(np.std(G.UK)) - 0.30) < 0.15          # Größenordnung der Streuung 0,30 (zwölf Ziehungen)


def test_draw_history_is_reproducible_and_prefix_consistent():
    X1, y1 = G.draw_history(200, 5)
    X2, y2 = G.draw_history(200, 5)
    X3, y3 = G.draw_history(50, 5)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert np.array_equal(X1[:50], X3) and np.array_equal(y1[:50], y3)       # Präfix: die ersten Zeilen hängen nicht von n ab
    assert not np.array_equal(G.draw_history(200, 6)[0], X1)


def test_draw_history_frozen_rows():
    X, y = G.draw_history(2000, 1)
    assert X[0].tolist() == [0.0, 6.0, 0.0, 1.0, 0.0, 0.3286238996822162]
    assert X[2].tolist()[:5] == [1.0, 1.0, 0.0, 0.0, 9.496842732371922]
    assert float(y[0]) == pytest.approx(0.7574905477977065, abs=1e-12)


def test_structural_rules_hold_in_every_row():
    X, _ = G.draw_history(3000, 2)
    art, zoll, vor = X[:, 0], X[:, 2], X[:, 4]
    assert np.all(zoll[art != 0] == 0)
    assert np.all(vor[np.isin(art, (0, 3))] == 0)
    assert np.all(vor[art == 1] >= 1) and np.all(vor[art == 1] <= 10)
    assert np.all(vor[art == 2] >= 0.5) and np.all(vor[art == 2] <= 6)


def test_distribution_in_a_band_with_standard_error():
    """Art-Anteile und Zollanteil gegen die exakten Werte (Band 4,5 Standardfehler); n = 20 000."""
    X, y = G.draw_history(20000, 3)
    n = len(X)
    for a, p in enumerate(G.ART_P):
        share = float(np.mean(X[:, 0] == a))
        assert abs(share - p) < 4.5 * math.sqrt(p * (1 - p) / n)
    imp = X[X[:, 0] == 0]
    p = G.P_ZOLL
    assert abs(float(imp[:, 2].mean()) - p) < 4.5 * math.sqrt(p * (1 - p) / len(imp))
    assert abs(float(X[:, 5].mean())) < 4.5 / math.sqrt(n) and abs(float(X[:, 5].std()) - 1) < 0.05      # Gewicht ~ N(0, 1)
    mu, s = G.log_median_and_s(X[:, 0].astype(int), X[:, 1].astype(int), X[:, 2], X[:, 3], X[:, 4])
    z = (y - mu) / s
    assert abs(float(z.mean())) < 4.5 / math.sqrt(n) and abs(float(z.std()) - 1) < 0.03                 # Rauschen ~ N(0, 1)


def test_truth_is_mu_against_log_threshold_and_probability_is_half_there():
    mu = np.log(4.0)
    assert bool(G.truth_longer(mu, 4)) and not bool(G.truth_longer(mu - 1e-9, 4))
    assert float(G.p_longer(mu, 0.5, 4)) == pytest.approx(0.5, abs=1e-12)
    assert float(G.p_longer(mu + 0.5, 0.5, 4)) == pytest.approx(0.8413447460685429, abs=1e-12)       # Phi(1)
