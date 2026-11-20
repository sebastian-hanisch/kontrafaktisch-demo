"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen: Instanz schon Klasse 0, Abstand größer als jede erreichbare Distanz, leere Politik, Rand des Rasters, Gleichstand,
Modell, das überall Klasse 1 sagt, Gegenprobe des Wegs mit der Modellvorhersage."""

import numpy as np
import pytest

import kfk_evaluation as EV
import kfk_fragility as F
import kfk_models as M
import kfk_recourse as R

SD_V = 2.4


def _lev(art=1, kunde=3, zoll=0, we=0, vor=8, g=6):
    return (art, kunde, zoll, we, vor, g)


def test_instance_already_class_zero_costs_nothing():
    lev = _lev()
    i = R.index_of(lev)
    sc = np.full(R.N_CAND, 1.0)
    sc[i] = -0.3
    assert R.cheapest(lev, R.POLICIES["A"], 0.0, sc, SD_V) == (0.0, i)
    assert R.cheapest(lev, R.POLICIES["A"], 0.5, sc, SD_V) is None            # mit Abstand 0,5 reicht der eigene Punkt (-0,3) nicht mehr: kein Weg, nicht Kosten 0
    sc[R.index_of(_lev(kunde=5))] = -0.8
    assert R.cheapest(lev, R.POLICIES["A"], 0.5, sc, SD_V)[0] == 1.0          # sondern der nächste Punkt (anderer Kunde, Kosten 1)


def test_margin_beyond_every_reachable_logit_gives_no_way_not_zero():
    sc = np.full(R.N_CAND, -0.5)
    assert R.cheapest(_lev(), R.POLICIES["A"], 1.0, sc, SD_V) is None
    assert R.cheapest(_lev(), R.POLICIES["A"], 0.5, sc, SD_V) is None         # -0,5 <= -0,5 - 1e-6 gilt nicht
    assert R.cheapest(_lev(), R.POLICIES["A"], 0.0, sc, SD_V)[0] == 0.0


def test_empty_policy_has_no_way_for_a_class_one_instance():
    lev = _lev()
    sc = np.full(R.N_CAND, -1.0)
    sc[R.index_of(lev)] = 2.0
    assert R.cheapest(lev, (0, 0, 0, 0, 0, 0), 0.0, sc, SD_V) is None


def test_instance_on_the_grid_edge_vorlauf_10_and_gewicht_edges():
    lev = _lev(vor=20, g=0)
    assert R.index_of(lev) >= 0 and R.index_of(_lev(vor=20, g=12)) >= 0
    sc = np.full(R.N_CAND, -1.0)
    sc[R.index_of(lev)] = 2.0
    c, i = R.cheapest(lev, R.POLICIES["C"], 0.0, sc, SD_V)                     # nur Vorlauf: der nächste Rasterpunkt, 0,5 Tage entfernt
    assert c == pytest.approx(0.5 / SD_V) and R.LEVELS[i][4] in (19,)


def test_structurally_impossible_candidates_do_not_exist():
    with pytest.raises(ValueError):
        R.index_of((0, 0, 1, 0, 3, 6))        # Import mit Vorlauf 1,5
    with pytest.raises(ValueError):
        R.index_of((1, 0, 1, 0, 4, 6))        # Zoll bei Export
    assert R.N_CAND == 13104 and R.N_GRID == 52416


def test_ties_are_resolved_by_the_smallest_candidate_index():
    lev = _lev()
    sc = np.full(R.N_CAND, 1.0)
    # zwei Kandidaten mit gleichen Kosten 1 (anderer Kunde), nur sie erfüllen die Bedingung
    a = R.index_of(_lev(kunde=5))
    b = R.index_of(_lev(kunde=9))
    sc[a] = sc[b] = -1.0
    assert R.cheapest(lev, R.POLICIES["A"], 0.0, sc, SD_V) == (1.0, min(a, b))
    assert EV.allowed_ties(lev, R.POLICIES["A"], 0.0, sc, SD_V, 1.0) == 2


def test_model_that_says_class_one_everywhere_gives_no_way():
    sc = np.full(R.N_CAND, 4.0)
    assert all(R.cheapest(_lev(), act, mg, sc, SD_V) is None for act in R.POLICIES.values() for mg in R.MARGINS)


@pytest.mark.parametrize("key", M.MODELS)
def test_every_way_is_class_zero_under_the_models_own_prediction(models, train, key):
    """Gegenprobe (Randpunkt-Falle): der Zielpunkt jedes Wegs wird mit der Vorhersage des Modells geprüft, nicht nur mit dem Quader; Logit <= -Abstand."""
    m = models[key]
    sc = m.cand_scores()
    n = 0
    for lev in F.instances_for(m, train["Xt"], 8):
        for mg in R.MARGINS:
            bf = R.cheapest(lev, R.POLICIES["A"], mg, sc, train["sd_v"])
            if bf is None:
                continue
            x = R.XC[bf[1]][None, :]
            assert m.logit(x)[0] <= -mg - R.EPS + 1e-9
            n += 1
    assert n >= 10


def test_ways_change_only_allowed_features_and_cost_is_zero_only_for_class_zero(models, train):
    m = models["boost"]
    sc = m.cand_scores()
    for lev in F.instances_for(m, train["Xt"], 6):
        assert sc[R.index_of(lev)] > 0                                        # Instanzen sind Klasse 1
        for pol, act in R.POLICIES.items():
            bf = R.cheapest(lev, act, 0.0, sc, train["sd_v"])
            if bf is None:
                continue
            assert bf[0] > 0
            changed = {j for j, name in enumerate(R.FEATURES) if name in R.changed_features(lev, bf[1])}
            assert all(act[j] for j in changed)


def test_level_text_and_way_for_clamp_the_instance_index(models, train):
    fit = EV.Fit(models["reg"], train["Xr"], train["sd_v"], train["Xt"], F.instances_for(models["reg"], train["Xt"]), 0.8, 1, "reg")
    w = EV.way_for(fit, 99, "A", 0.0)                                         # über die Zahl der Instanzen hinaus wird begrenzt
    assert w.instance == len(fit.levs) and w.cost is not None and w.agree is True
    assert EV.level_text((1, 3, 0, 1, 8, 7))["Vorlauf"] == "4,0 Tage" and EV.level_text((1, 3, 0, 1, 8, 7))["Gewicht"] == "0,5"
