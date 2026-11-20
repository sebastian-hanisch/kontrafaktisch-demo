"""Modellklassen: Hyperparameter, Logit an Instanzen gleich Logit an den Kandidaten, Genauigkeit im Band, Baum-Eigenschaften."""

import numpy as np
import pytest

import kfk_fragility as F
import kfk_generator as G
import kfk_models as M
import kfk_recourse as R


def test_hyperparameters_are_fixed(models):
    assert models["reg"].m.C == 1.0
    assert models["baum"].m.max_depth == 4 and models["baum"].m.min_samples_leaf == 30
    assert models["boost"].m.n_estimators == 100 and models["boost"].m.max_depth == 3
    assert models["netz"].m[-1].hidden_layer_sizes == (16,) and models["netz"].m[-1].alpha == 3.0
    assert M.OPT_DEPTH == 3 and set(M.MODELS) == set(M.MODEL_NAMES)


def test_logit_of_rows_equals_logit_at_the_same_candidates(models):
    """Ein Kandidat, als Zeile an `logit` gegeben, bekommt dasselbe Logit wie in cand_scores (zwei Wege zum selben Wert)."""
    pick = np.array([0, 17, 400, 5000, 9000, 13103])
    for key, m in models.items():
        assert np.allclose(m.logit(R.XC[pick]), m.cand_scores()[pick], atol=1e-9), key


def test_accuracy_in_a_band(models):
    X, y = G.draw_history(1500, 100777)
    Xt, lt = R.to_raster(X), F.labels(y)
    for key, m in models.items():
        acc = float(np.mean((m.logit(Xt) > 0) == (lt == 1)))
        assert 0.68 < acc < 0.86, (key, acc)


def test_tree_logits_are_clipped_leaf_fractions(models):
    for key in ("baum", "opt"):
        lg = models[key].cand_scores()
        assert lg.min() >= np.log(M.CLIP / (1 - M.CLIP)) - 1e-9 and lg.max() <= np.log((1 - M.CLIP) / M.CLIP) + 1e-9
    n = models["opt"].n_leaves()
    assert 2 <= n <= 8 and models["baum"].n_leaves() <= 16
    assert len(np.unique(models["opt"].cand_scores())) <= n


def test_optimal_tree_has_no_larger_training_error_than_the_sklearn_tree_of_the_same_depth(train):
    """Exakt gegen gierig: der exakt optimale Baum der Tiefe 3 ist auf dem Training nie schlechter als ein Baum von scikit-learn der Tiefe 3 mit denselben Schnittmöglichkeiten (Mindestblatt 10)."""
    from sklearn.tree import DecisionTreeClassifier
    m = M.Model("opt", 1).fit(train["Xr"], train["lab"])
    sk = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=1).fit(R.design(train["Xr"]), train["lab"])
    assert m.err <= int((sk.predict(R.design(train["Xr"])) != train["lab"]).sum())


def test_the_second_rule_gives_the_same_training_error(train):
    a = M.Model("opt", 1).fit(train["Xr"], train["lab"])
    b = M.Model("opt", 1, "groesster").fit(train["Xr"], train["lab"])
    assert a.err == b.err


def test_wrong_key_is_rejected_and_linear_weights_only_for_regression(models):
    with pytest.raises(ValueError):
        M.Model("xyz", 1)
    assert models["reg"].linear_weights()[0].shape == (20,) and models["boost"].linear_weights() is None and models["boost"].boxes() is None
