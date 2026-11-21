"""Die fünf Modellklassen der Demo (alle sehen dieselben 20 Spalten; Hyperparameter fest) und ihr Logit der Klasse 1 („länger als die Schwelle“).

* Logistische Regression (L2, C = 1), * Baum (scikit-learn, Tiefe 4, mindestens 30 Zeilen je Blatt), * Boosting (100 Bäume, Tiefe 3), * kleines Netz (16 Einheiten, L-BFGS, alpha = 3),
* exakt optimaler Baum (Tiefe 3, DP über das Raster aus `kfk_tree.py`, mindestens 10 Zeilen je Blatt, Gleichstandsregel kleinster Index; die Variante „größter Index“ zeigt die Spanne).
Bei Bäumen ist das Logit der Blattanteil der Klasse 1, auf [1e-4, 1 - 1e-4] begrenzt. Die Zufalls-Seeds der Modelle (Boosting, Netz, Baum) sind fest je Training.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import kfk_recourse as R
from kfk_tree import Problem, leaf_ids, n_leaves, raster_spec, split_matrix

MODELS = ("reg", "baum", "boost", "netz", "opt")
MODEL_NAMES = {
    "reg": "Logistische Regression",
    "baum": "Baum (scikit-learn, Tiefe 4)",
    "boost": "Boosting (100 Bäume, Tiefe 3)",
    "netz": "Kleines Netz (16 Einheiten)",
    "opt": "Exakt optimaler Baum (Tiefe 3)",
}
SPEC = raster_spec()
OPT_DEPTH = 3
NET_MAX_ITER = 500
CLIP = 1e-4


def clip_logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def sklearn_boxes(tree, d):
    """Blattquader (lo, hi] je Entwurfsspalte des scikit-learn-Baums und Blatt-Logit."""
    t = tree.tree_
    out = []

    def rec(node, lo, hi):
        if t.children_left[node] == -1:
            v = t.value[node][0]
            out.append((lo.copy(), hi.copy(), float(clip_logit(np.array([v[1] / v.sum()]))[0])))
            return
        j, thr = t.feature[node], t.threshold[node]
        hi2 = hi.copy()
        hi2[j] = min(hi2[j], thr)
        rec(t.children_left[node], lo, hi2)
        lo2 = lo.copy()
        lo2[j] = max(lo2[j], thr)
        rec(t.children_right[node], lo2, hi)

    rec(0, np.full(d, -np.inf), np.full(d, np.inf))
    return out


class Model:
    """fit(Xr, lab) mit gerasterten Trainingsdaten und 0/1-Labels; danach `logit(X)` und `cand_scores()` (Logit an allen 13 104 Kandidaten)."""

    def __init__(self, key, seed, rule="kleinster"):
        if key not in MODELS:
            raise ValueError(key)
        self.key, self.seed, self.rule = key, seed, rule

    def fit(self, Xr, lab):
        D = R.design(Xr)
        k = self.key
        if k == "reg":
            self.m = LogisticRegression(C=1.0, max_iter=2000).fit(D, lab)
        elif k == "baum":
            self.m = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=self.seed).fit(D, lab)
        elif k == "boost":
            self.m = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=self.seed).fit(D, lab)
        elif k == "netz":
            ct = ColumnTransformer([("p", "passthrough", list(range(16))), ("s", StandardScaler(), [16, 17, 18, 19])])
            self.m = make_pipeline(ct, MLPClassifier(hidden_layer_sizes=(16,), alpha=3.0, max_iter=NET_MAX_ITER, solver="lbfgs", random_state=self.seed)).fit(D, lab)
            self.converged = bool(self.m[-1].n_iter_ < NET_MAX_ITER)     # False: das Iterationslimit wurde erreicht (Meldung in der App)
        else:
            pb = Problem(Xr, lab.astype(float), SPEC, "cls")
            self.err, self.tree = pb.best(np.arange(len(Xr)), OPT_DEPTH, self.rule)
            ids, nl = leaf_ids(self.tree, pb.M)
            self.frac = np.array([lab[ids == j].mean() for j in range(nl)])
        return self

    def _logit(self, D, M):
        k = self.key
        if k in ("reg", "boost"):
            return self.m.decision_function(D)
        if k in ("baum", "netz"):
            return clip_logit(self.m.predict_proba(D)[:, 1])
        return clip_logit(self.frac[leaf_ids(self.tree, M)[0]])

    def logit(self, X):
        return self._logit(R.design(X), split_matrix(X, SPEC) if self.key == "opt" else None)

    def cand_scores(self):
        """Logit an allen zulässigen Kandidaten (Reihenfolge der Kandidaten wie in `kfk_recourse`)."""
        if not hasattr(self, "_cs"):
            self._cs = self._logit(R.DC, R.tree_cut_matrix(SPEC) if self.key == "opt" else None)
        return self._cs

    def n_leaves(self):
        if self.key == "baum":
            return int(self.m.get_n_leaves())
        if self.key == "opt":
            return n_leaves(self.tree)
        return None

    def leaf_of_candidates(self):
        """Blatt-Nummer je Kandidat (nur Bäume)."""
        if self.key == "baum":
            return self.m.apply(R.DC)
        if self.key == "opt":
            return leaf_ids(self.tree, R.tree_cut_matrix(SPEC))[0]
        return None

    def boxes(self):
        """Blattquader als Liste (Zugehörigkeit je Kandidat, Blatt-Logit), rein aus den Schnitten (nur Bäume)."""
        if self.key == "baum":
            out = []
            for lo, hi, lg in sklearn_boxes(self.m, R.DC.shape[1]):
                out.append((np.all((R.DC > lo[None, :]) & (R.DC <= hi[None, :]), axis=1), lg))
            return out
        if self.key == "opt":
            from kfk_tree import in_box, leaf_boxes
            Mc = R.tree_cut_matrix(SPEC)
            vals = np.array(clip_logit(self.frac))
            out = []
            for j, (cond, _) in enumerate(leaf_boxes(self.tree, Mc, Mc.shape[1])):
                out.append((in_box(Mc, cond), float(vals[j])))
            return out
        return None

    def linear_weights(self):
        if self.key != "reg":
            return None
        return self.m.coef_[0], float(self.m.intercept_[0])
