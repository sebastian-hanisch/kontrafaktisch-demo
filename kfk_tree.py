"""Exakt optimaler Entscheidungsbaum: dynamische Programmierung über Teilmengen, Aufzählung aller Bäume als Orakel, Gleichstandsregel, Weg über Blattquader.

Dieser Baum-Code wird beim Bau des Stücks `duenner-baum-demo` kopiert, nicht importiert (Kopie-Test gegen die eingefrorenen Werte aus `tests/test_tree_frozen.py`).

**Optimal** heißt: kleinster Trainingsfehler über alle Bäume der Tiefe <= d, deren Schnitte aus einem festen Raster stammen, mit Mindestblattgröße. Jede Aussage
„exakt optimal“ gilt über dieses Raster. Die DP ist exakt (nicht gierig): zu jeder Teilmenge der Zeilen gibt es den besten Teilbaum je Restiefe.

**Gleichstandsregel (Linienregel, Stück 4 und 6 wortgleich).** Der exakt optimale Baum ist meist nicht eindeutig. Unter allen Bäumen mit dem kleinsten Fehler (Gleichheit auf
`tol`) gewinnt der mit dem **kleinsten Index in der festen Aufzählungsreihenfolge**: die Vorordnung der Schnittindizes (Wurzel, links, rechts; Blatt = -1) wird lexikografisch
verglichen, ein Blatt gewinnt also gegen jeden Schnitt. Die zweite Regel `regel="groesster"` nimmt den lexikografisch größten Baum und zeigt die Spanne. Beide Regeln sind exakt
(geprüft gegen die Aufzählung aller Bäume), die Funktion des gewählten Baums außerhalb der Trainingszeilen hängt aber an der Regel.

Baum: ('blatt', wert) oder ('schnitt', k, links, rechts); der Schnitt k teilt die Zeilen in die mit M[:, k] wahr (links) und falsch (rechts).
"""

import numpy as np

MINLEAF = 10
TOL_CLS = 1e-9
TOL_REG = 1e-7


def raster_spec():
    """Raster der Schnitte (47): Art alle 7 Zweiteilungen, Kunde 11 ordinale Schwellen, Zoll, Wochenende, Vorlauf 20 Schwellen, Gewicht 7. Eintrag (Merkmal, Art, Parameter)."""
    spec = []
    for s in [(0,), (0, 1), (0, 2), (0, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3)]:
        spec.append((0, "menge", s))
    for k in range(11):
        spec.append((1, "le", k + 0.5))
    spec.append((2, "le", 0.5))
    spec.append((3, "le", 0.5))
    for t in np.arange(0.0, 10.0, 0.5):
        spec.append((4, "le", float(t) + (0.25 if t > 0 else 0.0)))
    for t in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
        spec.append((5, "le", t))
    return spec


def split_matrix(X, spec):
    """Boolesche Matrix (Zeilen x Schnitte): Zeile liegt links des Schnitts."""
    M = np.zeros((len(X), len(spec)), dtype=bool)
    for k, (j, art, p) in enumerate(spec):
        M[:, k] = np.isin(X[:, j], p) if art == "menge" else (X[:, j] <= p)
    return M


def tree_key(tree):
    """Vorordnung der Schnittindizes (Blatt = -1) als Tupel; lexikografisch kleiner = früher in der Aufzählungsreihenfolge."""
    if tree[0] == "blatt":
        return (-1,)
    return (tree[1],) + tree_key(tree[2]) + tree_key(tree[3])


def n_leaves(tree):
    return 1 if tree[0] == "blatt" else n_leaves(tree[2]) + n_leaves(tree[3])


def leaf_ids(tree, M):
    """Nummer des Blatts je Zeile (von links nach rechts gezählt) und Zahl der Blätter; rein aus den Schnitten, ohne Vorhersage."""
    ids = np.zeros(len(M), dtype=int)
    counter = [0]

    def rec(t, rows):
        if t[0] == "blatt":
            ids[rows] = counter[0]
            counter[0] += 1
            return
        m = M[rows, t[1]]
        rec(t[2], rows[m])
        rec(t[3], rows[~m])

    rec(tree, np.arange(len(M)))
    return ids, counter[0]


def predict(tree, M):
    """Blattwert je Zeile."""
    out = np.zeros(len(M))

    def rec(t, rows):
        if t[0] == "blatt":
            out[rows] = t[1]
            return
        m = M[rows, t[1]]
        rec(t[2], rows[m])
        rec(t[3], rows[~m])

    rec(tree, np.arange(len(M)))
    return out


def leaf_values(tree):
    """Blattwerte von links nach rechts (gleiche Reihenfolge wie `leaf_ids`)."""
    return [tree[1]] if tree[0] == "blatt" else leaf_values(tree[2]) + leaf_values(tree[3])


class Problem:
    """Trainingsdaten (Matrix der Schnitte M, Zielwerte y) für die exakte Suche. mode 'cls': Zielwerte 0/1, Fehler = Zahl der falsch eingestuften Zeilen;
    mode 'reg': Fehler = Summe der Abweichungsquadrate zum Blattmittel."""

    def __init__(self, X, y, spec, mode="cls", minleaf=MINLEAF):
        self.spec, self.mode, self.minleaf = spec, mode, minleaf
        self.tol = TOL_CLS if mode == "cls" else TOL_REG
        self.M = split_matrix(X, spec)
        self.Mf = self.M.astype(float)
        self.y = np.asarray(y, dtype=float)

    @classmethod
    def from_matrix(cls, M, y, mode="cls", minleaf=MINLEAF):
        """Direkt aus einer fertigen Schnittmatrix (kleine Handbeispiele und Tests)."""
        p = cls.__new__(cls)
        p.spec, p.mode, p.minleaf = None, mode, minleaf
        p.tol = TOL_CLS if mode == "cls" else TOL_REG
        p.M = np.asarray(M, dtype=bool)
        p.Mf = p.M.astype(float)
        p.y = np.asarray(y, dtype=float)
        return p

    def _val(self, idx):
        y = self.y[idx]
        return float(y.mean()) if self.mode == "reg" else float(y.sum() * 2 > len(y))

    def leaf_err(self, idx):
        y = self.y[idx]
        if self.mode == "reg":
            return float(((y - y.mean()) ** 2).sum())
        p = y.sum()
        return float(min(p, len(y) - p))

    def stump_errs(self, idx):
        """(Fehler als Blatt, Fehlerfeld aller Schnitte; inf, wo ein Teil kleiner als MINLEAF ist)."""
        y = self.y[idx]
        n = len(idx)
        Mi = self.Mf[idx]
        nl = Mi.sum(axis=0)
        nr = n - nl
        ok = (nl >= self.minleaf) & (nr >= self.minleaf)
        sl = y @ Mi
        if self.mode == "reg":
            sy, syy = y.sum(), (y * y).sum()
            leaf = syy - sy * sy / n
            with np.errstate(divide="ignore", invalid="ignore"):
                e = syy - sl ** 2 / nl - (sy - sl) ** 2 / nr
        else:
            p = y.sum()
            leaf = min(p, n - p)
            e = np.minimum(sl, nl - sl) + np.minimum(p - sl, nr - (p - sl))
        e = np.where(ok, e, np.inf)
        return float(leaf), e

    def best(self, idx, depth, regel="kleinster"):
        """(Fehler, Baum) des exakt besten Baums der Tiefe <= depth auf den Zeilen idx. Gleichstand nach `regel` ('kleinster' oder 'groesster' Index, siehe Kopf)."""
        leaf, e = self.stump_errs(idx)
        best_e, best_t = leaf, ("blatt", self._val(idx))
        if depth == 0:
            return best_e, best_t
        tol = self.tol
        kleinster = regel == "kleinster"
        if depth == 1:
            # Stumpf: der Einzelfehler e ist schon der Fehler des Baums; Gleichstand nach Regel (kleinster Index: Blatt vor Schnitt vor größerem Schnitt)
            fin = np.isfinite(e)
            if not fin.any():
                return best_e, best_t
            mn = float(e[fin].min())
            if mn < best_e - tol or (not kleinster and mn <= best_e + tol):
                ties = np.flatnonzero(fin & (e <= mn + tol))
                k = int(ties[0] if kleinster else ties[-1])
                m = self.M[idx, k]
                return float(e[k]), ("schnitt", k, ("blatt", self._val(idx[m])), ("blatt", self._val(idx[~m])))
            return best_e, best_t
        best_k = (-1,)
        # Schnitte nach ihrem Einzelfehler geordnet: ein früh gutes Optimum macht das Abschneiden wirksam; die Gleichstandsregel wirkt über den Vergleich der
        # Vorordnungen (nicht über die Reihenfolge der Schleife), das Ergebnis hängt also nicht an dieser Ordnung.
        for k in np.argsort(e, kind="stable"):
            if not np.isfinite(e[k]):
                break
            m = self.M[idx, k]
            el, tl = self.best(idx[m], depth - 1, regel)
            if el > best_e + tol:
                continue
            er, tr = self.best(idx[~m], depth - 1, regel)
            tot = el + er
            if tot > best_e + tol:
                continue
            cand = ("schnitt", int(k), tl, tr)
            ck = tree_key(cand)
            if tot < best_e - tol or (ck < best_k if kleinster else ck > best_k):
                best_e, best_t, best_k = tot, cand, ck
        return best_e, best_t

    def best_count(self, idx, depth):
        """(kleinster Fehler, Zahl der Bäume (Strukturen) der Tiefe <= depth mit genau diesem Fehler), DP über Teilmengen; ein Blatt und jede Schnittfolge zählen einzeln."""
        leaf, e = self.stump_errs(idx)
        opts = [(leaf, 1)]
        if depth > 0:
            for k in np.flatnonzero(np.isfinite(e)):
                m = self.M[idx, k]
                (e1, c1), (e2, c2) = self.best_count(idx[m], depth - 1), self.best_count(idx[~m], depth - 1)
                opts.append((e1 + e2, c1 * c2))
        mn = min(o[0] for o in opts)
        return mn, sum(c for er, c in opts if er <= mn + self.tol)

    # ---- Orakel: Aufzählung aller Bäume aus der Definition (nur für kleine Raster und kleine Zeilenzahl)
    def all_trees(self, idx, depth):
        """Alle Bäume der Tiefe <= depth auf idx als verschachtelte Tupel (ohne DP, ohne Fehlerfeld)."""
        out = [("blatt", self._val(idx))]
        if depth > 0:
            for k in range(self.M.shape[1]):
                m = self.M[idx, k]
                li, re = idx[m], idx[~m]
                if len(li) < self.minleaf or len(re) < self.minleaf:
                    continue
                for a in self.all_trees(li, depth - 1):
                    for b in self.all_trees(re, depth - 1):
                        out.append(("schnitt", k, a, b))
        return out

    def tree_error(self, tree, idx=None):
        """Trainingsfehler eines Baums, aus der Vorhersage gerechnet (anderer Rechenweg als `stump_errs`)."""
        idx = np.arange(len(self.y)) if idx is None else idx
        pred = predict(tree, self.M[idx])
        y = self.y[idx]
        if self.mode == "reg":
            return float(((y - pred) ** 2).sum())
        return float((pred != y).sum())


def leaf_boxes(tree, M_cols, n_cols):
    """Wege über Blattquader: je Blatt die Bedingungen als Vektor (-1 verlangt Schnitt falsch, +1 verlangt Schnitt wahr, 0 frei) und der Blattwert.
    Rasterpunkte gehören zu einem Blatt, wenn ihre Schnittmatrix (M_cols) die Bedingungen erfüllt; das braucht keine Vorhersage."""
    boxes = []

    def rec(t, cond):
        if t[0] == "blatt":
            boxes.append((cond.copy(), t[1]))
            return
        k = t[1]
        c1 = cond.copy()
        c1[k] = 1
        rec(t[2], c1)
        c2 = cond.copy()
        c2[k] = -1
        rec(t[3], c2)

    rec(tree, np.zeros(n_cols, dtype=int))
    return boxes


def in_box(M_cols, cond):
    """Welche Zeilen der Schnittmatrix liegen im Quader `cond` (Aufzählung über die Quader, ohne Baumvorhersage)."""
    ok = np.ones(len(M_cols), dtype=bool)
    for k in np.flatnonzero(cond):
        ok &= M_cols[:, k] if cond[k] > 0 else ~M_cols[:, k]
    return ok
