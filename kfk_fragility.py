"""Messprotokoll der Demo: wie viele Wege gelten nach einem Neu-Training nicht mehr, wie viele gelten in der Wahrheit nicht, wie viele Wege gibt es überhaupt.

Je **Zufalls-Seed** (ein Trainingsdatensatz mit n = 1 000 Containern, Klasse 1 = Verweilzeit länger als 4 Tage) werden die fünf Modellklassen trainiert (plus die Variante
„größter Index“ des exakt optimalen Baums), aus den ersten 25 als Klasse 1 eingestuften Testzeilen (**Instanzen**) je Politik und Sicherheitsabstand der billigste Weg gesucht und
gegen elf **Neu-Trainings** geprüft: R1 (10 % mehr Daten, frische Ziehung), R2 bis R11 (gleiche Menge, zehn frische Ziehungen). Gilt der Weg dort noch (Logit <= -1e-6)? Gilt er in
der Wahrheit des Generators (wahre Wahrscheinlichkeit der Klasse 1 am geänderten Punkt unter 0,5, Eingriffs-Lesart, eine Annahme dieser Demo)?
Die Ergebnisse stehen eingefroren in `kfk_fragilitaet.json` (erzeugt von `tools/kfk_make_fragilitaet.py`); die Live-App rechnet nur den einen Weg.
"""

import numpy as np

import kfk_generator as G
import kfk_models as M
import kfk_recourse as R

THR = 4                 # Klasse 1: Verweilzeit länger als 4 Tage
N_TRAIN = 1000
N_TEST = 400
N_INST = 25
N_DRAWS = 11            # R1 (+10 % Daten) und R2 bis R11 (gleiche Menge)
GROUP_SIZE = 12         # Zufalls-Seeds je Gruppe, fünf Gruppen
N_GROUPS = 5
EPS = R.EPS
KEYS = M.MODELS + ("opt_g",)   # opt_g: exakt optimaler Baum mit der zweiten Gleichstandsregel (größter Index)
CELLS = [(p, m) for p in R.POLICIES for m in R.MARGINS]


def cell_key(pol, mg):
    return f"{pol}|{mg:.1f}"


def stream_seeds(i, basis):
    """Zufalls-Seeds der Datenziehungen zu Zufalls-Seed i: Training, R1 bis R11, Test (getrennte Bereiche, nie doppelt)."""
    return {"train": basis + i, "r": [basis + 5000 + 1000 * r + i for r in range(N_DRAWS)], "test": basis + 100000 + i}


def _model(key, seed):
    return M.Model("opt", seed, "groesster") if key == "opt_g" else M.Model(key, seed)


def labels(y, thr=THR):
    return (np.exp(y) > thr).astype(int)


def instances_for(model, Xt, n_inst=N_INST):
    """Die ersten n_inst Testzeilen, die das Modell als Klasse 1 einstuft (Logit > 0), als Ebenen-Tupel."""
    pos = np.flatnonzero(model.logit(Xt) > 0)[:n_inst]
    return [R.levels(Xt[j]) for j in pos]


def way_record(lev, act, mg, sc0, scs, sd_v, thr=THR):
    """Ein Weg (billigster, kleinster Kandidaten-Index bei Gleichstand) und seine Prüfungen; None-Kosten heißt: kein Weg."""
    bf = R.cheapest(lev, act, mg, sc0, sd_v)
    if bf is None:
        return None
    cost, i = bf
    near = R.near_candidates(lev, act, mg, sc0, sd_v, cost)
    return {"cost": round(cost, 6), "inv": [bool(sc[i] > -EPS) for sc in scs], "wahr": not R.truth_valid(i, thr),
            "rausch": bool(R.CG[i] != lev[5]), "mehrweg": R.number_of_ways(lev, near) > 1}


def run_seed(i, basis, keys=KEYS, n_inst=N_INST, n_train=N_TRAIN, n_test=N_TEST, thr=THR):
    """Alles zu einem Zufalls-Seed: Genauigkeit, Blätter und je Modellklasse und Zelle (Politik, Abstand) die Zählungen über die Instanzen."""
    sd = stream_seeds(i, basis)
    X, y = G.draw_history(n_train, sd["train"])
    Xr = R.to_raster(X)
    lab = labels(y, thr)
    sd_v = R.sd_vorlauf(Xr)
    Xt, yt = G.draw_history(n_test, sd["test"])
    Xt = R.to_raster(Xt)
    lt = labels(yt, thr)
    draws = []
    for r, s in enumerate(sd["r"]):
        n = int(n_train * 1.1) if r == 0 else n_train
        Xn, yn = G.draw_history(n, s)
        draws.append((R.to_raster(Xn), labels(yn, thr), s))
    out = {"i": i, "acc": {}, "leaves": {}, "cells": {}}
    for key in keys:
        m0 = _model(key, sd["train"]).fit(Xr, lab)
        sc0 = m0.cand_scores()
        scs = [_model(key, s).fit(Xk, lk).cand_scores() for Xk, lk, s in draws]
        out["acc"][key] = round(float(np.mean((m0.logit(Xt) > 0) == (lt == 1))), 4)
        if m0.n_leaves() is not None:
            out["leaves"][key] = m0.n_leaves()
        levs = instances_for(m0, Xt, n_inst)
        cells = {}
        for pol, act in R.POLICIES.items():
            for mg in R.MARGINS:
                recs = [way_record(l0, act, mg, sc0, scs, sd_v, thr) for l0 in levs]
                fe = [r for r in recs if r is not None]
                cells[cell_key(pol, mg)] = {
                    "n": len(recs), "noway": len(recs) - len(fe),
                    "inv": [int(sum(r["inv"][d] for r in fe)) for d in range(N_DRAWS)],
                    "wahr": int(sum(r["wahr"] for r in fe)), "rausch": int(sum(r["rausch"] for r in fe)), "mehrweg": int(sum(r["mehrweg"] for r in fe)),
                    "kosten": [round(r["cost"], 3) for r in fe],
                }
        out["cells"][key] = cells
    return out


# ---------------------------------------------------------------- „Gleich gute Wege“: scikit-learn-Baum, Schwelle, Tiefe, Zählweise
TAB_THRESHOLDS = (4, 10)
TAB_DEPTHS = (3, 4, 5)
TAB_TRAIN = 3000
TAB_TEST = 600
TAB_INST = 60


def way_counts_seed(i, basis, n_train=TAB_TRAIN, n_test=TAB_TEST, n_inst=TAB_INST):
    """Zu einem Zufalls-Seed: je Schwelle und Tiefe die Zahl der Instanzen mit mehr als einem Blatt bzw. mehr als einer Menge geänderter Merkmale innerhalb von 10 % der
    Mindestkosten (Politik A, Abstand 0, Weg nur auf Rasterpunkten) und die Zahl der Instanzen mit Weg. Blatt = Blatt des Baums, in dem der Rasterpunkt liegt."""
    from sklearn.tree import DecisionTreeClassifier
    X, y = G.draw_history(n_train, basis + i)
    Xr = R.to_raster(X)
    sd_v = R.sd_vorlauf(Xr)
    Xt, _ = G.draw_history(n_test, basis + 100000 + i)
    Xt = R.to_raster(Xt)
    out = {"i": i}
    act = R.POLICIES["A"]
    for thr in TAB_THRESHOLDS:
        lab = labels(y, thr)
        for depth in TAB_DEPTHS:
            tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=30, random_state=basis + i).fit(R.design(Xr), lab)
            sc = M.clip_logit(tree.predict_proba(R.DC)[:, 1])
            leaf_c = tree.apply(R.DC)
            pos = np.flatnonzero(M.clip_logit(tree.predict_proba(R.design(Xt))[:, 1]) > 0)[:n_inst]
            n = mehr_blatt = mehr_menge = 0
            for j in pos:
                lev = R.levels(Xt[j])
                bf = R.cheapest(lev, act, 0.0, sc, sd_v)
                if bf is None:
                    continue
                near = R.near_candidates(lev, act, 0.0, sc, sd_v, bf[0])
                n += 1
                mehr_blatt += int(len(np.unique(leaf_c[near])) > 1)
                mehr_menge += int(R.number_of_ways(lev, near) > 1)
            out[f"{thr}|{depth}"] = {"n": n, "blatt": mehr_blatt, "menge": mehr_menge, "instanzen": len(pos)}
    return out
