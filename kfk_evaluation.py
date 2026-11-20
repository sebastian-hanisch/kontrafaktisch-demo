"""Kennzahlen und Verdikte. Die eingefrorene Messung (`kfk_fragilitaet.json`) wird nur ausgewertet, nie neu gerechnet; der einzelne Weg der Live-Ansicht wird gerechnet.

Zähleinheiten: **Instanzen** (je Zufalls-Seed bis zu 25 als Klasse 1 eingestufte Testzeilen), **Zufalls-Seeds** (ein Trainingsdatensatz mit n = 1 000), **Gruppen** (je 12
getrennte Zufalls-Seeds, also bis zu 300 Instanzen); jede Aussage über Anteile steht als Spanne über die fünf Gruppen und als Anteil über alle Instanzen, nie als Punktzahl aus
einer festen Ziehung. Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`."""

import json
import os
from dataclasses import dataclass
from statistics import median

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

import kfk_constants as C
import kfk_fragility as F
import kfk_generator as G
import kfk_models as M
import kfk_recourse as R

_DATA = {}


def data_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), C.DATA_FILE)


def load_data(path=None):
    """Eingefrorene Messung laden (einmal je Prozess)."""
    path = path or data_path()
    if path not in _DATA:
        with open(path, encoding="utf-8") as f:
            _DATA[path] = json.load(f)
    return _DATA[path]


def fmt(x, d=0):
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{d}f}".replace(".", ",")


def ntxt(n):
    """Ganzzahl mit Leerzeichen als Tausendertrenner (1 500)."""
    return f"{int(n):,}".replace(",", " ")


def pct(num, den):
    return 100.0 * num / den if den else float("nan")


def groups(data):
    """Die Zufalls-Seeds in Gruppen zu je GROUP_SIZE (in der Reihenfolge der Messung)."""
    seeds = data["seeds"]
    g = data["meta"]["gruppen"]["groesse"]
    return [seeds[k * g:(k + 1) * g] for k in range(len(seeds) // g)]


def _inv_count(cell, retrain):
    inv = cell["inv"]
    if retrain == "R1":
        return inv[0]
    if retrain == "R2":
        return inv[1]
    if retrain == "R3":
        return inv[2]
    if retrain == "R2-R11":
        return sum(inv[1:]) / len(inv[1:])          # Mittel der zehn Ziehungen gleicher Menge
    raise ValueError(retrain)


def group_stats(data, key, pol, mg, retrain="R1"):
    """Je Gruppe: Zahl der Instanzen, ohne Weg, mit Weg, ungültig nach dem Neu-Training, in der Wahrheit, über das Gewicht, mit mehr als einem Weg, Kosten (Liste)."""
    out = []
    ck = F.cell_key(pol, mg)
    for g in groups(data):
        s = dict(n=0, noway=0, inv=0.0, wahr=0, rausch=0, mehrweg=0, kosten=[])
        for seed in g:
            c = seed["cells"][key][ck]
            s["n"] += c["n"]
            s["noway"] += c["noway"]
            s["inv"] += _inv_count(c, retrain)
            s["wahr"] += c["wahr"]
            s["rausch"] += c["rausch"]
            s["mehrweg"] += c["mehrweg"]
            s["kosten"] += c["kosten"]
        s["withway"] = s["n"] - s["noway"]
        out.append(s)
    return out


def band(data, key, pol, mg, field, retrain="R1"):
    """Spanne eines Anteils (in Prozent) über die Gruppen und der Anteil über alle Instanzen.
    field: 'noway' (von allen Instanzen), 'inv', 'wahr', 'rausch', 'mehrweg' (von den Instanzen mit Weg), 'kosten' (Median)."""
    st_ = group_stats(data, key, pol, mg, retrain)
    if field == "kosten":
        per = [float(median(s["kosten"])) if s["kosten"] else float("nan") for s in st_]
        allk = [k for s in st_ for k in s["kosten"]]
        return {"per_group": per, "lo": min(per), "hi": max(per), "all": float(median(allk)) if allk else float("nan"), "n": len(allk)}
    if field == "noway":
        per = [pct(s["noway"], s["n"]) for s in st_]
        tot = pct(sum(s["noway"] for s in st_), sum(s["n"] for s in st_))
        return {"per_group": per, "lo": min(per), "hi": max(per), "all": tot, "n": sum(s["n"] for s in st_)}
    per = [pct(s[field], s["withway"]) for s in st_]
    tot = pct(sum(s[field] for s in st_), sum(s["withway"] for s in st_))
    return {"per_group": per, "lo": min(per), "hi": max(per), "all": tot, "n": sum(s["withway"] for s in st_)}


def draw_spread(data, key, pol, mg):
    """Anteil ungültig je einzelne Ziehung R2 bis R11 über alle Instanzen (kleinster und größter Wert): die Streuung des Neu-Trainings selbst."""
    ck = F.cell_key(pol, mg)
    vals = []
    for d in range(1, data["meta"]["neu_training"]):
        inv = sum(s["cells"][key][ck]["inv"][d] for s in data["seeds"])
        way = sum(s["cells"][key][ck]["n"] - s["cells"][key][ck]["noway"] for s in data["seeds"])
        vals.append(pct(inv, way))
    return min(vals), max(vals)


def accuracy(data, key):
    """Genauigkeit auf den Testzeilen (Mittel über die Zufalls-Seeds)."""
    return float(np.mean([s["acc"][key] for s in data["seeds"]]))


def tab_band(data, thr, depth, how):
    """Anteil der Instanzen mit mehr als einem Weg im Tab „Gleich gute Wege“: Spanne über die Gruppen zu je 6 Zufalls-Seeds und Anteil über alle Instanzen."""
    tab = data["tab"]
    gs = 6
    per, num_all, den_all = [], 0, 0
    for k in range(len(tab) // gs):
        part = tab[k * gs:(k + 1) * gs]
        num = sum(t[f"{thr}|{depth}"]["blatt" if how == "blaetter" else "menge"] for t in part)
        den = sum(t[f"{thr}|{depth}"]["n"] for t in part)
        per.append(pct(num, den))
        num_all += num
        den_all += den
    return {"per_group": per, "lo": min(per), "hi": max(per), "all": pct(num_all, den_all), "n": den_all}


# ---------------------------------------------------------------- Live: ein Weg
@dataclass
class Fit:
    model: object
    Xr: np.ndarray
    sd_v: float
    Xt: np.ndarray
    levs: list
    accuracy: float
    seed: int
    model_key: str


def fit_live(model_key, seed):
    """Ein Training (n = 1 000) und die ersten 25 als Klasse 1 eingestuften Testzeilen (Instanzen) zu einem Zufalls-Seed."""
    X, y = G.draw_history(C.N_TRAIN, seed)
    Xr = R.to_raster(X)
    lab = F.labels(y)
    m = F._model(model_key, seed).fit(Xr, lab)
    Xt, yt = G.draw_history(C.N_TEST, seed + 100000)
    Xt = R.to_raster(Xt)
    acc = float(np.mean((m.logit(Xt) > 0) == (F.labels(yt) == 1)))
    return Fit(m, Xr, R.sd_vorlauf(Xr), Xt, F.instances_for(m, Xt), acc, seed, model_key)


@dataclass
class Way:
    n_instances: int
    instance: int
    lev0: tuple
    lev1: tuple | None
    cost: float | None
    changed: list
    logit0: float
    logit1: float | None
    margin: float
    policy: str
    n_ways_near: int
    ways_near: list
    n_ties: int
    p_true0: float
    p_true1: float | None
    enum_cost: float | None
    other_cost: float | None      # Programm (Regression) oder Blattquader (Bäume), None bei Boosting und Netz
    other_name: str | None
    agree: bool | None


def way_for(fit, instance, policy, margin):
    """Der billigste Weg der gewählten Instanz (1-basiert; über die Zahl der Instanzen hinaus wird begrenzt) samt zweitem Rechenweg."""
    n = len(fit.levs)
    k = min(max(instance, 1), n) - 1
    lev0 = fit.levs[k]
    act = R.POLICIES[policy]
    sc = fit.model.cand_scores()
    i0 = R.index_of(lev0)
    logit0 = float(sc[i0])
    mu0, s0 = R.MU_C[i0], R.S_C[i0]
    p0 = float(G.p_longer(mu0, s0, F.THR))
    bf = R.cheapest(lev0, act, margin, sc, fit.sd_v)
    other = name = None
    if fit.model_key == "reg":
        w, b = fit.model.linear_weights()
        mp = R.milp_way(lev0, act, margin, w, b, fit.sd_v)
        other, name = (None if mp is None else mp[0]), "ganzzahliges Programm (HiGHS)"
    elif fit.model_key in ("baum", "opt", "opt_g"):
        bx = R.box_way(lev0, act, margin, fit.model.boxes(), fit.sd_v)
        other, name = (None if bx is None else bx[0]), "Blattquader (ohne Vorhersage)"
    if bf is None:
        agree = None if name is None else other is None
        return Way(n, k + 1, lev0, None, None, [], logit0, None, margin, policy, 0, [], 0, p0, None, None, other, name, agree)
    cost, i = bf
    near = R.near_candidates(lev0, act, margin, sc, fit.sd_v, cost)
    sigs = sorted({int(s) for s in R.change_signature(lev0, near)})
    ways = [[R.FEATURES[j] for j in range(6) if s >> j & 1] for s in sigs]
    ok = allowed_ties(lev0, act, margin, sc, fit.sd_v, cost)
    agree = None if name is None else (other is not None and abs(other - cost) < 1e-6)
    return Way(n, k + 1, lev0, tuple(int(v) for v in R.LEVELS[i]), cost, R.changed_features(lev0, i), logit0, float(sc[i]), margin, policy, len(sigs), ways, ok,
               p0, float(G.p_longer(R.MU_C[i], R.S_C[i], F.THR)), cost, other, name, agree)


def allowed_ties(lev, act, margin, scores, sd_v, cost):
    """Zahl der Kandidaten mit genau den Mindestkosten (Gleichstand; gewählt wird der kleinste Kandidaten-Index)."""
    ok = R.allowed(lev, act) & (scores <= -margin - R.EPS)
    c = np.where(ok, R.cost_cands(lev, sd_v), np.inf)
    return int(np.sum(c <= cost + 1e-9))


def level_text(lev):
    """Merkmale einer Ebenen-Tupel als Anzeigewerte."""
    a, k, z, w, v, g = lev
    art = ("Import", "Export", "Umschlag", "Leer")[a]
    return {"Art": art, "Kunde": f"Kunde {k + 1}", "Zoll": "ja" if z else "nein", "Wochenende": "ja" if w else "nein",
            "Vorlauf": f"{fmt(v * 0.5, 1)} Tage", "Gewicht": fmt((g - 6) * 0.5, 1)}


# ---------------------------------------------------------------- geschlossene Form (Handbeispiel Teil 5): warum der billigste Weg auf der Grenze liegt
def p_invalid_model(a, s):
    """P(Verschiebung > Spielraum): Spielraum U(0, a) über der Grenze, die Grenze verschiebt sich um N(0, s^2); P = (s/a) (t Phi(-t) + phi(0) - phi(t)), t = a/s."""
    t = a / s
    return (s / a) * (t * norm.cdf(-t) + norm.pdf(0) - norm.pdf(t))


def p_invalid_numeric(a, s):
    num, _ = quad(lambda x: norm.cdf(-x / s), 0, a)
    return num / a


# ---------------------------------------------------------------- Verdikte
def _range(b, d=0):
    """Spanne über die Gruppen als Text; sind beide Grenzen gleich (nach dem Runden), steht nur ein Wert."""
    lo, hi = fmt(b["lo"], d), fmt(b["hi"], d)
    return f"{lo} %" if lo == hi else f"{lo} bis {hi} %"


def verdict_way(w, bands):
    """Experiment 1: der Weg der Instanz. `bands`: eingefrorene Kennzahlen der Zelle {'kosten': band, 'mehrweg': band, 'noway': band}."""
    if w.cost is None:
        return "warning", (f"Für diese Instanz gibt es mit der Politik {R.POLICY_NAMES[w.policy]} und Abstand {fmt(w.margin, 1)} **keinen Weg**: kein erlaubter Rasterpunkt bringt das Logit "
                           f"unter {fmt(-w.margin, 1)}. Über alle Instanzen der Messung gilt das für {_range(bands['noway'])} (fünf Gruppen).")
    ch = " und ".join(w.changed) if w.changed else "nichts"
    txt = (f"Der billigste Weg ändert **{ch}** und kostet **{fmt(w.cost, 2)}**: das Logit sinkt von {fmt(w.logit0, 2)} auf {fmt(w.logit1, 2)}. Innerhalb von 10 % der Mindestkosten "
           f"gibt es {w.n_ways_near} Menge(n) geänderter Merkmale. Über alle Instanzen der Messung: Kosten im Median {fmt(bands['kosten']['all'], 2)}; bei {_range(bands['mehrweg'])} "
           f"der Instanzen mehr als eine Menge geänderter Merkmale innerhalb 10 % (fünf Gruppen zu je 12 Zufalls-Seeds).")
    return "info", txt


def verdict_policy(data, pol, mg):
    """Experiment 2: Anteil ohne Weg je Politik; Spanne über alle fünf Modellklassen und die fünf Gruppen."""
    def rng(p):
        los, his = [], []
        for key in M.MODELS:
            b = band(data, key, p, mg, "noway")
            los.append(b["lo"])
            his.append(b["hi"])
        return min(los), max(his)

    parts = []
    for p in ("C", "B", "A"):
        lo, hi = rng(p)
        parts.append(f"{R.POLICY_NAMES[p]}: " + ("jede Instanz hat einen Weg" if hi == 0 else "für " + (fmt(lo) + " bis " + fmt(hi) if fmt(lo) != fmt(hi) else fmt(lo)) + " % der Instanzen kein Weg"))
    kind = "warning" if rng(pol)[1] > 0 else "success"
    return kind, ("Bei Abstand " + fmt(mg, 1) + " (alle fünf Modellklassen, fünf Gruppen zu je 12 Zufalls-Seeds): " + "; ".join(parts) + ". Ob es einen Weg gibt, hängt hier stärker an der Liste der "
                  "veränderbaren Merkmale als am Modell; Politik und Kosten sind Annahmen des Verfassers.")


def verdict_retrain(data, model, pol, mg, retrain):
    """Experiment 3: Anteil der Wege, die nach dem Neu-Training nicht mehr gelten, für die gewählte Klasse und für alle fünf."""
    short = {"R1": "R1 (10 % mehr Daten)", "R2": "R2 (gleiche Menge)", "R3": "R3 (gleiche Menge)", "R2-R11": "den zehn Ziehungen gleicher Menge (Mittel)"}[retrain]
    b = band(data, model, pol, mg, "inv", retrain)
    others = ", ".join(f"{M.MODEL_NAMES[k].split(' (')[0]} {_range(band(data, k, pol, mg, 'inv', retrain))}" for k in M.MODELS if k != model)
    txt = (f"{M.MODEL_NAMES[model]}, Politik {pol}, Abstand {fmt(mg, 1)}: **{_range(b)}** der Wege gelten nach {short} nicht mehr (fünf Gruppen zu je 12 Zufalls-Seeds, bis zu 300 Instanzen je Gruppe; "
           f"{fmt(b['all'], 1)} % über alle {ntxt(b['n'])} Instanzen mit Weg). Die anderen vier Klassen: {others}.")
    kind = "success" if b["hi"] <= 3 else "warning"
    return kind, txt


def verdict_margin(data, pol, retrain):
    """Negativbefund zu Experiment 3: der Sicherheitsabstand 1,0 gegen Abstand 0, je Modellklasse."""
    parts = []
    for key in M.MODELS:
        b0, b1 = band(data, key, pol, 0.0, "inv", retrain), band(data, key, pol, 1.0, "inv", retrain)
        parts.append(f"{M.MODEL_NAMES[key].split(' (')[0]} {_range(b0)} auf {_range(b1)}")
    heal = [M.MODEL_NAMES[k].split(" (")[0] for k in M.MODELS if band(data, k, pol, 1.0, "inv", retrain)["hi"] <= 3.0]
    tail = (f" Auf höchstens 3 % in allen Gruppen sinkt nur: {', '.join(heal)}." if heal else " In keiner Klasse sinkt der Anteil in allen Gruppen auf höchstens 3 %.")
    return "info", f"Sicherheitsabstand 0 gegen 1,0 (Politik {pol}, ungültig, Spanne über fünf Gruppen): " + "; ".join(parts) + "." + tail


def verdict_truth(data, model, pol, mg):
    """Zweite Anzeige in Experiment 3: Wahrheit des Generators und Wege über das Merkmal ohne Wirkung."""
    bw, br = band(data, model, pol, mg, "wahr"), band(data, model, pol, mg, "rausch")
    others = ", ".join(f"{M.MODEL_NAMES[k].split(' (')[0]} {_range(band(data, k, pol, mg, 'rausch'), 1)}" for k in M.MODELS if k != model)
    return "info", (f"{M.MODEL_NAMES[model]}, Politik {pol}, Abstand {fmt(mg, 1)}: in der Wahrheit des Generators (Eingriffs-Lesart, eine Annahme dieser Demo) gelten {_range(bw)} der Wege nicht "
                    f"(wahre Wahrscheinlichkeit der Klasse 1 am geänderten Punkt mindestens 0,5; fünf Gruppen). Der billigste Weg ändert in {_range(br, 1)} der Instanzen auch das Gewicht (allein oder neben anderen Merkmalen), ein Merkmal ohne "
                    f"Wirkung ({others}).")


def verdict_tab(data, thr, depth, how):
    """Tab „Gleich gute Wege“: Anteil der Instanzen mit mehr als einem Weg, beide Zählweisen nebeneinander."""
    bl, me = tab_band(data, thr, depth, "blaetter"), tab_band(data, thr, depth, "mengen")
    mine = bl if how == "blaetter" else me
    other = me if how == "blaetter" else bl
    return "info", (f"Schwelle {thr} Tage, Tiefe {depth}, Zählung {C.COUNT_NAMES[how]}: **{_range(mine)}** der Instanzen haben mehr als einen Weg innerhalb 10 % (fünf Gruppen zu je 6 Zufalls-Seeds, "
                    f"{fmt(mine['all'], 1)} % über alle {ntxt(mine['n'])} Instanzen). Dieselben Bäume und Instanzen, Zählung {C.COUNT_NAMES['mengen' if how == 'blaetter' else 'blaetter']}: {_range(other)}. "
                    f"Die Zahl hängt also an der Zählweise und an der Schwelle, sie ist keine Eigenschaft von Gegenfaktischen.")
