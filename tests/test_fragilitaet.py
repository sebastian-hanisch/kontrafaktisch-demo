"""Die eingefrorene Messung (`kfk_fragilitaet.json`): Aufbau, innere Widersprüche (Eigenschaften je Zufalls-Seed) und Neu-Rechnen eines Zufalls-Seeds als Band."""

import importlib.util
import json
from pathlib import Path

import pytest

import kfk_evaluation as EV
import kfk_fragility as F
import kfk_models as M
import kfk_recourse as R

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def data():
    return EV.load_data()


def test_structure_and_provenance(data):
    meta = data["meta"]
    assert len(data["seeds"]) == 60 and len(EV.groups(data)) == 5 and all(len(g) == 12 for g in EV.groups(data))
    assert meta["schwelle_tage"] == 4 and meta["n_train"] == 1000 and meta["instanzen_je_seed"] == 25 and meta["neu_training"] == 11
    assert meta["quelle"] and meta["stand"] and meta["seeds"]["quelle"] and meta["gruppen"]["quelle"] and meta["tab"]["quelle"]
    assert len(data["tab"]) == 30 and [s["i"] for s in data["seeds"]] == list(range(60))
    for s in data["seeds"]:
        assert set(s["cells"]) == set(F.KEYS) and set(s["acc"]) == set(F.KEYS)
        for key in F.KEYS:
            assert set(s["cells"][key]) == {F.cell_key(p, m) for p, m in F.CELLS}


def test_counts_are_consistent_in_every_cell(data):
    for s in data["seeds"]:
        for key in F.KEYS:
            for ck, c in s["cells"][key].items():
                way = c["n"] - c["noway"]
                assert 0 <= c["n"] <= 25 and 0 <= c["noway"] <= c["n"] and len(c["inv"]) == 11 and len(c["kosten"]) == way
                assert all(0 <= v <= way for v in c["inv"]) and 0 <= c["wahr"] <= way and 0 <= c["rausch"] <= way and 0 <= c["mehrweg"] <= way
                assert all(k > 0 for k in c["kosten"])


def test_group_sizes_and_instances_per_group(data):
    for key in M.MODELS:
        st = EV.group_stats(data, key, "A", 0.0)
        assert all(250 <= s["n"] <= 300 for s in st)


def test_policy_property_narrower_policy_never_has_fewer_instances_without_way(data):
    """Je Zufalls-Seed und Modellklasse: kein Weg bei A <= B <= C und Politik A hat immer einen Weg (alle Merkmale veränderbar, Abstand bis 1,0)."""
    for s in data["seeds"]:
        for key in F.KEYS:
            for mg in R.MARGINS:
                nw = [s["cells"][key][F.cell_key(p, mg)]["noway"] for p in "ABC"]
                assert nw[0] <= nw[1] <= nw[2], (s["i"], key, mg, nw)
    assert all(s["cells"][k]["A|0.0"]["noway"] == 0 for s in data["seeds"] for k in F.KEYS)


def test_cost_property_larger_margin_never_costs_less(data):
    """Je Zufalls-Seed und Modellklasse (Politik A, alle Instanzen haben einen Weg): die Summe der Kosten wächst nicht mit dem Abstand fallend."""
    for s in data["seeds"]:
        for key in F.KEYS:
            tot = [sum(s["cells"][key][F.cell_key("A", mg)]["kosten"]) for mg in R.MARGINS]
            assert tot[0] <= tot[1] + 1e-6 <= tot[2] + 2e-6, (s["i"], key, tot)


def test_margin_makes_the_regression_way_valid_in_every_group(data):
    """Aussage mit Voraussetzungen: Abstand 1,0 senkt bei der logistischen Regression den Anteil in jeder Gruppe auf höchstens 3 % (Politik A, R1); bei den anderen vier Klassen nicht in jeder."""
    assert EV.band(data, "reg", "A", 1.0, "inv")["hi"] <= 3.0
    assert sum(EV.band(data, k, "A", 1.0, "inv")["hi"] > 3.0 for k in ("baum", "boost", "netz", "opt")) == 4


def test_accuracy_is_plausible_and_the_tree_family_is_not_better_than_regression(data):
    acc = {k: EV.accuracy(data, k) for k in F.KEYS}
    assert all(0.65 < a < 0.85 for a in acc.values())
    assert acc["reg"] >= max(acc["baum"], acc["opt"])


def test_recomputing_a_seed_reproduces_the_stored_counts_as_a_band(data):
    """Neu-Rechnen als Band: Zufalls-Seed 0, logistische Regression und Baum. Gleitkomma-Letztstellen dürfen einzelne Instanzen kippen (Toleranz: 2 Instanzen oder 10 %)."""
    basis = data["meta"]["seeds"]["basis"]
    new = F.run_seed(0, basis, keys=("reg", "baum"))
    for key in ("reg", "baum"):
        for ck, cell in new["cells"][key].items():
            old = data["seeds"][0]["cells"][key][ck]
            assert cell["n"] == old["n"]
            tol = max(2, 0.1 * cell["n"])
            for field in ("noway", "wahr", "rausch", "mehrweg"):
                assert abs(cell[field] - old[field]) <= tol, (key, ck, field)
            assert all(abs(a - b) <= tol for a, b in zip(cell["inv"], old["inv"])), (key, ck)


def test_the_tool_check_passes_on_the_frozen_file():
    spec = importlib.util.spec_from_file_location("kfk_make", ROOT / "tools" / "kfk_make_fragilitaet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.check(str(ROOT / "kfk_fragilitaet.json")) == 0


def test_json_is_valid_utf8_and_small_enough():
    p = ROOT / "kfk_fragilitaet.json"
    assert p.stat().st_size < 1_500_000
    with open(p, encoding="utf-8") as f:
        assert isinstance(json.load(f), dict)
