"""Verdikte an von Hand gesetzten Werten: jede Zahl im Text und jede Schwelle der Art steht hier. Die Messung wird durch ein kleines Datenobjekt mit fünf Gruppen zu je einem Zufalls-Seed ersetzt."""

import pytest

import kfk_evaluation as ev
import kfk_fragility as F
import kfk_models as M
import kfk_recourse as R


def cell(n=20, noway=0, inv=(2,) * 11, wahr=3, rausch=1, mehrweg=4, kosten=None):
    way = n - noway
    return {"n": n, "noway": noway, "inv": list(inv), "wahr": wahr, "rausch": rausch, "mehrweg": mehrweg, "kosten": kosten if kosten is not None else [1.0] * way}


def data_with(per_seed_cells):
    """per_seed_cells(i, key, pol, mg) -> Zelle; fünf Zufalls-Seeds, jeder eine Gruppe."""
    seeds = []
    for i in range(5):
        seeds.append({"i": i, "acc": {k: 0.75 for k in F.KEYS},
                      "cells": {k: {F.cell_key(p, m): per_seed_cells(i, k, p, m) for p, m in F.CELLS} for k in F.KEYS}})
    tab = [{"i": i, **{f"{t}|{d}": {"n": 10, "blatt": 5, "menge": 2, "instanzen": 12} for t in (4, 10) for d in (3, 4, 5)}} for i in range(12)]
    return {"meta": {"gruppen": {"groesse": 1}, "neu_training": 11}, "seeds": seeds, "tab": tab}


def test_band_is_the_spread_over_groups_and_the_share_over_all_instances():
    # Gruppe i hat i + 1 von 20 Wegen ungültig nach R1: 5, 10, 15, 20, 25 %
    d = data_with(lambda i, k, p, m: cell(inv=(i + 1,) + (2,) * 10))
    b = ev.band(d, "reg", "A", 0.0, "inv", "R1")
    assert b["per_group"] == [5.0, 10.0, 15.0, 20.0, 25.0] and (b["lo"], b["hi"]) == (5.0, 25.0) and b["all"] == 15.0 and b["n"] == 100


def test_retrain_selection_r2_to_r11_is_the_mean_of_the_ten_draws():
    d = data_with(lambda i, k, p, m: cell(inv=(0, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4)))
    assert ev.band(d, "reg", "A", 0.0, "inv", "R2")["all"] == 20.0 and ev.band(d, "reg", "A", 0.0, "inv", "R3")["all"] == 40.0
    assert ev.band(d, "reg", "A", 0.0, "inv", "R2-R11")["all"] == pytest.approx((4 * 9 + 8) / 10 / 20 * 100)        # Mittel 4,4 von 20 Wegen
    assert ev.draw_spread(d, "reg", "A", 0.0) == (20.0, 40.0)


def test_noway_is_a_share_of_all_instances_and_costs_use_the_median():
    d = data_with(lambda i, k, p, m: cell(n=20, noway=5, kosten=[0.5] * 5 + [1.0] * 5 + [3.0] * 5))
    b = ev.band(d, "reg", "C", 0.0, "noway")
    assert b["all"] == 25.0 and b["n"] == 100
    assert ev.band(d, "reg", "C", 0.0, "kosten")["all"] == 1.0


def test_verdict_retrain_names_the_class_the_range_and_the_other_classes():
    d = data_with(lambda i, k, p, m: cell(inv=((i + 1) if k == "reg" else 10,) + (2,) * 10))
    kind, text = ev.verdict_retrain(d, "reg", "A", 0.0, "R1")
    assert kind == "warning"                                                    # Obergrenze 25 % > 3 %
    assert "Logistische Regression, Politik A, Abstand 0,0: **5 bis 25 %**" in text and "15,0 % über alle 100 Instanzen" in text
    assert "Baum 50 %" in text and "Boosting 50 %" in text
    assert ev.verdict_retrain(data_with(lambda i, k, p, m: cell(inv=(0,) * 11)), "reg", "A", 1.0, "R1")[0] == "success"    # höchstens 3 %: Obergrenze inklusive
    assert ev.verdict_retrain(data_with(lambda i, k, p, m: cell(n=100, inv=(3,) + (0,) * 10)), "reg", "A", 1.0, "R1")[0] == "success"          # genau 3 %
    assert ev.verdict_retrain(data_with(lambda i, k, p, m: cell(n=100, inv=(4,) + (0,) * 10)), "reg", "A", 1.0, "R1")[0] == "warning"          # 4 %


def test_verdict_margin_lists_the_classes_that_drop_to_at_most_three_percent():
    d = data_with(lambda i, k, p, m: cell(n=100, inv=((0 if m == 1.0 else 30) if k == "reg" else (10 if m == 1.0 else 30),) + (0,) * 10))
    kind, text = ev.verdict_margin(d, "A", "R1")
    assert kind == "info" and "Logistische Regression 30 % auf 0 %" in text and "Baum 30 % auf 10 %" in text
    assert text.endswith("Auf höchstens 3 % in allen Gruppen sinkt nur: Logistische Regression.")
    none = ev.verdict_margin(data_with(lambda i, k, p, m: cell(n=100, inv=(30,) + (0,) * 10)), "A", "R1")[1]
    assert "In keiner Klasse sinkt der Anteil in allen Gruppen auf höchstens 3 %." in none


def test_verdict_policy_ranges_over_all_five_classes():
    d = data_with(lambda i, k, p, m: cell(n=20, noway={"A": 0, "B": 4, "C": 10}[p] + (2 if k == "boost" and p != "A" else 0)))
    kind, text = ev.verdict_policy(d, "C", 0.0)
    assert kind == "warning"
    assert "C: nur Vorlauf: für 50 bis 60 % der Instanzen kein Weg" in text and "B: Vorlauf und Zoll: für 20 bis 30 % der Instanzen kein Weg" in text
    assert "A: alles veränderbar: jede Instanz hat einen Weg" in text
    assert ev.verdict_policy(d, "A", 0.0)[0] == "success"                      # Politik A: kein Instanz ohne Weg


def test_verdict_truth_names_both_shares():
    d = data_with(lambda i, k, p, m: cell(n=100, wahr=20 + i, rausch=1 if k == "reg" else 7))
    kind, text = ev.verdict_truth(d, "reg", "A", 0.0)
    assert kind == "info" and "gelten 20 bis 24 % der Wege nicht" in text and "auch das Gewicht (allein oder neben anderen Merkmalen), ein Merkmal ohne Wirkung" in text
    assert "in 1,0 % der Instanzen" in text and "Boosting 7,0 %" in text


def test_verdict_tab_names_both_counting_rules():
    d = data_with(lambda i, k, p, m: cell())
    kind, text = ev.verdict_tab(d, 10, 5, "blaetter")
    assert kind == "info" and "Schwelle 10 Tage, Tiefe 5, Zählung nach Blättern des Baums: **50 %**" in text
    assert "nach Mengen geänderter Merkmale: 20 %" in text and "120 Instanzen" in text
    assert "Zählung nach Mengen geänderter Merkmale: **20 %**" in ev.verdict_tab(d, 10, 5, "mengen")[1]


def _way(cost=1.0, changed=("Zoll",), n_near=2):
    return ev.Way(25, 3, (0, 0, 1, 0, 0, 6), (0, 0, 0, 0, 0, 6), cost, list(changed), 1.234, -0.056, 0.0, "A", n_near, [], 1, 0.7, 0.3, cost, cost, "x", True)


def test_verdict_way_with_and_without_a_way():
    bands = {"kosten": {"all": 1.0}, "mehrweg": {"lo": 14.0, "hi": 19.0}, "noway": {"lo": 41.0, "hi": 55.0}}
    kind, text = ev.verdict_way(_way(), bands)
    assert kind == "info" and "ändert **Zoll** und kostet **1,00**" in text and "von 1,23 auf -0,06" in text and "2 Menge(n)" in text
    assert "Kosten im Median 1,00" in text and "14 bis 19 %" in text
    kind, text = ev.verdict_way(ev.Way(25, 3, (0, 0, 1, 0, 0, 6), None, None, [], 1.234, None, 0.5, "C", 0, [], 0, 0.7, None, None, None, None, None), bands)
    assert kind == "warning" and "**keinen Weg**" in text and "41 bis 55 %" in text and R.POLICY_NAMES["C"] in text and "Abstand 0,5" in text


def test_every_model_has_a_short_name_and_the_verdicts_cover_all_five():
    assert set(M.MODELS) == {"reg", "baum", "boost", "netz", "opt"}
    d = data_with(lambda i, k, p, m: cell())
    text = ev.verdict_retrain(d, "opt", "A", 0.0, "R1")[1]
    assert all(name in text for name in ("Logistische Regression", "Baum", "Boosting", "Kleines Netz")) and text.startswith("Exakt optimaler Baum (Tiefe 3)")
