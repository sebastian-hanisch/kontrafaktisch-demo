"""Claims: jede Zahl der README steht hier gegen die eingefrorene Messung (über dieselben Auswertungsfunktionen wie die App) und kommt in der README vor.
Nicht im Test (Messung dieser README): die Spalte „Vormessung“ der Korrekturtabelle, die zweite Messung mit 60 weiteren Zufalls-Seeds und die Vormessung mit dem Kreditscore.
Live nachgerechnet (als Band): die Lage des Wegs zur Grenze und die Zahl der nicht konvergierten Netze (am Dateiende)."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import kfk_constants as C
import kfk_evaluation as EV
import kfk_fragility as F
import kfk_models as M
import kfk_recourse as R
import kfk_texts as T

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def data():
    return EV.load_data()


def rng(b, d=0):
    return EV._range(b, d)


def one(x, d=1):
    return EV.fmt(x, d)


# (Schlüssel, Genauigkeit, Kosten 0, Kosten 1,0) je Zeile der Haupttabelle, wie sie in der README steht
ROWS = ["reg", "baum", "boost", "netz", "opt", "opt_g"]


def _row(data, k):
    g = lambda f, mg=0.0, rr="R1": EV.band(data, k, "A", mg, f, rr)   # noqa: E731
    return [
        f"{EV.fmt(100 * EV.accuracy(data, k), 1)} %",
        f"{EV.fmt(g('kosten')['all'], 2)} / {EV.fmt(g('kosten', 1.0)['all'], 2)}",
        f"{rng(g('inv'))} ({one(g('inv')['all'])})",
        f"{rng(g('inv', 0.5))} ({one(g('inv', 0.5)['all'])})",
        f"{rng(g('inv', 1.0))} ({one(g('inv', 1.0)['all'])})",
        f"{rng(g('inv', 0.0, 'R2-R11'))} ({one(g('inv', 0.0, 'R2-R11')['all'])})",
        "",                                                                      # Wahrheit: Zahlen werden unten gesondert verglichen
        rng(g("rausch"), 1),
    ]


def _table_line(k):
    names = {"reg": "Logistische Regression", "baum": "Baum (Tiefe 4)", "boost": "Boosting", "netz": "Kleines Netz", "opt": "Exakt optimaler Baum",
             "opt_g": "Exakt optimaler Baum, zweite Regel (größter Index)"}
    for line in README.splitlines():
        if line.startswith(f"| {names[k]} |"):
            return [c.strip() for c in line.strip().strip("|").split("|")]
    raise AssertionError(k)


@pytest.mark.parametrize("k", ROWS)
def test_main_table_row_equals_the_measurement(data, k):
    got = _table_line(k)[1:]
    want = _row(data, k)
    # die Wahrheitsspalte steht als „a bis b / c %“: Vergleich der Zahlen
    for g, w in zip(got[:6], want[:6]):
        assert g == w, (k, g, w)
    wg = re.findall(r"\d+", got[6])
    ww = re.findall(r"\d+", f"{EV._range(EV.band(data, k, 'A', 0.0, 'wahr'))} / {EV._range(EV.band(data, k, 'A', 1.0, 'wahr'))}")
    assert wg == ww, (k, got[6])
    assert got[7] == want[7], (k, got[7], want[7])


def test_headline_ranges_of_the_prose(data):
    # (2) Abstand 1,0: nur die Regression in jeder Gruppe auf höchstens 3 %, andere Spannen wie genannt
    assert EV.band(data, "reg", "A", 1.0, "inv")["hi"] == 0
    assert all(EV.band(data, k, "A", 1.0, "inv")["hi"] > 3 for k in ("baum", "boost", "netz", "opt"))
    assert "Spannen 0 bis 24, 8 bis 13, 8 bis 10, 11 bis 34 %" in README
    # (1) Anteil über alle Instanzen mit Weg 18 bis 39 %
    alle = [EV.band(data, k, "A", 0.0, "inv")["all"] for k in M.MODELS]
    assert (round(min(alle)), round(max(alle))) == (18, 39) and "Anteil über alle Instanzen mit Weg 18 bis 39 %" in README
    # (3) exakt optimaler Baum: R1 15 bis 40, Mittel der zehn Ziehungen 15 bis 28; in der Wahrheit 30 bis 36 und 21 bis 27
    assert rng(EV.band(data, "opt", "A", 0.0, "inv")) == "15 bis 40 %" and rng(EV.band(data, "opt", "A", 0.0, "inv", "R2-R11")) == "15 bis 28 %"
    assert "R1 gibt je Gruppe 15 bis 40 %, die Mittel der zehn Ziehungen gleicher Menge 15 bis 28 %" in README
    assert "in der Wahrheit 30 bis 36 % ungültig, mit Abstand 1,0 noch 21 bis 27 %" in README
    # (4) zweite Regel
    assert rng(EV.band(data, "opt_g", "A", 0.0, "inv")) == "15 bis 42 %" and "(15 bis 40 gegen 15 bis 42 %)" in README


def test_no_way_by_policy_over_the_five_classes(data):
    for pol, text in (("A", "Politik A (alles veränderbar) 0 %"), ("B", "Politik B (Vorlauf und Zoll) 30 bis 45 %"), ("C", "Politik C (nur Vorlauf) 41 bis 55 %")):
        los = [EV.band(data, k, pol, 0.0, "noway")["lo"] for k in M.MODELS]
        his = [EV.band(data, k, pol, 0.0, "noway")["hi"] for k in M.MODELS]
        got = "0 %" if max(his) == 0 else f"{EV.fmt(min(los))} bis {EV.fmt(max(his))} %"
        assert got in text and text in README, (pol, got)
    kind, txt = EV.verdict_policy(data, "C", 0.0)
    assert "C: nur Vorlauf: für 41 bis 55 %" in txt and "B: Vorlauf und Zoll: für 30 bis 45 %" in txt


def test_more_than_one_way_by_sets_of_changed_features(data):
    parts = [rng(EV.band(data, k, "A", 0.0, "mehrweg")) for k in ("reg", "baum", "boost", "netz", "opt")]
    assert parts == ["14 bis 19 %", "0 bis 3 %", "20 bis 27 %", "20 bis 23 %", "5 bis 7 %"]
    assert "Regression 14 bis 19 %, Baum 0 bis 3 %, Boosting 20 bis 27 %, Netz 20 bis 23 %, exakt optimaler Baum 5 bis 7 %" in README


def test_tab_numbers(data):
    b = EV.tab_band(data, 10, 5, "blaetter")
    assert rng(b) == "61 bis 87 %" and EV.fmt(b["all"], 1) == "75,6" and b["n"] == 324
    assert "Zählung nach Blättern 61 bis 87 % (75,6 % über alle 324 Instanzen), nach Mengen geänderter Merkmale 50 bis 87 %" in README
    assert rng(EV.tab_band(data, 10, 5, "mengen")) == "50 bis 87 %"
    assert rng(EV.tab_band(data, 4, 5, "blaetter")) == "2 bis 16 %" and rng(EV.tab_band(data, 4, 5, "mengen")) == "1 bis 12 %" and EV.tab_band(data, 4, 3, "blaetter")["all"] == 0
    assert "Schwelle 4 Tage, Tiefe 5: nach Blättern 2 bis 16 %, nach Mengen 1 bis 12 %; Tiefe 3 bei Schwelle 4 Tage: 0 %" in README
    ns = [EV.tab_band(data, 10, d, h)["n"] for d in (3, 4, 5) for h in ("blaetter", "mengen")]
    assert (min(ns), max(ns)) == (191, 327) and "191 bis 327" in README


def test_correction_table_build_column_is_the_measurement(data):
    """Die Spalte „Bau“ der Korrekturtabelle (Zeilen mit Messwerten dieser Demo)."""
    inv0 = " / ".join(rng(EV.band(data, k, "A", 0.0, "inv")) for k in M.MODELS)
    assert inv0.replace(" %", "") == "15 bis 22 / 35 bis 44 / 31 bis 35 / 26 bis 33 / 15 bis 40"
    assert "15 bis 22 / 35 bis 44 / 31 bis 35 / 26 bis 33 / 15 bis 40 %" in README
    inv1 = " / ".join(rng(EV.band(data, k, "A", 1.0, "inv")) for k in M.MODELS).replace(" %", "")
    assert inv1 == "0 / 0 bis 24 / 8 bis 13 / 8 bis 10 / 11 bis 34" and "0 / 0 bis 24 / 8 bis 13 / 8 bis 10 / 11 bis 34 %" in README
    assert "30 bis 36 / 21 bis 27 %" in README and "0,3 bis 1,3 / 0 / 4,0 bis 6,3 / 2,7 bis 7,3 / 0 bis 0,7 %" in README


def test_numbers_of_the_method_and_provenance(data):
    assert data["meta"]["rechenzeit_sekunden"] == 427 and "427 s" in README
    assert sum(s["cells"]["reg"]["A|0.0"]["n"] for s in data["seeds"]) == 1500 and "bis zu 1 500" in README
    assert [sum(s["cells"]["reg"]["A|0.0"]["n"] for s in g) for g in EV.groups(data)] == [300] * 5
    assert F.THR == 4 and F.N_TRAIN == 1000 and F.N_TEST == 400 and "400 statt 4 000" in README
    assert "108 von 108" in README and "45 von 45" in README
    assert "220 von 1 000" in README and "7 und 8 Blätter" in README


def test_shared_text_is_the_same_in_readme_and_app():
    assert T.LIMITS_MARKDOWN.strip() in README
    assert T.DATA_NOTE.replace("**", "") in README.replace("**", "")


def test_readme_test_count_equals_the_suite():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    n = int(re.findall(r"(\d+) tests? collected", out)[-1])
    assert f"{n} Tests (`pytest --collect-only -q tests`)" in README


def test_no_forbidden_words_in_texts():
    for text in (README, T.LIMITS_MARKDOWN, (ROOT / "app.py").read_text(encoding="utf-8")):
        for word in (r"Saatwert", r"Schwanz", r"\bträgt\b", r"\btragen\b"):
            assert not re.search(word, text), word


# ---------------------------------------------------------------- Nachprüfung nach der Erklärseite (2026-11-20)
def test_policy_a_stays_below_one_half_but_the_tighter_policies_do_not(data):
    """Der Satz „zwischen 0 und 1/2“ gilt nur bei Politik A; bei B und C überschreitet ihn der exakt optimale Baum (Anteil ungültig nach R1)."""
    keys = list(M.MODELS) + ["opt_g"]
    margins = (0.0, 0.5, 1.0)

    def top(pol, mgs=margins, ks=keys):
        return max(EV.band(data, k, pol, mg, "inv")["hi"] for k in ks for mg in mgs)

    assert round(top("A")) == 44 and "bei höchstens 44 %" in README
    assert round(top("B", (0.0,))) == 55 and round(top("B")) == 65 and round(top("C")) == 65
    assert top("B") > 50 and top("C") > 50
    assert top("B", ks=("reg", "baum", "boost", "netz")) < 50 and top("C", ks=("reg", "baum", "boost", "netz")) < 50   # nur der exakt optimale Baum
    assert "bis 65 % (Politik B bei Abstand 0 bis 55 %, bei Abstand 1,0 bis 65 %; Politik C bis 65 %)" in README
    assert "je Gruppe bis 65 %" in (ROOT / "app.py").read_text(encoding="utf-8") and "zwischen 0 und 44 %" in (ROOT / "app.py").read_text(encoding="utf-8")


def test_margin_preset_help_matches_the_measurement(data):
    """„auf etwa ein Viertel bis ein Drittel“ (Baum, Boosting, Netz), „ganz“ nur bei der Regression nach R1, „kaum“ beim exakt optimalen Baum."""
    ratio = {k: EV.band(data, k, "A", 1.0, "inv")["all"] / EV.band(data, k, "A", 0.0, "inv")["all"] for k in M.MODELS}
    assert all(0.25 <= ratio[k] <= 0.33 for k in ("baum", "boost", "netz")), ratio
    assert ratio["reg"] == 0 and EV.band(data, "reg", "A", 1.0, "inv")["hi"] == 0
    assert ratio["opt"] > 0.8
    help_text = C.PRESET_HELP["Mit Sicherheitsabstand 1,0"]
    assert "ein Viertel bis ein Drittel" in help_text and "nach R1 ganz" in help_text and "beim exakt optimalen Baum kaum" in help_text
    assert "nach R1 in jeder Gruppe auf 0 %" in README


def _logit_after_way(key, seeds):
    out = []
    for s in seeds:
        fit = EV.fit_live(key, s)
        sc = fit.model.cand_scores()
        for lev in fit.levs:
            bf = R.cheapest(lev, R.POLICIES["A"], 0.0, sc, fit.sd_v)
            if bf is not None:
                out.append(float(sc[bf[1]]))
    return np.array(out)


@pytest.mark.parametrize("key", ["reg", "boost", "baum"])
def test_the_cheapest_way_lies_behind_the_boundary_but_not_directly_on_it(key):
    """Lage des Wegs zur Grenze (README, „Lage des Wegs zur Grenze“): Logit nach dem Weg im Median zwischen −0,8 und −0,3, höchstens 20 % innerhalb von 0,1 der Grenze."""
    lg = _logit_after_way(key, range(40000, 40006))
    assert len(lg) > 100                                     # bis zu 25 Instanzen je Zufalls-Seed, Politik A hat immer einen Weg
    assert (lg <= -R.EPS).all()                              # jeder Weg liegt hinter der Grenze
    assert -0.8 <= float(np.median(lg)) <= -0.3, np.median(lg)
    assert float(np.mean(lg > -0.1)) <= 0.20


def test_position_numbers_in_the_readme():
    assert "im Median bei −0,44 (Regression), −0,54 (Boosting) und −0,64 (Baum); innerhalb von 0,1 der Entscheidungsgrenze liegen 12, 12 und 0 % der Wege" in README
    assert "Median zwischen −0,8 und −0,3, höchstens 20 % innerhalb von 0,1" in README


def test_the_small_net_mostly_hits_the_iteration_limit():
    """README (Grenzen): von 12 Trainings (Zufalls-Seeds 40000 bis 40011) erreichten 11 das Limit; als Band mindestens 9 (Plattform)."""
    hit = sum(not EV.fit_live("netz", s).model.converged for s in range(40000, 40012))
    assert 9 <= hit <= 12, hit
    assert "von 12 Trainings (Zufalls-Seeds 40000 bis 40011) taten es 11" in README and "mindestens 9 von 12" in README


def test_corrected_wording_in_app_and_readme():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "per Konstruktion" not in app and "per Konstruktion" not in README
    assert "nie als Punktzahl aus einer einzigen Ziehung" in app and "nie als Punktzahl aus einer Ziehung" not in app
    assert "öffnet „💬 In einfachen Worten“" in app
    assert "Zeitspanne in Tagen bis zur Schiffsabfahrt" in T.DATA_NOTE                                     # Vorlauf erklärt, README gleichlautend (Test oben)
    assert "Der Satz „zwischen 0 und 1/2“ gilt also nur bei Politik A" in README
    assert "Bei Umschlag erlaubt das Raster Vorlauf bis 10 Tage, der Generator zieht nur bis 6 Tage" in T.LIMITS_MARKDOWN
