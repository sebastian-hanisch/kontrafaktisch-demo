"""Erzeugt `kfk_fragilitaet.json` (eingefrorene Messung: Wege, Neu-Training, Wahrheit, gleich gute Wege) oder prüft sie mit `--check`.

Aufruf (Test-venv, im Ordner der Demo):
    python -X utf8 tools/kfk_make_fragilitaet.py [--workers 4] [--seeds 60] [--basis 40000] [--tab-seeds 30] [--out kfk_fragilitaet.json]
    python -X utf8 tools/kfk_make_fragilitaet.py --check          # rechnet Zufalls-Seed 0 (zwei Modellklassen) neu und vergleicht als Band

Die Messung ist schwer (je Zufalls-Seed rund 30 s bei einem Prozess, 60 Zufalls-Seeds also einige Minuten mit vier Prozessen), darum friert die App das Ergebnis ein.
Jedes Feld der Datei steht in `meta` mit Quelle und Stand. `OMP_NUM_THREADS=1` wird gesetzt (sonst hängen BLAS-Threads unter Last)."""

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import date

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

import kfk_fragility as F  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "kfk_fragilitaet.json")


def _job(args):
    i, basis = args
    return F.run_seed(i, basis)


def _job_tab(args):
    i, basis = args
    return F.way_counts_seed(i, basis)


def make(out, workers, seeds, basis, tab_seeds, tab_basis):
    import sklearn

    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_job, [(i, basis) for i in range(seeds)]))
        tab = list(ex.map(_job_tab, [(i, tab_basis) for i in range(tab_seeds)])) if tab_seeds else []
    meta = {
        "quelle": "tools/kfk_make_fragilitaet.py mit kfk_fragility.py (Generator kfk_generator.py, SplitMix64, erfunden, keine reale Datenquelle)",
        "stand": date.today().isoformat(),
        "python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
        "seeds": {"quelle": "Zufalls-Seed i = 0..%d; Training basis + i, Neu-Training basis + 5000 + 1000 r + i (r = 0 ist +10 %% Daten), Test basis + 100000 + i" % (seeds - 1), "basis": basis},
        "gruppen": {"quelle": "fünf Gruppen zu je %d Zufalls-Seeds in Reihenfolge" % F.GROUP_SIZE, "groesse": F.GROUP_SIZE, "n_gruppen": F.N_GROUPS},
        "schwelle_tage": F.THR, "n_train": F.N_TRAIN, "n_test": F.N_TEST, "instanzen_je_seed": F.N_INST, "neu_training": F.N_DRAWS,
        "zellen": "je Modellklasse Politik|Abstand: n Instanzen, noway ohne Weg, inv[11] ungültig nach R1..R11 (von n - noway), wahr ungültig in der Wahrheit, rausch Weg ändert das Gewicht, mehrweg mehr als eine Menge geänderter Merkmale innerhalb 10 %, kosten",
        "tab": {"quelle": "scikit-learn-Baum, n_train %d, bis zu %d Instanzen je Zufalls-Seed, Politik A, Abstand 0, Weg auf Rasterpunkten; Zufalls-Seeds Training %d + i, Test %d + i"
                % (F.TAB_TRAIN, F.TAB_INST, tab_basis, tab_basis + 100000), "basis": tab_basis, "seeds": tab_seeds},
        "rechenzeit_sekunden": round(time.time() - t0),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "seeds": res, "tab": tab}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"geschrieben: {out} ({os.path.getsize(out) // 1024} KB, {len(res)} Zufalls-Seeds, {len(tab)} Tab-Zufalls-Seeds, {time.time() - t0:.0f} s)")


def check(path):
    """Rechnet Zufalls-Seed 0 für zwei Modellklassen neu und vergleicht mit der Datei: Anzahlen als Band (Gleitkomma-Letztstelle kann einzelne Instanzen kippen)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    basis = data["meta"]["seeds"]["basis"]
    new = F.run_seed(0, basis, keys=("reg", "baum"))
    old = data["seeds"][0]
    bad = []
    for key in ("reg", "baum"):
        for ck, cell in new["cells"][key].items():
            o = old["cells"][key][ck]
            if cell["n"] != o["n"]:
                bad.append((key, ck, "n", cell["n"], o["n"]))
            for field in ("noway", "wahr", "rausch", "mehrweg"):
                if abs(cell[field] - o[field]) > max(2, 0.1 * max(cell["n"], 1)):
                    bad.append((key, ck, field, cell[field], o[field]))
            for d in range(len(o["inv"])):
                if abs(cell["inv"][d] - o["inv"][d]) > max(2, 0.1 * max(cell["n"], 1)):
                    bad.append((key, ck, f"inv{d + 1}", cell["inv"][d], o["inv"][d]))
    print("Abweichungen:", bad if bad else "keine (alle Zahlen im Band)")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=F.GROUP_SIZE * F.N_GROUPS)
    ap.add_argument("--basis", type=int, default=40000)
    ap.add_argument("--tab-seeds", type=int, default=30)
    ap.add_argument("--tab-basis", type=int, default=20000)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(check(a.out))
    make(a.out, a.workers, a.seeds, a.basis, a.tab_seeds, a.tab_basis)
