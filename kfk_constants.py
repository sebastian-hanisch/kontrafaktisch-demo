"""Konstanten: Reglergrenzen, feste Annahmen, Farben und Presets."""

import kfk_models as M
import kfk_recourse as R

DATA_FILE = "kfk_fragilitaet.json"

MODEL_OPTIONS = M.MODELS                      # reg, baum, boost, netz, opt
POLICY_OPTIONS = tuple(R.POLICIES)            # A, B, C
MARGIN_OPTIONS = R.MARGINS                    # 0,0 / 0,5 / 1,0
RETRAIN_OPTIONS = ("R1", "R2", "R3", "R2-R11")
RETRAIN_NAMES = {
    "R1": "R1: 10 % mehr Daten, eine frische Ziehung",
    "R2": "R2: gleiche Menge, frische Ziehung",
    "R3": "R3: gleiche Menge, zweite frische Ziehung",
    "R2-R11": "Mittel der zehn Ziehungen gleicher Menge (R2 bis R11)",
}
THRESHOLD_OPTIONS = (4, 10)
DEPTH_OPTIONS = (3, 4, 5)
COUNT_OPTIONS = ("blaetter", "mengen")
COUNT_NAMES = {"blaetter": "nach Blättern des Baums", "mengen": "nach Mengen geänderter Merkmale"}

INSTANCE_MIN, INSTANCE_MAX, DEFAULT_INSTANCE = 1, 25, 1
SEED_MAX, DEFAULT_SEED = 999999, 7
N_TRAIN, N_TEST = 1000, 400                   # Live-Rechnung wie die eingefrorene Messung

COLOR_MODEL = "#2F6B65"
COLOR_BAND = "rgba(76,120,168,0.25)"
COLOR_POINT = "#4c78a8"
COLOR_BAD = "#c0504d"
COLOR_OTHER = "rgba(150,150,150,0.55)"

_BASE = {"modell": "reg", "instanz": DEFAULT_INSTANCE, "politik": "A", "abstand": 0.0, "retrain": "R1", "schwelle": 4, "tiefe": 4, "zaehlweise": "blaetter", "seed": DEFAULT_SEED}

PRESETS = {
    "Ein Weg (Regression, Abstand 0)": dict(_BASE),
    "Gilt der Weg nach Neu-Training?": dict(_BASE, retrain="R2-R11"),
    "Mit Sicherheitsabstand 1,0": dict(_BASE, abstand=1.0),
    "Nur Vorlauf veränderbar": dict(_BASE, politik="C"),
    "Boosting statt Regression": dict(_BASE, modell="boost"),
    "Exakt optimaler Baum": dict(_BASE, modell="opt"),
    "Gleich gute Wege (Schwelle 10 Tage, Tiefe 5)": dict(_BASE, schwelle=10, tiefe=5),
}
PRESET_HELP = {
    "Ein Weg (Regression, Abstand 0)": "Der billigste Weg für eine Instanz der logistischen Regression: Alles darf sich ändern, kein Sicherheitsabstand (Experiment 1).",
    "Gilt der Weg nach Neu-Training?": "Wie Experiment 3 mit dem Mittel über zehn Ziehungen gleicher Menge statt einer einzigen Ziehung: Anteil der Wege, die nach einem Neu-Training nicht mehr gelten.",
    "Mit Sicherheitsabstand 1,0": "Der Weg muss das Logit bis unter -1,0 treiben statt nur unter 0: kostet mehr und hilft bei der Regression nach R1 ganz (in der zweiten Messung der README 0 bis 1 %), bei Baum, Boosting und Netz teilweise (auf etwa ein Viertel bis ein Drittel des Anteils bei Abstand 0), beim exakt optimalen Baum kaum.",
    "Nur Vorlauf veränderbar": "Nur der Vorlauf darf sich ändern (Politik C): Für einen großen Teil der Instanzen gibt es keinen Weg (Experiment 2).",
    "Boosting statt Regression": "Dieselbe Frage für Gradient Boosting: Der Weg ändert öfter ein Merkmal ohne Wirkung mit und gilt nach einem Neu-Training seltener.",
    "Exakt optimaler Baum": "Der exakt optimale Baum der Tiefe 3 (Dynamische Programmierung über das Raster) mit dem Weg über Blattquader; Gleichstandsregel kleinster Index.",
    "Gleich gute Wege (Schwelle 10 Tage, Tiefe 5)": "Stellt Schwelle 10 Tage, Tiefe 5 ein; die Ansicht steht im Tab „Gleich gute Wege“ von Experiment 1: Wie viele Wege es gibt, hängt an der Zählweise.",
}
