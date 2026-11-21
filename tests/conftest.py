"""Gemeinsame Fixtures. Schwere Rechnungen (Training, Wege) laufen einmal je Sitzung."""

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import pytest  # noqa: E402

import kfk_fragility as F  # noqa: E402
import kfk_generator as G  # noqa: E402
import kfk_models as M  # noqa: E402
import kfk_recourse as R  # noqa: E402


@pytest.fixture(scope="session")
def train():
    """Trainingsdaten (n = 1 000), gerastert, mit Labels, Vorlauf-Maßstab und Testzeilen zu Zufalls-Seed 1."""
    X, y = G.draw_history(1000, 1)
    Xr = R.to_raster(X)
    Xt, _ = G.draw_history(120, 100001)
    return {"Xr": Xr, "lab": F.labels(y), "sd_v": R.sd_vorlauf(Xr), "Xt": R.to_raster(Xt)}


@pytest.fixture(scope="session")
def models(train):
    """Je Modellklasse ein Modell auf `train` (einmal je Sitzung)."""
    return {k: M.Model(k, 1).fit(train["Xr"], train["lab"]) for k in M.MODELS}
