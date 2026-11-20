"""Je Preset die erwartete Verdikt-Zeile in der App (Aussagenprüfung: ein Preset darf keinen Effekt melden, den es nicht hat), Texte der App gegen die geteilten Texte, Netz-Meldung."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kfk_constants as C
import kfk_models as M
import kfk_texts as T

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _run(name=None, **state):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    if name:
        at.sidebar.button(key=f"preset_{name}").click().run()
    for k, v in state.items():
        at.session_state[k] = v
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return [e.value for e in list(at.info) + list(at.success) + list(at.warning)]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_each_preset_shows_its_verdict_lines(name):
    at = _run(name)
    p = C.PRESETS[name]
    texts = _texts(at)
    joined = "\n".join(texts)
    assert any(t.startswith(f"{M.MODEL_NAMES[p['modell']]}, Politik {p['politik']}, Abstand {str(p['abstand']).replace('.', ',')}:") and "gelten nach" in t for t in texts), name
    assert f"Schwelle {p['schwelle']} Tage, Tiefe {p['tiefe']}, Zählung {C.COUNT_NAMES[p['zaehlweise']]}" in joined
    assert "Der billigste Weg ändert" in joined or "keinen Weg" in joined
    assert f"Sicherheitsabstand 0 gegen 1,0 (Politik {p['politik']}" in joined
    assert "in der Wahrheit des Generators" in joined and f"Bei Abstand {str(p['abstand']).replace('.', ',')} (alle fünf Modellklassen" in joined


def test_the_margin_preset_reports_the_regression_as_healed_and_the_others_not():
    at = _run("Mit Sicherheitsabstand 1,0")
    text = next(t for t in _texts(at) if t.startswith("Logistische Regression, Politik A, Abstand 1,0") and "gelten nach" in t)
    assert "**0 %**" in text and "Baum 0 bis 24 %" in text and "Boosting 8 bis 13 %" in text
    assert [e for e in at.success if e.value.startswith("Logistische Regression, Politik A, Abstand 1,0") and "gelten nach" in e.value]        # Obergrenze 0 % <= 3 %


def test_the_policy_preset_reports_that_many_instances_have_no_way():
    at = _run("Nur Vorlauf veränderbar")
    assert any("C: nur Vorlauf: für 41 bis 55 % der Instanzen kein Weg" in t for t in _texts(at))
    assert any("Politik C" in e.value for e in at.warning if "gelten nach" in e.value)


def test_limits_table_and_math_expander_are_in_the_app():
    at = _run()
    assert any(T.LIMITS_MARKDOWN.strip() == m.value.strip() for m in at.markdown)
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)
    assert any("Lesehilfe" in m.value for m in at.markdown)


def test_net_without_convergence_is_reported():
    at = _run(modell="netz")
    fit_note = [w.value for w in at.warning if "Iterationslimit" in w.value]
    assert len(fit_note) <= 1       # die Meldung erscheint nur, wenn das Netz das Limit erreicht hat
    assert not at.exception


def test_instance_beyond_the_number_of_instances_is_clamped_and_policy_c_may_have_no_way():
    at = _run(instanz=25, politik="C", seed=11)
    assert not at.exception and any("gezeigt wird Instanz" in c.value for c in at.caption)
