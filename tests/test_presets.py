"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset einzigartig (sonst würde nur das erste hervorgehoben); Permalink begrenzt."""

import kfk_constants as C
import kfk_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    assert len(C.PRESETS) == 7
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_or_the_options():
    for preset in C.PRESETS.values():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            assert P.clamp(spec, value) == value, (key, value)


def test_default_preset_equals_the_default_settings_and_all_presets_differ():
    first = next(iter(C.PRESETS.values()))
    assert first == {k: s.default for k, s in P.SETTING_SPECS.items()}
    frozen = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(frozen)) == len(frozen)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_clamp_limits_numbers_and_rejects_unknown_options():
    s = P.SETTING_SPECS["instanz"]
    assert P.clamp(s, 0) == 1 and P.clamp(s, 99) == 25 and P.clamp(s, 7) == 7
    assert P.clamp(P.SETTING_SPECS["modell"], "xyz") == "reg" and P.clamp(P.SETTING_SPECS["modell"], "opt") == "opt"
    assert P.clamp(P.SETTING_SPECS["abstand"], 0.7) == 0.0 and P.clamp(P.SETTING_SPECS["seed"], -5) == 0


def test_presets_name_the_experiments_they_belong_to():
    assert C.PRESETS["Nur Vorlauf veränderbar"]["politik"] == "C" and C.PRESETS["Mit Sicherheitsabstand 1,0"]["abstand"] == 1.0
    assert C.PRESETS["Exakt optimaler Baum"]["modell"] == "opt" and C.PRESETS["Boosting statt Regression"]["modell"] == "boost"
    g = C.PRESETS["Gleich gute Wege (Schwelle 10 Tage, Tiefe 5)"]
    assert (g["schwelle"], g["tiefe"]) == (10, 5)
