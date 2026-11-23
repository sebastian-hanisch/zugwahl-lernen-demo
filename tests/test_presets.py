"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, eindeutige Hervorhebung, Permalink einrasten, vorgerechnete Modelle."""

import numpy as np
import pytest

import zwl_constants as C
import zwl_mlp as MLP
import zwl_presets as P


def test_jedes_preset_hat_hilfetext_und_alle_schluessel():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_werte_liegen_in_den_reglergrenzen():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.options is not None:
                assert preset[key] in spec.options, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)
        assert preset["seed"] <= P.seed_max(preset["n"])
    assert max(C.Z_SCHRITTE_STUFEN) == C.Z_SCHRITTE_MAX == 3000


def test_standard_preset_ist_das_erste_und_gleich_den_standardeinstellungen():
    erstes = C.PRESETS[next(iter(C.PRESETS))]
    standard = {key: P.SETTING_SPECS[state_key].default for key, state_key in P.PRESET_KEYS.items()}
    assert erstes == standard and next(iter(C.PRESETS)) == "Der Beweis kostet etwa die Hälfte" == P.DEFAULT_PRESET
    assert (erstes["n"], erstes["regel"], erstes["seed"]) == (20, "zufall", 0)


def test_setting_specs_decken_jeden_preset_schluessel_ab():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_unterscheiden_sich_so_dass_die_hervorhebung_eindeutig_ist():
    werte = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(werte)) == len(werte)


def test_presets_sind_die_geplanten_sieben():
    assert list(C.PRESETS) == ["Der Beweis kostet etwa die Hälfte", "Handbeispiel: sechs Orte", "Das Netz ordnet besser", "Gleiches Budget, kleiner Einsatz", "Größere Tour",
                               "Nachbarliste statt Beweis", "Abbruch statt Beweis"]
    p = C.PRESETS
    assert (p["Handbeispiel: sechs Orte"]["n"], p["Handbeispiel: sechs Orte"]["regel"]) == (6, "aufzaehlung")
    assert (p["Das Netz ordnet besser"]["regel"], p["Das Netz ordnet besser"]["quelle"], p["Das Netz ordnet besser"]["gewicht"]) == ("gelernt", "vorgerechnet", 0)
    assert (p["Gleiches Budget, kleiner Einsatz"]["budget"], p["Gleiches Budget, kleiner Einsatz"]["regel"], p["Gleiches Budget, kleiner Einsatz"]["n"]) == (2, "gelernt", 20)
    assert (p["Größere Tour"]["n"], p["Größere Tour"]["budget"]) == (50, 2)
    assert (p["Nachbarliste statt Beweis"]["n"], p["Nachbarliste statt Beweis"]["liste"]) == (50, 3)
    assert (p["Abbruch statt Beweis"]["n"], p["Abbruch statt Beweis"]["abbruch"], p["Abbruch statt Beweis"]["regel"]) == (50, 100, "gelernt")


def test_einrasten_des_permalinks():
    spec = P.SETTING_SPECS["n_select"]
    assert P.clamp(spec, 12) == 10 and P.clamp(spec, 100) == 50 and P.clamp(spec, 1) == 6 and P.clamp(spec, 30) == 20
    assert P.clamp(P.SETTING_SPECS["regel_select"], "unsinn") is None and P.clamp(P.SETTING_SPECS["regel_select"], "gelernt") == "gelernt"
    assert P.clamp(P.SETTING_SPECS["gewicht_input"], 50) == 4 and P.clamp(P.SETTING_SPECS["seed_input"], -3) == 0
    assert P.clamp(P.SETTING_SPECS["liste_select"], 4) == 3 and P.clamp(P.SETTING_SPECS["abbruch_select"], 70) == 50


def test_seed_grenzen_je_groesse():
    assert [P.seed_max(n) for n in C.N_STUFEN] == [0, 19, 19, 9]


@pytest.mark.parametrize("art,params", [("Z", C.Z_PARAMS), ("K", C.K_PARAMS)])
def test_vorgerechnete_modelle_liegen_vor(art, params):
    for seed in range(C.MODELL_SEEDS):
        p, meta = MLP.load_model(MLP.model_path(art, seed))
        assert MLP.n_params(p) == params and meta["kind"] == art and int(meta["seed"]) == seed and np.isfinite(p[0]).all()
        assert int(meta["feature_version"]) == C.FEATURE_VERSION
    assert MLP.model_path(art, 0).stat().st_size < 6000


def test_hilfetexte_der_presets_nennen_gemessene_zahlen_der_readme():
    """Die Zahlen der Hilfetexte stehen in der README (Experiment 2 und 4b) und sind dort an die Messung gebunden (`test_claims.py`)."""
    from pathlib import Path
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    for zahl in ("53,7 %", "3 057 von 12 875", "18,0 %"):
        assert zahl in readme, zahl
    assert "53,7 %" in C.PRESET_HELP["Der Beweis kostet etwa die Hälfte"]
    h = C.PRESET_HELP["Nachbarliste statt Beweis"]
    assert "3 057 von 12 875" in h and "rund 18 %" in h and "fast immer" not in h
