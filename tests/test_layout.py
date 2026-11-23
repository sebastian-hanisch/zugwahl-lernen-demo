"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt
sicher, dass wirklich JEDER Regler der Seitenleiste dort steht: sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import zwl_constants as C
import zwl_presets as P
import zwl_texts as TX

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht in der Seitenleiste")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def _anderer_wert(spec, wert):
    if spec.options is not None:
        return next(o for o in spec.options if o != wert)
    return spec.hi if wert != spec.hi else spec.lo


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value == TX.TITEL
    for wort in ("Konfiguration", "Parameterwahl", "Verfahrenswahl", "Wahlregeln"):
        assert wort not in at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [P.DEFAULT_PRESET] == ["Der Beweis kostet etwa die Hälfte"]


def test_erste_expander_und_titel_der_abschnitte():
    at = _app()
    namen = [e.label for e in at.expander]
    assert namen[0] == "💬 In einfachen Worten" and namen[-1] == "📐 Mathematische Formulierung"
    assert "So funktioniert die gelernte Zugwahl" in namen
    ueberschriften = [s.value for s in at.subheader]
    assert "🚧 Wo die Annahmen enden" in ueberschriften and sum(1 for u in ueberschriften if u.startswith("🔬")) == 5


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"


# Regler, die nur bei „Hier trainieren“ wirken (disabled sonst, AppTest darf sie dann nicht setzen; kein Preset trainiert): sie prüft der zustandslose Test unten
NUR_BEI_TRAINING = ("schritte_select", "trainings_inst_select")


@pytest.mark.parametrize("state_key", [k for k in P.SETTING_SPECS if k not in NUR_BEI_TRAINING])
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_des_vorher_aktiven_presets_auf(state_key):
    at = _app()
    start = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{start}").click().run()
    assert _typen(at)[start] == "primary"
    spec = P.SETTING_SPECS[state_key]
    regler = _regler(at, state_key)
    regler.set_value(_anderer_wert(spec, regler.value)).run()
    assert not at.exception, [str(e) for e in at.exception]
    typen = _typen(at)
    assert typen[start] != "primary"
    # ist danach ein anderes Preset markiert (die Änderung führt genau auf dessen Werte), dann stimmt die Markierung mit den Reglern überein
    for name, typ in typen.items():
        if typ == "primary":
            for key, sk in P.PRESET_KEYS.items():
                assert at.session_state[sk] == C.PRESETS[name][key], (name, key)
    # erneutes Anklicken markiert wieder
    at.sidebar.button(key=f"preset_{start}").click().run()
    assert _typen(at)[start] == "primary"


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_aktives_preset_zustandslos_jede_aenderung_hebt_auf(state_key, monkeypatch):
    """Ohne AppTest: eine Änderung jedes einzelnen Reglers (auch der nur bei einer Wahl wirkenden) hebt die Markierung des Presets auf."""
    name = "Gleiches Budget, kleiner Einsatz"
    zustand = {sk: C.PRESETS[name][key] for key, sk in P.PRESET_KEYS.items()}
    monkeypatch.setattr(P.st, "session_state", dict(zustand))
    assert P.aktives_preset() == name
    spec = P.SETTING_SPECS[state_key]
    P.st.session_state[state_key] = _anderer_wert(spec, zustand[state_key])
    assert P.aktives_preset() != name
