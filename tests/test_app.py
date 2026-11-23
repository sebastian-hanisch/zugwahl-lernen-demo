"""End-to-end-Rauchtest mit Streamlits AppTest: lädt app.py, klickt jedes Preset, fährt jeden Regler der Seitenleiste an seine Grenzen, klickt die Trainingsknöpfe und die Live-Proben.
Geprüft wird, dass kein Python-Fehler auftritt und die erwarteten Meldungen erscheinen (AppTest sieht keine Plotly-Fehler im Browser)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import zwl_constants as C
import zwl_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))
REGLER_ARTEN = ("select_slider", "number_input", "selectbox", "radio")


def _app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(elemente) -> str:
    return " ".join(e.value for e in elemente)


def _alle_meldungen(at) -> str:
    return " ".join([_texte(at.success), _texte(at.info), _texte(at.warning), _texte(at.error)])


def test_app_laedt_ohne_fehler_und_zeigt_die_verdikte_des_standards():
    at = _app()
    assert not at.error
    m = _alle_meldungen(at)
    assert "Zugauswertungen" in m and "beweisen nur das Lokalminimum" in m                      # E0
    assert "Das Netz ordnet die Züge klar besser als die Nachbarlisten-Reihenfolge" in m          # E1
    assert "der Zugauswertungen" in m and "beweist nur, dass kein Zug mehr verbessert" in m        # E2
    assert "Voraussetzung" in m or "Budget k = 2" in m                                           # E3 mit Voraussetzung
    assert "Die Nachbarliste (k = 3)" in m                                                       # E4b
    assert "Eine perfekte Startlösung spart bei Zufallsreihenfolge im Abstieg höchstens" in m                                      # E5


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_ohne_fehler(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, (name, [str(e) for e in at.exception])
    assert not at.error


def test_jeder_regler_der_seitenleiste_an_den_grenzen():
    at = _app()
    regler = [(art, w.key) for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key]
    assert {k for _, k in regler} == set(P.SETTING_SPECS)
    for art, key in regler:
        spec = P.SETTING_SPECS[key]
        werte = (spec.options[0], spec.options[-1]) if spec.options else (spec.lo, spec.hi)
        for wert in werte:
            at = _app()
            if key in ("schritte_select", "trainings_inst_select"):                    # nur bei „Hier trainieren“ bedienbar
                at.sidebar.radio(key="quelle_select").set_value("trainieren").run(timeout=TIMEOUT)
            getattr(at.sidebar, art)(key=key).set_value(wert).run(timeout=TIMEOUT)
            assert not at.exception, (key, wert, [str(e) for e in at.exception])


def test_jede_regel_laeuft():
    for regel in C.REGELN:
        at = _app()
        at.sidebar.selectbox(key="regel_select").set_value(regel).run(timeout=TIMEOUT)
        assert not at.exception, (regel, [str(e) for e in at.exception])


def test_schritt_regler_der_primaeransicht_an_den_grenzen():
    at = _app()
    key = next(s.key for s in at.slider if s.key and s.key.startswith("e0_schritt_"))
    s = at.slider(key=key)
    for wert in (s.min, s.max):
        at.slider(key=key).set_value(wert).run(timeout=TIMEOUT)
        assert not at.exception
    assert at.slider(key=key).min == 0


def test_handbeispiel_zeigt_tabelle_sieben_zuege_und_pruefrunde():
    at = _app()
    at.sidebar.button(key="preset_Handbeispiel: sechs Orte").click().run(timeout=TIMEOUT)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "M(6) = 93" in text and "verbessern 8 von 93" in text and "Verschiedene Nachbartouren sind es nur 29" in text
    assert "93 % von 100 Zugauswertungen beweisen nur das Lokalminimum" in _texte(at.info)
    assert "Experiment 1 braucht n von mindestens 10" in _texte(at.info)


def test_trainingsklick_zugnetz_kleinste_einstellung():
    at = _app()
    at.sidebar.radio(key="quelle_select").set_value("trainieren").run(timeout=TIMEOUT)
    assert "noch kein Zugnetz da" in _texte(at.info)                                       # vor dem Klick: Meldung statt Rechnung
    at.sidebar.select_slider(key="schritte_select").set_value(300).run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="trainings_inst_select").set_value(50).run(timeout=TIMEOUT)
    at.button(key="trainieren_knopf").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "Dieser Lauf: 300 Schritte × 4 096 = 1 228 800 Zeilen" in " ".join(c.value for c in at.caption)
    assert "noch kein Zugnetz da" not in _texte(at.info)


def test_gelernte_regel_ohne_trainiertes_zugnetz_meldet_statt_zu_rechnen():
    at = _app()
    at.sidebar.radio(key="quelle_select").set_value("trainieren").run(timeout=TIMEOUT)
    at.sidebar.selectbox(key="regel_select").set_value("gelernt").run(timeout=TIMEOUT)
    assert not at.exception and "braucht ein Zugnetz" in _texte(at.info)


def test_live_probe_e3_und_e4():
    at = _app()
    at.sidebar.button(key="preset_Gleiches Budget, kleiner Einsatz").click().run(timeout=TIMEOUT)
    at.button(key="probe_knopf").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "Eine Instanz, 5 Zufalls-Seeds, Mittel; das ist eine Probe, kein Befund." in " ".join(c.value for c in at.caption)
    at.sidebar.select_slider(key="abbruch_select").set_value(50).run(timeout=TIMEOUT)
    assert not at.exception
    assert "unabhängige Gegenprobe" in " ".join(c.value for c in at.caption)


def test_probetraining_merkmalsnetz_k():
    at = _app()
    at.radio(key="e5_kquelle").set_value("probetraining").run(timeout=TIMEOUT)
    assert "Noch kein Probetraining" in _texte(at.info)
    at.button(key="kprobe_knopf").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    meldungen = _alle_meldungen(at)
    assert ("Kaum gelernt" in meldungen) or ("Besser als Nächster Nachbar" in meldungen)


def test_seed_wird_bei_n50_auf_neun_begrenzt():
    at = _app()
    at.sidebar.number_input(key="seed_input").set_value(15).run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="n_select").set_value(50).run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["seed_input"] == 9


def test_budget_32_bei_n50_meldet_statt_zu_rechnen():
    at = _app()
    at.sidebar.select_slider(key="n_select").set_value(50).run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="budget_select").set_value(32).run(timeout=TIMEOUT)
    assert not at.exception and "nur die Budgets k = 2 und 8" in _texte(at.info)


def test_familien_ohne_eingefrorene_messung_rechnen_live():
    at = _app()
    at.sidebar.selectbox(key="familie_select").set_value("cluster").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "Für diese Zelle ist nichts gemessen" in _texte(at.info)
