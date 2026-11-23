"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Button (Standardmuster aus dem Demo-Portfolio).

SETTING_SPECS: die Regler der Seitenleiste; sie stehen im Permalink und entscheiden, welches Preset hervorgehoben ist. Regler, die nur bei einer Wahl wirken
(`schritte_select` und `trainings_inst_select` bei „Hier trainieren“), bleiben immer sichtbar und sind sonst abgeschaltet (disabled), sie verschwinden nicht."""

import random
from dataclasses import dataclass

import streamlit as st

import zwl_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None       # Auswahlregler: erlaubte Werte


SETTING_SPECS = {
    "familie_select": SettingSpec("familie", str, C.DEFAULT_FAMILIE, options=C.FAMILIEN),
    "n_select": SettingSpec("n", int, C.DEFAULT_N, options=C.N_STUFEN),
    "seed_input": SettingSpec("instanz", int, C.DEFAULT_SEED, 0, max(C.SEED_MAX.values())),
    "regel_select": SettingSpec("regel", str, C.DEFAULT_REGEL, options=C.REGELN),
    "quelle_select": SettingSpec("zugnetz", str, C.DEFAULT_QUELLE, options=C.QUELLEN),
    "gewicht_input": SettingSpec("gewichte", int, C.DEFAULT_GEWICHTS_SEED, 0, C.MODELL_SEEDS - 1),
    "schritte_select": SettingSpec("schritte", int, C.Z_SCHRITTE_STANDARD, options=C.Z_SCHRITTE_STUFEN),
    "trainings_inst_select": SettingSpec("trainingsinstanzen", int, C.Z_INSTANZEN_STANDARD, options=C.Z_INSTANZEN_STUFEN),
    "budget_select": SettingSpec("k", int, 2, options=C.BUDGET_K),
    "abbruch_select": SettingSpec("abbruch", int, 0, options=C.ABBRUCH_STUFEN),
    "liste_select": SettingSpec("liste", int, 0, options=C.LISTE_STUFEN),
}
PRESET_KEYS = {
    "familie": "familie_select", "n": "n_select", "seed": "seed_input", "regel": "regel_select", "quelle": "quelle_select", "gewicht": "gewicht_input",
    "schritte": "schritte_select", "trainings_inst": "trainings_inst_select", "budget": "budget_select", "abbruch": "abbruch_select", "liste": "liste_select",
}
DEFAULT_PRESET = next(iter(C.PRESETS))


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    """(Minimum, Maximum) eines Zahlenreglers."""
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def options(state_key) -> tuple:
    """Erlaubte Werte eines Auswahlreglers."""
    return SETTING_SPECS[state_key].options


def seed_max(n: int) -> int:
    """Größter Zufalls-Seed der Instanz bei n Orten (nur dafür sind Referenzlängen vorgerechnet): 19 bei n = 10 und 20, 9 bei n = 50, 0 beim Handbeispiel."""
    return C.SEED_MAX[n]


def clamp(spec: SettingSpec, value):
    """Wert aus der Adresszeile auf den Regler abbilden: Zahlen begrenzen, Auswahlwerte einrasten (Zahl: nächster Wert; Text: nur gültige)."""
    if spec.options is not None:
        if value in spec.options:
            return value
        if spec.caster is int:
            return min(spec.options, key=lambda o: (abs(o - value), o))
        return None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            value = clamp(spec, value)
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_instance():
    n = st.session_state.get("n_select", C.DEFAULT_N)
    st.session_state["seed_input"] = random.randint(0, seed_max(n))
