"""Textregeln (Auftrag und DEMO-PLAYBOOK §7): echte Umlaute, „Zufalls-Seed“, kein „trägt“, „Tail“ statt „Schwanz“, kein Gedankenstrich im Fließtext, kein „TLA“, Titel ohne verbotene Wörter."""

import re
from pathlib import Path

import zwl_constants as C
import zwl_texts as TX

ROOT = Path(__file__).resolve().parent.parent
ANZEIGE = {name: getattr(TX, name) for name in ("TITEL", "INTRO", "EINFACHE_WORTE", "SO_FUNKTIONIERT", "ZAEHLREGEL", "ERKLAERTABELLE", "ASSUMPTIONS_MARKDOWN", "LESEHILFE", "FORMELN")}
ANZEIGE["KERNFRAGEN"] = " ".join(TX.KERNFRAGEN)
ANZEIGE["PRESETS"] = " ".join(C.PRESET_HELP.values()) + " " + " ".join(C.PRESETS)
QUELLEN = {"README.md": (ROOT / "README.md").read_text(encoding="utf-8"), "app.py": (ROOT / "app.py").read_text(encoding="utf-8"), "zwl_evaluation.py": (ROOT / "zwl_evaluation.py").read_text(encoding="utf-8")}


def _alle():
    return {**ANZEIGE, **QUELLEN}


def test_verbotene_woerter():
    for name, text in _alle().items():
        for wort in ("trägt", "tragen", "TLA", "Schwanz", "Saatwert", "Gedankenstrich"):
            assert wort not in text.replace("tragen:", ""), (name, wort)


def test_kein_gedankenstrich_im_fliesstext_der_anzeigetexte():
    for name, text in ANZEIGE.items():
        assert "–" not in text and "—" not in text, name
    assert "–" not in QUELLEN["zwl_evaluation.py"]


def test_echte_umlaute_statt_ersatzschreibweise():
    for name, text in {**ANZEIGE, "README.md": QUELLEN["README.md"]}.items():
        for falsch in ("Verbesserung ueber", "fuer ", "ueber ", "moeglich", "Loesung", "Prüfung ae", "gross", "Zugauswertung ae"):
            assert falsch not in text, (name, falsch)
        assert len(text) < 200 or re.search(r"[äöüß]", text), name


def test_titel_ohne_verbotene_woerter():
    for wort in ("Konfiguration", "Parameterwahl", "Verfahrenswahl", "Wahlregeln"):
        assert wort not in TX.TITEL and wort not in QUELLEN["README.md"].splitlines()[0]
    for wort in ("hub", "sso", "prod", "ass", "qa"):
        for pfad in list(ROOT.glob("*.py")) + list(ROOT.glob("tests/*.py")) + list(ROOT.glob("tools/*.py")):
            assert wort not in pfad.stem, (pfad.name, wort)


def test_jede_display_gleichung_hat_einen_eigenen_block():
    """Je Display-Gleichung ein eigener $$-Block: kein Block enthält zwei Gleichungen, die Zahl der $$ ist gerade."""
    assert TX.FORMELN.count("$$") % 2 == 0
    for block in re.findall(r"\$\$(.*?)\$\$", TX.FORMELN, flags=re.S):
        assert "\n\n" not in block.strip()
