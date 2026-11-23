"""Kopier-Tests: die kopierten Dateien gleichen ihren Quellen nach dem Ersetzen des Präfixes (übersprungen, wenn die Quelle fehlt, z. B. in der CI).

Quellen: zeigernetz-demo (Stück 1: rngvec, instances, exact, stats; rng und rngblock stammen über Stück 1 aus Vorlage bzw. hyperheuristik-demo, siehe test_rng.py), modul_frisch.py."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
QUELLE_1 = ROOT.parent / "zeigernetz-demo"
QUELLE_HHW = ROOT.parent / "hyperheuristik-demo"


def _norm(p: Path) -> bytes:
    return p.read_bytes().replace(b"\r\n", b"\n")


@pytest.mark.parametrize("name", ["rng", "rngblock", "rngvec", "scenario", "instances", "exact", "stats"])
def test_kopie_aus_stueck_1_gleicht_der_quelle(name):
    quelle = QUELLE_1 / f"znz_{name}.py"
    if not quelle.exists():
        pytest.skip("zeigernetz-demo nicht vorhanden")
    assert _norm(ROOT / f"zwl_{name}.py") == _norm(quelle).replace(b"znz_", b"zwl_")


def test_modul_frisch_gleicht_der_quelle():
    quelle = QUELLE_1 / "modul_frisch.py"
    if not quelle.exists():
        pytest.skip("zeigernetz-demo nicht vorhanden")
    assert _norm(ROOT / "modul_frisch.py") == _norm(quelle)


def test_scenario_gleicht_hyperheuristik_demo_nach_ersetzung():
    quelle = QUELLE_HHW / "hhw_scenario.py"
    if not quelle.exists():
        pytest.skip("hyperheuristik-demo nicht vorhanden")
    assert _norm(ROOT / "zwl_scenario.py") == _norm(quelle).replace(b"hhw_", b"zwl_")


def test_formeln_der_zaehlung_gleichen_stueck_1():
    quelle = QUELLE_1 / "znz_cost.py"
    if not quelle.exists():
        pytest.skip("zeigernetz-demo nicht vorhanden")
    import importlib.util

    spec = importlib.util.spec_from_file_location("znz_cost_quelle", quelle)
    import sys

    sys.path.insert(0, str(QUELLE_1))
    try:
        modul = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modul)
    finally:
        sys.path.remove(str(QUELLE_1))
        sys.modules.pop("znz_constants", None)
    import zwl_cost as K

    for n in (5, 6, 10, 20, 50):
        assert K.k_konstruktion(n) == modul.k_konstruktion(n) and K.m_verschiedene_zuege(n) == modul.m_verschiedene_zuege(n)
    assert K.ops_kandidaten(190) == modul.ops_kandidaten(190) == 1520
