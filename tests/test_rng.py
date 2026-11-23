"""Zufallsquelle: zwl_rng.py ist die unveränderte Vorlage, Blockstrom und Ziehungen sind bitgleich zum skalaren Strom, die Quellen der Kopien sind unverändert."""

import math
from pathlib import Path

import numpy as np
import pytest

import zwl_constants as C
import zwl_rngvec as RV
import zwl_scenario as S
from zwl_rng import SplitMix64, exact_log
from zwl_rngblock import BlockSplitMix64, uniform_block

ROOT = Path(__file__).resolve().parent.parent
VORLAGE = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
QUELLE = ROOT.parent / "hyperheuristik-demo"


def _gleich(a: Path, b: Path, ersetze: bool = False) -> bool:
    ta = a.read_bytes().replace(b"\r\n", b"\n")
    tb = b.read_bytes().replace(b"\r\n", b"\n")
    return ta == (tb.replace(b"hhw_", b"zwl_") if ersetze else tb)


def test_kopie_der_vorlage_ist_unveraendert():
    if not VORLAGE.exists():
        pytest.skip("Vorlage nicht vorhanden (CI ohne Website-Repo)")
    assert _gleich(ROOT / "zwl_rng.py", VORLAGE)


def test_kopie_rngblock_gleicht_der_quelle_nach_ersetzung():
    if not (QUELLE / "hhw_rngblock.py").exists():
        pytest.skip("Quelle hyperheuristik-demo nicht vorhanden")
    assert _gleich(ROOT / "zwl_rngblock.py", QUELLE / "hhw_rngblock.py", ersetze=True)


def test_kopie_scenario_gleicht_der_quelle_nach_ersetzung():
    if not (QUELLE / "hhw_scenario.py").exists():
        pytest.skip("Quelle hyperheuristik-demo nicht vorhanden")
    assert _gleich(ROOT / "zwl_scenario.py", QUELLE / "hhw_scenario.py", ersetze=True)


def test_splitmix64_bekannte_werte():
    """Referenzwerte von SplitMix64 mit Zufalls-Seed 0 (Vigna, splitmix64.c): die ersten drei Ausgaben."""
    r = SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


@pytest.mark.parametrize("seed", [0, 1, 12345, (1 << 64) - 1, 0x9E3779B97F4A7C15])
def test_blockstrom_ist_bitgleich_zum_skalaren_strom(seed):
    skalar = SplitMix64(seed)
    werte = [skalar.random() for _ in range(300)]
    assert uniform_block(seed, 0, 300).tolist() == werte
    assert uniform_block(seed, 100, 200).tolist() == werte[100:300]
    block = BlockSplitMix64(seed, block=64)
    assert [block.random() for _ in range(300)] == werte


@pytest.mark.parametrize("seed", [0, 7, 123456789])
def test_ustream_ist_bitgleich_und_fuellt_in_c_reihenfolge(seed):
    skalar = SplitMix64(seed)
    zahlen = [skalar.random() for _ in range(12 + 30)]
    us = RV.UStream(seed)
    u = us.uniform((3, 4, 1))
    assert u[1, 2, 0] == zahlen[6]
    assert u[:, :, 0].reshape(-1).tolist() == zahlen[:12]
    assert us.position == 12
    assert us.uniform(30).tolist() == zahlen[12:42]         # der Strom läuft fortlaufend weiter
    assert us.position == 42


def test_ustream_skip_und_leere_form():
    us = RV.UStream(5)
    us.skip(10)
    assert us.uniform(()).shape == () and us.position == 11
    assert us.uniform((0, 3)).shape == (0, 3) and us.position == 11
    assert us.uniform(2).tolist() == uniform_block(5, 11, 2).tolist()


def test_normal_array_gleicht_dem_skalaren_box_muller():
    for seed in (1, 99):
        skalar = SplitMix64(seed)
        soll = np.array([skalar.normal() for _ in range(50)])
        ist = RV.normal_array(RV.UStream(seed), (50,))
        assert np.allclose(ist, soll, rtol=0, atol=1e-12)
    us = RV.UStream(3)
    RV.normal_array(us, (4, 5))
    assert us.position == 40                                  # zwei Gleichverteilte je Normalzahl


def test_gumbel_array_gleicht_der_skalaren_formel():
    skalar = SplitMix64(11)
    soll = [-exact_log(-exact_log(max(skalar.random(), 1e-12))) for _ in range(40)]
    ist = RV.gumbel_array(RV.UStream(11), (40,))
    assert np.allclose(ist, soll, rtol=1e-12, atol=1e-12)


def test_random_order_und_choice_sind_gueltige_ziehungen():
    us = RV.UStream(4)
    o = RV.random_order(us, 50, 9)
    assert o.shape == (50, 9) and np.all(np.sort(o, axis=1) == np.arange(9))
    assert len({tuple(z) for z in o}) > 40                    # verschiedene Anordnungen
    c = RV.choice_without_replacement(us, 30, 8, 3)
    assert c.shape == (30, 3) and all(len(set(z)) == 3 and 0 <= min(z) and max(z) < 8 for z in c)
    with pytest.raises(ValueError):
        RV.choice_without_replacement(us, 1, 3, 4)
    # deterministisch: gleicher Zufalls-Seed, gleiche Ziehung
    assert np.array_equal(RV.random_order(RV.UStream(4), 5, 6), RV.random_order(RV.UStream(4), 5, 6))


def test_uniform_array_ist_gleichverteilt_ueber_viele_zahlen():
    u = RV.uniform_array(RV.UStream(2), (200000,))
    assert abs(u.mean() - 0.5) < 0.005 and abs(u.std() - math.sqrt(1 / 12)) < 0.005 and 0 <= u.min() and u.max() < 1


def test_derive_ist_fest_und_trennt_zwecke_und_schluessel():
    a = S.derive(11, 0, 30, 0)
    assert a == S.derive(11, 0, 30, 0)
    andere = {S.derive(11, 0, 30, 1), S.derive(11, 1, 30, 0), S.derive(11, 0, 60, 0), S.derive(33, 0, 30, 0), S.derive(11, 0, 30, 0, 0)}
    assert a not in andere and len(andere) == 5


def test_zwecke_der_demo_liegen_im_reservierten_bereich_und_sind_verschieden():
    eigene = [C.ZWECK_Z_INIT, C.ZWECK_Z_SAMMEL, C.ZWECK_Z_ZEILEN, C.ZWECK_Z_BATCH, C.ZWECK_K_INIT, C.ZWECK_K_INST, C.ZWECK_K_START, C.ZWECK_K_ZIEHEN,
              C.ZWECK_SUCHE, C.ZWECK_E0, C.ZWECK_BOOT, C.ZWECK_E5]
    assert len(set(eigene)) == len(eigene) and all(100 <= z < 150 for z in eigene)
    assert (C.ZWECK_INSTANZ, C.ZWECK_START, C.ZWECK_LAUF) == (11, 33, 24)           # Werte der Quelle bleiben unberührt
    assert not set(eigene) & {11, 24, 33}
