"""Verdikte als reine Funktionen: Schwellenränder (genau 0,50, 0,25, 0,10, 0,05, 0,01), jede Zahl im Text, jede Kombination der Marken, Marken des gepaarten Vergleichs."""

import itertools

import numpy as np
import pytest

import zwl_constants as C
import zwl_evaluation as V


# ---------------------------------------------------------------------------------------------------------------- Formate
def test_zahlenformate():
    assert V.de(3.14159, 2) == "3,14" and V.de_vz(2.5, 1) == "+2,5" and V.de_vz(-0.25) == "-0,25"
    assert V.zahl(12875) == "12 875" and V.zahl(1850) == "1 850" and V.zahl(0) == "0"
    assert V.prozent(0.534) == "53" and V.prozent(0.0153, 1) == "1,5"
    assert V.zahl_k(26888) == "26 888" and V.zahl_k(2449128) == "2,45 Mio."


# ---------------------------------------------------------------------------------------------------------------- Marken
def _cmp(v, lo, hi):
    from zwl_stats import klasse
    return {"v": v, "klasse": klasse(v, lo, hi)}


def test_marken_klassen():
    assert V.marke(_cmp(-0.33, -0.65, -0.08)) == "schlägt"
    assert V.marke(_cmp(0.33, 0.08, 0.65)) == "verliert"
    assert V.marke(_cmp(-0.33, -0.65, 0.2)) == "gleich"                    # Intervall mit 0
    assert V.marke(_cmp(-0.05, -0.08, -0.02)) == "gleich"                  # unter 0,10
    assert V.marke(_cmp(-0.15, -0.2, -0.1)) == "dazwischen" and V.marke(_cmp(0.15, 0.1, 0.2)) == "dazwischen"
    assert V.marke(_cmp(-0.25, -0.3, -0.2)) == "schlägt"                   # genau 0,25 zählt als klar


def test_vergleich_zweier_bloecke_auf_gebauten_zahlen():
    ref = np.array([100.0] * 10)
    regel = np.full((10, 3, 2, 3), 101.0)                                  # Lücke 1 %
    gegner = np.full((10, 3, 2, 3), 102.0)                                 # Lücke 2 %
    gegner[:, :, :, 1] = 100.5
    r = V.vergleiche_bloecke(regel, gegner, ref, 0, 0)
    assert r["v"] == pytest.approx(-1.0) and r["marke"] == "schlägt" and r["luecke_regel"] == pytest.approx(1.0) and r["luecke_gegner"] == pytest.approx(2.0)
    assert V.vergleiche_bloecke(regel, gegner, ref, 1, 1)["marke"] == "verliert"
    assert V.vergleiche_bloecke(regel, regel, ref, 0, 0)["marke"] == "gleich"


# ---------------------------------------------------------------------------------------------------------------- E1
@pytest.mark.parametrize("d,art,wort", [(0.0501, "success", "klar besser"), (0.05, "success", "klar besser"), (0.0499, "info", "etwas besser"), (0.01, "info", "etwas besser"),
                                        (0.0099, "warning", "nicht besser"), (-0.1, "warning", "nicht besser")])
def test_verdict_guete_schwellen(d, art, wort):
    a, t = V.verdict_guete(0.9 + d, 0.9, 171.7, 8.5, 20)
    assert a == art and wort in t


def test_verdict_guete_nennt_immer_beide_zahlen_und_die_voraussetzung():
    for netz in (0.973, 0.91, 0.88):
        a, t = V.verdict_guete(netz, 0.876, 171.7, 8.5, 20)
        assert V.de(netz, 3) in t and "0,876" in t and "8,5" in t and "171,7" in t and "n = 20" in t and "n = 20 trainiert" in t
    assert "Nachbarlisten-Reihenfolge" in V.verdict_guete(0.973, 0.876, 171.7, 8.5, 20)[1]


# ---------------------------------------------------------------------------------------------------------------- E2
@pytest.mark.parametrize("anteil,start", [(0.534, "Mehr als die Hälfte"), (0.5, "Mehr als die Hälfte"), (0.4999, "Zwischen einem Viertel und der Hälfte"),
                                          (0.25, "Zwischen einem Viertel und der Hälfte"), (0.2499, "Weniger als ein Viertel"), (0.0, "Weniger als ein Viertel")])
def test_verdict_beweis_schwellen(anteil, start):
    a, t = V.verdict_beweis(anteil, 20, 20)
    assert t.startswith(start) and f"({V.prozent(anteil)} %)" in t and f"{V.prozent(1 - anteil)} %" in t and "n = 20" in t and "20 Instanzen" in t


def test_verdict_beweis_prozente_summieren_sich_auf_hundert():
    for anteil in (0.534, 0.4, 0.123, 0.987):
        _, t = V.verdict_beweis(anteil, 50)
        p, q = int(V.prozent(anteil)), int(V.prozent(1 - anteil))
        assert p + q == 100 and f"({p} %)" in t and f"{q} %" in t


# ---------------------------------------------------------------------------------------------------------------- E3
def _t(marken):
    return V.verdict_zugwahl(marken, 20, "gleich", 2, 20, 10)


def test_verdict_zugwahl_fuenf_faelle():
    art, t = _t({"frei": "schlägt", "K": "schlägt", "O": "schlägt"})
    assert art == "success" and "auch bei Zählung der Operationen" in t and "belastbar" in t
    art, t = _t({"frei": "schlägt", "K": "schlägt", "O": "gleich"})
    assert art == "info" and "gewinnt in Kandidatenbewertungen (K)" in t and "(gleich)" in t and "zwischen K und O" in t
    art, t = _t({"frei": "schlägt", "K": "verliert", "O": "verliert"})
    assert art == "info" and "Nur wenn die Merkmalszeilen nichts kosten" in t and "schon in K nicht mehr (verliert)" in t and "K das Netz begünstigt" in t
    for m in ("gleich", "dazwischen"):
        art, t = _t({"frei": m, "K": "verliert", "O": "verliert"})
        assert art == "warning" and "nicht besser als Zufall" in t
    art, t = _t({"frei": "verliert", "K": "verliert", "O": "verliert"})
    assert art == "warning" and "schlechter als Zufall" in t


def test_verdict_zugwahl_jede_kombination_hat_voraussetzung_und_zaehlung():
    for frei, k, o in itertools.product(("schlägt", "gleich", "dazwischen", "verliert"), repeat=3):
        art, t = _t({"frei": frei, "K": k, "O": o})
        assert art in ("success", "info", "warning") and "n = 20" in t and "Gleichverteilt" in t and "Budget k = 2" in t and "20 Instanzen mit je 10 Zufalls-Seeds" in t
        assert "Zugnetz bei n = 20 trainiert" in t and "Zufall" in t


# ---------------------------------------------------------------------------------------------------------------- E4
@pytest.mark.parametrize("anteil,teil", [(0.25, "kein Beweis mehr"), (0.2499, "Näherungsbeweis"), (0.05, "Näherungsbeweis"), (0.0499, "fast nie")])
def test_verdict_abbruch_schwellen_und_zweiter_satz(anteil, teil):
    art, t = V.verdict_abbruch(anteil, {"frei": "schlägt", "K": "verliert", "O": "verliert"}, 100, 50, "gleich", 2, 10, 5)
    assert teil in t and f"{V.prozent(anteil)} %" in t and "nach 100 Kandidaten" in t
    assert "Nur wenn die Merkmalszeilen nichts kosten" in t                                   # zweiter Satz nach verdict_zugwahl


def test_verdict_nachbarliste_faelle():
    art, t = V.verdict_nachbarliste(718, 1850, 0.0, 0.33, 3)
    assert art == "success" and "von 1 850 auf 718" in t and "ohne Qualitätsverlust" in t and "+0,33 Prozentpunkte" in t and "Don't-Look-Bits" in t
    art, t = V.verdict_nachbarliste(718, 1850, 0.10, 0.1, 3)
    assert art == "warning" and "schlechterer Endtouren" in t and "10 %" in t                 # genau 0,10 nicht mehr unter der Schwelle
    art, t = V.verdict_nachbarliste(718, 1850, 0.0, 0.5, 3)
    assert art == "warning" and "schlechterer Endtouren" in t
    art, t = V.verdict_nachbarliste(718, 1850, 0.099, 0.499, 3)
    assert art == "success"
    art, t = V.verdict_nachbarliste(925, 1850, 0.0, 0.0, 8)                                   # genau die Hälfte: spart nur
    assert art == "warning" and "spart damit nur 50 %" in t


def test_verdict_liste_gelernt():
    assert "verkürzt auch" in V.verdict_liste_gelernt({"frei": "schlägt", "K": "verliert", "O": "verliert"}, 3, 50)[1]
    assert "in K ebenfalls" in V.verdict_liste_gelernt({"frei": "schlägt", "K": "schlägt", "O": "verliert"}, 3, 50)[1]
    art, t = V.verdict_liste_gelernt({"frei": "gleich", "K": "verliert", "O": "verliert"}, 3, 50)
    assert art == "warning" and "bringt die gelernte Reihenfolge nichts" in t and "(gleich)" in t and "Nachbarliste k = 3" in t


# ---------------------------------------------------------------------------------------------------------------- E5 und E0
def test_verdict_start_und_fehler():
    art, t = V.verdict_start(5.3, 47.0, 190)
    assert "höchstens 47 %" in t and "das Netz als Start 5 %" in t and "190 Kandidatenbewertungen" in t
    with pytest.raises(ValueError):
        V.verdict_start(47.0, 47.0, 190)
    with pytest.raises(ValueError):
        V.verdict_start(50.0, 47.0, 190)


def test_verdict_probe_k():
    assert V.verdict_probe_k(19.0, 18.5)[0] == "warning" and "Kaum gelernt" in V.verdict_probe_k(19.0, 18.5)[1]
    assert V.verdict_probe_k(18.5, 18.5)[0] == "warning"
    art, t = V.verdict_probe_k(12.0, 18.5)
    assert art == "success" and "Besser als Nächster Nachbar" in t and "12,0 %" in t and "18,5 %" in t


def test_zeile_e0_handbeispiel_und_varianten():
    t = V.verdict_instanz_e0(1, 7, 93)
    assert "Angenommene Züge: 1" in t and "93 Zugauswertungen" in t and "93 % von 100" in t
    assert "Abbruch nach 50" in V.verdict_instanz_e0(3, 100, 50, abbruch=50, falsch=True) and "kein Lokalminimum" in V.verdict_instanz_e0(3, 100, 50, abbruch=50, falsch=True)
    assert "Nachbarliste (k = 3)" in V.verdict_instanz_e0(3, 100, 718, liste=3)
    assert V.verdict_instanz_e0(0, 0, 0).startswith("Angenommene Züge: 0")                            # Nenner 0: kein Fehler


def test_trainingszaehlung():
    assert V.training_zeilen(1500) == 1500 * 4096 == 6144000 and V.training_touren_k(150) == 38400 and C.K_SCHRITTE_FROZEN * C.K_BATCH_TOUREN == 960000
