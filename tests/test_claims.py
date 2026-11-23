"""Jede Zahl der README gegen die eingefrorene Messung (zwl_frozen.json), die Referenzen (zwl_refs.json) und die Module der Demo.

„Exakt aus der Datei“: die Auswertung der gespeicherten Längen ergibt genau die Zeile der README (`tests/claims_zeilen.py` baut sie). Zahlen, die live neu gerechnet werden (AUC,
Zugauswertungen je Zug, Anteil der Prüfrunde, Startlösungen), stehen als Band: die README nennt den Wert der Messung dieses Standes, der Test rechnet neu und prüft ein Band um ihn
(Plattformen rechnen im letzten Bit anders). Zahlen ohne Test stehen in der README als „Messung dieser README, nicht im Test“ (Zeiten der Cloud-Näherung, Spalte „Plan“).
Die Datei selbst rechnet tests/test_frozen.py an Ausschnitten als Band neu; die Zahlen des Handbeispiels stehen exakt in tests/test_moves.py, tests/test_search.py und tests/test_handbeispiele.py."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from claims_zeilen import (
    e3_andere_regeln_zeilen,
    e3_pruef_anteil_zeilen,
    e3_zeilen,
    e4a_zeilen,
    e4b_abstieg_zeilen,
    e4b_ils_zeilen,
    e5_zeilen,
    zaehlungen,
)

import zwl_constants as C
import zwl_construct as CO
import zwl_cost as K
import zwl_evaluation as V
import zwl_experiments as X
import zwl_mlp as MLP
import zwl_moves as M
import zwl_texts as TX
import zwl_zugnetz as Z

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
F = V.load_frozen()
R = V.load_refs()
de, zahl = V.de, V.zahl


def steht(*teile):
    for teil in teile:
        assert teil in README, f"nicht in der README: {teil}"


def zeilen_stehen(zeilen):
    assert zeilen
    for z in zeilen:
        assert z in README, f"Zeile nicht in der README: {z}"


# ------------------------------------------------------------------------------------------------------------- Experiment 1
def test_befund_e1_guete():
    Fv, yv, ns = Z.validierungszeilen(C.Z_VAL_INSTANZEN_FROZEN)
    tabs = [Z.auc_tabelle(Fv, yv, X.lade_modell("Z", s)[0]) for s in range(C.MODELL_SEEDS)]
    assert all(0.975 <= t["netz"] <= 0.981 for t in tabs)                                        # Band um 0,978 (Plan 0,973)
    assert 0.876 <= tabs[0]["hand"] <= 0.882 and all(0.76 <= tabs[0][k] <= 0.86 for k in Z.EINZELMERKMALE)
    assert (len(yv), ns) == (197600, 988) and 0.014 <= yv.mean() <= 0.017
    steht("AUC 0,978 gegen 0,879", "0,978 (fünf Gewichts-Zufalls-Seeds: 0,978 bis 0,978)", "0,879; 0,77 bis 0,85", "988 Zustände, 197 600 Zeilen, 1,5 % verbessernde Züge",
          "mittlere entfernte Länge 0,84, größte 0,82, kleinster Rang 0,85, größter Rang 0,77")
    assert round(tabs[0]["hand"], 3) == 0.879 and round(tabs[0]["netz"], 3) == 0.978
    assert [round(tabs[0][k], 2) for k in Z.EINZELMERKMALE] == [0.84, 0.82, 0.85, 0.77]
    assert tabs[0]["netz"] - tabs[0]["hand"] >= C.AUC_KLAR                                         # Verdikt „klar besser“
    for s in range(C.MODELL_SEEDS):                                                               # Gütegrenze der gespeicherten Modelle (Metadaten der Erzeugung)
        assert 0.975 <= float(X.lade_modell("Z", s)[1]["val_auc"]) <= 0.981


def test_befund_e1_je_zug():
    netz = X.lade_modell("Z", 0)[0]
    for n, count, soll in ((20, 20, (185.2, 41.5, 12.0, 10.2)), (50, 10, (900.8, 188.5, 37.9, 32.4))):
        d = X.e1_je_zug(netz, "gleich", n, count)
        ist = tuple(d[r]["je_zug"] for r in ("zufall", "nachbarliste", "gelernt", "gelernt_det"))
        assert all(abs(i - s) <= 0.2 * s for i, s in zip(ist, soll)), (n, ist)                    # Band ±20 % um die Messung dieser README
        assert d["steilster"]["je_zug"] == M.neighborhood_size(n)
        assert ist[0] > 4 * ist[1] and ist[1] > 2.5 * ist[2]                                                 # Zufall > Nachbarlisten-Reihenfolge > gelernt, mit Abstand
        steht(*(de(s, 1) for s in soll))
    steht("1 850", "12 875")


def test_befund_e1_zugarten():
    z = X.e1_zugarten(X.lade_modell("Z", 0)[0], "gleich", 20, 20)
    assert [round(100 * a) for a in z["anteil_m"]] == [9, 19, 37, 35]
    assert all(abs(100 * a - s) <= 2.0 for a, s in zip(z["anteil_top"], (22, 32, 30, 15))) and z["zustaende"] == 173
    steht("2-opt 22 % (Anteil an M: 9 %), Or-opt 1: 32 % (19 %), Or-opt 2: 30 % (37 %), Or-opt 3: 15 % (35 %)", "173 Zustände")


# ------------------------------------------------------------------------------------------------------------- Experiment 2
def test_befund_e2_pruefrunde():
    soll = {10: (512, 128, 73.2, 536, 84.9), 20: (3446, 1070, 53.7, 3701, 59.7), 30: (9510, 2583, 46.5, 10874, 48.4), 40: (18505, 5364, 43.8, 22740, 42.3), 50: (32914, 8386, 39.1, 44421, 34.4)}
    for n in C.E2_N_LISTE:
        a, b = X.e2_beweis("gleich", n, 20), X.e2_beweis("gleich", n, 20, True)
        mittel, sd, anteil, mittel_alt, anteil_alt = soll[n]
        assert (a["M"], b["M"]) == (M.neighborhood_size(n), M.neighborhood_size(n, True))
        assert abs(a["mittel"] - mittel) <= 0.05 * mittel and abs(100 * a["anteil"] - anteil) <= 2.5 and abs(b["mittel"] - mittel_alt) <= 0.05 * mittel_alt and abs(100 * b["anteil"] - anteil_alt) <= 2.5
        steht(f"| {n} | {zahl(a['M'])} | {zahl(mittel)} ({zahl(sd)}) | {de(anteil, 1)} % | {zahl(b['M'])} | {zahl(mittel_alt)} | {de(anteil_alt, 1)} % |")
    steht("1 850 bei n = 20, 12 875 bei n = 50", "53,7 % bzw. 39,1 %")


def test_befund_e2_kicks():
    zeilen_stehen(e3_pruef_anteil_zeilen())
    mit_zeilen = {zelle: [V.e3_pruef_anteil(F, zelle, r) for r in ("nachbarliste", "gelernt", "gelernt_det")] for zelle in F["e3"]}
    assert (round(100 * min(mit_zeilen["n20_gleich"])), round(100 * max(mit_zeilen["n20_gleich"]))) == (94, 99)
    assert (round(100 * min(mit_zeilen["n50_gleich"])), round(100 * max(mit_zeilen["n50_gleich"]))) == (82, 98)
    assert (round(100 * min(min(v) for v in mit_zeilen.values())), round(100 * max(max(v) for v in mit_zeilen.values()))) == (59, 99)
    steht("94 bis 99 % (n = 20) bzw. 82 bis 98 % (n = 50 gleichverteilt)", "59 bis 99 % (Regeln mit Zeilen, alle Zellen; Gruppen n = 50 am niedrigsten)")


# ------------------------------------------------------------------------------------------------------------- Experiment 3
def test_befund_e3_hauptvergleich():
    zeilen_stehen(e3_zeilen())
    for zelle, z in F["e3"].items():
        assert z["zufalls_seeds"] == 10 and z["instanzen"] == (20 if zelle == "n20_gleich" else 10) and z["gewichts_seeds"] == (5 if zelle == "n20_gleich" else 3)
    streu = V.e3_gewichts_streuung(F, "n20_gleich", "gelernt", 2, "K", R)
    assert (round(min(streu), 2), round(max(streu), 2)) == (10.97, 13.62)
    steht("10,97 bis 13,62 %")
    # Erwartung der Planung als Marken: Satz (3) bei n = 20 und k = 2, Satz (4) bei k = 8 und 32; in K überall nicht besser
    m = {k: V.e3_marken(F, "n20_gleich", "gelernt", k, "zufall", R) for k in (2, 8, 32)}
    assert m[2]["frei"] == "schlägt" and m[8]["frei"] == m[32]["frei"] == "gleich" and all(m[k]["K"] != "schlägt" for k in m)
    art, text = V.verdict_zugwahl(m[2], 20, "gleich", 2, 20, 10)
    assert "Nur wenn die Merkmalszeilen nichts kosten" in text


def test_befund_e3_zaehlung():
    z = zaehlungen()
    assert z == {"zellen": 7, "frei": {"schlägt": 3, "gleich": 4, "dazwischen": 0, "verliert": 0}, "K": {"schlägt": 0, "gleich": 0, "dazwischen": 1, "verliert": 6},
                 "O": {"schlägt": 0, "gleich": 0, "dazwischen": 0, "verliert": 7}}
    steht("nur Zugauswertungen: 3-mal schlägt, 4-mal gleich; K: 6-mal verliert, 1-mal dazwischen; O: 7-mal verliert", "in 3 von 7 Zellen", "in 4 von 7 gleich gut",
          "In K (Zeilen zählen wie Zugauswertungen) verliert es in 6 von 7 Zellen", "(+0,12 Prozentpunkte)", "in O (grobe Operationen) verliert es in 7 von 7 Zellen")
    assert V.e3_vergleich(F, "n20_gleich", "gelernt", 32, "K", "zufall", R)["v"] == pytest.approx(0.12, abs=0.005)


def test_befund_e3_o_ohne_einen_schritt():
    assert C.SICHT_GEWICHT_Z["O"] == 22 and C.O_JE_ZEILE_Z == 176 and C.OPS_JE_KANDIDAT == 8
    ohne_schritt, nn_luecken = 0, {}
    for zelle, z in F["e3"].items():
        ref = V.e3_ref(F, zelle, R)
        nn = np.mean([100 * (M.tour_length(X.kontext(z["familie"], z["n"], s).D, CO.nn_tour_np(X.kontext(z["familie"], z["n"], s).D, 0)) - ref[s]) / ref[s] for s in range(z["instanzen"])])
        nn_luecken[zelle] = nn
        for k in z["ks"]:
            if k < 22:                                                                            # ein Schritt bewertet M Zeilen = 22 M Zugauswertungen in O
                assert V.e3_luecken(F, zelle, "gelernt", k, "O", R) == pytest.approx(nn, abs=2e-3), (zelle, k)
                ohne_schritt += 1
            else:
                assert V.e3_luecken(F, zelle, "gelernt", k, "O", R) < nn - 1.0
    assert ohne_schritt == 6
    assert [round(nn_luecken[z], 2) for z in ("n20_gleich", "n50_gleich", "n50_cluster")] == [16.68, 21.57, 20.88]
    steht("in 6 von 7 Zellen nicht für einen Schritt des Netzes", "22 · M Zugauswertungen", "16,68 % bei n = 20, 21,57 % bei n = 50 gleichverteilt, 20,88 % in Gruppen")


def test_befund_e3_andere_regeln():
    zeilen_stehen(e3_andere_regeln_zeilen())


# ------------------------------------------------------------------------------------------------------------- Experiment 4
def test_befund_e4a_abbruch():
    zeilen_stehen(e4a_zeilen())
    z20, z50 = F["e4a"]["n20_gleich"], F["e4a"]["n50_gleich"]
    assert (z20["instanzen"], z20["zufalls_seeds"], z20["ks"], z50["instanzen"], z50["zufalls_seeds"], z50["ks"]) == (20, 10, [2, 8], 10, 5, [2])
    f = {(zz, r): round(100 * V.e4a_falsch_anteil(F, zz, 50, r)) for zz in ("n20_gleich", "n50_gleich") for r in C.REGELN_E4A}
    assert f[("n20_gleich", "zufall")] == 99 and f[("n50_gleich", "zufall")] == 100 and f[("n20_gleich", "gelernt")] == 7 and f[("n50_gleich", "gelernt")] == 46
    steht("99 % (n = 20) bzw. 100 % (n = 50) der Abstiegsenden in der Suche mit Kicks (Zufallsreihenfolge)", "beim Netz sind es 7 % bzw. 46 %")
    # mit Abbruch gewinnt das Netz in „nur Zugauswertungen“ gegen Zufall, in K und O nicht (in den neun Zellen: frei 9-mal schlägt oder dazwischen, K nie schlägt)
    for zz, z in F["e4a"].items():
        for g in C.E4A_G:
            for k in z["ks"]:
                assert V.e4a_vergleich(F, zz, g, "gelernt", k, "frei", "zufall", R)["v"] < 0
                assert V.e4a_vergleich(F, zz, g, "gelernt", k, "K", "zufall", R)["marke"] != "schlägt"


def test_befund_e4b_nachbarliste():
    zeilen_stehen(e4b_abstieg_zeilen())
    zeilen_stehen(e4b_ils_zeilen())
    a20, a50 = V.e4b_abstieg(F, "n20_gleich", 3, "zufall"), V.e4b_abstieg(F, "n50_gleich", 3, "zufall")
    assert (round(a20["pruef"]), round(a50["pruef"])) == (648, 3057) and a20["M"] == 1850 and a50["M"] == 12875
    assert round(100 * a50["nicht_voll"]) == 18 and round(100 * a20["nicht_voll"], 1) == 3.5
    steht("von 1 850 auf 648", "von 12 875 auf 3 057", "enden dann aber 18 % der Abstiege nicht im vollen Lokalminimum")
    for zz in ("n20_gleich", "n50_gleich"):
        assert V.e4b_abstieg(F, zz, 3, "zufall")["pruef"] < V.e4b_abstieg(F, zz, 5, "zufall")["pruef"] < V.e4b_abstieg(F, zz, 8, "zufall")["pruef"] < F["e4b"][zz]["M"]
    art, text = V.verdict_nachbarliste(a50["pruef"], a50["M"], a50["nicht_voll"], a50["laenge_gegen_voll"], 3)
    assert "schlechterer Endtouren" in text                                                       # n = 50, k = 3: nicht ohne Qualitätsverlust


# ------------------------------------------------------------------------------------------------------------- Experiment 5
def test_befund_e5_startloesungen():
    zeilen_stehen(e5_zeilen())
    for n, count, (zufall, farthest, netz, perfekt) in ((20, 20, (-4, 26.5, 10.1, 46.7)), (50, 10, (-21, 26.9, 9.9, 61.4))):
        z = {r["art"]: r for r in X.e5_starts("gleich", n, count, X.lade_modell("K", 0)[0], V.refs_vektor("gleich", n, count, R))}
        assert abs(z["zufall"]["ersparnis"] - zufall) <= 1.5 and abs(z["farthest"]["ersparnis"] - farthest) <= 1.5 and abs(z["netz_k"]["ersparnis"] - netz) <= 1.5
        assert abs(z["optimum"]["ersparnis"] - perfekt) <= 1.5 and z["optimum"]["ersparnis"] > z["farthest"]["ersparnis"] > z["netz_k"]["ersparnis"] > 0 and z["nn"]["ersparnis"] == 0
        assert z["optimum"]["ev_pruef"] == z["optimum"]["ev_gesamt"] == M.neighborhood_size(n) and z["nn"]["k_start"] == K.k_konstruktion(n)
    steht("höchstens 47 % (n = 20) bzw. 61 % (n = 50)", "spart 10 % bzw. 10 %, Farthest Insertion 27 % bzw. 27 %", "190 bzw. 1 225")


# ------------------------------------------------------------------------------------------------------------- Training und Zählung
def test_befund_modellguete():
    luecken = [float(X.lade_modell("K", s)[1]["val_gap_greedy"]) for s in range(C.MODELL_SEEDS)]
    assert [round(x, 2) for x in luecken] == [9.02, 7.95, 8.41, 7.48, 8.24]
    steht("9,02 / 7,95 / 8,41 / 7,48 / 8,24 % gegen 18,52 %")
    for s in range(C.MODELL_SEEDS):
        meta = X.lade_modell("K", s)[1]
        assert int(meta["n_train_tours"]) == 960000 == C.K_SCHRITTE_FROZEN * C.K_BATCH_TOUREN and int(meta["steps"]) == 3750 and int(meta["val_instances"]) == 50
        assert int(meta["n_train_rows"]) == 960000 * K.zeilen_featnet(20) == 182400000
        z = X.lade_modell("Z", s)[1]
        assert int(z["steps"]) == 3000 and int(z["n_train_rows"]) == 3000 * 4096 and int(z["n_train_instances"]) == 300


def test_befund_trainingszahlen():
    F_, y_, ns = Z.sammle_zustaende(Z.train_instanzen(C.Z_INSTANZEN_FROZEN))
    assert (len(y_), ns) == (2311600, 11558) and 3000 * 4096 == 12288000 and 1500 * 4096 == 6144000
    steht("2 311 600 Zeilen aus 11 558 Zuständen", "12,3 Mio. Zeilen", "6,1 Mio. Zeilen", "182,4 Mio. Zeilen", "960 000 Touren", "150 Schritte = 38 400 Touren")
    assert C.K_SCHRITTE_PROBE * C.K_BATCH_TOUREN == 38400 and 960000 * K.zeilen_featnet(20) == 182400000


def test_befund_zaehlung_k_und_o():
    assert (K.k_konstruktion(20), K.k_konstruktion(50)) == (190, 1225) and (K.m_verschiedene_zuege(20), K.m_verschiedene_zuege(50)) == (1850, 12875)
    assert (C.O_JE_ZEILE_Z, C.O_JE_ZEILE_K, C.OPS_JE_KANDIDAT, C.O_JE_ZEILE_HAND) == (176, 144, 8, 8) and (C.Z_PARAMS, C.K_PARAMS) == (177, 145)
    assert (M.neighborhood_size(20, True), M.neighborhood_size(50, True)) == (2210, 15275)
    steht("n(11n − 35)/2", "1 850 bei n = 20 und 12 875 bei n = 50", "2 210 und 15 275", "176 je Zeile des Zugnetzes", "177 Parametern", "145 Parameter")


def test_cp_sat_referenzen_bewiesen():
    bewiesen = sum(r["bewiesen"] for fam in C.FAMILIEN for r in R["refs"][fam]["50"])
    assert bewiesen == 28
    gemessen = sum(r["bewiesen"] for fam in ("gleich", "cluster") for r in R["refs"][fam]["50"])          # die Zellen der Messungen: gleichverteilt und Gruppen (Gitter nicht gemessen)
    assert gemessen == 18
    steht("in den gemessenen Zellen 18 von 20 Instanzen als optimal bewiesen, mit der Familie Gitter 28 von 30")


# ------------------------------------------------------------------------------------------------------------- Texte
def test_annahmenblock_der_app_steht_wortgleich_in_der_readme():
    assert TX.ASSUMPTIONS_MARKDOWN.strip() in README


def test_testzahl_der_readme_stimmt_mit_der_suite_ueberein():
    m = re.search(r"(\d+) Tests in (\d+) Dateien", README)
    assert m, "Testzahl fehlt in der README"
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=170).stdout
    zeilen = [z for z in out.splitlines() if "::" in z]
    assert int(m.group(1)) == len(zeilen)
    assert int(m.group(2)) == len(list((ROOT / "tests").glob("test_*.py")))


def test_jeder_testverweis_der_readme_gibt_es():
    """Die Spalte „Test“ der Befundtabellen nennt Datei und Testfunktion: beide müssen existieren."""
    verweise = re.findall(r"`(test_[a-z_]+\.py)(?:::(test_[a-z0-9_]+))?`", README)
    assert len(verweise) >= 15
    for datei, funktion in verweise:
        pfad = ROOT / "tests" / datei
        assert pfad.exists(), datei
        if funktion:
            assert f"def {funktion}(" in pfad.read_text(encoding="utf-8"), (datei, funktion)


def test_readme_hat_die_pflichtabschnitte_in_der_reihenfolge():
    kopf = [z for z in README.splitlines() if z.startswith("## ")]
    erwartet = ["## Kernfrage und Befund in Kürze", "## Warum dieses Problem", "## Modell", "## Methodik", "## Befunde (gemessen, keine Behauptungen)", "## Training und Zähleinheiten",
                "## Befunde und Korrekturen gegenüber dem Plan", "## Wo die Annahmen enden", "## Ehrliche Grenzen", "## Verifikation", "## Dateistruktur", "## Bewusst nicht enthalten", "## Quellen (mit Belegstufe)",
                "## Verweise", "## Lokal ausführen"]
    assert kopf == erwartet
    assert "Q1" not in README and "BAUPLAN" not in README and "Phase 0" not in README and "Stück" not in README      # keine Begriffe der eigenen Planung


def test_modelldateien_stehen_in_der_readme_mit_namen():
    steht("`zugnetz_s0..4.npz`, `featnet_s0..4.npz`", "zwl-mlp-1")
    for s in range(C.MODELL_SEEDS):
        assert MLP.model_path("Z", s).exists() and MLP.model_path("K", s).exists()


def test_befund_plan_gegen_neumessung():
    """Spalte „Neumessung“ der Tabelle „Befunde und Korrekturen gegenüber dem Plan“; die Spalte „Plan“ ist die Messung der Vormessung und steht nicht im Test."""
    netz = X.lade_modell("Z", 0)[0]
    d20, d50 = X.e1_je_zug(netz, "gleich", 20, 20), X.e1_je_zug(netz, "gleich", 50, 10)
    assert round(d20["zufall"]["je_zug"] / d20["gelernt"]["je_zug"]) == 15 and round(d50["zufall"]["je_zug"] / d50["gelernt"]["je_zug"]) == 24
    steht("185,2 / 41,5 / 12,0 / 10,2", "900,8 / 188,5 / 37,9 / 32,4", "Faktor 15 statt 20", "Faktor 24 statt 51", "0,978 / 0,879", "53,7 % / 39,1 %", "59,7 % / 34,4 %")
    assert 171.7 / 8.5 == pytest.approx(20.2, abs=0.05) and 845.9 / 16.7 == pytest.approx(50.7, abs=0.05)
    pruef = {(zz, k): round(V.e4b_abstieg(F, zz, k, "zufall")["pruef"]) for zz in ("n20_gleich", "n50_gleich") for k in (3, 5, 8)}
    assert [pruef[("n20_gleich", k)] for k in (3, 5, 8)] == [648, 1220, 1654] and [pruef[("n50_gleich", k)] for k in (3, 5, 8)] == [3057, 6040, 8984]
    steht("648 / 1 220 / 1 654", "3 057 / 6 040 / 8 984", "3,5 % (n = 20), 18 % (n = 50)")
    v = [V.e3_vergleich(F, "n20_gleich", "gelernt", k, "frei", "zufall", R)["v"] for k in (2, 8, 32)]
    assert [V.de_vz(x) for x in v] == ["-0,27", "-0,01", "0,00"]
    kk = [V.e3_vergleich(F, "n20_gleich", "gelernt", k, "K", "zufall", R)["v"] for k in (2, 8, 32)]
    assert [V.de_vz(x) for x in kk] == ["+11,31", "+1,49", "+0,12"]
    n50 = {k: V.e3_vergleich(F, "n50_gleich", "gelernt", k, "frei", "zufall", R) for k in (2, 8)}
    assert V.de_vz(n50[2]["v"]) == "-0,72" and n50[2]["marke"] == "schlägt" and V.de_vz(n50[8]["v"]) == "-0,32" and n50[8]["marke"] == "gleich"
    steht("-0,27 / -0,01 / 0,00", "+11,31 / +1,49 / +0,12", "bei k = 2 (-0,72), bei k = 8 gleich (-0,32)")
    assert F["e4b"]["n20_gleich"]["zufalls_seeds"] == 10 and F["e4b"]["n50_gleich"]["zufalls_seeds"] == 5 and F["e4b"]["n50_gleich"]["ks"] == [2]
