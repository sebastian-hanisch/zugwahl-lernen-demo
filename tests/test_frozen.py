"""Eingefrorene Messreihen und Referenzen: Format, Formen, Monotonie, keine Spalte aus lauter gleichen Werten, Ausschnitt als Band neu gerechnet (nie bytegleich: Plattformen)."""

import numpy as np
import pytest

import zwl_constants as C
import zwl_evaluation as V
import zwl_experiments as X
import zwl_moves as M


@pytest.fixture(scope="module")
def frozen():
    return V.load_frozen()


@pytest.fixture(scope="module")
def refs():
    return V.load_refs()


def test_format_und_zellen(frozen):
    assert frozen["format"] == "zwl-frozen-1" and frozen["sichten"] == ["frei", "K", "O"]
    assert list(frozen["e3"]) == ["n20_gleich", "n50_gleich", "n50_cluster"]
    assert list(frozen["e4a"]) == ["n20_gleich", "n50_gleich"] and list(frozen["e4b"]) == ["n20_gleich", "n50_gleich"]
    z = frozen["e3"]["n20_gleich"]
    assert (z["instanzen"], z["zufalls_seeds"], z["gewichts_seeds"], z["ks"], z["M"]) == (20, 10, 5, [2, 8, 32], 1850)
    for name in ("n50_gleich", "n50_cluster"):
        z = frozen["e3"][name]
        assert (z["instanzen"], z["zufalls_seeds"], z["gewichts_seeds"], z["ks"], z["M"]) == (10, 10, 3, [2, 8], 12875)


def test_formen_der_bloecke(frozen):
    for name, z in frozen["e3"].items():
        for regel in C.REGELN_E3:
            r = z["regeln"][regel]
            assert np.array(r["best"]).shape == (z["instanzen"], z["zufalls_seeds"], len(z["ks"]), 3), (name, regel)
            assert np.array(r["enden"]).shape == np.array(r["falsch"]).shape == np.array(r["ev"]).shape == (z["instanzen"], z["zufalls_seeds"])
    for name, z in frozen["e4a"].items():
        assert sorted(z["abbruch"], key=int) == ["50", "100", "200"] and z["ks"] == ([2, 8] if name == "n20_gleich" else [2])
        for g, bl in z["abbruch"].items():
            assert set(bl) == set(C.REGELN_E4A)
            assert np.array(bl["gelernt"]["best"]).shape == (z["instanzen"], z["zufalls_seeds"], len(z["ks"]), 3)
    for name, z in frozen["e4b"].items():
        assert sorted(z["liste"], key=int) == ["3", "5", "8"]
        for k, bl in z["liste"].items():
            assert set(bl["abstieg"]) == set(C.REGELN_E4B) and set(bl["ils"]) == set(C.REGELN_E4B)
            assert np.array(bl["abstieg"]["zufall"]["laenge"]).shape == (z["instanzen"], z["zufalls_seeds"])


def test_besser_mit_mehr_budget_und_teurere_sicht_ist_nie_besser(frozen):
    for z in frozen["e3"].values():
        for regel in C.REGELN_E3:
            b = np.array(z["regeln"][regel]["best"])
            assert (np.diff(b, axis=2) <= 1e-9).all()                       # mehr Budget: nie schlechter
            assert (np.diff(b, axis=3) >= -1e-9).all()                      # Sicht frei <= K <= O (teurere Sicht: weniger Ereignisse im Budget)
    z = frozen["e3"]["n20_gleich"]
    for regel in ("zufall", "steilster"):                                   # ohne Merkmalszeilen sind die Sichten gleich
        b = np.array(z["regeln"][regel]["best"])
        assert (b[..., 0] == b[..., 1]).all() and (b[..., 1] == b[..., 2]).all()
    b = np.array(z["regeln"]["gelernt"]["best"])
    assert (b[..., 2] > b[..., 0]).any()


def test_luecken_sind_nicht_negativ_gegen_bewiesene_referenz(frozen, refs):
    z = frozen["e3"]["n20_gleich"]
    ref = V.e3_ref(frozen, "n20_gleich", refs)
    for regel in C.REGELN_E3:
        b = np.array(z["regeln"][regel]["best"])
        assert (V.luecke(b, ref[:, None, None, None]) >= -1e-4).all()


def test_keine_spalte_aus_lauter_gleichen_werten_und_regeln_weichen_ab(frozen, refs):
    for name, z in frozen["e3"].items():
        spalten = {r: V.e3_luecken(frozen, name, r, z["ks"][0], "K", refs) for r in C.REGELN_E3}
        assert len(set(np.round(list(spalten.values()), 6))) == 5, name
    for name, z in frozen["e4a"].items():
        for r in C.REGELN_E4A:                                                        # jede Regel weicht bei jedem Abbruch von den anderen Abbruchstufen ab (Anteil falscher Lokalminima, Lücke)
            falsch = [V.e4a_falsch_anteil(frozen, name, int(g), r) for g in z["abbruch"]]
            assert len(set(np.round(falsch, 6))) == 3 or set(np.round(falsch, 6)) == {1.0}, (name, r)          # Zufallsreihenfolge bei n = 50: jeder Abstieg endet falsch
        for g in z["abbruch"]:
            assert len({round(V.e4a_luecke(frozen, name, int(g), r, 2, "frei", refs), 6) for r in ("zufall", "nachbarliste")}) == 2
        zufall = [V.e4a_luecke(frozen, name, int(g), "zufall", 2, "frei", refs) for g in z["abbruch"]]
        assert len(set(np.round(zufall, 6))) == 3


def test_ohne_abbruch_gibt_es_keine_falschen_lokalminima(frozen):
    for z in frozen["e3"].values():
        for regel in ("zufall", "nachbarliste", "gelernt", "gelernt_det"):
            assert np.sum(z["regeln"][regel]["falsch"]) == 0 and np.sum(z["regeln"][regel]["enden"]) > 0


def test_referenzen_format_und_bewiesen(refs):
    assert refs["format"] == "zwl-refs-1"
    for fam in C.FAMILIEN:
        for n, anzahl in ((10, 20), (20, 20), (50, 10)):
            liste = refs["refs"][fam][str(n)]
            assert len(liste) == anzahl and all(r["laenge"] > 0 for r in liste)
            if n <= 20:
                assert all(r["bewiesen"] for r in liste)
            else:
                assert all(r["untergrenze"] <= r["laenge"] + 0.1 for r in liste)                  # ganzzahlige Skalierung mit 1 000: Rundung bis 0,05
    assert len(refs["val"]["gleich"]["20"]) == C.Z_VAL_INSTANZEN_FROZEN


def test_ausschnitt_wird_als_band_neu_gerechnet(frozen):
    """3 Instanzen, 2 Zufalls-Seeds, n = 20, k = 2 und 8: Zufall und gelernt neu gerechnet gleichen dem eingefrorenen Wert (Band, nicht bytegleich)."""
    z = frozen["e3"]["n20_gleich"]
    for regel in ("zufall", "gelernt"):
        for inst in range(3):
            ctx = X.kontext("gleich", 20, inst)
            for sd in range(2):
                scorer = X.lade_modell("Z", X.gewicht_seed(sd, 5))[0] if regel == "gelernt" else None
                r = X.lauf(ctx, regel, scorer, (2, 8), X.suche_strom("gleich", 20, inst, sd))
                soll = np.array(z["regeln"][regel]["best"][inst][sd])[:2]
                assert np.allclose(np.array(r["best"]), soll, rtol=1e-6), (regel, inst, sd)


def test_eingefrorene_laengen_sind_laengen_von_touren(frozen):
    """Die eingefrorene beste Länge einer Suche ist die Länge einer wirklichen Tour: neu gerechnet und mit der Tourlänge verglichen."""
    ctx = X.kontext("gleich", 20, 0)
    r = X.lauf(ctx, "zufall", None, (2,), X.suche_strom("gleich", 20, 0, 0))
    assert M.tour_length(ctx.D, r["res"]["best_tour"]) == pytest.approx(r["res"]["best_len"]) and r["best"][0][0] >= r["res"]["best_len"] - 1e-9
