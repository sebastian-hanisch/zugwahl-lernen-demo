"""Suche: Handbeispiel (7 + 93 = 100 Zugauswertungen, 93 % Beweis), Lokalminimum auf zwei Wegen, Prüfrunde = M, Budget, Abbruch, Nachbarliste, Zweig-Tests."""

import itertools

import numpy as np
import pytest
from conftest import kanon, tour_len, zuege_listen, zufalls_D, zufallstour_lcg

import zwl_construct as CO
import zwl_hand as H
import zwl_moves as M
import zwl_search as S
import zwl_zugnetz as Z
from zwl_mlp import load_model, model_path
from zwl_rngvec import UStream


def _modell():
    return load_model(model_path("Z", 0))[0]


def test_handbeispiel_abstieg_aus_naechstem_nachbar_in_aufzaehlung(hand_ctx):
    r = S.descend(hand_ctx, np.array(H.START), "aufzaehlung", protokoll=True)
    assert r["zuege"] == 1 and r["ev_zuege"] == 7 and r["ev_pruef"] == 93 == r["n_pruef"]
    assert r["ev_zuege"] + r["ev_pruef"] == 100 and 100 * r["ev_pruef"] / 100 == 93
    assert r["tour"].tolist() == [0, 1, 5, 2, 4, 3] and M.tour_length(hand_ctx.D, r["tour"]) == H.OPTIMUM == 27
    assert r["protokoll"][0]["stelle"] == 7 and r["falsch"] is False


def test_handbeispiel_abstieg_aus_schlechter_starttour(hand_ctx):
    r = S.descend(hand_ctx, np.array(H.SCHLECHTER_START), "aufzaehlung", protokoll=True)
    assert M.tour_length(hand_ctx.D, H.SCHLECHTER_START) == 42
    assert [(p["stelle"], p["delta"]) for p in r["protokoll"]] == [(1, -6), (4, -1), (5, -5), (9, -3)]
    assert r["zuege"] == 4 and M.tour_length(hand_ctx.D, r["tour"]) == 27
    assert r["ev_zuege"] + r["ev_pruef"] == 112 and round(100 * 93 / 112) == 83


def test_handbeispiel_nachbarlisten_reihenfolge_erste_verbesserung_nach_einer_auswertung(hand_ctx):
    r = S.descend(hand_ctx, np.array(H.START), "nachbarliste", us=UStream(1), protokoll=True)
    assert r["protokoll"][0]["stelle"] == 1 and r["protokoll"][0]["delta"] == -3 and r["protokoll"][0]["art"] == 0
    assert r["rows"] == 2 * 93                                           # zwei Schritte (ein Zug, die Prüfrunde), je alle 93 Zeilen


def test_handbeispiel_steilster_nimmt_das_beste_delta_und_kostet_m(hand_ctx):
    z = S.waehle_zug(hand_ctx, np.array(H.START), "steilster")
    assert z["kosten"] == 93 and z["delta"][z["idx"]] == -3 and z["improving"]


def test_handbeispiel_optimum_durch_aufzaehlung_aller_touren(hand_D):
    alle = {}
    for p in itertools.permutations(range(1, 6)):
        t = [0, *p]
        alle[kanon(t)] = tour_len(H.TABELLE, t)
    assert len(alle) == 60 and min(alle.values()) == 27


@pytest.mark.parametrize("regel", ["aufzaehlung", "zufall", "nachbarliste", "steilster"])
def test_abstieg_endet_im_lokalminimum_auf_zwei_wegen(regel):
    for n, seed in ((12, 1), (20, 2), (30, 3)):
        D = zufalls_D(n, seed)
        ctx = M.Kontext(D)
        r = S.descend(ctx, CO.nn_tour_np(D, 0), regel, UStream(seed))
        assert M.is_permutation(r["tour"], n)
        assert S.is_local_min_independent(D, r["tour"])
        assert M.deltas(D, r["tour"], ctx.nb).min() >= -1e-12
        assert r["ev_pruef"] == ctx.nb.M and r["falsch"] is False


def test_gegenprobe_eine_zufallstour_ist_kein_lokalminimum():
    D = zufalls_D(12, 4)
    assert not S.is_local_min_independent(D, zufallstour_lcg(12, 1))


def test_pruefrunde_auf_lokalminimum_ist_1850_und_12875():
    for n, soll in ((20, 1850), (50, 12875)):
        D = zufalls_D(n, 7)
        ctx = M.Kontext(D)
        r = S.descend(ctx, CO.nn_tour_np(D, 0), "zufall", UStream(1))
        z = S.waehle_zug(ctx, r["tour"], "zufall", UStream(2))
        assert z["kosten"] == soll == M.neighborhood_size(n) and not z["improving"]


def test_rechnung_ist_wiederholbar_und_haengt_am_zufalls_seed():
    D = zufalls_D(20, 9)
    ctx = M.Kontext(D)
    nn = CO.nn_tour_np(D, 0)
    a = S.descend(ctx, nn, "zufall", UStream(5))
    b = S.descend(ctx, nn, "zufall", UStream(5))
    c = [S.descend(ctx, nn, "zufall", UStream(s))["ev_zuege"] for s in range(6)]
    assert a["ev_zuege"] == b["ev_zuege"] and a["tour"].tolist() == b["tour"].tolist() and len(set(c)) > 1


def test_ils_budget_best_so_far_und_ereignisse():
    D = zufalls_D(20, 11)
    ctx = M.Kontext(D)
    nn = CO.nn_tour_np(D, 0)
    m = ctx.nb.M
    res = S.ils(ctx, nn, "zufall", 8 * m, UStream(3))
    assert res["ev"][0] == 190 and res["ev"][-1] >= 8 * m and res["ev"][-1] < 8 * m + m + 1       # das Budget wird höchstens um einen Schritt überschritten
    assert (np.diff(res["ev"]) > 0).all() and (np.diff(res["rows"]) >= 0).all()
    werte = [S.bester_wert(res, b) for b in (m, 2 * m, 4 * m, 8 * m)]
    assert all(a >= b - 1e-12 for a, b in zip(werte, werte[1:])) and werte[-1] == pytest.approx(res["best_len"])
    assert M.is_permutation(res["best_tour"], 20) and M.tour_length(D, res["best_tour"]) == pytest.approx(res["best_len"])
    assert res["kicks"] >= 1 and res["falsch"] == 0 and res["enden"] >= res["kicks"]
    assert np.isnan(S.bester_wert(res, 10))                                                        # schon die Startlösung kostet 190


def test_ils_schlaegt_oder_gleicht_dem_ersten_abstieg():
    D = zufalls_D(20, 12)
    ctx = M.Kontext(D)
    res = S.ils(ctx, CO.nn_tour_np(D, 0), "zufall", 8 * ctx.nb.M, UStream(2))
    assert res["best_len"] <= res["erster_abstieg"][4] + 1e-12


def test_bester_wert_gewicht_der_zeilen():
    res = {"ev": np.array([10, 20, 30]), "rows": np.array([0, 100, 200]), "ln": np.array([5.0, 4.0, 3.0])}
    assert S.bester_wert(res, 30, 0.0) == 3.0                    # Zeilen gratis
    assert S.bester_wert(res, 130, 1.0) == 4.0                   # 20 + 100 = 120 <= 130, 30 + 200 = 230 > 130
    assert S.bester_wert(res, 130, 22.0) == 5.0                  # 20 + 2 200 > 130


def test_abbruch_gleich_m_ist_die_volle_pruefung():
    D = zufalls_D(20, 13)
    ctx = M.Kontext(D)
    nn = CO.nn_tour_np(D, 0)
    a = S.descend(ctx, nn, "zufall", UStream(4))
    b = S.descend(ctx, nn, "zufall", UStream(4), abbruch=ctx.nb.M)
    assert a["tour"].tolist() == b["tour"].tolist() and a["ev_zuege"] == b["ev_zuege"] and b["ev_pruef"] == ctx.nb.M


def test_abbruch_beendet_nach_g_erfolglosen_kandidaten_und_kann_ein_falsches_lokalminimum_liefern():
    n = 50
    D = zufalls_D(n, 14)
    ctx = M.Kontext(D)
    nn = CO.nn_tour_np(D, 0)
    falsch = 0
    for s in range(6):
        r = S.descend(ctx, nn, "zufall", UStream(s), abbruch=50)
        assert r["ev_pruef"] == 50
        falsch += int(r["falsch"])
        assert r["falsch"] == (M.deltas(D, r["tour"], ctx.nb).min() < -1e-12)
        assert r["falsch"] == (not S.is_local_min_independent(D, r["tour"]))      # unabhängige Gegenprobe
    assert falsch >= 1


def _unabhaengige_liste_groesse(D, t, k):
    """Zweiter Weg: jeden Nachbartour ausdrücklich bauen, die neuen Kanten als Städtepaare bestimmen und deren Rang aus den sortierten Zeilen der Matrix lesen."""
    n = len(t)
    rang = {}
    for u in range(n):
        ordn = sorted((v for v in range(n) if v != u), key=lambda v: (D[u][v], v))
        for r, v in enumerate(ordn):
            rang[(u, v)] = r

    def kanten(x):
        return {frozenset((x[i], x[(i + 1) % n])) for i in range(n)}

    alt = kanten(t)
    zaehler = 0
    for z in zuege_listen(t):
        neu = kanten(z[4]) - alt
        if any(min(rang[tuple(e)], rang[tuple(e)[::-1]]) < k for e in map(tuple, neu)):
            zaehler += 1
    return zaehler


@pytest.mark.parametrize("k", [1, 3, 5, 100])
def test_nachbarliste_groesse_gleich_zweitem_rechenweg(k):
    n = 10
    D = zufalls_D(n, 15)
    ctx = M.Kontext(D)
    t = np.array(zufallstour_lcg(n, 2))
    z = S.waehle_zug(ctx, t, "zufall", UStream(1), liste=k)
    assert z["n_cand"] == _unabhaengige_liste_groesse(D.tolist(), t.tolist(), k)
    if k >= n - 1:
        assert z["n_cand"] == ctx.nb.M


def test_nachbarliste_verkleinert_die_pruefrunde_und_k_gleich_n_minus_1_ist_voll():
    n = 20
    D = zufalls_D(n, 16)
    ctx = M.Kontext(D)
    nn = CO.nn_tour_np(D, 0)
    pr = [S.descend(ctx, nn, "zufall", UStream(1), liste=k)["n_pruef"] for k in (3, 5, 8)]
    assert pr[0] < pr[1] < pr[2] < ctx.nb.M
    assert S.descend(ctx, nn, "zufall", UStream(1), liste=n - 1)["n_pruef"] == ctx.nb.M


def test_zweig_jede_regel_weicht_von_jeder_anderen_ab():
    D = zufalls_D(30, 17)
    ctx = M.Kontext(D)
    nn = CO.nn_tour_np(D, 0)
    netz = _modell()
    ergebnis = {}
    for regel in ("aufzaehlung", "zufall", "nachbarliste", "gelernt", "gelernt_det", "steilster"):
        r = S.descend(ctx, nn, regel, UStream(3), scorer=netz)
        ergebnis[regel] = (r["ev_zuege"], r["tour"].tolist())
    assert len({v[0] for v in ergebnis.values()}) == 6                      # keine Spalte aus lauter gleichen Werten
    assert ergebnis["gelernt"][0] != ergebnis["gelernt_det"][0]


def test_zweig_abbruch_und_nachbarliste_aendern_das_ergebnis():
    D = zufalls_D(50, 18)
    ctx = M.Kontext(D)
    nn = CO.nn_tour_np(D, 0)
    e = {g: S.ils(ctx, nn, "zufall", 2 * ctx.nb.M, UStream(1), abbruch=g)["best_len"] for g in (0, 50, 100)}
    assert len(set(e.values())) == 3
    assert S.ils(ctx, nn, "zufall", 2 * ctx.nb.M, UStream(1), liste=3)["best_len"] != e[0]


def test_gelernte_regel_braucht_weniger_auswertungen_je_zug_als_zufall():
    """Das Zugnetz ordnet besser: Mittel über 10 Instanzen der Validierung (n = 20), Auswertungen je angenommenem Zug."""
    netz = _modell()
    erg = {"zufall": [0, 0], "gelernt": [0, 0]}
    for ctx in Z.val_instanzen(10):
        for regel in erg:
            r = S.descend(ctx, CO.nn_tour_np(ctx.D, 0), regel, UStream(1), scorer=netz)
            erg[regel][0] += r["ev_zuege"]
            erg[regel][1] += r["zuege"]
    je_zug = {k: v[0] / v[1] for k, v in erg.items()}
    assert je_zug["gelernt"] * 5 < je_zug["zufall"]


def test_unbekannte_regel():
    D = zufalls_D(10, 1)
    with pytest.raises(ValueError):
        S.waehle_zug(M.Kontext(D), np.arange(10), "gibtesnicht", UStream(1))
