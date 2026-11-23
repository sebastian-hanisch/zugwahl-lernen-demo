"""Experimente E0 bis E5 live: Form der Ergebnisse, Zählungen von Hand (Handbeispiel), Zählgleichungen, Wiederholbarkeit."""

import numpy as np
import pytest

import zwl_constants as C
import zwl_experiments as X
import zwl_hand as H
import zwl_moves as M
import zwl_search as S


@pytest.fixture(scope="module")
def netz():
    return X.lade_modell("Z", 0)[0]


def test_e0_handbeispiel_mit_kanten_der_zuege():
    ctx = X.kontext("hand", 6, 0)
    r = X.e0_abstieg(ctx, "aufzaehlung", None, "hand", 0)
    assert r["start"].tolist() == list(H.START) and r["start_laenge"] == 30 and r["end_laenge"] == 27 and r["zuege"] == 1
    s = r["schritte"][0]
    assert s["stelle"] == 7 and s["delta"] == -3 and len(s["entfernt"]) == len(s["neu"])
    t = s["tour_davor"].tolist()
    kanten = lambda x: {frozenset((x[i], x[(i + 1) % 6])) for i in range(6)}  # noqa: E731
    assert {frozenset(e) for e in s["entfernt"]} - {frozenset(e) for e in s["neu"]} == kanten(t) - kanten(s["tour"].tolist())
    assert {frozenset(e) for e in s["neu"]} - {frozenset(e) for e in s["entfernt"]} == kanten(s["tour"].tolist()) - kanten(t)


def test_e0_schritte_haengen_zusammen(netz):
    ctx = X.kontext("gleich", 20, 0)
    r = X.e0_abstieg(ctx, "gelernt", netz, "gleich", 0)
    assert len(r["schritte"]) == r["zuege"] and sum(p["stelle"] for p in r["schritte"]) == r["ev_zuege"]
    assert r["schritte"][0]["tour_davor"].tolist() == r["start"].tolist() and r["schritte"][-1]["tour"].tolist() == r["tour"].tolist()
    assert r["rows"] == (r["zuege"] + 1) * ctx.nb.M                                           # je Schritt alle M Zeilen
    again = X.e0_abstieg(ctx, "gelernt", netz, "gleich", 0)
    assert again["tour"].tolist() == r["tour"].tolist()


def test_e1_guete_form_und_band(netz):
    g = X.e1_guete(netz, 8)
    assert set(g["auc"]) == {"netz", "hand", "rem_mean", "rem_max", "minrank", "maxrank"} and g["zeilen"] == g["zustaende"] * 200
    assert 0.9 < g["auc"]["netz"] < 1.0 and g["auc"]["netz"] > g["auc"]["hand"] > max(g["auc"][k] for k in ("rem_mean", "rem_max", "minrank", "maxrank")) - 0.05


def test_e1_je_zug_zufall_viel_mehr_als_gelernt_und_steilster_gleich_m(netz):
    d = X.e1_je_zug(netz, "gleich", 20, 6)
    assert d["steilster"]["je_zug"] == M.neighborhood_size(20) and d["steilster"]["ev_pruef"] == 1850
    assert d["zufall"]["je_zug"] > 5 * d["gelernt"]["je_zug"] and d["nachbarliste"]["je_zug"] < d["zufall"]["je_zug"]
    assert d["zufall"]["zeilen"] == 0 and d["gelernt"]["zeilen"] == (d["gelernt"]["zuege"] + 1) * 1850


def test_e1_zugarten_anteile_summieren_zu_eins(netz):
    z = X.e1_zugarten(netz, "gleich", 20, 4)
    assert sum(z["anteil_top"]) == pytest.approx(1.0) and sum(z["anteil_m"]) == pytest.approx(1.0) and z["zustaende"] >= 4
    assert z["anteil_m"] == pytest.approx([20 * 17 / 2 / 1850, 20 * 18 / 1850, 2 * 20 * 17 / 1850, 2 * 20 * 16 / 1850])


def test_e2_beweis_prueft_formel_und_zaehlweisen():
    neu = X.e2_beweis("gleich", 20, 8)
    alt = X.e2_beweis("gleich", 20, 8, doppelt=True)
    assert neu["M"] == 1850 and alt["M"] == 2210 and neu["pruef"] == 1850 and alt["pruef"] == 2210
    assert 0.3 < neu["anteil"] < 0.8 and neu["gespart_max"] == pytest.approx(1 - neu["anteil"])
    assert X.e2_beweis("gleich", 20, 8) == neu                                                  # wiederholbar


def test_lauf_liefert_best_je_budget_und_sicht(netz):
    ctx = X.kontext("gleich", 20, 1)
    r = X.lauf(ctx, "gelernt", netz, (2, 8), X.suche_strom("gleich", 20, 1, 0))
    b = np.array(r["best"])
    assert b.shape == (2, 3) and (b[1] <= b[0] + 1e-12).all() and b[0, 0] <= b[0, 1] <= b[0, 2] and r["ev"] >= 8 * 1850
    z = X.lauf(ctx, "zufall", None, (2,), X.suche_strom("gleich", 20, 1, 0))
    assert z["best"][0][0] == z["best"][0][1] == z["best"][0][2] and z["rows"] == 0


def test_e3_probe_und_kurve(netz):
    ctx = X.kontext("gleich", 20, 0)
    p = X.e3_probe(ctx, "gleich", 0, 2, netz, seeds=2, regeln=("zufall", "gelernt"))
    assert p["gelernt"]["best"].shape == (2, 3)
    raster = np.geomspace(1850, 2 * 1850, 6)
    k = X.kurve(p["gelernt"]["runs"], "gelernt", "K", raster)
    assert k.shape == (6,) and (np.diff(k[~np.isnan(k)]) <= 1e-9).all()


def test_e4_probe_abbruch_und_liste(netz):
    ctx = X.kontext("gleich", 50, 0)
    a = X.e4_probe(ctx, "gleich", 0, "zufall", None, 50, 0)
    assert a["probe"]["ev_pruef"] == 50 and a["voll"]["ev_pruef"] == 12875 and a["kein_lokalminimum"] == a["probe"]["falsch"]
    b = X.e4_probe(ctx, "gleich", 0, "gelernt", netz, 0, 3)
    assert b["probe"]["n_pruef"] < 12875 and b["kein_lokalminimum"] == b["probe"]["falsch"]


def test_e5_startloesungen_zaehlungen_und_obergrenze():
    import zwl_featnet as FN
    params_k = X.lade_modell("K", 0)[0]
    zeilen = X.e5_starts("gleich", 20, 6, params_k, None)
    art = {z["art"]: z for z in zeilen}
    assert set(art) == set(C.START_ARTEN) and art["optimum"]["ev_zuege"] == 0 and art["optimum"]["ev_pruef"] == 1850
    assert art["optimum"]["ersparnis"] > art["farthest"]["ersparnis"] and art["optimum"]["ersparnis"] > art["netz_k"]["ersparnis"] > -50
    assert art["nn"]["ersparnis"] == 0 and art["zufall"]["ersparnis"] < 0
    assert art["nn"]["k_start"] == 190 and art["netz_k"]["k_start"] == FN.greedy_tours(np.eye(20) + 1, params_k)[2] == 190 and art["optimum"]["k_gesamt"] == 1850
    assert art["farthest"]["o_start"] == 1600 and art["netz_k"]["o_start"] == 144 * 190


def test_k_greedy_gap_gegen_referenz():
    from zwl_evaluation import load_refs
    gp, gn = X.k_greedy_gap(X.lade_modell("K", 1)[0], load_refs(), count=10)
    assert gp < gn and 5 < gp < 14                                                               # Band um die Validierungslücke 7,95 % des Modells (50 Instanzen)


def test_training_in_der_app_zaehlt():
    p, verlauf, cnt = X.trainiere_zugnetz(100, 10, 0)
    assert cnt["schritte"] == 100 and cnt["zeilen"] == 100 * 4096 and cnt["instanzen"] == 10 and cnt["daten_zeilen"] == cnt["daten_zustaende"] * 200


def test_ils_wirklich_mit_abbruch_hat_viele_enden_und_falsche_minima():
    ctx = X.kontext("gleich", 50, 0)
    r = X.lauf(ctx, "zufall", None, (2,), X.suche_strom("gleich", 50, 0, 0), abbruch=50)
    assert r["enden"] > 50 and r["falsch"] > 0
    assert S.bester_wert(r["res"], 2 * 12875) == pytest.approx(r["best"][0][0])
