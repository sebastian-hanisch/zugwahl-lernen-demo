"""Randfälle: kleinste Instanzen, Nenner 0, Gleichstände, Budget unter den Startkosten, leere Nachbarliste, fehlende Dateien, Intervall mit einer Instanz."""

import json

import numpy as np
import pytest
from conftest import zufalls_D

import zwl_construct as CO
import zwl_evaluation as V
import zwl_mlp as MLP
import zwl_moves as M
import zwl_search as S
import zwl_stats as ST
import zwl_zugnetz as Z
from zwl_rngvec import UStream


def test_kleinste_instanz_fuenf_orte_hat_zuege_und_findet_ein_lokalminimum():
    n = 5
    D = zufalls_D(n, 3)
    ctx = M.Kontext(D)
    assert ctx.nb.M == M.neighborhood_size(5) == 5 * (55 - 35) // 2 == 50
    r = S.descend(ctx, CO.nn_tour_np(D, 0), "zufall", UStream(1))
    assert S.is_local_min_independent(D, r["tour"]) and r["ev_pruef"] == 50


def test_gleichstaende_in_der_reihenfolge_sind_stabil():
    """Alle Entfernungen gleich: kein Zug verbessert (Delta 0), jede Regel bewertet genau M Züge und endet sofort."""
    n = 8
    D = np.ones((n, n)) - np.eye(n)
    ctx = M.Kontext(D)
    for regel in ("aufzaehlung", "zufall", "nachbarliste", "steilster"):
        r = S.descend(ctx, np.arange(n), regel, UStream(2))
        assert r["zuege"] == 0 and r["ev_pruef"] == ctx.nb.M and r["tour"].tolist() == list(range(n))


def test_budget_unter_den_startkosten_gibt_nur_die_startloesung():
    D = zufalls_D(12, 5)
    ctx = M.Kontext(D)
    res = S.ils(ctx, CO.nn_tour_np(D, 0), "zufall", 10, UStream(1))
    assert len(res["ev"]) == 1 and res["schritte"] == 0 and res["erster_abstieg"] is None and np.isnan(S.bester_wert(res, 5))


def test_nachbarliste_k1_hat_wenige_kandidaten_und_aendert_nichts_an_der_korrektheit():
    D = zufalls_D(20, 7)
    ctx = M.Kontext(D)
    r = S.descend(ctx, CO.nn_tour_np(D, 0), "zufall", UStream(1), liste=1)
    assert 0 < r["n_pruef"] < ctx.nb.M and M.is_permutation(r["tour"], 20)


def test_abbruch_groesser_als_zugmenge_ist_die_volle_pruefung():
    D = zufalls_D(10, 8)
    ctx = M.Kontext(D)
    nn = CO.nn_tour_np(D, 0)
    a = S.descend(ctx, nn, "zufall", UStream(1), abbruch=10 ** 6)
    b = S.descend(ctx, nn, "zufall", UStream(1))
    assert a["ev_pruef"] == b["ev_pruef"] == ctx.nb.M and a["tour"].tolist() == b["tour"].tolist()


def test_zwei_gleiche_orte_ergeben_keinen_absturz_der_merkmale():
    pts = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    ctx = M.Kontext(D)
    _, F, _ = M.move_features(ctx.D, np.arange(6), ctx.nb, ctx.s, ctx.Rs)
    assert np.isfinite(F).all()


def test_modelldatei_fehlt_gibt_oserror(tmp_path):
    with pytest.raises(OSError):
        MLP.load_model(tmp_path / "gibt_es_nicht.npz")


def test_eingefrorene_messung_fehlt_oder_hat_falsches_format(tmp_path):
    with pytest.raises(FileNotFoundError):
        V.load_frozen(str(tmp_path / "gibt_es_nicht.json"))
    p = tmp_path / "falsch.json"
    p.write_text(json.dumps({"format": "anders"}))
    with pytest.raises(ValueError):
        V.load_frozen(str(p))
    with pytest.raises(ValueError):
        V.load_refs(str(p))


def test_referenz_nicht_vorgerechnet_gibt_none():
    refs = {"refs": {"gleich": {"20": [{"laenge": 1.0, "bewiesen": True}]}}}
    assert V.ref_laenge("gleich", 20, 0, refs) == (1.0, True)
    assert V.ref_laenge("gleich", 20, 1, refs) is None and V.ref_laenge("gitter", 20, 0, refs) is None and V.ref_laenge("gleich", 50, 0, refs) is None


def test_intervall_mit_einer_instanz_und_leere_zaehlung():
    assert ST.bootstrap_ci([3.0]) is None and ST.unter_gegner([1.0], [2.0]) is False
    r = ST.compare(np.array([1.0]), np.array([2.0]))
    assert r["klasse"] is None and V.marke(r) == "dazwischen"


def test_zugnetz_training_mit_einer_trainingsinstanz_und_wenigen_schritten():
    F, y, ns = Z.sammle_zustaende(Z.train_instanzen(1), k_budget=1)
    assert ns >= 1 and len(y) == ns * 200
    p, _, cnt = Z.train_zugnetz(F, y, 3, batch=16)
    assert cnt["schritte"] == 3 and MLP.n_params(p) == 177


def test_auc_ohne_positive_beispiele_meldet_fehler():
    with pytest.raises(ValueError):
        Z.auc([0.1, 0.2], [0, 0])
