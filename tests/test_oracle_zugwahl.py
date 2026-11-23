"""Orakel-Tests (ORACLE-PLAYBOOK §2 und §5): jede Kernroutine gegen einen unabhängigen Rechenweg, nie dieselbe Formel zweimal.

Held-Karp gegen Aufzählung und CP-SAT; Zugmenge, Delta und Lokalminimum gegen Listen und volle Neuberechnung; Prüfrunde = M; Rückwärtspässe gegen zentrale Differenzen;
AUC gegen Rangsumme; Gumbel-Reihenfolge gegen Plackett-Luce; SplitMix64-Vektor gegen den skalaren Strom; Abbruch und Nachbarliste gegen eine zweite Umsetzung; Zählung von Hand; Zweig-Tests."""

import itertools
import os

import numpy as np
import pytest
from conftest import kanon, tour_len, zuege_listen, zufalls_D, zufallstour_lcg

import zwl_construct as CO
import zwl_exact as EX
import zwl_hand as H
import zwl_moves as M
import zwl_rngvec as RV
import zwl_scenario as SC
import zwl_search as S
import zwl_zugnetz as Z
from zwl_mlp import load_model, model_path
from zwl_rng import SplitMix64


def _brute_force(D) -> float:
    n = len(D)
    best = 1e300
    for perm in itertools.permutations(range(1, n)):
        if perm[0] > perm[-1]:
            continue
        best = min(best, D[0][perm[0]] + D[perm[-1]][0] + sum(D[a][b] for a, b in zip(perm, perm[1:])))
    return best


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
def test_held_karp_gleich_aufzaehlung_aller_touren(n):
    D = SC.make_instance("gleich", n, n)[1]
    assert EX.held_karp(np.array(D)) == pytest.approx(_brute_force(D), abs=1e-9)
    laenge, tour = EX.held_karp_tour(np.array(D))
    assert laenge == pytest.approx(tour_len(D, tour), abs=1e-9) and sorted(tour) == list(range(n))


def test_held_karp_gleich_cpsat_an_drei_instanzen_mit_zwanzig_orten():
    cp = pytest.importorskip("ortools.sat.python.cp_model")
    for seed in range(3):
        D = np.array(SC.make_instance("gleich", 20, seed)[1])
        m = cp.CpModel()
        lits, kosten = [], []
        for i in range(20):
            for j in range(20):
                if i != j:
                    b = m.NewBoolVar("")
                    lits.append((i, j, b))
                    kosten.append(int(round(D[i][j] * 1000)) * b)
        m.AddCircuit(lits)
        m.Minimize(sum(kosten))
        s = cp.CpSolver()
        s.parameters.max_time_in_seconds = 60
        s.parameters.num_workers = min(4, os.cpu_count() or 1)
        assert s.Solve(m) == cp.OPTIMAL
        assert s.ObjectiveValue() / 1000 == pytest.approx(EX.held_karp(D), abs=0.05)             # ganzzahlige Skalierung mit 1 000: Rundung höchstens 0,0005 je Kante


def test_handbeispiel_optimum_auf_zwei_wegen():
    assert EX.held_karp(H.distanzen()) == H.OPTIMUM == _brute_force(H.TABELLE) == 27


@pytest.mark.parametrize("n", [6, 10, 12, 20, 50])
def test_zugmenge_und_delta_gegen_listen_und_volle_neuberechnung(n):
    nb = M.neighborhood(n)
    D = zufalls_D(n, 40 + n)
    t = zufallstour_lcg(n, 4)
    listen = zuege_listen(t)
    assert nb.M == len(listen) == n * (11 * n - 35) // 2
    soll = np.array([tour_len(D.tolist(), z[4]) - tour_len(D.tolist(), t) for z in listen])
    assert np.abs(M.deltas(D, np.array(t), nb) - soll).max() < 1e-12
    assert all(M.is_permutation(M.apply_move(np.array(t), k, nb), n) for k in range(0, nb.M, 7))


def test_lokalminimum_auf_zwei_wegen_und_gegenprobe():
    for seed in range(3):
        D = zufalls_D(16, 60 + seed)
        ctx = M.Kontext(D)
        r = S.descend(ctx, CO.nn_tour_np(D, 0), "zufall", RV.UStream(seed))
        listen = zuege_listen(r["tour"].tolist())
        assert r["ev_pruef"] == ctx.nb.M == len(listen)                                          # zweite Prüfrunde findet nichts: Zähler = M
        assert min(tour_len(D.tolist(), z[4]) for z in listen) >= tour_len(D.tolist(), r["tour"].tolist()) - 1e-12
        assert S.is_local_min_independent(D, r["tour"])
    assert not S.is_local_min_independent(zufalls_D(16, 70), zufallstour_lcg(16, 3))             # eine Zufallstour fällt durch


def test_beweis_zaehler_verschieden_und_doppelt_gezaehlt():
    for n, soll, soll_alt in ((20, 1850, 2210), (50, 12875, 15275)):
        D = zufalls_D(n, 80 + n)
        for doppelt, erwartet in ((False, soll), (True, soll_alt)):
            ctx = M.Kontext(D, doppelt)
            r = S.descend(ctx, CO.nn_tour_np(D, 0), "zufall", RV.UStream(1))
            assert r["ev_pruef"] == erwartet and ctx.nb.M == erwartet
            assert S.is_local_min_independent(D, r["tour"])


def test_doppelt_gezaehlte_zuege_haben_gleiche_deltas_wie_ihr_gegenstueck():
    n = 12
    nb2 = M.neighborhood(n, True)
    D = zufalls_D(n, 5)
    t = np.array(zufallstour_lcg(n, 1))
    d = M.deltas(D, t, nb2)
    assert nb2.M == M.neighborhood_size(n, True) and len(set(np.round(d, 12).tolist())) < len(d)
    voll = np.array([tour_len(D.tolist(), z[4]) - tour_len(D.tolist(), t.tolist()) for z in zuege_listen(t.tolist(), doppelt=True)])
    assert np.abs(d - voll).max() < 1e-12


def test_nachbarliste_gegen_schleife_ueber_orte_und_listen():
    """Zweite Umsetzung der Zugmenge der Nachbarliste: für jeden Zug die wirklich neuen Kanten (Kantenmengen der Touren) mit dem Rang aus sortierten Zeilen."""
    n, k = 10, 3
    D = zufalls_D(n, 90)
    Dl = D.tolist()
    t = zufallstour_lcg(n, 6)
    rang = {}
    for u in range(n):
        for r, v in enumerate(sorted((v for v in range(n) if v != u), key=lambda v: (Dl[u][v], v))):
            rang[(u, v)] = r
    kanten = lambda x: {frozenset((x[i], x[(i + 1) % n])) for i in range(n)}  # noqa: E731
    zaehler = sum(1 for z in zuege_listen(t) if any(min(rang[tuple(e)], rang[tuple(e)[::-1]]) < k for e in map(tuple, kanten(z[4]) - kanten(t))))
    assert S.waehle_zug(M.Kontext(D), np.array(t), "zufall", RV.UStream(1), liste=k)["n_cand"] == zaehler
    assert S.waehle_zug(M.Kontext(D), np.array(t), "zufall", RV.UStream(1), liste=n - 1)["n_cand"] == M.neighborhood_size(n)


def test_abbruch_gleich_m_ist_die_volle_pruefung_und_anteil_falscher_minima_unabhaengig_gezaehlt():
    D = zufalls_D(30, 91)
    ctx = M.Kontext(D)
    nn = CO.nn_tour_np(D, 0)
    voll = S.descend(ctx, nn, "zufall", RV.UStream(3))
    assert S.descend(ctx, nn, "zufall", RV.UStream(3), abbruch=ctx.nb.M)["tour"].tolist() == voll["tour"].tolist()
    falsch = [S.descend(ctx, nn, "zufall", RV.UStream(s), abbruch=30) for s in range(8)]
    assert [r["falsch"] for r in falsch] == [not S.is_local_min_independent(D, r["tour"]) for r in falsch]


def test_splitmix64_vektor_gegen_skalaren_strom_bitgleich():
    for seed in (0, 7, 2 ** 63 + 5):
        us = RV.UStream(seed)
        skalar = SplitMix64(seed)
        assert us.uniform(50).tolist() == [skalar.random() for _ in range(50)]
        assert us.position == 50
    o = RV.UStream(3).random_order(4, 17)
    assert all(sorted(r.tolist()) == list(range(17)) for r in o)


def test_auc_gegen_rangsumme_und_zweite_umsetzung_mit_sortieren():
    from scipy.stats import rankdata
    us = RV.UStream(11)
    score = np.round(us.uniform(500) * 20) / 20
    y = (us.uniform(500) < 0.2).astype(float)
    r = rankdata(score)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    rang = (r[y > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    neg = np.sort(score[y == 0])
    zweite = np.mean([(np.searchsorted(neg, s, "left") + 0.5 * (np.searchsorted(neg, s, "right") - np.searchsorted(neg, s, "left"))) / n0 for s in score[y > 0]])
    assert Z.auc(score, y) == pytest.approx(rang, abs=1e-12) == pytest.approx(zweite, abs=1e-12)


def test_netz_z_rueckwaertspass_gegen_zentrale_differenzen_voller_groesse():
    from zwl_mlp import bce_loss_grad, init_mlp
    p = init_mlp(9, 16, RV.UStream(2))
    us = RV.UStream(3)
    X, y = us.uniform((6, 9)), (us.uniform(6) > 0.5).astype(float)
    _, g = bce_loss_grad(p, X, y)
    maxabw = 0.0
    for P, gP in zip(p, g):
        it = np.nditer(P, flags=["multi_index"], op_flags=["readwrite"])
        for _ in it:
            i = it.multi_index
            alt = P[i]
            P[i] = alt + 1e-6
            fp = bce_loss_grad(p, X, y)[0]
            P[i] = alt - 1e-6
            fm = bce_loss_grad(p, X, y)[0]
            P[i] = alt
            maxabw = max(maxabw, abs((fp - fm) / 2e-6 - gP[i]))
    assert maxabw / max(np.abs(a).max() for a in g) < 1e-6


def test_zweig_jede_regel_weicht_ab_und_keine_spalte_aus_lauter_gleichen_werten():
    netz = load_model(model_path("Z", 0))[0]
    spalten = {r: [] for r in ("aufzaehlung", "zufall", "nachbarliste", "gelernt", "gelernt_det", "steilster")}
    for seed in range(5):
        D = zufalls_D(25, 100 + seed)
        ctx = M.Kontext(D)
        for regel in spalten:
            spalten[regel].append(S.descend(ctx, CO.nn_tour_np(D, 0), regel, RV.UStream(seed), netz)["ev_zuege"])
    for a, b in itertools.combinations(spalten, 2):
        assert spalten[a] != spalten[b], (a, b)
    assert all(len(set(v)) > 1 for v in spalten.values())
    # Abbruch 50 gegen 100: andere Ergebnisse der Suche
    D = zufalls_D(50, 7)
    ctx = M.Kontext(D)
    nn = CO.nn_tour_np(D, 0)
    assert len({S.ils(ctx, nn, "zufall", 2 * ctx.nb.M, RV.UStream(1), abbruch=g)["best_len"] for g in (50, 100)}) == 2


def test_kanonische_form_unterscheidet_29_nachbartouren_im_handbeispiel():
    t = list(H.START)
    assert len({kanon(z[4]) for z in zuege_listen(t)} - {kanon(t)}) == 29
