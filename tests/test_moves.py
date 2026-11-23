"""Zugmenge, Delta, Merkmale, Anwenden und Kick: Handbeispiel (sechs Orte) von Hand, Formel M(n) gegen Aufzählung, Positionsfelder gegen Listen, Delta gegen volle Neuberechnung."""

import numpy as np
import pytest
from conftest import kanon, tour_len, zuege_listen, zufalls_D, zufallstour_lcg

import zwl_hand as H
import zwl_moves as M
from zwl_rngvec import UStream


def test_handbeispiel_zugmenge_nach_arten(hand_ctx):
    assert M.m_arten(6) == {"2-opt": 9, "Or-opt 1": 24, "Or-opt 2": 36, "Or-opt 3": 24}
    assert hand_ctx.nb.M == 93 == M.neighborhood_size(6) == 9 + 24 + 36 + 24
    assert M.neighborhood_size(6, doppelt=True) == 117 == 93 + 24
    assert [int((hand_ctx.nb.art == a).sum()) for a in range(4)] == [9, 24, 36, 24]


@pytest.mark.parametrize("n", range(5, 13))
def test_formel_gleich_aufzaehlung(n):
    t = list(range(n))
    assert len(zuege_listen(t)) == M.neighborhood_size(n) == M.neighborhood(n).M
    assert len(zuege_listen(t, doppelt=True)) == M.neighborhood_size(n, doppelt=True)
    assert n * (11 * n - 35) // 2 == M.neighborhood_size(n)


def test_groessen_20_und_50():
    assert M.neighborhood_size(20) == 1850 and M.neighborhood_size(50) == 12875
    assert M.neighborhood_size(20, True) == 2210 and M.neighborhood_size(50, True) == 15275
    assert M.neighborhood(20).M == 1850 and M.neighborhood(50).M == 12875
    assert M.neighborhood_size(20, True) - M.neighborhood_size(20) == 20 * 18 == 360


def test_zu_wenige_orte():
    with pytest.raises(ValueError):
        M.Neighborhood(4)


def test_handbeispiel_delta_gleich_volle_neuberechnung(hand_D, hand_ctx):
    t = list(H.START)
    L0 = tour_len(H.TABELLE, t)
    assert L0 == 30
    delta = M.deltas(hand_D, np.array(t), hand_ctx.nb)
    voll = np.array([tour_len(H.TABELLE, z[4]) - L0 for z in zuege_listen(t)], dtype=float)
    assert np.abs(delta - voll).max() < 1e-12
    assert int((delta < 0).sum()) == 8 and delta.min() == -3 and int((delta == -3).sum()) == 5


def test_handbeispiel_erste_verbesserung_in_aufzaehlung_an_stelle_7(hand_D, hand_ctx):
    delta = M.deltas(hand_D, np.array(H.START), hand_ctx.nb)
    assert int(np.argmax(delta < 0)) + 1 == 7


def test_handbeispiel_verschiedene_nachbarn_weniger_als_zuege():
    t = list(H.START)
    nachbarn = {kanon(z[4]) for z in zuege_listen(t)} - {kanon(t)}
    assert len(nachbarn) == 29 < 93


@pytest.mark.parametrize("n", [6, 10, 12, 20, 50])
def test_delta_felder_gleich_listen_auf_zufallstouren(n):
    nb = M.neighborhood(n)
    D = zufalls_D(n, n)
    Dl = D.tolist()
    for s in range(3):
        t = zufallstour_lcg(n, s)
        L0 = tour_len(Dl, t)
        delta = M.deltas(D, np.array(t), nb)
        voll = np.array([tour_len(Dl, z[4]) - L0 for z in zuege_listen(t)])
        assert np.abs(delta - voll).max() < 1e-12


def test_delta_felder_gleich_listen_auf_dreissig_touren_im_handbeispiel(hand_D, hand_ctx):
    for s in range(30):
        t = zufallstour_lcg(6, s)
        voll = np.array([tour_len(H.TABELLE, z[4]) - tour_len(H.TABELLE, t) for z in zuege_listen(t)], dtype=float)
        assert np.abs(M.deltas(hand_D, np.array(t), hand_ctx.nb) - voll).max() < 1e-12


@pytest.mark.parametrize("n", [6, 10, 20])
def test_apply_move_ergibt_permutation_mit_stimmender_laenge(n):
    nb = M.neighborhood(n)
    D = zufalls_D(n, 100 + n)
    t = np.array(zufallstour_lcg(n, 3))
    delta = M.deltas(D, t, nb)
    L0 = M.tour_length(D, t)
    for k in range(nb.M):
        u = M.apply_move(t, k, nb)
        assert M.is_permutation(u, n)
        assert abs(M.tour_length(D, u) - (L0 + delta[k])) < 1e-12


def test_apply_move_erzeugt_dieselben_touren_wie_die_zugliste():
    n = 8
    nb = M.neighborhood(n)
    t = zufallstour_lcg(n, 5)
    soll = [kanon(z[4]) for z in zuege_listen(t)]
    ist = [kanon(M.apply_move(np.array(t), k, nb).tolist()) for k in range(nb.M)]
    assert soll == ist


def test_blindwerte_und_kantenzahlen(hand_ctx):
    nb = hand_ctx.nb
    assert set(nb.nrem.tolist()) == {2, 3} and int((nb.nrem == 2).sum()) == 9
    assert nb.blind_add.sum() == 9 and nb.rem.shape == (93, 3, 2)


def test_rangtabelle_und_massstab_im_handbeispiel(hand_D):
    Rs = M.rank_table(hand_D)
    assert Rs[0, 1] == 0 and Rs[1, 0] == 0                    # Ort 1 ist der nächste Nachbar von Ort 0 (und umgekehrt)
    assert Rs[5, 2] == 1 and Rs[4, 3] == 0                    # von Hand: Platz von 2 unter den Nachbarn von 5 ist 3, Platz von 5 unter denen von 2 ist 1; Rs = Minimum
    assert (Rs == Rs.T).all() and (np.diag(Rs) == 5).all()      # die Diagonale (unendlich) steht hinten; sie wird nie gelesen
    assert M.scale_s(hand_D) == (1 + 1 + 4 + 6 + 4 + 2) / 6 == 3.0


def test_merkmale_des_ersten_zugs_der_nachbarlisten_reihenfolge(hand_D, hand_ctx):
    """2-opt auf den Positionen 2 und 4: entfernt (5,4) und (2,3), neu (5,2) und (4,3), Delta 5 + 6 - 4 - 10 = -3."""
    nb, t = hand_ctx.nb, np.array(H.START)
    delta, F, min_rank = M.move_features(hand_D, t, nb, hand_ctx.s, hand_ctx.Rs)
    k = int(np.flatnonzero((nb.art == 0) & (nb.a == 2) & (nb.b == 4))[0])
    assert delta[k] == 5 + 6 - 4 - 10 == -3
    assert F[k, 0] == pytest.approx((4 + 10) / 2 / 3.0) and F[k, 1] == pytest.approx(10 / 3.0)
    assert min_rank[k] == 0 and F[k, 2] == 0.0 and F[k, 3] == pytest.approx(1 / 8)
    assert F[k, 4] == 1 and F[k, 5:].sum() == 0
    assert np.abs(delta - M.deltas(hand_D, t, nb)).max() == 0


def test_merkmale_flags_und_grenzen(hand_ctx, hand_D):
    nb = hand_ctx.nb
    _, F, _ = M.move_features(hand_D, np.array(H.START), nb, hand_ctx.s, hand_ctx.Rs)
    assert F.shape == (93, 9) and (F[:, 4:8].sum(1) == 1).all()
    assert (F[:, 8] == nb.rev).all() and F[:, 8].sum() == 36 / 2 + 24 / 2
    assert F[:, 0].max() <= 6 and F[:, 1].max() <= 6 and F[:, 2:4].max() <= 1


def test_merkmale_invariant_gegen_skalieren(hand_D, hand_ctx):
    nb = hand_ctx.nb
    t = np.array(H.START)
    a = M.move_features(hand_D, t, nb, hand_ctx.s, hand_ctx.Rs)
    k5 = M.Kontext(hand_D * 5.0)
    b = M.move_features(hand_D * 5.0, t, nb, k5.s, k5.Rs)
    assert np.allclose(a[1], b[1], atol=1e-12) and (a[2] == b[2]).all()


def test_kick_ist_permutation_mit_drei_verschiedenen_schnitten():
    n = 20
    t = np.arange(n)
    us = UStream(3)
    for _ in range(50):
        u = M.kick_double_bridge(t, us)
        assert M.is_permutation(u, n) and u[0] == 0 and not (u == t).all()
    # Double-Bridge von Hand: Schnitte 2, 5, 8 aus 0..9 ergeben A C B D
    t10 = np.arange(10)
    assert np.concatenate([t10[:2], t10[5:8], t10[2:5], t10[8:]]).tolist() == [0, 1, 5, 6, 7, 2, 3, 4, 8, 9]


def test_tour_laenge_und_permutation(hand_D):
    assert M.tour_length(hand_D, [0, 1, 5, 4, 2, 3]) == 30
    assert M.is_permutation([2, 0, 1], 3) and not M.is_permutation([0, 0, 1], 3) and not M.is_permutation([0, 1], 3)


@pytest.mark.parametrize("n, soll", [(8, 108), (20, 1570), (50, 12175)])
def test_zuege_sind_zugbeschreibungen_nicht_verschiedene_nachbartouren(n, soll):
    """M(n) zählt Zugbeschreibungen; als Nachbartouren (Kantenmengen der Tour nach dem Zug) sind es weniger: 108 von 212 bei n = 8, 1 570 von 1 850 bei n = 20, 12 175 von 12 875 bei n = 50.
    Die Zählung der Prüfrunde bleibt M(n): sie wertet jede Zugbeschreibung einmal aus."""
    nb = M.neighborhood(n)
    t = np.arange(n)

    def kanten(u):
        return frozenset(frozenset((int(u[i]), int(u[(i + 1) % n]))) for i in range(n))

    touren = {kanten(M.apply_move(t, k, nb)) for k in range(nb.M)}
    assert nb.M == M.neighborhood_size(n) and len(touren) == soll < nb.M
