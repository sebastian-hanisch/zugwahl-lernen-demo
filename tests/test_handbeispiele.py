"""Die Prüfungen aus handbeispiele_04.py als echte Tests (gleiche enge Toleranzen): Teil C (AUC, Gumbel-Reihenfolge) und Teil D (Zählregel im Kleinen und als Umrechnung der Vormessung).
Teil A (sechs Orte) steht in test_moves.py und test_search.py, Teil B (Mini-Netz, Politikverlust) in test_mlp.py und test_policy.py."""

import math

import numpy as np
import pytest

import zwl_constants as C
import zwl_cost as K
import zwl_hand as H
import zwl_moves as M
import zwl_rngvec as RV


def test_tabelle_ist_symmetrisch_mit_diagonale_null_und_gerundete_entfernungen():
    t = H.TABELLE
    assert all(t[i][j] == t[j][i] for i in range(6) for j in range(6)) and all(t[i][i] == 0 for i in range(6))
    assert all(t[i][j] == round(math.dist(H.PUNKTE[i], H.PUNKTE[j])) for i in range(6) for j in range(6))


def test_nachster_nachbar_und_konstruktionskosten_von_hand():
    D = H.TABELLE
    t, frei = [0], set(range(1, 6))
    kandidaten = 0
    while frei:
        kandidaten += len(frei)
        j = min(sorted(frei), key=lambda x: D[t[-1]][x])
        t.append(j)
        frei.remove(j)
    assert t == list(H.START) and kandidaten == 5 + 4 + 3 + 2 + 1 == K.k_konstruktion(6) == 15


def test_gumbel_reihenfolge_zwei_kandidaten_gleich_softmax_quadratur():
    z1, z2, m = 0.8, -0.3, 400000
    u = (np.arange(m) + 0.5) / m
    g1 = -np.log(-np.log(u))
    quad = float(np.mean(np.exp(-np.exp(-(z1 + g1 - z2)))))          # P(z1 + g1 > z2 + g2) = E_u[F(z1 + g1 - z2)], F(x) = exp(-exp(-x))
    soft = 1 / (1 + math.exp(-(z1 - z2)))
    assert round(soft, 4) == 0.7503 and abs(quad - soft) < 2e-4


def test_gumbel_ziehung_der_demo_haeufigkeiten_gegen_plackett_luce():
    """Reihenfolge nach z + Gumbel aus dem SplitMix64-Strom: Häufigkeit jeder Reihenfolge von drei Kandidaten gegen das Plackett-Luce-Produkt (Band über 20 000 Ziehungen)."""
    z = np.array([0.8, -0.3, 0.1])
    g = RV.UStream(5).gumbel((20000, 3))
    ordnung = np.argsort(-(z + g), axis=1, kind="stable")
    e = np.exp(z)
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        p = 1.0
        rest = list(range(3))
        for c in perm:
            p *= e[c] / e[rest].sum()
            rest.remove(c)
        haeufigkeit = float(np.mean((ordnung == np.array(perm)).all(axis=1)))
        assert abs(haeufigkeit - p) < 0.012, (perm, haeufigkeit, p)
    assert all(sorted(row.tolist()) == [0, 1, 2] for row in ordnung[:100])


def test_zaehlregel_im_kleinen_eine_runde_bei_sechs_orten():
    M6 = M.neighborhood_size(6)
    assert M6 == 93 and C.O_JE_ZEILE_Z == 176 and C.O_JE_ZEILE_K == 144
    assert K.k_gesamt(1, M6) == 94 and K.k_gesamt(7, 0) == 7                                       # Netz-Reihenfolge: 93 Zeilen + 1 Auswertung; Aufzählung: 7
    assert K.o_gesamt("gelernt", 1, M6) == 93 * 176 + 1 * 8 == 16376 and K.o_gesamt("aufzaehlung", 7, 0) == 7 * 8 == 56


def test_umrechnung_der_vormessung_tabelle_6_in_k_und_o():
    """n = 20, erster Abstieg aus Nächster Nachbar (Mittel): Zugauswertungen und Merkmalszeilen aus der Vormessung, in K und O umgerechnet (reine Arithmetik)."""
    t6 = {"zufall": (3361, 0), "nachbarliste": (2260, 12367), "gelernt": (2101, 13820), "gelernt_det": (2077, 10508)}
    k = {r: K.k_gesamt(ev, zl) for r, (ev, zl) in t6.items()}
    o = {r: K.o_gesamt(r, ev, zl) for r, (ev, zl) in t6.items()}
    assert (k["zufall"], k["nachbarliste"], k["gelernt"], k["gelernt_det"]) == (3361, 14627, 15921, 12585)
    assert o["zufall"] == 26888 and o["gelernt"] == 2449128 and o["gelernt_det"] == 8 * 2077 + 176 * 10508 == 1866024
    assert o["gelernt"] // o["zufall"] == 91 and o["gelernt"] / o["zufall"] >= 50                  # nur die Zehnerpotenz ist belastbar
    assert o["nachbarliste"] == 8 * (2260 + 12367) == 117016                                       # Zeile der Nachbarlisten-Reihenfolge mit 8 Operationen (Untergrenze, Annahme der Demo)


def test_konstruktion_190_und_1225_und_start_kosten():
    assert K.k_konstruktion(20) == 190 and K.k_konstruktion(50) == 1225
    assert K.start_kosten("nn", 20) == (190, 1520) and K.start_kosten("zufall", 20) == (1, 8) and K.start_kosten("optimum", 20) == (0, 0)
    assert K.start_kosten("netz_k", 20) == (190, 144 * 190) and K.start_kosten("farthest", 20) == (190, 4 * 400)
    with pytest.raises(ValueError):
        K.start_kosten("gibtesnicht", 20)


def test_zeilengewichte_der_sichten():
    assert K.zeilen_gewicht("gelernt", "frei") == 0 and K.zeilen_gewicht("gelernt", "K") == 1 and K.zeilen_gewicht("gelernt", "O") == 22
    assert K.zeilen_gewicht("nachbarliste", "O") == 1 and K.zeilen_gewicht("zufall", "O") == 0 and K.zeilen_gewicht("steilster", "K") == 0
    assert K.kosten("gelernt", 100, 10, "O") == 100 + 22 * 10 and K.ops_zeilen("gelernt", 3) == 528 and K.ops_zeilen("zufall", 3) == 0


def test_formel_und_zaehlungen_der_zugmenge():
    for n in (20, 50):
        assert K.m_verschiedene_zuege(n) == M.neighborhood_size(n)
    assert K.zeilen_featnet(20) == 190 and K.zeilen_featnet(6) == 15
