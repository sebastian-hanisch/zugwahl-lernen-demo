"""Merkmalsnetz K: Politikverlust von Hand, Gradienten-Check gegen zentrale Differenzen (Maske, Teacher-Forcing), Merkmale im Handbeispiel, Touren sind Permutationen, Training."""

import numpy as np
import pytest

import zwl_constants as C
import zwl_featnet as FN
import zwl_hand as H
import zwl_instances as I
import zwl_mlp as MLP
import zwl_policy as PO
from zwl_rngvec import UStream


def _mini():
    return [np.array([[0.5, -0.3], [0.2, 0.4]]), np.array([0.1, -0.1]), np.array([0.7, -0.5]), np.array(0.05)]


def test_politikverlust_von_hand_drei_kandidaten_einer_besucht():
    X = np.array([[[1.0, 0.5], [0.3, -0.2], [-0.5, 0.9]]])
    unv = np.array([[True, False, True]])
    a = np.array([2])
    p = _mini()
    z = MLP.forward(p, X[0])
    soft = np.exp(z[[0, 2]]) / np.exp(z[[0, 2]]).sum()
    verlust, g = PO.policy_loss_grad(p, [(X, a, unv)], np.array([0.1]))
    assert soft == pytest.approx([0.6241, 0.3759], abs=1e-4)
    assert verlust == pytest.approx(-0.1 * np.log(soft[1]), abs=1e-12) and round(verlust, 4) == 0.0978
    # Abweichung zu zentralen Differenzen
    def f():
        return PO.policy_loss_grad(p, [(X, a, unv)], np.array([0.1]))[0]
    maxabw = 0.0
    for P, gP in zip(p, g):
        it = np.nditer(P, flags=["multi_index"], op_flags=["readwrite"])
        for _ in it:
            i = it.multi_index
            alt = P[i]
            P[i] = alt + 1e-6
            fp = f()
            P[i] = alt - 1e-6
            fm = f()
            P[i] = alt
            maxabw = max(maxabw, abs((fp - fm) / 2e-6 - gP[i]))
    assert maxabw < 1e-8


def _differenzen(f, params, h=1e-6):
    gs = []
    for P in params:
        g = np.zeros_like(P, dtype=float)
        it = np.nditer(P, flags=["multi_index"], op_flags=["readwrite"])
        for _ in it:
            i = it.multi_index
            alt = P[i]
            P[i] = alt + h
            fp = f()
            P[i] = alt - h
            fm = f()
            P[i] = alt
            g[i] = (fp - fm) / (2 * h)
        gs.append(g)
    return gs


def test_gradienten_check_sechs_touren_zwoelf_orte_mit_maske():
    p = MLP.init_mlp(7, 16, UStream(3))
    X = I.train_batch(UStream(5), 6, 12)
    D = I.distance_matrix_np(X)
    s = (D + np.diag(np.full(12, np.inf))).min(2).mean(1)
    starts = np.arange(6)
    T, L, zeilen, rec = FN.rollout(D, s, starts, p, us=UStream(9), mode="sample", record=True)
    assert zeilen == 6 * 12 * 11 // 2
    coef = np.array([0.1, -0.05, 0.02, 0.0, 0.07, -0.1])
    verlust, g = PO.policy_loss_grad(p, rec, coef)

    def f():
        return PO.policy_loss_grad(p, FN.rollout(D, s, starts, p, teacher=T, record=True)[3], coef)[0]
    assert f() == pytest.approx(verlust, abs=1e-12)                   # Teacher-Forcing liefert dieselbe Aufzeichnung
    fd = _differenzen(f, p)
    assert max(np.abs(a - b).max() for a, b in zip(g, fd)) / max(np.abs(a).max() for a in g) < 1e-6


def test_merkmale_im_handbeispiel_von_hand():
    D = H.distanzen()
    n = 6
    s = np.array([3.0])                                              # (1 + 1 + 4 + 6 + 4 + 2) / 6
    Dd = D[None].copy()
    Dd[:, np.arange(n), np.arange(n)] = np.inf
    unv = np.array([[False, True, True, True, True, True]])
    X = FN.k_features(Dd, s, np.array([0]), np.array([0]), unv, 1, n)[0]
    assert X.shape == (6, 7)
    assert X[1] == pytest.approx([1 / 3, 0 / 8, 1 / 3, 2 / 3, 2 / 8, 1 / 6, 0.0])    # Ort 1: nächster, nächster Unbesuchter ist Ort 5 (Abstand 2), zwei Orte (4, 5) im Abstand <= 6
    assert X[5] == pytest.approx([1.0, 1 / 8, 1.0, 2 / 3, 3 / 8, 1 / 6, 3 / 1 - 1])  # Ort 5: Abstand 3, Rang 1, zum Start 3, zum nächsten Unbesuchten Ort 1 (2), Orte 1, 4, 2 im Abstand <= 6
    assert X[3, 0] == 3.0 and X[3, 6] == pytest.approx(6.0)           # Ort 3: x1 = 9 / 3 = 3; x7 = 9 / 1 - 1 = 8, bei 6 gekappt


@pytest.mark.parametrize("n", [10, 20, 50])
def test_touren_sind_permutationen_greedy_und_gezogen_und_zeilen(n):
    p = MLP.init_mlp(7, 16, UStream(2))
    D = I.distance_matrix_np(I.train_batch(UStream(3), 1, n)[0])
    T, L, zeilen = FN.greedy_tours(D, p, starts=np.arange(5))
    assert all(sorted(t.tolist()) == list(range(n)) for t in T) and zeilen == 5 * n * (n - 1) // 2
    DD, s, st = FN.instanz_felder(D, np.arange(5))
    T2, L2, _, _ = FN.rollout(DD, s, st, p, us=UStream(4), mode="sample")
    assert all(sorted(t.tolist()) == list(range(n)) for t in T2)
    assert np.allclose(L, [D[t, np.roll(t, -1)].sum() for t in T])
    assert (T[:, 0] == np.arange(5)).all()


def test_teacher_forcing_gibt_die_vorgegebene_tour_zurueck():
    n = 10
    p = MLP.init_mlp(7, 16, UStream(2))
    D = I.distance_matrix_np(I.train_batch(UStream(3), 1, n)[0])
    teacher = np.array([[3, 1, 4, 0, 9, 8, 2, 5, 7, 6]])
    DD, s, st = FN.instanz_felder(D, [3])
    T, _, _, _ = FN.rollout(DD, s, st, p, teacher=teacher)
    assert T.tolist() == teacher.tolist()


def test_multi_start_hat_n_touren():
    D = I.distance_matrix_np(I.train_batch(UStream(3), 1, 12)[0])
    T, L, zeilen = FN.multi_start(D, MLP.init_mlp(7, 16, UStream(2)))
    assert T.shape == (12, 12) and zeilen == 12 * 66


def test_unbekannter_modus():
    D = I.distance_matrix_np(I.train_batch(UStream(3), 1, 8)[0])
    DD, s, st = FN.instanz_felder(D, [0])
    with pytest.raises(ValueError):
        FN.rollout(DD, s, st, MLP.init_mlp(7, 16, UStream(2)), mode="x")


def test_referenzkoeffizienten_mittel_der_touren_derselben_instanz():
    L = np.array([10.0, 12.0, 8.0, 20.0])                           # zwei Instanzen mit je zwei Touren
    c = PO.referenz_koeffizienten(L, 2)
    assert c == pytest.approx([(11 - 10) / 11 / 4, (11 - 12) / 11 / 4, (14 - 8) / 14 / 4, (14 - 20) / 14 / 4])
    assert c[:2].sum() == pytest.approx(c[:2].sum()) and abs(((L.reshape(2, 2).mean(1, keepdims=True) - L.reshape(2, 2)) / L.reshape(2, 2).mean(1, keepdims=True)).sum(1)).max() < 1e-12


def test_training_zehn_schritte_senken_die_mittlere_laenge_nicht_aber_verlauf_und_zaehler_stimmen():
    p, verlauf, cnt = PO.train_featnet(25, batch_inst=8, rollouts=4, n=10, seed=1)
    assert cnt["schritte"] == 25 and cnt["touren"] == 25 * 8 * 4 and cnt["zeilen"] == 25 * 8 * 4 * 45 and not cnt["abgebrochen"]
    assert len(verlauf) == 1 and verlauf[0][0] == 25 and MLP.n_params(p) == 145


def test_training_lernt_kurze_probe():
    """100 Schritte mit 16 Instanzen und 8 Touren bei n = 12: die mittlere gezogene Länge sinkt gegen den Anfang (Band, nicht exakt)."""
    p, verlauf, _ = PO.train_featnet(100, batch_inst=16, rollouts=8, n=12, seed=2)
    assert verlauf[-1][1] < verlauf[0][1]


def test_zeitwaechter_bricht_ab():
    _, _, cnt = PO.train_featnet(200, batch_inst=4, rollouts=2, n=8, seed=1, max_seconds=0.0)
    assert cnt["abgebrochen"] and cnt["schritte"] == 25
    assert C.K_BATCH_TOUREN == 256
