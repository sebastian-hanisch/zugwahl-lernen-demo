"""Zugnetz Z: AUC (Paarzählung gegen Rangsumme), Datensammlung, Training, vorgerechnete Modelle, Startlösungen."""

import numpy as np
import pytest
from conftest import zufalls_D

import zwl_constants as C
import zwl_construct as CO
import zwl_mlp as MLP
import zwl_scenario as SC
import zwl_zugnetz as Z
from zwl_rngvec import UStream


def test_auc_beispiel_von_hand_und_rangsumme():
    pos, neg = [0.9, 0.8, 0.35], [0.7, 0.35, 0.3, 0.2, 0.1]
    gewinn = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    assert gewinn == 13.5 and Z.auc(pos + neg, [1] * 3 + [0] * 5) == pytest.approx(0.9, abs=1e-15) == pytest.approx(13.5 / 15)


def test_auc_gegen_scipy_und_paarzaehlung_auf_zufallswerten():
    from scipy.stats import rankdata  # nur Test (requirements-dev)
    us = UStream(3)
    for _ in range(5):
        score = np.round(us.uniform(300) * 10) / 10                    # viele Gleichstände
        y = (us.uniform(300) < 0.3).astype(float)
        r = rankdata(score)
        n1, n0 = int(y.sum()), int((1 - y).sum())
        rang = (r[y > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        a, b = score[y > 0], score[y == 0]
        paare = ((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (n1 * n0)
        assert Z.auc(score, y) == pytest.approx(rang, abs=1e-12) == pytest.approx(paare, abs=1e-12)


def test_auc_randfaelle():
    assert Z.auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0 and Z.auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0 and Z.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    with pytest.raises(ValueError):
        Z.auc([1, 2], [1, 1])


def test_instanzen_sind_getrennt_test_val_train():
    t = Z.kontext_von("gleich", 20, 0).D
    assert not np.array_equal(t, Z.val_instanzen(1)[0].D) and not np.array_equal(t, Z.train_instanzen(1)[0].D)
    assert not np.array_equal(Z.val_instanzen(1)[0].D, Z.train_instanzen(1)[0].D)
    _, D = SC.make_instance("gleich", 20, 1000)
    assert np.array_equal(Z.val_instanzen(1)[0].D, np.array(D))


def test_datensammlung_zeilen_etiketten_und_praefix_stabilitaet():
    kt = Z.train_instanzen(4)
    F, y, ns = Z.sammle_zustaende(kt[:2], k_budget=2)
    assert F.shape == (ns * C.Z_PRO_ZUSTAND, 9) and y.shape == (ns * C.Z_PRO_ZUSTAND,) and set(np.unique(y)) <= {0.0, 1.0}
    F2, y2, ns2 = Z.sammle_zustaende(kt, k_budget=2)
    assert ns2 > ns and np.array_equal(F2[: len(F)], F) and np.array_equal(y2[: len(y)], y)       # die ersten Instanzen liefern dieselben Zeilen
    assert 0.003 < y2.mean() < 0.05                                                                # etwa 1,5 % der Züge verbessern (Vormessung 1,51 %)


def test_etiketten_stimmen_mit_dem_delta_ueberein():
    """Ein Zustand: die gesammelten Merkmale und Etiketten stammen aus derselben Zeile (Delta < 0 <-> y = 1), geprüft über den Anteil und die Merkmale der verbessernden Zeilen."""
    kt = Z.train_instanzen(3)
    F, y, _ = Z.sammle_zustaende(kt, k_budget=3)
    assert F[y == 1][:, 0].mean() > F[y == 0][:, 0].mean()                                         # verbessernde Züge entfernen im Mittel längere Kanten


def test_training_senkt_den_verlust_und_zaehlt():
    kt = Z.train_instanzen(20)
    F, y, _ = Z.sammle_zustaende(kt, k_budget=3)
    p, verlauf, cnt = Z.train_zugnetz(F, y, 200, seed=0)
    assert cnt == {"schritte": 200, "zeilen": 200 * 4096, "sekunden": cnt["sekunden"], "abgebrochen": False}
    assert verlauf[-1][1] < verlauf[0][1] and MLP.n_params(p) == 177


def test_training_ist_wiederholbar_im_band_und_haengt_am_seed():
    F, y, _ = Z.sammle_zustaende(Z.train_instanzen(10), k_budget=3)
    a, _, _ = Z.train_zugnetz(F, y, 60, seed=0)
    b, _, _ = Z.train_zugnetz(F, y, 60, seed=0)
    c, _, _ = Z.train_zugnetz(F, y, 60, seed=1)
    assert np.allclose(a[0], b[0], rtol=1e-9, atol=1e-12) and not np.allclose(a[0], c[0])


def test_zeitwaechter_des_zugnetz_trainings():
    F, y, _ = Z.sammle_zustaende(Z.train_instanzen(5), k_budget=2)
    _, _, cnt = Z.train_zugnetz(F, y, 500, max_seconds=0.0)
    assert cnt["abgebrochen"] and cnt["schritte"] == 50


def test_vorgerechnete_modelle_format_parameterzahl_und_guete_als_band():
    Fv, yv, _ = Z.validierungszeilen(10)
    for seed in range(C.MODELL_SEEDS):
        p, meta = MLP.load_model(MLP.model_path("Z", seed))
        assert MLP.n_params(p) == 177 and meta["kind"] == "Z" and meta["seed"] == seed and meta["steps"] == 3000
        tab = Z.auc_tabelle(Fv, yv, p)
        assert 0.95 < tab["netz"] < 0.995 and tab["netz"] > tab["hand"] + 0.05
        assert 0.80 < tab["hand"] < 0.93 and all(0.7 < tab[k] < 0.9 for k in Z.EINZELMERKMALE)


def test_rauchtest_1500_schritte_auc_ueber_handregel():
    kt = Z.train_instanzen(C.Z_INSTANZEN_STANDARD)
    F, y, _ = Z.sammle_zustaende(kt)
    p, _, _ = Z.train_zugnetz(F, y, C.Z_SCHRITTE_STANDARD, seed=0)
    Fv, yv, _ = Z.validierungszeilen(10)
    tab = Z.auc_tabelle(Fv, yv, p)
    assert 0.94 <= tab["netz"] <= 0.995 and tab["netz"] > tab["hand"] + 0.04


def test_handregel_wert_hoher_rang_schlecht():
    F = np.zeros((2, 9))
    F[0, 3], F[1, 3] = 1 / 8, 3 / 8
    assert Z.hand_score(F)[0] > Z.hand_score(F)[1]


# ---------------------------------------------------------------------------------------------------------------- Startlösungen
def test_nn_und_farthest_insertion_permutationen():
    for n in (8, 20, 50):
        D = zufalls_D(n, n)
        for t in (CO.nn_tour_np(D, 0), CO.farthest_insertion(D, 0), CO.zufallstour(UStream(1), n)):
            assert sorted(t.tolist()) == list(range(n))
    D = zufalls_D(12, 3)
    assert CO.nn_tour_np(D, 4).tolist() == SC.nn_tour(D.tolist(), 4)


def test_farthest_insertion_im_handbeispiel_von_hand():
    import zwl_hand as H
    t = CO.farthest_insertion(H.distanzen(), 0)
    # Start 0, weitester Ort ist 3 (9); dann Ort 2 (Abstand zur Tour {0, 3}: min(8, 10) = 8), Zuwachs an beiden Kanten 8 + 10 - 9 = 9 (Gleichstand: erste Stelle); weitere Orte kommen dazwischen
    assert t[0] == 0 and sorted(t.tolist()) == list(range(6)) and 3 in t.tolist()
    assert sum(H.TABELLE[t[k]][t[(k + 1) % 6]] for k in range(6)) >= H.OPTIMUM
