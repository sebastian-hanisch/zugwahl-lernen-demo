"""Instanzen: eingefrorene Werte der Quelle (Planmessung 2026-11-23, Rundung auf 6 Stellen), Test-/Validierungs-/Trainingsmengen, Eingabenormierung."""

import math

import numpy as np
import pytest

import zwl_constants as C
import zwl_instances as I
import zwl_scenario as S
from zwl_rngvec import UStream

EINGEFROREN = [
    ("gleich", 10, 0, [(693.476267, 983.595745), (527.951359, 154.407698)], 3319.318),
    ("gleich", 20, 0, [(943.37747, 897.216781)], 4593.652),
    ("cluster", 20, 0, [(88.936716, 828.737122)], 1919.971),
    ("gitter", 20, 0, [(88.133422, 586.130351)], 5032.548),
]


@pytest.mark.parametrize("familie,n,seed,erste,nn_laenge", EINGEFROREN)
def test_make_instance_stimmt_mit_den_eingefrorenen_werten_der_quelle(familie, n, seed, erste, nn_laenge):
    punkte, D = S.make_instance(familie, n, seed)
    assert len(punkte) == n
    for (x, y), (ex, ey) in zip(punkte, erste):
        assert abs(x - ex) < 1e-5 and abs(y - ey) < 1e-5
    assert abs(S.tour_cost(D, S.nn_tour(D, 0)) - nn_laenge) < 1e-3


def test_testmenge_validierungsmenge_haben_getrennte_zufalls_seeds():
    t, v = I.test_set(10, 5), I.val_set(10, 5)
    assert t.shape == v.shape == (5, 10, 2)
    assert np.all((0 <= t) & (t <= 1)) and np.all((0 <= v) & (v <= 1))
    assert not np.allclose(t, v)
    assert np.array_equal(t[3], I.instance_unit("gleich", 10, 3))
    assert np.array_equal(v[3], I.instance_unit("gleich", 10, C.VAL_OFFSET + 3))
    assert np.array_equal(I.test_set(10, 3), t[:3])           # die ersten Instanzen hängen nicht von der Größe des Satzes ab


def test_offsets_der_instanzmengen():
    assert C.VAL_OFFSET == 1000 and C.TRAIN_OFFSET == 2000 and C.SEED_MAX == {6: 0, 10: 19, 20: 19, 50: 9}


def test_trainingsbatch_kommt_aus_dem_blockstrom_und_ist_gleichverteilt_im_einheitsquadrat():
    X = I.train_batch(UStream(1), 400, 20)
    assert X.shape == (400, 20, 2)
    assert abs(X.mean() - 0.5) < 0.01 and abs(X.std() - math.sqrt(1 / 12)) < 0.01
    assert np.array_equal(I.train_batch(UStream(1), 400, 20), X)
    # Testinstanzen derselben Verteilung (nicht derselben Folge): Mittel und Streuung liegen im selben Band
    T = I.test_set(20, 200)
    assert abs(T.mean() - 0.5) < 0.02 and abs(T.std() - math.sqrt(1 / 12)) < 0.02
    assert not np.allclose(T[:5], X[:5])


def test_normalize_unit_ist_invariant_gegen_verschieben_und_skalieren():
    rng = np.random.default_rng(0)
    X = rng.random((6, 12, 2)) * 3 + 1
    Y = I.normalize_unit(X)
    assert Y.shape == X.shape and np.allclose(Y.min(axis=1), 0) and np.allclose(Y.max(axis=1).max(axis=1), 1)
    assert np.allclose(I.normalize_unit(X * 37.5), Y, atol=1e-12)
    assert np.allclose(I.normalize_unit(X + np.array([100.0, -50.0])), Y, atol=1e-12)
    assert np.allclose(I.normalize_unit(X[0]), Y[0])           # eine Instanz (n, 2) geht auch


def test_normalize_unit_randfaelle():
    assert np.all(I.normalize_unit(np.zeros((1, 4, 2))) == 0)             # lauter gleiche Punkte
    assert np.all(I.normalize_unit(np.array([[3.0, 4.0]])) == 0)          # ein Punkt
    stab = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 1.0]])                 # alle auf einer Geraden: Spanne in y
    assert np.allclose(I.normalize_unit(stab)[:, 1], [0, 1, 0.5])


def test_abstandsfunktionen_stimmen_ueberein():
    x = I.instance_unit("gleich", 8, 2)
    assert np.allclose(np.array(I.distance_lists(x)), I.distance_matrix_np(x), atol=1e-15)
