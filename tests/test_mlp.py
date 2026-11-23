"""Netz von Hand: Mini-Netz 2-2-1 (Teil B des Handbeispiels) Vorwärts- und Rückwärtspass gegen Handrechnung und zentrale Differenzen, 9-16-1 und 7-16-1, Adam, Modelldatei."""

import math

import numpy as np
import pytest

import zwl_constants as C
import zwl_mlp as MLP
from zwl_rngvec import UStream


def mini():
    return [np.array([[0.5, -0.3], [0.2, 0.4]]), np.array([0.1, -0.1]), np.array([0.7, -0.5]), np.array(0.05)]


def zentral(f, params, h=1e-6):
    gs = []
    for P in params:
        g = np.zeros_like(P, dtype=float)
        it = np.nditer(P, flags=["multi_index"], op_flags=["readwrite"])
        for _ in it:
            idx = it.multi_index
            alt = P[idx]
            P[idx] = alt + h
            fp = f()
            P[idx] = alt - h
            fm = f()
            P[idx] = alt
            g[idx] = (fp - fm) / (2 * h)
        gs.append(g)
    return gs


def test_vorwaertspass_von_hand():
    h1, h2 = math.tanh(0.5 * 1.0 - 0.3 * 0.5 + 0.1), math.tanh(0.2 * 1.0 + 0.4 * 0.5 - 0.1)
    assert h1 == pytest.approx(0.4218990053, abs=1e-9) and h2 == pytest.approx(0.2913126125, abs=1e-9)
    z = 0.7 * h1 - 0.5 * h2 + 0.05
    assert round(z, 4) == 0.1997
    X, y = np.array([[1.0, 0.5]]), np.array([1.0])
    assert MLP.forward(mini(), X)[0] == pytest.approx(z, abs=1e-12)
    verlust, _ = MLP.bce_loss_grad(mini(), X, y)
    assert round(float(MLP.sigmoid(np.array(z))), 4) == 0.5498 and round(verlust, 4) == 0.5983
    assert verlust == pytest.approx(math.log(1 + math.exp(z)) - z, abs=1e-12)


def test_rueckwaertspass_von_hand_und_gegen_differenzen():
    X, y, p = np.array([[1.0, 0.5]]), np.array([1.0]), mini()
    h1, h2 = math.tanh(0.45), math.tanh(0.30)
    z = 0.7 * h1 - 0.5 * h2 + 0.05
    dz = 1 / (1 + math.exp(-z)) - 1.0
    _, g = MLP.bce_loss_grad(p, X, y)
    assert g[2] == pytest.approx([dz * h1, dz * h2], abs=1e-12) and g[3] == pytest.approx(dz, abs=1e-12)
    dp1, dp2 = dz * 0.7 * (1 - h1 ** 2), dz * (-0.5) * (1 - h2 ** 2)
    assert np.allclose(g[0], [[dp1, dp1 * 0.5], [dp2, dp2 * 0.5]], atol=1e-12) and g[1] == pytest.approx([dp1, dp2], abs=1e-12)
    assert round(dz, 4) == -0.4502 and round(g[2][0], 4) == -0.1900 and round(g[2][1], 4) == -0.1312
    assert round(g[0][0][0], 4) == -0.2591 and round(g[0][0][1], 4) == -0.1295
    fd = zentral(lambda: MLP.bce_loss_grad(p, X, y)[0], p)
    assert max(np.abs(a - b).max() for a, b in zip(g, fd)) < 1e-8


def test_mittel_ueber_zeilen_ist_faktor_1_durch_n():
    p = mini()
    X, y = np.array([[1.0, 0.5], [-0.4, 0.8]]), np.array([1.0, 0.0])
    v, g = MLP.bce_loss_grad(p, X, y)
    va, ga = MLP.bce_loss_grad(p, X[:1], y[:1])
    vb, gb = MLP.bce_loss_grad(p, X[1:], y[1:])
    assert v == pytest.approx((va + vb) / 2, abs=1e-12)
    assert all(np.allclose(a, (b + c) / 2, atol=1e-12) for a, b, c in zip(g, ga, gb))


@pytest.mark.parametrize("d", [C.Z_MERKMALE, C.K_MERKMALE])
def test_gradienten_check_voller_groesse(d):
    p = MLP.init_mlp(d, C.H, UStream(7))
    us = UStream(8)
    X, y = us.uniform((5, d)), (us.uniform(5) > 0.5).astype(float)
    _, g = MLP.bce_loss_grad(p, X, y)
    fd = zentral(lambda: MLP.bce_loss_grad(p, X, y)[0], p)
    groesst = max(np.abs(a).max() for a in g)
    assert max(np.abs(a - b).max() for a, b in zip(g, fd)) / groesst < 1e-6


def test_parameterzahlen_und_form():
    z, k = MLP.init_mlp(9, 16, UStream(1)), MLP.init_mlp(7, 16, UStream(1))
    assert MLP.n_params(z) == 177 == C.Z_PARAMS and MLP.n_params(k) == 145 == C.K_PARAMS
    assert [a.shape for a in z] == [(16, 9), (16,), (16,), ()] and (z[1] == 0).all() and z[3] == 0
    assert 9 * 16 + 16 + 16 == C.O_JE_ZEILE_Z == 176 and 7 * 16 + 16 + 16 == C.O_JE_ZEILE_K == 144


def test_anfangsgewichte_sind_wiederholbar_und_haengen_am_seed():
    a, b, c = MLP.init_mlp(9, 16, UStream(1)), MLP.init_mlp(9, 16, UStream(1)), MLP.init_mlp(9, 16, UStream(2))
    assert all((x == y).all() for x, y in zip(a, b)) and not (a[0] == c[0]).all()
    assert abs(a[0].std() - 1 / 3) < 0.1 and abs(a[2].std() - 0.075) < 0.04


def test_adam_erster_schritt_und_kappung():
    p = [np.array([1.0, 2.0])]
    opt = MLP.Adam(p, lr=0.1)
    norm = opt.step(p, [np.array([3.0, -4.0])])
    assert norm == pytest.approx(5.0) and p[0] == pytest.approx([1.0 - 0.1, 2.0 + 0.1], abs=1e-6)      # erster Schritt: lr * Vorzeichen
    q = [np.array([0.0, 0.0])]
    o2 = MLP.Adam(q, lr=0.1)
    o2.step(q, [np.array([30.0, 40.0])], clip=5.0)
    assert q[0] == pytest.approx([-0.1, -0.1], abs=1e-6) and o2.m[0] == pytest.approx([0.1 * 3.0, 0.1 * 4.0], abs=1e-12)    # gekappter Gradient (3, 4)


def test_training_senkt_den_verlust():
    us = UStream(3)
    X = us.uniform((400, 9))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(float)
    p = MLP.init_mlp(9, 16, UStream(4))
    opt = MLP.Adam(p, 0.02)
    v0 = MLP.bce_loss_grad(p, X, y)[0]
    for _ in range(200):
        _, g = MLP.bce_loss_grad(p, X, y)
        opt.step(p, g)
    assert MLP.bce_loss_grad(p, X, y)[0] < 0.5 * v0


def test_modelldatei_rundreise_und_pruefungen(tmp_path):
    p = MLP.init_mlp(9, 16, UStream(5))
    pfad = tmp_path / "m.npz"
    MLP.save_model(pfad, p, {"kind": "Z", "seed": 3, "steps": 10, "lr": 0.01})
    q, meta = MLP.load_model(pfad)
    assert all((a == b).all() for a, b in zip(p, q)) and meta["seed"] == 3 and meta["kind"] == "Z" and meta["format"] == "zwl-mlp-1"
    MLP.save_model(tmp_path / "falsch.npz", p, {"kind": "K"})                  # Z-Gewichte als K gespeichert: falsche Form
    with pytest.raises(ValueError):
        MLP.load_model(tmp_path / "falsch.npz")
    np.savez(tmp_path / "alt.npz", W1=p[0], b1=p[1], W2=p[2], b2=p[3], format=np.array("anders"), kind=np.array("Z"))
    with pytest.raises(ValueError):
        MLP.load_model(tmp_path / "alt.npz")
    nan = [a.copy() for a in p]
    nan[1][0] = np.nan
    MLP.save_model(tmp_path / "nan.npz", nan, {"kind": "Z"})
    with pytest.raises(ValueError):
        MLP.load_model(tmp_path / "nan.npz")


def test_modellpfade():
    assert MLP.model_path("Z", 2).name == "zugnetz_s2.npz" and MLP.model_path("K", 0).name == "featnet_s0.npz"
    assert MLP.model_path("Z", 0).parent.name == C.MODELL_ORDNER
