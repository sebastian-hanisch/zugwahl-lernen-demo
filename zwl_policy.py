"""Politikgradient für das Merkmalsnetz K (Quelle für die Folge-Stücke): REINFORCE mit dem Mittel der Touren derselben Instanz (verschiedene Startorte) als Referenzwert.

Verlust einer Tour r: -c_r * Summe_k log pi(Ort_k+1 | Ort_k, Unbesucht_k) mit c_r = (b - L_r) / b (b = Mittel der Touren derselben Instanz): der Verlust sinkt, wenn kürzere Touren
wahrscheinlicher werden. Rückwärtspass von Hand über die aufgezeichneten Schritte (Gradienten-Check gegen zentrale Differenzen: tests/test_policy.py)."""

import time

import numpy as np

import zwl_constants as C
import zwl_instances as I
import zwl_mlp as MLP
import zwl_scenario as SC
from zwl_featnet import rollout
from zwl_rngvec import UStream


def policy_loss_grad(params, rec, coef: np.ndarray):
    """Verlust - sum_r coef_r sum_k log pi(a_rk) und sein Gradient [gW1, gb1, gW2, gb2]. `rec`: Aufzeichnung von `rollout(record=True)`, `coef` (R,)."""
    W1, b1, W2, b2 = params
    gW1, gb1, gW2, gb2 = np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros(())
    verlust = 0.0
    R = len(coef)
    ar = np.arange(R)
    for X, a, unv in rec:
        H = np.tanh(X @ W1.T + b1)
        z = np.where(unv, H @ W2 + b2, -np.inf)
        e = np.exp(z - z.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        verlust += float(-(coef * np.log(p[ar, a] + 1e-300)).sum())
        onehot = np.zeros_like(p)
        onehot[ar, a] = 1.0
        dz = -coef[:, None] * (onehot - p) * unv
        gW2 += np.einsum("rnh,rn->h", H, dz)
        gb2 += dz.sum()
        dpre = dz[..., None] * W2 * (1.0 - H ** 2)
        gW1 += np.einsum("rnh,rnd->hd", dpre, X)
        gb1 += dpre.sum((0, 1))
    return verlust, [gW1, gb1, gW2, gb2]


def referenz_koeffizienten(L: np.ndarray, rollouts: int) -> np.ndarray:
    """c_r / R: Vorteil (b - L) / b mit b = Mittel der `rollouts` Touren derselben Instanz, geteilt durch die Zahl aller Touren."""
    Lr = L.reshape(-1, rollouts)
    b = Lr.mean(1, keepdims=True)
    return (((b - Lr) / b).ravel()) / L.size


def reinforce_schritt(params, opt, D: np.ndarray, s: np.ndarray, starts: np.ndarray, us, rollouts: int, clip: float = C.K_CLIP) -> dict:
    """Ein Trainingsschritt: gezogene Touren (R = Instanzen * rollouts, in Instanzen gruppiert), Gradient, Adam. Gibt Verlust, mittlere Länge, Zeilen."""
    T, L, zeilen, rec = rollout(D, s, starts, params, us=us, mode="sample", record=True)
    verlust, grads = policy_loss_grad(params, rec, referenz_koeffizienten(L, rollouts))
    opt.step(params, grads, clip=clip)
    return {"verlust": verlust, "laenge": float(L.mean()), "zeilen": zeilen, "touren": len(L)}


def train_featnet(steps: int, batch_inst: int = C.K_INST, rollouts: int = C.K_ROLL, lr: float = C.K_LERNRATE, seed: int = 0, n: int = C.K_N_TRAIN,
                  progress=None, max_seconds: float | None = None):
    """Training des Merkmalsnetzes K bei n Orten: je Schritt `batch_inst` frische gleichverteilte Instanzen mit je `rollouts` gezogenen Touren (verschiedene Startorte),
    Referenzwert = Mittel dieser Touren, Adam mit Normkappung 5. Gibt (Parameter, Verlauf [(Schritt, mittlere gezogene Länge der letzten 25 Schritte)], Zähler)."""
    params = MLP.init_mlp(C.K_MERKMALE, C.H, UStream(SC.derive(C.ZWECK_K_INIT, seed)))
    opt = MLP.Adam(params, lr)
    us_i = UStream(SC.derive(C.ZWECK_K_INST, seed))
    us_s = UStream(SC.derive(C.ZWECK_K_START, seed))
    us_g = UStream(SC.derive(C.ZWECK_K_ZIEHEN, seed))
    verlauf, summe, t0, getan, abgebrochen, zeilen = [], 0.0, time.time(), 0, False, 0
    diag = np.diag(np.full(n, np.inf))
    for step in range(steps):
        Ds = I.distance_matrix_np(I.train_batch(us_i, batch_inst, n))
        ss = (Ds + diag).min(2).mean(1)
        starts = us_s.choice(batch_inst, n, rollouts).ravel()
        info = reinforce_schritt(params, opt, np.repeat(Ds, rollouts, axis=0), np.repeat(ss, rollouts), starts, us_g, rollouts)
        getan += 1
        zeilen += info["zeilen"]
        summe += info["laenge"]
        if (step + 1) % 25 == 0:
            verlauf.append((step + 1, summe / 25))
            summe = 0.0
            if progress is not None:
                progress(step + 1, steps)
            if max_seconds is not None and time.time() - t0 > max_seconds and step + 1 < steps:
                abgebrochen = True
                break
    return params, verlauf, {"schritte": getan, "touren": getan * batch_inst * rollouts, "zeilen": zeilen, "sekunden": time.time() - t0, "abgebrochen": abgebrochen}
