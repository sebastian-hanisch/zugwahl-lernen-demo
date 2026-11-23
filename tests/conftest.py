"""Gemeinsame Fixtures: das Handbeispiel (sechs Orte, Entfernungstabelle), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b), und die Zugliste als zweiter Rechenweg.

Zugliste (Weg 1, Listenoperationen) und Positionsfelder (Weg 2, `zwl_moves`) werden gegeneinander geprüft; die Zugliste liegt nur hier in den Tests."""
# ruff: noqa: E402

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import zwl_hand as H  # noqa: E402
import zwl_moves as M  # noqa: E402


@pytest.fixture
def hand_D():
    return H.distanzen()


@pytest.fixture
def hand_ctx(hand_D):
    return M.Kontext(hand_D)


def tour_len(D, t) -> float:
    """Länge einer Tour in reinem Python (zweiter Weg, kein numpy-Indexieren)."""
    return sum(D[t[k]][t[(k + 1) % len(t)]] for k in range(len(t)))


def kanon(t):
    """Kanonische Form eines Rundwegs (Start bei 0, kleinere Richtung)."""
    k = list(t).index(0)
    u = list(t)[k:] + list(t)[:k]
    v = [u[0]] + u[1:][::-1]
    return tuple(min(u, v))


def zuege_listen(t, doppelt=False):
    """Alle 2-opt- und Or-opt-Züge (Länge 1 bis 3) als (Art, a, b, rev, neue Tour). `doppelt` zählt die Einzelortverschiebung auch umgekehrt (Zählweise der Vormessung)."""
    t = list(t)
    n = len(t)
    out = []
    for i in range(n - 1):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            out.append(("2opt", i, j, 0, t[: i + 1] + t[i + 1: j + 1][::-1] + t[j + 1:]))
    for L in (1, 2, 3):
        for i in range(n):
            seg = [t[(i + k) % n] for k in range(L)]
            rest = [t[(i + L + k) % n] for k in range(n - L)]
            for pos in range(len(rest) - 1):
                for rev in (0, 1):
                    if L == 1 and rev == 1 and not doppelt:
                        continue
                    s = seg[::-1] if rev else seg
                    out.append((f"or{L}", i, pos, rev, rest[: pos + 1] + s + rest[pos + 1:]))
    return out


def zufallstour_lcg(n: int, seed: int):
    """Zufallsanordnung mit kleinem Lineargenerator (nur für die Tests, kein numpy.random)."""
    t = list(range(n))
    r = 97 + seed
    for k in range(n - 1, 0, -1):
        r = (r * 1103515245 + 12345) % (2 ** 31)
        j = r % (k + 1)
        t[k], t[j] = t[j], t[k]
    return t


def zufalls_D(n: int, seed: int) -> np.ndarray:
    """Zufallsinstanz im Einheitsquadrat aus dem Quellstrom (SplitMix64, nicht numpy.random)."""
    import zwl_instances as I
    from zwl_rngvec import UStream
    return I.distance_matrix_np(I.train_batch(UStream(seed), 1, n)[0])
