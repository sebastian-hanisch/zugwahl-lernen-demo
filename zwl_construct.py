"""Startlösungen der Rundtour (numpy): Nächster Nachbar, Farthest Insertion, Zufallstour. Das Merkmalsnetz K als Startlösung steht in `zwl_featnet`."""

import numpy as np


def nn_tour_np(D, start: int = 0) -> np.ndarray:
    """Nächster-Nachbar-Tour ab `start`; bei gleichem Abstand gewinnt der kleinere Index (wie `zwl_scenario.nn_tour`)."""
    D = np.asarray(D, dtype=np.float64)
    n = len(D)
    besucht = np.zeros(n, dtype=bool)
    besucht[start] = True
    t = [start]
    cur = start
    for _ in range(n - 1):
        d = np.where(besucht, np.inf, D[cur])
        j = int(np.argmin(d))
        t.append(j)
        besucht[j] = True
        cur = j
    return np.array(t, dtype=np.int64)


def farthest_insertion(D, start: int = 0) -> np.ndarray:
    """Farthest Insertion: Start mit `start` und dem von ihm am weitesten entfernten Ort; dann wird der Ort mit dem größten Abstand zur Tour dort eingefügt,
    wo die Tour am wenigsten länger wird (Gleichstände: kleinster Index)."""
    D = np.asarray(D, dtype=np.float64)
    n = len(D)
    zweiter = int(np.argmax(D[start]))
    tour = [start, zweiter]
    in_tour = np.zeros(n, dtype=bool)
    in_tour[[start, zweiter]] = True
    abstand = np.minimum(D[start], D[zweiter])
    while len(tour) < n:
        c = int(np.argmax(np.where(in_tour, -1.0, abstand)))
        tt = np.array(tour)
        nxt = np.roll(tt, -1)
        zuwachs = D[tt, c] + D[c, nxt] - D[tt, nxt]
        tour.insert(int(np.argmin(zuwachs)) + 1, c)
        in_tour[c] = True
        abstand = np.minimum(abstand, D[c])
    return np.array(tour, dtype=np.int64)


def zufallstour(us, n: int) -> np.ndarray:
    """Zufällige Anordnung der Orte 0..n-1 (Rangfolge von n Gleichverteilten aus dem Strom `us`)."""
    return us.random_order(1, n)[0]
