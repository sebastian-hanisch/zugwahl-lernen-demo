"""Exakte Länge der kürzesten Rundtour: Teilmengen-DP nach Held und Karp, vektorisiert nach Teilmengengröße (n bis etwa 20).

Zustand (Teilmenge S der Orte 1..n-1, letzter Ort j in S): dp[S, j] = kürzester Weg von Ort 0 über alle Orte von S, endend in j. Bei n = 20 hat das Feld 2^19 x 19 Zahlen
(rund 80 MB in float64): live nur für die gezeigte Instanz, Testsätze sind eingefroren."""

import numpy as np

_LAYER_CACHE: dict = {}


def _layers(m: int) -> list:
    """Teilmengen (als Bitmasken) von 0..m-1 nach Größe sortiert."""
    if m not in _LAYER_CACHE:
        masks = np.arange(1 << m, dtype=np.int64)
        pc = np.zeros(1 << m, dtype=np.int64)
        for j in range(m):
            pc += (masks >> j) & 1
        _LAYER_CACHE[m] = [masks[pc == p] for p in range(m + 1)]
    return _LAYER_CACHE[m]


def _dp(D: np.ndarray):
    n = D.shape[0]
    m = n - 1
    layers = _layers(m)
    dp = np.full((1 << m, m), np.inf)
    for j in range(m):
        dp[1 << j, j] = D[0, j + 1]
    Dm = D[1:, 1:]
    for p in range(2, m + 1):
        masks = layers[p]
        for j in range(m):
            sel = masks[((masks >> j) & 1) == 1]
            prev = sel ^ (1 << j)
            dp[sel, j] = (dp[prev, :] + Dm[:, j][None, :]).min(1)
    return dp, Dm


def held_karp(D) -> float:
    """Länge der kürzesten geschlossenen Rundtour für die Abstandsmatrix D (n, n), n >= 2."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if n == 2:
        return float(2.0 * D[0, 1])
    dp, _ = _dp(D)
    return float((dp[(1 << (n - 1)) - 1, :] + D[1:, 0]).min())


def held_karp_tour(D):
    """(Länge, Tour ab Ort 0) einer kürzesten Rundtour, mit Rückverfolgung über das DP-Feld."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if n == 2:
        return float(2.0 * D[0, 1]), [0, 1]
    m = n - 1
    dp, Dm = _dp(D)
    full = (1 << m) - 1
    schluss = dp[full, :] + D[1:, 0]
    j = int(schluss.argmin())
    laenge = float(schluss[j])
    rueck = [j + 1]
    mask = full
    while True:
        prev = mask ^ (1 << j)
        if prev == 0:
            break
        i = int((dp[prev, :] + Dm[:, j]).argmin())
        rueck.append(i + 1)
        mask, j = prev, i
    return laenge, [0] + rueck[::-1]


def dp_zustaende(n: int) -> int:
    """Zahl der Zustände (nichtleere Teilmenge von n - 1 Orten, letzter Ort in ihr): (n - 1) 2^(n-2)."""
    m = n - 1
    return m * (1 << (m - 1)) if m >= 1 else 0


def dp_uebergaenge(n: int) -> int:
    """Zahl der Übergänge einschließlich der m = n - 1 Abschlüsse: m (m - 1) 2^(m-2) + m (Handbeispiel n = 5: 52)."""
    m = n - 1
    return (m * (m - 1) * (1 << m)) // 4 + m
