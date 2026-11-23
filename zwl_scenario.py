"""Instanzen der Rundtour: drei erzeugte Familien, das Handbeispiel, Abstandsmatrix, Tourlänge und Startlösung.

Zufall nur über SplitMix64 (zwl_rng); jeder Zweck (Instanz, Startlösung, Lauf, Referenz) hat einen eigenen Zufalls-Seed aus `derive`.
Abstände mit `math.sqrt` (nach IEEE 754 exakt gerundet, auf jeder Plattform dieselben Bits)."""

import math

import zwl_constants as C
from zwl_rng import MASK, SplitMix64

HAND_PUNKTE = ((0.0, 0.0), (4.0, 0.0), (8.0, 0.0), (8.0, 3.0), (4.0, 3.0), (0.0, 3.0))   # Rand eines Rechtecks 8 x 3
HAND_START = (0, 2, 1, 3, 4, 5)                                                           # eine Kreuzung, Länge 28
HAND_OPTIMUM = 22.0                                                                       # Umfang des Rechtecks


def derive(zweck: int, *schluessel: int) -> int:
    """Zufalls-Seed für einen Zweck und seine Schlüssel (Familie, n, Instanz, Lauf): SplitMix64-Mischen je Schlüssel, kein hash()."""
    x = SplitMix64(zweck & MASK).next_u64()
    for k in schluessel:
        x = SplitMix64((x ^ ((int(k) + 0x9E3779B97F4A7C15) & MASK)) & MASK).next_u64()
    return x


def distance_matrix(punkte) -> list[list[float]]:
    """Euklidische Abstände als Liste von Listen (schneller Zugriff in der Suchschleife)."""
    sqrt = math.sqrt
    return [[sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by)) for bx, by in punkte] for ax, ay in punkte]


def _punkte_gleich(n, rng):
    return [(rng.random() * C.SEITE, rng.random() * C.SEITE) for _ in range(n)]


def _punkte_cluster(n, rng):
    """Fünf Gauß-Haufen mit Streuung 40, Mittelpunkte gleichverteilt in [100, 900]^2."""
    breite = C.SEITE - 200.0
    mitten = [(100.0 + rng.random() * breite, 100.0 + rng.random() * breite) for _ in range(C.CLUSTER_ZAHL)]
    punkte = []
    for _ in range(n):
        mx, my = mitten[rng.below(C.CLUSTER_ZAHL)]
        punkte.append((mx + C.CLUSTER_SIGMA * rng.normal(), my + C.CLUSTER_SIGMA * rng.normal()))
    return punkte


def gitter_zellen(n, rng) -> tuple[int, list[int]]:
    """Gitterweite g = ceil(sqrt(1,5 n)) und n verschiedene Zellen (keine Zelle doppelt)."""
    g = int(math.ceil(math.sqrt(C.GITTER_FAKTOR * n)))
    zellen = list(range(g * g))
    rng.shuffle(zellen)
    return g, zellen[:n]


def _punkte_gitter(n, rng):
    g, zellen = gitter_zellen(n, rng)
    sp = C.SEITE / g
    return [((z % g + 0.5) * sp + C.GITTER_RAUSCHEN * sp * rng.normal(), (z // g + 0.5) * sp + C.GITTER_RAUSCHEN * sp * rng.normal()) for z in zellen]


_ERZEUGER = {"gleich": _punkte_gleich, "cluster": _punkte_cluster, "gitter": _punkte_gitter}


def make_instance(familie: str, n: int, seed: int):
    """Instanz (Punkte, Abstandsmatrix). `seed` 0 bis 29: Instanzen der App und der Messung; ab 1000: Probeinstanzen."""
    if familie == C.FAMILIE_HAND:
        return hand_instance()
    rng = SplitMix64(derive(C.ZWECK_INSTANZ, C.FAMILIEN.index(familie), n, seed))
    punkte = _ERZEUGER[familie](n, rng)
    return punkte, distance_matrix(punkte)


def hand_instance():
    """Das Handbeispiel: sechs Orte auf dem Rand eines Rechtecks 8 x 3."""
    punkte = list(HAND_PUNKTE)
    return punkte, distance_matrix(punkte)


def tour_cost(D, t) -> float:
    """Länge der Rundtour t (volle Neubewertung, zyklisch)."""
    s = 0.0
    prev = t[-1]
    for c in t:
        s += D[prev][c]
        prev = c
    return s


def nn_tour(D, start: int) -> list[int]:
    """Nächster-Nachbar-Tour ab `start`; bei gleichem Abstand gewinnt der kleinere Index."""
    n = len(D)
    seen = [False] * n
    t = [start]
    seen[start] = True
    cur = start
    for _ in range(n - 1):
        row = D[cur]
        best = -1
        bd = 1e300
        for j in range(n):
            if not seen[j] and row[j] < bd:
                bd = row[j]
                best = j
        t.append(best)
        seen[best] = True
        cur = best
    return t


def nn_start(D, rng) -> list[int]:
    """Startlösung: Nächster-Nachbar-Tour ab einem zufälligen Ort."""
    return nn_tour(D, rng.below(len(D)))


def start_tour(D, familie: str, n: int, seed: int, lauf: int) -> list[int]:
    """Gemeinsame Startlösung je (Instanz, Lauf), für alle Varianten gleich; das Handbeispiel startet immer mit seiner Kreuzungstour."""
    if familie == C.FAMILIE_HAND:
        return list(HAND_START)
    return nn_start(D, SplitMix64(derive(C.ZWECK_START, C.FAMILIEN.index(familie), n, seed, lauf)))


def run_seed(familie: str, n: int, seed: int, lauf: int) -> int:
    """Zufalls-Seed des Suchlaufs je (Instanz, Lauf); alle Varianten eines Laufs lesen denselben Strom von vorn."""
    fi = 9 if familie == C.FAMILIE_HAND else C.FAMILIEN.index(familie)
    return derive(C.ZWECK_LAUF, fi, n, seed, lauf)
