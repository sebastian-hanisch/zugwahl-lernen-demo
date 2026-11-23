"""Gemeinsame vektorisierte Ziehung der Reihe (Quelle für alle Stücke, Q3): ein SplitMix64-Strom, Ziehungen als numpy-Arrays.

Grundlage ist `zwl_rngblock.uniform_block` (bitgleich zu `zwl_rng.SplitMix64(seed).random()`, Vorlage `hhw_rngblock`). Der Strom ist ein Zähler: die Zahl Nummer k
(ab 0 gezählt) hängt nur von Zufalls-Seed und k ab. Jede Ziehung verbraucht eine feste, dokumentierte Zahl Gleichverteilter in C-Reihenfolge:

* `uniform_array(us, shape)`: prod(shape) Gleichverteilte in [0, 1), bitgleich zum skalaren Strom.
* `normal_array(us, shape)`: Box-Muller wie `SplitMix64.normal()` (je Normalzahl zwei Gleichverteilte u1, u2 nacheinander; u1 nach unten bei 1e-12 begrenzt).
  Gleichverteilte sind bitgleich; Logarithmus und Kosinus kommen aus numpy, weichen also vom exakten `decimal`-Weg der Vorlage höchstens in der letzten Binärstelle ab (Test: 1e-12).
* `gumbel_array(us, shape)`: -log(-log(u)) mit u nach unten bei 1e-12 begrenzt (eine Gleichverteilte je Wert; Gumbel-Max-Ziehung).
* `random_order(us, m, n)`: m zufällige Anordnungen von 0..n-1 (Sortieren von m * n Gleichverteilten, stabil), Form (m, n).
* `choice_without_replacement(us, m, n, k)`: je Zeile k verschiedene Zahlen aus 0..n-1 (die ersten k von `random_order`), Form (m, k).
* `UStream(seed)`: der Strom mit Position; `us.uniform(shape)`, `us.normal(shape)`, `us.gumbel(shape)`, `us.random_order(m, n)`, `us.choice(m, n, k)`, `us.position`, `us.skip(k)`.

Kein `numpy.random`."""

import math

import numpy as np

from zwl_rngblock import MASK, uniform_block

UMIN = 1e-12                                   # untere Grenze vor dem Logarithmus (wie `SplitMix64.normal()`)


class UStream:
    """Strom gleichverteilter Zahlen: jeder Aufruf liest die nächsten Zahlen des SplitMix64-Stroms `seed`."""

    def __init__(self, seed: int):
        self.seed = int(seed) & MASK
        self.position = 0

    def uniform(self, shape) -> np.ndarray:
        """Die nächsten prod(shape) Gleichverteilten in C-Reihenfolge als Array der Form `shape`."""
        shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(int(s) for s in shape)
        count = int(math.prod(shape))
        out = uniform_block(self.seed, self.position, count).reshape(shape)
        self.position += count
        return out

    def skip(self, k: int) -> None:
        """Die nächsten k Zahlen überspringen (Position vorrücken)."""
        self.position += int(k)

    def normal(self, shape) -> np.ndarray:
        return normal_array(self, shape)

    def gumbel(self, shape) -> np.ndarray:
        return gumbel_array(self, shape)

    def random_order(self, m: int, n: int) -> np.ndarray:
        return random_order(self, m, n)

    def choice(self, m: int, n: int, k: int) -> np.ndarray:
        return choice_without_replacement(self, m, n, k)


def uniform_array(us: UStream, shape) -> np.ndarray:
    """Gleichverteilte in [0, 1) der Form `shape` (verbraucht prod(shape) Zahlen)."""
    return us.uniform(shape)


def normal_array(us: UStream, shape) -> np.ndarray:
    """Standardnormalverteilte der Form `shape` (Box-Muller, verbraucht 2 * prod(shape) Zahlen: je Wert u1, u2 nacheinander)."""
    shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(int(s) for s in shape)
    count = int(math.prod(shape))
    u = us.uniform((count, 2))
    u1 = np.maximum(u[:, 0], UMIN)
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * math.pi * u[:, 1])
    return z.reshape(shape)


def gumbel_array(us: UStream, shape) -> np.ndarray:
    """Standard-Gumbel-Verteilte der Form `shape` (-log(-log(u)), verbraucht prod(shape) Zahlen)."""
    u = np.maximum(us.uniform(shape), UMIN)
    return -np.log(-np.log(u))


def random_order(us: UStream, m: int, n: int) -> np.ndarray:
    """m zufällige Anordnungen von 0..n-1, Form (m, n): Rangfolge von m * n Gleichverteilten (stabil sortiert; Gleichstand praktisch ausgeschlossen)."""
    keys = us.uniform((m, n))
    return np.argsort(keys, axis=1, kind="stable").astype(np.int64)


def choice_without_replacement(us: UStream, m: int, n: int, k: int) -> np.ndarray:
    """Je Zeile k verschiedene Zahlen aus 0..n-1, Form (m, k) (die ersten k Spalten von `random_order`; verbraucht m * n Zahlen)."""
    if k > n:
        raise ValueError("k größer als n")
    return random_order(us, m, n)[:, :k]
