"""Blockweiser SplitMix64-Strom: dieselben Gleichverteilten wie `zwl_rng.SplitMix64(seed).random()`, aber in Blöcken mit numpy erzeugt.

Die Suche zieht je Auswertung zwei bis drei Zufallszahlen; mit dem skalaren Generator kostet das mehr als die Zugart selbst. numpy rechnet
vorzeichenlose 64-Bit-Zahlen modulo 2^64, darum sind die Blöcke bitgleich zur skalaren Vorlage (tests/test_rng.py prüft das). Der Zustand
nach k Ziehungen ist seed + k * GAMMA, ein Block braucht also nur den Zähler.

Schnittstelle wie die Vorlage, soweit die Suche sie braucht: `rng.random()` liefert die nächste Gleichverteilte in [0, 1)."""

import numpy as np

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)
_SCALE = 1.0 / float(1 << 53)
MASK = (1 << 64) - 1


def uniform_block(seed: int, start: int, count: int) -> np.ndarray:
    """Die Gleichverteilten Nummer start bis start + count - 1 (ab 0 gezählt) des Stroms mit dem Zufalls-Seed `seed`."""
    k = np.arange(start + 1, start + count + 1, dtype=np.uint64)
    z = k * _GAMMA + np.full(count, seed & MASK, dtype=np.uint64)
    z = (z ^ (z >> _S30)) * _M1
    z = (z ^ (z >> _S27)) * _M2
    z = z ^ (z >> _S31)
    return (z >> _S11).astype(np.float64) * _SCALE


def _stream(seed: int, block: int):
    start = 0
    while True:
        yield from uniform_block(seed, start, block).tolist()
        start += block


class BlockSplitMix64:
    """Zufallsstrom für die Suche: `random()` ist bitgleich zu `SplitMix64(seed).random()`."""

    def __init__(self, seed: int, block: int = 2048):
        self.seed = seed & MASK
        self.random = _stream(self.seed, block).__next__
