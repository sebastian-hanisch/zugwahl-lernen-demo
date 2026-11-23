"""Instanzen für das Netz: Test-, Validierungs- und Trainingsmengen im Einheitsquadrat, das Handbeispiel und die Eingabenormierung.

Test- und Validierungsinstanzen kommen aus `zwl_scenario.make_instance("gleich", n, seed)` (Quelle hyperheuristik-demo, 1000er-Quadrat, hier durch 1000 geteilt);
Trainingsinstanzen aus dem Blockstrom (`UStream`). Beides ist die Gleichverteilung im Einheitsquadrat, aber nicht dieselbe Folge (Test `test_instances`).
Das Netz bekommt absolute Koordinaten: der Aufrufer normiert mit `normalize_unit`, wenn er Skala und Lage ausschließen will (Stück 5)."""

import math

import numpy as np

import zwl_constants as C
import zwl_scenario as S
from zwl_rngvec import UStream


def scale_unit(punkte) -> np.ndarray:
    """Punkte im 1000er-Quadrat (Liste von (x, y)) -> Array der Form (n, 2) im Einheitsquadrat."""
    return np.asarray(punkte, dtype=np.float64) / C.SEITE


def instance_unit(familie: str, n: int, seed: int) -> np.ndarray:
    """Eine Instanz der Quelle als (n, 2)-Array im Einheitsquadrat."""
    punkte, _ = S.make_instance(familie, n, seed)
    return scale_unit(punkte)


def test_set(n: int, count: int, familie: str = "gleich") -> np.ndarray:
    """Testinstanzen, Form (count, n, 2): `make_instance(familie, n, seed)` für Zufalls-Seed 0 bis count - 1."""
    return np.stack([instance_unit(familie, n, s) for s in range(count)])


def val_set(n: int, count: int, familie: str = "gleich") -> np.ndarray:
    """Validierungsinstanzen, Form (count, n, 2): Zufalls-Seeds ab 1000 (nie dieselben wie im Testsatz)."""
    return np.stack([instance_unit(familie, n, C.VAL_OFFSET + s) for s in range(count)])


def train_batch(us: UStream, m: int, n: int) -> np.ndarray:
    """m frische Trainingsinstanzen, Form (m, n, 2), gleichverteilt im Einheitsquadrat, direkt aus dem Blockstrom (m * n * 2 Zahlen in C-Reihenfolge)."""
    return us.uniform((m, n, 2))


def hand_instance() -> np.ndarray:
    """Das Handbeispiel (Rechteck 4 x 3 mit Mitte), Form (5, 2), Koordinaten durch 4 geteilt: die Eingabe des Mini-Netzes."""
    return np.asarray(C.HAND_PUNKTE, dtype=np.float64) / C.HAND_SKALA


def normalize_unit(X: np.ndarray) -> np.ndarray:
    """Eingabenormierung je Instanz: (X - Minimum je Achse) / größte Spannweite. Wirkt auf (n, 2) oder (M, n, 2).

    Invariant gegen Verschieben und positives Skalieren der Koordinaten (bis auf Rundung); eine Instanz mit nur einem Punkt
    oder lauter gleichen Punkten (Spannweite 0) wird auf Nullen abgebildet."""
    X = np.asarray(X, dtype=np.float64)
    einzeln = X.ndim == 2
    if einzeln:
        X = X[None]
    mn = X.min(axis=1, keepdims=True)
    spanne = (X.max(axis=1, keepdims=True) - mn).max(axis=2, keepdims=True)
    spanne = np.where(spanne > 0, spanne, 1.0)
    out = (X - mn) / spanne
    return out[0] if einzeln else out


def distance_matrix_np(X: np.ndarray) -> np.ndarray:
    """Abstandsmatrizen (M, n, n) oder (n, n) aus Koordinaten (numpy)."""
    diff = X[..., :, None, :] - X[..., None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def distance_lists(x: np.ndarray) -> list[list[float]]:
    """Abstände einer Instanz als Liste von Listen (für die Gegner in reinem Python, `math.sqrt` wie die Quelle)."""
    sqrt = math.sqrt
    pts = [(float(a), float(b)) for a, b in x]
    return [[sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by)) for bx, by in pts] for ax, ay in pts]
