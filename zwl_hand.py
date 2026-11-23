"""Handbeispiel: sechs Orte als Entfernungstabelle (ganze Zahlen, gerundete euklidische Entfernungen der Punkte (2,0), (3,0), (9,3), (1,9), (6,5), (4,2)).

Von Hand nachrechenbar: Nächster Nachbar ab Ort 0 ergibt 0, 1, 5, 4, 2, 3 (Länge 30), das Optimum ist 27, die Zugmenge hat M(6) = 93 Züge, 8 davon verbessern am Start."""

import numpy as np

TABELLE = (
    (0, 1, 8, 9, 6, 3),
    (1, 0, 7, 9, 6, 2),
    (8, 7, 0, 10, 4, 5),
    (9, 9, 10, 0, 6, 8),
    (6, 6, 4, 6, 0, 4),
    (3, 2, 5, 8, 4, 0),
)
PUNKTE = ((2, 0), (3, 0), (9, 3), (1, 9), (6, 5), (4, 2))   # Herkunft der Tabelle (die App zeigt die Tabelle)
START = (0, 1, 5, 4, 2, 3)                                  # Nächster Nachbar ab Ort 0, Länge 30
SCHLECHTER_START = (0, 2, 1, 3, 5, 4)                       # Länge 42
OPTIMUM = 27
N = 6


def distanzen() -> np.ndarray:
    """Die Tabelle als float-Feld (6, 6)."""
    return np.array(TABELLE, dtype=np.float64)
