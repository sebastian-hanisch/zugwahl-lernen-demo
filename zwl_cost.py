"""Zählung der Linie für Stück 4 (Zuschnitt 2, Abschnitt 10; Q5): K = Kandidatenbewertungen, O = grobe Operationen.

Die Formeln der Konstruktion und der Zugmenge haben dieselben Namen und Werte wie `znz_cost` (Stück 1); dazu die Zeilen der beiden Netze und der Startlösungen.
K begünstigt das Netz (eine Merkmalszeile des Netzes ist in Operationen viel teurer als eine Zugauswertung), O benachteiligt es. Eine Konstruktion kostet nie 1:
n(n-1)/2 Kandidatenbewertungen (Nächster Nachbar, Netz K, Farthest Insertion nach Annahme); eine Zufallstour kostet 1. Das Wort „TLA“ kommt nirgends vor."""

import zwl_constants as C


def k_konstruktion(n: int) -> int:
    """Kandidatenbewertungen einer Konstruktion (Nächster Nachbar oder Netz): n (n - 1) / 2."""
    return n * (n - 1) // 2


def m_verschiedene_zuege(n: int) -> int:
    """Verschiedene Züge einer vollen Prüfrunde (2-opt und Or-opt, duplikatfrei, Q2): n (11 n - 35) / 2; n = 20: 1 850, n = 50: 12 875."""
    return n * (11 * n - 35) // 2


def ops_kandidaten(k: float) -> float:
    """O für k Zugauswertungen oder Kandidatenbewertungen der klassischen Verfahren: 8 je Stück (Annahme)."""
    return C.OPS_JE_KANDIDAT * k


def ops_zeilen(regel: str, zeilen: float) -> float:
    """O für Merkmalszeilen: Zugnetz Z 176 je Zeile, Nachbarlisten-Reihenfolge 8 je Zeile (Annahme, Untergrenze), sonst 0 (Merkmalsbildung nicht mitgezählt)."""
    if regel in C.REGELN_MIT_NETZ:
        return C.O_JE_ZEILE_Z * zeilen
    if regel == "nachbarliste":
        return C.O_JE_ZEILE_HAND * zeilen
    return 0.0


def zeilen_gewicht(regel: str, sicht: str) -> float:
    """Gewicht einer Merkmalszeile in Zugauswertungen je Sicht: frei 0, K 1, O 176 / 8 = 22 (Zugnetz) bzw. 1 (Nachbarlisten-Reihenfolge)."""
    if regel in C.REGELN_MIT_NETZ:
        return C.SICHT_GEWICHT_Z[sicht]
    if regel == "nachbarliste":
        return C.SICHT_GEWICHT_HAND[sicht]
    return 0.0


def kosten(regel: str, ev: float, zeilen: float, sicht: str) -> float:
    """Kosten in Einheiten einer Zugauswertung (O: Operationen geteilt durch 8): Zugauswertungen plus gewichtete Merkmalszeilen."""
    return ev + zeilen_gewicht(regel, sicht) * zeilen


def k_gesamt(ev: float, zeilen: float) -> float:
    """K = Zugauswertungen (mit Konstruktion und Kicks) plus Merkmalszeilen."""
    return ev + zeilen


def o_gesamt(regel: str, ev: float, zeilen: float) -> float:
    """O = 8 je Zugauswertung plus Operationen der Zeilen (Zugnetz 176, Nachbarlisten-Reihenfolge 8)."""
    return ops_kandidaten(ev) + ops_zeilen(regel, zeilen)


def zeilen_featnet(n: int) -> int:
    """Merkmalszeilen einer Tour des Merkmalsnetzes K (greedy): n(n-1)/2 (je Schritt alle unbesuchten Orte)."""
    return n * (n - 1) // 2


def k_start_netz(n: int) -> int:
    """K der Startlösung aus dem Merkmalsnetz K: n(n-1)/2 Merkmalszeilen, die Konstruktion zählt als eine Kandidatenbewertung je Zeile."""
    return zeilen_featnet(n)


def o_start_netz(n: int) -> int:
    """O der Startlösung aus dem Merkmalsnetz K: 144 je Zeile, Merkmalsbildung nicht mitgezählt (Untergrenze)."""
    return C.O_JE_ZEILE_K * zeilen_featnet(n)


def start_kosten(art: str, n: int) -> tuple[int, int]:
    """(K, O) der Konstruktion der Startlösung. Zufallstour 1 und 8, Nächster Nachbar n(n-1)/2 und 8 n(n-1)/2, Farthest Insertion n(n-1)/2 und 4 n^2 (Annahme),
    Merkmalsnetz K n(n-1)/2 und 144 n(n-1)/2; „schon Lokalminimum“ ist die Obergrenze der Ersparnis und kostet 0."""
    if art == "zufall":
        return 1, int(ops_kandidaten(1))
    if art == "nn":
        return k_konstruktion(n), int(ops_kandidaten(k_konstruktion(n)))
    if art == "farthest":
        return k_konstruktion(n), C.O_FARTHEST_FAKTOR * n * n
    if art == "netz_k":
        return k_start_netz(n), o_start_netz(n)
    if art == "optimum":
        return 0, 0
    raise ValueError(art)
