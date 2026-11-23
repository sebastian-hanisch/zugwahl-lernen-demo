"""Zugmenge der Lokalsuche: 2-opt und Or-opt (Segmentlänge 1 bis 3), Delta nach Kantenformel, Merkmale der Züge, Anwenden eines Zugs, Kick.

Quelle für die Folge-Stücke (Kopie heißt `<präfix>_moves.py`). Die Zugmenge ist **duplikatfrei** (Q2): die Verschiebung eines Einzelorts zählt nur einmal,
M(n) = n (11 n - 35) / 2 (n = 20: 1 850, n = 50: 12 875). Die Zählweise der Vormessung (Einzelortverschiebung in beiden Richtungen) hat M(n) + n (n - 2).

Eine Tour ist ein numpy-Feld t (Position -> Ort). Ein Zug ist ein Index k in der Zugmenge `Neighborhood(n)`; sie hält je Zug die Positionspaare der
entfernten und der neuen Kanten (drei Paare, Blindwert = Position n mit Entfernung 0), so dass das Delta aller Züge ein Aufruf ohne Schleife ist:

    Delta_k = sum(Entfernung der neuen Kanten) - sum(Entfernung der entfernten Kanten).

Reihenfolge der Züge: erst alle 2-opt-Züge (i < j, j - i >= 2, nicht (0, n-1)), dann Or-opt Länge 1, 2, 3 (je Länge: Anfangsposition i, Zielstelle x, Umkehr)."""

import numpy as np

import zwl_constants as C


# ---------------------------------------------------------------------------------------------------------------- Größe der Zugmenge
def m_arten(n: int) -> dict:
    """Zahl der Züge je Art: 2-opt n(n-3)/2, Or-opt 1: n(n-2), Or-opt 2: 2n(n-3), Or-opt 3: 2n(n-4)."""
    return {"2-opt": n * (n - 3) // 2, "Or-opt 1": n * (n - 2), "Or-opt 2": 2 * n * (n - 3), "Or-opt 3": 2 * n * (n - 4)}


def neighborhood_size(n: int, doppelt: bool = False) -> int:
    """M(n): Zahl der verschiedenen Züge einer vollen Prüfrunde. Mit `doppelt=True` die Zählweise der Vormessung (Einzelortverschiebung auch umgekehrt): M(n) + n(n-2)."""
    m = sum(m_arten(n).values())
    return m + (n * (n - 2) if doppelt else 0)


# ---------------------------------------------------------------------------------------------------------------- Zugmenge als Positionsfelder
class Neighborhood:
    """Alle Züge einer Tour mit n Orten als Felder: Art (0 = 2-opt, 1 bis 3 = Länge des Or-opt-Segments), Beschreibung (a, b, rev) und die Kanten als Positionspaare."""

    def __init__(self, n: int, doppelt: bool = False):
        if n < 5:
            raise ValueError("mindestens fünf Orte")
        self.n = n
        self.doppelt = doppelt
        art, a, b, rev, rem, add = [], [], [], [], [], []
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                art.append(0), a.append(i), b.append(j), rev.append(0)
                rem.append([(i, i + 1), (j, (j + 1) % n), (n, n)])
                add.append([(i, j), (i + 1, (j + 1) % n), (n, n)])
        for L in C.OR_LAENGEN:
            for i in range(n):
                s0, sl, p, q = i, (i + L - 1) % n, (i - 1) % n, (i + L) % n
                for x in range(n - L - 1):
                    u = (i + L + x) % n
                    v = (u + 1) % n
                    for r in (0, 1):
                        if L == 1 and r == 1 and not doppelt:     # doppelt: Zählweise der Vormessung (dieselbe Verschiebung in beiden Richtungen)
                            continue
                        art.append(L), a.append(i), b.append(x), rev.append(r)
                        rem.append([(p, s0), (sl, q), (u, v)])
                        add.append([(p, q), (u, sl) if r else (u, s0), (s0, v) if r else (sl, v)])
        self.art = np.array(art, dtype=np.int64)
        self.a = np.array(a, dtype=np.int64)
        self.b = np.array(b, dtype=np.int64)
        self.rev = np.array(rev, dtype=bool)
        self.rem = np.array(rem, dtype=np.int64)            # (M, 3, 2)
        self.add = np.array(add, dtype=np.int64)            # (M, 3, 2)
        self.M = len(self.art)
        self.blind_rem = self.rem[:, :, 0] == n             # (M, 3): Blindwert (2-opt hat nur zwei Kanten)
        self.blind_add = self.add[:, :, 0] == n
        self.nrem = (~self.blind_rem).sum(1)                # entfernte Kanten je Zug: 2 oder 3
        # „neue“ Kante: nicht Blindwert und nicht dieselbe Kante, die der Zug zugleich entfernt (ein Or-opt mit u = q und Umkehr setzt die Kante (sl, q) zurück)
        gleich = np.zeros_like(self.blind_add)
        for i in range(3):
            for j in range(3):
                a0, a1, r0, r1 = self.add[:, i, 0], self.add[:, i, 1], self.rem[:, j, 0], self.rem[:, j, 1]
                gleich[:, i] |= ((a0 == r0) & (a1 == r1) | (a0 == r1) & (a1 == r0)) & ~self.blind_rem[:, j]
        self.add_neu = ~self.blind_add & ~gleich            # (M, 3)
        w = n + 1
        self.rem_flat = self.rem[:, :, 0] * w + self.rem[:, :, 1]
        self.add_flat = self.add[:, :, 0] * w + self.add[:, :, 1]
        self.is_2opt = self.art == 0


_NB: dict = {}


def neighborhood(n: int, doppelt: bool = False) -> Neighborhood:
    """Zwischengespeicherte Zugmenge für n Orte (`doppelt`: Zählweise der Vormessung mit M(n) + n(n-2) Zügen)."""
    if (n, doppelt) not in _NB:
        _NB[(n, doppelt)] = Neighborhood(n, doppelt)
    return _NB[(n, doppelt)]


# ---------------------------------------------------------------------------------------------------------------- Instanzen-Hilfen
def tour_length(D: np.ndarray, t) -> float:
    """Länge der Rundtour t (volle Neuberechnung)."""
    t = np.asarray(t)
    return float(D[t, np.roll(t, -1)].sum())


def scale_s(D: np.ndarray) -> float:
    """s = mittlerer Abstand eines Ortes zu seinem nächsten Nachbarn (Maßstab der Merkmale)."""
    n = len(D)
    return float((D + np.diag(np.full(n, np.inf))).min(axis=1).mean())


def rank_table(D: np.ndarray) -> np.ndarray:
    """Rs[u, v] = kleinerer der beiden Richtungsränge (Platz von v unter den Nachbarn von u nach Entfernung, 0 = nächster; stabil bei Gleichstand)."""
    n = len(D)
    Dd = D + np.diag(np.full(n, np.inf))
    ordn = np.argsort(Dd, axis=1, kind="stable")
    rk = np.empty((n, n), dtype=np.int64)
    rk[np.arange(n)[:, None], ordn] = np.arange(n)[None, :]
    return np.minimum(rk, rk.T)


class Kontext:
    """Eine Instanz mit allem, was Suche und Netz brauchen: Entfernungen D (numpy), Maßstab s, Rangtabelle Rs und die Zugmenge."""

    def __init__(self, D, doppelt: bool = False):
        self.D = np.asarray(D, dtype=np.float64)
        self.n = len(self.D)
        self.s = scale_s(self.D)
        self.Rs = rank_table(self.D)
        self.nb = neighborhood(self.n, doppelt)


# ---------------------------------------------------------------------------------------------------------------- Delta und Merkmale
def _padded(D: np.ndarray, t: np.ndarray) -> np.ndarray:
    n = len(t)
    P = np.zeros((n + 1, n + 1))
    P[:n, :n] = D[np.ix_(t, t)]
    return P


def deltas(D: np.ndarray, t, nb: Neighborhood) -> np.ndarray:
    """Delta aller M Züge der Tour t (negativ = der Zug verkürzt die Tour), Form (M,)."""
    t = np.asarray(t)
    P = _padded(D, t).ravel()
    return P[nb.add_flat].sum(1) - P[nb.rem_flat].sum(1)


def move_features(D: np.ndarray, t, nb: Neighborhood, s: float, Rs: np.ndarray):
    """(Delta (M,), Merkmale F (M, 9), kleinster Rang der **neuen** Kanten (M,) ohne Kappung für die Nachbarliste).
    Die Merkmale 2 und 3 nehmen alle hinzugefügten Kanten; „neue“ Kanten (Nachbarliste) sind die hinzugefügten ohne die, die der Zug zugleich entfernt
    (ein Or-opt mit u = q und Umkehr setzt die Kante (sl, q) zurück).

    Merkmale des Zugnetzes Z (Delta selbst ist kein Merkmal): 0 mittlere Länge der entfernten Kanten / s (bei 6 gekappt), 1 größte Länge der entfernten Kanten / s (bei 6 gekappt),
    2 kleinster Rang der neuen Kanten (bei 8 gekappt) / 8, 3 größter Rang der neuen Kanten (bei 8 gekappt) / 8, 4 ist 2-opt, 5 bis 7 Or-opt Länge 1 bis 3, 8 Segment umgekehrt."""
    t = np.asarray(t)
    n = nb.n
    P = _padded(D, t).ravel()
    rem_len = P[nb.rem_flat]
    delta = P[nb.add_flat].sum(1) - rem_len.sum(1)
    Rpad = np.zeros((n + 1, n + 1), dtype=np.int64)
    Rpad[:n, :n] = Rs[np.ix_(t, t)]
    ranks = Rpad.ravel()[nb.add_flat]
    min_rank_neu = np.where(nb.add_neu, ranks, 1 << 20).min(1)
    min_rank = np.where(nb.blind_add, 1 << 20, ranks).min(1)         # Merkmale: alle hinzugefügten Kanten (wie die Vormessung)
    max_rank = np.where(nb.blind_add, 0, ranks).max(1)
    F = np.zeros((nb.M, C.Z_MERKMALE))
    F[:, 0] = np.minimum(rem_len.sum(1) / nb.nrem / s, C.LAENGEN_KAPPE)
    F[:, 1] = np.minimum(rem_len.max(1) / s, C.LAENGEN_KAPPE)
    F[:, 2] = np.minimum(min_rank, C.RANG_KAPPE) / C.RANG_KAPPE
    F[:, 3] = np.minimum(max_rank, C.RANG_KAPPE) / C.RANG_KAPPE
    F[nb.is_2opt, 4] = 1.0
    for L in C.OR_LAENGEN:
        F[nb.art == L, 4 + L] = 1.0
    F[:, 8] = nb.rev
    return delta, F, min_rank_neu


# ---------------------------------------------------------------------------------------------------------------- Anwenden, Kick
def apply_move(t, k: int, nb: Neighborhood) -> np.ndarray:
    """Neue Tour nach Zug k. 2-opt kehrt die Positionen a+1..b um; Or-opt setzt das Segment ab Position a (Länge L) hinter die Zielstelle b der übrigen Orte (rotierte Tour)."""
    t = np.asarray(t)
    art, a, b, rev = int(nb.art[k]), int(nb.a[k]), int(nb.b[k]), bool(nb.rev[k])
    if art == 0:
        return np.concatenate([t[: a + 1], t[a + 1: b + 1][::-1], t[b + 1:]])
    u = np.roll(t, -a)
    seg, rest = u[:art], u[art:]
    if rev:
        seg = seg[::-1]
    return np.concatenate([rest[: b + 1], seg, rest[b + 1:]])


def kick_double_bridge(t, us) -> np.ndarray:
    """Double-Bridge: drei verschiedene Schnittstellen 1 <= p1 < p2 < p3 <= n-1 (aus dem Strom `us`, verbraucht n-1 Zahlen); Tour A B C D wird zu A C B D."""
    t = np.asarray(t)
    n = len(t)
    cuts = np.sort(us.choice(1, n - 1, 3)[0] + 1)
    p1, p2, p3 = (int(c) for c in cuts)
    return np.concatenate([t[:p1], t[p2:p3], t[p1:p2], t[p3:]])


def is_permutation(t, n: int) -> bool:
    """Ist t eine Anordnung der Orte 0..n-1?"""
    t = np.asarray(t)
    return len(t) == n and sorted(t.tolist()) == list(range(n))
