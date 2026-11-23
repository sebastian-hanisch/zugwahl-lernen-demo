"""Merkmalsnetz K (Konstruktion, Quelle für die Folge-Stücke, Q1): ein Netz 7-16-1 mit 145 Parametern bewertet jede noch nicht besuchte Stadt vom aktuellen Ort aus;
die Tour entsteht schrittweise (greedy: größter Wert, gezogen: Softmax über die unbesuchten). Sieben Merkmale, alle durch den Maßstab s geteilt, also unabhängig von Größe der Koordinaten.

Merkmale x1..x7 der Stadt j vom aktuellen Ort i aus: x1 = d_ij / s (bei 6 gekappt), x2 = Rang von j unter den unbesuchten Nachbarn von i (bei 8 gekappt) / 8, x3 = Abstand von j zum Start / s
(bei 6 gekappt), x4 = Abstand von j zum nächsten anderen unbesuchten Ort / s (bei 6 gekappt), x5 = Zahl unbesuchter Orte im Abstand <= 2 s von j (bei 8 gekappt) / 8, x6 = Anteil besuchter Orte,
x7 = d_ij / (Abstand zum nächsten Unbesuchten) - 1 (bei 6 gekappt). Alle Rechnungen laufen für R Touren gleichzeitig (Felder der Form (R, n, ...))."""

import numpy as np

import zwl_mlp as MLP


def k_features(Dd: np.ndarray, s: np.ndarray, cur: np.ndarray, start: np.ndarray, unv: np.ndarray, n_besucht: int, n: int) -> np.ndarray:
    """Merkmale (R, n, 7). `Dd` (R, n, n) mit Diagonale unendlich, `s` (R,), `cur` und `start` (R,) Orte, `unv` (R, n) boolesch (noch nicht besucht)."""
    R = Dd.shape[0]
    ar = np.arange(R)
    sc = s[:, None]
    dcur = Dd[ar, cur, :]
    dm = np.where(unv, dcur, np.inf)
    x1 = np.minimum(dm / sc, 6.0)
    ordn = np.argsort(dm, axis=1, kind="stable")
    rk = np.empty_like(ordn)
    np.put_along_axis(rk, ordn, np.broadcast_to(np.arange(n)[None, :], (R, n)), axis=1)
    x2 = np.minimum(rk, 8) / 8.0
    x3 = np.minimum(Dd[ar, start, :] / sc, 6.0)
    Dm = np.where(unv[:, None, :], Dd, np.inf)
    x4 = np.minimum(Dm.min(2) / sc, 6.0)
    x5 = np.minimum((Dm <= 2 * sc[:, :, None]).sum(2), 8) / 8.0
    x6 = np.full((R, n), n_besucht / n)
    dmin = dm.min(1, keepdims=True)
    x7 = np.minimum(np.where(unv, dm / np.maximum(dmin, 1e-12) - 1.0, 0.0), 6.0)
    return np.stack([x1, x2, x3, x4, x5, x6, x7], axis=-1)


def rollout(D: np.ndarray, s: np.ndarray, starts: np.ndarray, params, us=None, mode: str = "greedy", teacher: np.ndarray | None = None, record: bool = False):
    """R Touren gleichzeitig. `D` (R, n, n), `s` (R,), `starts` (R,). `mode` greedy oder sample (Gumbel-Max aus dem Strom `us`: R n Zahlen je Schritt);
    `teacher` (R, n): vorgegebene Touren (Teacher-Forcing, für den Gradienten-Check und das Lernen von Vorbildern).

    Gibt (Touren (R, n), Längen (R,), Zahl der Merkmalszeilen, Aufzeichnung für den Gradienten). Die Zeilen sind die unbesuchten Orte je Schritt: n(n-1)/2 je Tour."""
    R, n, _ = D.shape
    Dd = D.copy()
    ii = np.arange(n)
    Dd[:, ii, ii] = np.inf
    ar = np.arange(R)
    unv = np.ones((R, n), dtype=bool)
    starts = np.asarray(starts, dtype=np.int64)
    unv[ar, starts] = False
    cur = starts.copy()
    tour = [starts.copy()]
    rec, zeilen = [], 0
    for schritt in range(n - 1):
        X = k_features(Dd, s, cur, starts, unv, schritt + 1, n)
        z = np.where(unv, MLP.forward(params, X), -np.inf)
        zeilen += int(unv.sum())
        if teacher is not None:
            a = teacher[:, schritt + 1]
        elif mode == "greedy":
            a = np.argmax(z, axis=1)
        elif mode == "sample":
            a = np.argmax(z + us.gumbel(z.shape), axis=1)
        else:
            raise ValueError(f"unbekannter Modus {mode!r}")
        if record:
            rec.append((X, a.copy(), unv.copy()))
        unv[ar, a] = False
        cur = a
        tour.append(a.copy())
    T = np.stack(tour, axis=1)
    L = D[ar[:, None], T, np.roll(T, -1, axis=1)].sum(1)
    return T, L, zeilen, rec


def instanz_felder(D: np.ndarray, starts) -> tuple:
    """(D (R, n, n), s (R,), starts (R,)) für eine Instanz D (n, n) und R Startorte."""
    n = len(D)
    starts = np.atleast_1d(np.asarray(starts, dtype=np.int64))
    s = float((D + np.diag(np.full(n, np.inf))).min(axis=1).mean())
    return np.broadcast_to(D, (len(starts), n, n)).copy(), np.full(len(starts), s), starts


def greedy_tours(D: np.ndarray, params, starts=(0,)):
    """Greedy-Touren des Netzes K für eine Instanz D (n, n) ab den Startorten `starts`: (Touren (R, n), Längen (R,), Zahl der Merkmalszeilen)."""
    DD, s, st = instanz_felder(np.asarray(D, dtype=np.float64), starts)
    T, L, zeilen, _ = rollout(DD, s, st, params, mode="greedy")
    return T, L, zeilen


def multi_start(D: np.ndarray, params):
    """Greedy-Touren von allen n Startorten aus (Mehrfachstart): (Touren (n, n), Längen (n,), Zeilen)."""
    return greedy_tours(D, params, starts=np.arange(len(D)))
