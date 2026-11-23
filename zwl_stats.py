"""Statistik des Vergleichs: Lücke in Prozent, gepaarte Differenz, Bootstrap-Intervall über Instanzen (SplitMix64-Blockstrom), Klassen, Zählung „unter dem Gegner“.

v = mittlere Lücke des Netzes minus mittlere Lücke des Gegners in Prozentpunkten, gepaart je Instanz; negativ heißt: das Netz ist besser."""

from functools import lru_cache

import numpy as np

import zwl_constants as C
from zwl_rngblock import uniform_block


def gap(laenge, referenz):
    """Lücke in Prozent über dem Referenzwert (Zahl oder Array)."""
    return 100.0 * (np.asarray(laenge, dtype=np.float64) - referenz) / referenz


def paired_diff(a, b) -> np.ndarray:
    """Differenz je Instanz a - b (gleich viele Werte)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("a und b brauchen gleich viele Werte")
    return a - b


@lru_cache(maxsize=8)
def _boot_index(seed: int, m: int, ziehungen: int) -> np.ndarray:
    return (uniform_block(seed, 0, ziehungen * m) * m).astype(np.int32).reshape(ziehungen, m)


def bootstrap_ci(werte, seed: int = C.ZWECK_BOOT, ziehungen: int = C.BOOTSTRAP_ZIEHUNGEN):
    """95-%-Intervall des Mittels (Perzentile 2,5 und 97,5 von `ziehungen` Ziehungen mit Zurücklegen); None bei weniger als zwei Werten."""
    x = np.asarray(werte, dtype=np.float64)
    m = len(x)
    if m < 2:
        return None
    mittel = x[_boot_index(seed, m, ziehungen)].mean(axis=1)
    return float(np.percentile(mittel, 2.5)), float(np.percentile(mittel, 97.5))


def klasse(v: float, lo: float, hi: float) -> str:
    """'klar' bei |v| >= 0,25 und Intervall ohne 0; 'gleich' bei |v| < 0,10 oder Intervall mit 0; sonst 'klein'."""
    ohne_null = lo > 0 or hi < 0
    if abs(v) >= C.MARKE_KLAR and ohne_null:
        return "klar"
    if abs(v) < C.MARKE_GLEICH or not ohne_null:
        return "gleich"
    return "klein"


def compare(gap_netz, gap_gegner, seed: int = C.ZWECK_BOOT) -> dict:
    """Gepaarter Vergleich über Instanzen: v (Prozentpunkte), Intervall, Klasse, Zahl der Instanzen m. Ohne Intervall (m < 2): lo, hi, klasse None."""
    d = paired_diff(gap_netz, gap_gegner)
    m = len(d)
    v = float(d.mean()) if m else 0.0
    ci = bootstrap_ci(d, seed)
    if ci is None:
        return {"v": v, "lo": None, "hi": None, "klasse": None, "m": m}
    return {"v": v, "lo": ci[0], "hi": ci[1], "klasse": klasse(v, ci[0], ci[1]), "m": m}


def unter_gegner(gap_netz, gap_gegner, seed: int = C.ZWECK_BOOT) -> bool:
    """Liegt das Netz unter dem Gegner? Ja, wenn die mittlere Lücke kleiner ist und das gepaarte Bootstrap-Intervall der Differenz 0 ausschließt."""
    r = compare(gap_netz, gap_gegner, seed)
    return r["lo"] is not None and r["v"] < 0 and r["hi"] < 0


def zaehle_unter(flags) -> tuple[int, int]:
    """(Zahl der Seeds unter dem Gegner, Zahl der Seeds)."""
    flags = list(flags)
    return sum(1 for f in flags if f), len(flags)
