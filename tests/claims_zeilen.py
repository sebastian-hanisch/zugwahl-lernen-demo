"""Zeilen der Befundtabellen der README, aus der eingefrorenen Messung gebaut: die README steht wörtlich so, `tests/test_claims.py` prüft jede Zeile.
Dieselben Funktionen druckt `python tests/claims_zeilen.py`, damit die README nicht von Hand abgeschrieben wird."""

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import zwl_constants as C  # noqa: E402
import zwl_evaluation as V  # noqa: E402

F = V.load_frozen()
R = V.load_refs()
de, vz = V.de, V.de_vz
ZELLEN_E3 = (("n20_gleich", "n = 20 gleichverteilt"), ("n50_gleich", "n = 50 gleichverteilt"), ("n50_cluster", "n = 50 Gruppen"))


def _c(c):
    return f"{vz(c['v'])} [{vz(c['lo'])}; {vz(c['hi'])}] {c['marke']}"


def e3_zeilen() -> list:
    """Hauptvergleich: je Zelle und Budget k die Lücke von Zufall und gelernt (nur Zugauswertungen / K / O) und gelernt gegen Zufall."""
    out = []
    for zelle, name in ZELLEN_E3:
        for k in F["e3"][zelle]["ks"]:
            zu = V.e3_luecken(F, zelle, "zufall", k, "frei", R)
            ge = [V.e3_luecken(F, zelle, "gelernt", k, s, R) for s in C.SICHTEN]
            cmp_ = [V.e3_vergleich(F, zelle, "gelernt", k, s, "zufall", R) for s in C.SICHTEN]
            out.append(f"| {name} | {k} | {de(zu)} % | {de(ge[0])} / {de(ge[1])} / {de(ge[2])} % | {_c(cmp_[0])} | {_c(cmp_[1])} | {_c(cmp_[2])} |")
    return out


def e3_andere_regeln_zeilen() -> list:
    """Nachbarlisten-Reihenfolge und gelernt deterministisch gegen Zufall (nur Marken, v in K), je Zelle und Budget."""
    out = []
    for zelle, name in ZELLEN_E3:
        for k in F["e3"][zelle]["ks"]:
            zeile = []
            for regel in ("nachbarliste", "gelernt_det", "steilster"):
                m = V.e3_marken(F, zelle, regel, k, "zufall", R)
                zeile.append(f"{m['frei']} / {m['K']} / {m['O']}")
            out.append(f"| {name} | {k} | " + " | ".join(zeile) + " |")
    return out


def e3_pruef_anteil_zeilen() -> list:
    return [f"| {name} | " + " | ".join(V.prozent(V.e3_pruef_anteil(F, zelle, r)) + " %" for r in ("zufall", "nachbarliste", "gelernt", "gelernt_det")) + " |" for zelle, name in ZELLEN_E3]


def e4a_zeilen() -> list:
    out = []
    for zelle, name in (("n20_gleich", "n = 20 gleichverteilt"), ("n50_gleich", "n = 50 gleichverteilt")):
        z = F["e4a"][zelle]
        for g in C.E4A_G:
            for k in z["ks"]:
                t = {x["regel"]: x for x in V.e4a_tabelle(F, zelle, g, k, R)}
                ge = t["gelernt"]
                out.append(f"| {name} | {g} | {k} | {V.prozent(t['zufall']['falsch'])} / {V.prozent(t['nachbarliste']['falsch'])} / {V.prozent(ge['falsch'])} % | "
                           f"{de(t['zufall']['frei'])} / {de(ge['frei'])} % | {_c(ge['cmp_frei'])} | {_c(ge['cmp_K'])} | {_c(ge['cmp_O'])} |")
    return out


def e4b_abstieg_zeilen() -> list:
    out = []
    for zelle, name in (("n20_gleich", "n = 20"), ("n50_gleich", "n = 50")):
        for kl in (3, 5, 8):
            a = {r: V.e4b_abstieg(F, zelle, kl, r) for r in C.REGELN_E4B}
            z = a["zufall"]
            out.append(f"| {name} | {kl} | {V.zahl(z['pruef'])} von {V.zahl(z['M'])} | {V.zahl(z['ev'])} (voll {V.zahl(z['ev_voll'])}) | {vz(z['laenge_gegen_voll'])} % | {V.prozent(z['nicht_voll'], 1)} % | "
                       f"{V.zahl(a['gelernt']['pruef'])} | {vz(a['gelernt']['laenge_gegen_voll'])} % | {V.prozent(a['gelernt']['nicht_voll'], 1)} % |")
    return out


def e4b_ils_zeilen() -> list:
    out = []
    for zelle, name in (("n20_gleich", "n = 20"), ("n50_gleich", "n = 50")):
        k = F["e4b"][zelle]["ks"][0]
        for kl in (3, 5, 8):
            voll = V.e4b_gegen_volle_menge(F, zelle, kl, "zufall", k, "frei", R)
            g = {s: V.e4b_vergleich(F, zelle, kl, "gelernt", k, s, "zufall", R) for s in C.SICHTEN}
            out.append(f"| {name} | {kl} | {k} | {_c(voll)} | {_c(g['frei'])} | {_c(g['K'])} | {_c(g['O'])} |")
    return out


def e5_zeilen() -> list:
    """Startlösungen (live neu gerechnet, Merkmalsnetz K mit Gewichts-Zufalls-Seed 0): je n alle Startarten."""
    import zwl_experiments as X

    out = []
    for n, count in ((20, 20), (50, 10)):
        for z in X.e5_starts("gleich", n, count, X.lade_modell("K", 0)[0], V.refs_vektor("gleich", n, count, R)):
            out.append(f"| {n} | {C.START_ANZEIGE[z['art']]} | {de(z['gap_start'])} % | {V.zahl(z['ev_gesamt'])} ({V.zahl(z['ev_pruef'])}) | {de(z['ersparnis'], 0)} % | "
                       f"{V.zahl(z['k_start'])} | {V.zahl(z['k_gesamt'])} | {V.zahl(z['o_gesamt'])} |")
    return out


def zaehlungen() -> dict:
    """Zählungen über alle Zellen und Budgets: in wie vielen Zellen gewinnt oder verliert die gelernte Reihenfolge je Sicht."""
    out = {s: {"schlägt": 0, "gleich": 0, "dazwischen": 0, "verliert": 0} for s in C.SICHTEN}
    zellen = 0
    for zelle, _ in ZELLEN_E3:
        for k in F["e3"][zelle]["ks"]:
            zellen += 1
            for s, m in V.e3_marken(F, zelle, "gelernt", k, "zufall", R).items():
                out[s][m] += 1
    return {"zellen": zellen, **out}


if __name__ == "__main__":
    for name, f in (("E3", e3_zeilen), ("E3 andere Regeln", e3_andere_regeln_zeilen), ("E3 Prüfanteil", e3_pruef_anteil_zeilen), ("E4a", e4a_zeilen), ("E4b Abstieg", e4b_abstieg_zeilen), ("E4b ILS", e4b_ils_zeilen)):
        print(f"--- {name}")
        print("\n".join(f()))
    print("--- Zählung", zaehlungen())
