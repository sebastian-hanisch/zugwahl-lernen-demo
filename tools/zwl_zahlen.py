"""Druckt alle Zahlen der eingefrorenen Messreihen (E3, E4a, E4b, Prüfrunden-Anteil bei Kicks) für README und Bericht; rechnet nichts neu.

Aufruf (Ordner der Demo): python tools/zwl_zahlen.py"""

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import zwl_constants as C  # noqa: E402
import zwl_evaluation as V  # noqa: E402

F, R = V.load_frozen(), V.load_refs()


def zeile(c):
    return "Gegner" if c is None else f"{V.de_vz(c['v'])} [{V.de_vz(c['lo'])}; {V.de_vz(c['hi'])}] {c['marke']}"


def main() -> None:
    for zelle, z in F["e3"].items():
        for k in z["ks"]:
            print(f"\nE3 {zelle} k = {k} ({z['instanzen']} Instanzen x {z['zufalls_seeds']} Seeds)")
            for t in V.e3_tabelle(F, zelle, k, R):
                print(f"  {t['regel']:13s} frei {V.de(t['frei'])} K {V.de(t['K'])} O {V.de(t['O'])} | frei {zeile(t['cmp_frei'])} | K {zeile(t['cmp_K'])} | O {zeile(t['cmp_O'])}")
            m = V.e3_marken(F, zelle, "gelernt", k, "zufall", R)
            print("  Verdikt:", V.verdict_zugwahl(m, z["n"], z["familie"], k, z["instanzen"], z["zufalls_seeds"])[1][:120])
            print("  Gewichts-Streuung gelernt:", {s: [V.de(x) for x in V.e3_gewichts_streuung(F, zelle, "gelernt", k, s, R)] for s in C.SICHTEN})
        print("  Anteil Prüfrunde bei Kicks:", {r: V.prozent(V.e3_pruef_anteil(F, zelle, r)) for r in ("zufall", "nachbarliste", "gelernt", "gelernt_det")})
    for zelle, z in F["e4a"].items():
        for g in C.E4A_G:
            for k in z["ks"]:
                print(f"\nE4a {zelle} Abbruch {g} k = {k} ({z['instanzen']} Instanzen x {z['zufalls_seeds']} Seeds)")
                for t in V.e4a_tabelle(F, zelle, g, k, R):
                    print(f"  {t['regel']:13s} falsch {V.prozent(t['falsch'])} % | frei {V.de(t['frei'])} K {V.de(t['K'])} O {V.de(t['O'])} | frei {zeile(t['cmp_frei'])} | K {zeile(t['cmp_K'])} | O {zeile(t['cmp_O'])}")
    for zelle, z in F["e4b"].items():
        for kl in (3, 5, 8):
            print(f"\nE4b {zelle} Nachbarliste {kl}")
            for r in C.REGELN_E4B:
                a = V.e4b_abstieg(F, zelle, kl, r)
                print(f"  {r:8s} Prüfrunde {a['pruef']:.0f} von {a['M']}, ev {a['ev']:.0f} (voll {a['ev_voll']:.0f}), Länge {V.de_vz(a['laenge_gegen_voll'])} %, nicht voll {V.prozent(a['nicht_voll'], 1)} %")
            for k in z["ks"]:
                m = {s: V.e4b_vergleich(F, zelle, kl, "gelernt", k, s, "zufall", R) for s in C.SICHTEN}
                print(f"  ILS k = {k}, gelernt gegen Zufall (beide mit Liste):", {s: zeile(c) for s, c in m.items()})
                g = {s: V.e4b_gegen_volle_menge(F, zelle, kl, "zufall", k, s, R) for s in C.SICHTEN}
                print(f"  ILS k = {k}, Zufall mit Liste gegen Zufall voll:", {s: zeile(c) for s, c in g.items()})


if __name__ == "__main__":
    main()
