"""Planzahlen neu messen (Bau-Vormessung J7): druckt die Zahlen, die der Plan aus den Vormessungen übernommen hat, neu gerechnet mit den App-Strömen (SplitMix64).

Aufruf (Ordner der Demo, ein Prozess): python tools/zwl_planzahlen.py   # schreibt tools/_zwischen/planzahlen.json
Inhalt: AUC der fünf vorgerechneten Zugnetze (50 Validierungsinstanzen) und der Handregel, Zugauswertungen je angenommenem Zug (n = 20 mit 20 Instanzen, n = 50 mit 10, je Gewichts-Seed 0 bis 4),
Anteil der Prüfrunde (20 Instanzen je n, beide Zählweisen), E5 (Startlösungen)."""

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import zwl_constants as C  # noqa: E402
import zwl_evaluation as V  # noqa: E402
import zwl_experiments as X  # noqa: E402
import zwl_zugnetz as Z  # noqa: E402

ZW = ROOT / "tools" / "_zwischen"


def main() -> None:
    out = {}
    Fv, yv, ns = Z.validierungszeilen(C.Z_VAL_INSTANZEN_FROZEN)
    aucs = [Z.auc_tabelle(Fv, yv, X.lade_modell("Z", s)[0]) for s in range(C.MODELL_SEEDS)]
    out["auc"] = {k: [a[k] for a in aucs] for k in aucs[0]}
    out["auc_zeilen"], out["auc_zustaende"] = int(len(yv)), ns
    print("AUC Netz", np.round(out["auc"]["netz"], 4).tolist(), "Handregel", round(out["auc"]["hand"][0], 4), "Einzelmerkmale",
          {k: round(out["auc"][k][0], 4) for k in Z.EINZELMERKMALE}, flush=True)
    out["je_zug"] = {}
    for n, count in ((20, 20), (50, 10)):
        je = {r: [] for r in C.REGELN_E3}
        for s in range(C.MODELL_SEEDS):
            d = X.e1_je_zug(X.lade_modell("Z", s)[0], "gleich", n, count)
            for r in je:
                je[r].append(d[r]["je_zug"])
        out["je_zug"][str(n)] = {r: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for r, v in je.items()}
        print(f"je angenommenem Zug n = {n}:", {r: round(v[0], 1) for r, v in out["je_zug"][str(n)].items()}, flush=True)
    out["pruef"] = {}
    for n in (20, 50):
        for doppelt in (False, True):
            r = X.e2_beweis("gleich", n, 20, doppelt)
            out["pruef"][f"{n}_{'alt' if doppelt else 'neu'}"] = r
            print(f"Prüfrunde n = {n} doppelt={doppelt}: M {r['M']}, Mittel {r['mittel']:.0f} (SD {r['sd']:.0f}), Anteil {100 * r['anteil']:.1f} %", flush=True)
    for n, count in ((20, 20), (50, 10)):
        refs = V.refs_vektor("gleich", n, count) if (ROOT / "zwl_refs.json").exists() else None
        zeilen = X.e5_starts("gleich", n, count, X.lade_modell("K", 0)[0], refs)
        out[f"e5_{n}"] = zeilen
        print(f"E5 n = {n}:", [(z["art"], round(z["ersparnis"], 1), None if z["gap_start"] is None else round(z["gap_start"], 2)) for z in zeilen], flush=True)
    (ZW / "planzahlen.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
