"""Zeit und Speicher in der Cloud-Näherung: ein Prozess, ein BLAS-Thread, begrenzter Speicher (Entwicklungsrechner, nicht die Streamlit Cloud; Zeit ist kein Befund).

Aufruf (Ordner der Demo): python tools/zwl_kalibrierung.py   # schreibt tools/_zwischen/kalibrierung.json und druckt die Messwerte
Gemessen: Datensammlung und Training des Zugnetzes (Voreinstellung 100 Instanzen, 1 500 Schritte; Obergrenze 300 Instanzen, 3 000 Schritte), Probetraining des Merkmalsnetzes K
(150 Schritte), ein Abstieg bei n = 50, die Live-Probe (4 Regeln x 5 Zufalls-Seeds, n = 20, k = 8), die Spitze der Speicherbelegung (tracemalloc, numpy-Zuteilungen)."""

import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import zwl_constants as C  # noqa: E402
import zwl_experiments as X  # noqa: E402
import zwl_policy as PO  # noqa: E402
import zwl_search as S  # noqa: E402

ZW = ROOT / "tools" / "_zwischen"
ZW.mkdir(parents=True, exist_ok=True)


def messen(name, f):
    tracemalloc.start()
    t0 = time.time()
    f()
    sek = time.time() - t0
    spitze = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    print(f"{name}: {sek:.1f} s, Spitze {spitze:.0f} MB", flush=True)
    return {"sekunden": round(sek, 2), "spitze_mb": round(spitze, 1)}


def main() -> None:
    out = {}
    params_z = X.lade_modell("Z", 0)[0]
    out["z_standard"] = messen("Zugnetz Voreinstellung (100 Instanzen, 1 500 Schritte)", lambda: X.trainiere_zugnetz(C.Z_SCHRITTE_STANDARD, C.Z_INSTANZEN_STANDARD, 0))
    out["z_maximum"] = messen("Zugnetz Obergrenze (300 Instanzen, 3 000 Schritte)", lambda: X.trainiere_zugnetz(C.Z_SCHRITTE_MAX, 300, 0))
    out["k_probe"] = messen("Merkmalsnetz K Probetraining (150 Schritte x 256 Touren)", lambda: PO.train_featnet(C.K_SCHRITTE_PROBE, seed=0))
    ctx50 = X.kontext("gleich", 50, 0)
    out["abstieg_n50"] = messen("ein Abstieg n = 50 (Zufall)", lambda: S.descend(ctx50, X.start_tour(ctx50), "zufall", X.suche_strom("gleich", 50, 0, 0)))
    ctx20 = X.kontext("gleich", 20, 0)
    out["probe_n20_k8"] = messen("Live-Probe n = 20, k = 8 (4 Regeln x 5 Zufalls-Seeds)", lambda: X.e3_probe(ctx20, "gleich", 0, 8, params_z))
    out["probe_n50_k2"] = messen("Live-Probe n = 50, k = 2 (4 Regeln x 5 Zufalls-Seeds)", lambda: X.e3_probe(ctx50, "gleich", 0, 2, params_z))
    out["e2_alle"] = messen("Experiment 2 (5 Größen x 20 Instanzen)", lambda: [X.e2_beweis("gleich", n, 20) for n in C.E2_N_LISTE])
    (ZW / "kalibrierung.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
