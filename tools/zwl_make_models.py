"""Vorgerechnete Modelle: Zugnetz Z (zwl_modelle/zugnetz_s0..4.npz) und Merkmalsnetz K (zwl_modelle/featnet_s0..4.npz).

Aufruf (vom Ordner der Demo aus, ein Prozess, BLAS-Threads 1):
  python tools/zwl_make_models.py Z            # 5 Gewichts-Zufalls-Seeds, 3 000 Schritte x 4 096 Zeilen, 300 Trainingsinstanzen (rund 1 min)
  python tools/zwl_make_models.py K [seeds]    # 3 750 Schritte x 256 Touren = 960 000 Touren je Modell (rund 3 bis 10 min je Modell), wiederaufnehmbar
Schreibt nur nach zwl_modelle/ und tools/_zwischen/ (Protokoll, Validierung). Referenzen der Validierung: Held-Karp (n = 20, 50 Instanzen ab Zufalls-Seed 1000)."""

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import zwl_constants as C  # noqa: E402
import zwl_exact as EX  # noqa: E402
import zwl_featnet as FN  # noqa: E402
import zwl_mlp as MLP  # noqa: E402
import zwl_policy as PO  # noqa: E402
import zwl_zugnetz as Z  # noqa: E402

ZW = ROOT / "tools" / "_zwischen"
ZW.mkdir(parents=True, exist_ok=True)
(ROOT / C.MODELL_ORDNER).mkdir(exist_ok=True)


def log(text: str) -> None:
    print(text, flush=True)
    with open(ZW / "models_log.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def val_refs(count: int = 50) -> np.ndarray:
    """Held-Karp-Längen der Validierungsinstanzen (n = 20 gleich, Zufalls-Seed ab 1000), zwischengespeichert."""
    p = ZW / f"val_ref_{count}.json"
    if p.exists():
        return np.array(json.loads(p.read_text()))
    refs = [EX.held_karp(k.D) for k in Z.val_instanzen(count)]
    p.write_text(json.dumps(refs))
    return np.array(refs)


def machen_z() -> None:
    t0 = time.time()
    F, y, ns = Z.sammle_zustaende(Z.train_instanzen(C.Z_INSTANZEN_FROZEN))
    log(f"Z Daten: {C.Z_INSTANZEN_FROZEN} Instanzen, {ns} Zustände, {len(y)} Zeilen, Anteil verbessernd {y.mean():.4f}, {time.time() - t0:.0f} s")
    Fv, yv, nsv = Z.validierungszeilen(C.Z_VAL_INSTANZEN_FROZEN)
    log(f"Z Validierung: {C.Z_VAL_INSTANZEN_FROZEN} Instanzen, {nsv} Zustände, {len(yv)} Zeilen, Anteil verbessernd {yv.mean():.4f}")
    for seed in range(C.MODELL_SEEDS):
        params, verlauf, cnt = Z.train_zugnetz(F, y, C.Z_SCHRITTE_FROZEN, seed=seed)
        tab = Z.auc_tabelle(Fv, yv, params)
        meta = {"kind": "Z", "seed": seed, "steps": cnt["schritte"], "lr": C.Z_LERNRATE, "n_train_rows": cnt["zeilen"], "n_train_instances": C.Z_INSTANZEN_FROZEN,
                "feature_version": C.FEATURE_VERSION, "val_auc": tab["netz"], "val_auc_hand": tab["hand"], "n": C.Z_N_TRAIN}
        MLP.save_model(MLP.model_path("Z", seed, ROOT), params, meta)
        log(f"Z Gewichts-Seed {seed}: AUC Validierung {tab['netz']:.4f} (Nachbarlisten-Reihenfolge {tab['hand']:.4f}), {cnt['sekunden']:.0f} s, Verlust zuletzt {verlauf[-1][1]:.4f}")


def machen_k(seeds) -> None:
    kontexte = Z.val_instanzen(50)
    refs = val_refs(50)
    D = np.stack([k.D for k in kontexte])
    from zwl_construct import nn_tour_np
    nn = np.array([float(k.D[nn_tour_np(k.D, 0), np.roll(nn_tour_np(k.D, 0), -1)].sum()) for k in kontexte])
    log(f"K Validierung: 50 Instanzen n = 20, Nächster Nachbar {100 * np.mean((nn - refs) / refs):.2f} % über Held-Karp")
    for seed in seeds:
        p = MLP.model_path("K", seed, ROOT)
        if p.exists():
            log(f"K Gewichts-Seed {seed}: schon vorhanden, übersprungen")
            continue
        t0 = time.time()
        params, verlauf, cnt = PO.train_featnet(C.K_SCHRITTE_FROZEN, seed=seed, progress=lambda s, g: print(f"  K seed {seed} Schritt {s}/{g}", flush=True) if s % 500 == 0 else None)
        ss = np.array([k.s for k in kontexte])
        _, L, _ = FN.greedy_tours(kontexte[0].D, params)  # Formtest
        T, L, _, _ = FN.rollout(D, ss, np.zeros(len(D), dtype=np.int64), params, mode="greedy")
        gap = float(np.mean((L - refs) / refs) * 100)
        meta = {"kind": "K", "seed": seed, "steps": cnt["schritte"], "lr": C.K_LERNRATE, "n_train_tours": cnt["touren"], "n_train_rows": cnt["zeilen"], "feature_version": C.FEATURE_VERSION,
                "val_gap_greedy": gap, "val_instances": 50, "n": C.K_N_TRAIN}
        MLP.save_model(p, params, meta)
        log(f"K Gewichts-Seed {seed}: greedy-Lücke Validierung {gap:.2f} % (Nächster Nachbar {100 * np.mean((nn - refs) / refs):.2f} %), {cnt['touren']} Touren, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    art = sys.argv[1]
    if art == "Z":
        machen_z()
    elif art == "K":
        machen_k([int(x) for x in sys.argv[2:]] or list(range(C.MODELL_SEEDS)))
    else:
        raise SystemExit("Aufruf: zwl_make_models.py Z | K [seeds]")
