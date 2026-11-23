"""Referenzlängen der Instanzen der Seitenleiste: n = 10 und 20 Held-Karp (alle drei Familien, Zufalls-Seed 0 bis 19), n = 50 CP-SAT (drei Familien, Zufalls-Seed 0 bis 9).

Aufruf (Ordner der Demo, ein Prozess): python tools/zwl_make_refs.py [n50]   # ohne Angabe: n = 10 und 20, mit `n50`: nur die CP-SAT-Läufe (braucht ortools aus requirements-dev.txt)
Schreibt zwl_refs.json (Format zwl-refs-1): refs[familie][n] = [{"laenge", "bewiesen", "untergrenze"}], Längen im 1000er-Quadrat.
CP-SAT: AddCircuit, ganzzahlige Skalierung mit 1 000 (Rundungsfehler unter 0,05 je Tour), Zeitgrenze 30 s je Instanz, 4 Arbeiter; `bewiesen` heißt OPTIMAL, sonst ist die Lücke eine Untergrenze."""

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
import zwl_scenario as SC  # noqa: E402

PFAD = ROOT / "zwl_refs.json"


def cpsat_tour(D, zeit: float = 30.0):
    from ortools.sat.python import cp_model

    n = len(D)
    skala = 1000
    m = cp_model.CpModel()
    lits, kosten = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                b = m.NewBoolVar(f"x{i}_{j}")
                lits.append((i, j, b))
                kosten.append(int(round(D[i][j] * skala)) * b)
    m.AddCircuit(lits)
    m.Minimize(sum(kosten))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = zeit
    s.parameters.num_workers = min(4, os.cpu_count() or 1)
    status = s.Solve(m)
    nach = {i: j for i, j, b in lits if s.Value(b)}
    t, cur = [0], 0
    for _ in range(n - 1):
        cur = nach[cur]
        t.append(cur)
    laenge = sum(D[t[k]][t[(k + 1) % n]] for k in range(n))
    return laenge, status == cp_model.OPTIMAL, s.BestObjectiveBound() / skala


def main(nur_n50: bool) -> None:
    daten = json.loads(PFAD.read_text(encoding="utf-8")) if PFAD.exists() else {"format": C.REFS_FORMAT, "refs": {f: {} for f in C.FAMILIEN}}
    t0 = time.time()
    for fam in C.FAMILIEN:
        for n in (50,) if nur_n50 else (10, 20):
            liste = []
            for seed in range(C.N_REF_SEEDS[n]):
                _, D = SC.make_instance(fam, n, seed)
                if n == 50:
                    laenge, bewiesen, ug = cpsat_tour(D)
                else:
                    laenge, bewiesen, ug = EX.held_karp(np.array(D)), True, None
                liste.append({"laenge": round(float(laenge), 6), "bewiesen": bool(bewiesen), "untergrenze": None if ug is None else round(float(ug), 6)})
            daten["refs"][fam][str(n)] = liste
            print(f"{fam} n = {n}: {len(liste)} Instanzen, bewiesen {sum(r['bewiesen'] for r in liste)}, {time.time() - t0:.0f} s", flush=True)
    if not nur_n50:                                   # Validierungsinstanzen des Probetrainings (n = 20 gleich, Zufalls-Seed ab 1000)
        daten["val"] = {"gleich": {"20": [round(float(EX.held_karp(np.array(SC.make_instance("gleich", 20, C.VAL_OFFSET + i)[1]))), 6) for i in range(C.Z_VAL_INSTANZEN_FROZEN)]}}
        print(f"Validierung: {C.Z_VAL_INSTANZEN_FROZEN} Instanzen n = 20, {time.time() - t0:.0f} s", flush=True)
    PFAD.write_text(json.dumps(daten), encoding="utf-8")


if __name__ == "__main__":
    main("n50" in sys.argv[1:])
