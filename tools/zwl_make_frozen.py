"""Eingefrorene Messreihen (zwl_frozen.json, Format zwl-frozen-1): E3 (Hauptvergleich), E4a (Abbruch), E4b (Nachbarliste, erster Abstieg und Suche mit Kicks).

Aufruf (Ordner der Demo, ein Prozess, BLAS-Threads 1; wiederaufnehmbar, Zwischenstände in tools/_zwischen/):
  python tools/zwl_make_frozen.py e3 | e4a | e4b | alles      # Stufen; `alles` rechnet fehlende Stufen und schreibt zwl_frozen.json
  python tools/zwl_make_frozen.py --groesse                    # zeigt die Größe jeder Stufe
Alle Läufe starten beim Nächsten Nachbarn ab Ort 0, Suche mit erster Verbesserung und Double-Bridge-Kicks (zwl_search.ils), SplitMix64-Ströme
(zwl_experiments.suche_strom), Zugnetz Z aus zwl_modelle/zugnetz_s*.npz (gelernte Läufe: Gewichts-Zufalls-Seed = Zufalls-Seed der Suche modulo Zahl der Gewichts-Seeds).
Lage: Längen im 1000er-Quadrat, Referenzen stehen in zwl_refs.json."""

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import zwl_constants as C  # noqa: E402
import zwl_experiments as X  # noqa: E402
import zwl_moves as M  # noqa: E402
import zwl_search as S  # noqa: E402

ZW = ROOT / "tools" / "_zwischen"
ZW.mkdir(parents=True, exist_ok=True)
FROZEN = ROOT / "zwl_frozen.json"

ZELLEN = {
    "n20_gleich": dict(n=20, familie="gleich", inst=20, ks=(2, 8, 32), seeds=10, nw=5),
    "n50_gleich": dict(n=50, familie="gleich", inst=10, ks=(2, 8), seeds=10, nw=3),
    "n50_cluster": dict(n=50, familie="cluster", inst=10, ks=(2, 8), seeds=10, nw=3),
}
E4A_ZELLEN = {"n20_gleich": dict(ZELLEN["n20_gleich"], ks=(2, 8)), "n50_gleich": dict(ZELLEN["n50_gleich"], ks=(2,), seeds=5)}
E4A_G = C.E4A_G
E4A_REGELN = C.REGELN_E4A
E4B_ZELLEN = {"n20_gleich": dict(ZELLEN["n20_gleich"], ks=(2, 8)), "n50_gleich": dict(ZELLEN["n50_gleich"], ks=(2,), seeds=5)}
E4B_LISTE = (3, 5, 8)
E4B_REGELN = C.REGELN_E4B


def log(text: str) -> None:
    print(text, flush=True)


def scorer_fuer(regel: str, sd: int, nw: int):
    return X.lade_modell("Z", X.gewicht_seed(sd, nw))[0] if regel in C.REGELN_MIT_NETZ else None


def runde(x):
    return [runde(v) for v in x] if isinstance(x, (list, tuple)) else round(float(x), 3)


def zelle_laufen(cf: dict, regeln, abbruch: int = 0, liste: int = 0) -> dict:
    """Alle Läufe einer Zelle: best[inst][seed][k][sicht], enden, falsch, ev (je [inst][seed])."""
    out = {r: {"best": [], "enden": [], "falsch": [], "ev": []} for r in regeln}
    for inst in range(cf["inst"]):
        ctx = X.kontext(cf["familie"], cf["n"], inst)
        for regel in regeln:
            b, e, f, ev = [], [], [], []
            for sd in range(cf["seeds"]):
                r = X.lauf(ctx, regel, scorer_fuer(regel, sd, cf["nw"]), cf["ks"], X.suche_strom(cf["familie"], cf["n"], inst, sd), abbruch, liste)
                b.append(runde(r["best"]))
                e.append(r["enden"])
                f.append(r["falsch"])
                ev.append(r["ev"])
            out[regel]["best"].append(b)
            out[regel]["enden"].append(e)
            out[regel]["falsch"].append(f)
            out[regel]["ev"].append(ev)
    return out


def stufe_e3() -> dict:
    p = ZW / "e3.json"
    if p.exists():
        return json.loads(p.read_text())
    erg = {}
    for name, cf in ZELLEN.items():
        t0 = time.time()
        erg[name] = {"n": cf["n"], "familie": cf["familie"], "M": M.neighborhood_size(cf["n"]), "instanzen": cf["inst"], "ks": list(cf["ks"]), "zufalls_seeds": cf["seeds"],
                     "gewichts_seeds": cf["nw"], "regeln": zelle_laufen(cf, C.REGELN_E3)}
        log(f"E3 {name}: {cf['inst']} Instanzen, {len(C.REGELN_E3)} Regeln, {cf['seeds']} Seeds, {time.time() - t0:.0f} s")
    p.write_text(json.dumps(erg))
    return erg


def stufe_e4a() -> dict:
    p = ZW / "e4a.json"
    if p.exists():
        return json.loads(p.read_text())
    erg = {}
    for name, cf in E4A_ZELLEN.items():
        erg[name] = {"n": cf["n"], "familie": cf["familie"], "M": M.neighborhood_size(cf["n"]), "instanzen": cf["inst"], "ks": list(cf["ks"]), "zufalls_seeds": cf["seeds"],
                     "gewichts_seeds": cf["nw"], "abbruch": {}}
        for g in E4A_G:
            teil = ZW / f"e4a_{name}_{g}.json"
            if teil.exists():
                erg[name]["abbruch"][str(g)] = json.loads(teil.read_text())
                continue
            t0 = time.time()
            erg[name]["abbruch"][str(g)] = zelle_laufen(cf, E4A_REGELN, abbruch=g)
            teil.write_text(json.dumps(erg[name]["abbruch"][str(g)]))
            log(f"E4a {name} Abbruch {g}: {time.time() - t0:.0f} s")
    p.write_text(json.dumps(erg))
    return erg


def stufe_e4b() -> dict:
    p = ZW / "e4b.json"
    if p.exists():
        return json.loads(p.read_text())
    erg = {}
    for name, cf in E4B_ZELLEN.items():
        erg[name] = {"n": cf["n"], "familie": cf["familie"], "M": M.neighborhood_size(cf["n"]), "instanzen": cf["inst"], "ks": list(cf["ks"]), "zufalls_seeds": cf["seeds"],
                     "gewichts_seeds": cf["nw"], "liste": {}}
        for k in E4B_LISTE:
            teil = ZW / f"e4b_{name}_{k}.json"
            if teil.exists():
                erg[name]["liste"][str(k)] = json.loads(teil.read_text())
                continue
            t0 = time.time()
            ils = zelle_laufen(cf, E4B_REGELN, liste=k)
            erster = {r: {"voll_laenge": [], "laenge": [], "ev": [], "ev_voll": [], "n_pruef": [], "falsch": []} for r in E4B_REGELN}
            for inst in range(cf["inst"]):
                ctx = X.kontext(cf["familie"], cf["n"], inst)
                for regel in E4B_REGELN:
                    z = {key: [] for key in erster[regel]}
                    for sd in range(cf["seeds"]):
                        sc = scorer_fuer(regel, sd, cf["nw"])
                        st = X.start_tour(ctx)
                        voll = S.descend(ctx, st, regel, X.suche_strom(cf["familie"], cf["n"], inst, sd), sc)
                        li = S.descend(ctx, st, regel, X.suche_strom(cf["familie"], cf["n"], inst, sd), sc, liste=k)
                        z["voll_laenge"].append(round(M.tour_length(ctx.D, voll["tour"]), 3))
                        z["laenge"].append(round(M.tour_length(ctx.D, li["tour"]), 3))
                        z["ev"].append(li["ev_zuege"] + li["ev_pruef"])
                        z["ev_voll"].append(voll["ev_zuege"] + voll["ev_pruef"])
                        z["n_pruef"].append(li["n_pruef"])
                        z["falsch"].append(int(li["falsch"]))
                    for key in z:
                        erster[regel][key].append(z[key])
            erg[name]["liste"][str(k)] = {"abstieg": erster, "ils": ils}
            teil.write_text(json.dumps(erg[name]["liste"][str(k)]))
            log(f"E4b {name} Nachbarliste {k}: {time.time() - t0:.0f} s")
    p.write_text(json.dumps(erg))
    return erg


def zusammenbauen() -> None:
    daten = {"format": C.FROZEN_FORMAT, "sichten": list(C.SICHTEN), "e3": stufe_e3(), "e4a": stufe_e4a(), "e4b": stufe_e4b()}
    FROZEN.write_text(json.dumps(daten, separators=(",", ":")), encoding="utf-8")
    log(f"zwl_frozen.json geschrieben, {FROZEN.stat().st_size / 1e6:.2f} MB")


def groesse() -> None:
    for name, cf in ZELLEN.items():
        log(f"E3 {name}: {cf['inst'] * len(C.REGELN_E3) * cf['seeds']} Läufe bis {max(cf['ks'])} M")
    for name, cf in E4A_ZELLEN.items():
        log(f"E4a {name}: {cf['inst'] * len(E4A_REGELN) * cf['seeds'] * len(E4A_G)} Läufe bis {max(cf['ks'])} M")
    for name, cf in E4B_ZELLEN.items():
        log(f"E4b {name}: {cf['inst'] * len(E4B_REGELN) * cf['seeds'] * len(E4B_LISTE)} Läufe bis {max(cf['ks'])} M und ebenso viele erste Abstiege")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--groesse"
    {"e3": stufe_e3, "e4a": stufe_e4a, "e4b": stufe_e4b, "alles": zusammenbauen, "--groesse": groesse}[arg]()
