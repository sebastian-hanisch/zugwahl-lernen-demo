"""Experimente E0 bis E5, die live in der App und in der Messung (tools/zwl_make_frozen.py) gleich laufen. Dünn: ruft nur die Einheiten auf (Suche, Netze, Zählung)."""

from functools import lru_cache

import numpy as np

import zwl_constants as C
import zwl_cost as K
import zwl_featnet as FN
import zwl_hand as H
import zwl_mlp as MLP
import zwl_scenario as SC
import zwl_search as S
import zwl_zugnetz as Z
from zwl_construct import farthest_insertion, nn_tour_np, zufallstour
from zwl_moves import Kontext, move_features, neighborhood, tour_length
from zwl_rngvec import UStream


# ---------------------------------------------------------------------------------------------------------------- Modelle und Instanzen
@lru_cache(maxsize=16)
def lade_modell(art: str, seed: int):
    """Vorgerechnetes Modell (Z oder K) mit Gewichts-Zufalls-Seed 0 bis 4: (Parameter, Angaben)."""
    return MLP.load_model(MLP.model_path(art, seed))


def kontext(familie: str, n: int, seed: int, doppelt: bool = False) -> Kontext:
    """Kontext der Instanz; das Handbeispiel (n = 6, Familie „hand“) ist die Tabelle."""
    if familie == C.FAMILIE_HAND or n == C.N_HAND:
        return Kontext(H.distanzen(), doppelt)
    _, D = SC.make_instance(familie, n, seed)
    return Kontext(np.array(D), doppelt)


def start_tour(ctx: Kontext) -> np.ndarray:
    """Startlösung der Suche: Nächster Nachbar ab Ort 0."""
    return nn_tour_np(ctx.D, 0)


def fam_index(familie: str) -> int:
    return 9 if familie == C.FAMILIE_HAND else C.FAMILIEN.index(familie)


def suche_strom(familie: str, n: int, inst: int, sd: int) -> UStream:
    """Strom der Suche je (Instanz, Zufalls-Seed der Suche): alle Regeln beginnen mit demselben Strom, damit der Vergleich gepaart ist."""
    return UStream(SC.derive(C.ZWECK_SUCHE, fam_index(familie), n, inst, sd))


def gewicht_seed(sd: int, anzahl: int) -> int:
    """Gewichts-Zufalls-Seed, der zum Zufalls-Seed der Suche gepaart ist (gelernte Regeln, reihum)."""
    return sd % anzahl


# ---------------------------------------------------------------------------------------------------------------- E0: ein Abstieg Schritt für Schritt
def kanten_des_zugs(t, k: int, nb) -> tuple:
    """(entfernte Kanten, neue Kanten) des Zugs k auf der Tour t als Paare von Orten (die Positionspaare des Zugs, übersetzt)."""
    t = np.asarray(t)
    n = nb.n
    entfernt = [(int(t[a]), int(t[b])) for a, b in nb.rem[k] if a != n]
    neu = [(int(t[a]), int(t[b])) for a, b in nb.add[k] if a != n]
    return entfernt, neu


def e0_abstieg(ctx: Kontext, regel: str, scorer, familie: str, seed: int, abbruch: int = 0, liste: int = 0) -> dict:
    """Ein Abstieg aus dem Nächsten Nachbarn mit Protokoll je Zug (Tour davor, entfernte und neue Kanten, Delta, Stelle in der Reihenfolge)."""
    start = start_tour(ctx)
    us = UStream(SC.derive(C.ZWECK_E0, fam_index(familie), ctx.n, seed))
    r = S.descend(ctx, start, regel, us, scorer, abbruch, liste, protokoll=True)
    t = start
    schritte = []
    for p in r["protokoll"]:
        entfernt, neu = kanten_des_zugs(t, p["k"], ctx.nb)
        schritte.append({**p, "tour_davor": t, "entfernt": entfernt, "neu": neu})
        t = p["tour"]
    r["schritte"] = schritte
    r["start"] = start
    r["start_laenge"] = tour_length(ctx.D, start)
    r["end_laenge"] = tour_length(ctx.D, r["tour"])
    return r


# ---------------------------------------------------------------------------------------------------------------- E1: ordnet das Netz besser?
def e1_guete(scorer, count: int = C.Z_VAL_INSTANZEN) -> dict:
    """AUC des Netzes, der Nachbarlisten-Reihenfolge und der Einzelmerkmale auf Validierungszuständen (n = 20 gleich, Zufalls-Seeds ab 1000)."""
    F, y, zustaende = Z.validierungszeilen(count)
    tab = Z.auc_tabelle(F, y, scorer)
    return {"auc": tab, "zeilen": int(len(y)), "zustaende": int(zustaende), "instanzen": count, "anteil_verbessernd": float(y.mean())}


E2_STROEME = 5                                  # Zufalls-Seeds der Suche je Instanz in Experiment 2 (Zufallsreihenfolge)
E1_STROEME = 5                                 # Zufalls-Seeds der Suche je Instanz in Experiment 1 (Mittel über Instanzen und Seeds)


def e1_je_zug(scorer, familie: str, n: int, count: int, regeln=C.REGELN_E3, stroeme: int = E1_STROEME) -> dict:
    """Erster Abstieg aus dem Nächsten Nachbarn je Regel über `count` Instanzen mit je `stroeme` Zufalls-Seeds der Suche: Zugauswertungen je angenommenem Zug
    (Summe der Auswertungen bis zu den Zügen / Summe der Züge), Merkmalszeilen (Mittel je Abstieg)."""
    out = {}
    for regel in regeln:
        ev = zuege = zeilen = pruef = 0
        for seed in range(count):
            ctx = kontext(familie, n, seed)
            for sd in range(stroeme):
                r = S.descend(ctx, start_tour(ctx), regel, suche_strom(familie, n, seed, sd), scorer)
                ev += r["ev_zuege"]
                zuege += r["zuege"]
                zeilen += r["rows"]
                pruef += r["ev_pruef"]
        m = count * stroeme
        out[regel] = {"je_zug": ev / max(zuege, 1), "zuege": zuege / m, "ev_zuege": ev / m, "ev_pruef": pruef / m, "zeilen": zeilen / m, "abstiege": m}
    return out


def e1_zugarten(scorer, familie: str, n: int, count: int, top: int = 20) -> dict:
    """Welche Zugart bevorzugt das Netz (Reserve R, teilweise gefüllt)? Anteil der Zugarten unter den `top` bestbewerteten Kandidaten je Zustand, gemittelt über alle Zustände
    der Abstiege (Zufallsregel aus dem Nächsten Nachbarn), gegen den Anteil der Art an der Zugmenge M. Zustand = Tour vor jedem Zug und vor der Prüfrunde."""
    nb = neighborhood(n)
    anteil_top = np.zeros(4)
    zustaende = 0
    for seed in range(count):
        ctx = kontext(familie, n, seed)
        r = S.descend(ctx, start_tour(ctx), "zufall", suche_strom(familie, n, seed, 0), protokoll=True)
        touren = [start_tour(ctx)] + [p["tour"] for p in r["protokoll"]]
        for t in touren:
            _, F, _ = move_features(ctx.D, t, nb, ctx.s, ctx.Rs)
            z = MLP.forward(scorer, F)
            best = np.argsort(-z, kind="stable")[:top]
            anteil_top += np.bincount(nb.art[best], minlength=4) / top
            zustaende += 1
    anteil_m = np.bincount(nb.art, minlength=4) / nb.M
    return {"zugarten": list(C.ZUGARTEN), "anteil_top": (anteil_top / zustaende).tolist(), "anteil_m": anteil_m.tolist(), "zustaende": zustaende, "top": top}


# ---------------------------------------------------------------------------------------------------------------- E2: wie viel vom Aufwand ist der Beweis?
def e2_beweis(familie: str, n: int, count: int, doppelt: bool = False) -> dict:
    """Erster Abstieg aus dem Nächsten Nachbarn über `count` Instanzen: Zugauswertungen bis zum Lokalminimum (mit Prüfrunde), Anteil der Prüfrunde.
    Standard: Zufallsreihenfolge (je Instanz `E2_STROEME` Zufalls-Seeds der Suche). `doppelt`: Zählweise der Vormessung, also Einzelortverschiebung in beiden Richtungen
    **und** Aufzählungsreihenfolge (2-opt zuerst), wie die frühere Vormessung sie zählte."""
    gesamt, pruef = [], []
    for seed in range(count):
        ctx = kontext(familie, n, seed, doppelt)
        for sd in range(1 if doppelt else E2_STROEME):
            r = S.descend(ctx, start_tour(ctx), "aufzaehlung" if doppelt else "zufall", suche_strom(familie, n, seed, sd))
            gesamt.append(r["ev_zuege"] + r["ev_pruef"])
            pruef.append(r["ev_pruef"])
    m = neighborhood(n, doppelt).M
    mittel = float(np.mean(gesamt))
    return {"n": n, "M": m, "mittel": mittel, "sd": float(np.std(gesamt)), "anteil": m / mittel, "instanzen": count, "pruef": int(pruef[0]), "gespart_max": 1 - m / mittel}


# ---------------------------------------------------------------------------------------------------------------- E3 und E4: Suche mit Kicks bis k M
def lauf(ctx: Kontext, regel: str, scorer, ks, strom: UStream, abbruch: int = 0, liste: int = 0) -> dict:
    """Eine Suche (Abstieg, Kicks) aus dem Nächsten Nachbarn bis max(ks) M Zugauswertungen. best[k][sicht]: beste Länge bei Budget k M je Sicht
    (frei, K, O; Zeilen kosten 0, 1 bzw. 22 Zugauswertungen); dazu Abstiegsenden, davon falsche Lokalminima, Zugauswertungen am Ende, Merkmalszeilen, erster Abstieg."""
    m = ctx.nb.M
    res = S.ils(ctx, start_tour(ctx), regel, max(ks) * m, strom, scorer, abbruch, liste)
    best = [[S.bester_wert(res, k * m, K.zeilen_gewicht(regel, s)) for s in C.SICHTEN] for k in ks]
    return {"best": best, "enden": res["enden"], "falsch": res["falsch"], "ev": int(res["ev"][-1]), "rows": int(res["rows_total"]), "kicks": res["kicks"],
            "erster": res["erster_abstieg"], "res": res}


def kurve(res_liste, regel: str, sicht: str, raster) -> np.ndarray:
    """Mittel der besten Länge über die Läufe bei den Kosten des Rasters (NaN, wo die Startlösung schon teurer ist)."""
    w = K.zeilen_gewicht(regel, sicht)
    return np.array([np.nanmean([S.bester_wert(r, b, w) for r in res_liste]) for b in raster])


def e3_probe(ctx: Kontext, familie: str, inst: int, k: int, scorer, seeds: int = C.E3_LIVE_SEEDS, regeln=("zufall", "nachbarliste", "gelernt", "gelernt_det")) -> dict:
    """Live-Probe des Hauptvergleichs: eine Instanz, `seeds` Zufalls-Seeds, die Regeln bei Budget k M. Ergebnis je Regel: Läufe, beste Länge je Sicht je Seed."""
    out = {}
    for regel in regeln:
        runs = [lauf(ctx, regel, scorer, (k,), suche_strom(familie, ctx.n, inst, sd), 0, 0) for sd in range(seeds)]
        out[regel] = {"runs": [r["res"] for r in runs], "best": np.array([r["best"][0] for r in runs])}      # (seeds, 3)
    return out


def e4_probe(ctx: Kontext, familie: str, inst: int, regel: str, scorer, abbruch: int, liste: int) -> dict:
    """Live-Probe von E4: ein Abstieg aus dem Nächsten Nachbarn mit Abbruch bzw. Nachbarliste und derselbe Abstieg voll (gleicher Strom), mit unabhängiger Gegenprobe."""
    start = start_tour(ctx)
    voll = S.descend(ctx, start, regel, suche_strom(familie, ctx.n, inst, 0), scorer)
    r = S.descend(ctx, start, regel, suche_strom(familie, ctx.n, inst, 0), scorer, abbruch, liste)
    return {"voll": voll, "probe": r, "laenge_voll": tour_length(ctx.D, voll["tour"]), "laenge": tour_length(ctx.D, r["tour"]),
            "kein_lokalminimum": not S.is_local_min_independent(ctx.D, r["tour"]), "m": ctx.nb.M}


# ---------------------------------------------------------------------------------------------------------------- E5: Startlösungen
def netz_k_start(ctx: Kontext, params_k) -> tuple:
    """Startlösung des Merkmalsnetzes K (greedy ab Ort 0): (Tour, Zahl der Merkmalszeilen)."""
    T, _, zeilen = FN.greedy_tours(ctx.D, params_k, starts=(0,))
    return T[0], zeilen


def e5_starts(familie: str, n: int, count: int, params_k, refs=None) -> list:
    """Je Startart ein Abstieg (Zufallsreihenfolge) über `count` Instanzen. Zeilen: Startart, mittlere Lücke der Startlösung (`refs`: Referenzlängen je Zufalls-Seed oder None),
    Zugauswertungen bis zum Lokalminimum (ohne Prüfrunde, mit Prüfrunde), Ersparnis gegen Nächster Nachbar in Prozent, K und O gesamt (Konstruktion + Abstieg)."""
    daten = {a: {"gap": [], "ev": [], "pruef": [], "zuege": []} for a in C.START_ARTEN}
    for seed in range(count):
        ctx = kontext(familie, n, seed)
        us0 = UStream(SC.derive(C.ZWECK_E5, fam_index(familie), n, seed))
        nn = nn_tour_np(ctx.D, 0)
        nn_end = S.descend(ctx, nn, "zufall", UStream(SC.derive(C.ZWECK_E5, fam_index(familie), n, seed, 1)))["tour"]
        starts = {"zufall": zufallstour(us0, n), "nn": nn, "farthest": farthest_insertion(ctx.D, 0), "optimum": nn_end}
        if params_k is not None:
            starts["netz_k"] = netz_k_start(ctx, params_k)[0]
        for art, t in starts.items():
            r = S.descend(ctx, t, "zufall", UStream(SC.derive(C.ZWECK_E5, fam_index(familie), n, seed, 2)))
            daten[art]["ev"].append(r["ev_zuege"])
            daten[art]["pruef"].append(r["ev_pruef"])
            daten[art]["zuege"].append(r["zuege"])
            if refs is not None:
                daten[art]["gap"].append(100.0 * (tour_length(ctx.D, t) - refs[seed]) / refs[seed])
    nn_ev = float(np.mean(np.array(daten["nn"]["ev"]) + np.array(daten["nn"]["pruef"])))
    zeilen = []
    for art in C.START_ARTEN:
        d = daten[art]
        if not d["ev"]:
            continue
        ev, pr = float(np.mean(d["ev"])), float(np.mean(d["pruef"]))
        k_s, o_s = K.start_kosten(art, n)
        zeilen.append({"art": art, "gap_start": float(np.mean(d["gap"])) if d["gap"] else None, "ev_zuege": ev, "ev_pruef": pr, "ev_gesamt": ev + pr,
                       "ersparnis": 100.0 * (nn_ev - (ev + pr)) / nn_ev, "k_start": k_s, "o_start": o_s, "k_gesamt": k_s + ev + pr, "o_gesamt": o_s + K.ops_kandidaten(ev + pr), "instanzen": count})
    return zeilen


def k_greedy_gap(params_k, refs: dict, count: int = C.Z_VAL_INSTANZEN) -> tuple:
    """(Greedy-Lücke des Merkmalsnetzes K, Lücke des Nächsten Nachbarn) in Prozent über Held-Karp, Mittel über `count` Validierungsinstanzen (n = 20 gleich, ab Zufalls-Seed 1000), Start Ort 0."""
    ref = np.array(refs["val"]["gleich"]["20"][:count])
    netz, nn = [], []
    for i, ctx in enumerate(Z.val_instanzen(count)):
        netz.append(float(FN.greedy_tours(ctx.D, params_k, starts=(0,))[1][0]))
        nn.append(tour_length(ctx.D, nn_tour_np(ctx.D, 0)))
    return float(np.mean(100 * (np.array(netz) - ref) / ref)), float(np.mean(100 * (np.array(nn) - ref) / ref))


# ---------------------------------------------------------------------------------------------------------------- Training in der App
def trainiere_zugnetz(schritte: int, instanzen: int, seed: int, familie: str = "gleich", progress=None, max_seconds: float | None = None) -> tuple:
    """Training des Zugnetzes in der App: Daten sammeln (Instanzen ab Zufalls-Seed 2000), dann `schritte` Schritte zu 4 096 Zeilen. Gibt (Parameter, Verlauf, Zähler)."""
    F, y, zustaende = Z.sammle_zustaende(Z.train_instanzen(instanzen, C.Z_N_TRAIN, familie))
    params, verlauf, cnt = Z.train_zugnetz(F, y, schritte, seed=seed, progress=progress, max_seconds=max_seconds)
    cnt.update({"daten_zeilen": int(len(y)), "daten_zustaende": int(zustaende), "instanzen": instanzen})
    return params, verlauf, cnt
