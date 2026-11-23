"""Lokale Suche mit erster Verbesserung: Reihenfolge der Kandidaten (Regeln), Abstieg bis zum Lokalminimum, iterierte lokale Suche mit Kicks (ILS),
Abbruch der Prüfung und Nachbarliste als Zugmenge. Quelle für die Folge-Stücke (Kopie heißt `<präfix>_search.py`).

Zählung (Q5): **Zugauswertung** = ein bewerteter Kandidat (Delta aus Kanten; ein Kick zählt als eine), **Merkmalszeile** = ein vom Netz oder von der Nachbarlisten-Reihenfolge
bewerteter Kandidat. Die Reihenfolge gilt für die Kandidaten der Zugmenge (alle M oder die der Nachbarliste); die Suche nimmt den ersten verbessernden Zug. Findet sie keinen,
hat sie die ganze Zugmenge bewertet (die **Prüfrunde**: M Zugauswertungen, Beweis des Lokalminimums) oder, mit Abbruch nach g Kandidaten, nur g (dann ist es kein Beweis).

Regeln: `aufzaehlung` (Reihenfolge der Zugliste), `zufall`, `nachbarliste` (Nachbarlisten-Reihenfolge: kleinster größter Rang der neuen Kanten zuerst, dann längere entfernte
Kanten zuerst; sie ordnet nur und bezahlt dafür M Merkmalszeilen), `gelernt` (Zugnetz Z, Reihenfolge gezogen nach Gumbel), `gelernt_det` (absteigende Logits), `steilster` (bester Zug, M Auswertungen).
Zufall nur über den übergebenen Strom `us` (UStream, SplitMix64); jeder Schritt der Suche liest die Zahl der Kandidaten an Zufallszahlen (außer `aufzaehlung`, `steilster`)."""

import numpy as np

import zwl_constants as C
import zwl_cost as K
import zwl_mlp as MLP
from zwl_moves import (
    Kontext,
    apply_move,
    deltas,
    kick_double_bridge,
    move_features,
    tour_length,
)

EPS = C.EPS


# ---------------------------------------------------------------------------------------------------------------- Reihenfolge und ein Schritt
def ordne(regel: str, cand: np.ndarray, delta, F, us, scorer):
    """(Reihenfolge der Kandidaten, Zahl der Merkmalszeilen). `cand`: Indizes der Zugmenge, die in Frage kommen."""
    m = len(cand)
    if regel == "aufzaehlung":
        return cand, 0
    if regel == "zufall":
        return cand[np.argsort(us.uniform(m), kind="stable")], 0
    if regel == "nachbarliste":
        schluessel = np.rint(F[cand, 3] * C.RANG_KAPPE) * 100.0 - F[cand, 0] + 1e-6 * us.uniform(m)
        return cand[np.argsort(schluessel, kind="stable")], m
    if regel == "gelernt":
        z = MLP.forward(scorer, F[cand])
        return cand[np.argsort(-(z + us.gumbel(m)), kind="stable")], m
    if regel == "gelernt_det":
        z = MLP.forward(scorer, F[cand])
        return cand[np.argsort(-z + 1e-9 * us.uniform(m), kind="stable")], m
    raise ValueError(f"unbekannte Regel {regel!r}")


def waehle_zug(ctx: Kontext, t, regel: str, us=None, scorer=None, abbruch: int = 0, liste: int = 0, brauche_f: bool = False) -> dict:
    """Ein Schritt der Suche auf der Tour t: der erste verbessernde Kandidat in der Reihenfolge der Regel.

    `abbruch` > 0: nach so vielen erfolglosen Kandidaten aufgeben; `liste` > 0: nur Züge, bei denen mindestens eine neue Kante einen Rang unter `liste` hat.
    Ergebnis: idx (Zug oder -1), improving, kosten (Zugauswertungen dieses Schritts), rows (Merkmalszeilen), n_cand (Größe der Zugmenge dieses Schritts),
    full_improvable (gäbe es in der ganzen Zugmenge einen verbessernden Zug? Gegenprobe für Abbruch und Nachbarliste), delta, F."""
    nb = ctx.nb
    braucht = regel in C.REGELN_MIT_ZEILEN or liste > 0 or brauche_f
    if braucht:
        delta, F, min_rank = move_features(ctx.D, t, nb, ctx.s, ctx.Rs)
    else:
        delta, F, min_rank = deltas(ctx.D, t, nb), None, None
    cand = np.arange(nb.M) if liste <= 0 else np.nonzero(min_rank < liste)[0]
    m = len(cand)
    voll = bool((delta < -EPS).any())
    if m == 0:
        return dict(idx=-1, improving=False, kosten=0, rows=0, n_cand=0, full_improvable=voll, delta=delta, F=F)
    if regel == "steilster":
        idx = int(cand[int(np.argmin(delta[cand]))])
        return dict(idx=idx if delta[idx] < -EPS else -1, improving=bool(delta[idx] < -EPS), kosten=m, rows=0, n_cand=m, full_improvable=voll, delta=delta, F=F)
    order, rows = ordne(regel, cand, delta, F, us, scorer)
    imp = delta[order] < -EPS
    if abbruch > 0:
        imp = imp[:abbruch]
    if imp.any():
        pos = int(imp.argmax())
        return dict(idx=int(order[pos]), improving=True, kosten=pos + 1, rows=rows, n_cand=m, full_improvable=voll, delta=delta, F=F)
    return dict(idx=-1, improving=False, kosten=m if abbruch <= 0 else min(abbruch, m), rows=rows, n_cand=m, full_improvable=voll, delta=delta, F=F)


# ---------------------------------------------------------------------------------------------------------------- Abstieg bis zum Lokalminimum
def descend(ctx: Kontext, t0, regel: str, us=None, scorer=None, abbruch: int = 0, liste: int = 0, protokoll: bool = False) -> dict:
    """Abstieg mit erster Verbesserung bis keiner der Kandidaten mehr verbessert (mit `abbruch` oder `liste`: bis zum Abbruch bzw. zum Lokalminimum der Nachbarliste).

    Ergebnis: tour, zuege (angenommene Züge), ev_zuege (Zugauswertungen bis zu den angenommenen Zügen, ohne die letzte Runde), ev_pruef (Zugauswertungen der letzten Runde:
    bei voller Zugmenge M, der Beweis), rows (Merkmalszeilen), falsch (die Endtour ist kein Lokalminimum der vollen Zugmenge), n_pruef (Größe der Zugmenge der letzten Runde)
    und mit `protokoll` je Zug: Zug, Art, Delta, Stelle in der Reihenfolge, Tour danach."""
    t = np.asarray(t0).copy()
    nb = ctx.nb
    zuege, ev_zuege, rows, pro = 0, 0, 0, []
    while True:
        z = waehle_zug(ctx, t, regel, us, scorer, abbruch, liste)
        rows += z["rows"]
        if not z["improving"]:
            return dict(tour=t, zuege=zuege, ev_zuege=ev_zuege, ev_pruef=z["kosten"], rows=rows, falsch=z["full_improvable"], n_pruef=z["n_cand"], protokoll=pro)
        k = z["idx"]
        t = apply_move(t, k, nb)
        zuege += 1
        ev_zuege += z["kosten"]
        if protokoll:
            pro.append(dict(k=k, art=int(nb.art[k]), delta=float(z["delta"][k]), stelle=z["kosten"], tour=t.copy(), rows=z["rows"]))


# ---------------------------------------------------------------------------------------------------------------- Iterierte lokale Suche
def ils(ctx: Kontext, t0, regel: str, budget: int, us, scorer=None, abbruch: int = 0, liste: int = 0, start_kosten: int | None = None,
        snap=None, brauche_f: bool = False) -> dict:
    """Abstieg, Kick (Double-Bridge von der besten bisherigen Tour, angenommen wird bei Länge <= bisher beste), wieder Abstieg, bis `budget` Zugauswertungen verbraucht sind.

    Das Budget ist in Zugauswertungen (Zeilen gratis); die Merkmalszeilen stehen daneben (`rows`), die Sichten K und O rechnen sie später dazu (zwl_evaluation).
    Die Konstruktion der Startlösung (Standard n(n-1)/2, Nächster Nachbar) ist in `ev` schon enthalten; ein Kick zählt eine Zugauswertung.
    Ergebnis: Ereignisse als Felder ev, rows, ln (kumulierte Zugauswertungen, Merkmalszeilen, Länge der Tour nach jedem Ereignis), erster_abstieg
    (angenommene Züge, Zugauswertungen bis dahin, ev, rows, Länge beim ersten Ende), enden (Zahl der Abstiegsenden), falsch (davon keine Lokalminima der vollen Zugmenge), kicks, schritte, best_tour."""
    nb, D = ctx.nb, ctx.D
    t = np.asarray(t0).copy()
    cur = tour_length(D, t)
    ev = K.k_konstruktion(ctx.n) if start_kosten is None else int(start_kosten)
    rows = 0
    evs, rws, lns = [ev], [0], [cur]
    inc_t, inc_len = t.copy(), cur
    best_len, best_tour = cur, t.copy()
    acc_steps = acc_evals = 0
    erster = None
    kicks = schritte = enden = falsch = 0
    while ev < budget:
        z = waehle_zug(ctx, t, regel, us, scorer, abbruch, liste, brauche_f=brauche_f or snap is not None)
        schritte += 1
        rows += z["rows"]
        if snap is not None:
            snap(t, z)
        ev += z["kosten"]
        if z["improving"]:
            acc_steps += 1
            acc_evals += z["kosten"]
            t = apply_move(t, z["idx"], nb)
            cur += float(z["delta"][z["idx"]])
            evs.append(ev), rws.append(rows), lns.append(cur)
            continue
        enden += 1
        falsch += int(z["full_improvable"])
        if erster is None:
            erster = (acc_steps, acc_evals, ev, rows, tour_length(D, t))
        cur = tour_length(D, t)
        evs.append(ev), rws.append(rows), lns.append(cur)
        if cur <= best_len + EPS:
            inc_t, inc_len = t.copy(), cur
        if cur < best_len - EPS:
            best_tour = t.copy()
        best_len = min(best_len, cur)
        t = kick_double_bridge(inc_t, us)
        ev += 1
        kicks += 1
        cur = tour_length(D, t)
        evs.append(ev), rws.append(rows), lns.append(cur)
    return dict(ev=np.array(evs), rows=np.array(rws), ln=np.array(lns), erster_abstieg=erster, enden=enden, falsch=falsch, kicks=kicks, schritte=schritte,
                acc_steps=acc_steps, acc_evals=acc_evals, rows_total=rows, best_tour=best_tour, best_len=float(min(lns)), inc_len=inc_len)


def bester_wert(res: dict, budget: float, gewicht: float = 0.0) -> float:
    """Beste Tourlänge, die bis zum Budget erreicht war. Kosten eines Ereignisses = Zugauswertungen + gewicht * Merkmalszeilen
    (gewicht 0: Zeilen gratis, 1: K, 22: O bei einer Zeile des Zugnetzes mit 176 Operationen gegen 8 je Zugauswertung). NaN, wenn schon die Startlösung das Budget sprengt."""
    kosten = res["ev"] + gewicht * res["rows"]
    ok = kosten <= budget
    if not ok.any():
        return float("nan")
    return float(res["ln"][ok].min())


# ---------------------------------------------------------------------------------------------------------------- Gegenprobe des Lokalminimums (zweiter Rechenweg)
def is_local_min_independent(D, t) -> bool:
    """Ist t ein Lokalminimum von 2-opt und Or-opt (Segment 1 bis 3, beide Richtungen)? Anderer Rechenweg als `deltas`: jede Nachbartour wird ausdrücklich gebaut
    (2-opt: Teilstück umkehren; Or-opt: Segment herausnehmen und in jede Lücke der übrigen Tour einsetzen) und ihre Länge voll neu berechnet."""
    D = np.asarray(D, dtype=np.float64)
    t = [int(x) for x in t]
    n = len(t)
    L0 = float(D[np.asarray(t), np.roll(np.asarray(t), -1)].sum())

    def laenge(u):
        u = np.asarray(u)
        return float(D[u, np.roll(u, -1)].sum())

    for i in range(n - 1):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if laenge(t[: i + 1] + t[i + 1: j + 1][::-1] + t[j + 1:]) < L0 - EPS:
                return False
    for L in C.OR_LAENGEN:
        for i in range(n):
            seg = [t[(i + k) % n] for k in range(L)]
            rest = [t[(i + L + k) % n] for k in range(n - L)]
            for g in range(len(rest) - 1):                    # Lücke zwischen rest[g] und rest[g+1]; die Lücke rest[-1], rest[0] ist die ursprüngliche
                for s in (seg, seg[::-1]) if L > 1 else (seg,):
                    if laenge(rest[: g + 1] + s + rest[g + 1:]) < L0 - EPS:
                        return False
    return True
