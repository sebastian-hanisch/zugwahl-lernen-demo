"""Auswertung: Zahlenformate, eingefrorene Messreihen lesen (E3, E4), Referenzlängen, Marken und alle Verdikte als reine Funktionen.

Jedes Verdikt gibt `(art, text)` zurück (art: "success", "info", "warning"; die App ruft `getattr(st, art)(text)`). Schwellen sind gesetzt, nicht abgeleitet (zwl_constants):
Marken des Vergleichs nach DEMO-PLAYBOOK §4 (schlägt: mindestens 0,25 Prozentpunkte und Intervall ohne 0; gleich: unter 0,10 oder Intervall mit 0; sonst dazwischen)."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import zwl_constants as C
import zwl_cost as K
import zwl_stats as S

FROZEN_PFAD = Path(__file__).resolve().parent / "zwl_frozen.json"
REFS_PFAD = Path(__file__).resolve().parent / "zwl_refs.json"


# ---------------------------------------------------------------------------------------------------------------- Zahlenformate (deutsch)
def de(x: float, stellen: int = 2) -> str:
    """Zahl mit Dezimalkomma (eine Zahl, die auf null rundet, ohne Minuszeichen)."""
    s = f"{x:.{stellen}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace(".", ",")


def de_vz(x: float, stellen: int = 2) -> str:
    """Zahl mit Dezimalkomma und Vorzeichen (eine Zahl, die auf null rundet, als „0,00“ ohne Vorzeichen)."""
    s = f"{x:+.{stellen}f}"
    if float(s) == 0:
        s = s[1:]
    return s.replace(".", ",")


def zahl(n) -> str:
    """Ganzzahl mit Leerzeichen als Tausendertrenner (1 850)."""
    return f"{int(round(n)):,}".replace(",", " ")


def prozent(x: float, stellen: int = 0) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Dezimalkomma, ohne Zeichen."""
    return de(100.0 * x, stellen)


def zahl_k(x: float) -> str:
    """Kandidatenbewertungen und Operationen: ganze Zahl mit Tausendertrenner, ab einer Million als Zahl mit „Mio.“."""
    if x >= 1e6:
        return f"{de(x / 1e6, 2)} Mio."
    return zahl(x)


# ---------------------------------------------------------------------------------------------------------------- Eingefrorene Messung und Referenzen
@lru_cache(maxsize=2)
def load_frozen(pfad: str = str(FROZEN_PFAD)) -> dict:
    """Die eingefrorene Messung (zwl_frozen.json); FileNotFoundError, wenn sie fehlt."""
    daten = json.loads(Path(pfad).read_text(encoding="utf-8"))
    if daten.get("format") != C.FROZEN_FORMAT:
        raise ValueError("unbekanntes Format der eingefrorenen Messung")
    return daten


@lru_cache(maxsize=2)
def load_refs(pfad: str = str(REFS_PFAD)) -> dict:
    """Referenzlängen (zwl_refs.json)."""
    daten = json.loads(Path(pfad).read_text(encoding="utf-8"))
    if daten.get("format") != C.REFS_FORMAT:
        raise ValueError("unbekanntes Format der Referenzen")
    return daten


def ref_laenge(familie: str, n: int, seed: int, refs: dict | None = None):
    """(Referenzlänge, bewiesen) der Instanz oder None, wenn für sie keine vorgerechnet ist (Zufalls-Seed 0 bis 19 bei n = 10, 20; 0 bis 9 bei n = 50)."""
    refs = refs or load_refs()
    liste = refs["refs"].get(familie, {}).get(str(n))
    if liste is None or seed >= len(liste):
        return None
    return liste[seed]["laenge"], liste[seed]["bewiesen"]


def refs_vektor(familie: str, n: int, anzahl: int, refs: dict | None = None) -> np.ndarray:
    """Referenzlängen der ersten `anzahl` Instanzen."""
    refs = refs or load_refs()
    return np.array([r["laenge"] for r in refs["refs"][familie][str(n)][:anzahl]])


def luecke(laenge, ref) -> np.ndarray:
    """Lücke in Prozent über der Referenz."""
    return S.gap(laenge, ref)


# ---------------------------------------------------------------------------------------------------------------- Vergleich und Marken
def marke(vergleich: dict) -> str:
    """Marke eines gepaarten Vergleichs (v = Regel minus Gegner in Prozentpunkten, negativ = die Regel ist besser): schlägt, gleich, dazwischen oder verliert."""
    k = vergleich["klasse"]
    if k == "klar":
        return "schlägt" if vergleich["v"] < 0 else "verliert"
    if k == "gleich":
        return "gleich"
    return "dazwischen"


def vergleiche_bloecke(best_regel: np.ndarray, best_gegner: np.ndarray, ref: np.ndarray, ki: int, si: int) -> dict:
    """Gepaarter Vergleich zweier Blöcke best[inst][seed][k][sicht]: Lücke je Instanz (Mittel über die Seeds), Differenz Regel minus Gegner, Bootstrap-Intervall über die Instanzen."""
    return vergleiche_luecken(luecke(best_regel[:, :, ki, si], ref[:, None]).mean(1), luecke(best_gegner[:, :, ki, si], ref[:, None]).mean(1))


def vergleiche_luecken(g_r: np.ndarray, g_g: np.ndarray) -> dict:
    """Gepaarter Vergleich zweier Lücken je Instanz: Differenz, Intervall, Klasse und Marke."""
    r = S.compare(g_r, g_g)
    r.update({"luecke_regel": float(g_r.mean()), "luecke_gegner": float(g_g.mean()), "marke": None})
    r["marke"] = marke(r)
    return r


# ---------------------------------------------------------------------------------------------------------------- E3: Hauptvergleich
def e3_zellen(frozen: dict) -> list:
    return list(frozen["e3"])


def e3_block(frozen: dict, zelle: str, regel: str) -> np.ndarray:
    """best[inst][seed][k][sicht] als Feld."""
    return np.array(frozen["e3"][zelle]["regeln"][regel]["best"])


def e3_ref(frozen: dict, zelle: str, refs: dict | None = None) -> np.ndarray:
    z = frozen["e3"][zelle]
    return refs_vektor(z["familie"], z["n"], z["instanzen"], refs)


def e3_luecken(frozen: dict, zelle: str, regel: str, k: int, sicht: str, refs: dict | None = None) -> float:
    """Mittlere Lücke (Prozent) einer Regel bei Budget k M in einer Sicht, Mittel über Instanzen und Seeds."""
    z = frozen["e3"][zelle]
    return float(luecke(e3_block(frozen, zelle, regel)[:, :, z["ks"].index(k), C.SICHTEN.index(sicht)], e3_ref(frozen, zelle, refs)[:, None]).mean())


def e3_vergleich(frozen: dict, zelle: str, regel: str, k: int, sicht: str, gegner: str = "zufall", refs: dict | None = None) -> dict:
    """Regel gegen Gegner (Standard Zufall) bei Budget k M in einer Sicht."""
    z = frozen["e3"][zelle]
    return vergleiche_bloecke(e3_block(frozen, zelle, regel), e3_block(frozen, zelle, gegner), e3_ref(frozen, zelle, refs), z["ks"].index(k), C.SICHTEN.index(sicht))


def e3_marken(frozen: dict, zelle: str, regel: str, k: int, gegner: str = "zufall", refs: dict | None = None) -> dict:
    """Marken in den drei Sichten (frei, K, O) für `regel` gegen `gegner`."""
    return {s: e3_vergleich(frozen, zelle, regel, k, s, gegner, refs)["marke"] for s in C.SICHTEN}


def e3_gewichts_streuung(frozen: dict, zelle: str, regel: str, k: int, sicht: str, refs: dict | None = None) -> list:
    """Mittlere Lücke je Gewichts-Seed (gelernte Regeln: Läufe mit Zufalls-Seed sd benutzen Gewichts-Seed sd modulo Zahl der Gewichts-Seeds)."""
    z = frozen["e3"][zelle]
    nw = z["gewichts_seeds"]
    g = luecke(e3_block(frozen, zelle, regel)[:, :, z["ks"].index(k), C.SICHTEN.index(sicht)], e3_ref(frozen, zelle, refs)[:, None])
    return [float(g[:, [s for s in range(g.shape[1]) if s % nw == w]].mean()) for w in range(nw)]


def e3_pruef_anteil(frozen: dict, zelle: str, regel: str) -> float:
    """Anteil der Prüfrunden an den Zugauswertungen einer Suche mit Kicks: (Abstiegsenden x M) / Zugauswertungen am Ende, über alle Läufe der Regel."""
    z = frozen["e3"][zelle]
    r = z["regeln"][regel]
    return float(np.sum(r["enden"]) * z["M"] / np.sum(r["ev"]))


# ---------------------------------------------------------------------------------------------------------------- E4a: Abbruch
def e4a_block(frozen: dict, zelle: str, g: int, regel: str) -> np.ndarray:
    return np.array(frozen["e4a"][zelle]["abbruch"][str(g)][regel]["best"])


def e4a_falsch_anteil(frozen: dict, zelle: str, g: int, regel: str) -> float:
    """Anteil der Abstiegsenden, die kein Lokalminimum der vollen Zugmenge sind (Gegenprobe mit voller Prüfung)."""
    r = frozen["e4a"][zelle]["abbruch"][str(g)][regel]
    return float(np.sum(r["falsch"]) / np.sum(r["enden"]))


def e4a_vergleich(frozen: dict, zelle: str, g: int, regel: str, k: int, sicht: str, gegner: str = "zufall", refs: dict | None = None) -> dict:
    """Regel gegen Gegner beim Abbruch nach g Kandidaten (beide mit demselben g)."""
    z = frozen["e4a"][zelle]
    ref = refs_vektor(z["familie"], z["n"], z["instanzen"], refs)
    return vergleiche_bloecke(e4a_block(frozen, zelle, g, regel), e4a_block(frozen, zelle, g, gegner), ref, z["ks"].index(k), C.SICHTEN.index(sicht))


def e4a_luecke(frozen: dict, zelle: str, g: int, regel: str, k: int, sicht: str, refs: dict | None = None) -> float:
    z = frozen["e4a"][zelle]
    ref = refs_vektor(z["familie"], z["n"], z["instanzen"], refs)
    return float(luecke(e4a_block(frozen, zelle, g, regel)[:, :, z["ks"].index(k), C.SICHTEN.index(sicht)], ref[:, None]).mean())


# ---------------------------------------------------------------------------------------------------------------- E4b: Nachbarliste
def e4b_abstieg(frozen: dict, zelle: str, k_liste: int, regel: str) -> dict:
    """Erster Abstieg mit Nachbarliste gegen den vollen Abstieg (gleicher Strom): Größe der Prüfrunde, Zugauswertungen, Länge in Prozent gegen voll, Anteil Endtouren ohne volles Lokalminimum."""
    z = frozen["e4b"][zelle]
    a = z["liste"][str(k_liste)]["abstieg"][regel]
    l_li, l_vo = np.array(a["laenge"]), np.array(a["voll_laenge"])
    return {"pruef": float(np.mean(a["n_pruef"])), "M": z["M"], "ev": float(np.mean(a["ev"])), "ev_voll": float(np.mean(a["ev_voll"])),
            "laenge_gegen_voll": float(np.mean(100.0 * (l_li - l_vo) / l_vo)), "nicht_voll": float(np.mean(a["falsch"]))}


def e4b_ils_block(frozen: dict, zelle: str, k_liste: int, regel: str) -> np.ndarray:
    return np.array(frozen["e4b"][zelle]["liste"][str(k_liste)]["ils"][regel]["best"])


def e4b_vergleich(frozen: dict, zelle: str, k_liste: int, regel: str, k: int, sicht: str, gegner: str = "zufall", refs: dict | None = None) -> dict:
    """Suche mit Kicks innerhalb der Nachbarliste: `regel` gegen `gegner` (beide mit derselben Liste)."""
    z = frozen["e4b"][zelle]
    ref = refs_vektor(z["familie"], z["n"], z["instanzen"], refs)
    return vergleiche_bloecke(e4b_ils_block(frozen, zelle, k_liste, regel), e4b_ils_block(frozen, zelle, k_liste, gegner), ref, z["ks"].index(k), C.SICHTEN.index(sicht))


def e4b_gegen_volle_menge(frozen: dict, zelle: str, k_liste: int, regel: str, k: int, sicht: str, refs: dict | None = None) -> dict:
    """Suche mit Kicks mit Nachbarliste gegen dieselbe Regel auf der vollen Zugmenge (E3-Läufe, gleicher Budgetfaktor)."""
    z = frozen["e4b"][zelle]
    ref = refs_vektor(z["familie"], z["n"], z["instanzen"], refs)
    si = C.SICHTEN.index(sicht)
    g_l = luecke(e4b_ils_block(frozen, zelle, k_liste, regel)[:, :, z["ks"].index(k), si], ref[:, None]).mean(1)
    g_v = luecke(e3_block(frozen, zelle, regel)[:, :, frozen["e3"][zelle]["ks"].index(k), si], ref[:, None]).mean(1)
    return vergleiche_luecken(g_l, g_v)


# ---------------------------------------------------------------------------------------------------------------- Kosten K und O einer Rechnung
def kosten_zeile(regel: str, ev: float, zeilen: float) -> dict:
    """K (Zugauswertungen plus Merkmalszeilen) und O (8 je Zugauswertung plus Operationen der Zeilen) einer Rechnung, dazu der Grenzfall „Zeilen gratis“."""
    return {"frei": ev, "K": K.k_gesamt(ev, zeilen), "O": K.o_gesamt(regel, ev, zeilen)}


def training_zeilen(schritte: int, batch: int = C.Z_BATCH) -> int:
    """Zeilen des Z-Trainings: Schritte mal Zeilen je Schritt."""
    return schritte * batch


def training_touren_k(schritte: int) -> int:
    """Touren des K-Trainings: Schritte mal 256 (32 Instanzen mal 8 Touren)."""
    return schritte * C.K_BATCH_TOUREN


# ---------------------------------------------------------------------------------------------------------------- Verdikte
def verdict_instanz_e0(zuege: int, ev_zuege: int, ev_pruef: int, abbruch: int = 0, liste: int = 0, falsch: bool = False) -> str:
    """Eine Zeile zur Primäransicht (kein Verdikt): Züge, Zugauswertungen und Anteil der letzten Runde."""
    gesamt = ev_zuege + ev_pruef
    anteil = prozent(ev_pruef / gesamt) if gesamt else "0"
    if abbruch:
        return (f"Angenommene Züge: {zuege}; {zahl(ev_zuege)} Zugauswertungen bis zum letzten Zug und {zahl(ev_pruef)} in der letzten Runde (Abbruch nach {abbruch} erfolglosen Kandidaten): "
                f"{anteil} % der Zugauswertungen. " + ("Die Endtour ist kein Lokalminimum der vollen Zugmenge." if falsch else "Die Endtour ist ein Lokalminimum der vollen Zugmenge."))
    if liste:
        return (f"Angenommene Züge: {zuege}; {zahl(ev_zuege)} Zugauswertungen bis zum letzten Zug und {zahl(ev_pruef)} in der letzten Runde über die Nachbarliste (k = {liste}): {anteil} % der Zugauswertungen. "
                + ("Die Endtour ist kein Lokalminimum der vollen Zugmenge." if falsch else "Die Endtour ist auch ein Lokalminimum der vollen Zugmenge."))
    return (f"Angenommene Züge: {zuege}; {zahl(ev_zuege)} Zugauswertungen bis zum letzten Zug, dann die Prüfrunde mit {zahl(ev_pruef)} Zugauswertungen, keine verbessert: "
            f"{anteil} % von {zahl(gesamt)} Zugauswertungen beweisen nur das Lokalminimum.")


def verdict_guete(auc_netz: float, auc_hand: float, ev_rand: float, ev_netz: float, n: int | None = None) -> tuple:
    """E1: Ordnet das Netz die Züge besser als die Nachbarlisten-Reihenfolge? d = AUC Netz minus AUC Nachbarlisten-Reihenfolge; ab 0,05 klar, ab 0,01 etwas besser, sonst nicht besser.
    Der zweite Satzteil nennt immer beide Zahlen (Zugauswertungen je angenommenem Zug: Netz, Zufall)."""
    d = auc_netz - auc_hand
    a, b = de(auc_netz, 3), de(auc_hand, 3)
    zusatz = (f"es braucht im Mittel {de(ev_netz, 1)} statt {de(ev_rand, 1)} Zugauswertungen je angenommenem Zug (Zufallsreihenfolge), "
              f"zusätzlich bewertet das Netz in jedem Schritt die Merkmalszeilen aller M Kandidaten.")
    voraus = f" Voraussetzung: erster Abstieg aus dem Nächsten Nachbarn bei n = {n}, gleichverteilte Orte, Zugnetz bei n = 20 trainiert." if n is not None else ""
    if d >= C.AUC_KLAR:
        return "success", f"Das Netz ordnet die Züge klar besser als die Nachbarlisten-Reihenfolge (AUC {a} gegen {b}); {zusatz}{voraus}"
    if d >= C.AUC_ETWAS:
        return "info", f"Das Netz ordnet die Züge etwas besser als die Nachbarlisten-Reihenfolge (AUC {a} gegen {b}); {zusatz}{voraus}"
    return "warning", (f"Das Netz ordnet nicht besser als die Nachbarlisten-Reihenfolge (AUC {a} gegen {b}); das Lernen bringt hier nichts, was die Regel nicht schon leistet. "
                       f"Zugauswertungen je angenommenem Zug: {de(ev_netz, 1)} (Netz), {de(ev_rand, 1)} (Zufall).{voraus}")


def verdict_beweis(anteil: float, n: int, instanzen: int | None = None) -> tuple:
    """E2: Anteil der Prüfrunde an den Zugauswertungen eines Abstiegs; Schwellen 0,50 und 0,25 sind Darstellungsgrenzen, keine Messgrößen."""
    p = prozent(anteil)
    q = prozent(1 - anteil)
    voraus = f" (n = {n}, erster Abstieg aus dem Nächsten Nachbarn, Zufallsreihenfolge" + (f", Mittel über {instanzen} Instanzen)" if instanzen else ")")
    if anteil >= C.ANTEIL_HAELFTE:
        return "success", f"Mehr als die Hälfte der Zugauswertungen ({p} %) beweist nur, dass kein Zug mehr verbessert; eine bessere Startlösung kann höchstens die übrigen {q} % sparen.{voraus}"
    if anteil >= C.ANTEIL_VIERTEL:
        return "info", f"Zwischen einem Viertel und der Hälfte der Zugauswertungen ({p} %) beweist nur, dass kein Zug mehr verbessert; eine bessere Startlösung kann höchstens die übrigen {q} % sparen.{voraus}"
    return "info", f"Weniger als ein Viertel der Zugauswertungen ({p} %) beweist nur, dass kein Zug mehr verbessert; eine bessere Startlösung kann höchstens die übrigen {q} % sparen.{voraus}"


def verdict_zugwahl(marken: dict, n: int, familie: str, k: int, instanzen: int, seeds: int, regel_name: str = "gelernte Reihenfolge") -> tuple:
    """E3: `marken[sicht]` in {schlägt, gleich, dazwischen, verliert} für „gelernt gegen Zufall“ in den Sichten frei, K, O.
    (1) O schlägt; (2) sonst K schlägt; (3) sonst frei schlägt; (4) sonst frei gleich oder dazwischen; (5) sonst schlechter als Zufall. Jeder Satz nennt n, Familie, k, Instanzen und Zufalls-Seeds."""
    voraus = (f" (n = {n}, {C.FAMILIE_ANZEIGE.get(familie, familie)}, Budget k = {k}, {instanzen} Instanzen mit je {seeds} Zufalls-Seeds, Gegner Zufallsreihenfolge, "
              f"Zugnetz bei n = 20 trainiert, eigene Umsetzung ohne Nachbarlisten.)")
    if marken["O"] == "schlägt":
        return "success", f"Das Netz gewinnt auch bei Zählung der Operationen; das ist belastbar.{voraus}"
    if marken["K"] == "schlägt":
        return "info", f"Das Netz gewinnt in Kandidatenbewertungen (K), in Operationen (O) nicht ({marken['O']}): die Entscheidung liegt zwischen K und O.{voraus}"
    if marken["frei"] == "schlägt":
        return "info", (f"Nur wenn die Merkmalszeilen nichts kosten, gewinnt die {regel_name}; schon in K nicht mehr ({marken['K']}), und das ist die Richtung, in der K das Netz begünstigt.{voraus}")
    if marken["frei"] in ("gleich", "dazwischen"):
        return "warning", f"Auch bei kostenloser Bewertung ist die {regel_name} nicht besser als Zufall.{voraus}"
    return "warning", f"Die {regel_name} ist auch bei kostenloser Bewertung schlechter als Zufall.{voraus}"


def verdict_abbruch(anteil_falsch: float, marken: dict, g: int, n: int, familie: str, k: int, instanzen: int, seeds: int, regel_name: str = "Zufallsreihenfolge") -> tuple:
    """E4a: Mit Abbruch nach g erfolglosen Kandidaten ist das Ende kein Beweis mehr. `anteil_falsch` = Anteil der Abstiegsenden ohne volles Lokalminimum (Gegenprobe); der zweite Satz
    ist das Verdikt von E3 für gelernt gegen Zufall mit demselben Abbruch."""
    p = prozent(anteil_falsch)
    if anteil_falsch >= C.FALSCH_HOCH:
        s1 = f"Mit Abbruch nach {g} Kandidaten endet die Suche mit {regel_name} in {p} % der Abstiege an einem falschen Lokalminimum; das ist kein Beweis mehr."
    elif anteil_falsch >= C.FALSCH_NIEDRIG:
        s1 = f"Mit Abbruch nach {g} Kandidaten endet die Suche mit {regel_name} in {p} % der Abstiege an einem falschen Lokalminimum; das Ergebnis ist ein Näherungsbeweis."
    else:
        s1 = f"Mit Abbruch nach {g} Kandidaten endet die Suche mit {regel_name} in {p} % der Abstiege an einem falschen Lokalminimum, also fast nie."
    art, s2 = verdict_zugwahl(marken, n, familie, k, instanzen, seeds)
    return art, f"{s1} {s2}"


def verdict_nachbarliste(pruef_k: float, M: int, nicht_voll: float, luecke_pp: float, k: int) -> tuple:
    """E4b: Prüfrunde der Nachbarliste (k) gegen M. Spart sie über die Hälfte (pruef_k / M < 0,5), endet sie in unter 10 % der Abstiege nicht im vollen Lokalminimum und liegt die Länge
    unter 0,5 Prozentpunkte neben dem vollen Abstieg, spart sie ohne Qualitätsverlust; sonst mit schlechteren Endtouren, oder sie spart nur wenig."""
    p = prozent(nicht_voll)
    anteil = pruef_k / M
    kopf = f"Die Nachbarliste (k = {k}) senkt die Prüfrunde von {zahl(M)} auf {zahl(pruef_k)} Zugauswertungen"
    voraus = " Voraussetzung: erster Abstieg ohne Kicks, einfache Umsetzung ohne Don't-Look-Bits, Zählung der Zugauswertungen ohne den Aufbau der Listen."
    if anteil >= C.LISTE_SPAREN:
        return "warning", f"{kopf} und spart damit nur {prozent(1 - anteil)} %; {p} % der Abstiege enden nicht im vollen Lokalminimum, Länge {de_vz(luecke_pp)} Prozentpunkte gegen den vollen Abstieg.{voraus}"
    if nicht_voll >= C.LISTE_NICHT_VOLL or abs(luecke_pp) >= C.LISTE_LUECKE:
        return "warning", f"{kopf}, spart, aber um den Preis schlechterer Endtouren: {p} % der Abstiege enden nicht im vollen Lokalminimum, Länge {de_vz(luecke_pp)} Prozentpunkte gegen den vollen Abstieg.{voraus}"
    return "success", f"{kopf} und endet in {p} % der Abstiege nicht im vollen Lokalminimum; Länge {de_vz(luecke_pp)} Prozentpunkte gegen den vollen Abstieg: sie spart ohne Qualitätsverlust.{voraus}"


def verdict_liste_gelernt(marken: dict, k_liste: int, n: int) -> tuple:
    """E4b (c): Hilft die gelernte Reihenfolge innerhalb der Nachbarliste? Reine Funktion über die Marken gelernt gegen Zufall (beide mit derselben Liste)."""
    ende = f" (n = {n}, Nachbarliste k = {k_liste}, Suche mit Kicks, Marken gegen Zufallsreihenfolge mit derselben Liste)."
    if marken["frei"] == "schlägt":
        zusatz = "in K ebenfalls" if marken["K"] == "schlägt" else f"in K nicht ({marken['K']})"
        return "info", f"Die gelernte Reihenfolge verkürzt auch die Suche innerhalb der Nachbarliste, wenn die Merkmalszeilen nichts kosten; {zusatz}.{ende}"
    return "warning", f"Innerhalb der Nachbarliste bringt die gelernte Reihenfolge nichts: schon bei kostenloser Bewertung ist sie nicht besser als Zufall ({marken['frei']}).{ende}"


def verdict_start(ersparnis_netz: float, ersparnis_max: float, c_konstr: float) -> tuple:
    """E5: Eine perfekte Startlösung (schon Lokalminimum) ist die Obergrenze der Ersparnis; das Netz als Start liegt darunter (sonst ein Fehler)."""
    if ersparnis_netz >= ersparnis_max:
        raise ValueError("Die Ersparnis des Netzes kann nicht die der perfekten Startlösung erreichen")
    return "info", (f"Eine perfekte Startlösung spart bei Zufallsreihenfolge im Abstieg höchstens {de(ersparnis_max, 0)} % der Zugauswertungen, das Netz als Start {de(ersparnis_netz, 0)} %; "
                    f"in K kostet die Konstruktion {zahl(c_konstr)} Kandidatenbewertungen.")


def verdict_probe_k(gap_probe: float, gap_nn: float) -> tuple:
    """E5, Probetraining des Merkmalsnetzes K: noch über oder schon unter dem Nächsten Nachbarn (Greedy-Lücke gegen Lücke des Nächsten Nachbarn)."""
    if gap_probe >= gap_nn:
        return "warning", f"Kaum gelernt: Greedy-Lücke {de(gap_probe, 1)} % liegt noch nicht unter dem Nächsten Nachbarn ({de(gap_nn, 1)} %)."
    return "success", f"Besser als Nächster Nachbar: Greedy-Lücke {de(gap_probe, 1)} % gegen {de(gap_nn, 1)} %."


# ---------------------------------------------------------------------------------------------------------------- Tabellen für die App
def e3_tabelle(frozen: dict, zelle: str, k: int, refs: dict | None = None) -> list:
    """Zeilen für Experiment 3: je Regel die mittlere Lücke in den drei Sichten und der Vergleich mit Zufall (v, Intervall, Marke) je Sicht."""
    zeilen = []
    for regel in C.REGELN_E3:
        z = {"regel": regel}
        for s in C.SICHTEN:
            z[s] = e3_luecken(frozen, zelle, regel, k, s, refs)
            z[f"cmp_{s}"] = None if regel == "zufall" else e3_vergleich(frozen, zelle, regel, k, s, "zufall", refs)
        zeilen.append(z)
    return zeilen


def e4a_tabelle(frozen: dict, zelle: str, g: int, k: int, refs: dict | None = None) -> list:
    """Zeilen für Experiment 4a: je Regel Anteil falscher Lokalminima, Lücke je Sicht und Vergleich mit Zufall (beide mit Abbruch nach g)."""
    zeilen = []
    for regel in C.REGELN_E4A:
        z = {"regel": regel, "falsch": e4a_falsch_anteil(frozen, zelle, g, regel)}
        for s in C.SICHTEN:
            z[s] = e4a_luecke(frozen, zelle, g, regel, k, s, refs)
            z[f"cmp_{s}"] = None if regel == "zufall" else e4a_vergleich(frozen, zelle, g, regel, k, s, "zufall", refs)
        zeilen.append(z)
    return zeilen
