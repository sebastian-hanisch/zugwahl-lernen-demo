"""Zugnetz Z: ordnet die Kandidaten einer Zugmenge nach „verbessert dieser Zug die Tour jetzt?“ (überwachtes Lernen, nicht das spätere Ergebnis).

Neun Merkmale je Zug (`zwl_moves.move_features`), Netz 9-16-1 mit 177 Parametern. Trainingsdaten: Zustände aus der Suche mit Zufallsregel (Nächster-Nachbar-Start, Kicks),
je Zustand 200 zufällige Zeilen mit dem Etikett „Delta < 0“. Das Netz sieht nie das Delta. Nur in Stück 4, nie kopiert (Q1). Zufall nur über `UStream`."""

import time

import numpy as np

import zwl_constants as C
import zwl_instances as I
import zwl_mlp as MLP
import zwl_scenario as SC
from zwl_construct import nn_tour_np
from zwl_moves import Kontext
from zwl_rngvec import UStream
from zwl_search import ils


def kontext_von(familie: str, n: int, seed: int) -> Kontext:
    """Kontext der Instanz (familie, n, seed) der Quelle (1000er-Quadrat, Abstände wie `zwl_scenario`)."""
    _, D = SC.make_instance(familie, n, seed)
    return Kontext(np.array(D))


def kontext_aus_punkten(punkte) -> Kontext:
    """Kontext aus Koordinaten (n, 2)."""
    return Kontext(I.distance_matrix_np(np.asarray(punkte, dtype=np.float64)))


def sammle_zustaende(kontexte, k_budget: int = C.Z_SAMMEL_K, pro_zustand: int = C.Z_PRO_ZUSTAND, seed_basis: int = 0):
    """Zeilen für das Training: je Instanz eine Suche mit Zufallsregel aus dem Nächsten Nachbarn bis k_budget * M Zugauswertungen; an jedem Zustand `pro_zustand` verschiedene Züge.

    Jede Instanz hat eigene Ströme (abgeleitet aus Nummer und `seed_basis`), die ersten 100 Instanzen sind also bei 300 Instanzen dieselben. Gibt (F, y, Zahl der Zustände); y = 1, wenn Delta < 0."""
    Fs, ys = [], []
    for i, ctx in enumerate(kontexte):
        us_lauf = UStream(SC.derive(C.ZWECK_Z_SAMMEL, seed_basis, i))
        us_zeilen = UStream(SC.derive(C.ZWECK_Z_ZEILEN, seed_basis, i))
        m = ctx.nb.M

        def snap(t, z, us_zeilen=us_zeilen, m=m):
            sel = us_zeilen.choice(1, m, pro_zustand)[0]
            Fs.append(z["F"][sel].astype(np.float32))
            ys.append((z["delta"][sel] < -C.EPS).astype(np.float32))

        ils(ctx, nn_tour_np(ctx.D, 0), "zufall", k_budget * m, us_lauf, snap=snap)
    return np.concatenate(Fs), np.concatenate(ys), len(Fs)


def train_instanzen(count: int, n: int = C.Z_N_TRAIN, familie: str = "gleich") -> list:
    """Trainingsinstanzen des Zugnetzes: Zufalls-Seeds ab 2000 (nie dieselben wie Test 0 bis 29 und Validierung ab 1000)."""
    return [kontext_von(familie, n, C.TRAIN_OFFSET + i) for i in range(count)]


def val_instanzen(count: int, n: int = C.Z_N_TRAIN, familie: str = "gleich") -> list:
    """Validierungsinstanzen: Zufalls-Seeds ab 1000."""
    return [kontext_von(familie, n, C.VAL_OFFSET + i) for i in range(count)]


def train_zugnetz(F: np.ndarray, y: np.ndarray, steps: int, batch: int = C.Z_BATCH, lr: float = C.Z_LERNRATE, seed: int = 0,
                  progress=None, max_seconds: float | None = None):
    """Training mit Adam auf zufälligen Zeilen (mit Zurücklegen). Gibt (Parameter, Verlauf [(Schritt, Verlust)], Zähler {schritte, zeilen, sekunden, abgebrochen}).

    `progress(schritt, gesamt)` wird alle 50 Schritte gerufen; `max_seconds` ist der Zeitwächter (das Zwischenmodell bleibt nutzbar)."""
    params = MLP.init_mlp(C.Z_MERKMALE, C.H, UStream(SC.derive(C.ZWECK_Z_INIT, seed)))
    opt = MLP.Adam(params, lr)
    us = UStream(SC.derive(C.ZWECK_Z_BATCH, seed))
    Fd, yd = F.astype(np.float64), y.astype(np.float64)
    N = len(yd)
    verlauf, summe, t0, getan, abgebrochen = [], 0.0, time.time(), 0, False
    for step in range(steps):
        bi = np.minimum((us.uniform(batch) * N).astype(np.int64), N - 1)
        loss, grads = MLP.bce_loss_grad(params, Fd[bi], yd[bi])
        opt.step(params, grads)
        getan += 1
        summe += loss
        if (step + 1) % 50 == 0:
            verlauf.append((step + 1, summe / 50))
            summe = 0.0
            if progress is not None:
                progress(step + 1, steps)
            if max_seconds is not None and time.time() - t0 > max_seconds and step + 1 < steps:
                abgebrochen = True
                break
    return params, verlauf, {"schritte": getan, "zeilen": getan * batch, "sekunden": time.time() - t0, "abgebrochen": abgebrochen}


def auc(score, y) -> float:
    """AUC als Wahrscheinlichkeit, dass ein zufälliger positiver Zug höher bewertet wird als ein zufälliger negativer (Gleichstand zählt halb): Rangsumme mit gemittelten Rängen."""
    score = np.asarray(score, dtype=np.float64)
    pos = np.asarray(y) > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("AUC braucht positive und negative Beispiele")
    order = np.argsort(score, kind="stable")
    s = score[order]
    neu = np.concatenate([[True], s[1:] != s[:-1]])
    gruppe = np.cumsum(neu) - 1
    start = np.flatnonzero(neu)
    ende = np.concatenate([start[1:], [len(s)]])
    rang_je_gruppe = (start + ende + 1) / 2.0                      # mittlerer Rang (1-basiert) der Gruppe
    ranks = np.empty(len(s))
    ranks[order] = rang_je_gruppe[gruppe]
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def hand_score(F: np.ndarray) -> np.ndarray:
    """Wert der Nachbarlisten-Reihenfolge (hoch = zuerst): minus (größter Rang * 100 - mittlere Länge der entfernten Kanten)."""
    return -(np.rint(F[:, 3] * C.RANG_KAPPE) * 100.0 - F[:, 0])


EINZELMERKMALE = ("rem_mean", "rem_max", "minrank", "maxrank")


def einzelmerkmal_score(F: np.ndarray, k: int) -> np.ndarray:
    """Wert eines Einzelmerkmals mit dem Vorzeichen, bei dem AUC > 0,5 erwartet wird: lange entfernte Kanten gut (Merkmal 0, 1), kleine Ränge gut (Merkmal 2, 3)."""
    return F[:, k] if k < 2 else -F[:, k]


def auc_tabelle(F: np.ndarray, y: np.ndarray, params) -> dict:
    """AUC des Netzes, der Nachbarlisten-Reihenfolge und der vier Einzelmerkmale auf denselben Zeilen."""
    out = {"netz": auc(MLP.forward(params, F.astype(np.float64)), y), "hand": auc(hand_score(F), y)}
    for k, name in enumerate(EINZELMERKMALE):
        out[name] = auc(einzelmerkmal_score(F, k), y)
    return out


def validierungszeilen(count: int = C.Z_VAL_INSTANZEN, n: int = C.Z_N_TRAIN, familie: str = "gleich"):
    """Zeilen der Validierung: Instanzen ab Zufalls-Seed 1000, Suche bis 3 M, je Zustand 200 Zeilen (eigene Ströme, `seed_basis` 1)."""
    return sammle_zustaende(val_instanzen(count, n, familie), C.Z_VAL_K, C.Z_PRO_ZUSTAND, seed_basis=1)
