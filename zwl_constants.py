"""Konstanten: Zugmenge, Netze, Training, Instanzen, Zufalls-Seeds je Zweck, Zählannahmen, Marken der Verdikte, Regler-Grenzen, Farben und Presets.

Die Namen und Werte der Instanz-, Zweck- und Statistik-Konstanten (SEITE, FAMILIEN, ZWECK_INSTANZ, MARKE_KLAR, ...) sind Schnittstelle zu den kopierten Dateien
(zwl_scenario, zwl_instances, zwl_stats); die Netz- und Zählkonstanten (O_JE_ZEILE_Z, O_JE_ZEILE_K, K_*) kopieren die Folge-Stücke mit gleichem Namen und Wert."""

# ---------------------------------------------------------------- Instanzen (Quelle: hyperheuristik-demo, hhw_scenario; Kopie über Stück 1)
FAMILIEN = ("gleich", "cluster", "gitter")      # Index = Schlüssel der Zufalls-Seeds
FAMILIE_ANZEIGE = {"gleich": "Gleichverteilt", "cluster": "Gruppen", "gitter": "Gitter"}
FAMILIE_HAND = "hand"
SEITE = 1000.0                                  # Quadrat 1000 x 1000
CLUSTER_ZAHL, CLUSTER_SIGMA = 5, 40.0
GITTER_FAKTOR, GITTER_RAUSCHEN = 1.5, 0.1
VAL_OFFSET = 1000                               # Validierungsinstanzen: Zufalls-Seeds ab 1000
TRAIN_OFFSET = 2000                             # Trainingsinstanzen des Zugnetzes: Zufalls-Seeds ab 2000 (nie dieselben wie Test und Validierung)
HAND_PUNKTE = ((2.0, 0.0), (3.0, 0.0), (9.0, 3.0), (1.0, 9.0), (6.0, 5.0), (4.0, 2.0))   # Herkunft der Handtabelle (nur Dokumentation)
HAND_SKALA = 1.0
N_STUFEN = (6, 10, 20, 50)                      # 6: nur das Handbeispiel
N_HAND = 6
SEED_MAX = {6: 0, 10: 19, 20: 19, 50: 9}        # größter Zufalls-Seed der Instanz je n (Referenzen vorgerechnet)
N_REF_SEEDS = {10: 20, 20: 20, 50: 10}

# ---------------------------------------------------------------- Zugmenge und Zählung (Q2, Q5)
ZUGARTEN = ("2-opt", "Or-opt 1", "Or-opt 2", "Or-opt 3")
OR_LAENGEN = (1, 2, 3)
EPS = 1e-12                                     # ein Zug verbessert, wenn Delta < -EPS
RANG_KAPPE = 8                                  # Ränge werden bei 8 gekappt (Merkmale)
LAENGEN_KAPPE = 6.0                             # Längen durch s werden bei 6 gekappt
OPS_JE_KANDIDAT = 8                             # Annahme: grobe Operationen je Zugauswertung (Delta aus Kanten); wie znz_cost
O_JE_ZEILE_Z = 9 * 16 + 16 + 16                 # Zugnetz Z 9-16-1: 144 + 16 (tanh) + 16 = 176 (Merkmalsbildung nicht mitgezählt)
O_JE_ZEILE_K = 7 * 16 + 16 + 16                 # Merkmalsnetz K 7-16-1: 112 + 16 + 16 = 144 (Merkmalsbildung nicht mitgezählt)
O_JE_ZEILE_HAND = OPS_JE_KANDIDAT               # Annahme, Untergrenze: eine Zeile der Nachbarlisten-Reihenfolge kostet wie eine Zugauswertung
O_FARTHEST_FAKTOR = 4                           # Annahme: Farthest Insertion 4 n^2 grobe Operationen
M_MODELL_ZEICHEN = "M"

# ---------------------------------------------------------------- Regeln (Reihenfolge, in der die Suche die Kandidaten probiert)
REGELN = ("zufall", "nachbarliste", "gelernt", "gelernt_det", "steilster", "aufzaehlung")
REGEL_ANZEIGE = {
    "zufall": "Zufall",
    "nachbarliste": "Nachbarlisten-Reihenfolge",
    "gelernt": "Gelernt (gezogen)",
    "gelernt_det": "Gelernt (deterministisch)",
    "steilster": "Steilster Abstieg",
    "aufzaehlung": "Aufzählung (Handbeispiel)",
}
REGEL_KURZ = {"zufall": "Zufall", "nachbarliste": "Nachbarlisten-Reihenfolge", "gelernt": "gelernt", "gelernt_det": "gelernt det.", "steilster": "steilster", "aufzaehlung": "Aufzählung"}
REGELN_E3 = ("zufall", "nachbarliste", "gelernt", "gelernt_det", "steilster")
REGELN_E4A = ("zufall", "nachbarliste", "gelernt")       # Regeln mit Abbruch der Prüfung
REGELN_E4B = ("zufall", "gelernt")                       # Regeln mit Nachbarliste
E4A_G = (50, 100, 200)
REGELN_MIT_NETZ = ("gelernt", "gelernt_det")
REGELN_MIT_ZEILEN = ("nachbarliste", "gelernt", "gelernt_det")     # diese Regeln bewerten alle Kandidaten der Zugmenge je Schritt (Merkmalszeilen)
SICHTEN = ("frei", "K", "O")                    # Zeilen gratis (Grenzfall), Kandidatenbewertungen, grobe Operationen
SICHT_ANZEIGE = {"frei": "nur Zugauswertungen (Zeilen gratis)", "K": "K (Zeilen zählen wie Auswertungen)", "O": "O (grobe Operationen)"}
SICHT_GEWICHT_Z = {"frei": 0.0, "K": 1.0, "O": O_JE_ZEILE_Z / OPS_JE_KANDIDAT}      # Zeilen des Zugnetzes in Zugauswertungen gerechnet (176 / 8 = 22)
SICHT_GEWICHT_HAND = {"frei": 0.0, "K": 1.0, "O": O_JE_ZEILE_HAND / OPS_JE_KANDIDAT}

# ---------------------------------------------------------------- Netze (Q1: Z = Zugnetz mit 9 Merkmalen, K = Merkmalsnetz zur Konstruktion mit 7)
H = 16                                          # versteckte Einheiten beider Netze
Z_MERKMALE, K_MERKMALE = 9, 7
Z_PARAMS = H * Z_MERKMALE + H + H + 1           # 177
K_PARAMS = H * K_MERKMALE + H + H + 1           # 145
Z_SCHRITTE_STUFEN = (300, 500, 1000, 1500, 2000, 3000)
Z_SCHRITTE_STANDARD, Z_SCHRITTE_MAX = 1500, 3000
Z_SCHRITTE_FROZEN = 3000                        # vorgerechnete Modelle: volle Größe
Z_INSTANZEN_STUFEN = (50, 100, 200, 300)
Z_INSTANZEN_STANDARD, Z_INSTANZEN_FROZEN = 100, 300
Z_BATCH = 4096                                  # Zeilen je Trainingsschritt
Z_LERNRATE = 0.01
Z_PRO_ZUSTAND = 200                             # Zeilen je gesammeltem Zustand
Z_SAMMEL_K = 6                                  # Datensammlung: Suche mit Zufallsregel bis 6 M Zugauswertungen je Trainingsinstanz
Z_N_TRAIN = 20
Z_VAL_INSTANZEN, Z_VAL_INSTANZEN_FROZEN = 20, 50    # Validierungsinstanzen der AUC (live / eingefroren)
Z_VAL_K = 3                                     # Validierungszustände: Suche bis 3 M
K_LERNRATE = 0.03
K_CLIP = 5.0
K_INST, K_ROLL = 32, 8                          # Instanzen und Touren (verschiedene Startstädte) je Trainingsschritt
K_BATCH_TOUREN = K_INST * K_ROLL                # 256
K_SCHRITTE_FROZEN = 3750                        # 3 750 x 256 = 960 000 Touren (Rezept von L2)
K_SCHRITTE_PROBE = 150                          # Probetraining in der App: 150 x 256 = 38 400 Touren
K_N_TRAIN = 20
K_BEGRENZUNG_ZEIGER = 6.0
MODELL_FORMAT = "zwl-mlp-1"
FROZEN_FORMAT = "zwl-frozen-1"
REFS_FORMAT = "zwl-refs-1"
MODELL_ORDNER = "zwl_modelle"
MODELL_SEEDS = 5                                # Gewichts-Zufalls-Seeds 0 bis 4
FEATURE_VERSION = 1
ZEITWAECHTER_SEKUNDEN = 100.0                   # Training in der App bricht danach ab (Zwischenmodell bleibt nutzbar)

# ---------------------------------------------------------------- Zwecke der Zufalls-Seeds (zwl_scenario.derive); reservierter Bereich der Demo: 100 bis 149
ZWECK_INSTANZ, ZWECK_START, ZWECK_LAUF = 11, 33, 24     # Werte der Quelle hhw_scenario (von zwl_scenario.py gebraucht)
ZWECK_Z_INIT, ZWECK_Z_SAMMEL, ZWECK_Z_ZEILEN, ZWECK_Z_BATCH = 100, 101, 102, 103
ZWECK_K_INIT, ZWECK_K_INST, ZWECK_K_START, ZWECK_K_ZIEHEN = 104, 105, 106, 107
ZWECK_SUCHE, ZWECK_E0, ZWECK_BOOT, ZWECK_E5 = 108, 109, 110, 111

# ---------------------------------------------------------------- Verdikte (gesetzt, nicht abgeleitet)
MARKE_KLAR, MARKE_GLEICH = 0.25, 0.10           # Prozentpunkte (DEMO-PLAYBOOK §4)
BOOTSTRAP_ZIEHUNGEN = 2000
AUC_KLAR, AUC_ETWAS = 0.05, 0.01                # Verdikt E1: Abstand der AUC zur Nachbarlisten-Reihenfolge
ANTEIL_HAELFTE, ANTEIL_VIERTEL = 0.50, 0.25     # Verdikt E2 (Darstellungsgrenzen, keine Messgrößen)
FALSCH_HOCH, FALSCH_NIEDRIG = 0.25, 0.05        # Verdikt E4 (a)
LISTE_SPAREN, LISTE_NICHT_VOLL, LISTE_LUECKE = 0.5, 0.1, 0.5   # Verdikt E4 (b)

# ---------------------------------------------------------------- Regler
BUDGET_K = (2, 8, 32)
ABBRUCH_STUFEN = (0, 50, 100, 200)              # 0 = aus (volle Prüfrunde)
LISTE_STUFEN = (0, 3, 5, 8)                     # 0 = aus
LISTE_DEFAULT_K = 3
E2_N_LISTE = (10, 20, 30, 40, 50)
E2_INSTANZEN = 20
E3_LIVE_SEEDS = 5
DEFAULT_FAMILIE, DEFAULT_N, DEFAULT_SEED, DEFAULT_REGEL = "gleich", 20, 0, "zufall"
DEFAULT_QUELLE, DEFAULT_GEWICHTS_SEED = "vorgerechnet", 0
QUELLEN = ("vorgerechnet", "trainieren")
QUELLE_ANZEIGE = {"vorgerechnet": "Vorgerechnet", "trainieren": "Hier trainieren"}
START_ARTEN = ("zufall", "nn", "farthest", "netz_k", "optimum")
START_ANZEIGE = {"zufall": "Zufallstour", "nn": "Nächster Nachbar", "farthest": "Farthest Insertion", "netz_k": "Merkmalsnetz K (greedy)", "optimum": "schon Lokalminimum"}

# ---------------------------------------------------------------- Farben
COLOR_ZUFALL = "#8c8c8c"
COLOR_LISTE = "#f58518"
COLOR_GELERNT = "#2F6B65"
COLOR_GELERNT_DET = "#54a24b"
COLOR_STEIL = "#b279a2"
COLOR_BEWEIS = "#e45756"
COLOR_ZUG = "#4c78a8"
COLOR_GRAU = "rgba(150,150,150,0.75)"
REGEL_FARBE = {"zufall": COLOR_ZUFALL, "nachbarliste": COLOR_LISTE, "gelernt": COLOR_GELERNT, "gelernt_det": COLOR_GELERNT_DET, "steilster": COLOR_STEIL, "aufzaehlung": "#9d755d"}

# ---------------------------------------------------------------- Presets (das erste ist die Voreinstellung beim Öffnen)
_BASIS = {
    "familie": DEFAULT_FAMILIE, "n": DEFAULT_N, "seed": DEFAULT_SEED, "regel": DEFAULT_REGEL, "quelle": DEFAULT_QUELLE, "gewicht": DEFAULT_GEWICHTS_SEED,
    "schritte": Z_SCHRITTE_STANDARD, "trainings_inst": Z_INSTANZEN_STANDARD, "budget": 2, "abbruch": 0, "liste": 0,
}
PRESETS = {
    "Der Beweis kostet etwa die Hälfte": dict(_BASIS),
    "Handbeispiel: sechs Orte": {**_BASIS, "n": 6, "regel": "aufzaehlung"},
    "Das Netz ordnet besser": {**_BASIS, "regel": "gelernt", "budget": 8},
    "Gleiches Budget, kleiner Einsatz": {**_BASIS, "regel": "gelernt", "budget": 2},
    "Größere Tour": {**_BASIS, "n": 50, "budget": 2},
    "Nachbarliste statt Beweis": {**_BASIS, "n": 50, "liste": 3},
    "Abbruch statt Beweis": {**_BASIS, "n": 50, "regel": "gelernt", "abbruch": 100},
}
PRESET_HELP = {
    "Der Beweis kostet etwa die Hälfte": "20 Orte gleichverteilt, Zufalls-Seed 0, Regel Zufall: ein Abstieg aus dem Nächsten Nachbarn; zeigt, wie viele Zugauswertungen nur beweisen, dass kein Zug mehr verbessert (im Mittel über 20 Instanzen 53,7 %, Experiment 2).",
    "Handbeispiel: sechs Orte": "Sechs Orte als Entfernungstabelle zum Nachrechnen (93 Züge), Regel Aufzählung: ein Zug, dann die Prüfrunde.",
    "Das Netz ordnet besser": "20 Orte, Regel gelernt, vorgerechnetes Zugnetz (Gewichts-Zufalls-Seed 0, 3 000 Schritte, 300 Trainingsinstanzen): Güte der Reihenfolge und Zugauswertungen je Zug (Budget k = 8 gilt erst bei Experiment 3).",
    "Gleiches Budget, kleiner Einsatz": "20 Orte, Budget k = 2 (2 M Zugauswertungen), Regel gelernt: der Hauptvergleich in drei Zählweisen.",
    "Größere Tour": "50 Orte, Budget k = 2: der Anteil der Prüfrunde sinkt, der Hauptvergleich in K bleibt.",
    "Nachbarliste statt Beweis": "50 Orte, Nachbarliste k = 3: die Prüfrunde schrumpft auf 3 057 von 12 875 Zugauswertungen, aber rund 18 % der Endtouren sind kein volles Lokalminimum.",
    "Abbruch statt Beweis": "50 Orte, Abbruch der Prüfung nach 100 erfolglosen Kandidaten, Regel gelernt: ohne Beweis, mit Anteil falscher Lokalminima.",
}
