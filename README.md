# Gelernte Zugwahl: wo steckt der Aufwand? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-zugwahl-lernen-demo.streamlit.app/)** -->

Vierte Demo der Linie **Neuronale kombinatorische Optimierung erklärt** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Kante vom Zeigernetz (erste Demo der Linie) ist ein **Kontrast**: Verbesserung statt Konstruktion.
Die Demo ist außerdem die Quelle des **Merkmalsnetzes K** (Konstruktion, 145 Parameter) für die Folge-Demos und deckt die „zustandsabhängige Auswahl“ (Reserve R der Hyperheuristiken) nur teilweise ab: gemessen ist die Zugwahl aus dem Zustand der Tour, nicht die Zugart nach Suchphase.
Vehikel ist die Rundtour (TSP): ein von Hand in numpy gebautes Netz (Zugnetz Z, 177 Parameter) ordnet die 2-opt- und Or-opt-Züge, die Suche probiert sie in dieser Reihenfolge, die Demo zählt, wo der Aufwand steckt.

## Kernfrage und Befund in Kürze

**Ein kleines Netz ordnet die Züge nach AUC und Zugauswertungen je Zug klar besser als die Nachbarlisten-Reihenfolge; bei gleichem Budget verliert es gegen die Zufallsreihenfolge, sobald seine Merkmalszeilen mitgezählt werden (in K in 6 von 7 gemessenen Zellen, in O in 7 von 7, davon 6 mangels Budget für einen Schritt des Netzes): wo steckt der Aufwand?** Jede Aussage gilt für diese Voraussetzungen: eigene Umsetzung, erzeugte Rundtouren, einfachste lokale Suche (2-opt und Or-opt bis Länge 3, erste Verbesserung, Kicks) ohne Nachbarlisten als Standard und ohne Don't-Look-Bits, Zugnetz nur bei n = 20 trainiert.

* **Das Netz ordnet besser** (Experiment 1). AUC 0,978 gegen 0,879 der Nachbarlisten-Reihenfolge (Validierung mit 50 Instanzen, n = 20, 197 600 Zeilen aus 988 Zuständen mit je 200 gezogenen Zeilen von 1 850; die AUC ist über alle Zeilen gemeinsam gerechnet, je Zustand gemittelt 0,980 gegen 0,912; alle fünf Gewichts-Zufalls-Seeds 0,978). Je angenommenem Zug braucht die Suche bei n = 20 im Mittel 12,0 Zugauswertungen mit dem Netz (zusätzlich zu den Merkmalszeilen für alle M Kandidaten je Schritt; gezogen; deterministisch 10,2) statt 185,2 beim Zufall und 41,5 bei der Nachbarlisten-Reihenfolge (erster Abstieg aus dem Nächsten Nachbarn, 20 Instanzen mit je 5 Zufalls-Seeds); bei n = 50 37,9 statt 900,8 (10 Instanzen).
* **Der Beweis ist der große Posten** (Experiment 2). Die Prüfrunde bewertet alle M(n) = n(11n − 35)/2 Züge (Zugbeschreibungen): 1 850 bei n = 20, 12 875 bei n = 50. Das sind im Mittel 53,7 % bzw. 39,1 % der Zugauswertungen eines Abstiegs mit Zufallsreihenfolge aus dem Nächsten Nachbarn (mit Durchläufen statt Zufallsreihenfolge sind es in `rechnen-beim-einsatz-demo` bei n = 20 67 %); eine bessere Startlösung kann also höchstens den Rest sparen. In einer Suche mit Kicks sind es bei den Regeln mit Merkmalszeilen 94 bis 99 % (n = 20) bzw. 82 bis 98 % (n = 50 gleichverteilt).
* **Bei gleichem Budget** (Experiment 3, sieben Zellen aus Größe, Familie und Budget k): nur mit gratis Merkmalszeilen gewinnt das Netz, und das in 3 von 7 Zellen (n = 20 mit k = 2, n = 50 mit k = 2 gleichverteilt und in Gruppen); in 4 von 7 gleich gut. In K (Zeilen zählen wie Zugauswertungen) verliert es in 6 von 7 Zellen, in der siebten (n = 20, k = 32) ist der Unterschied klein (+0,12 Prozentpunkte); in O (grobe Operationen) verliert es in 7 von 7 Zellen, in 6 davon, weil das Budget nicht einmal für einen Schritt des Netzes reicht (siehe unten).
* **Abbruch und Nachbarliste** (Experiment 4). Der Abbruch der Prüfung spart Zugauswertungen, ist aber kein Beweis mehr: bei Abbruch nach 50 Kandidaten enden 99 % (n = 20) bzw. 100 % (n = 50) der Abstiegsenden in der Suche mit Kicks (Zufallsreihenfolge) an einem Lokalminimum, das eine volle Prüfrunde noch verbessert; beim Netz sind es 7 % bzw. 46 % (n = 50 nur mit Budget k = 2, 10 Instanzen, das bei n = 20 trainierte Netz). Die Nachbarliste (k = 3) senkt die Prüfrunde bei n = 20 von 1 850 auf 648 und bei n = 50 von 12 875 auf 3 057 Zugauswertungen; bei n = 50 enden dann aber 18 % der Abstiege nicht im vollen Lokalminimum.
* **Startlösung** (Experiment 5). Eine perfekte Startlösung (schon Lokalminimum) spart bei Zufallsreihenfolge im Abstieg höchstens 47 % (n = 20) bzw. 61 % (n = 50) der Zugauswertungen; das Merkmalsnetz K als Start spart 10 % bzw. 10 %, Farthest Insertion 27 % bzw. 27 %. Die Konstruktion kostet in K n(n − 1)/2: 190 bzw. 1 225.

Gezählt wird in drei Sichten nebeneinander, **K begünstigt das Netz, O benachteiligt es**; „das Netz verliert“ gilt als belastbar, wenn es schon in K verliert. Zeit ist nie ein Befund.

## Warum dieses Problem

Die erste Demo der Linie baut die Tour mit einem Netz und bleibt hinter Nächster Nachbar plus 2-opt zurück. Naheliegend ist die Gegenfrage: Was kann das Netz in der **Verbesserung** leisten? Die Antwort hängt an der Zähleinheit. Ein Netz, das die Reihenfolge der Züge lernt, spart Zugauswertungen bis zum nächsten verbessernden Zug, muss dafür aber in jedem Schritt alle M Kandidaten bewerten, und wenn kein Zug mehr verbessert, hilft keine Reihenfolge: die Prüfrunde bewertet alle M Züge, egal wie sie geordnet sind.

## Modell

Rundtour (TSP) mit n Orten und euklidischen Entfernungen. **Zugmenge:** 2-opt und Or-opt mit Segmenten von 1 bis 3 Orten (bei mehr als einem Ort auch umgekehrt), gezählt als Zugbeschreibungen, die Verschiebung eines einzelnen Ortes nur einmal: M(n) = n(11n − 35)/2 Züge, also 1 850 bei n = 20 und 12 875 bei n = 50. Einzelne Züge liefern dieselbe Nachbartour (zum Beispiel ergibt ein Or-opt-Segment der Länge 2, an die Nachbarstelle gesetzt, dieselbe Tour wie ein 2-opt-Zug oder wie ein Or-opt-Zug des Nachbarortes); bei n = 20 sind es 1 570 verschiedene Nachbartouren, bei n = 50 12 175. Die Prüfrunde bewertet alle M(n) Züge je einmal. Die frühere Vormessung zählte die Verschiebung eines einzelnen Ortes in beiden Richtungen (M(n) + n(n − 2): 2 210 und 15 275); die App hält diese Zählweise hinter einem Schalter.
**Suche:** Nächster Nachbar als Startlösung, erste Verbesserung, Double-Bridge-Kicks (angenommen bei Länge höchstens der besten bisherigen). **Zugnetz Z:** neun Merkmale je Zug (mittlere und größte Länge der entfernten Kanten, kleinster und größter Rang der neuen Kanten, Zugart, Umkehr; durch den mittleren Abstand zum nächsten Nachbarn geteilt), ein Netz 9-16-1 mit 177 Parametern, logistischer Verlust auf dem Etikett „Delta < 0“, von Hand in numpy (Rückwärtspass gegen zentrale Differenzen geprüft). **Merkmalsnetz K:** sieben Merkmale je noch nicht besuchtem Ort, 7-16-1 mit 145 Parametern, REINFORCE mit dem Mittel der 8 Touren derselben Instanz als Referenzwert. Alle Formeln stehen im Expander „📐 Mathematische Formulierung“ der App.

## Methodik

* **Regeln** (Reihenfolge der Kandidaten): Zufall, Nachbarlisten-Reihenfolge (eine feste Handregel, die nur ordnet und dafür M Zeilen je Schritt bewertet), gelernt gezogen (Gumbel), gelernt deterministisch, steilster Abstieg (alle M Züge bewerten); dazu die Aufzählung im Handbeispiel.
* **Gegner der gelernten Reihenfolge:** der Zufall bei gleichem Budget; die **Nachbarliste** als beschränkte Zugmenge (Experiment 4, zwei Gegner: Nachbarlisten-Reihenfolge ordnet nur, die klassische Nachbarliste beschränkt) und der Abbruch der Prüfung. Das Zugnetz bekommt dieselben Startlösungen, Zufalls-Seeds (gepaart) und Budgets wie der Zufall.
* **Zählung:** drei Sichten nebeneinander: nur Zugauswertungen (Merkmalszeilen gratis, Grenzfall), K (Zeilen zählen wie Zugauswertungen) und O (grobe Operationen: 8 je Zugauswertung, 176 je Zeile des Zugnetzes; die Zeile der Nachbarlisten-Reihenfolge mit 8, eine Untergrenze). K begünstigt das Netz, O benachteiligt es.
* **Messgröße:** Lücke in Prozent über der Referenz (Held-Karp bei n = 20, CP-SAT bei n = 50, dort teils nur Untergrenze), Mittel über Zufalls-Seeds je Instanz, gepaarte Differenz zum Zufall mit 95-%-Bootstrap-Intervall über die Instanzen. Marken: schlägt ab 0,25 Prozentpunkten mit Intervall ohne 0; gleich unter 0,10 oder mit 0 im Intervall; sonst dazwischen.
* **Zufall:** SplitMix64 (`zwl_rng.py` ist die Vorlage, `zwl_rngvec.py` die vektorisierte Ziehung aus der ersten Demo der Linie), kein `numpy.random` in App und Kern.

## Befunde (gemessen, keine Behauptungen)

Alle Messungen mit SplitMix64-Strömen der Demo (`zwl_rng.py`, `zwl_rngvec.py`), Trainingsläufe nur als Band wiederholbar. Referenzen: Held-Karp bei n = 20, CP-SAT bei n = 50 (in den gemessenen Zellen 18 von 20 Instanzen als optimal bewiesen, mit der Familie Gitter 28 von 30; sonst Lücke zur besten bekannten Länge, eine Untergrenze). „Test“ nennt die Datei und Testfunktion, die die Zahl prüft.

### Experiment 1: Ordnet das Netz besser?

| Größe | Wert | Voraussetzung | Test |
|---|---|---|---|
| AUC Zugnetz | 0,978 (fünf Gewichts-Zufalls-Seeds: 0,978 bis 0,978) | 50 Validierungsinstanzen, n = 20 gleichverteilt, 988 Zustände, 197 600 Zeilen, 1,5 % verbessernde Züge | `test_claims.py::test_befund_e1_guete` |
| AUC Nachbarlisten-Reihenfolge, Einzelmerkmale | 0,879; 0,77 bis 0,85 (mittlere entfernte Länge 0,84, größte 0,82, kleinster Rang 0,85, größter Rang 0,77) | wie oben | `test_claims.py::test_befund_e1_guete` |
| Zugauswertungen je angenommenem Zug, n = 20 | Zufall 185,2; Nachbarlisten-Reihenfolge 41,5; gelernt 12,0; gelernt deterministisch 10,2; steilster Abstieg 1 850 | erster Abstieg aus Nächster Nachbar, 20 Instanzen mit je 5 Zufalls-Seeds, Zugnetz mit Gewichts-Zufalls-Seed 0 | `test_claims.py::test_befund_e1_je_zug` |
| dasselbe, n = 50 | 900,8; 188,5; 37,9; 32,4; 12 875 | 10 Instanzen mit je 5 Zufalls-Seeds | `test_claims.py::test_befund_e1_je_zug` |
| Zugarten unter den 20 bestbewerteten Kandidaten | 2-opt 22 % (Anteil an M: 9 %), Or-opt 1: 32 % (19 %), Or-opt 2: 30 % (37 %), Or-opt 3: 15 % (35 %) | 20 Instanzen, 173 Zustände der Abstiege (Zufallsregel), Zugnetz Seed 0 | `test_claims.py::test_befund_e1_zugarten` |

Das Netz bevorzugt kurze Umordnungen: 2-opt und kurze Or-opt-Segmente stehen überproportional oben. Gemessen ist die Zugwahl aus dem Zustand der Tour, nicht die Zugart nach Suchphase oder Stagnation.

### Experiment 2: Wie viel der Arbeit beweist nur, dass es nicht weitergeht?

Erster Abstieg aus dem Nächsten Nachbarn, Zufallsreihenfolge, 20 gleichverteilte Instanzen mit je 5 Zufalls-Seeds; Zählweise der Demo (Zugbeschreibungen, Einzelortverschiebung einmal) und Zählweise der Vormessung (Einzelortverschiebung doppelt, Aufzählungsreihenfolge, 1 Lauf je Instanz).

| n | M(n) | Zugauswertungen bis zum Lokalminimum (Mittel, SD) | Anteil der Prüfrunde | M(n) alt | Mittel alt | Anteil alt |
|---|---|---|---|---|---|---|
| 10 | 375 | 512 (128) | 73,2 % | 455 | 536 | 84,9 % |
| 20 | 1 850 | 3 446 (1 070) | 53,7 % | 2 210 | 3 701 | 59,7 % |
| 30 | 4 425 | 9 510 (2 583) | 46,5 % | 5 265 | 10 874 | 48,4 % |
| 40 | 8 100 | 18 505 (5 364) | 43,8 % | 9 620 | 22 740 | 42,3 % |
| 50 | 12 875 | 32 914 (8 386) | 39,1 % | 15 275 | 44 421 | 34,4 % |

Test: `test_claims.py::test_befund_e2_pruefrunde` (Bänder, neu gerechnet). Aufschlüsselung von M(n): 2-opt n(n − 3)/2, Or-opt 1 n(n − 2), Or-opt 2 2n(n − 3), Or-opt 3 2n(n − 4) (`test_moves.py::test_formel_gleich_aufzaehlung`).
**Suche mit Kicks:** Anteil der Prüfrunden an allen Zugauswertungen (alle Läufe der Zelle, Budget bis k_max · M; Test `test_claims.py::test_befund_e2_kicks`):

| Zelle | Zufall | Nachbarlisten-Reihenfolge | gelernt | gelernt deterministisch |
|---|---|---|---|---|
| n = 20 gleichverteilt | 59 % | 94 % | 98 % | 99 % |
| n = 50 gleichverteilt | 42 % | 82 % | 97 % | 98 % |
| n = 50 Gruppen | 40 % | 59 % | 84 % | 86 % |

### Experiment 3: Wer gewinnt bei gleichem Budget?

Suche aus dem Nächsten Nachbarn mit Kicks bis k · M Zugauswertungen; **Differenz = Lücke gelernt minus Lücke Zufall in Prozentpunkten (negativ = das Netz ist besser)**, Intervall: 95 % Bootstrap über die Instanzen. Zellen: n = 20 gleichverteilt (20 Instanzen × 10 Zufalls-Seeds, gelernt mit 5 Gewichts-Seeds reihum), n = 50 gleichverteilt und Gruppen (je 10 Instanzen × 10 Zufalls-Seeds, 3 Gewichts-Seeds). Spalte „gelernt“: Lücke in nur Zugauswertungen / K / O. Test: `test_claims.py::test_befund_e3_hauptvergleich`.

| Zelle | k | Zufall (Lücke) | gelernt (Lücke) | gelernt gegen Zufall: nur Zugauswertungen | in K | in O |
|---|---|---|---|---|---|---|
| n = 20 gleichverteilt | 2 | 0,51 % | 0,23 / 11,82 / 16,68 % | -0,27 [-0,60; -0,02] schlägt | +11,31 [+8,60; +14,02] verliert | +16,17 [+12,39; +19,78] verliert |
| n = 20 gleichverteilt | 8 | 0,06 % | 0,05 / 1,55 / 16,68 % | -0,01 [-0,08; +0,06] gleich | +1,49 [+0,72; +2,34] verliert | +16,62 [+12,76; +20,21] verliert |
| n = 20 gleichverteilt | 32 | 0,00 % | 0,00 / 0,12 / 11,82 % | 0,00 [0,00; 0,00] gleich | +0,12 [0,00; +0,27] dazwischen | +11,82 [+8,98; +14,63] verliert |
| n = 50 gleichverteilt | 2 | 2,50 % | 1,77 / 19,75 / 21,57 % | -0,72 [-1,01; -0,44] schlägt | +17,25 [+13,05; +21,37] verliert | +19,07 [+14,60; +23,52] verliert |
| n = 50 gleichverteilt | 8 | 1,23 % | 0,91 / 9,81 / 21,57 % | -0,32 [-0,73; +0,07] gleich | +8,58 [+6,32; +10,66] verliert | +20,34 [+15,81; +24,74] verliert |
| n = 50 Gruppen | 2 | 0,85 % | 0,48 / 19,58 / 20,88 % | -0,37 [-0,54; -0,19] schlägt | +18,73 [+13,88; +23,52] verliert | +20,03 [+15,05; +24,94] verliert |
| n = 50 Gruppen | 8 | 0,30 % | 0,20 / 12,46 / 20,88 % | -0,10 [-0,15; -0,04] gleich | +12,16 [+8,60; +15,61] verliert | +20,58 [+15,76; +25,40] verliert |

* **Zählung über die sieben Zellen** (`test_claims.py::test_befund_e3_zaehlung`): nur Zugauswertungen: 3-mal schlägt, 4-mal gleich; K: 6-mal verliert, 1-mal dazwischen; O: 7-mal verliert.
* **In O reicht das Budget in 6 von 7 Zellen nicht für einen Schritt des Netzes** (`test_claims.py::test_befund_e3_o_ohne_einen_schritt`): eine Merkmalszeile kostet 176 Operationen, eine Zugauswertung 8; ein Schritt des Netzes bewertet M Zeilen, also 22 · M Zugauswertungen. Budgets unter 22 M (k = 2 und 8) lassen keinen Schritt zu, die Lücke ist dann die der Startlösung (Nächster Nachbar: 16,68 % bei n = 20, 21,57 % bei n = 50 gleichverteilt, 20,88 % in Gruppen). Das ist die Leserichtung von O, nicht ein Lernfehler: die Merkmalsbildung ist dabei noch nicht mitgezählt.
* **Andere Regeln** (Marken gegen Zufall, nur Zugauswertungen / K / O; Test `test_claims.py::test_befund_e3_andere_regeln`):

| Zelle | k | Nachbarlisten-Reihenfolge | gelernt deterministisch | steilster Abstieg |
|---|---|---|---|---|
| n = 20 gleichverteilt | 2 | schlägt / verliert / verliert | schlägt / verliert / verliert | verliert / verliert / verliert |
| n = 20 gleichverteilt | 8 | gleich / verliert / verliert | gleich / verliert / verliert | gleich / gleich / gleich |
| n = 20 gleichverteilt | 32 | gleich / gleich / gleich | gleich / gleich / verliert | dazwischen / dazwischen / dazwischen |
| n = 50 gleichverteilt | 2 | gleich / verliert / verliert | gleich / verliert / verliert | verliert / verliert / verliert |
| n = 50 gleichverteilt | 8 | schlägt / verliert / verliert | gleich / verliert / verliert | verliert / verliert / verliert |
| n = 50 Gruppen | 2 | gleich / verliert / verliert | schlägt / verliert / verliert | verliert / verliert / verliert |
| n = 50 Gruppen | 8 | gleich / verliert / verliert | gleich / verliert / verliert | verliert / verliert / verliert |

* **Streuung über die Gewichts-Seeds** (gelernt, n = 20, k = 2, K): 10,97 bis 13,62 % (Lücke je Gewichts-Seed; `test_claims.py::test_befund_e3_hauptvergleich`).

### Experiment 4: Lässt sich der Beweis abkürzen, und was kostet das?

**(a) Abbruch der Prüfung** nach g erfolglosen Kandidaten (Test `test_claims.py::test_befund_e4a_abbruch`). Zellen: n = 20 gleichverteilt (20 Instanzen × 10 Zufalls-Seeds, k = 2 und 8) und n = 50 gleichverteilt (10 Instanzen × 5 Zufalls-Seeds, nur k = 2, weil Läufe mit Abbruch wegen der vielen Kicks rund das Zehnfache an Rechenschritten kosten). „Falsch“: Anteil der Abstiegsenden, bei denen eine volle Prüfrunde noch einen verbessernden Zug findet (Gegenprobe), für Zufall / Nachbarlisten-Reihenfolge / gelernt. Differenz: gelernt minus Zufall mit demselben Abbruch.

| Zelle | g | k | Abstiege an falschem Lokalminimum | Lücke Zufall / gelernt (nur Zugauswertungen) | gelernt gegen Zufall: nur Zugauswertungen | in K | in O |
|---|---|---|---|---|---|---|---|
| n = 20 gleichverteilt | 50 | 2 | 99 / 38 / 7 % | 3,81 / 0,00 % | -3,81 [-4,43; -3,20] schlägt | +8,03 [+5,52; +10,65] verliert | +12,87 [+9,30; +16,32] verliert |
| n = 20 gleichverteilt | 50 | 8 | 99 / 38 / 7 % | 1,02 / 0,00 % | -1,02 [-1,33; -0,75] schlägt | +0,68 [-0,09; +1,53] gleich | +15,66 [+11,95; +19,19] verliert |
| n = 20 gleichverteilt | 100 | 2 | 96 / 26 / 2 % | 1,96 / 0,00 % | -1,96 [-2,28; -1,64] schlägt | +9,86 [+7,23; +12,50] verliert | +14,72 [+11,07; +18,14] verliert |
| n = 20 gleichverteilt | 100 | 8 | 96 / 26 / 2 % | 0,24 / 0,00 % | -0,24 [-0,34; -0,14] dazwischen | +1,34 [+0,55; +2,20] verliert | +16,44 [+12,62; +20,06] verliert |
| n = 20 gleichverteilt | 200 | 2 | 85 / 7 / 0 % | 1,09 / 0,03 % | -1,06 [-1,38; -0,78] schlägt | +10,73 [+8,04; +13,39] verliert | +15,59 [+11,85; +19,18] verliert |
| n = 20 gleichverteilt | 200 | 8 | 85 / 7 / 0 % | 0,11 / 0,00 % | -0,11 [-0,19; -0,05] dazwischen | +1,46 [+0,66; +2,34] verliert | +16,57 [+12,72; +20,17] verliert |
| n = 50 gleichverteilt | 50 | 2 | 100 / 99 / 46 % | 15,40 / 0,02 % | -15,38 [-17,55; -12,79] schlägt | +4,60 [+2,13; +7,14] verliert | +6,17 [+3,61; +8,82] verliert |
| n = 50 gleichverteilt | 100 | 2 | 100 / 96 / 30 % | 14,11 / 0,01 % | -14,09 [-16,16; -11,79] schlägt | +5,90 [+2,77; +9,04] verliert | +7,46 [+4,45; +10,69] verliert |
| n = 50 gleichverteilt | 200 | 2 | 100 / 96 / 16 % | 10,80 / 0,09 % | -10,71 [-12,04; -9,28] schlägt | +9,21 [+5,37; +12,90] verliert | +10,77 [+6,97; +14,58] verliert |

**(b) Nachbarliste als Zugmenge** (Zug zugelassen, wenn mindestens eine neue Kante einen Rang unter k hat; Test `test_claims.py::test_befund_e4b_nachbarliste`). Erster Abstieg aus dem Nächsten Nachbarn gegen den vollen Abstieg mit demselben Strom, ohne Kicks; n = 20: 20 Instanzen × 10 Zufalls-Seeds, n = 50: 10 Instanzen × 5 Zufalls-Seeds. „nicht voll“: Anteil der Endtouren, bei denen die volle Prüfrunde noch einen verbessernden Zug findet.

| n | Liste k | Prüfrunde Zufall | Zugauswertungen (voll) | Länge gegen voll | nicht voll | Prüfrunde gelernt | Länge gegen voll | nicht voll |
|---|---|---|---|---|---|---|---|---|
| n = 20 | 3 | 648 von 1 850 | 1 205 (voll 3 470) | +0,14 % | 3,5 % | 643 | +0,18 % | 3,0 % |
| n = 20 | 5 | 1 220 von 1 850 | 2 331 (voll 3 470) | -0,03 % | 0,0 % | 1 217 | -0,03 % | 0,0 % |
| n = 20 | 8 | 1 654 von 1 850 | 2 991 (voll 3 470) | -0,01 % | 0,0 % | 1 658 | -0,02 % | 0,0 % |
| n = 50 | 3 | 3 057 von 12 875 | 8 044 (voll 32 765) | +0,11 % | 18,0 % | 3 024 | +0,21 % | 18,0 % |
| n = 50 | 5 | 6 040 von 12 875 | 15 418 (voll 32 765) | -0,02 % | 0,0 % | 5 989 | +0,13 % | 0,0 % |
| n = 50 | 8 | 8 984 von 12 875 | 23 419 (voll 32 765) | 0,00 % | 0,0 % | 8 978 | +0,40 % | 0,0 % |

Suche mit Kicks innerhalb der Nachbarliste, Budget k = 2 (Differenz in Prozentpunkten): Zufall mit Liste gegen Zufall auf der vollen Zugmenge, dann gelernt gegen Zufall (beide mit Liste) in den drei Sichten:

| n | Liste k | Budget k | Zufall mit Liste gegen voll | gelernt gegen Zufall: nur Zugauswertungen | in K | in O |
|---|---|---|---|---|---|---|
| n = 20 | 3 | 2 | -0,28 [-0,54; -0,08] schlägt | -0,13 [-0,36; +0,08] gleich | +3,33 [+2,18; +4,56] verliert | +16,46 [+12,63; +20,08] verliert |
| n = 20 | 5 | 2 | -0,25 [-0,48; -0,07] schlägt | -0,10 [-0,22; +0,02] gleich | +6,76 [+4,75; +8,88] verliert | +16,42 [+12,60; +20,03] verliert |
| n = 20 | 8 | 2 | -0,11 [-0,26; +0,01] gleich | -0,24 [-0,53; +0,01] gleich | +8,19 [+6,01; +10,44] verliert | +16,29 [+12,44; +19,89] verliert |
| n = 50 | 3 | 2 | -1,10 [-1,61; -0,50] schlägt | -0,57 [-0,86; -0,26] schlägt | +8,21 [+5,46; +10,79] verliert | +20,17 [+15,37; +24,84] verliert |
| n = 50 | 5 | 2 | -0,63 [-1,02; -0,19] schlägt | -0,44 [-0,81; -0,02] schlägt | +12,03 [+8,90; +14,92] verliert | +19,70 [+15,30; +24,12] verliert |
| n = 50 | 8 | 2 | -0,19 [-0,71; +0,35] gleich | -0,81 [-1,60; -0,06] schlägt | +15,51 [+11,68; +19,16] verliert | +19,26 [+14,81; +23,79] verliert |

### Experiment 5: Was kann eine bessere Startlösung höchstens sparen?

Erster Abstieg (Zufallsreihenfolge) aus der Startlösung, 20 Instanzen bei n = 20, 10 bei n = 50, gleichverteilt; Merkmalsnetz K mit Gewichts-Zufalls-Seed 0 (greedy ab Ort 0). Ersparnis: Zugauswertungen bis zum Lokalminimum (mit Prüfrunde) gegen den Nächsten Nachbarn. Test: `test_claims.py::test_befund_e5_startloesungen` (Bänder, neu gerechnet).

| n | Startlösung | Lücke der Startlösung | Zugauswertungen bis zum Lokalminimum (davon Prüfrunde) | Ersparnis | K Konstruktion | K gesamt | O gesamt |
|---|---|---|---|---|---|---|---|
| 20 | Zufallstour | 165,31 % | 3 616 (1 850) | -4 % | 1 | 3 617 | 28 934 |
| 20 | Nächster Nachbar | 16,68 % | 3 469 (1 850) | 0 % | 190 | 3 659 | 29 273 |
| 20 | Farthest Insertion | 2,43 % | 2 548 (1 850) | 27 % | 190 | 2 738 | 21 988 |
| 20 | Merkmalsnetz K (greedy) | 5,82 % | 3 118 (1 850) | 10 % | 190 | 3 308 | 52 302 |
| 20 | schon Lokalminimum | 0,27 % | 1 850 (1 850) | 47 % | 0 | 1 850 | 14 800 |
| 50 | Zufallstour | 361,99 % | 40 486 (12 875) | -21 % | 1 | 40 487 | 323 898 |
| 50 | Nächster Nachbar | 21,57 % | 33 382 (12 875) | 0 % | 1 225 | 34 607 | 276 857 |
| 50 | Farthest Insertion | 5,44 % | 24 406 (12 875) | 27 % | 1 225 | 25 631 | 205 245 |
| 50 | Merkmalsnetz K (greedy) | 18,20 % | 30 061 (12 875) | 10 % | 1 225 | 31 286 | 416 891 |
| 50 | schon Lokalminimum | 2,66 % | 12 875 (12 875) | 61 % | 0 | 12 875 | 103 000 |

Das Netz als Start ist in K nur wenig billiger als der Nächste Nachbar und in O teurer (144 Operationen je Zeile, Merkmalsbildung nicht mitgezählt); Farthest Insertion spart mehr, ohne gelernt zu sein. Das Training des Merkmalsnetzes K (960 000 Touren je Modell) steht nicht in diesen Zahlen.

## Training und Zähleinheiten

* **Zugnetz Z:** 3 000 Schritte × 4 096 Zeilen = 12,3 Mio. Zeilen je vorgerechnetem Modell, Daten aus 300 Trainingsinstanzen (n = 20, Zufalls-Seed ab 2000; 2 311 600 Zeilen aus 11 558 Zuständen). In der App: Voreinstellung 1 500 Schritte (6,1 Mio. Zeilen) aus 100 Instanzen, höchstens 3 000 Schritte aus 300 Instanzen; Zeitwächter 100 s.
* **Merkmalsnetz K:** 3 750 Schritte × 256 Touren (32 Instanzen × 8 Touren) = 960 000 Touren = 182,4 Mio. Zeilen je Modell (Rezept der früheren Vormessung, Lernrate 0,03, Normkappung 5). Greedy-Lücke der fünf Modelle auf 50 Validierungsinstanzen (n = 20): 9,02 / 7,95 / 8,41 / 7,48 / 8,24 % gegen 18,52 % des Nächsten Nachbarn (`test_claims.py::test_befund_modellguete`). Probetraining in der App: 150 Schritte = 38 400 Touren.
* **Cloud-Näherung** (ein Prozess, ein BLAS-Thread, Entwicklungsrechner, nicht die Streamlit Cloud; Zeit ist kein Befund; Messung dieser README, nicht im Test): Zugnetz Voreinstellung 5,5 s (Spitze der numpy-Zuteilungen 96 MB), Obergrenze 13,2 s (280 MB); Probetraining K 8,4 s; Live-Probe n = 20 mit k = 8 0,5 s, n = 50 mit k = 2 1,7 s; Experiment 2 1,6 s. Die Streamlit Cloud ist nicht gemessen.

## Befunde und Korrekturen gegenüber dem Plan

Die Planzahlen stammen aus Vormessungen mit einem anderen Zufallsstrom (`numpy.default_rng`) und teils anderer Zählung; hier mit SplitMix64 neu gemessen. Spalte „Plan“: Messung der Vormessung, nicht im Test.

| Größe | Plan | Neumessung | Bemerkung |
|---|---|---|---|
| AUC Netz / Nachbarlisten-Reihenfolge | 0,973 / 0,876 | 0,978 / 0,879 | gleiche Aussage |
| Zugauswertungen je Zug, n = 20 (Zufall, Nachbarliste, gelernt, deterministisch) | 171,7 / 33,2 / 8,5 / 6,4 | 185,2 / 41,5 / 12,0 / 10,2 | Netz weniger stark als im Plan, Faktor 15 statt 20 |
| dasselbe, n = 50 | 845,9 / 157,6 / 16,7 / 18,3 | 900,8 / 188,5 / 37,9 / 32,4 | Netz weniger stark, Faktor 24 statt 51 |
| Anteil der Prüfrunde, n = 20 / n = 50 | 53 % / 40 % (Planmessung) | 53,7 % / 39,1 % | bestätigt |
| Anteil in der Zählweise der Vormessung | 62 % / 42 % (eigener Nachbau), 64 % / 42 % (frühere Vormessung) | 59,7 % / 34,4 % | n = 50 niedriger |
| Prüfrunde der Nachbarliste k = 3 / 5 / 8, n = 20 | 718 / 1 232 / 1 648 | 648 / 1 220 / 1 654 | |
| dasselbe, n = 50 | 3 183 / 5 862 / 9 007 | 3 057 / 6 040 / 8 984 | |
| Endtouren ohne volles Lokalminimum, k = 3 | 0 von 40 (n = 20), 1 von 20 (n = 50) | 3,5 % (n = 20), 18 % (n = 50) | **bei n = 50 und k = 3 ist die Nachbarliste nicht ohne Qualitätsverlust** |
| Differenz gelernt minus Zufall, nur Zugauswertungen, n = 20, k = 2 / 8 / 32 | +0,33 (Vorteil des Netzes) / nicht besser / nicht besser | -0,27 / -0,01 / 0,00 | gleiche Richtung |
| in K, n = 20, k = 2 / 8 / 32 | −13,92 / −2,17 / −0,31 | +11,31 / +1,49 / +0,12 | Vorzeichen jetzt „gelernt minus Zufall“; gleiche Größenordnung |
| n = 50 gleichverteilt, nur Zugauswertungen: Netz schlägt Zufall | bei k = 8 (+0,72) | bei k = 2 (-0,72), bei k = 8 gleich (-0,32) | **verschoben** |
| Anteil Prüfrunde bei Kicks | 85 bis 99 % | 59 bis 99 % (Regeln mit Zeilen, alle Zellen; Gruppen n = 50 am niedrigsten) | **weiter gefasst** |
| Ersparnis durch das Netz als Start (n = 20 / 50) | 5,3 % / 7,1 % | 10 % / 10 % | andere Zählweise |
| Ersparnis Farthest Insertion | 15,5 % / 12,2 % | 27 % / 27 % | andere Zählweise |
| höchstens zu sparen durch perfekte Startlösung | 36 % / 58 % (Zählweise der früheren Vormessung) | 47 % / 61 % | Einzelortverschiebung einmal gezählt, Zufallsreihenfolge |

Weitere Korrekturen: (a) die Prüfrunde zählt die Einzelortverschiebung nur einmal (1 850 und 12 875 Züge), nicht 2 210 und 15 275 der früheren Vormessung (dort war die Einzelortverschiebung doppelt gezählt); (b) die Vormessung nannte die Sichten „a, b“, hier sind es „nur Zugauswertungen“ (Grenzfall, keine Zählregel), K und O; (c) die „Handregel“ der Vormessung ist eine Reihenfolge (Nachbarlisten-Reihenfolge); die klassische Nachbarliste beschränkt die Zugmenge und ist ein eigener Gegner (Experiment 4b); (d) „Abbruch ist kein Beweis“: der Anteil falscher Lokalminima wird mit einer unabhängigen Gegenprobe gezählt, je Abstieg;
(e) die Zeile der Nachbarlisten-Reihenfolge in O kostet 8 Operationen (Untergrenze), nicht 176 wie im Plan; (f) „neue Kante“ der Nachbarliste: eine Kante, die der Zug zugleich entfernt, zählt nicht als neu; (g) Abbruch bei n = 50 nur mit k = 2 und 5 Zufalls-Seeds (Rechenaufwand).

## Wo die Annahmen enden

| Annahme | Was passiert, wenn sie nicht gilt | Wer setzt an |
|---|---|---|
| Alle Kandidaten werden in jedem Schritt bewertet | Wer nur die durch einen Zug geänderten Kandidaten neu bewertet, spart Zeilen; das ist nicht gemessen und würde das Netz in K und O besser stellen | inkrementelle Neubewertung |
| Die Zugmenge ist 2-opt und Or-opt bis Länge 3 | Lin-Kernighan und Kicks mit Lin-Kernighan sind hier nicht gebaut | `lin-kernighan-demo` |
| Die Nachbarliste ist als „mindestens eine neue Kante mit Rang unter k“ definiert, ohne Don't-Look-Bits | Eine andere Definition oder Don't-Look-Bits verkleinern die Prüfrunde anders; der Aufbau der Listen ist nicht mitgezählt | `hill-climbing-demo` |
| Das Zugnetz wurde bei n = 20 trainiert | Bei n = 50 gilt es nur als Null-Schuss (ohne neues Training); andere Familien und n ab 100 sind nicht gemessen | `netz-fremde-instanzen-demo` der Linie |
| K und O beruhen auf Annahmen (8 Operationen je Zugauswertung, 176 je Zeile) | Die Wahrheit liegt zwischen K und O; nur Zehnerpotenzen sind belastbar | Zählregel der Linie |
| Das Netz lernt nur die Reihenfolge, überwacht (verbessert dieser Zug jetzt?) | Lernen über mehrere Schritte ist nicht gemessen | Reinforcement Learning in der Suche |
| Gemessen ist die Zugwahl aus dem Zustand der Tour | Die Zugart nach Suchphase oder Stagnation (Reserve R der Hyperheuristiken) ist nicht gemessen | `hyperheuristik-demo` |
| Eigene Umsetzung, einfachste lokale Suche, erzeugte Rundtouren | Keine Aussage über Praxis oder starke Löser (Lin-Kernighan-Löser und exakte Löser sind nicht im Bestand) | Experimente an realen Daten |

## Ehrliche Grenzen

* **Eigene Umsetzung, einfachste lokale Suche.** 2-opt und Or-opt bis Länge 3 mit erster Verbesserung und Double-Bridge-Kicks, **ohne** Nachbarlisten als Standard und **ohne** Don't-Look-Bits; die Nachbarliste ist nur in Experiment 4 und nur in einer Definition gemessen. Keine Aussage über Lin-Kernighan-Löser, LKH oder Concorde (nicht im Bestand).
* **Erzeugte Rundtouren, kleine n.** Gemessen sind n = 20 gleichverteilt und n = 50 (gleichverteilt, Gruppen) mit je 20 bzw. 10 Instanzen; `gitter` und n ab 100 sind nicht gemessen. Das Zugnetz Z ist nur bei n = 20 trainiert; bei n = 50 gilt es als Null-Schuss (ohne neues Training).
* **Handgeschriebenes Netz mit 177 Parametern.** Kein GPU-Training, kein Sprachmodell; die Aussage gilt für dieses Netz, diese Merkmale und dieses Training (überwacht: „verbessert dieser Zug jetzt?“), nicht für Netze im Allgemeinen. Gelernt wird nur die Reihenfolge, nicht der Verlauf über mehrere Schritte.
* **Das Netz bewertet in jedem Schritt alle Kandidaten.** Wer nur die vom letzten Zug geänderten Kandidaten neu bewertet, spart Zeilen; das ist nicht gemessen und würde das Netz in K und O besser stellen.
* **K und O beruhen auf Annahmen** (8 Operationen je Zugauswertung, 176 je Zeile des Zugnetzes, 8 je Zeile der Nachbarlisten-Reihenfolge als Untergrenze; Merkmalsbildung nicht mitgezählt). K begünstigt das Netz, O benachteiligt es; „das Netz verliert“ gilt als belastbar, wenn es schon in K verliert. Zeit ist nie ein Befund.
* **Abbruch ist kein Beweis.** Das Wort „Beweis des Lokalminimums“ gilt nur für die volle Prüfrunde; mit Abbruch endet die Suche in manchen Abstiegen an falschen Lokalminima, die Demo zählt sie mit einer unabhängigen Gegenprobe.
* **Reserve R der Hyperheuristiken nur teilweise gefüllt:** gemessen ist die Zugwahl aus dem Zustand der Tour (Experiment 1, Zugarten), nicht die Zugart nach Suchphase oder Stagnation.
* **Stichproben sind klein.** 20 bzw. 10 Instanzen je Zelle; Marken stützen sich auf gepaarte Differenzen mit Bootstrap-Intervall über die Instanzen, eine Zelle mit 10 Instanzen kann eine Marke nur grob belegen. Die Nachbarliste bei n = 50 ruht auf 50 Abstiegen je Regel.
* **Streamlit Cloud nicht gemessen; Linux-Check steht aus.** Zeiten und Speicher stammen vom Entwicklungsrechner (ein BLAS-Thread). Trainingsläufe sind nur als Band wiederholbar: Normalzahlen für die Anfangsgewichte nutzen `numpy.log` und `numpy.cos`, Windows und Linux rechnen im letzten Bit verschieden; die Tests prüfen Bänder, nie Bitgleichheit der Trainings.

## Verifikation

* **Kern:** Zugmenge (M(n) gegen Aufzählung für n = 5 bis 12), Delta gegen volle Neuberechnung (Listen, Zufallstouren, n = 6 bis 50), Lokalminimum auf zwei Wegen, Prüfrunde = M, Held-Karp gegen Aufzählung und CP-SAT, Handbeispiel von Hand (`test_moves.py`, `test_search.py`, `test_handbeispiele.py`, `test_oracle_zugwahl.py`).
* **Netze:** Rückwärtspass von Hand (Mini-Netz 2-2-1) und gegen zentrale Differenzen in voller Größe, Politikgradient mit Maske und Teacher-Forcing, AUC gegen Rangsumme, Gumbel-Reihenfolge gegen Plackett-Luce (`test_mlp.py`, `test_policy.py`, `test_zugnetz.py`).
* **Kopien:** Kopier-Tests gegen `zeigernetz-demo` und `hyperheuristik-demo` (`test_copies.py`, `test_rng.py`), übersprungen, wenn die Quelle fehlt.
* **Messreihen:** Form, Monotonie, Ausschnitt als Band neu gerechnet (`test_frozen.py`); jede Zahl dieser README in `test_claims.py`.
* **Oberfläche:** AppTest-Rauchtest (jedes Preset, jeder Regler an den Grenzen, Trainingsknöpfe, Live-Proben), Layout (Schnellstart oben, fester Kopf, aktives Preset hervorgehoben; jede Änderung eines Reglers hebt die Markierung auf), Footer, Laufzeit-Importe.
* **Tests:** 330 Tests in 22 Dateien (ein Prozess, ohne xdist).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Kopf, Primäransicht, fünf Experimente, Grenzen, Formeln) |
| `zwl_moves.py` | Zugmenge (2-opt, Or-opt), Delta, Merkmale, Anwenden, Kick; **Quelle für die Folge-Demos** |
| `zwl_search.py` | Reihenfolgen, Abstieg, Suche mit Kicks, Abbruch, Nachbarliste; **Quelle für die Folge-Demos** |
| `zwl_mlp.py`, `zwl_featnet.py`, `zwl_policy.py` | Netz von Hand, Merkmalsnetz K, Politikgradient; **Quelle für die Folge-Demos** |
| `zwl_zugnetz.py` | Zugnetz Z: Datensammlung, Training, AUC (nur hier) |
| `zwl_construct.py`, `zwl_cost.py` | Startlösungen, Zählung K und O |
| `zwl_experiments.py` | Experimente live und in der Messung |
| `zwl_evaluation.py`, `zwl_stats.py` | Verdikte als reine Funktionen, Tabellen, Statistik |
| `zwl_texts.py`, `zwl_presets.py`, `zwl_layout.py`, `zwl_visualization.py`, `zwl_constants.py` | Texte, Presets, Kopf, Figuren, Konstanten |
| `zwl_hand.py` | Handbeispiel (sechs Orte) |
| `zwl_rng.py`, `zwl_rngblock.py`, `zwl_rngvec.py`, `zwl_scenario.py`, `zwl_instances.py`, `zwl_exact.py`, `modul_frisch.py` | Kopien (Zufall, Instanzen, Held-Karp) |
| `zwl_frozen.json`, `zwl_refs.json` | eingefrorene Messreihen, Referenzlängen |
| `zwl_modelle/` | vorgerechnete Modelle `zugnetz_s0..4.npz`, `featnet_s0..4.npz` (Format `zwl-mlp-1`) |
| `tools/zwl_make_models.py`, `tools/zwl_make_refs.py`, `tools/zwl_make_frozen.py` | Modelle, Referenzen, Messreihen erzeugen |
| `tools/zwl_planzahlen.py`, `tools/zwl_zahlen.py`, `tools/zwl_kalibrierung.py` | Planzahlen neu messen, Zahlen drucken, Zeit und Speicher |
| `tests/` | Tests (siehe Verifikation), `tests/claims_zeilen.py` baut die Tabellenzeilen dieser README |
| `.github/workflows/tests.yml`, `requirements.txt`, `requirements-dev.txt` | CI, Laufzeit (streamlit, numpy, plotly), Gegenproben (pytest, scipy, ortools) |

## Bewusst nicht enthalten

* Zugart nach Suchphase oder Stagnation (Reserve R in der Fassung der Hyperheuristiken), Lernen über mehrere Schritte (Reinforcement Learning in der Suche).
* Lin-Kernighan und Kicks mit Lin-Kernighan, Don't-Look-Bits (siehe `hill-climbing-demo`), inkrementelle Neubewertung der Kandidaten.
* n ab 100, die Familie `gitter` in den eingefrorenen Tabellen, andere Netzgrößen, GPU, Sprachmodell, Praxisnutzen.
* Mehrfachstart und Gesamtbudget mit dem Merkmalsnetz K als Start (das misst `rechnen-beim-einsatz-demo` der Linie); hier nur der Einzelstart und die Obergrenze der Ersparnis.
* Eine Zeitmessung als Befund, eine Messung auf der Streamlit Cloud.

## Quellen (mit Belegstufe)

* Wu, Song, Cao, Zhang, Lim 2021 (gelernte Wahl des nächsten 2-opt-Zugs; Belegstufe V, Ort S): bei n = 20 und n = 50 gleichauf mit „beste Verbesserung mit Neustart“, erst bei n = 100 vorn; Zähleinheit dort Verbesserungsschritte, nicht Zugauswertungen, und ein anderes Netz. Das stützt den Befund „kleines n: kein Vorteil in der Endqualität“, ist aber kein Nachweis für diese Demo.
* Mazyavkina u. a. 2021 (Belegstufe V): Einteilung in Konstruktions- und Verbesserungsverfahren.
* Chen und Tian 2019, NeuRewriter (Belegstufe V): Regionswahl und Regelwahl, aber für Tourenplanung mit Kapazität, nicht für die Rundtour.
* Survey arXiv 2602.21761, Abschnitt VI-C (Belegstufe V): Verbesserungsnetze mit kleiner Nachbarschaft hätten hohen Rechenaufwand beim Paarwählen und begrenzte Wirkung; das passt zum Befund „alle Kandidaten je Schritt bewerten kostet“, ist aber eine Aussage der Survey, hier nicht nachgemessen.
* da Costa u. a. 2020: nicht gelesen, nicht zitiert.

## Verweise

Live-Demos: [`hill-climbing-demo`](https://sebastianhanisch-hill-climbing-demo.streamlit.app/) (Kandidatenlisten und Don't-Look-Bits für 2-opt, hier nicht nachgebaut), [`iterated-local-search-demo`](https://sebastianhanisch-iterated-local-search-demo.streamlit.app/) (Kicks),
[`lin-kernighan-demo`](https://sebastianhanisch-lin-kernighan-demo.streamlit.app/) (stärkerer Gegner, hier nicht gebaut), [`akzeptanz-mal-auswahl-demo`](https://sebastianhanisch-akzeptanz-mal-auswahl-demo.streamlit.app/) (gelernte Auswahl der Zugart gegen Zufall),
[`policy-gradient-demo`](https://sebastianhanisch-policy-gradient-demo.streamlit.app/) (REINFORCE-Idee für das Merkmalsnetz K), [`actor-critic-demo`](https://sebastianhanisch-actor-critic-demo.streamlit.app/),
[`einstellen-lohnt-demo`](https://sebastianhanisch-einstellen-lohnt-demo.streamlit.app/) (abgestimmte Gegner), [`vergleich-kippt-demo`](https://sebastianhanisch-vergleich-kippt-demo.streamlit.app/) (Rangfolge kippt mit der Zähleinheit).
Nur mit Namen (kein Live-Link): `hyperheuristik-demo`, `auswahlregeln-demo`, `gelernter-verzweiger-umstauen-demo`, `monte-carlo-gradient-demo` sowie die weiteren Demos dieser Linie (`zeigernetz-demo`, `basislinie-lernen-demo`, `rechnen-beim-einsatz-demo`, `netz-fremde-instanzen-demo`, `netz-gegen-gegner-demo`).
Das Merkmalsnetz K (`zwl_featnet.py`, `zwl_policy.py`, `zwl_modelle/featnet_s0..4.npz`) ist die Quelle für `rechnen-beim-einsatz-demo` und `netz-fremde-instanzen-demo`.

## Lokal ausführen

```
python -m pip install -r requirements.txt
streamlit run app.py
```

Tests: `python -m pip install -r requirements-dev.txt`, dann `python -m pytest tests -q`.
Messreihen neu erzeugen (je ein Prozess, `OMP_NUM_THREADS=1`): `python tools/zwl_make_models.py Z`, `python tools/zwl_make_models.py K`, `python tools/zwl_make_refs.py` und `python tools/zwl_make_refs.py n50`, `python tools/zwl_make_frozen.py e3`, `e4a`, `e4b`, `alles`.

Gebaut mit Streamlit, Plotly und numpy.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Neuronale kombinatorische Optimierung](https://sebastianhanisch.net/konzepte-neuronale-kombinatorische-optimierung.html).
