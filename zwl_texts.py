"""Alle Anzeigetexte der App (ein Ort, damit README und App dieselben Sätze führen). Verdikte stehen in zwl_evaluation.py, weil sie Zahlen einsetzen."""

TITEL = "🧭 Gelernte Zugwahl: wo steckt der Aufwand?"

INTRO = (
    "Diese Demo gehört zur Linie **Neuronale kombinatorische Optimierung erklärt** und stellt eine Frage, die die Konstruktion aus dem ersten Stück offen lässt: "
    "Was kann ein kleines Netz in der **Verbesserung** leisten? Ein Netz ordnet hier die möglichen **2-opt- und Or-opt-Züge** einer Rundtour, und die Suche probiert sie in dieser Reihenfolge. "
    "Die Demo zählt, **wo der Aufwand steckt**, und stellt dem Netz die **Nachbarliste** gegenüber, damit nicht dem Lernen gutgeschrieben wird, was eine alte Idee der Lokalsuche schon leistet. "
    "Die Expander am Ende nennen die Annahmen und die Formeln."
)

EINFACHE_WORTE = (
    "Eine Rundtour wird verbessert, indem man kleine Änderungen probiert: zwei Strecken tauschen, einen Ort an eine andere Stelle setzen. Fast keine Änderung hilft. "
    "Ein Netz kann lernen, welche Änderungen sich zuerst lohnen, so wie ein erfahrener Planer die aussichtsreichen zuerst ansieht. Das spart Versuche, bis ein Zug gefunden ist. "
    "Aber wenn keine Änderung mehr hilft, muss die Suche **alle** möglichen Änderungen einmal ansehen, um sicher zu sein. Dafür hilft keine Reihenfolge. "
    "Außerdem muss das Netz in jedem Schritt alle Kandidaten bewerten, und auch das kostet. Die Demo zählt beides getrennt und lässt dem Netz die Wahl, ob seine Arbeit mitgezählt wird oder nicht."
)

SO_FUNKTIONIERT = """
**1. Die Tour und ihre Züge.** Eine Rundtour besucht alle Orte einmal. Ein **Zug** ändert sie ein wenig: **2-opt** ersetzt zwei Kanten durch zwei andere (ein Stück der Tour wird umgekehrt),
**Or-opt** nimmt ein Stück von einem, zwei oder drei Orten heraus und setzt es an anderer Stelle wieder ein, bei mehr als einem Ort auch umgekehrt. Das **Delta** eines Zugs ist die neue minus die alte Länge;
negativ heißt: der Zug verkürzt die Tour. Die Zugmenge hat bei n Orten n (11 n − 35) / 2 Züge (M), gezählt als Zugbeschreibungen (die Verschiebung eines einzelnen Ortes nur einmal); einzelne Züge liefern dieselbe Nachbartour, bei n = 20 sind es 1 570 verschiedene Nachbartouren.

**2. Die Suche.** Die Suche beginnt mit dem Nächsten Nachbarn (Nearest Neighbor) und probiert die Züge in einer **Reihenfolge**. Sie nimmt den ersten Zug, der verbessert (erste Verbesserung, First Improvement),
und beginnt von vorn. Findet sie keinen mehr, hat sie alle M Züge bewertet: die **Prüfrunde**. Sie ist der Beweis, dass die Tour ein Lokalminimum der vollen Zugmenge ist (mit Abbruch oder Nachbarliste gilt das nicht, siehe Experiment 4), und sie hängt nicht von der Reihenfolge ab.

**3. Wer die Reihenfolge macht.** Zufall; die **Nachbarlisten-Reihenfolge** (Züge mit kurzen neuen Kanten zuerst, eine feste Handregel); das **Zugnetz** Z (neun Merkmale je Zug, ein Netz mit 177 Parametern, gelernt
aus Zuständen früherer Suchen: verbessert dieser Zug jetzt?); der **steilste Abstieg** (alle M Züge bewerten und den besten nehmen).

**4. Was gezählt wird.** Eine **Zugauswertung** ist ein bewerteter Kandidat. Das Netz muss in jedem Schritt alle Kandidaten bewerten (**Merkmalszeilen**), auch die, die die Suche nie probiert. Experiment 3 zählt in drei Sichten:
Merkmalszeilen gratis (Grenzfall), als Kandidatenbewertung (K) und als grobe Operationen (O).

**5. Der Gegner.** Die **Nachbarliste** (Neighbor List) beschränkt die Zugmenge statt sie nur zu ordnen: Nur Züge mit mindestens einer kurzen neuen Kante werden überhaupt geprüft. Der **Abbruch** der Prüfung nach g erfolglosen Kandidaten
spart ebenfalls, ist aber kein Beweis mehr.
"""

ZAEHLREGEL = """
**K (Kandidatenbewertungen).** Eine Zugauswertung zählt 1, eine Merkmalszeile des Netzes zählt 1, ein Kick zählt 1, die Konstruktion (Nächster Nachbar, Farthest Insertion, Merkmalsnetz K) zählt n(n−1)/2, eine Zufallstour 1.

**O (grobe Operationen).** Annahme: 8 Operationen je Zugauswertung (Delta aus vier Kanten) und je Kandidatenbewertung der klassischen Verfahren; 176 je Merkmalszeile des Zugnetzes (9 · 16 + 16 + 16),
144 je Zeile des Merkmalsnetzes K (7 · 16 + 16 + 16); die Nachbarlisten-Reihenfolge rechnen wir mit 8 je Zeile (Annahme, Untergrenze). Die Bildung der Merkmale ist nicht mitgezählt, O ist für das Netz also eine Untergrenze.
Nur die Zehnerpotenz ist belastbar.

**Grenzfall „Zeilen gratis“.** Die Merkmalszeilen kosten nichts; das ist keine Zählregel, es zeigt, was das Netz höchstens leisten könnte.

**Leserichtung.** K begünstigt das Netz (eine Merkmalszeile ist in Operationen viel teurer als eine Zugauswertung), O benachteiligt es (die Operationen des Netzes sind gezählt, die der Merkmalsbildung nicht).
„Das Netz verliert“ gilt als belastbar, wenn es schon in K verliert. **Training** ist ein dritter Posten (Schritte × Zeilen bzw. Touren). **Zeit** ist nie ein Befund.
"""

ERKLAERTABELLE = """
| Name | Was es ist | Hier |
|---|---|---|
| 2-opt | ersetzt zwei Kanten einer Tour durch zwei andere | Zugart, n(n−3)/2 Züge |
| Or-opt | verschiebt ein Stück von 1 bis 3 Orten | Zugart, 3 Längen, bei Stücken mit mehr als einem Ort auch umgekehrt |
| Erste Verbesserung (First Improvement) | der erste verbessernde Kandidat wird genommen | Regel der Suche |
| Prüfrunde | alle M Züge ohne Verbesserung bewertet | Beweis des Lokalminimums |
| Nachbarliste (Neighbor List) | je Ort nur die nächsten k Nachbarn zulassen | beschränkt die Zugmenge (Experiment 4) |
| Nachbarlisten-Reihenfolge | Züge mit kurzen neuen Kanten zuerst probieren | ordnet nur, bezahlt M Zeilen je Schritt |
| Kick (Double-Bridge) | die Tour an drei Stellen schneiden und umordnen | Iterated Local Search (ILS) |
| Gumbel-Ziehung | Ziehen ohne Zurücklegen aus dem Softmax der Logits | gelernte Reihenfolge (gezogen) |
| AUC | Wahrscheinlichkeit, dass ein verbessernder Zug vor einem nicht verbessernden steht (hier über alle Zeilen aller Validierungszustände gemeinsam gerechnet, nicht je Zustand) | Güte der Reihenfolge |
| REINFORCE | Gradient aus gezogenen Touren, gewichtet mit dem Vorteil | Training des Merkmalsnetzes K |
| Farthest Insertion | den vom Tourteil am weitesten entfernten Ort einfügen, wo es am wenigsten kostet | Startlösung (Experiment 5) |
"""

KERNFRAGEN = [
    "Ein Abstieg Schritt für Schritt",
    "Ordnet das Netz die Züge besser als eine einfache Regel?",
    "Wie viel der Arbeit beweist nur, dass es nicht weitergeht?",
    "Wer gewinnt bei gleichem Budget?",
    "Lässt sich der Beweis abkürzen, und was kostet das?",
    "Was kann eine bessere Startlösung höchstens sparen?",
]

ASSUMPTIONS_MARKDOWN = """
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
"""

LESEHILFE = """
**Gegeben** sind Orte mit Entfernungen $d_{ij}$ und eine Rundtour $t$. **Gesucht** ist ein Zug aus der Menge $\\mathcal{M}(t)$, der die Tour verkürzt, und die Zahl der Bewertungen, die das kostet.
Die Formeln sind ein vollständiges Modell der Demo (Zugmenge, Delta, Zugnetz, Ordnung, Zählregel); was nicht dazugehört, steht unter „Wo die Annahmen enden“.

| Symbol | Bedeutung |
|---|---|
| $n$, $d_{ij}$, $t$, $L(t)$ | Zahl der Orte, Entfernung, Rundtour, Tourlänge |
| $\\mathcal{M}(t)$, $M(n)$ | Zugmenge, ihre Größe (Zugbeschreibungen) |
| $\\Delta$ | Längenänderung eines Zugs (negativ = Verbesserung) |
| $s$ | mittlerer Abstand eines Ortes zum nächsten Nachbarn der Instanz |
| $x, z, \\sigma$ | Merkmalsvektor, Netzausgabe (Logit), Sigmoid |
| $\\rho_{uv}$ | Rang der Kante $(u,v)$ unter den Nachbarn (0 = nächster), kleinerer der beiden Richtungen |
| $k$ | Grenze der Nachbarliste (Experiment 4) bzw. Budgetfaktor (Experiment 3); in der Oberfläche getrennt benannt |
| $E, R, K, O$ | Zugauswertungen, Merkmalszeilen, Kandidatenbewertungen, grobe Operationen |
"""

FORMELN = r"""
Tourlänge und Delta eines 2-opt-Zugs (Positionen $i<j$, $j-i\ge 2$, nicht $(0,n-1)$; die Kanten $(t_i,t_{i+1})$ und $(t_j,t_{j+1})$ werden durch $(t_i,t_j)$ und $(t_{i+1},t_{j+1})$ ersetzt):

$$L(t)=\sum_{k=0}^{n-1} d_{t_k,t_{k+1 \bmod n}}$$

$$\Delta_{\text{2opt}}(i,j)=d_{t_i t_j}+d_{t_{i+1}t_{j+1}}-d_{t_i t_{i+1}}-d_{t_j t_{j+1}}$$

Or-opt (Segment $s_0..s_\ell$ der Länge $\ell\in\{1,2,3\}$ mit Vorgänger $p$ und Nachfolger $q$ wird zwischen die Nachbarn $u,v$ eingesetzt, bei $\ell\ge 2$ in beiden Richtungen):

$$\Delta_{\text{Or}}=d_{pq}-d_{ps_0}-d_{s_\ell q}-d_{uv}+\begin{cases} d_{us_0}+d_{s_\ell v} & \text{vorwärts}\\ d_{us_\ell}+d_{s_0v} & \text{rückwärts}\end{cases}$$

Zahl der Züge (verschiedene Zugauswertungen einer Prüfrunde):

$$M(n)=\tfrac{n(n-3)}{2}+n(n-2)+2n(n-3)+2n(n-4)=\tfrac{n(11n-35)}{2}$$

Die Zählweise der früheren Vormessung zählt die Verschiebung eines Einzelorts doppelt, $M_{\text{alt}}(n)=M(n)+n(n-2)$ (Schalter in Experiment 2).

Zugnetz Z (Merkmale $x\in\mathbb{R}^9$: mittlere und größte Länge der entfernten Kanten geteilt durch $s$ und bei 6 gekappt; kleinster und größter Rang der neuen Kanten, bei 8 gekappt, geteilt durch 8; Kennzeichen 2-opt, Or-opt $\ell=1,2,3$; Umkehr):

$$z(x)=w_2^{\top}\tanh(W_1x+b_1)+b_2,\qquad W_1\in\mathbb{R}^{16\times 9}$$

Verlust (logistisch, Etikett $y=1$ genau wenn $\Delta<0$), Mittel über $N$ Zeilen, und Gradient nach dem Logit:

$$\mathcal{L}=\frac1N\sum_{r}\bigl[\operatorname{softplus}(z_r)-y_rz_r\bigr]$$

$$\frac{\partial\mathcal{L}}{\partial z_r}=\frac{\sigma(z_r)-y_r}{N}$$

Ordnung der Kandidaten: gezogen nach Gumbel (Ziehen ohne Zurücklegen aus dem Softmax) oder absteigende Logits; die Nachbarlisten-Reihenfolge nimmt einen festen Schlüssel:

$$\text{Rang}\;=\;\operatorname{argsort}_{\downarrow}\bigl(z_m+g_m\bigr),\quad g_m=-\ln(-\ln u_m),\; u_m\sim U(0,1)$$

$$\text{Schlüssel}_m=\min(\rho_m^{\max},8)\cdot 100-\min(\bar e_m/s,\,6)$$

($\rho_m^{\max}$ größter Rang der neuen Kanten, $\bar e_m$ mittlere Länge der entfernten Kanten; kleiner Schlüssel zuerst, Gleichstand zufällig). Die Nachbarliste mit Grenze $k$ ist die Zugmenge

$$\mathcal{M}_k(t)=\{m:\ \min_{e\in\text{neu}(m)}\rho_e<k\},\qquad \rho_{uv}=\min(\text{Rang}_u(v),\text{Rang}_v(u))$$

Güte der Ordnung (AUC, Paarzählung mit halber Gewichtung bei Gleichstand):

$$\mathrm{AUC}=\frac{1}{|P||N|}\sum_{a\in P}\sum_{b\in N}\Bigl(\mathbf{1}[z_a>z_b]+\tfrac12\mathbf{1}[z_a=z_b]\Bigr)$$

Merkmalsnetz K (Konstruktion, Experiment 5): Politik über die noch nicht besuchten Orte $j$ vom aktuellen Ort $i$, sieben Merkmale $x_{ij}$, $z_{ij}=w_2^\top\tanh(W_1x_{ij}+b_1)+b_2$ mit $W_1\in\mathbb{R}^{16\times 7}$, 145 Parameter:

$$\pi(j\mid i,U)=\frac{e^{z_{ij}}}{\sum_{l\in U}e^{z_{il}}}$$

$$\nabla_\theta J=-\mathbb{E}\Bigl[(b-L(\tau))\,\nabla_\theta\sum_{k}\ln\pi(\tau_{k+1}\mid\tau_k,U_k)\Bigr]$$

Der Referenzwert $b$ ist das Mittel der Touren der 8 Rollouts (verschiedene Startorte) derselben Instanz, der Vorteil ist $(b-L)/b$; der Politikverlust $-\sum_r c_r\sum_k\ln\pi$ mit $c_r=(b_r-L_r)/b_r$ sinkt, wenn kürzere Touren wahrscheinlicher werden.

Zählregel: Konstruktion nie 1 (Zufallstour und Kick ausgenommen):

$$K=c_{\text{Konstr}}+E+R,\qquad c_{\text{Konstr}}=\tfrac{n(n-1)}{2}$$

$$O=8\,c_{\text{Konstr}}+8E+176R\qquad(\text{Nächster Nachbar, Zugnetz Z}),\qquad E=\text{Zugauswertungen},\ R=\text{Merkmalszeilen}$$

Nicht als Formel stehen im Text: die Akzeptanzregel der Kicks (angenommen wird bei Länge höchstens der besten bisherigen), der Kick (Double-Bridge mit drei verschiedenen gleichverteilten Schnittstellen) und Adam.
Don't-Look-Bits und Lin-Kernighan fehlen bewusst.
"""
