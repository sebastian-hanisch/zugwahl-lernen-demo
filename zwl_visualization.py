"""Plotly-Figuren: Karte der Tour mit Zug, Balken der Zugauswertungen, AUC, Zugauswertungen je Zug, Zugarten, Lernkurve, Bestwert über Kosten, Startlösungen.
Alle Achsen fest (`fixedrange`), keine Auswahl, Beschriftungen als Annotationen (DEMO-PLAYBOOK §3)."""

import numpy as np
import plotly.graph_objects as go

import zwl_constants as C

HOEHE = 380


def _fest(fig: go.Figure, titel: str | None = None, hoehe: int = HOEHE, legende: bool = True) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(title=dict(text=titel, font=dict(size=14)) if titel else None, height=hoehe, margin=dict(l=50, r=20, t=50 if titel else 20, b=50), showlegend=legende,
                      legend=dict(orientation="h", y=-0.2), dragmode=False)
    return fig


def _kanten_spur(punkte, paare, farbe, breite, name, gestrichelt=False):
    xs, ys = [], []
    for a, b in paare:
        xs += [punkte[a][0], punkte[b][0], None]
        ys += [punkte[a][1], punkte[b][1], None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=farbe, width=breite, dash="dash" if gestrichelt else "solid"), name=name, hoverinfo="skip")


def karte(punkte, tour, entfernt=(), neu=(), start=None, titel: str | None = None) -> go.Figure:
    """Orte mit Tour (blau), Startlösung (grau gestrichelt), entfernte Kanten des Zugs (rot) und neue (grün). `punkte`: Liste von (x, y)."""
    n = len(punkte)
    paare = lambda t: [(int(t[i]), int(t[(i + 1) % len(t)])) for i in range(len(t))]  # noqa: E731
    fig = go.Figure()
    if start is not None:
        fig.add_trace(_kanten_spur(punkte, paare(start), C.COLOR_GRAU, 2, "Startlösung", gestrichelt=True))
    fig.add_trace(_kanten_spur(punkte, paare(tour), C.COLOR_ZUG, 3, "Tour"))
    if len(entfernt):
        fig.add_trace(_kanten_spur(punkte, list(entfernt), C.COLOR_BEWEIS, 5, "entfernte Kanten"))
    if len(neu):
        fig.add_trace(_kanten_spur(punkte, list(neu), "#54a24b", 5, "neue Kanten"))
    fig.add_trace(go.Scatter(x=[p[0] for p in punkte], y=[p[1] for p in punkte], mode="markers", marker=dict(size=9, color="#333333"), name="Orte", hoverinfo="skip",
                             selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
    for i in range(n):
        fig.add_annotation(x=punkte[i][0], y=punkte[i][1], text=str(i), showarrow=False, yshift=12, font=dict(size=11), bgcolor="rgba(255,255,255,0.8)")
    xs, ys = [p[0] for p in punkte], [p[1] for p in punkte]
    dx, dy = (max(xs) - min(xs)) * 0.06 + 1e-9, (max(ys) - min(ys)) * 0.06 + 1e-9
    fig.update_xaxes(range=[min(xs) - dx, max(xs) + dx], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[min(ys) - dy, max(ys) + dy], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1)
    return _fest(fig, titel, 420)


def balken_aufwand(ev_zuege: float, ev_pruef: float, zeilen: float = 0.0) -> go.Figure:
    """Zugauswertungen: für angenommene Züge und für den Beweis (ein Balken), darunter die Merkmalszeilen."""
    fig = go.Figure()
    fig.add_trace(go.Bar(y=["Zugauswertungen"], x=[ev_zuege], orientation="h", name="bis zu den angenommenen Zügen", marker_color=C.COLOR_ZUG))
    fig.add_trace(go.Bar(y=["Zugauswertungen"], x=[ev_pruef], orientation="h", name="für den Beweis (Prüfrunde)", marker_color=C.COLOR_BEWEIS))
    if zeilen:
        fig.add_trace(go.Bar(y=["Merkmalszeilen"], x=[zeilen], orientation="h", name="Merkmalszeilen", marker_color=C.COLOR_GELERNT))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Anzahl")
    fig = _fest(fig, None, 300)
    fig.update_layout(legend=dict(orientation="h", y=-0.55), margin=dict(l=50, r=20, t=20, b=90))
    return fig


def auc_balken(tab: dict) -> go.Figure:
    """AUC des Netzes, der Nachbarlisten-Reihenfolge und der vier Einzelmerkmale."""
    namen = ["Netz Z", "Nachbarlisten-Reihenfolge", "mittlere entfernte Länge", "größte entfernte Länge", "kleinster Rang", "größter Rang"]
    werte = [tab["netz"], tab["hand"], tab["rem_mean"], tab["rem_max"], tab["minrank"], tab["maxrank"]]
    farben = [C.COLOR_GELERNT, C.COLOR_LISTE] + [C.COLOR_ZUFALL] * 4
    fig = go.Figure(go.Bar(x=namen, y=werte, marker_color=farben, text=[f"{w:.3f}".replace(".", ",") for w in werte], textposition="outside", showlegend=False))
    fig.update_yaxes(range=[0.5, 1.02], title="AUC")
    return _fest(fig, "Güte der Reihenfolge (AUC, höher ist besser)", legende=False)


def je_zug_balken(daten: dict) -> go.Figure:
    """Zugauswertungen je angenommenem Zug je Regel (logarithmische Achse)."""
    regeln = [r for r in C.REGELN_E3 if r in daten]
    fig = go.Figure(go.Bar(x=[C.REGEL_KURZ[r] for r in regeln], y=[daten[r]["je_zug"] for r in regeln], marker_color=[C.REGEL_FARBE[r] for r in regeln],
                           text=[f"{daten[r]['je_zug']:.1f}".replace(".", ",") for r in regeln], textposition="outside", showlegend=False))
    fig.update_yaxes(type="log", title="Zugauswertungen je angenommenem Zug")
    return _fest(fig, "Zugauswertungen je angenommenem Zug (erster Abstieg)", legende=False)


def zugarten_balken(z: dict) -> go.Figure:
    """Anteil der Zugarten unter den bestbewerteten Kandidaten gegen ihren Anteil an der Zugmenge."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=z["zugarten"], y=[100 * a for a in z["anteil_top"]], name=f"unter den {z['top']} bestbewerteten", marker_color=C.COLOR_GELERNT))
    fig.add_trace(go.Bar(x=z["zugarten"], y=[100 * a for a in z["anteil_m"]], name="Anteil an der Zugmenge M", marker_color=C.COLOR_ZUFALL))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Anteil in %")
    return _fest(fig, "Welche Zugart bevorzugt das Netz?")


def lernkurve(verlauf, titel: str = "Verlust beim Training", achse: str = "Verlust") -> go.Figure:
    """Verlauf [(Schritt, Wert)]."""
    fig = go.Figure(go.Scatter(x=[s for s, _ in verlauf], y=[v for _, v in verlauf], mode="lines+markers", line=dict(color=C.COLOR_GELERNT), showlegend=False))
    fig.update_xaxes(title="Trainingsschritte")
    fig.update_yaxes(title=achse)
    return _fest(fig, titel, 300, legende=False)


def beweis_anteil_balken(zeilen: list) -> go.Figure:
    """Anteil der Prüfrunde je n (Experiment 2)."""
    fig = go.Figure(go.Bar(x=[f"n = {z['n']}" for z in zeilen], y=[100 * z["anteil"] for z in zeilen], marker_color=C.COLOR_BEWEIS, showlegend=False,
                           text=[f"{100 * z['anteil']:.0f} %" for z in zeilen], textposition="outside"))
    fig.update_yaxes(range=[0, 105], title="Anteil der Prüfrunde in %")
    return _fest(fig, "Anteil der Prüfrunde an den Zugauswertungen eines Abstiegs", legende=False)


def kurve_bestwert(raster, kurven: dict, achse_x: str, titel: str) -> go.Figure:
    """Bester Wert über Kosten: `kurven[regel]` = Werte am Raster."""
    fig = go.Figure()
    for regel, y in kurven.items():
        fig.add_trace(go.Scatter(x=list(raster), y=list(y), mode="lines", name=C.REGEL_KURZ[regel], line=dict(color=C.REGEL_FARBE[regel], width=2)))
    fig.update_xaxes(title=achse_x, type="log")
    fig.update_yaxes(title="beste Tourlänge (Mittel der Zufalls-Seeds)")
    return _fest(fig, titel)


def luecken_balken(zeilen: list, titel: str) -> go.Figure:
    """Lücke in % je Regel und Sicht (`zeilen`: {regel, frei, K, O})."""
    fig = go.Figure()
    for s, farbe in zip(C.SICHTEN, ("#8c8c8c", "#4c78a8", "#e45756")):
        fig.add_trace(go.Bar(x=[C.REGEL_KURZ[z["regel"]] for z in zeilen], y=[z[s] for z in zeilen], name=C.SICHT_ANZEIGE[s], marker_color=farbe))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Lücke zur Referenz in %")
    return _fest(fig, titel)


def start_balken(zeilen: list) -> go.Figure:
    """Zugauswertungen bis zum Lokalminimum je Startlösung (mit Prüfrunde als zweitem Teil)."""
    namen = [C.START_ANZEIGE[z["art"]] for z in zeilen]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=namen, y=[z["ev_zuege"] for z in zeilen], name="bis zu den angenommenen Zügen", marker_color=C.COLOR_ZUG))
    fig.add_trace(go.Bar(x=namen, y=[z["ev_pruef"] for z in zeilen], name="Prüfrunde", marker_color=C.COLOR_BEWEIS))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Zugauswertungen")
    return _fest(fig, "Zugauswertungen bis zum Lokalminimum je Startlösung")


def raster_kosten(m: int, k_max: float, punkte: int = 40) -> np.ndarray:
    """Logarithmisches Raster der Kosten von etwa 1 M bis k_max M."""
    return np.geomspace(m, k_max * m, punkte)
