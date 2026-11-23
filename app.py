"""Gelernte Zugwahl: wo steckt der Aufwand? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein von Hand in numpy gebautes Zugnetz ordnet die 2-opt- und Or-opt-Züge einer Rundtour; die Suche probiert sie in dieser Reihenfolge. Gezählt wird, wo der Aufwand steckt:
Zugauswertungen, Merkmalszeilen des Netzes (K und O) und der Beweis, dass kein Zug mehr verbessert. Aufbau nach DEMO-PLAYBOOK §2: Kopf, Einleitung, Erklär-Expander, Primäransicht
(ein Abstieg Schritt für Schritt), fünf Experimente, Grenzen, Formeln.

Lauffähig mit: streamlit run app.py
"""
# ruff: noqa: E402  (die BLAS-Umgebungsvariablen müssen vor dem ersten Import von numpy stehen)

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # vor dem ersten Import von numpy: ein BLAS-Thread

import numpy as np
import streamlit as st

import modul_frisch
import zwl_constants as C
import zwl_cost as K
import zwl_evaluation as V
import zwl_experiments as X
import zwl_hand as H
import zwl_layout as L
import zwl_moves as M
import zwl_policy as PO
import zwl_presets as PR
import zwl_scenario as SC
import zwl_texts as TX
import zwl_visualization as VIS

modul_frisch.aktualisieren(__file__)

st.set_page_config(page_title="Gelernte Zugwahl: wo steckt der Aufwand? – Sebastian Hanisch", layout="wide")

ZELLE_NAME = {"n20_gleich": "n = 20 gleichverteilt", "n50_gleich": "n = 50 gleichverteilt", "n50_cluster": "n = 50 Gruppen"}


# ---------------------------------------------------------------------------------------------------------------- Zwischenspeicher
@st.cache_data(show_spinner=False)
def _frozen():
    try:
        return V.load_frozen()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def _refs():
    try:
        return V.load_refs()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def _ctx(familie, n, seed, doppelt=False):
    return X.kontext(familie, n, seed, doppelt)


@st.cache_data(show_spinner=False)
def _punkte(familie, n, seed):
    return [tuple(p) for p in (H.PUNKTE if n == C.N_HAND else SC.make_instance(familie, n, seed)[0])]


@st.cache_data(show_spinner=False)
def _e0(familie, n, seed, regel, modell_id, abbruch, liste, _scorer):
    return X.e0_abstieg(_ctx(familie, n, seed), regel, _scorer, familie, seed, abbruch, liste)


@st.cache_data(show_spinner=False)
def _e1_guete(modell_id, _scorer):
    return X.e1_guete(_scorer)


@st.cache_data(show_spinner=False)
def _e1_je_zug(modell_id, familie, n, count, _scorer):
    return X.e1_je_zug(_scorer, familie, n, count)


@st.cache_data(show_spinner=False)
def _e1_zugarten(modell_id, familie, n, count, _scorer):
    return X.e1_zugarten(_scorer, familie, n, count)


@st.cache_data(show_spinner=False)
def _e2(familie, doppelt):
    return [X.e2_beweis(familie, n, C.E2_INSTANZEN, doppelt) for n in C.E2_N_LISTE]


@st.cache_data(show_spinner=False)
def _e5(familie, n, count, kmodell_id, _params_k):
    refs = _refs()
    ref = None
    if refs is not None and familie in refs["refs"] and str(n) in refs["refs"][familie]:
        ref = V.refs_vektor(familie, n, count, refs)
    return X.e5_starts(familie, n, count, _params_k, ref)


# ---------------------------------------------------------------------------------------------------------------- Kopf, Einleitung, Erklärung
st.html(L.kopf_css(L.dunkles_design()))
with st.container(key="kopf"):
    st.title(TX.TITEL)
st.markdown(TX.INTRO)

with st.expander("💬 In einfachen Worten"):
    st.markdown(TX.EINFACHE_WORTE)

with st.expander("So funktioniert die gelernte Zugwahl", expanded=True):
    st.markdown(TX.SO_FUNKTIONIERT)

with st.expander("Wie hier gezählt wird (K und O)"):
    st.markdown(TX.ZAEHLREGEL)

with st.expander("Benannte Verfahren kurz erklärt"):
    st.markdown(TX.ERKLAERTABELLE)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider, einfach kopieren, um ein Szenario zu teilen.")

PR.load_permalink_settings()
PR.init_session_state_defaults()

# ---------------------------------------------------------------------------------------------------------------- Seitenleiste
with st.sidebar:
    PR.sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Szenario")
    st.caption("Benutzt von allen Abschnitten. n = 6 ist nur das Handbeispiel (feste Tabelle).")
    n_vorher = st.session_state["n_select"]
    familie = st.selectbox("Familie der Orte", options=PR.options("familie_select"), format_func=C.FAMILIE_ANZEIGE.get, key="familie_select", disabled=n_vorher == C.N_HAND,
                           help="Gleichverteilt, Gruppen oder Gitter. Die eingefrorenen Tabellen gibt es nur für n = 20 gleichverteilt, n = 50 gleichverteilt und n = 50 Gruppen; sonst rechnet die App live.")
    n = st.select_slider("Orte n", options=PR.options("n_select"), key="n_select",
                         help="Größe der Instanz. n = 6 ist nur das Handbeispiel; die Zugmenge hat n (11 n − 35) / 2 Züge.")
    if st.session_state["seed_input"] > PR.seed_max(n):
        st.session_state["seed_input"] = PR.seed_max(n)
    seed = st.number_input("Zufalls-Seed der Instanz", PR.bounds("seed_input")[0], PR.seed_max(n) if n != C.N_HAND else 1, key="seed_input", step=1, disabled=n == C.N_HAND,
                           help="Welche Instanz die Experimente zeigen. Für Zufalls-Seed 0 bis 19 (bei n = 50: 0 bis 9) sind Referenzlängen vorgerechnet.")
    st.button("🎲 Neue Instanz", width="stretch", on_click=PR.randomize_instance, disabled=n == C.N_HAND)

    st.header("⚙️ Zugwahl")
    regel = st.selectbox("Regel (Reihenfolge der Kandidaten)", options=PR.options("regel_select"), format_func=C.REGEL_ANZEIGE.get, key="regel_select",
                         help="Benutzt von der Primäransicht und dem Vergleich in Experiment 4. Die Regeln mit Netz brauchen ein Zugnetz.")
    st.markdown("**Zugnetz Z** (Experiment 1; Regeln „gelernt“)")
    quelle = st.radio("Zugnetz", options=PR.options("quelle_select"), format_func=C.QUELLE_ANZEIGE.get, key="quelle_select",
                      help="Vorgerechnet: fünf Modelle (Gewichts-Zufalls-Seed 0 bis 4, 3 000 Schritte, 300 Trainingsinstanzen). Hier trainieren: Training in der App, Knopf bei Experiment 1.")
    gewicht = st.number_input("Gewichts-Zufalls-Seed", *PR.bounds("gewicht_input"), key="gewicht_input", step=1,
                              help="Welches vorgerechnete Modell (0 bis 4), bei „Hier trainieren“ der Zufalls-Seed von Anfangsgewichten und Zeilenziehung. Gilt auch für das Merkmalsnetz K in Experiment 5.")
    trainiert_ok = quelle == "trainieren"
    schritte = st.select_slider("Schritte (nur bei „Hier trainieren“)", options=PR.options("schritte_select"), key="schritte_select", disabled=not trainiert_ok,
                                help="Trainingsschritte zu je 4 096 Zeilen; höchstens 3 000, danach bricht ein Zeitwächter nach 100 Sekunden ab.")
    tinst = st.select_slider("Trainingsinstanzen (nur bei „Hier trainieren“)", options=PR.options("trainings_inst_select"), key="trainings_inst_select", disabled=not trainiert_ok,
                             help="Instanzen mit n = 20 (Zufalls-Seed ab 2000), aus deren Suchen die Trainingszeilen stammen.")
    st.markdown("**Budget und Beweis** (Experimente 3 und 4)")
    budget = st.select_slider("Budget k (k · M Zugauswertungen)", options=PR.options("budget_select"), key="budget_select",
                              help="Experiment 3: die Suche darf k · M Zugauswertungen ausgeben (M = Zahl der Züge). Bei n = 50 gibt es nur k = 2 und 8.")
    abbruch = st.select_slider("Abbruch der Prüfung (0 = aus)", options=PR.options("abbruch_select"), key="abbruch_select",
                               help="Experiment 4 und Primäransicht: nach so vielen erfolglosen Kandidaten gibt die Suche auf; das ist kein Beweis mehr. 0 = volle Prüfrunde.")
    liste = st.select_slider("Nachbarliste k (0 = aus)", options=PR.options("liste_select"), key="liste_select",
                             help="Experiment 4 und Primäransicht: nur Züge, bei denen mindestens eine neue Kante einen Rang unter k hat. 0 = alle Züge.")

n, seed, gewicht, schritte, tinst, budget, abbruch, liste = (int(v) for v in (n, seed, gewicht, schritte, tinst, budget, abbruch, liste))
PR.sync_query_params({"familie_select": familie, "n_select": n, "seed_input": seed, "regel_select": regel, "quelle_select": quelle, "gewicht_input": gewicht, "schritte_select": schritte,
                      "trainings_inst_select": tinst, "budget_select": budget, "abbruch_select": abbruch, "liste_select": liste})

frozen, refs = _frozen(), _refs()
hand = n == C.N_HAND
fam_anzeige = "Handbeispiel" if hand else C.FAMILIE_ANZEIGE[familie]
zelle = f"n{n}_{familie}"
zelle_da = frozen is not None and zelle in frozen["e3"]
ctx = _ctx(familie, n, seed)
M_n = ctx.nb.M


# ---------------------------------------------------------------------------------------------------------------- Zugnetz bestimmen
def _zugnetz():
    """(Parameter, Angaben) des Zugnetzes Z je nach Quelle; (None, None), wenn „Hier trainieren“ noch nicht gelaufen ist."""
    if quelle == "vorgerechnet":
        try:
            params, meta = X.lade_modell("Z", gewicht)
        except (OSError, ValueError) as fehler:
            st.error(f"Das vorgerechnete Zugnetz ist unbrauchbar: {fehler}")
            return None, None
        return params, {"art": "vorgerechnet", "id": f"v{gewicht}", "schritte": int(meta["steps"]), "zeilen": int(meta["n_train_rows"]), "instanzen": int(meta["n_train_instances"]), "seed": gewicht}
    eintrag = st.session_state.get("zmodell")
    if eintrag is not None and eintrag["key"] == (gewicht, schritte, tinst):
        c = eintrag["counts"]
        return eintrag["params"], {"art": "trainiert", "id": f"t{gewicht}_{schritte}_{tinst}_{c['schritte']}", "schritte": c["schritte"], "zeilen": c["zeilen"], "instanzen": tinst, "seed": gewicht}
    return None, None


Z_params, Z_info = _zugnetz()
braucht_netz = regel in C.REGELN_MIT_NETZ

# ---------------------------------------------------------------------------------------------------------------- E0: ein Abstieg Schritt für Schritt
st.markdown(f"## 🧭 {TX.KERNFRAGEN[0]}")
st.caption(f"Regler: Familie, n, Zufalls-Seed, Regel, Abbruch, Nachbarliste{', Zugnetz' if braucht_netz else ''}. Die Suche beginnt mit dem Nächsten Nachbarn ab Ort 0 und nimmt jeweils den ersten verbessernden Zug.")
if braucht_netz and Z_params is None:
    st.info("Die Regel „gelernt“ braucht ein Zugnetz: wähle „Vorgerechnet“ oder trainiere eines bei Experiment 1. Bis dahin zeigt die Primäransicht nichts.")
else:
    r0 = _e0(familie, n, seed, regel, Z_info["id"] if Z_info and braucht_netz else "-", abbruch, liste, Z_params if braucht_netz else None)
    punkte = _punkte(familie, n, seed)
    zuege = r0["zuege"]
    cfg = f"{familie}_{n}_{seed}_{regel}_{abbruch}_{liste}_{Z_info['id'] if Z_info and braucht_netz else '-'}"
    schritt = st.slider("Schritt (0 = Startlösung, dann Zug für Zug, zuletzt die Prüfrunde)", 0, zuege + 1, min(1, zuege + 1), key=f"e0_schritt_{cfg}",
                        help="Je Schritt: die Tour nach dem Zug, entfernte Kanten rot, neue grün, das Delta und wo der Zug in der Reihenfolge stand.")
    c_karte, c_rechts = st.columns([3, 2])
    if schritt == 0:
        fig = VIS.karte(punkte, r0["start"], start=None, titel=f"Startlösung (Nächster Nachbar), Länge {V.de(r0['start_laenge'], 1)}" if not hand else f"Startlösung (Nächster Nachbar), Länge {V.zahl(r0['start_laenge'])}")
        text_schritt = "Startlösung: Nächster Nachbar ab Ort 0."
    elif schritt <= zuege:
        p = r0["schritte"][schritt - 1]
        bisher = sum(q["stelle"] for q in r0["schritte"][:schritt])
        fig = VIS.karte(punkte, p["tour_davor"], p["entfernt"], p["neu"], start=r0["start"], titel=f"Zug {schritt} von {zuege}: Tour vor dem Zug")
        text_schritt = (f"**Zug {schritt}** ({C.ZUGARTEN[p['art']]}), Delta {V.de_vz(p['delta'], 1 if hand else 3)}: der Zug stand an Stelle {V.zahl(p['stelle'])} der Reihenfolge, "
                        f"bis hierher insgesamt {V.zahl(bisher)}. Entfernt (rot): {', '.join(f'{a}-{b}' for a, b in p['entfernt'])}; neu (grün): {', '.join(f'{a}-{b}' for a, b in p['neu'])}.")
    else:
        fig = VIS.karte(punkte, r0["tour"], start=r0["start"], titel=f"Endtour, Länge {V.de(r0['end_laenge'], 1) if not hand else V.zahl(r0['end_laenge'])}")
        text_schritt = (f"**Prüfrunde:** {V.zahl(r0['n_pruef'])} Kandidaten bewertet, " + ("keiner verbessert." if not abbruch else f"keiner der ersten {abbruch} verbessert (Abbruch, kein Beweis)."))
    with c_karte:
        st.plotly_chart(fig, width="stretch", key=f"e0_karte_{cfg}_{schritt}")
    with c_rechts:
        st.markdown(text_schritt)
        st.plotly_chart(VIS.balken_aufwand(r0["ev_zuege"], r0["ev_pruef"], r0["rows"]), width="stretch", key=f"e0_balken_{cfg}")
        kosten = V.kosten_zeile(regel, r0["ev_zuege"] + r0["ev_pruef"], r0["rows"])
        st.markdown(f"| Zählweise | Aufwand dieses Abstiegs |\n|---|---|\n| nur Zugauswertungen | {V.zahl(kosten['frei'])} |\n| K (Zeilen zählen mit) | {V.zahl(kosten['K'])} |\n| O (grobe Operationen) | {V.zahl(kosten['O'])} |")
    st.info(V.verdict_instanz_e0(zuege, r0["ev_zuege"], r0["ev_pruef"], abbruch, liste, r0["falsch"]))
    if hand:
        st.markdown("**Handbeispiel: sechs Orte als Entfernungstabelle** (gerundete Entfernungen; die Punkte dahinter stehen nur für die Karte).")
        kopf = "| | " + " | ".join(str(i) for i in range(6)) + " |\n|---|" + "---|" * 6 + "\n"
        st.markdown(kopf + "\n".join(f"| **{i}** | " + " | ".join(str(v) for v in H.TABELLE[i]) + " |" for i in range(6)))
        d_start = M.deltas(ctx.D, np.array(H.START), ctx.nb)
        st.markdown(f"Die Zugmenge hat M(6) = {M.neighborhood_size(6)} Züge ({' + '.join(str(v) for v in M.m_arten(6).values())}). Am Start (Länge {V.zahl(M.tour_length(ctx.D, H.START))}) verbessern "
                    f"{int((d_start < 0).sum())} von {M.neighborhood_size(6)} Zügen, das beste Delta ist {V.de_vz(float(d_start.min()), 0)}. Das Optimum ist {H.OPTIMUM} (alle 60 verschiedenen Rundwege geprüft). "
                    f"Verschiedene Nachbartouren sind es nur 29: ein 2-opt über zwei Orte ist dasselbe wie eine Verschiebung um eine Stelle.")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- E1: Ordnet das Netz besser?
st.subheader(f"🔬 {TX.KERNFRAGEN[1]}")
st.caption("Regler: Zugnetz (Quelle, Gewichts-Zufalls-Seed, Schritte, Trainingsinstanzen), Familie, n. Das Netz sieht nie das Delta, nur neun Merkmale des Zugs.")
if quelle == "trainieren":
    zeilen_plan = V.training_zeilen(schritte)
    st.caption(f"Training: Schritte × Zeilen = {V.zahl(schritte)} × {V.zahl(C.Z_BATCH)} = {V.zahl(zeilen_plan)} Zeilen, gezogen aus den Zuständen von {V.zahl(tinst)} Trainingsinstanzen (n = 20, Zufalls-Seed ab 2000), "
               f"Lernrate {V.de(C.Z_LERNRATE, 2)}. Nach {int(C.ZEITWAECHTER_SEKUNDEN)} Sekunden bricht ein Zeitwächter ab, das Zwischenmodell bleibt nutzbar.")
    if st.button(f"🏋️ Zugnetz trainieren ({V.zahl(schritte)} Schritte × {V.zahl(C.Z_BATCH)} Zeilen = {V.zahl(zeilen_plan)} Zeilen)", key="trainieren_knopf"):
        balken = st.progress(0.0, text="Datensammlung")

        def _fortschritt(s, gesamt):
            balken.progress(s / gesamt, text=f"Schritt {V.zahl(s)} von {V.zahl(gesamt)}")

        with st.spinner(f"Training: {V.zahl(schritte)} Schritte × {V.zahl(C.Z_BATCH)} Zeilen = {V.zahl(zeilen_plan)} Zeilen aus {V.zahl(tinst)} Trainingsinstanzen"):
            p_neu, verlauf, cnt = X.trainiere_zugnetz(schritte, tinst, gewicht, progress=_fortschritt, max_seconds=C.ZEITWAECHTER_SEKUNDEN)
        st.session_state["zmodell"] = {"key": (gewicht, schritte, tinst), "params": p_neu, "verlauf": verlauf, "counts": cnt}      # nur das jüngste Modell bleibt im Speicher
        st.rerun()
    eintrag = st.session_state.get("zmodell")
    if eintrag is not None and eintrag["key"] == (gewicht, schritte, tinst):
        c = eintrag["counts"]
        if c["abgebrochen"]:
            st.warning(f"Nach {V.zahl(c['schritte'])} von {V.zahl(schritte)} Schritten abgebrochen (Zeitwächter); das Zwischenmodell wird gezeigt.")
        st.caption(f"Dieser Lauf: {V.zahl(c['schritte'])} Schritte × {V.zahl(C.Z_BATCH)} = {V.zahl(c['zeilen'])} Zeilen aus {V.zahl(c['daten_zeilen'])} gesammelten Zeilen ({V.zahl(c['daten_zustaende'])} Zustände), "
                   f"{V.de(c['sekunden'], 1)} Sekunden (Zeit ist keine Aussage über das Verfahren).")
        st.plotly_chart(VIS.lernkurve(eintrag["verlauf"], "Verlust beim Training (Mittel über 50 Schritte)"), width="stretch", key="e1_lernkurve")
if hand:
    st.info("Experiment 1 braucht n von mindestens 10: wähle eine andere Größe in der Seitenleiste.")
elif Z_params is None:
    st.info("Für diese Einstellung ist noch kein Zugnetz da: wähle „Vorgerechnet“ oder starte das Training.")
else:
    g = _e1_guete(Z_info["id"], Z_params)
    count_ev = 20 if n <= 20 else 10
    je_zug = _e1_je_zug(Z_info["id"], familie, n, count_ev, Z_params)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(VIS.auc_balken(g["auc"]), width="stretch", key="e1_auc")
        st.caption(f"AUC auf {V.zahl(g['zeilen'])} Zeilen aus {V.zahl(g['zustaende'])} Validierungszuständen ({g['instanzen']} Instanzen, n = 20 gleichverteilt, Zufalls-Seed ab 1000; je Zustand {C.Z_PRO_ZUSTAND} gezogene Zeilen, "
                   f"die AUC ist über alle Zeilen gemeinsam gerechnet, nicht je Zustand); Anteil verbessernder Züge {V.de(100 * g['anteil_verbessernd'], 1)} %.")
    with c2:
        st.plotly_chart(VIS.je_zug_balken(je_zug), width="stretch", key="e1_je_zug")
        st.caption(f"Erster Abstieg aus dem Nächsten Nachbarn, {count_ev} Instanzen ({fam_anzeige}, n = {n}); gelernt gezogen und deterministisch mit Gewichts-Zufalls-Seed {Z_info['seed']}.")
    art, text = V.verdict_guete(g["auc"]["netz"], g["auc"]["hand"], je_zug["zufall"]["je_zug"], je_zug["gelernt"]["je_zug"], n)
    getattr(st, art)(text)
    zug = _e1_zugarten(Z_info["id"], familie, n, count_ev, Z_params)
    with st.expander("Welche Zugart bevorzugt das Netz? (Zustand und Zugart)"):
        st.plotly_chart(VIS.zugarten_balken(zug), width="stretch", key="e1_zugarten")
        st.caption(f"Über {V.zahl(zug['zustaende'])} Zustände der Abstiege ({count_ev} Instanzen). Gemessen ist die Zugwahl aus dem Zustand der Tour, nicht die Zugart nach Suchphase oder Stagnation: "
                   "die Reserve R der Hyperheuristiken ist damit nur teilweise gefüllt.")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- E2: Wie viel ist der Beweis?
st.subheader(f"🔬 {TX.KERNFRAGEN[2]}")
st.caption("Regler: Familie, Zufalls-Seeds der Instanzen (0 bis 19, je 20 Instanzen). Die Regel spielt keine Rolle: die Prüfrunde hängt nicht von der Reihenfolge ab.")
doppelt = st.toggle("Zählweise der Vormessung (Einzelortverschiebung doppelt gezählt)", key="e2_doppelt",
                    help="Die frühere Vormessung zählte die Verschiebung eines einzelnen Ortes in beiden Richtungen, also denselben Zug zweimal: M(n) + n(n − 2) Züge.")
fam_e2 = "gleich" if hand else familie
tab2 = _e2(fam_e2, doppelt)
st.markdown("| n | M(n) = Prüfrunde | Zugauswertungen bis zum Lokalminimum (Mittel, SD) | Anteil der Prüfrunde | höchstens gespart durch eine perfekte Startlösung |\n|---|---|---|---|---|\n" + "\n".join(
    f"| {z['n']} | {V.zahl(z['M'])} | {V.zahl(z['mittel'])} ({V.zahl(z['sd'])}) | {V.prozent(z['anteil'])} % | {V.prozent(z['gespart_max'])} % |" for z in tab2))
st.plotly_chart(VIS.beweis_anteil_balken(tab2), width="stretch", key="e2_balken")
n_e2 = n if n in C.E2_N_LISTE else 20
z_e2 = next(z for z in tab2 if z["n"] == n_e2)
art, text = V.verdict_beweis(z_e2["anteil"], n_e2, C.E2_INSTANZEN)
getattr(st, art)(text)
a = M.m_arten(n_e2)
st.markdown(f"**Aufschlüsselung der Prüfrunde bei n = {n_e2}:** 2-opt {V.zahl(a['2-opt'])}, Or-opt 1 {V.zahl(a['Or-opt 1'])}, Or-opt 2 {V.zahl(a['Or-opt 2'])}, Or-opt 3 {V.zahl(a['Or-opt 3'])}; "
            f"zusammen M({n_e2}) = {V.zahl(M.neighborhood_size(n_e2))}.")
if frozen is not None:
    zeilen_k = [f"| {ZELLE_NAME.get(zz, zz)} | " + " | ".join(V.prozent(V.e3_pruef_anteil(frozen, zz, r)) + " %" for r in ("zufall", "nachbarliste", "gelernt", "gelernt_det")) + " |" for zz in frozen["e3"]]
    st.markdown("**Suche mit Kicks:** Anteil der Prüfrunden an allen Zugauswertungen (eingefroren, Budget bis k_max · M, alle Läufe).\n\n| Zelle | Zufall | Nachbarlisten-Reihenfolge | gelernt | gelernt det. |\n|---|---|---|---|---|\n"
                + "\n".join(zeilen_k))

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- E3: Hauptvergleich
st.subheader(f"🔬 {TX.KERNFRAGEN[3]}")
st.caption("Regler: Familie, n, Budget k, Zugnetz (Gewichts-Zufalls-Seed für die Live-Probe). Die eingefrorene Tabelle gilt für das fertige Modell der Messung; Marken gegen Zufall mit gepaartem Bootstrap-Intervall über die Instanzen.")
if hand or not zelle_da or refs is None:
    st.info("Für diese Zelle ist nichts gemessen (eingefroren sind n = 20 gleichverteilt, n = 50 gleichverteilt und n = 50 Gruppen); die Live-Probe unten rechnet trotzdem, wenn n von mindestens 10 gewählt ist.")
elif budget not in frozen["e3"][zelle]["ks"]:
    st.info(f"Bei n = {n} gibt es nur die Budgets k = 2 und 8; wähle ein anderes k.")
else:
    z3 = frozen["e3"][zelle]
    zeilen3 = V.e3_tabelle(frozen, zelle, budget, refs)

    def _zelle_cmp(c):
        return "Gegner" if c is None else f"{V.de_vz(c['v'])} [{V.de_vz(c['lo'])}; {V.de_vz(c['hi'])}] {c['marke']}"

    st.markdown(f"**{fam_anzeige}, n = {n}, Budget k = {budget} (= {V.zahl(budget * z3['M'])} Zugauswertungen), {z3['instanzen']} Instanzen × {z3['zufalls_seeds']} Zufalls-Seeds** (gelernte Regeln: {z3['gewichts_seeds']} Gewichts-Seeds, reihum)")
    st.markdown("| Regel | Lücke nur Zugauswertungen | Lücke K | Lücke O | gegen Zufall (nur Zugauswertungen) | gegen Zufall (K) | gegen Zufall (O) |\n|---|---|---|---|---|---|---|\n" + "\n".join(
        f"| {C.REGEL_ANZEIGE[z['regel']]} | {V.de(z['frei'])} % | {V.de(z['K'])} % | {V.de(z['O'])} % | {_zelle_cmp(z['cmp_frei'])} | {_zelle_cmp(z['cmp_K'])} | {_zelle_cmp(z['cmp_O'])} |" for z in zeilen3))
    st.caption("Lücke = Länge über der Referenz (Held-Karp bei n = 20, CP-SAT bei n = 50, dort teils nur Untergrenze) in Prozent. Differenz in Prozentpunkten (Regel minus Zufall, negativ = besser), Intervall: 95 % Bootstrap über die Instanzen. "
               "Marken: schlägt ab 0,25 Prozentpunkten mit Intervall ohne 0, gleich unter 0,10 oder mit 0 im Intervall.")
    if budget < C.SICHT_GEWICHT_Z["O"]:
        st.caption(f"In O kostet ein Schritt des Netzes {V.zahl(C.SICHT_GEWICHT_Z['O'] * z3['M'])} Zugauswertungen (M Zeilen zu je 176 statt 8 Operationen); bei k = {budget} reicht das Budget nicht für einen Schritt, die Lücke in O ist darum die der Startlösung (Nächster Nachbar).")
    st.plotly_chart(VIS.luecken_balken(zeilen3, f"Lücke zur Referenz bei Budget k = {budget}"), width="stretch", key="e3_balken")
    marken = V.e3_marken(frozen, zelle, "gelernt", budget, "zufall", refs)
    art, text = V.verdict_zugwahl(marken, n, familie, budget, z3["instanzen"], z3["zufalls_seeds"])
    getattr(st, art)(text)
    streu = [V.e3_gewichts_streuung(frozen, zelle, "gelernt", budget, s, refs) for s in C.SICHTEN]
    st.caption("Streuung über die Gewichts-Seeds (gelernt, Mittel je Gewichts-Seed): " + "; ".join(f"{C.SICHT_ANZEIGE[s]}: {V.de(min(v))} bis {V.de(max(v))} %" for s, v in zip(C.SICHTEN, streu)))

with st.expander("Live-Probe: eine Instanz, fünf Zufalls-Seeds, vier Regeln"):
    live_ok = (not hand) and ((n == 20 and budget <= 8) or (n == 50 and budget == 2) or n == 10) and (Z_params is not None)
    if hand:
        st.info("Die Live-Probe braucht n von mindestens 10.")
    elif Z_params is None:
        st.info("Die Live-Probe braucht ein Zugnetz.")
    elif not live_ok:
        st.info("Live läuft n = 20 mit k bis 8 und n = 50 mit k = 2; die größeren Budgets gibt es nur eingefroren.")
    else:
        gesamt = 4 * C.E3_LIVE_SEEDS * budget * M_n
        key_probe = f"probe_{familie}_{n}_{seed}_{budget}_{Z_info['id']}"
        if st.button(f"Probe starten: 4 Regeln × {C.E3_LIVE_SEEDS} Zufalls-Seeds × k · M = {V.zahl(gesamt)} Zugauswertungen", key="probe_knopf"):
            with st.spinner(f"Live-Probe: 4 Regeln × {C.E3_LIVE_SEEDS} Zufalls-Seeds × {V.zahl(budget * M_n)} Zugauswertungen = {V.zahl(gesamt)}"):
                pr = X.e3_probe(ctx, familie, seed, budget, Z_params)
            raster = VIS.raster_kosten(M_n, budget)
            st.session_state["probe"] = {"key": key_probe, "best": {r: v["best"] for r, v in pr.items()},
                                         "kurven": {s: {r: X.kurve(v["runs"], r, s, raster) for r, v in pr.items()} for s in ("K", "O")}, "raster": raster}
        p = st.session_state.get("probe")
        if p is not None and p["key"] == key_probe:
            ref = V.ref_laenge(familie, n, seed, refs) if refs is not None else None
            bez = ref[0] if ref else float(min(np.nanmin(v) for v in p["best"].values()))
            st.markdown("| Regel | Lücke nur Zugauswertungen | Lücke K | Lücke O |\n|---|---|---|---|\n" + "\n".join(
                f"| {C.REGEL_ANZEIGE[r]} | " + " | ".join(V.de(100 * (float(np.nanmean(v[:, i])) - bez) / bez) + " %" for i in range(3)) + " |" for r, v in p["best"].items()))
            st.caption(("Referenz: " + ("bewiesenes Optimum" if ref[1] else "beste bekannte Länge (CP-SAT, Untergrenze nicht erreicht)") if ref else "Keine Referenz vorgerechnet: Lücke zur besten gefundenen Länge dieser Probe")
                       + f". Eine Instanz, {C.E3_LIVE_SEEDS} Zufalls-Seeds, Mittel; das ist eine Probe, kein Befund.")
            k1, k2 = st.columns(2)
            for col, s in ((k1, "K"), (k2, "O")):
                with col:
                    st.plotly_chart(VIS.kurve_bestwert(p["raster"] * (1 if s == "K" else C.OPS_JE_KANDIDAT), p["kurven"][s], "Kosten in K" if s == "K" else "Kosten in O", f"Bester Wert über Kosten ({s})"),
                                    width="stretch", key=f"probe_kurve_{s}")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- E4: Beweis abkürzen
st.subheader(f"🔬 {TX.KERNFRAGEN[4]}")
st.caption("Regler: Abbruch der Prüfung, Nachbarliste k, Regel (Live-Probe), Budget k. Beides spart Zugauswertungen: der Abbruch gibt den Beweis auf, die Nachbarliste beschränkt die Zugmenge.")
st.markdown("**(a) Abbruch der Prüfung**")
zk4 = zelle if (frozen is not None and zelle in frozen["e4a"]) else None
if hand or zk4 is None:
    st.info("Für diese Zelle ist Experiment 4a nicht gemessen (eingefroren sind n = 20 und n = 50, gleichverteilt).")
else:
    z4 = frozen["e4a"][zk4]
    k4 = budget if budget in z4["ks"] else z4["ks"][-1]
    g_anz = abbruch if abbruch in C.E4A_G else 100
    if abbruch not in C.E4A_G:
        st.caption("Der Abbruch ist in der Seitenleiste aus (oder keine gemessene Stufe): gezeigt wird Abbruch nach 100 Kandidaten.")
    zeilen4 = V.e4a_tabelle(frozen, zk4, g_anz, k4, refs)
    st.markdown(f"**Abbruch nach {g_anz} erfolglosen Kandidaten, Budget k = {k4}** ({z4['instanzen']} Instanzen × {z4['zufalls_seeds']} Zufalls-Seeds; gelernt: {z4['gewichts_seeds']} Gewichts-Seeds)")
    st.markdown("| Regel | Abstiege an falschem Lokalminimum | Lücke nur Zugauswertungen | Lücke K | Lücke O | gelernt gegen Zufall (K) |\n|---|---|---|---|---|---|\n" + "\n".join(
        f"| {C.REGEL_ANZEIGE[z['regel']]} | {V.prozent(z['falsch'])} % | {V.de(z['frei'])} % | {V.de(z['K'])} % | {V.de(z['O'])} % | "
        f"{'-' if z['cmp_K'] is None else f'{V.de_vz(z['cmp_K']['v'])} {z['cmp_K']['marke']}'} |" for z in zeilen4))
    st.caption("Falsches Lokalminimum: eine vollständige Prüfrunde fände noch einen verbessernden Zug (Gegenprobe). Mit Abbruch ist das Ende kein Beweis. Gelernt gegen Zufall mit demselben Abbruch.")
    marken4 = {s: next(z for z in zeilen4 if z["regel"] == "gelernt")[f"cmp_{s}"]["marke"] for s in C.SICHTEN}
    z_zufall = next(z for z in zeilen4 if z["regel"] == "zufall")
    art, text = V.verdict_abbruch(z_zufall["falsch"], marken4, g_anz, n, familie, k4, z4["instanzen"], z4["zufalls_seeds"])
    getattr(st, art)(text)

st.markdown("**(b) Nachbarliste statt Beweis**")
zk4b = zelle if (frozen is not None and zelle in frozen["e4b"]) else None
if hand or zk4b is None:
    st.info("Für diese Zelle ist Experiment 4b nicht gemessen.")
else:
    z4b = frozen["e4b"][zk4b]
    kl = liste if liste in (3, 5, 8) else C.LISTE_DEFAULT_K
    if liste not in (3, 5, 8):
        st.caption(f"Die Nachbarliste ist in der Seitenleiste aus: gezeigt wird k = {kl}.")
    ab = {r: V.e4b_abstieg(frozen, zk4b, kl, r) for r in C.REGELN_E4B}
    st.markdown(f"**Erster Abstieg aus dem Nächsten Nachbarn, Nachbarliste k = {kl}** ({z4b['instanzen']} Instanzen × {z4b['zufalls_seeds']} Zufalls-Seeds, ohne Kicks)")
    st.markdown("| Regel | Prüfrunde statt M | Zugauswertungen gesamt (voll) | Länge gegen den vollen Abstieg | Endtouren ohne volles Lokalminimum |\n|---|---|---|---|---|\n" + "\n".join(
        f"| {C.REGEL_ANZEIGE[r]} | {V.zahl(a_['pruef'])} statt {V.zahl(a_['M'])} | {V.zahl(a_['ev'])} ({V.zahl(a_['ev_voll'])}) | {V.de_vz(a_['laenge_gegen_voll'])} % | {V.prozent(a_['nicht_voll'])} % |" for r, a_ in ab.items()))
    art, text = V.verdict_nachbarliste(ab["zufall"]["pruef"], ab["zufall"]["M"], ab["zufall"]["nicht_voll"], ab["zufall"]["laenge_gegen_voll"], kl)
    getattr(st, art)(text)
    k4b = 2 if 2 in z4b["ks"] else z4b["ks"][0]
    marken_l = {s: V.e4b_vergleich(frozen, zk4b, kl, "gelernt", k4b, s)["marke"] for s in C.SICHTEN}
    art, text = V.verdict_liste_gelernt(marken_l, kl, n)
    getattr(st, art)(text)
    st.caption(f"(c) gelernte Reihenfolge innerhalb der Nachbarliste: Suche mit Kicks, Budget k = {k4b}, {z4b['instanzen']} Instanzen × {z4b['zufalls_seeds']} Zufalls-Seeds. "
               "Die Erzeugung der Kandidaten aus den Listen ist nicht gezählt; Don't-Look-Bits sind nicht gebaut.")

with st.expander("Live-Probe: ein Abstieg mit den Einstellungen der Seitenleiste"):
    if hand or (braucht_netz and Z_params is None):
        st.info("Die Live-Probe braucht n von mindestens 10 und für gelernte Regeln ein Zugnetz.")
    elif abbruch == 0 and liste == 0:
        st.info("Stelle in der Seitenleiste einen Abbruch oder eine Nachbarliste ein.")
    else:
        pr4 = X.e4_probe(ctx, familie, seed, regel, Z_params if braucht_netz else None, abbruch, liste)
        p_, v_ = pr4["probe"], pr4["voll"]
        st.markdown("| | Zugauswertungen gesamt | letzte Runde | Länge |\n|---|---|---|---|\n"
                    f"| voller Abstieg | {V.zahl(v_['ev_zuege'] + v_['ev_pruef'])} | {V.zahl(v_['ev_pruef'])} | {V.de(pr4['laenge_voll'], 1)} |\n"
                    f"| mit Abbruch {abbruch}, Nachbarliste {liste} | {V.zahl(p_['ev_zuege'] + p_['ev_pruef'])} | {V.zahl(p_['ev_pruef'])} | {V.de(pr4['laenge'], 1)} |")
        st.caption("Die Endtour ist " + ("kein" if pr4["kein_lokalminimum"] else "ein") + " Lokalminimum der vollen Zugmenge (unabhängige Gegenprobe über alle Nachbartouren). Ein Abstieg, ein Zufalls-Seed: das ist eine Probe, kein Befund.")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- E5: Startlösung
st.subheader(f"🔬 {TX.KERNFRAGEN[5]}")
st.caption("Regler: Familie, n, Gewichts-Zufalls-Seed (Merkmalsnetz K vorgerechnet), Probetraining. Nicht hier: Mehrfachstart und Gesamtbudget.")
if hand:
    st.info("Experiment 5 braucht n von mindestens 10.")
else:
    kq = st.radio("Merkmalsnetz K", ["vorgerechnet", "probetraining"], format_func=lambda s: "Vorgerechnet (Gewichts-Zufalls-Seed der Seitenleiste)" if s == "vorgerechnet" else "Probetraining", horizontal=True, key="e5_kquelle")
    params_k, k_id = None, "-"
    if kq == "vorgerechnet":
        try:
            params_k, kmeta = X.lade_modell("K", gewicht)
            k_id = f"v{gewicht}"
        except (OSError, ValueError):
            st.info("Das vorgerechnete Merkmalsnetz K fehlt: wähle das Probetraining.")
    else:
        tr = K.k_konstruktion(20)
        st.caption(f"Probetraining: {C.K_SCHRITTE_PROBE} Schritte × {C.K_INST} Instanzen × {C.K_ROLL} Touren = {V.zahl(C.K_SCHRITTE_PROBE * C.K_BATCH_TOUREN)} Touren, "
                   f"{V.zahl(C.K_SCHRITTE_PROBE * C.K_BATCH_TOUREN * tr)} Merkmalszeilen (n = 20, je Tour {tr} Zeilen), Lernrate {V.de(C.K_LERNRATE, 2)}.")
        if st.button(f"🏋️ Probetraining starten ({C.K_SCHRITTE_PROBE} Schritte × {C.K_BATCH_TOUREN} Touren)", key="kprobe_knopf"):
            with st.spinner(f"Probetraining: {C.K_SCHRITTE_PROBE} Schritte × {C.K_BATCH_TOUREN} Touren = {V.zahl(C.K_SCHRITTE_PROBE * C.K_BATCH_TOUREN)} Touren"):
                pk, verl, cnt_k = PO.train_featnet(C.K_SCHRITTE_PROBE, seed=gewicht, max_seconds=C.ZEITWAECHTER_SEKUNDEN)
            st.session_state["kprobe"] = {"seed": gewicht, "params": pk, "counts": cnt_k}
            st.rerun()
        ep = st.session_state.get("kprobe")
        if ep is not None and ep["seed"] == gewicht:
            params_k, k_id = ep["params"], f"p{gewicht}_{ep['counts']['schritte']}"
            if refs is not None and "val" in refs:
                gp, gn = X.k_greedy_gap(params_k, refs)
                art, text = V.verdict_probe_k(gp, gn)
                getattr(st, art)(text + f" (Mittel über {C.Z_VAL_INSTANZEN} Validierungsinstanzen, n = 20, Held-Karp als Referenz; {V.zahl(ep['counts']['touren'])} Touren, {V.de(ep['counts']['sekunden'], 1)} Sekunden, Zeit ist kein Befund.)")
        else:
            st.info("Noch kein Probetraining für diesen Gewichts-Zufalls-Seed.")
    count5 = 20 if n <= 20 else 10
    zeilen5 = _e5(familie, n, count5, k_id, params_k)
    ref5 = refs is not None and familie in refs["refs"] and str(n) in refs["refs"][familie] and len(refs["refs"][familie][str(n)]) >= count5
    st.markdown("| Startlösung | Lücke der Startlösung | Zugauswertungen bis zum Lokalminimum | davon Prüfrunde | Ersparnis gegen Nächster Nachbar | Konstruktion K | Konstruktion O | Gesamt K | Gesamt O |\n|---|---|---|---|---|---|---|---|---|\n" + "\n".join(
        f"| {C.START_ANZEIGE[z['art']]} | {V.de(z['gap_start']) + ' %' if z['gap_start'] is not None else '-'} | {V.zahl(z['ev_gesamt'])} | {V.zahl(z['ev_pruef'])} | {V.de(z['ersparnis'], 0)} % | "
        f"{V.zahl(z['k_start'])} | {V.zahl(z['o_start'])} | {V.zahl(z['k_gesamt'])} | {V.zahl_k(z['o_gesamt'])} |" for z in zeilen5))
    st.caption(f"{count5} Instanzen ({fam_anzeige}, n = {n}), Abstieg mit Zufallsreihenfolge aus der Startlösung. Zugauswertungen ohne Konstruktion; die Zeile „schon Lokalminimum“ ist die Obergrenze der Ersparnis. "
               + ("Lücke zur Referenz (Held-Karp bzw. CP-SAT)." if ref5 else "Für diese Zelle ist keine Referenz vorgerechnet, die Lücke der Startlösung fehlt."))
    st.plotly_chart(VIS.start_balken(zeilen5), width="stretch", key="e5_balken")
    zk = next((z for z in zeilen5 if z["art"] == "netz_k"), None)
    zmax = next(z for z in zeilen5 if z["art"] == "optimum")
    if zk is not None:
        art, text = V.verdict_start(zk["ersparnis"], zmax["ersparnis"], zk["k_start"])
        getattr(st, art)(text)

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------- Grenzen und Formeln
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(TX.LESEHILFE)
    st.markdown(TX.FORMELN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Neuronale kombinatorische Optimierung](https://sebastianhanisch.net/konzepte-neuronale-kombinatorische-optimierung.html)."
)
