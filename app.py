"""Was müsste anders sein? Wie lange ein Weg gilt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Kontrafaktische Erklärung auf einem erfundenen Container-Generator: der billigste Weg zu „kein Langsteher“, fünf Modellklassen, Neu-Training und Wahrheit.
Aufbau und Regeln: DEMO-PLAYBOOK §2. Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import kfk_constants as C
import kfk_evaluation as EV
import kfk_models as M
import kfk_recourse as R
from kfk_layout import dunkles_design, kopf_css
from kfk_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from kfk_texts import DATA_NOTE, LIMITS_MARKDOWN
from kfk_visualization import build_bands, build_logit, build_tab, short

st.set_page_config(page_title="Was müsste anders sein? Wie lange ein Weg gilt – Sebastian Hanisch", layout="wide")


@st.cache_resource(show_spinner=False)
def _fit(model_key, seed):
    return EV.fit_live(model_key, seed)


@st.cache_resource(show_spinner=False)
def _data():
    return EV.load_data()


def _fmt_retrain(key):
    return C.RETRAIN_NAMES[key]


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kfk_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Was müsste anders sein? Wie lange ein Weg gilt")
st.markdown(
    f"""
Eine **kontrafaktische Erklärung** sagt, welche **kleinste erlaubte Änderung** der Merkmale die Entscheidung eines Modells kippt: hier von „Langsteher“ (Verweilzeit über 4 Tage) zu „kein Langsteher“.
Der **Weg** ist die billigste solche Änderung auf einem festen Raster; wegen der kleinen Kosten liegt er oft dicht an der Entscheidungsgrenze, aber nicht immer unmittelbar auf ihr (Messung in der README). Die Demo zeigt, **wie viele Wege nach einem Neu-Training des Modells nicht mehr
gelten**, wie viele in der Wahrheit des Generators nicht gelten, wie viele Wege es gibt und wann es gar keinen gibt, jeweils für **fünf Modellklassen**. {DATA_NOTE}

Wer zuerst ohne Fachbegriffe verstehen will, worum es geht, öffnet „💬 In einfachen Worten“. Weiter unten: die **Experimente 1 bis 3**, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Gezählt werden **Instanzen** (bis zu 25 je Zufalls-Seed),
**Zufalls-Seeds** (je ein Trainingsdatensatz) und **Gruppen** (je 12 Zufalls-Seeds); Anteile stehen als Spanne über fünf Gruppen, nie als Punktzahl aus einer einzigen Ziehung; die Übersichtstabelle am Ende von Experiment 3 nennt den Anteil über alle Instanzen der Messung.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/kontrafaktischer-rat-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Ein Modell sagt zu einem Container „Langsteher“. Die Demo fragt: Was müsste anders sein, damit es „kein Langsteher“ sagt? Die Antwort ist ein **Weg**, die billigste erlaubte Änderung der Merkmale nach einer festgelegten Kostenrechnung. Weil der Weg oft dicht an der Entscheidungsgrenze liegt, kann er ungültig werden, wenn das Modell mit frischen Daten neu gelernt wird. Die Demo zählt, bei wie vielen Wegen das nach einem Neu-Training oder in der Wahrheit des ausgedachten Generators passiert, wie viele Wege es gibt und wann es keinen gibt, jeweils für fünf Modellklassen.

Ein Sicherheitsabstand macht den Weg teurer. Er schützt bei der Regression fast ganz, bei drei Modellklassen teilweise und beim exakt optimalen Baum kaum. Die Aussagen gelten für ausgedachte Container, nicht für einen echten Betrieb.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/kontrafaktischer-rat-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert ein Weg", expanded=True):
    st.markdown(
        """
1. **Modell:** Eine von fünf Modellklassen (logistische Regression, Baum, Boosting, kleines Netz, exakt optimaler Baum) lernt aus 1 000 ausgedachten Containern, ob die Verweilzeit über 4 Tage liegt.
2. **Weg:** Unter allen Rasterpunkten (13 104 strukturell mögliche Kandidaten) wird der **billigste** gesucht, an dem das Modell „kein Langsteher“ sagt (Logit höchstens −Abstand). Nur Merkmale der gewählten **Politik** dürfen sich ändern.
3. **Prüfung:** Ein Zweitmodell derselben Klasse, auf frischen Daten trainiert (**Neu-Training**), und die bekannte Formel des Generators (**Wahrheit je Funktional**, Eingriffs-Lesart) sagen, ob der Weg noch gilt.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Modell und Instanz**")
    modell = st.selectbox("Modellklasse", C.MODEL_OPTIONS, key="modell", format_func=lambda k: M.MODEL_NAMES[k], help="Welche Modellklasse den Weg liefert (Experiment 1, hervorgehoben in Experiment 2 und 3).")
    instanz = st.slider("Instanz", *bounds("instanz"), key="instanz", help="Die k-te der ersten 25 Testzeilen, die das Modell als Langsteher einstuft.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed"), key="seed", step=1, help="Zufalls-Seed des Trainingsdatensatzes (Experiment 1).")
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)
    st.markdown("**Was darf sich ändern**")
    politik = st.selectbox("Politik", C.POLICY_OPTIONS, key="politik", format_func=lambda p: R.POLICY_NAMES[p], help="Liste der veränderbaren Merkmale.")
    abstand = st.select_slider("Sicherheitsabstand (Logit)", C.MARGIN_OPTIONS, key="abstand", format_func=lambda v: EV.fmt(v, 1), help="Verlangt wird Logit höchstens −Abstand, nicht nur höchstens 0.")
    st.markdown("**Neu-Training (Experiment 3)**")
    retrain = st.selectbox("Neu-Training", C.RETRAIN_OPTIONS, key="retrain", format_func=_fmt_retrain, help="Welches zweite Modell den Weg prüft.")
    st.markdown("**Gleich gute Wege (Tab in Experiment 1)**")
    schwelle = st.selectbox("Schwelle (Tage)", C.THRESHOLD_OPTIONS, key="schwelle", help="Klasse 1 = Verweilzeit über dieser Schwelle.")
    tiefe = st.selectbox("Tiefe des Baums", C.DEPTH_OPTIONS, key="tiefe")
    zaehl = st.selectbox("Zählweise", C.COUNT_OPTIONS, key="zaehlweise", format_func=lambda k: C.COUNT_NAMES[k])

sync_query_params({k: st.session_state[k] for k in ("modell", "instanz", "politik", "abstand", "retrain", "schwelle", "tiefe", "zaehlweise", "seed")})

data = _data()

# --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment 1 – Kernfrage: Welcher Weg ist am billigsten?")
with st.spinner("Training und Weg werden gerechnet (rund 1 bis 3 Sekunden; der exakt optimale Baum etwas länger) …"):
    fit = _fit(modell, int(seed))
    way = EV.way_for(fit, int(instanz), politik, float(abstand))
st.caption(
    f"{M.MODEL_NAMES[modell]}, Zufalls-Seed {int(seed)}: Genauigkeit {EV.fmt(100 * fit.accuracy, 1)} % auf {C.N_TEST} Testzeilen; {way.n_instances} Instanzen (als Langsteher eingestufte Testzeilen); "
    f"gezeigt wird Instanz {way.instance}."
)
if modell == "netz" and not fit.model.converged:
    st.warning(f"Das kleine Netz hat das Iterationslimit ({M.NET_MAX_ITER}) erreicht, ohne zu konvergieren; es wird trotzdem verwendet (so auch in der eingefrorenen Messung).")
tab_way, tab_calc, tab_equal = st.tabs(["Der Weg", "Rechenweg", "Gleich gute Wege"])

with tab_way:
    m1, m2, m3 = st.columns(3)
    m1.metric("Kosten des Wegs", "kein Weg" if way.cost is None else EV.fmt(way.cost, 2))
    m2.metric("Geänderte Größen", "-" if not way.changed else ", ".join(way.changed))
    m3.metric("Logit vor → nach", EV.fmt(way.logit0, 2) + " → " + ("-" if way.logit1 is None else EV.fmt(way.logit1, 2)))
    a0, a1 = EV.level_text(way.lev0), (EV.level_text(way.lev1) if way.lev1 else None)
    rows = []
    for name in R.FEATURES:
        rows.append({"Merkmal": name, "Instanz": a0[name], "Ziel": "-" if a1 is None else a1[name], "geändert": "ja" if name in way.changed else ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.plotly_chart(build_logit(way.logit0, way.logit1, way.margin), width="stretch", key="logit_chart")
    p_txt = f"Wahre Wahrscheinlichkeit der Klasse 1 (Formel des Generators): am Start {EV.fmt(way.p_true0, 2)}" + ("" if way.p_true1 is None else f", am Ziel {EV.fmt(way.p_true1, 2)}") + "."
    st.caption(p_txt)
    bands = {"kosten": EV.band(data, modell, politik, float(abstand), "kosten"), "mehrweg": EV.band(data, modell, politik, float(abstand), "mehrweg"),
             "noway": EV.band(data, modell, politik, float(abstand), "noway")}
    kind, text = EV.verdict_way(way, bands)
    getattr(st, kind)(text)

with tab_calc:
    st.markdown(
        "Das **Hauptverfahren** ist die Aufzählung über alle 13 104 Kandidaten (Gleichstand: kleinster Kandidaten-Index). Ein zweiter Rechenweg prüft sie, wo es einen gibt."
    )
    c1, c2 = st.columns(2)
    c1.metric("Aufzählung über das Raster", "kein Weg" if way.enum_cost is None and way.cost is None else EV.fmt(way.cost or 0.0, 4))
    if way.other_name is None:
        c2.metric("Zweiter Rechenweg", "nicht anwendbar")
        st.info("Boosting und kleines Netz sind weder linear noch aus Blättern gebaut: ein ganzzahliges Programm und Blattquader sind hier nicht anwendbar, die Aufzählung ist das Verfahren.")
    else:
        c2.metric(way.other_name, "kein Weg" if way.other_cost is None else EV.fmt(way.other_cost, 4))
        (st.success if way.agree else st.error)("Beide Rechenwege geben dieselben Kosten (auf 10⁻⁶): gleich." if way.agree else "Die Rechenwege weichen ab.")
    if way.cost is not None:
        st.caption(f"{way.n_ties} Kandidat(en) mit genau den Mindestkosten; gewählt ist der mit dem kleinsten Index. Wege innerhalb 10 %: "
                   + "; ".join(" + ".join(w) for w in way.ways_near) + ".")
    if modell in ("baum", "opt"):
        st.markdown(
            "**Randpunkt-Falle:** Wird der Weg als „Quaderrand plus ε“ gebildet, kann die Bibliothek (Vergleich in 32-Bit-Zahlen) den Randpunkt anders einstufen. Hier liegen alle Wege auf Rasterpunkten, "
            "und die Modellvorhersage prüft jeden Weg gegen."
        )
    if modell == "opt":
        fit2 = _fit("opt_g", int(seed))
        w2 = EV.way_for(fit2, int(instanz), politik, float(abstand))
        st.markdown(
            "**Gleichstandsregel des exakt optimalen Baums:** unter allen Bäumen mit dem kleinsten Trainingsfehler gewinnt der mit dem **kleinsten Index in der festen Aufzählungsreihenfolge der Schnitte** "
            "(Vorordnung Wurzel, links, rechts; Blatt zählt −1). Die zweite Regel („größter Index“) zeigt die Spanne."
        )
        st.dataframe(pd.DataFrame([
            {"Regel": "kleinster Index", "Trainingsfehler": int(fit.model.err), "Blätter": fit.model.n_leaves(), "Genauigkeit Test": f"{EV.fmt(100 * fit.accuracy, 1)} %",
             "Kosten des Wegs": "kein Weg" if way.cost is None else EV.fmt(way.cost, 2), "Geänderte Größen": ", ".join(way.changed) or "-"},
            {"Regel": "größter Index", "Trainingsfehler": int(fit2.model.err), "Blätter": fit2.model.n_leaves(), "Genauigkeit Test": f"{EV.fmt(100 * fit2.accuracy, 1)} %",
             "Kosten des Wegs": "kein Weg" if w2.cost is None else EV.fmt(w2.cost, 2), "Geänderte Größen": ", ".join(w2.changed) or "-"},
        ]), hide_index=True, width="stretch")
        st.caption("Gleicher Trainingsfehler, verschiedene Bäume: außerhalb der Trainingszeilen antworten sie verschieden; derselbe Weg kann dort anders ausfallen.")

with tab_equal:
    st.markdown(
        "Eingefroren gerechnet: ein **Baum** (scikit-learn, mindestens 30 Zeilen je Blatt, n = 3 000), Politik A, Abstand 0, bis zu 60 Instanzen je Zufalls-Seed, Wege nur auf Rasterpunkten. "
        "Gezählt wird nach **Blättern** (Rasterpunkte innerhalb 10 % der Mindestkosten in verschiedenen Blättern) oder nach **Mengen geänderter Merkmale**."
    )
    kind, text = EV.verdict_tab(data, int(schwelle), int(tiefe), zaehl)
    rows3 = [{"depth": d, **EV.tab_band(data, int(schwelle), d, zaehl)} for d in C.DEPTH_OPTIONS]
    st.plotly_chart(build_tab(rows3, int(tiefe)), width="stretch", key="tab_chart")
    getattr(st, kind)(text)

# --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment 2 – Kernfrage: Was darf sich ändern?")
st.caption(f"Anteil der Instanzen ohne Weg, Politik {R.POLICY_NAMES[politik]}, Abstand {EV.fmt(float(abstand), 1)}; eingefroren gerechnet (fünf Gruppen zu je 12 Zufalls-Seeds, bis zu 300 Instanzen je Gruppe).")
rows2 = [{"key": k, **EV.band(data, k, politik, float(abstand), "noway")} for k in M.MODELS]
st.plotly_chart(build_bands(rows2, modell, "Instanzen ohne Weg (%)", "Gruppe: "), width="stretch", key="noway_chart")
tbl = []
for k in M.MODELS:
    row = {"Modellklasse": short(k)}
    for p in R.POLICIES:
        b = EV.band(data, k, p, float(abstand), "noway")
        row[f"Politik {p}"] = f"{EV.fmt(b['lo'])} bis {EV.fmt(b['hi'])} %"
    tbl.append(row)
st.dataframe(pd.DataFrame(tbl), hide_index=True, width="stretch")
kind, text = EV.verdict_policy(data, politik, float(abstand))
getattr(st, kind)(text)

# --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🔬 Experiment 3 – Kernfragen: Gilt der Weg nach einem Neu-Training? Gilt er in der Wahrheit?")
st.caption(f"Anteil der Wege (nur Instanzen mit Weg), die nach {C.RETRAIN_NAMES[retrain]} nicht mehr gelten; Politik {R.POLICY_NAMES[politik]}, Abstand {EV.fmt(float(abstand), 1)}; eingefroren gerechnet.")
rows_inv = [{"key": k, **EV.band(data, k, politik, float(abstand), "inv", retrain)} for k in M.MODELS]
st.plotly_chart(build_bands(rows_inv, modell, "Wege, die nicht mehr gelten (%)", "Gruppe: "), width="stretch", key="inv_chart")
kind, text = EV.verdict_retrain(data, modell, politik, float(abstand), retrain)
getattr(st, kind)(text)
lo, hi = EV.draw_spread(data, modell, politik, float(abstand))
st.caption(f"Streuung der Ziehung selbst: einzelne Ziehungen gleicher Datenmenge (R2 bis R11) geben für {short(modell)} {EV.fmt(lo, 1)} bis {EV.fmt(hi, 1)} % über alle Instanzen.")
kind, text = EV.verdict_margin(data, politik, retrain)
getattr(st, kind)(text)

st.markdown("**Zweite Anzeige: Gilt der Weg in der Wahrheit?**")
rows_w = [{"key": k, **EV.band(data, k, politik, float(abstand), "wahr")} for k in M.MODELS]
st.plotly_chart(build_bands(rows_w, modell, "Wege, die in der Wahrheit nicht gelten (%)", "Gruppe: "), width="stretch", key="wahr_chart")
kind, text = EV.verdict_truth(data, modell, politik, float(abstand))
getattr(st, kind)(text)

summary = []
for k in M.MODELS + ("opt_g",):
    b = lambda f, r="R1": EV.band(data, k, politik, float(abstand), f, r)  # noqa: E731
    summary.append({
        "Modellklasse": M.MODEL_NAMES.get(k, "Exakt optimaler Baum, zweite Regel (größter Index)"),
        "Genauigkeit Test": f"{EV.fmt(100 * EV.accuracy(data, k), 1)} %",
        "kein Weg": f"{EV.fmt(b('noway')['all'], 1)} %", "Kosten (Median)": EV.fmt(b("kosten")["all"], 2),
        "ungültig R1": f"{EV.fmt(b('inv')['all'], 1)} %", "ungültig R2 bis R11 (Mittel)": f"{EV.fmt(b('inv', 'R2-R11')['all'], 1)} %",
        "in der Wahrheit": f"{EV.fmt(b('wahr')['all'], 1)} %", "ändert das Gewicht": f"{EV.fmt(b('rausch')['all'], 1)} %",
    })
st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
st.caption("Alle Zahlen über alle Instanzen der Messung (60 Zufalls-Seeds); die letzte Zeile ist dieselbe Frage mit der zweiten Gleichstandsregel des exakt optimalen Baums.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(LIMITS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/kontrafaktischer-rat-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind eine Instanz $x_0$ mit Ebenen $(a_0, k_0, z_0, w_0, v_0, g_0)$ (Art, Kunde, Zoll, Wochenende, Vorlauf, Gewicht), ein Modell mit Logit $f$ (Klasse 1 = länger als 4 Tage, $f>0$), eine Politik $P$ (Menge der veränderbaren Merkmale) und ein Sicherheitsabstand $m \ge 0$. Gesucht ist der billigste Rasterpunkt $x$, an dem das Modell „kein Langsteher“ sagt. Einige Buchstaben haben in verschiedenen Formeln verschiedene Bedeutungen; die Tabelle nennt jede Verwendung einzeln.

| Symbol | Bedeutung |
|---|---|
| $x_0, x$ | Instanz und Zielpunkt auf dem Raster (13 104 zulässige Kandidaten) |
| $f(x)$ | Logit der Klasse 1 (bei Bäumen: Blattanteil, auf $[10^{-4}, 1-10^{-4}]$ begrenzt) |
| $m$ | Sicherheitsabstand, $\varepsilon = 10^{-6}$ |
| $P$ | Politik: Menge der veränderbaren Merkmale |
| $\sigma_v$ | Standardabweichung der von 0 verschiedenen Vorlaufzeiten im Training |
| $c(x_0, x)$ | gesetzte Kosten |
| $a, a_0$ | Art des Containers (Ebene; $a_0$ die der Instanz) in der Kostenformel und in der Summe $\sum_a$ des ganzzahligen Programms |
| $k_0, z_0, v_0, g_0$ | Ebenen der Instanz für Kunde, Zoll, Vorlauf (Tage) und Gewicht; $k, z, v, g$ sind die Ebenen des Zielpunkts |
| $w, w_0$ | Ebene Wochenende (Kostenformel), nicht zu verwechseln mit den Gewichten $w_a, w_k, w_z, w_w, w_v, w_g$ |
| $b$, $w_a, w_k, w_z, w_w, w_v, w_g$ | Achsenabschnitt und Gewichte des linearen Modells im Logit $f$; der Index nennt das Merkmal: Art, Kunde, Zoll, Wochenende, Vorlauf (je Tag), Gewicht |
| $\alpha_a, \kappa_k, \zeta, \omega$ | Binärvariablen des ganzzahligen Programms für die Art $a$, den Kunden $k$, den Zoll und das Wochenende (1 = gewählt) |
| $\nu_l, \gamma_l$ | Binärvariablen für die Vorlaufstufe $l$ (Vorlauf $l/2$ Tage) und die Gewichtsstufe $l$ (Gewicht $(l-6)/2$) |
| $a$ (letzte Formel) | etwas anderes: Obergrenze des Spielraums $u$ über der Grenze |
| $s$ (letzte Formel) | Streuung der Verschiebung $\Delta$ der Grenze (im Generator ist $s$ die Streuung der log-Verweilzeit, nicht in den Formeln der App) |
| $u$ | Spielraum des Weges über der Grenze, $u \sim U(0, a)$ |
| $\Delta$ | Verschiebung der Grenze durch das neue Modell, $\Delta \sim N(0, s^2)$ |
| $t$ | $t = a/s$ |
| $\Phi, \varphi$ | Verteilungsfunktion und Dichte der Standardnormalverteilung |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/kontrafaktischer-rat-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Der Weg** (Aufzählung über alle zulässigen Kandidaten $x$):

$$\min_{x}\; c(x_0, x) \quad \text{mit} \quad f(x) \le -m - \varepsilon,\;\; x_j = (x_0)_j \;\text{für}\; j \notin P,\;\; x \;\text{strukturell zulässig.}$$

$$c(x_0, x) = [a \ne a_0] + [k \ne k_0] + [z \ne z_0] + [w \ne w_0] + \frac{|v - v_0|}{\sigma_v} + |g - g_0|.$$

Strukturell zulässig heißt: Zoll nur bei Import; Vorlauf 0 bei Import und Leer, mindestens 1 Tag bei Export, mindestens 0,5 Tage bei Umschlag (Obergrenze 10 Tage für beide; der Generator zieht Umschlag nur bis 6 Tage). Bei Gleichstand gewinnt der kleinste Kandidaten-Index.

**Ganzzahliges Programm (lineares Modell, zweiter Rechenweg).** Binärvariablen $\alpha_a$ (4 Arten), $\kappa_k$ (12 Kunden), $\zeta, \omega \in \{0,1\}$, $\nu_l$ (21 Vorlaufstufen von 0 bis 10 Tage), $\gamma_l$ (13 Gewichtsstufen von −3 bis 3):

$$\sum_a \alpha_a = 1,\quad \sum_k \kappa_k = 1,\quad \sum_l \nu_l = 1,\quad \sum_l \gamma_l = 1,\quad b + \sum_a w_a \alpha_a + \sum_k w_k \kappa_k + w_z \zeta + w_w \omega + \sum_l w_v \tfrac{l}{2} \nu_l + \sum_l w_g \tfrac{l-6}{2} \gamma_l \le -m - \varepsilon.$$

Dazu die Strukturbedingungen (Zoll nur bei Import, Vorlaufstufen nach Art) und die Festlegung der nicht veränderbaren Merkmale; die Kosten sind linear in den Binärvariablen. Für Bäume gilt der **Weg über Blattquader**:
je Blatt mit Logit $\le -m-\varepsilon$ die billigste erlaubte Zeile im Quader, das Minimum über alle Blätter, ohne die Vorhersage des Baums zu rufen.

**Exakt optimaler Baum.** Dynamische Programmierung über Teilmengen der Trainingszeilen: der kleinste Fehler über alle Bäume der Tiefe höchstens 3 mit Schnitten aus dem festen Raster (47 Schnitte, mindestens 10 Zeilen je Blatt). Bei Gleichstand gewinnt der Baum mit der lexikografisch kleinsten Vorordnung der Schnittindizes.

**Warum ein Weg nahe an der Grenze leicht ungültig wird (Modell, kein Messwert).** Liegt der Weg mit Spielraum $u \sim U(0, a)$ über der Grenze und verschiebt das neue Modell die Grenze um $\Delta \sim N(0, s^2)$, so ist der Weg ungültig, wenn $\Delta > u$:

$$P(\text{ungültig}) = \frac{s}{a}\left(t\,\Phi(-t) + \varphi(0) - \varphi(t)\right), \qquad t = \frac{a}{s}.$$

Für $a = 0{,}3$, $s = 0{,}2$ ist das $0{,}2464$; für $a \to 0$ geht es gegen $1/2$. Die Formel gilt nur für eine Verschiebung der Grenze bei gleichverteiltem Spielraum. Der gemessene Anteil in Experiment 3 (nach R1) liegt bei Politik A in jeder Gruppe und bei allen drei Abständen zwischen 0 und 44 %, also zwischen 0 und $1/2$; bei den engeren Politiken B und C kann er beim exakt optimalen Baum $1/2$ überschreiten (je Gruppe bis 65 %).

**Literatur** (Stufe der Lektüre: A = Zusammenfassung gelesen, sonst nicht gelesen). Wachter, Mittelstadt, Russell 2018 (kleinste Änderung der Eingabe, die das Ergebnis ändert; A); Ustun, Spangher, Liu 2019 (veränderbare Merkmale, ganzzahlige Programme; A);
Rawal, Kamar, Lakkaraju (Modellupdates machen Wege ungültig; A); Upadhyay u. a. 2021 (robuste Wege; A); Pawelczyk u. a. 2020 (Modell-Vielfachheit; A); Mothilal u. a. (DiCE, Vielfalt; A); Karimi u. a. (Survey; A, kausale Fassung nicht gelesen).
Dies ist keine neue Methode: alles ist Lehrbuch oder Forschung; neu ist die nachrechenbare Vorführung mit Zähleinheit, fünf Modellklassen und einer Wahrheit (Vorbilder: XAI-Bench, OpenXAI, Hooker).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
