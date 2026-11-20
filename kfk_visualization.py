"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import kfk_constants as C
import kfk_models as M


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def short(key):
    return M.MODEL_NAMES[key].split(" (")[0]


def build_logit(logit0, logit1, margin):
    """Logit der Instanz vor und nach dem Weg gegen die verlangte Grenze (-Abstand)."""
    labels, vals, cols = ["vor dem Weg"], [logit0], [C.COLOR_BAD if logit0 > 0 else C.COLOR_MODEL]
    if logit1 is not None:
        labels.append("nach dem Weg")
        vals.append(logit1)
        cols.append(C.COLOR_MODEL)
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=cols, hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.add_vline(x=-margin, line_dash="dash", line_color="gray", annotation_text="verlangte Grenze", annotation_position="top")
    fig.update_xaxes(title_text="Logit der Klasse 1 (größer null: Langsteher)", zeroline=True)
    return _base(fig, 200)


def build_bands(rows, selected, ytitle, title_hover):
    """Je Modellklasse: Anteil über alle Instanzen (Balken) und die fünf Gruppen (Punkte); die gewählte Klasse ist farbig.
    rows: Liste {'key', 'all', 'per_group'}."""
    fig = go.Figure()
    xs = [short(r["key"]) for r in rows]
    fig.add_trace(go.Bar(x=xs, y=[r["all"] for r in rows], marker_color=[C.COLOR_MODEL if r["key"] == selected else C.COLOR_OTHER for r in rows], name="alle Instanzen",
                         hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    px, py = [], []
    for r in rows:
        for v in r["per_group"]:
            px.append(short(r["key"]))
            py.append(v)
    fig.add_trace(go.Scatter(x=px, y=py, mode="markers", marker=dict(color=C.COLOR_POINT, size=9, symbol="diamond"), name="eine Gruppe (12 Zufalls-Seeds)",
                             hovertemplate=title_hover + "%{y:.1f} %<extra></extra>"))
    fig.update_yaxes(title_text=ytitle, rangemode="tozero")
    return _base(fig)


def build_tab(rows, selected_depth):
    """Tab „Gleich gute Wege“: je Tiefe Anteil über alle Instanzen und die Gruppen."""
    fig = go.Figure()
    xs = [f"Tiefe {r['depth']}" for r in rows]
    fig.add_trace(go.Bar(x=xs, y=[r["all"] for r in rows], marker_color=[C.COLOR_MODEL if r["depth"] == selected_depth else C.COLOR_OTHER for r in rows], name="alle Instanzen"))
    fig.add_trace(go.Scatter(x=[f"Tiefe {r['depth']}" for r in rows for _ in r["per_group"]], y=[v for r in rows for v in r["per_group"]], mode="markers",
                             marker=dict(color=C.COLOR_POINT, size=9, symbol="diamond"), name="eine Gruppe (6 Zufalls-Seeds)"))
    fig.update_yaxes(title_text="Instanzen mit mehr als einem Weg (%)", range=[0, 100])
    return _base(fig)
