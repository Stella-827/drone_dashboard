"""
Motor Characterization Page
Mirrors: Kyle_&_Jose_Motor_Characterization_V1 / Optimizer V3 Step 2

Fixes vs v2:
- ALL motors (built-in CSV + manually added) are stored in a unified store
- Every motor has a working delete button
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL
import dash_bootstrap_components as dbc

from data_engine import load_motor_csv, compute_motor_params

dash.register_page(__name__, path="/motor", name="Motor", title="Motor Characterization")

COLORS = px.colors.qualitative.Plotly


def _initial_motor_store():
    """Load all motors from CSV into the store on first visit."""
    df     = load_motor_csv()
    params = compute_motor_params(df)
    return {name: {**p, "source": "file"} for name, p in params.items()}


# ── Chart builders ────────────────────────────────────────────────────────────

def build_params_table(store, selected):
    rows = [{"Motor": n,
             "Source": store[n].get("source", "manual"),
             "Mass (mg)": f"{store[n]['mass_mg']:.0f}",
             "R (Ω)": f"{store[n]['R']:.3f}",
             "K (Nm/A)": f"{store[n]['K']:.6f}"}
            for n in selected if n in store]
    if not rows:
        return html.Div("No motors selected.")
    return dbc.Table(
        [html.Thead(html.Tr([html.Th(k) for k in rows[0]]))] +
        [html.Tbody([html.Tr([html.Td(r[k]) for k in r]) for r in rows])],
        striped=True, bordered=True, hover=True, size="sm", className="text-white")


def build_comparison(store, selected):
    names  = [n for n in selected if n in store]
    K_vals = [store[n]["K"]       for n in names]
    R_vals = [store[n]["R"]       for n in names]
    mass   = [store[n]["mass_mg"] for n in names]

    fig = make_subplots(rows=1, cols=3, subplot_titles=["K (Nm/A)", "R (Ω)", "Mass (mg)"])
    c   = COLORS[:len(names)]
    fig.add_trace(go.Bar(x=names, y=K_vals, marker_color=c, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=R_vals, marker_color=c, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=mass,   marker_color=c, showlegend=False), row=1, col=3)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title="Motor Parameters Comparison",
                      margin=dict(l=50, r=10, t=70, b=100), height=420)
    fig.update_xaxes(tickangle=-40)
    return fig


def build_efficiency(store, selected, V=3.7):
    """
    Motor efficiency curve from Optimizer V3 Step 2 model:
      ω = (V − I·R) / K
      P_shaft = K·ω·I
      η = P_shaft / (V·I)
    """
    fig = go.Figure()
    for i, name in enumerate([n for n in selected if n in store]):
        K = store[name]["K"]
        R = max(store[name]["R"], 0.001)
        I = np.linspace(0, (V / R) * 0.95, 400)
        omega   = (V - I * R) / K
        P_shaft = K * omega * I
        P_elec  = V * I
        eta     = np.where(P_elec > 0, P_shaft / P_elec, 0)
        fig.add_trace(go.Scatter(
            x=P_elec * 1000, y=eta * 100,
            name=name, line=dict(color=COLORS[i % len(COLORS)], width=2)))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title=f"Motor Efficiency vs Electrical Power  (V = {V:.1f} V)",
        xaxis_title="Electrical Power (mW)", yaxis_title="Efficiency (%)",
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50))
    return fig


def build_torque_speed(store, selected, V=3.7):
    """
    Torque–speed from Optimizer V3 Step 2:
      τ = K·(V − K·ω) / R
    """
    fig = go.Figure()
    for i, name in enumerate([n for n in selected if n in store]):
        K = store[name]["K"]
        R = max(store[name]["R"], 0.001)
        omega  = np.linspace(0, V / K, 400)
        torque = np.clip(K * (V - K * omega) / R, 0, None)
        rpm    = omega * 60 / (2 * np.pi)
        fig.add_trace(go.Scatter(
            x=rpm, y=torque * 1000,
            name=name, line=dict(color=COLORS[i % len(COLORS)], width=2)))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title=f"Torque vs RPM  (V = {V:.1f} V)",
        xaxis_title="RPM", yaxis_title="Torque (mNm)",
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50))
    return fig


def motor_list_ui(store):
    """Every motor gets a delete button."""
    if not store:
        return html.Div("No motors saved yet.", className="text-muted small")
    items = []
    for name in sorted(store):
        p      = store[name]
        source = p.get("source", "manual")
        tag    = "📄" if source == "file" else "🖊"
        items.append(dbc.ListGroupItem(
            dbc.Row([
                dbc.Col(html.Div([
                    html.Span(f"{tag} {name}", className="small fw-bold"),
                    html.Br(),
                    html.Span(f"K={p['K']:.5f}  R={p['R']:.2f}Ω  {p['mass_mg']:.0f}mg",
                              className="small text-muted"),
                ]), width=9),
                dbc.Col(
                    dbc.Button("✕", id={"type": "motor-delete-btn", "index": name},
                               color="danger", size="sm", className="py-0 px-2"),
                    width=3, className="text-end"),
            ], align="center"),
            className="bg-transparent border-secondary text-white py-1",
        ))
    return dbc.ListGroup(items, flush=True)


# ── Layout ────────────────────────────────────────────────────────────────────
layout = dbc.Container(fluid=True, children=[
    dcc.Store(id="motor-store", data=None),
    dcc.Store(id="motor-init-trigger", data=0),

    html.H3("⚙️ Motor Characterization", className="mb-4 mt-2 text-info"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Nominal Voltage (V)"),
                dbc.CardBody(dcc.Slider(
                    id="motor-voltage-slider", min=2.0, max=5.0, step=0.1, value=3.7,
                    marks={v: str(v) for v in [2, 3, 3.7, 4, 5]},
                    tooltip={"placement": "bottom"})),
            ], className="mb-3"),

            dbc.Accordion([
                dbc.AccordionItem(title="➕  Add Motor", children=[
                    dbc.Label("Motor Name", className="small mb-0"),
                    dbc.Input(id="motor-name", placeholder="e.g. MyMotor 300mg", size="sm", className="mb-2"),
                    dbc.Label("Mass (mg)", className="small mb-0"),
                    dbc.Input(id="motor-mass", type="number", placeholder="e.g. 300", size="sm", className="mb-2"),
                    dbc.Label("ESR / Resistance (Ω)", className="small mb-0"),
                    dbc.Input(id="motor-R", type="number", placeholder="e.g. 4.6", size="sm", className="mb-2"),
                    dbc.Label("K — Motor Constant (Nm/A)", className="small mb-0"),
                    dbc.Input(id="motor-K", type="number", placeholder="e.g. 0.00035", size="sm", className="mb-2"),
                    html.P("K is computed from K = Σ(ω·back_EMF) / Σ(ω²) in the Colab. "
                           "Enter the value from your characterization.",
                           className="small text-muted"),
                    dbc.Button("Save Motor", id="motor-save-btn", color="success", size="sm", className="w-100 mt-1"),
                    html.Div(id="motor-save-status", className="small mt-2"),
                ]),
            ], start_collapsed=True, className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Saved Motors  (📄 file · 🖊 manual)"),
                dbc.CardBody(html.Div(id="motor-list-ui")),
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Plot Selection"),
                dbc.CardBody(dcc.Checklist(
                    id="motor-selector", options=[], value=[],
                    labelStyle={"display": "block", "marginBottom": "4px"},
                    inputStyle={"marginRight": "6px"})),
            ]),
        ], md=3),

        dbc.Col([
            dbc.Tabs([
                dbc.Tab(html.Div(id="motor-param-table", className="mt-3"),        label="Parameters"),
                dbc.Tab(dcc.Graph(id="motor-compare-fig",    config={"displayModeBar": False}), label="Comparison"),
                dbc.Tab(dcc.Graph(id="motor-efficiency-fig", config={"displayModeBar": False}), label="Efficiency"),
                dbc.Tab(dcc.Graph(id="motor-torque-fig",     config={"displayModeBar": False}), label="Torque–Speed"),
            ]),
        ], md=9),
    ]),
])


# ── Init ──────────────────────────────────────────────────────────────────────
@callback(
    Output("motor-store", "data", allow_duplicate=True),
    Input("motor-init-trigger", "data"),
    State("motor-store", "data"),
    prevent_initial_call='initial_duplicate',
)
def init_store(_, existing):
    if existing:
        return dash.no_update
    return _initial_motor_store()


# ── Save motor ────────────────────────────────────────────────────────────────
@callback(
    Output("motor-store",       "data",    allow_duplicate=True),
    Output("motor-save-status", "children"),
    Output("motor-name",        "value"),
    Output("motor-mass",        "value"),
    Output("motor-R",           "value"),
    Output("motor-K",           "value"),
    Input("motor-save-btn",     "n_clicks"),
    State("motor-name",         "value"),
    State("motor-mass",         "value"),
    State("motor-R",            "value"),
    State("motor-K",            "value"),
    State("motor-store",        "data"),
    prevent_initial_call=True,
)
def save_motor(_, name, mass, R, K, store):
    store = dict(store or {})
    nd    = dash.no_update
    if not name or not name.strip():
        return store, "⚠ Name required.", nd, nd, nd, nd
    if K is None or R is None:
        return store, "⚠ K and R are required.", nd, nd, nd, nd
    store[name.strip()] = {
        "mass_mg": float(mass or 0),
        "R":       float(R),
        "K":       float(K),
        "source":  "manual",
    }
    return store, f"✓ Saved '{name.strip()}'", "", None, None, None


# ── Delete motor ──────────────────────────────────────────────────────────────
@callback(
    Output("motor-store", "data", allow_duplicate=True),
    Input({"type": "motor-delete-btn", "index": ALL}, "n_clicks"),
    State("motor-store", "data"),
    prevent_initial_call=True,
)
def delete_motor(_, store):
    store     = dict(store or {})
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        store.pop(triggered.get("index", ""), None)
    return store


# ── Refresh list + selector ───────────────────────────────────────────────────
@callback(
    Output("motor-list-ui",  "children"),
    Output("motor-selector", "options"),
    Output("motor-selector", "value"),
    Input("motor-store",     "data"),
)
def refresh_list(store):
    store     = store or {}
    all_names = sorted(store.keys())
    return motor_list_ui(store), [{"label": n, "value": n} for n in all_names], all_names


# ── Update charts ─────────────────────────────────────────────────────────────
@callback(
    Output("motor-param-table",    "children"),
    Output("motor-compare-fig",    "figure"),
    Output("motor-efficiency-fig", "figure"),
    Output("motor-torque-fig",     "figure"),
    Input("motor-selector",        "value"),
    Input("motor-voltage-slider",  "value"),
    State("motor-store",           "data"),
)
def update_charts(selected, V, store):
    empty = go.Figure()
    empty.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    store    = store or {}
    selected = [s for s in (selected or []) if s in store]
    if not selected:
        return "No motors selected.", empty, empty, empty
    return (
        build_params_table(store, selected),
        build_comparison(store, selected),
        build_efficiency(store, selected, float(V or 3.7)),
        build_torque_speed(store, selected, float(V or 3.7)),
    )
