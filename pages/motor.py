"""
Motor Characterization Page — manual entry, no CSV upload
"""

import json
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

# ── Field definitions for the "Add Motor" form ───────────────────────────────
MOTOR_FIELDS = [
    ("motor-name",     "Motor Name",           "text",   "e.g. MicroDC - subgram"),
    ("motor-mass",     "Mass (mg)",             "number", "e.g. 283"),
    ("motor-R",        "ESR / Resistance (Ω)",  "number", "e.g. 13.5"),
    ("motor-K",        "K — Motor Constant (Nm/A)", "number", "e.g. 0.00035"),
    ("motor-voltage",  "Nominal Voltage (V)",   "number", "e.g. 3.7"),
]


# ── Chart builders ────────────────────────────────────────────────────────────
def build_motor_params_table(motor_params: dict) -> dbc.Table:
    rows = [{"Motor": n, "Mass (mg)": f"{p['mass_mg']:.0f}",
             "R (Ω)": f"{p['R']:.3f}", "K (Nm/A)": f"{p['K']:.6f}"}
            for n, p in motor_params.items()]
    if not rows:
        return html.Div("No motors selected.")
    return dbc.Table(
        [html.Thead(html.Tr([html.Th(k) for k in rows[0]]))] +
        [html.Tbody([html.Tr([html.Td(r[k]) for k in r]) for r in rows])],
        striped=True, bordered=True, hover=True, size="sm", className="text-white",
    )


def build_K_comparison(motor_params: dict, selected: list) -> go.Figure:
    names  = [n for n in selected if n in motor_params]
    K_vals = [motor_params[n]["K"] for n in names]
    R_vals = [motor_params[n]["R"] for n in names]
    mass   = [motor_params[n]["mass_mg"] for n in names]
    fig = make_subplots(rows=1, cols=3, subplot_titles=["K (Nm/A)", "R (Ω)", "Mass (mg)"])
    fig.add_trace(go.Bar(x=names, y=K_vals, marker_color=COLORS[:len(names)], showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=R_vals, marker_color=COLORS[:len(names)], showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=mass,   marker_color=COLORS[:len(names)], showlegend=False), row=1, col=3)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title="Motor Parameters Comparison",
                      margin=dict(l=50, r=10, t=70, b=100), height=420)
    fig.update_xaxes(tickangle=-40)
    return fig


def build_efficiency_fig(motor_params: dict, selected: list, V: float = 3.7) -> go.Figure:
    fig = go.Figure()
    for i, name in enumerate([n for n in selected if n in motor_params]):
        p = motor_params[name]
        K = p["K"]; R = max(p["R"], 0.001)
        I = np.linspace(0, (V / R) * 0.95, 300)
        omega   = (V - I * R) / K
        P_shaft = K * omega * I
        P_elec  = V * I
        eta     = np.where(P_elec > 0, P_shaft / P_elec, 0)
        fig.add_trace(go.Scatter(x=P_elec * 1000, y=eta * 100,
                                 name=name, line=dict(color=COLORS[i % len(COLORS)], width=2)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title=f"Motor Efficiency vs Electrical Power  (V = {V:.1f} V)",
                      xaxis_title="Electrical Power (mW)", yaxis_title="Efficiency (%)",
                      legend=dict(orientation="v", x=1.01),
                      margin=dict(l=50, r=10, t=50, b=50))
    return fig


def build_torque_speed_fig(motor_params: dict, selected: list, V: float = 3.7) -> go.Figure:
    fig = go.Figure()
    for i, name in enumerate([n for n in selected if n in motor_params]):
        p = motor_params[name]
        K = p["K"]; R = max(p["R"], 0.001)
        omega = np.linspace(0, V / K, 300)
        torque = np.clip(K * (V - K * omega) / R, 0, None)
        rpm    = omega * 60 / (2 * np.pi)
        fig.add_trace(go.Scatter(x=rpm, y=torque * 1000,
                                 name=name, line=dict(color=COLORS[i % len(COLORS)], width=2)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title=f"Torque vs RPM  (V = {V:.1f} V)",
                      xaxis_title="RPM", yaxis_title="Torque (mNm)",
                      legend=dict(orientation="v", x=1.01),
                      margin=dict(l=50, r=10, t=50, b=50))
    return fig


def motor_list_ui(motor_params: dict) -> html.Div:
    """Render the saved motors list with individual delete buttons."""
    if not motor_params:
        return html.Div("No motors saved yet.", className="text-muted small")
    items = []
    for name in motor_params:
        items.append(
            dbc.ListGroupItem(
                dbc.Row([
                    dbc.Col(html.Span(name, className="small"), width=9),
                    dbc.Col(
                        dbc.Button("✕", id={"type": "motor-delete-btn", "index": name},
                                   color="danger", size="sm", className="py-0 px-2"),
                        width=3, className="text-end"
                    ),
                ], align="center"),
                className="bg-transparent border-secondary text-white py-1",
            )
        )
    return dbc.ListGroup(items, flush=True)


# ── Layout ────────────────────────────────────────────────────────────────────
layout = dbc.Container(fluid=True, children=[
    dcc.Store(id="motor-custom-store", data={}),  # {name: {mass_mg, R, K}}

    html.H3("⚙️ Motor Characterization", className="mb-4 mt-2 text-info"),

    dbc.Row([
        # ── Left panel ───────────────────────────────────────────────────────
        dbc.Col([
            # Voltage slider
            dbc.Card([
                dbc.CardHeader("Nominal Voltage (V)"),
                dbc.CardBody(dcc.Slider(id="motor-voltage-slider", min=2.0, max=5.0,
                                        step=0.1, value=3.7,
                                        marks={v: str(v) for v in [2, 3, 3.7, 4, 5]},
                                        tooltip={"placement": "bottom"})),
            ], className="mb-3"),

            # Add motor accordion
            dbc.Accordion([
                dbc.AccordionItem(
                    title="➕  Add Motor",
                    children=[
                        dbc.Row([
                            dbc.Col([
                                dbc.Label(label, className="small mb-0"),
                                dbc.Input(id=field_id, type=ftype,
                                          placeholder=ph,
                                          className="mb-2",
                                          size="sm"),
                            ], width=12)
                            for field_id, label, ftype, ph in MOTOR_FIELDS
                        ]),
                        dbc.Button("Save Motor", id="motor-save-btn",
                                   color="success", size="sm", className="w-100 mt-1"),
                        html.Div(id="motor-save-status", className="small mt-2"),
                    ],
                ),
            ], start_collapsed=True, className="mb-3"),

            # Saved motors list
            dbc.Card([
                dbc.CardHeader("Saved Motors"),
                dbc.CardBody(html.Div(id="motor-list-ui")),
            ], className="mb-3"),

            # Select which motors to plot
            dbc.Card([
                dbc.CardHeader("Plot Selection"),
                dbc.CardBody(dcc.Checklist(
                    id="motor-selector", options=[], value=[],
                    labelStyle={"display": "block", "marginBottom": "4px"},
                    inputStyle={"marginRight": "6px"},
                )),
            ]),
        ], md=3),

        # ── Right panel ──────────────────────────────────────────────────────
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(html.Div(id="motor-param-table", className="mt-3"), label="Parameters"),
                dbc.Tab(dcc.Graph(id="motor-compare-fig",     config={"displayModeBar": False}), label="Comparison"),
                dbc.Tab(dcc.Graph(id="motor-efficiency-fig",  config={"displayModeBar": False}), label="Efficiency"),
                dbc.Tab(dcc.Graph(id="motor-torquespeed-fig", config={"displayModeBar": False}), label="Torque–Speed"),
            ]),
        ], md=9),
    ]),
])


# ── Save a new motor ──────────────────────────────────────────────────────────
@callback(
    Output("motor-custom-store", "data"),
    Output("motor-save-status",  "children"),
    Output("motor-name",         "value"),
    Output("motor-mass",         "value"),
    Output("motor-R",            "value"),
    Output("motor-K",            "value"),
    Output("motor-voltage",      "value"),
    Input("motor-save-btn",      "n_clicks"),
    State("motor-name",          "value"),
    State("motor-mass",          "value"),
    State("motor-R",             "value"),
    State("motor-K",             "value"),
    State("motor-voltage",       "value"),
    State("motor-custom-store",  "data"),
    prevent_initial_call=True,
)
def save_motor(_, name, mass, R, K, V, store):
    store = store or {}
    if not name or name.strip() == "":
        return store, "⚠ Motor name is required.", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    if K is None or R is None:
        return store, "⚠ K and R are required.", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    store[name.strip()] = {"mass_mg": float(mass or 0), "R": float(R), "K": float(K)}
    return store, f"✓ Saved '{name.strip()}'", "", None, None, None, None


# ── Delete a motor ────────────────────────────────────────────────────────────
@callback(
    Output("motor-custom-store", "data", allow_duplicate=True),
    Input({"type": "motor-delete-btn", "index": ALL}, "n_clicks"),
    State("motor-custom-store", "data"),
    prevent_initial_call=True,
)
def delete_motor(n_clicks_list, store):
    store = store or {}
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict) and triggered.get("type") == "motor-delete-btn":
        name = triggered["index"]
        store.pop(name, None)
    return store


# ── Rebuild selector + list UI whenever store changes ────────────────────────
@callback(
    Output("motor-list-ui",   "children"),
    Output("motor-selector",  "options"),
    Output("motor-selector",  "value"),
    Input("motor-custom-store", "data"),
)
def refresh_motor_list(store):
    store = store or {}
    df_base = load_motor_csv()
    base_params = compute_motor_params(df_base)
    all_params  = {**base_params, **store}
    list_ui = motor_list_ui(all_params)
    opts  = [{"label": n, "value": n} for n in sorted(all_params)]
    value = sorted(all_params)
    return list_ui, opts, value


# ── Update charts ─────────────────────────────────────────────────────────────
@callback(
    Output("motor-param-table",     "children"),
    Output("motor-compare-fig",     "figure"),
    Output("motor-efficiency-fig",  "figure"),
    Output("motor-torquespeed-fig", "figure"),
    Input("motor-selector",         "value"),
    Input("motor-voltage-slider",   "value"),
    State("motor-custom-store",     "data"),
)
def update_charts(selected, V, store):
    df_base = load_motor_csv()
    base_params = compute_motor_params(df_base)
    all_params  = {**base_params, **(store or {})}
    selected    = [s for s in (selected or []) if s in all_params]

    empty = go.Figure()
    empty.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")

    if not selected:
        return "No motors selected.", empty, empty, empty

    return (
        build_motor_params_table({n: all_params[n] for n in selected}),
        build_K_comparison(all_params, selected),
        build_efficiency_fig(all_params, selected, V or 3.7),
        build_torque_speed_fig(all_params, selected, V or 3.7),
    )
