"""
Thrust / Propeller Characterization Page
Mirrors: Optimizer V3 Steps 1 & 3

Fixes vs v2:
- Built-in prop_lift_tests entries are loaded into the store and are deletable
- N/W surface fit equation verified against Colab Step 3
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL
import dash_bootstrap_components as dbc

from data_engine import load_thrust_csv, fit_thrust_model

dash.register_page(__name__, path="/thrust", name="Thrust", title="Thrust Characterization")

COLORS = px.colors.qualitative.Plotly


def _initial_thrust_store():
    """Load built-in prop_lift_tests CSV into the store."""
    df = load_thrust_csv()
    if df.empty:
        return []
    records = df.to_dict("records")
    for r in records:
        r["source"] = "file"
    return records


def _store_to_df(store):
    if not store:
        return pd.DataFrame()
    return pd.DataFrame(store)


# ── Chart builders ────────────────────────────────────────────────────────────

def build_NW_vs_angle(df):
    fig = go.Figure()
    if df.empty or "flexure_angle_deg" not in df.columns:
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          title="No data — add prop test entries to see this chart")
        return fig
    for i, (angle, grp) in enumerate(df.groupby("flexure_angle_deg")):
        fig.add_trace(go.Scatter(
            x=grp["prop_length_mm"], y=grp["N_per_W"],
            mode="markers", name=f"{angle}°",
            marker=dict(color=COLORS[i % len(COLORS)], size=8)))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title="N/W vs Prop Length by Flexure Angle  (Optimizer V3 Step 1)",
        xaxis_title="Prop Length (mm)", yaxis_title="N/W (N/W)",
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50))
    return fig


def build_thrust_vs_power(df):
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        return fig
    col = "motor_type" if "motor_type" in df.columns else "prop_width_mm"
    for i, (mt, grp) in enumerate(df.groupby(col)):
        fig.add_trace(go.Scatter(
            x=grp["power_W"] * 1000, y=grp["thrust_N"] * 1000,
            mode="markers", name=str(mt),
            marker=dict(color=COLORS[i % len(COLORS)], size=8, opacity=0.8)))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title="Thrust vs Electrical Power",
        xaxis_title="Power (mW)", yaxis_title="Thrust (mN)",
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50))
    return fig


def build_NW_scatter(df):
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        return fig
    col = "motor_type" if "motor_type" in df.columns else "prop_width_mm"
    fig = px.scatter(df, x="prop_length_mm", y="N_per_W", color=col,
                     hover_data=df.columns.tolist(),
                     title="N/W vs Prop Length", template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="v", x=1.01),
                      margin=dict(l=50, r=10, t=50, b=50))
    return fig


def build_NW_surface(model):
    """
    3D surface of the fit  N/W = a·L + b·W + c  (Optimizer V3 Step 3).
    """
    lengths = np.linspace(20, 55, 60)
    widths  = np.linspace(3.0, 7.0, 60)
    L, W    = np.meshgrid(lengths, widths)
    NW      = model["a_L"] * L + model["a_W"] * W + model["c"]
    fig = go.Figure(data=[go.Surface(
        z=NW, x=L, y=W, colorscale="Viridis",
        colorbar=dict(title="N/W (N/W)"))])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title=f"N/W Surface Fit  ({model['eq_str']})  R²={model['R2']:.3f}",
        scene=dict(xaxis_title="Prop Length (mm)",
                   yaxis_title="Prop Width (mm)",
                   zaxis_title="Thrust Efficiency (N/W)"),
        margin=dict(l=0, r=0, t=50, b=0), height=500)
    return fig


def thrust_list_ui(store):
    """Every entry gets a delete button."""
    if not store:
        return html.Div("No prop test entries yet.", className="text-muted small")
    items = []
    for i, e in enumerate(store):
        source = e.get("source", "manual")
        tag    = "📄" if source == "file" else "🖊"
        label  = (f"{tag} {e.get('motor_type','?')}  "
                  f"L={e['prop_length_mm']}mm W={e['prop_width_mm']}mm  "
                  f"N/W={e['N_per_W']:.5f}  α={e.get('flexure_angle_deg','?')}°")
        items.append(dbc.ListGroupItem(
            dbc.Row([
                dbc.Col(html.Span(label, className="small"), width=9),
                dbc.Col(dbc.Button("✕", id={"type": "thrust-delete-btn", "index": i},
                                   color="danger", size="sm", className="py-0 px-2"),
                        width=3, className="text-end"),
            ], align="center"),
            className="bg-transparent border-secondary text-white py-1",
        ))
    return dbc.ListGroup(items, flush=True)


# ── Layout ────────────────────────────────────────────────────────────────────
layout = dbc.Container(fluid=True, children=[
    dcc.Store(id="thrust-store", data=None),
    dcc.Store(id="thrust-init-trigger", data=0),

    html.H3("💨 Thrust & Propeller Characterization", className="mb-4 mt-2 text-info"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Fit Flexure Angle (°)"),
                dbc.CardBody([
                    dcc.Input(id="thrust-angle", type="number", value=20,
                              min=0, max=90, step=5,
                              className="form-control form-control-sm"),
                    html.P("Angle used for N/W surface fit (20° optimal from your data).",
                           className="small text-muted mt-1"),
                ]),
            ], className="mb-3"),

            dbc.Accordion([
                dbc.AccordionItem(title="➕  Add Prop Test", children=[
                    dbc.Label("Motor Type / Label", className="small mb-0"),
                    dbc.Input(id="thrust-motor-type", placeholder="e.g. Florida 4.6Ω", size="sm", className="mb-2"),
                    dbc.Label("Prop Length (mm)", className="small mb-0"),
                    dbc.Input(id="thrust-length", type="number", placeholder="e.g. 33.5", size="sm", className="mb-2"),
                    dbc.Label("Prop Width (mm)", className="small mb-0"),
                    dbc.Input(id="thrust-width", type="number", placeholder="e.g. 4.35", size="sm", className="mb-2"),
                    dbc.Label("Prop Mass (mg)", className="small mb-0"),
                    dbc.Input(id="thrust-prop-mass", type="number", placeholder="e.g. 51", size="sm", className="mb-2"),
                    dbc.Label("Voltage (V)", className="small mb-0"),
                    dbc.Input(id="thrust-volt", type="number", placeholder="e.g. 5.0", size="sm", className="mb-2"),
                    dbc.Label("Current (mA)", className="small mb-0"),
                    dbc.Input(id="thrust-current", type="number", placeholder="e.g. 700", size="sm", className="mb-2"),
                    dbc.Label("Thrust (N)", className="small mb-0"),
                    dbc.Input(id="thrust-N", type="number", placeholder="e.g. 0.0256", size="sm", className="mb-2"),
                    dbc.Label("Flexure Angle (°)", className="small mb-0"),
                    dbc.Input(id="thrust-flex-angle", type="number", placeholder="e.g. 20", size="sm", className="mb-2"),
                    html.P("N/W is auto-computed from Thrust(N) / Power(W).",
                           className="small text-muted"),
                    dbc.Button("Save Entry", id="thrust-save-btn", color="success", size="sm", className="w-100 mt-1"),
                    html.Div(id="thrust-save-status", className="small mt-2"),
                ]),
            ], start_collapsed=True, className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Prop Test Entries  (📄 file · 🖊 manual)"),
                dbc.CardBody(html.Div(id="thrust-list-ui"),
                             style={"maxHeight": "350px", "overflowY": "auto"}),
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Fit Info"),
                dbc.CardBody(html.Div(id="thrust-fit-info", className="small")),
            ]),
        ], md=3),

        dbc.Col([
            dbc.Tabs([
                dbc.Tab(dcc.Graph(id="thrust-nw-angle-fig",   config={"displayModeBar": False}), label="N/W vs Angle"),
                dbc.Tab(dcc.Graph(id="thrust-power-fig",      config={"displayModeBar": False}), label="Thrust vs Power"),
                dbc.Tab(dcc.Graph(id="thrust-nw-scatter-fig", config={"displayModeBar": False}), label="N/W Scatter"),
                dbc.Tab(dcc.Graph(id="thrust-nw-surface-fig", config={"displayModeBar": False}), label="N/W Surface"),
            ]),
        ], md=9),
    ]),
])


# ── Init ──────────────────────────────────────────────────────────────────────
@callback(
    Output("thrust-store", "data", allow_duplicate=True),
    Input("thrust-init-trigger", "data"),
    State("thrust-store", "data"),
    prevent_initial_call='initial_duplicate',
)
def init_store(_, existing):
    if existing:
        return dash.no_update
    return _initial_thrust_store()


# ── Save entry ────────────────────────────────────────────────────────────────
@callback(
    Output("thrust-store",        "data",    allow_duplicate=True),
    Output("thrust-save-status",  "children"),
    Output("thrust-motor-type",   "value"),
    Output("thrust-length",       "value"),
    Output("thrust-width",        "value"),
    Output("thrust-prop-mass",    "value"),
    Output("thrust-volt",         "value"),
    Output("thrust-current",      "value"),
    Output("thrust-N",            "value"),
    Output("thrust-flex-angle",   "value"),
    Input("thrust-save-btn",      "n_clicks"),
    State("thrust-motor-type",    "value"),
    State("thrust-length",        "value"),
    State("thrust-width",         "value"),
    State("thrust-prop-mass",     "value"),
    State("thrust-volt",          "value"),
    State("thrust-current",       "value"),
    State("thrust-N",             "value"),
    State("thrust-flex-angle",    "value"),
    State("thrust-store",         "data"),
    prevent_initial_call=True,
)
def save_entry(_, motor_type, length, width, prop_mass, volt, current, thrust_N, flex_angle, store):
    store = list(store or [])
    nd    = dash.no_update
    if length is None or width is None or thrust_N is None:
        return store, "⚠ Prop length, width, and thrust are required.", nd, nd, nd, nd, nd, nd, nd, nd
    V      = float(volt or 0)
    I_mA   = float(current or 0)
    P_W    = V * I_mA / 1000
    N_per_W = float(thrust_N) / P_W if P_W > 0 else 0.0
    store.append({
        "motor_type":        str(motor_type or "unknown"),
        "prop_length_mm":    float(length),
        "prop_width_mm":     float(width),
        "prop_mass_mg":      float(prop_mass or 0),
        "voltage_V":         V,
        "current_mA":        I_mA,
        "power_W":           P_W,
        "thrust_N":          float(thrust_N),
        "N_per_W":           N_per_W,
        "flexure_angle_deg": float(flex_angle or 0),
        "source":            "manual",
    })
    return store, f"✓ Saved (N/W = {N_per_W:.5f})", "", None, None, None, None, None, None, None


# ── Delete entry ──────────────────────────────────────────────────────────────
@callback(
    Output("thrust-store", "data", allow_duplicate=True),
    Input({"type": "thrust-delete-btn", "index": ALL}, "n_clicks"),
    State("thrust-store", "data"),
    prevent_initial_call=True,
)
def delete_entry(_, store):
    store     = list(store or [])
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        idx = triggered.get("index", -1)
        if 0 <= idx < len(store):
            store.pop(idx)
    return store


# ── Refresh list UI ───────────────────────────────────────────────────────────
@callback(
    Output("thrust-list-ui", "children"),
    Input("thrust-store",    "data"),
)
def refresh_list(store):
    return thrust_list_ui(store or [])


# ── Update charts ─────────────────────────────────────────────────────────────
@callback(
    Output("thrust-nw-angle-fig",   "figure"),
    Output("thrust-power-fig",      "figure"),
    Output("thrust-nw-scatter-fig", "figure"),
    Output("thrust-nw-surface-fig", "figure"),
    Output("thrust-fit-info",       "children"),
    Input("thrust-angle",           "value"),
    Input("thrust-store",           "data"),
)
def update_charts(angle, store):
    df    = _store_to_df(store)
    empty = go.Figure()
    empty.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")

    if df.empty:
        msg = html.P("No data yet. Built-in prop_lift_tests data loads automatically "
                     "if prop_lift_tests_33p5mm.csv is in data/thrust/.",
                     className="text-muted small")
        return build_NW_vs_angle(df), empty, empty, empty, msg

    model    = fit_thrust_model(df, flexure_angle=float(angle or 20))
    fit_info = [
        html.P(f"Fit:  {model['eq_str']}", className="mb-1"),
        html.P(f"R² = {model['R2']:.4f}", className="mb-1"),
        html.P(f"Baseline prop length: {model['baseline_length_mm']:.1f} mm", className="mb-1"),
        html.P(f"Fit uses {angle}° flexure angle rows only.", className="text-muted small"),
    ]
    return (
        build_NW_vs_angle(df),
        build_thrust_vs_power(df),
        build_NW_scatter(df),
        build_NW_surface(model),
        fit_info,
    )
