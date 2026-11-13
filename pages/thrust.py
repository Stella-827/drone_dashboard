"""
Thrust / Propeller Characterization Page — manual entry, no CSV upload
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL
import dash_bootstrap_components as dbc

from data_engine import load_all_thrust, fit_thrust_model

dash.register_page(__name__, path="/thrust", name="Thrust", title="Thrust Characterization")

COLORS = px.colors.qualitative.Plotly


# ── Chart builders ────────────────────────────────────────────────────────────
def build_NW_vs_angle(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "flexure_angle_deg" not in df.columns or df.empty:
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          title="No data — add prop test entries to see this chart")
        return fig
    for i, (angle, grp) in enumerate(df.groupby("flexure_angle_deg")):
        fig.add_trace(go.Scatter(x=grp["prop_length_mm"], y=grp["N_per_W"],
                                 mode="markers", name=f"{angle}°",
                                 marker=dict(color=COLORS[i % len(COLORS)], size=8)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title="N/W vs Prop Length by Flexure Angle",
                      xaxis_title="Prop Length (mm)", yaxis_title="N/W (N/W)",
                      legend=dict(orientation="v", x=1.01),
                      margin=dict(l=50, r=10, t=50, b=50))
    return fig


def build_thrust_vs_power(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        return fig
    motor_types = df["motor_type"].unique() if "motor_type" in df.columns else ["all"]
    for i, mt in enumerate(motor_types):
        sub = df[df["motor_type"] == mt] if "motor_type" in df.columns else df
        fig.add_trace(go.Scatter(x=sub["power_W"] * 1000, y=sub["thrust_N"] * 1000,
                                 mode="markers", name=str(mt),
                                 marker=dict(color=COLORS[i % len(COLORS)], size=8, opacity=0.8)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title="Thrust vs Electrical Power",
                      xaxis_title="Power (mW)", yaxis_title="Thrust (mN)",
                      legend=dict(orientation="v", x=1.01),
                      margin=dict(l=50, r=10, t=50, b=50))
    return fig


def build_NW_scatter(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        return fig
    color_col = "motor_type" if "motor_type" in df.columns else "prop_width_mm"
    fig = px.scatter(df, x="prop_length_mm", y="N_per_W", color=color_col,
                     hover_data=df.columns.tolist(),
                     title="N/W vs Prop Length", template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="v", x=1.01),
                      margin=dict(l=50, r=10, t=50, b=50))
    return fig


def build_NW_surface(model: dict) -> go.Figure:
    lengths = np.linspace(20, 55, 50)
    widths  = np.linspace(3.0, 7.0, 50)
    L, W    = np.meshgrid(lengths, widths)
    NW      = model["a_L"] * L + model["a_W"] * W + model["c"]
    fig = go.Figure(data=[go.Surface(z=NW, x=L, y=W, colorscale="Viridis",
                                     colorbar=dict(title="N/W"))])
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title=f"N/W Surface Fit  ({model['eq_str']})",
                      scene=dict(xaxis_title="Prop Length (mm)",
                                 yaxis_title="Prop Width (mm)",
                                 zaxis_title="Thrust Efficiency (N/W)"),
                      margin=dict(l=0, r=0, t=50, b=0), height=500)
    return fig


def thrust_list_ui(entries: list) -> html.Div:
    if not entries:
        return html.Div("No prop test entries yet.", className="text-muted small")
    items = []
    for i, e in enumerate(entries):
        label = (f"L={e['prop_length_mm']}mm W={e['prop_width_mm']}mm "
                 f"N/W={e['N_per_W']:.4f} α={e.get('flexure_angle_deg', '—')}°")
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
    dcc.Store(id="thrust-manual-store", data=[]),

    html.H3("💨 Thrust & Propeller Characterization", className="mb-4 mt-2 text-info"),

    dbc.Row([
        dbc.Col([
            # Flexure angle filter
            dbc.Card([
                dbc.CardHeader("Fit Flexure Angle (°)"),
                dbc.CardBody([
                    dcc.Input(id="thrust-angle", type="number", value=20,
                              min=0, max=90, step=5,
                              className="form-control form-control-sm"),
                    html.P("Angle used for N/W surface fit.", className="small text-muted mt-1"),
                ]),
            ], className="mb-3"),

            # Add prop test accordion
            dbc.Accordion([
                dbc.AccordionItem(
                    title="➕  Add Prop Test",
                    children=[
                        dbc.Label("Motor Type / Label", className="small mb-0"),
                        dbc.Input(id="thrust-motor-type", placeholder="e.g. Florida 4.6Ω", size="sm", className="mb-2"),

                        dbc.Label("Prop Length (mm)", className="small mb-0"),
                        dbc.Input(id="thrust-length", type="number", placeholder="e.g. 33.5", size="sm", className="mb-2"),

                        dbc.Label("Prop Width (mm)", className="small mb-0"),
                        dbc.Input(id="thrust-width", type="number", placeholder="e.g. 4.35", size="sm", className="mb-2"),

                        dbc.Label("Prop Mass (mg)", className="small mb-0"),
                        dbc.Input(id="thrust-prop-mass", type="number", placeholder="e.g. 51", size="sm", className="mb-2"),

                        dbc.Label("Voltage (V)", className="small mb-0"),
                        dbc.Input(id="thrust-volt", type="number", placeholder="e.g. 5", size="sm", className="mb-2"),

                        dbc.Label("Current (mA)", className="small mb-0"),
                        dbc.Input(id="thrust-current", type="number", placeholder="e.g. 700", size="sm", className="mb-2"),

                        dbc.Label("Thrust (N)", className="small mb-0"),
                        dbc.Input(id="thrust-N", type="number", placeholder="e.g. 0.0256", size="sm", className="mb-2"),

                        dbc.Label("Flexure Angle (°)", className="small mb-0"),
                        dbc.Input(id="thrust-flex-angle", type="number", placeholder="e.g. 20", size="sm", className="mb-2"),

                        dbc.Button("Save Entry", id="thrust-save-btn",
                                   color="success", size="sm", className="w-100 mt-1"),
                        html.Div(id="thrust-save-status", className="small mt-2"),
                    ],
                ),
            ], start_collapsed=True, className="mb-3"),

            # Saved entries list
            dbc.Card([
                dbc.CardHeader("Saved Prop Tests"),
                dbc.CardBody(html.Div(id="thrust-list-ui")),
            ], className="mb-3"),

            # Fit info
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


# ── Save entry ────────────────────────────────────────────────────────────────
@callback(
    Output("thrust-manual-store", "data"),
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
    State("thrust-manual-store",  "data"),
    prevent_initial_call=True,
)
def save_thrust(_, motor_type, length, width, prop_mass, volt, current, thrust_N, flex_angle, store):
    store = list(store or [])
    if length is None or width is None or thrust_N is None:
        return (store, "⚠ Prop length, width, and thrust (N) are required.",
                *[dash.no_update] * 8)
    V     = float(volt or 0)
    I_mA  = float(current or 0)
    P_W   = V * I_mA / 1000
    N_per_W = float(thrust_N) / P_W if P_W > 0 else 0.0
    entry = {
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
    }
    store.append(entry)
    return store, f"✓ Saved entry (N/W = {N_per_W:.5f})", "", None, None, None, None, None, None, None


# ── Delete entry ──────────────────────────────────────────────────────────────
@callback(
    Output("thrust-manual-store", "data", allow_duplicate=True),
    Input({"type": "thrust-delete-btn", "index": ALL}, "n_clicks"),
    State("thrust-manual-store", "data"),
    prevent_initial_call=True,
)
def delete_thrust(_, store):
    store = list(store or [])
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict) and triggered.get("type") == "thrust-delete-btn":
        idx = triggered["index"]
        if 0 <= idx < len(store):
            store.pop(idx)
    return store


# ── Refresh list UI ───────────────────────────────────────────────────────────
@callback(
    Output("thrust-list-ui", "children"),
    Input("thrust-manual-store", "data"),
)
def refresh_thrust_list(store):
    return thrust_list_ui(store or [])


# ── Update charts ─────────────────────────────────────────────────────────────
@callback(
    Output("thrust-nw-angle-fig",   "figure"),
    Output("thrust-power-fig",      "figure"),
    Output("thrust-nw-scatter-fig", "figure"),
    Output("thrust-nw-surface-fig", "figure"),
    Output("thrust-fit-info",       "children"),
    Input("thrust-angle",           "value"),
    Input("thrust-manual-store",    "data"),
)
def update_charts(angle, manual_store):
    manual_dfs = []
    if manual_store:
        df_manual = pd.DataFrame(manual_store)
        if not df_manual.empty:
            manual_dfs.append(df_manual)

    df = load_all_thrust(manual_dfs)
    empty = go.Figure()
    empty.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")

    if df.empty:
        msg = html.P("Add prop test entries using the form to populate charts.",
                     className="text-muted")
        return (build_NW_vs_angle(df), empty, empty, empty, msg)

    model = fit_thrust_model(df, flexure_angle=float(angle or 20))
    fit_info = [
        html.P(model["eq_str"]),
        html.P(f"R² = {model['R2']:.3f}"),
        html.P(f"Baseline length: {model['baseline_length_mm']:.1f} mm"),
    ]
    return (
        build_NW_vs_angle(df),
        build_thrust_vs_power(df),
        build_NW_scatter(df),
        build_NW_surface(model),
        fit_info,
    )
