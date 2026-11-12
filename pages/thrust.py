"""
Thrust / Propeller Characterization Page
Mirrors: prop_lift_tests data and V3 Step 1 & 3
"""

import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

from data_engine import load_all_thrust, load_thrust_csv, fit_thrust_model

dash.register_page(__name__, path="/thrust", name="Thrust", title="Thrust Characterization")

COLORS = px.colors.qualitative.Plotly


def build_thrust_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                         color_col: str, title: str) -> go.Figure:
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col, hover_data=df.columns.tolist(),
        title=title,
        template="plotly_dark",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="v", x=1.01),
                      margin=dict(l=50, r=10, t=50, b=50))
    return fig


def build_NW_surface(model: dict) -> go.Figure:
    lengths = np.linspace(20, 55, 50)
    widths  = np.linspace(3.0, 7.0, 50)
    L, W    = np.meshgrid(lengths, widths)
    NW      = model["a_L"] * L + model["a_W"] * W + model["c"]

    fig = go.Figure(data=[go.Surface(z=NW, x=L, y=W,
                                     colorscale="Viridis",
                                     colorbar=dict(title="N/W"))])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title=f"N/W Surface Fit  ({model['eq_str']})",
        scene=dict(xaxis_title="Prop Length (mm)",
                   yaxis_title="Prop Width (mm)",
                   zaxis_title="Thrust Efficiency (N/W)"),
        margin=dict(l=0, r=0, t=50, b=0), height=500,
    )
    return fig


def build_NW_vs_angle(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "flexure_angle_deg" not in df.columns or df.empty:
        return fig
    for i, (angle, grp) in enumerate(df.groupby("flexure_angle_deg")):
        fig.add_trace(go.Scatter(
            x=grp["prop_length_mm"], y=grp["N_per_W"],
            mode="markers", name=f"{angle}°",
            marker=dict(color=COLORS[i % len(COLORS)], size=7),
        ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title="N/W vs Prop Length by Flexure Angle",
        xaxis_title="Prop Length (mm)", yaxis_title="N/W (N/W)",
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50),
    )
    return fig


def build_thrust_vs_power(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    motor_types = df["motor_type"].unique() if "motor_type" in df.columns else ["all"]
    fig = go.Figure()
    for i, mt in enumerate(motor_types):
        sub = df[df["motor_type"] == mt] if "motor_type" in df.columns else df
        fig.add_trace(go.Scatter(
            x=sub["power_W"] * 1000, y=sub["thrust_N"] * 1000,
            mode="markers", name=str(mt),
            marker=dict(color=COLORS[i % len(COLORS)], size=7, opacity=0.75),
        ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title="Thrust vs Electrical Power",
        xaxis_title="Power (mW)", yaxis_title="Thrust (mN)",
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50),
    )
    return fig


# ── Layout ───────────────────────────────────────────────────────────────────
layout = dbc.Container(fluid=True, children=[
    dcc.Store(id="thrust-extra-store", data=None),

    html.H3("🌀 Thrust & Propeller Characterization", className="mb-4 mt-2 text-info"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Flexure Angle Filter"),
                dbc.CardBody([
                    html.Label("Use rows at angle (°):"),
                    dcc.Input(id="thrust-angle", type="number", value=20,
                              min=0, max=90, step=5,
                              className="form-control form-control-sm"),
                    html.P("Leave blank to use all rows.", className="small text-muted mt-1"),
                ]),
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Upload Thrust CSV"),
                dbc.CardBody([
                    html.P("Same column format as prop_lift_tests_33p5mm.csv",
                           className="small text-muted"),
                    dcc.Upload(
                        id="thrust-upload",
                        children=html.Div(["Drag & Drop or ", html.A("Select File")]),
                        style={"borderWidth": "1px", "borderStyle": "dashed",
                               "borderRadius": "5px", "borderColor": "#6c757d",
                               "textAlign": "center", "padding": "20px"},
                        multiple=True,
                    ),
                    html.Div(id="thrust-upload-status", className="mt-2 small"),
                ]),
            ]),

            dbc.Card([
                dbc.CardHeader("Fit Info"),
                dbc.CardBody(html.Div(id="thrust-fit-info", className="small")),
            ], className="mt-3"),
        ], md=3),

        dbc.Col([
            dbc.Tabs([
                dbc.Tab(dcc.Graph(id="thrust-nw-angle-fig",    config={"displayModeBar": False}), label="N/W vs Angle"),
                dbc.Tab(dcc.Graph(id="thrust-power-fig",       config={"displayModeBar": False}), label="Thrust vs Power"),
                dbc.Tab(dcc.Graph(id="thrust-nw-scatter-fig",  config={"displayModeBar": False}), label="N/W Scatter"),
                dbc.Tab(dcc.Graph(id="thrust-nw-surface-fig",  config={"displayModeBar": False}), label="N/W Surface"),
            ]),
        ], md=9),
    ]),
])


@callback(
    Output("thrust-extra-store",   "data"),
    Output("thrust-upload-status", "children"),
    Input("thrust-upload",         "contents"),
    State("thrust-upload",         "filename"),
    State("thrust-extra-store",    "data"),
    prevent_initial_call=True,
)
def handle_upload(contents_list, filenames, existing):
    if not contents_list:
        return existing, ""
    all_b64 = list(existing or [])
    msgs = []
    for content, fname in zip(contents_list, filenames):
        _, encoded = content.split(",")
        raw = base64.b64decode(encoded)
        try:
            df_test = load_thrust_csv(raw_bytes=raw)
            if df_test.empty:
                msgs.append(f"⚠ '{fname}' produced no valid rows.")
            else:
                all_b64.append(encoded)
                msgs.append(f"✓ Loaded '{fname}' ({len(df_test)} rows)")
        except Exception as e:
            msgs.append(f"⚠ Error in '{fname}': {e}")
    return all_b64, html.Ul([html.Li(m) for m in msgs])


@callback(
    Output("thrust-nw-angle-fig",   "figure"),
    Output("thrust-power-fig",      "figure"),
    Output("thrust-nw-scatter-fig", "figure"),
    Output("thrust-nw-surface-fig", "figure"),
    Output("thrust-fit-info",       "children"),
    Input("thrust-angle",           "value"),
    State("thrust-extra-store",     "data"),
    Input("thrust-upload",          "contents"),  # triggers on new upload too
)
def update_thrust_charts(angle, extra_b64_list, _):
    extra_dfs = []
    for b64 in (extra_b64_list or []):
        raw = base64.b64decode(b64)
        df  = load_thrust_csv(raw_bytes=raw)
        if not df.empty:
            extra_dfs.append(df)

    df = load_all_thrust(extra_dfs)
    empty = go.Figure()
    empty.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")

    if df.empty:
        return empty, empty, empty, empty, "No thrust data loaded."

    model = fit_thrust_model(df, flexure_angle=float(angle) if angle else 20.0)

    fit_info = [
        html.P(model["eq_str"]),
        html.P(f"R² = {model['R2']:.3f}"),
        html.P(f"Baseline length: {model['baseline_length_mm']:.1f} mm"),
    ]

    return (
        build_NW_vs_angle(df),
        build_thrust_vs_power(df),
        build_thrust_scatter(df, "prop_length_mm", "N_per_W",
                             "motor_type" if "motor_type" in df.columns else "prop_width_mm",
                             "N/W vs Prop Length"),
        build_NW_surface(model),
        fit_info,
    )
