"""
Motor Characterization Page
Mirrors: Kyle_&_Jose_Motor_Characterization_V1
"""

import base64
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

from data_engine import load_motor_csv, compute_motor_params

dash.register_page(__name__, path="/motor", name="Motor", title="Motor Characterization")

COLORS = px.colors.qualitative.Plotly


def build_motor_params_table(motor_params: dict) -> dbc.Table:
    rows = []
    for name, p in motor_params.items():
        rows.append({"Motor": name, "Mass (mg)": f"{p['mass_mg']:.0f}",
                     "R (Ω)": f"{p['R']:.3f}", "K (Nm/A)": f"{p['K']:.6f}"})
    if not rows:
        return html.Div("No motor data loaded.")
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

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["K (Nm/A)", "R (Ω)", "Mass (mg)"])

    fig.add_trace(go.Bar(x=names, y=K_vals, marker_color=COLORS[:len(names)],
                         showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=R_vals, marker_color=COLORS[:len(names)],
                         showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=mass,   marker_color=COLORS[:len(names)],
                         showlegend=False), row=1, col=3)

    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title="Motor Parameters Comparison",
                      margin=dict(l=50, r=10, t=70, b=100),
                      height=420)
    fig.update_xaxes(tickangle=-40)
    return fig


def build_efficiency_fig(motor_params: dict, selected: list, V_nominal: float = 3.7) -> go.Figure:
    """
    Plot motor shaft power vs. electrical power for each motor at nominal voltage.
    P_shaft = (V − io*R) * omega / R  (simplified from K/R model)
    """
    fig = go.Figure()
    for i, name in enumerate([n for n in selected if n in motor_params]):
        p  = motor_params[name]
        K  = p["K"]
        R  = p["R"] if p["R"] > 0 else 0.001
        # Sweep current from 0 to stall
        I_max = V_nominal / R
        I     = np.linspace(0, I_max * 0.95, 300)
        omega = (V_nominal - I * R) / K
        P_shaft  = K * omega * I
        P_elec   = V_nominal * I
        eta      = np.where(P_elec > 0, P_shaft / P_elec, 0)

        fig.add_trace(go.Scatter(
            x=P_elec * 1000, y=eta * 100,
            name=name, line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title=f"Motor Efficiency vs Electrical Power  (V = {V_nominal:.1f} V)",
        xaxis_title="Electrical Power (mW)", yaxis_title="Efficiency (%)",
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50),
    )
    return fig


def build_torque_speed_fig(motor_params: dict, selected: list, V_nominal: float = 3.7) -> go.Figure:
    fig = go.Figure()
    for i, name in enumerate([n for n in selected if n in motor_params]):
        p = motor_params[name]
        K = p["K"]; R = p["R"] if p["R"] > 0 else 0.001
        omega_max = V_nominal / K
        omega = np.linspace(0, omega_max, 300)
        torque = K * (V_nominal - K * omega) / R
        torque = np.clip(torque, 0, None)
        rpm    = omega * 60 / (2 * np.pi)
        fig.add_trace(go.Scatter(
            x=rpm, y=torque * 1000,
            name=name, line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title=f"Torque vs RPM  (V = {V_nominal:.1f} V)",
        xaxis_title="RPM", yaxis_title="Torque (mNm)",
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50),
    )
    return fig


# ── Layout ───────────────────────────────────────────────────────────────────
layout = dbc.Container(fluid=True, children=[
    dcc.Store(id="motor-extra-store", data=None),

    html.H3("⚙️ Motor Characterization", className="mb-4 mt-2 text-info"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Select Motors"),
                dbc.CardBody([
                    dcc.Checklist(id="motor-selector", options=[], value=[],
                                  labelStyle={"display": "block", "marginBottom": "6px"},
                                  inputStyle={"marginRight": "8px"}),
                ]),
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Voltage (V)"),
                dbc.CardBody([
                    dcc.Slider(id="motor-voltage", min=2.0, max=5.0,
                               step=0.1, value=3.7,
                               marks={v: str(v) for v in [2, 3, 3.7, 4, 5]},
                               tooltip={"placement": "bottom"}),
                ]),
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Upload Motor CSV"),
                dbc.CardBody([
                    html.P("Same column format as motor_characteristics.csv",
                           className="small text-muted"),
                    dcc.Upload(
                        id="motor-upload",
                        children=html.Div(["Drag & Drop or ", html.A("Select File")]),
                        style={"borderWidth": "1px", "borderStyle": "dashed",
                               "borderRadius": "5px", "borderColor": "#6c757d",
                               "textAlign": "center", "padding": "20px"},
                        multiple=False,
                    ),
                    html.Div(id="motor-upload-status", className="mt-2 small"),
                ]),
            ]),
        ], md=3),

        dbc.Col([
            dbc.Tabs([
                dbc.Tab(html.Div(id="motor-param-table", className="mt-3"), label="Parameters"),
                dbc.Tab(dcc.Graph(id="motor-compare-fig", config={"displayModeBar": False}), label="Comparison"),
                dbc.Tab(dcc.Graph(id="motor-efficiency-fig", config={"displayModeBar": False}), label="Efficiency"),
                dbc.Tab(dcc.Graph(id="motor-torquespeed-fig", config={"displayModeBar": False}), label="Torque–Speed"),
            ]),
        ], md=9),
    ]),
])


@callback(
    Output("motor-selector",       "options"),
    Output("motor-selector",       "value"),
    Output("motor-extra-store",    "data"),
    Output("motor-upload-status",  "children"),
    Input("motor-upload",          "contents"),
    State("motor-upload",          "filename"),
    State("motor-extra-store",     "data"),
    prevent_initial_call=True,
)
def handle_upload(content, fname, existing_b64):
    import pandas as pd
    if not content:
        return dash.no_update, dash.no_update, dash.no_update, ""
    _, encoded = content.split(",")
    raw = base64.b64decode(encoded)
    try:
        df_up = load_motor_csv(raw_bytes=raw)
        # Merge with base
        df_base = load_motor_csv()
        df_all  = pd.concat([df_base, df_up], ignore_index=True).drop_duplicates(subset=["Motor Name"])
    except Exception as e:
        return dash.no_update, dash.no_update, existing_b64, f"⚠ Error: {e}"

    motor_params = compute_motor_params(df_all)
    opts  = [{"label": n, "value": n} for n in sorted(motor_params)]
    value = sorted(motor_params)
    return opts, value, encoded, f"✓ Loaded '{fname}'"


@callback(
    Output("motor-selector", "options", allow_duplicate=True),
    Output("motor-selector", "value",   allow_duplicate=True),
    Input("motor-selector",  "id"),
    prevent_initial_call=False,
)
def init_motor_selector(_):
    df = load_motor_csv()
    motor_params = compute_motor_params(df)
    all_names = sorted(motor_params)
    opts = [{"label": n, "value": n} for n in all_names]
    return opts, all_names


@callback(
    Output("motor-param-table",     "children"),
    Output("motor-compare-fig",     "figure"),
    Output("motor-efficiency-fig",  "figure"),
    Output("motor-torquespeed-fig", "figure"),
    Input("motor-selector",         "value"),
    Input("motor-voltage",          "value"),
    State("motor-extra-store",      "data"),
)
def update_charts(selected, V, extra_b64):
    import pandas as pd
    df_base = load_motor_csv()
    if extra_b64:
        try:
            raw = base64.b64decode(extra_b64)
            df_up = load_motor_csv(raw_bytes=raw)
            df_all = pd.concat([df_base, df_up], ignore_index=True).drop_duplicates(subset=["Motor Name"])
        except Exception:
            df_all = df_base
    else:
        df_all = df_base

    motor_params = compute_motor_params(df_all)
    selected = [s for s in (selected or []) if s in motor_params]

    empty = go.Figure()
    empty.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")

    if not selected:
        return "No motors selected.", empty, empty, empty

    return (
        build_motor_params_table({n: motor_params[n] for n in selected}),
        build_K_comparison(motor_params, selected),
        build_efficiency_fig(motor_params, selected, V or 3.7),
        build_torque_speed_fig(motor_params, selected, V or 3.7),
    )
