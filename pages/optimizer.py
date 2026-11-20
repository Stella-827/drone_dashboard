"""
Flight Time Optimizer Page — Mirrors Optimizer V3 Steps 4-5
"""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc

from data_engine import (
    load_motor_csv, compute_motor_params,
    load_thrust_csv, fit_thrust_model,
    run_optimizer, DEFAULT_BATTERY_MODELS,
)

dash.register_page(__name__, path="/optimizer", name="Optimizer", title="Flight Time Optimizer")
COLORS = px.colors.qualitative.Plotly


def build_top_bar(df, n=10):
    top = df.head(n)
    labels = [f"{r['motor'][:18]}…\n{r['battery'][:14]}…\nL={r['prop_length_mm']}mm W={r['prop_width_mm']}mm"
              for _, r in top.iterrows()]
    fig = go.Figure(go.Bar(
        x=top["flight_time_s"], y=labels, orientation="h",
        marker_color=COLORS[:n],
        text=[f"{v:.1f} s" for v in top["flight_time_s"]],
        textposition="auto"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title=f"Top {n} Configurations by Flight Time",
                      xaxis_title="Flight Time (s)", yaxis=dict(autorange="reversed"),
                      margin=dict(l=280, r=20, t=50, b=50), height=max(400, n * 45))
    return fig


def build_heatmap(df, motor, battery):
    sub = df[(df["motor"] == motor) & (df["battery"] == battery)]
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          title="No data for selected motor/battery.")
        return fig
    pivot = sub.pivot_table(values="flight_time_s",
                            index="prop_width_mm", columns="prop_length_mm", aggfunc="max")
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns.tolist(),
                                y=pivot.index.tolist(), colorscale="Viridis",
                                colorbar=dict(title="Flight Time (s)")))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title=f"Flight Time Heatmap – {motor[:30]} × {battery[:30]}",
                      xaxis_title="Prop Length (mm)", yaxis_title="Prop Width (mm)",
                      margin=dict(l=80, r=20, t=60, b=50))
    return fig


def build_scatter_matrix(df):
    cols = ["prop_length_mm", "prop_width_mm", "total_mass_mg", "P_elec_mW", "flight_time_s"]
    present = [c for c in cols if c in df.columns]
    fig = px.scatter_matrix(df.head(500), dimensions=present,
                            color="flight_time_s", color_continuous_scale="Viridis",
                            title="Parameter Scatter Matrix (top 500)",
                            template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=50, r=50, t=50, b=50))
    return fig


def build_mass_fig(df):
    top = df.head(15)
    fig = go.Figure(go.Bar(x=list(range(1, len(top)+1)), y=top["total_mass_mg"].tolist(),
                           marker_color="steelblue"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title="Total System Mass — Top 15 Configs",
                      xaxis_title="Rank", yaxis_title="Mass (mg)",
                      margin=dict(l=50, r=10, t=50, b=50))
    return fig


layout = dbc.Container(fluid=True, children=[
    dcc.Store(id="opt-results-store", data=None),
    html.H3("🚁 Flight Time Optimizer", className="mb-4 mt-2 text-info"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Select Motors"),
                dbc.CardBody(dcc.Checklist(id="opt-motor-selector", options=[], value=[],
                    labelStyle={"display":"block","marginBottom":"4px"},
                    inputStyle={"marginRight":"6px"})),
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader("Select Batteries"),
                dbc.CardBody(dcc.Checklist(id="opt-battery-selector", options=[], value=[],
                    labelStyle={"display":"block","marginBottom":"4px"},
                    inputStyle={"marginRight":"6px"})),
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader("Prop Search Space"),
                dbc.CardBody([
                    html.Label("Length range (mm):", className="small"),
                    dcc.RangeSlider(id="opt-length-range", min=15, max=70, step=2, value=[20,50],
                        marks={v:str(v) for v in [15,25,33,40,50,60,70]},
                        tooltip={"placement":"bottom"}),
                    html.Label("Width range (mm):", className="small mt-2"),
                    dcc.RangeSlider(id="opt-width-range", min=2.0, max=10.0, step=0.5, value=[3.5,6.5],
                        marks={v:str(v) for v in [2,3.5,5,6.5,8,10]},
                        tooltip={"placement":"bottom"}),
                ]),
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader("Physical Constants"),
                dbc.CardBody([
                    dbc.Row([dbc.Col(html.Label("Body mass (mg):",className="small")),
                             dbc.Col(dbc.Input(id="opt-body-mass",type="number",value=450,
                                              className="form-control form-control-sm"))],className="mb-2"),
                    dbc.Row([dbc.Col(html.Label("Lift margin (mg):",className="small")),
                             dbc.Col(dbc.Input(id="opt-lift-margin",type="number",value=100,
                                              className="form-control form-control-sm"))]),
                ]),
            ], className="mb-3"),
            dbc.Button("🔄 Run Optimizer", id="opt-run-btn", color="success", className="w-100 mb-3"),
            html.Div(id="opt-status", className="small text-muted"),
        ], md=3),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(dcc.Graph(id="opt-topbar-fig",  config={"displayModeBar":False}), label="Top Configs"),
                dbc.Tab(dcc.Graph(id="opt-heatmap-fig", config={"displayModeBar":False}), label="Heatmap"),
                dbc.Tab(dcc.Graph(id="opt-scatter-fig", config={"displayModeBar":False}), label="Scatter Matrix"),
                dbc.Tab(dcc.Graph(id="opt-mass-fig",    config={"displayModeBar":False}), label="Mass Breakdown"),
                dbc.Tab(html.Div([
                    dbc.Row([
                        dbc.Col([html.Label("Motor for heatmap:",className="small"),
                                 dcc.Dropdown(id="opt-heatmap-motor",options=[],value=None,className="mb-2",
                                              style={"backgroundColor":"#2d2d2d","color":"#fff"})]),
                        dbc.Col([html.Label("Battery for heatmap:",className="small"),
                                 dcc.Dropdown(id="opt-heatmap-battery",options=[],value=None,className="mb-2",
                                              style={"backgroundColor":"#2d2d2d","color":"#fff"})]),
                    ]),
                    html.Div(id="opt-results-table",className="mt-2",style={"overflowX":"auto"}),
                ]), label="Results Table"),
            ]),
        ], md=9),
    ]),
])


@callback(
    Output("opt-motor-selector",   "options"),
    Output("opt-motor-selector",   "value"),
    Output("opt-battery-selector", "options"),
    Output("opt-battery-selector", "value"),
    Input("opt-motor-selector",    "id"),
    prevent_initial_call=False,
)
def init_selectors(_):
    df = load_motor_csv()
    params = compute_motor_params(df)
    motor_opts = [{"label": n, "value": n} for n in sorted(params)]
    bat_opts   = [{"label": b[0], "value": b[0]} for b in DEFAULT_BATTERY_MODELS]
    return motor_opts, sorted(params), bat_opts, [b[0] for b in DEFAULT_BATTERY_MODELS]


@callback(
    Output("opt-results-store",    "data"),
    Output("opt-status",           "children"),
    Output("opt-heatmap-motor",    "options"),
    Output("opt-heatmap-battery",  "options"),
    Output("opt-heatmap-motor",    "value"),
    Output("opt-heatmap-battery",  "value"),
    Input("opt-run-btn",           "n_clicks"),
    State("opt-motor-selector",    "value"),
    State("opt-battery-selector",  "value"),
    State("opt-length-range",      "value"),
    State("opt-width-range",       "value"),
    State("opt-body-mass",         "value"),
    State("opt-lift-margin",       "value"),
    prevent_initial_call=True,
)
def run_opt(_, sel_motors, sel_bats, l_range, w_range, body_mass, lift_margin):
    df_motors    = load_motor_csv()
    motor_params = compute_motor_params(df_motors)
    df_thrust    = load_thrust_csv()
    thrust_model = fit_thrust_model(df_thrust)

    prop_lengths = np.arange(l_range[0], l_range[1] + 1, 2)
    prop_widths  = np.arange(w_range[0], w_range[1] + 0.1, 0.5)

    results = run_optimizer(
        motor_params=motor_params,
        batteries_model=DEFAULT_BATTERY_MODELS,
        thrust_model=thrust_model,
        prop_lengths_mm=prop_lengths,
        prop_widths_mm=prop_widths,
        body_mass_mg=float(body_mass or 450),
        lift_margin_mg=float(lift_margin or 100),
        selected_motors=sel_motors,
        selected_batteries=sel_bats,
    )

    if results.empty:
        return None, "⚠ No valid combinations found.", [], [], None, None

    csv_data = results.to_json(orient="records")
    best     = results.iloc[0]
    status   = (f"✓ {len(results):,} combinations. Best: {best['motor'][:20]}… + "
                f"{best['battery'][:16]}…  →  {best['flight_time_s']:.1f} s")
    motors   = [{"label": m, "value": m} for m in results["motor"].unique()]
    bats     = [{"label": b, "value": b} for b in results["battery"].unique()]
    return csv_data, status, motors, bats, results["motor"].iloc[0], results["battery"].iloc[0]


@callback(
    Output("opt-topbar-fig",    "figure"),
    Output("opt-heatmap-fig",   "figure"),
    Output("opt-scatter-fig",   "figure"),
    Output("opt-mass-fig",      "figure"),
    Output("opt-results-table", "children"),
    Input("opt-results-store",  "data"),
    Input("opt-heatmap-motor",  "value"),
    Input("opt-heatmap-battery","value"),
)
def update_charts(csv_data, hm_motor, hm_battery):
    empty = go.Figure()
    empty.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    if not csv_data:
        return empty, empty, empty, empty, "Run the optimizer first."

    df = pd.read_json(csv_data, orient="records")
    if df.empty:
        return empty, empty, empty, empty, "No results."

    display_cols = ["motor","battery","prop_length_mm","prop_width_mm",
                    "total_mass_mg","P_elec_mW","E_mWh","flight_time_s"]
    display_cols = [c for c in display_cols if c in df.columns]
    top100 = df[display_cols].head(100).reset_index(drop=True)
    top100.insert(0, "Rank", range(1, len(top100)+1))

    tbl = dash_table.DataTable(
        data=top100.to_dict("records"),
        columns=[{"name": c, "id": c} for c in top100.columns],
        style_table={"overflowX":"auto"},
        style_header={"backgroundColor":"#1e1e2e","color":"white","fontWeight":"bold"},
        style_cell={"backgroundColor":"#2a2a3e","color":"white","border":"1px solid #444","fontSize":12},
        style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":"#252535"}],
        page_size=20, sort_action="native", filter_action="native",
    )
    return (build_top_bar(df), build_heatmap(df, hm_motor or "", hm_battery or ""),
            build_scatter_matrix(df), build_mass_fig(df), tbl)
