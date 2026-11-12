"""
Battery Characterization Page
Mirrors: Kyle_&_Jose_Battery_Characterization_V1
"""

import base64, io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc

from data_engine import (
    load_all_batteries, load_battery_csv,
    fit_battery_energy_model, NOMINAL_V,
)

dash.register_page(__name__, path="/battery", name="Battery", title="Battery Characterization")

# ── Helpers ──────────────────────────────────────────────────────────────────
TAB_COLORS = px.colors.qualitative.Plotly

def build_capacity_fig(batteries: dict, selected: list[str]) -> go.Figure:
    fig = go.Figure()
    for bat_name in selected:
        runs = batteries.get(bat_name, [])
        for i, r in enumerate(runs):
            color = TAB_COLORS[i % len(TAB_COLORS)]
            I_dis = r["discharge_mA"]

            dis = r["df_discharge"]
            fig.add_trace(go.Scatter(
                x=dis["Q discharge/mA.h"], y=dis["Ewe/V"],
                name=f"{bat_name} – Discharge @ {I_dis:.0f} mA",
                line=dict(color=color, width=2),
            ))

            chg = r["df_charge"]
            if not chg.empty:
                fig.add_trace(go.Scatter(
                    x=chg["Q charge/mA.h"], y=chg["Ewe/V"],
                    name=f"{bat_name} – Charge @ {r['charge_mA']:.0f} mA",
                    line=dict(color=color, width=2, dash="dash"),
                ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title="Voltage vs. Capacity (Charge & Discharge)",
        xaxis_title="Capacity (mAh)", yaxis_title="Voltage (V)",
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50),
    )
    return fig


def build_time_fig(batteries: dict, selected: list[str]) -> go.Figure:
    fig = go.Figure()
    for bat_name in selected:
        runs = batteries.get(bat_name, [])
        for i, r in enumerate(runs):
            color = TAB_COLORS[i % len(TAB_COLORS)]
            df = r["df_full"]
            t_hours = df["time/s"] / 3600
            I_dis = r["discharge_mA"]

            dis_mask = df["ox/red"] == 0
            fig.add_trace(go.Scatter(
                x=t_hours[dis_mask], y=df["Ewe/V"][dis_mask],
                name=f"{bat_name} – Discharge @ {I_dis:.0f} mA",
                line=dict(color=color, width=2),
            ))

            chg_mask = df["ox/red"] == 1
            fig.add_trace(go.Scatter(
                x=t_hours[chg_mask], y=df["Ewe/V"][chg_mask],
                name=f"{bat_name} – Charge @ {r['charge_mA']:.0f} mA",
                line=dict(color=color, width=2, dash="dash"),
            ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title="Voltage vs. Time",
        xaxis_title="Time (hours)", yaxis_title="Voltage (V)",
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50),
    )
    return fig


def build_energy_fig(batteries: dict, selected: list[str]) -> go.Figure:
    fig = go.Figure()
    for bat_name in selected:
        runs = batteries.get(bat_name, [])
        fit  = fit_battery_energy_model(runs)

        powers   = np.array([r["discharge_mA"] * NOMINAL_V for r in runs])
        energies = np.array([r["energy_mWh"] for r in runs])
        labels   = [f"{r['discharge_mA']:.0f} mA" for r in runs]

        fig.add_trace(go.Scatter(
            x=powers, y=energies, mode="markers+text",
            text=labels, textposition="top right",
            marker=dict(size=10), name=f"{bat_name} data",
        ))

        if len(powers) >= 2:
            P_fit = np.linspace(0, powers.max() * 1.3, 200)
            E_fit = fit["m"] * P_fit + fit["b"]
            fig.add_trace(go.Scatter(
                x=P_fit, y=E_fit, mode="lines",
                name=f"{bat_name} – {fit['eq_str']}",
                line=dict(width=2, dash="dot"),
            ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title="Total Energy vs. Discharge Power (Linear Fit)",
        xaxis_title="Discharge Power (mW)", yaxis_title="Total Energy (mWh)",
        xaxis=dict(rangemode="tozero"), yaxis=dict(rangemode="tozero"),
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50),
    )
    return fig


def build_summary_table(batteries: dict, selected: list[str]) -> list:
    rows = []
    for bat_name in selected:
        runs = batteries.get(bat_name, [])
        fit  = fit_battery_energy_model(runs)
        for r in runs:
            rows.append({
                "Battery":        bat_name,
                "Discharge (mA)": f"{r['discharge_mA']:.0f}",
                "Charge (mA)":    f"{r['charge_mA']:.0f}",
                "Energy (mWh)":   f"{r['energy_mWh']:.2f}",
                "Max Charge V":   f"{r['max_voltage']:.4f}",
                "Fit":            fit["eq_str"],
            })
    return rows


# ── Layout ───────────────────────────────────────────────────────────────────
layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Store(id="bat-extra-store", data=[]),   # extra uploaded records (serialisable form)

        html.H3("🔋 Battery Characterization", className="mb-4 mt-2 text-info"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Select Batteries"),
                    dbc.CardBody([
                        dcc.Checklist(
                            id="bat-selector",
                            options=[],   # filled by callback
                            value=[],
                            labelStyle={"display": "block", "marginBottom": "6px"},
                            inputStyle={"marginRight": "8px"},
                        ),
                    ]),
                ], className="mb-3"),

                dbc.Card([
                    dbc.CardHeader("Upload New Battery CSV"),
                    dbc.CardBody([
                        html.P(
                            "File name must follow:\n"
                            "<Name>-<X>mA_discharge-<Y>mA_charge_..._GCPL_C01.csv",
                            className="small text-muted",
                        ),
                        dcc.Upload(
                            id="bat-upload",
                            children=html.Div(["Drag & Drop or ", html.A("Select Files")]),
                            style={
                                "borderWidth": "1px", "borderStyle": "dashed",
                                "borderRadius": "5px", "borderColor": "#6c757d",
                                "textAlign": "center", "padding": "20px",
                            },
                            multiple=True,
                        ),
                        html.Div(id="bat-upload-status", className="mt-2 small"),
                    ]),
                ]),
            ], md=3),

            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(dcc.Graph(id="bat-cap-fig",    config={"displayModeBar": False}), label="Capacity Curves"),
                    dbc.Tab(dcc.Graph(id="bat-time-fig",   config={"displayModeBar": False}), label="Voltage vs Time"),
                    dbc.Tab(dcc.Graph(id="bat-energy-fig", config={"displayModeBar": False}), label="Energy vs Power"),
                    dbc.Tab(
                        html.Div([
                            html.Div(id="bat-summary-table", className="mt-3"),
                        ]),
                        label="Summary Table",
                    ),
                ]),
            ], md=9),
        ]),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────
@callback(
    Output("bat-selector",      "options"),
    Output("bat-selector",      "value"),
    Output("bat-extra-store",   "data"),
    Output("bat-upload-status", "children"),
    Input("bat-upload",         "contents"),
    State("bat-upload",         "filename"),
    State("bat-extra-store",    "data"),
    prevent_initial_call=True,
)
def handle_upload(contents_list, filenames, existing_extra):
    if not contents_list:
        return dash.no_update, dash.no_update, dash.no_update, ""

    new_records = list(existing_extra or [])
    msgs = []

    for content, fname in zip(contents_list, filenames):
        _, encoded = content.split(",")
        raw = base64.b64decode(encoded)
        rec = load_battery_csv(fname, raw_bytes=raw)
        if rec is None:
            msgs.append(f"⚠ Skipped '{fname}' — filename doesn't match pattern.")
            continue
        # Store only serialisable metadata (not DataFrames)
        new_records.append({
            "battery":      rec["battery"],
            "discharge_mA": rec["discharge_mA"],
            "charge_mA":    rec["charge_mA"],
            "energy_mWh":   rec["energy_mWh"],
            "max_voltage":  rec["max_voltage"],
            "source_file":  rec["source_file"],
            "_csv_b64":     encoded,
        })
        msgs.append(f"✓ Loaded '{fname}'")

    # Rebuild options from disk + extras
    batteries = load_all_batteries()
    extra_bat_names = {r["battery"] for r in new_records}
    all_names = sorted(set(batteries.keys()) | extra_bat_names)
    opts  = [{"label": n, "value": n} for n in all_names]
    value = all_names

    return opts, value, new_records, html.Ul([html.Li(m) for m in msgs])


@callback(
    Output("bat-selector",  "options", allow_duplicate=True),
    Output("bat-selector",  "value",   allow_duplicate=True),
    Input("bat-selector",   "id"),     # fires on page load
    prevent_initial_call=False,
)
def init_selector(_):
    batteries = load_all_batteries()
    all_names = sorted(batteries.keys())
    opts  = [{"label": n, "value": n} for n in all_names]
    return opts, all_names


@callback(
    Output("bat-cap-fig",    "figure"),
    Output("bat-time-fig",   "figure"),
    Output("bat-energy-fig", "figure"),
    Output("bat-summary-table", "children"),
    Input("bat-selector",    "value"),
    State("bat-extra-store", "data"),
)
def update_charts(selected, extra_store):
    if not selected:
        empty = go.Figure()
        empty.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        return empty, empty, empty, "No batteries selected."

    # Reconstruct extra records with DataFrames from stored base64
    extra_records = []
    for r in (extra_store or []):
        raw = base64.b64decode(r["_csv_b64"])
        rec = load_battery_csv(r["source_file"], raw_bytes=raw)
        if rec:
            extra_records.append(rec)

    batteries = load_all_batteries(extra_records)
    selected  = [s for s in selected if s in batteries]

    rows = build_summary_table(batteries, selected)
    if rows:
        tbl = dbc.Table(
            [html.Thead(html.Tr([html.Th(k) for k in rows[0].keys()]))] +
            [html.Tbody([html.Tr([html.Td(row[k]) for k in row]) for row in rows])],
            striped=True, bordered=True, hover=True, size="sm",
            className="text-white",
        )
    else:
        tbl = "No data."

    return (
        build_capacity_fig(batteries, selected),
        build_time_fig(batteries, selected),
        build_energy_fig(batteries, selected),
        tbl,
    )
