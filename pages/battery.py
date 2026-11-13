"""
Battery Characterization Page — manual entry, no CSV upload
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL
import dash_bootstrap_components as dbc

from data_engine import load_all_batteries, fit_battery_energy_model, NOMINAL_V

dash.register_page(__name__, path="/battery", name="Battery", title="Battery Characterization")

TAB_COLORS = px.colors.qualitative.Plotly

# ── Chart builders ────────────────────────────────────────────────────────────
def build_capacity_fig(batteries: dict, selected: list) -> go.Figure:
    fig = go.Figure()
    for bat_name in selected:
        runs = batteries.get(bat_name, [])
        for i, r in enumerate(runs):
            color = TAB_COLORS[i % len(TAB_COLORS)]
            dis = r["df_discharge"]
            fig.add_trace(go.Scatter(
                x=dis["Q discharge/mA.h"], y=dis["Ewe/V"],
                name=f"{bat_name} – Discharge @ {r['discharge_mA']:.0f} mA",
                line=dict(color=color, width=2),
            ))
            chg = r["df_charge"]
            if not chg.empty:
                fig.add_trace(go.Scatter(
                    x=chg["Q charge/mA.h"], y=chg["Ewe/V"],
                    name=f"{bat_name} – Charge @ {r['charge_mA']:.0f} mA",
                    line=dict(color=color, width=2, dash="dash"),
                ))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title="Voltage vs. Capacity (Charge & Discharge)",
                      xaxis_title="Capacity (mAh)", yaxis_title="Voltage (V)",
                      legend=dict(orientation="v", x=1.01),
                      margin=dict(l=50, r=10, t=50, b=50))
    return fig


def build_time_fig(batteries: dict, selected: list) -> go.Figure:
    fig = go.Figure()
    for bat_name in selected:
        runs = batteries.get(bat_name, [])
        for i, r in enumerate(runs):
            color = TAB_COLORS[i % len(TAB_COLORS)]
            df = r["df_full"]
            t_hours = df["time/s"] / 3600
            dis_mask = df["ox/red"] == 0
            chg_mask = df["ox/red"] == 1
            fig.add_trace(go.Scatter(x=t_hours[dis_mask], y=df["Ewe/V"][dis_mask],
                                     name=f"{bat_name} – Discharge @ {r['discharge_mA']:.0f} mA",
                                     line=dict(color=color, width=2)))
            fig.add_trace(go.Scatter(x=t_hours[chg_mask], y=df["Ewe/V"][chg_mask],
                                     name=f"{bat_name} – Charge @ {r['charge_mA']:.0f} mA",
                                     line=dict(color=color, width=2, dash="dash")))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title="Voltage vs. Time",
                      xaxis_title="Time (hours)", yaxis_title="Voltage (V)",
                      legend=dict(orientation="v", x=1.01),
                      margin=dict(l=50, r=10, t=50, b=50))
    return fig


def build_energy_fig(batteries: dict, selected: list, manual_store: dict) -> go.Figure:
    fig = go.Figure()
    # Empirical batteries from CSV files
    for bat_name in selected:
        runs = batteries.get(bat_name, [])
        if not runs:
            continue
        fit = fit_battery_energy_model(runs)
        powers   = np.array([r["discharge_mA"] * NOMINAL_V for r in runs])
        energies = np.array([r["energy_mWh"] for r in runs])
        labels   = [f"{r['discharge_mA']:.0f} mA" for r in runs]
        fig.add_trace(go.Scatter(x=powers, y=energies, mode="markers+text",
                                 text=labels, textposition="top right",
                                 marker=dict(size=10), name=f"{bat_name} data"))
        if len(powers) >= 2:
            P_fit = np.linspace(0, powers.max() * 1.3, 200)
            fig.add_trace(go.Scatter(x=P_fit, y=fit["m"] * P_fit + fit["b"], mode="lines",
                                     name=f"{bat_name} – {fit['eq_str']}",
                                     line=dict(width=2, dash="dot")))

    # Manual batteries — plot as single point (P, E) from their m/b model
    for bat_name, bdata in (manual_store or {}).items():
        if bat_name not in selected:
            continue
        P_mW = float(bdata.get("discharge_mA", 100)) * NOMINAL_V
        E_mWh = float(bdata.get("energy_mWh", 0))
        fig.add_trace(go.Scatter(x=[P_mW], y=[E_mWh], mode="markers",
                                 marker=dict(size=12, symbol="star"),
                                 name=f"{bat_name} (manual)"))

    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      title="Total Energy vs. Discharge Power",
                      xaxis_title="Discharge Power (mW)", yaxis_title="Total Energy (mWh)",
                      xaxis=dict(rangemode="tozero"), yaxis=dict(rangemode="tozero"),
                      legend=dict(orientation="v", x=1.01),
                      margin=dict(l=50, r=10, t=50, b=50))
    return fig


def build_summary_table(batteries: dict, selected: list, manual_store: dict) -> html.Div:
    rows = []
    for bat_name in selected:
        runs = batteries.get(bat_name, [])
        if runs:
            fit = fit_battery_energy_model(runs)
            for r in runs:
                rows.append({"Battery": bat_name, "Type": "Empirical",
                              "Discharge (mA)": f"{r['discharge_mA']:.0f}",
                              "Charge (mA)": f"{r['charge_mA']:.0f}",
                              "Energy (mWh)": f"{r['energy_mWh']:.2f}",
                              "Max V": f"{r['max_voltage']:.4f}",
                              "Fit": fit["eq_str"]})
        elif bat_name in (manual_store or {}):
            bdata = manual_store[bat_name]
            rows.append({"Battery": bat_name, "Type": "Manual",
                         "Discharge (mA)": str(bdata.get("discharge_mA", "—")),
                         "Charge (mA)": str(bdata.get("charge_mA", "—")),
                         "Energy (mWh)": str(bdata.get("energy_mWh", "—")),
                         "Max V": str(bdata.get("max_voltage", "—")),
                         "Fit": f"m={bdata.get('m', 0):.4f}, b={bdata.get('b', 0):.2f}"})
    if not rows:
        return html.Div("No batteries selected.")
    return dbc.Table(
        [html.Thead(html.Tr([html.Th(k) for k in rows[0]]))] +
        [html.Tbody([html.Tr([html.Td(r[k]) for k in r]) for r in rows])],
        striped=True, bordered=True, hover=True, size="sm", className="text-white")


def battery_list_ui(all_names: list, manual_names: set) -> html.Div:
    if not all_names:
        return html.Div("No batteries saved yet.", className="text-muted small")
    items = []
    for name in all_names:
        tag = " 🖊" if name in manual_names else " 📄"
        items.append(dbc.ListGroupItem(
            dbc.Row([
                dbc.Col(html.Span(name + tag, className="small"), width=9),
                dbc.Col(
                    dbc.Button("✕", id={"type": "bat-delete-btn", "index": name},
                               color="danger", size="sm", className="py-0 px-2",
                               disabled=(name not in manual_names)),
                    width=3, className="text-end"
                ),
            ], align="center"),
            className="bg-transparent border-secondary text-white py-1",
        ))
    return dbc.ListGroup(items, flush=True)


# ── Layout ────────────────────────────────────────────────────────────────────
layout = dbc.Container(fluid=True, children=[
    dcc.Store(id="bat-manual-store", data={}),

    html.H3("🔋 Battery Characterization", className="mb-4 mt-2 text-info"),

    dbc.Row([
        dbc.Col([
            # Add battery accordion
            dbc.Accordion([
                dbc.AccordionItem(
                    title="➕  Add Battery",
                    children=[
                        dbc.Label("Battery Name", className="small mb-0"),
                        dbc.Input(id="bat-name", placeholder="e.g. Polybattery 1300mg", size="sm", className="mb-2"),

                        dbc.Label("Mass (mg)", className="small mb-0"),
                        dbc.Input(id="bat-mass", type="number", placeholder="e.g. 1300", size="sm", className="mb-2"),

                        dbc.Label("Discharge Current (mA)", className="small mb-0"),
                        dbc.Input(id="bat-discharge-mA", type="number", placeholder="e.g. 300", size="sm", className="mb-2"),

                        dbc.Label("Charge Current (mA)", className="small mb-0"),
                        dbc.Input(id="bat-charge-mA", type="number", placeholder="e.g. 5", size="sm", className="mb-2"),

                        dbc.Label("Total Energy (mWh)", className="small mb-0"),
                        dbc.Input(id="bat-energy", type="number", placeholder="e.g. 28.3", size="sm", className="mb-2"),

                        dbc.Label("Max Charge Voltage (V)", className="small mb-0"),
                        dbc.Input(id="bat-maxV", type="number", placeholder="e.g. 4.2", size="sm", className="mb-2"),

                        html.Hr(className="border-secondary my-2"),
                        html.P("Optional: linear model coefficients  E = m·P + b", className="small text-muted mb-1"),

                        dbc.Label("m (slope)", className="small mb-0"),
                        dbc.Input(id="bat-m", type="number", placeholder="e.g. -0.042", size="sm", className="mb-2"),

                        dbc.Label("b (intercept, mWh)", className="small mb-0"),
                        dbc.Input(id="bat-b", type="number", placeholder="e.g. 54.0", size="sm", className="mb-2"),

                        dbc.Button("Save Battery", id="bat-save-btn",
                                   color="success", size="sm", className="w-100 mt-1"),
                        html.Div(id="bat-save-status", className="small mt-2"),
                    ],
                ),
            ], start_collapsed=True, className="mb-3"),

            # Saved batteries list
            dbc.Card([
                dbc.CardHeader("Saved Batteries  (📄 = from file, 🖊 = manual)"),
                dbc.CardBody(html.Div(id="bat-list-ui")),
            ], className="mb-3"),

            # Plot selection
            dbc.Card([
                dbc.CardHeader("Plot Selection"),
                dbc.CardBody(dcc.Checklist(
                    id="bat-selector", options=[], value=[],
                    labelStyle={"display": "block", "marginBottom": "6px"},
                    inputStyle={"marginRight": "8px"},
                )),
            ]),
        ], md=3),

        dbc.Col([
            dbc.Tabs([
                dbc.Tab(dcc.Graph(id="bat-cap-fig",    config={"displayModeBar": False}), label="Capacity Curves"),
                dbc.Tab(dcc.Graph(id="bat-time-fig",   config={"displayModeBar": False}), label="Voltage vs Time"),
                dbc.Tab(dcc.Graph(id="bat-energy-fig", config={"displayModeBar": False}), label="Energy vs Power"),
                dbc.Tab(html.Div(id="bat-summary-table", className="mt-3"),               label="Summary Table"),
            ]),
        ], md=9),
    ]),
])


# ── Save battery ──────────────────────────────────────────────────────────────
@callback(
    Output("bat-manual-store", "data"),
    Output("bat-save-status",  "children"),
    Output("bat-name",         "value"),
    Output("bat-mass",         "value"),
    Output("bat-discharge-mA", "value"),
    Output("bat-charge-mA",    "value"),
    Output("bat-energy",       "value"),
    Output("bat-maxV",         "value"),
    Output("bat-m",            "value"),
    Output("bat-b",            "value"),
    Input("bat-save-btn",      "n_clicks"),
    State("bat-name",          "value"),
    State("bat-mass",          "value"),
    State("bat-discharge-mA",  "value"),
    State("bat-charge-mA",     "value"),
    State("bat-energy",        "value"),
    State("bat-maxV",          "value"),
    State("bat-m",             "value"),
    State("bat-b",             "value"),
    State("bat-manual-store",  "data"),
    prevent_initial_call=True,
)
def save_battery(_, name, mass, dis_mA, chg_mA, energy, maxV, m, b, store):
    store = store or {}
    if not name or name.strip() == "":
        return (store, "⚠ Battery name is required.",
                *[dash.no_update] * 8)
    if energy is None:
        return (store, "⚠ Total energy is required.",
                *[dash.no_update] * 8)
    store[name.strip()] = {
        "mass_mg":      float(mass or 0),
        "discharge_mA": float(dis_mA or 0),
        "charge_mA":    float(chg_mA or 0),
        "energy_mWh":   float(energy),
        "max_voltage":  float(maxV or 0),
        "m":            float(m or 0),
        "b":            float(b or energy),
    }
    return store, f"✓ Saved '{name.strip()}'", "", None, None, None, None, None, None, None


# ── Delete battery ────────────────────────────────────────────────────────────
@callback(
    Output("bat-manual-store", "data", allow_duplicate=True),
    Input({"type": "bat-delete-btn", "index": ALL}, "n_clicks"),
    State("bat-manual-store", "data"),
    prevent_initial_call=True,
)
def delete_battery(_, store):
    store = store or {}
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict) and triggered.get("type") == "bat-delete-btn":
        store.pop(triggered["index"], None)
    return store


# ── Refresh list + selector ───────────────────────────────────────────────────
@callback(
    Output("bat-list-ui",  "children"),
    Output("bat-selector", "options"),
    Output("bat-selector", "value"),
    Input("bat-manual-store", "data"),
)
def refresh_battery_list(manual_store):
    manual_store = manual_store or {}
    file_batteries = load_all_batteries()
    all_names = sorted(set(file_batteries.keys()) | set(manual_store.keys()))
    manual_names = set(manual_store.keys())
    list_ui = battery_list_ui(all_names, manual_names)
    opts  = [{"label": n, "value": n} for n in all_names]
    value = all_names
    return list_ui, opts, value


# ── Update charts ─────────────────────────────────────────────────────────────
@callback(
    Output("bat-cap-fig",       "figure"),
    Output("bat-time-fig",      "figure"),
    Output("bat-energy-fig",    "figure"),
    Output("bat-summary-table", "children"),
    Input("bat-selector",       "value"),
    State("bat-manual-store",   "data"),
)
def update_charts(selected, manual_store):
    empty = go.Figure()
    empty.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    if not selected:
        return empty, empty, empty, "No batteries selected."

    batteries = load_all_batteries()
    return (
        build_capacity_fig(batteries, selected),
        build_time_fig(batteries, selected),
        build_energy_fig(batteries, selected, manual_store or {}),
        build_summary_table(batteries, selected, manual_store or {}),
    )
