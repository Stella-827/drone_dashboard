"""
Battery Characterization Page
Mirrors: Kyle_&_Jose_Battery_Characterization_V1

Fixes vs v2:
- ALL batteries (file-based + manual) are stored in one unified store and deletable
- Energy vs Power graph correctly shows fit lines for manual batteries with m/b coefficients
- Single-point manual batteries show a marker + model line extrapolated from m/b
"""

import base64, io
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL
import dash_bootstrap_components as dbc

from data_engine import load_all_batteries, load_battery_csv, fit_battery_energy_model, NOMINAL_V

dash.register_page(__name__, path="/battery", name="Battery", title="Battery Characterization")

COLORS = px.colors.qualitative.Plotly


# ── Helpers ───────────────────────────────────────────────────────────────────

def _initial_store():
    """
    Build the initial store from files in data/batteries/.
    Each entry: {name, discharge_mA, charge_mA, energy_mWh, max_voltage,
                 source, _csv_b64 (file-based only) OR m/b (manual only)}
    """
    batteries = load_all_batteries()
    store = {}
    for bat_name, runs in batteries.items():
        fit = fit_battery_energy_model(runs)
        for r in runs:
            key = f"{bat_name}||{r['discharge_mA']}"
            store[key] = {
                "name":         bat_name,
                "discharge_mA": r["discharge_mA"],
                "charge_mA":    r["charge_mA"],
                "energy_mWh":   r["energy_mWh"],
                "max_voltage":  r["max_voltage"],
                "source":       "file",
                "m":            fit["m"],
                "b":            fit["b"],
            }
    return store


def _store_to_batteries(store):
    """
    Re-group store entries into {bat_name: [run_dict, ...]} for charting.
    File-based entries carry _csv_b64 so we can reconstruct DataFrames.
    """
    grouped = {}
    for key, entry in (store or {}).items():
        name = entry["name"]
        grouped.setdefault(name, []).append((key, entry))
    return grouped


def build_capacity_fig(store, selected_names):
    fig = go.Figure()
    grouped = _store_to_batteries(store)
    color_idx = 0
    for bat_name in selected_names:
        if bat_name not in grouped:
            continue
        for key, entry in sorted(grouped[bat_name], key=lambda x: x[1]["discharge_mA"]):
            color = COLORS[color_idx % len(COLORS)]
            color_idx += 1
            if "_csv_b64" not in entry:
                continue  # manual entries have no raw curves
            rec = load_battery_csv(entry["_fname"], raw_bytes=base64.b64decode(entry["_csv_b64"]))
            if not rec:
                continue
            dis = rec["df_discharge"]
            chg = rec["df_charge"]
            fig.add_trace(go.Scatter(
                x=dis["Q discharge/mA.h"], y=dis["Ewe/V"],
                name=f"{bat_name} – Discharge @ {entry['discharge_mA']:.0f} mA",
                line=dict(color=color, width=2)))
            if not chg.empty:
                fig.add_trace(go.Scatter(
                    x=chg["Q charge/mA.h"], y=chg["Ewe/V"],
                    name=f"{bat_name} – Charge @ {entry['charge_mA']:.0f} mA",
                    line=dict(color=color, width=2, dash="dash")))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title="Voltage vs. Capacity  (file-based batteries only)",
        xaxis_title="Capacity (mAh)", yaxis_title="Voltage (V)",
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50))
    if not fig.data:
        fig.add_annotation(text="No file-based batteries selected — capacity curves require raw CSV data",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(color="#aaa", size=14))
    return fig


def build_time_fig(store, selected_names):
    fig = go.Figure()
    grouped = _store_to_batteries(store)
    color_idx = 0
    for bat_name in selected_names:
        if bat_name not in grouped:
            continue
        for key, entry in sorted(grouped[bat_name], key=lambda x: x[1]["discharge_mA"]):
            color = COLORS[color_idx % len(COLORS)]
            color_idx += 1
            if "_csv_b64" not in entry:
                continue
            rec = load_battery_csv(entry["_fname"], raw_bytes=base64.b64decode(entry["_csv_b64"]))
            if not rec:
                continue
            df = rec["df_full"]
            t  = df["time/s"] / 3600
            fig.add_trace(go.Scatter(
                x=t[df["ox/red"] == 0], y=df["Ewe/V"][df["ox/red"] == 0],
                name=f"{bat_name} – Discharge @ {entry['discharge_mA']:.0f} mA",
                line=dict(color=color, width=2)))
            fig.add_trace(go.Scatter(
                x=t[df["ox/red"] == 1], y=df["Ewe/V"][df["ox/red"] == 1],
                name=f"{bat_name} – Charge @ {entry['charge_mA']:.0f} mA",
                line=dict(color=color, width=2, dash="dash")))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title="Voltage vs. Time  (file-based batteries only)",
        xaxis_title="Time (hours)", yaxis_title="Voltage (V)",
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50))
    if not fig.data:
        fig.add_annotation(text="No file-based batteries selected — time curves require raw CSV data",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(color="#aaa", size=14))
    return fig


def build_energy_fig(store, selected_names):
    """
    Energy vs Power plot (Battery Colab Step 4).
    - File-based batteries: scatter points + linear fit line
    - Manual batteries with m & b: fit line extrapolated + single marker at their data point
    - Manual batteries with only energy_mWh: single star marker
    """
    fig = go.Figure()
    grouped = _store_to_batteries(store)
    color_idx = 0

    for bat_name in selected_names:
        if bat_name not in grouped:
            continue
        entries = sorted(grouped[bat_name], key=lambda x: x[1]["discharge_mA"])
        color   = COLORS[color_idx % len(COLORS)]
        color_idx += 1

        # Collect (P_mW, E_mWh) points
        points = [(e["discharge_mA"] * NOMINAL_V, e["energy_mWh"]) for _, e in entries]
        P_arr  = np.array([p[0] for p in points])
        E_arr  = np.array([p[1] for p in points])
        source = entries[0][1].get("source", "manual")

        # --- Scatter points ---
        fig.add_trace(go.Scatter(
            x=P_arr, y=E_arr,
            mode="markers+text",
            text=[f"{e['discharge_mA']:.0f} mA" for _, e in entries],
            textposition="top right",
            marker=dict(size=10, color=color,
                        symbol="circle" if source == "file" else "star"),
            name=f"{bat_name} ({'file' if source == 'file' else 'manual'})"))

        # --- Fit / model line ---
        m = entries[0][1].get("m", 0.0)
        b = entries[0][1].get("b", 0.0)

        if len(P_arr) >= 2 and source == "file":
            # Recompute fit from actual data points (most accurate)
            fit = fit_battery_energy_model([
                {"discharge_mA": e["discharge_mA"], "energy_mWh": e["energy_mWh"]}
                for _, e in entries
            ])
            m, b = fit["m"], fit["b"]
            label = f"{bat_name}: {fit['eq_str']}"
        elif m != 0.0 or b != 0.0:
            label = f"{bat_name}: E = {m:.4f}·P + {b:.2f} (manual model)"
        else:
            continue  # no line to draw

        P_max = max(float(P_arr.max()) * 1.5, 500.0)
        P_fit = np.linspace(0, P_max, 300)
        E_fit = m * P_fit + b
        # Only plot where E_fit > 0
        mask  = E_fit > 0
        fig.add_trace(go.Scatter(
            x=P_fit[mask], y=E_fit[mask],
            mode="lines",
            name=label,
            line=dict(color=color, width=2, dash="dot")))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        title="Total Energy vs. Discharge Power  (E = m·P + b, Battery Colab Step 4)",
        xaxis_title="Discharge Power P (mW)", yaxis_title="Total Energy E (mWh)",
        xaxis=dict(rangemode="tozero"), yaxis=dict(rangemode="tozero"),
        legend=dict(orientation="v", x=1.01),
        margin=dict(l=50, r=10, t=50, b=50))
    return fig


def build_summary_table(store, selected_names):
    grouped = _store_to_batteries(store)
    rows = []
    for bat_name in selected_names:
        if bat_name not in grouped:
            continue
        entries = sorted(grouped[bat_name], key=lambda x: x[1]["discharge_mA"])
        m = entries[0][1].get("m", 0.0)
        b = entries[0][1].get("b", 0.0)
        for _, e in entries:
            rows.append({
                "Battery":        bat_name,
                "Source":         e.get("source", "manual"),
                "Discharge (mA)": f"{e['discharge_mA']:.0f}",
                "Charge (mA)":    f"{e['charge_mA']:.0f}",
                "Energy (mWh)":   f"{e['energy_mWh']:.2f}",
                "Max V":          f"{e['max_voltage']:.4f}" if e['max_voltage'] else "—",
                "Model":          f"m={m:.4f}, b={b:.2f}",
            })
    if not rows:
        return html.Div("No batteries selected.")
    return dbc.Table(
        [html.Thead(html.Tr([html.Th(k) for k in rows[0]]))] +
        [html.Tbody([html.Tr([html.Td(r[k]) for k in r]) for r in rows])],
        striped=True, bordered=True, hover=True, size="sm", className="text-white")


def battery_list_ui(store):
    """Render all batteries with delete buttons — every entry is deletable."""
    grouped = _store_to_batteries(store or {})
    if not grouped:
        return html.Div("No batteries saved yet.", className="text-muted small")
    items = []
    for bat_name in sorted(grouped):
        entries = sorted(grouped[bat_name], key=lambda x: x[1]["discharge_mA"])
        source  = entries[0][1].get("source", "manual")
        tag     = "📄" if source == "file" else "🖊"
        rates   = ", ".join(f"{e['discharge_mA']:.0f}mA" for _, e in entries)
        items.append(dbc.ListGroupItem(
            dbc.Row([
                dbc.Col(html.Div([
                    html.Span(f"{tag} {bat_name}", className="small fw-bold"),
                    html.Br(),
                    html.Span(f"Discharge: {rates}", className="small text-muted"),
                ]), width=9),
                dbc.Col(
                    dbc.Button("✕", id={"type": "bat-delete-btn", "index": bat_name},
                               color="danger", size="sm", className="py-0 px-2"),
                    width=3, className="text-end"),
            ], align="center"),
            className="bg-transparent border-secondary text-white py-1",
        ))
    return dbc.ListGroup(items, flush=True)


# ── Layout ────────────────────────────────────────────────────────────────────
layout = dbc.Container(fluid=True, children=[
    dcc.Store(id="bat-store", data=None),   # None = not yet initialised
    dcc.Store(id="bat-init-trigger", data=0),

    html.H3("🔋 Battery Characterization", className="mb-4 mt-2 text-info"),

    dbc.Row([
        dbc.Col([
            dbc.Accordion([
                dbc.AccordionItem(title="➕  Add Battery", children=[
                    dbc.Label("Battery Name", className="small mb-0"),
                    dbc.Input(id="bat-name", placeholder="e.g. MyBattery 500mg", size="sm", className="mb-2"),

                    dbc.Label("Mass (mg)", className="small mb-0"),
                    dbc.Input(id="bat-mass", type="number", placeholder="e.g. 500", size="sm", className="mb-2"),

                    dbc.Label("Discharge Current (mA)", className="small mb-0"),
                    dbc.Input(id="bat-dis-mA", type="number", placeholder="e.g. 300", size="sm", className="mb-2"),

                    dbc.Label("Charge Current (mA)", className="small mb-0"),
                    dbc.Input(id="bat-chg-mA", type="number", placeholder="e.g. 5", size="sm", className="mb-2"),

                    dbc.Label("Total Energy (mWh)", className="small mb-0"),
                    dbc.Input(id="bat-energy", type="number", placeholder="e.g. 28.3", size="sm", className="mb-2"),

                    dbc.Label("Max Charge Voltage (V)", className="small mb-0"),
                    dbc.Input(id="bat-maxV", type="number", placeholder="e.g. 4.2", size="sm", className="mb-2"),

                    html.Hr(className="border-secondary my-2"),
                    html.P("Optional: linear model  E = m·P + b", className="small text-muted mb-1"),
                    dbc.Label("m (slope, mWh/mW)", className="small mb-0"),
                    dbc.Input(id="bat-m", type="number", placeholder="e.g. -0.042", size="sm", className="mb-2"),
                    dbc.Label("b (intercept, mWh)", className="small mb-0"),
                    dbc.Input(id="bat-b", type="number", placeholder="e.g. 54.0", size="sm", className="mb-2"),

                    dbc.Button("Save Battery", id="bat-save-btn", color="success", size="sm", className="w-100 mt-1"),
                    html.Div(id="bat-save-status", className="small mt-2"),
                ]),
            ], start_collapsed=True, className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Saved Batteries  (📄 file · 🖊 manual)"),
                dbc.CardBody(html.Div(id="bat-list-ui")),
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Plot Selection"),
                dbc.CardBody(dcc.Checklist(
                    id="bat-selector", options=[], value=[],
                    labelStyle={"display": "block", "marginBottom": "6px"},
                    inputStyle={"marginRight": "8px"})),
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


# ── Initialise store from files on first load ─────────────────────────────────
@callback(
    Output("bat-store", "data", allow_duplicate=True),
    Input("bat-init-trigger", "data"),
    State("bat-store", "data"),
    prevent_initial_call='initial_duplicate',
)
def init_store(_, existing):
    if existing:
        return dash.no_update
    return _initial_store()


# ── Save a new battery ────────────────────────────────────────────────────────
@callback(
    Output("bat-store",       "data",    allow_duplicate=True),
    Output("bat-save-status", "children"),
    Output("bat-name",        "value"),
    Output("bat-mass",        "value"),
    Output("bat-dis-mA",      "value"),
    Output("bat-chg-mA",      "value"),
    Output("bat-energy",      "value"),
    Output("bat-maxV",        "value"),
    Output("bat-m",           "value"),
    Output("bat-b",           "value"),
    Input("bat-save-btn",     "n_clicks"),
    State("bat-name",         "value"),
    State("bat-mass",         "value"),
    State("bat-dis-mA",       "value"),
    State("bat-chg-mA",       "value"),
    State("bat-energy",       "value"),
    State("bat-maxV",         "value"),
    State("bat-m",            "value"),
    State("bat-b",            "value"),
    State("bat-store",        "data"),
    prevent_initial_call=True,
)
def save_battery(_, name, mass, dis_mA, chg_mA, energy, maxV, m, b, store):
    store = dict(store or {})
    nd    = dash.no_update
    if not name or not name.strip():
        return store, "⚠ Name required.", nd, nd, nd, nd, nd, nd, nd, nd
    if energy is None:
        return store, "⚠ Total energy required.", nd, nd, nd, nd, nd, nd, nd, nd

    dis_mA = float(dis_mA or 0)
    key    = f"{name.strip()}||{dis_mA}"
    store[key] = {
        "name":         name.strip(),
        "discharge_mA": dis_mA,
        "charge_mA":    float(chg_mA or 0),
        "energy_mWh":   float(energy),
        "max_voltage":  float(maxV or 0),
        "source":       "manual",
        "m":            float(m or 0),
        "b":            float(b or energy),
    }
    return store, f"✓ Saved '{name.strip()}'", "", None, None, None, None, None, None, None


# ── Delete a battery ──────────────────────────────────────────────────────────
@callback(
    Output("bat-store", "data", allow_duplicate=True),
    Input({"type": "bat-delete-btn", "index": ALL}, "n_clicks"),
    State("bat-store", "data"),
    prevent_initial_call=True,
)
def delete_battery(_, store):
    store     = dict(store or {})
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return store
    bat_name  = triggered.get("index")
    # Remove all keys belonging to this battery name
    to_del = [k for k in store if store[k]["name"] == bat_name]
    for k in to_del:
        del store[k]
    return store


# ── Refresh list + selector ───────────────────────────────────────────────────
@callback(
    Output("bat-list-ui",  "children"),
    Output("bat-selector", "options"),
    Output("bat-selector", "value"),
    Input("bat-store",     "data"),
)
def refresh_list(store):
    grouped   = _store_to_batteries(store or {})
    all_names = sorted(grouped.keys())
    list_ui   = battery_list_ui(store)
    opts      = [{"label": n, "value": n} for n in all_names]
    return list_ui, opts, all_names


# ── Update charts ─────────────────────────────────────────────────────────────
@callback(
    Output("bat-cap-fig",       "figure"),
    Output("bat-time-fig",      "figure"),
    Output("bat-energy-fig",    "figure"),
    Output("bat-summary-table", "children"),
    Input("bat-selector",       "value"),
    State("bat-store",          "data"),
)
def update_charts(selected, store):
    empty = go.Figure()
    empty.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    if not selected or not store:
        return empty, empty, empty, "No batteries selected."
    return (
        build_capacity_fig(store, selected),
        build_time_fig(store, selected),
        build_energy_fig(store, selected),
        build_summary_table(store, selected),
    )
