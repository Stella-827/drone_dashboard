"""
Drone Flight Optimizer Dashboard
---------------------------------
Based on Kyle & Jose Colab notebooks:
  - Battery Characterization V1
  - Motor Characterization V1
  - Flight Time Optimizer V3

Run:  python app.py
Then open http://127.0.0.1:8050 in your browser.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Drone Flight Optimizer",
)
server = app.server  # for deployment (Gunicorn / Railway / Render)

# ── Sidebar nav ──────────────────────────────────────────────────────────────
sidebar = dbc.Nav(
    [
        html.Div(
            [
                html.Img(src="/assets/logo.svg", height="40px", className="me-2"),
                html.Span("Flight Optimizer", className="fs-5 fw-bold text-white"),
            ],
            className="d-flex align-items-center px-3 py-3 border-bottom border-secondary",
        ),
        dbc.NavLink("🔋 Battery", href="/battery", active="exact", className="py-3"),
        dbc.NavLink("⚙️ Motor",   href="/motor",   active="exact", className="py-3"),
        dbc.NavLink("💨 Thrust",  href="/thrust",  active="exact", className="py-3"),
        dbc.NavLink("🚁 Optimizer", href="/optimizer", active="exact", className="py-3"),
    ],
    vertical=True,
    pills=True,
    className="bg-dark border-end border-secondary",
    style={"width": "200px", "minHeight": "100vh", "position": "fixed", "top": 0, "left": 0},
)

app.layout = html.Div(
    [
        sidebar,
        html.Div(
            [dash.page_container],
            style={"marginLeft": "200px", "padding": "1.5rem"},
        ),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
