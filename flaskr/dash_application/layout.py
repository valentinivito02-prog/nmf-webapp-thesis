import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


def serve_layout() -> html.Div:
    return html.Div([
        dcc.Location(id='url', refresh=False),
        dcc.Store(id="session-store", storage_type="session"),

        # Font Awesome
        html.Link(
            rel='stylesheet',
            href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
        ),

        # Google Fonts
        html.Link(
            rel='stylesheet',
            href='https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap'
        ),

        # Sidebar
        html.Div([
            html.Div(
                [
                    html.H2("NMF Web App", className="mb-2"),
                    html.P(
                        "Clustering & Fuzzy Analysis",
                        className="text-light small"
                    )
                ],
                className="sidebar-logo",
                style={"padding": "20px"}
            )
        ], className="sidebar"),

        # Main content
        html.Div([
            html.Div(id="page-content", className="content-inner")
        ], className="content"),

    ], className="app-container")