from dash import dcc, html
import dash_bootstrap_components as dbc

def layout() -> html.Div:
    return html.Div([
        dcc.Store(id='inference-data'),
        dcc.Store(id='rule-memberships', data={}),
        dcc.Store(id='is-classification', data=False),
        dcc.Store(id="winner-term-store", data={}),
        dcc.Store(id="inference-input-ids", data=[]),

        html.Div(
            id="inference-content",
            className="content",
            style={"display": "block", "position": "relative"},
            children=[
                dbc.Row(
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4("Test Inference Fuzzy", className="card-title mb-0"),
                                dbc.Badge("Simulation", color="warning", className="ml-2")
                            ], className="card-header-gradient d-flex justify-content-between align-items-center"),
                            dbc.CardBody(
                                html.Div(id="test-page-content")
                            )
                        ], className="main-card"),
                        width={"size": 10, "offset": 1},
                        className="py-4"
                    )
                )
            ]
        )
    ])
