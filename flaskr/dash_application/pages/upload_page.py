from dash import dcc, html
import dash_bootstrap_components as dbc

def layout() -> html.Div:
    return html.Div([
        dcc.Store(id="dataset-store", storage_type="session"),

        dbc.Row(
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        html.H4("Step 1: Upload Dataset", className="card-title mb-0"),
                        className="card-header-gradient"
                    ),

                    dbc.CardBody([
                        html.P(
                            "Upload a CSV dataset to start the analysis workflow.",
                            className="text-muted mb-4"
                        ),

                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                html.I(className="fas fa-file-upload me-2"),
                                html.Span("Drag and Drop or "),
                                html.A("Select CSV File")
                            ]),
                            style={
                                'width': '100%',
                                'height': '100px',
                                'lineHeight': '100px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '12px',
                                'textAlign': 'center',
                                'margin-bottom': '20px',
                                'backgroundColor': '#f8f9fa',
                                'cursor': 'pointer'
                            },
                            multiple=False
                        ),

                        html.Div(
                            id='upload-output',
                            children=dbc.Alert(
                                "No dataset uploaded yet.",
                                color="light",
                                className="mt-3"
                            )
                        ),
                    ]),

                    dbc.CardFooter(
                    html.Div([
                        dbc.Button(
                            [html.I(className="fas fa-arrow-right me-2"), "Next"],
                            id="upload-next-btn",
                            color="primary",
                            className="nav-btn"
                        ),
                        html.Div(
                            id="upload-warning",
                            className="mt-2"
                        )
                    ], className="d-flex flex-column align-items-end"),
                    className="card-footer-gradient"
                )
                ], className="main-card"),
                width=10
            ),
            justify="center",
            className="py-4"
        )
    ])