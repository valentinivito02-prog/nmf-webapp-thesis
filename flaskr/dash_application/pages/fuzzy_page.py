from dash import dcc, html
import dash_bootstrap_components as dbc


def layout() -> html.Div:
    return html.Div([
        dcc.Store(id="dataset-store", storage_type="session"),
        dcc.Store(id="nmf-results-store", storage_type="session"),
        dcc.Store(id="fuzzy-settings", storage_type="session"),
        dcc.Store(id="fuzzy-results", storage_type="session"),

        dcc.Download(id="download-w-explanations"),
        dcc.Download(id="download-h-explanations"),
        dcc.Download(id="download-example-explanations"),

        dbc.Row(
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        html.H4(
                            "Step 4: Fuzzy Explanations",
                            className="card-title mb-0"
                        ),
                        className="card-header-gradient"
                    ),

                    dbc.CardBody([

                        html.P(
                            "Configure the fuzzy explanation settings to interpret "
                            "the NMF results.",
                            className="text-muted mb-4"
                        ),

                        dbc.Row([

                            dbc.Col([
                                html.H5(
                                    "Number of Fuzzy Sets",
                                    className="mb-3"
                                ),

                                dcc.Input(
                                    id="num-fuzzy-sets",
                                    type="number",
                                    min=2,
                                    value=3,
                                    style={"width": "100%"}
                                ),
                            ], md=4),

                            dbc.Col([
                                html.H5(
                                    "Creation Method",
                                    className="mb-3"
                                ),

                                dcc.RadioItems(
                                    id="fuzzy-method",
                                    options=[
                                        {
                                            "label": " Equidistant",
                                            "value": "equidistant"
                                        },
                                        {
                                            "label": " Quartile-based",
                                            "value": "quartile"
                                        },
                                        {
                                            "label": " Manual (not implemented)",
                                            "value": "manual"
                                        },
                                    ],
                                    value="equidistant"
                                ),
                            ], md=4),

                            dbc.Col([
                                html.H5(
                                    "Membership Function Shape",
                                    className="mb-3"
                                ),

                                dcc.Dropdown(
                                    id="fuzzy-shape",
                                    options=[
                                        {
                                            "label": "Gaussian",
                                            "value": "gaussian"
                                        },
                                        {
                                            "label": "Triangular",
                                            "value": "triangular"
                                        },
                                        {
                                            "label": "Trapezoidal",
                                            "value": "trapezoidal"
                                        },
                                    ],
                                    value="gaussian"
                                ),
                            ], md=4),

                        ], className="mb-4"),

                        html.H5(
                            "Apply to",
                            className="mb-3"
                        ),

                        dbc.Card(
                            dbc.CardBody([
                                dcc.Checklist(
                                    id="fuzzy-target",
                                    options=[
                                        {
                                            "label": " Matrix W (Latent Factors)",
                                            "value": "W"
                                        },
                                        {
                                            "label": " Matrix H (Clusters)",
                                            "value": "H"
                                        },
                                    ],
                                    value=["W"]
                                ),
                            ]),
                            className="mb-4"
                        ),

                        html.Div(
                            dbc.Button(
                                "Generate Fuzzy Explanations",
                                id="run-fuzzy",
                                color="primary",
                                className="px-4"
                            ),
                            className="d-flex justify-content-center mb-2"
                        ),

                        html.Div(
                            id="fuzzy-run-warning",
                            className="mt-2"
                        ),

                        html.H5(
                            "Results",
                            className="mb-3 mt-4"
                        ),

                        dbc.Card(
                            dbc.CardBody([

                                html.Div(
                                    dcc.Tabs(
                                        id="fuzzy-results-tabs",
                                        value="tab-fuzzy-w",

                                        colors={
                                            "border": "#dee2e6",
                                            "primary": "#52b2cf",
                                            "background": "#f8f9fa"
                                        },

                                        children=[

                                            dcc.Tab(
                                                label="W Explanations",
                                                value="tab-fuzzy-w",

                                                style={
                                                    "padding": "10px",
                                                    "fontWeight": "500"
                                                },

                                                selected_style={
                                                    "padding": "10px",
                                                    "fontWeight": "600",
                                                    "borderTop": "3px solid #52b2cf",
                                                    "backgroundColor": "white"
                                                },

                                                children=[
                                                    html.Div(
                                                        id="fuzzy-w-output",

                                                        children=dbc.Alert(
                                                            "W explanations will "
                                                            "be displayed here.",
                                                            color="light"
                                                        ),

                                                        className="mt-3"
                                                    )
                                                ]
                                            ),

                                            dcc.Tab(
                                                label="H Explanations",
                                                value="tab-fuzzy-h",

                                                style={
                                                    "padding": "10px",
                                                    "fontWeight": "500"
                                                },

                                                selected_style={
                                                    "padding": "10px",
                                                    "fontWeight": "600",
                                                    "borderTop": "3px solid #52b2cf",
                                                    "backgroundColor": "white"
                                                },

                                                children=[
                                                    html.Div(
                                                        id="fuzzy-h-output",

                                                        children=dbc.Alert(
                                                            "H explanations will "
                                                            "be displayed here.",
                                                            color="light"
                                                        ),

                                                        className="mt-3"
                                                    )
                                                ]
                                            ),

                                            dcc.Tab(
                                                label="Example Interpretations",
                                                value="tab-fuzzy-examples",

                                                style={
                                                    "padding": "10px",
                                                    "fontWeight": "500"
                                                },

                                                selected_style={
                                                    "padding": "10px",
                                                    "fontWeight": "600",
                                                    "borderTop": "3px solid #52b2cf",
                                                    "backgroundColor": "white"
                                                },

                                                children=[
                                                    html.Div(
                                                        id="fuzzy-examples",

                                                        children=dbc.Alert(
                                                            "Example explanations "
                                                            "will be displayed here.",
                                                            color="light"
                                                        ),

                                                        className="mt-3"
                                                    )
                                                ]
                                            ),

                                            dcc.Tab(
                                                label="Configuration Summary",
                                                value="tab-fuzzy-summary",

                                                style={
                                                    "padding": "10px",
                                                    "fontWeight": "500"
                                                },

                                                selected_style={
                                                    "padding": "10px",
                                                    "fontWeight": "600",
                                                    "borderTop": "3px solid #52b2cf",
                                                    "backgroundColor": "white"
                                                },

                                                children=[
                                                    html.Div(
                                                        id="fuzzy-summary",

                                                        children=dbc.Alert(
                                                            "Fuzzy configuration "
                                                            "summary will be "
                                                            "displayed here.",
                                                            color="light"
                                                        ),

                                                        className="mt-3"
                                                    )
                                                ]
                                            ),

                                        ]
                                    ),

                                    style={
                                        "backgroundColor": "#f8f9fa",
                                        "borderRadius": "8px",
                                        "padding": "5px"
                                    }
                                ),

                                html.Div([

                                    dbc.Button(
                                        [
                                            html.I(
                                                className="fas fa-download me-2"
                                            ),
                                            "Download W Explanations"
                                        ],
                                        id="download-w-explanations-btn",
                                        color="success",
                                        className="me-2"
                                    ),

                                    dbc.Button(
                                        [
                                            html.I(
                                                className="fas fa-download me-2"
                                            ),
                                            "Download H Explanations"
                                        ],
                                        id="download-h-explanations-btn",
                                        color="secondary",
                                        className="me-2"
                                    ),

                                    dbc.Button(
                                        [
                                            html.I(
                                                className="fas fa-download me-2"
                                            ),
                                            "Download Examples"
                                        ],
                                        id="download-example-explanations-btn",
                                        color="primary"
                                    ),

                                ], className="d-flex justify-content-end mt-3"),

                                html.Hr(className="my-4"),

                                # --------------------------------------------------
                                # Sezione API Export / Integrazione con Fuxplainer
                                # --------------------------------------------------

                                dbc.Alert(
                                    [
                                        html.I(className="fas fa-plug me-2"),

                                        html.Strong(
                                            "Explain with Fuxplainer: "
                                        ),

                                        "Fuzzy results are exposed via API "
                                        "to the endpoint ",

                                        html.Code(
                                            "GET http://localhost:5000/api/explanations",
                                            style={"fontSize": "0.85em"}
                                        ),

                                        ". Fuxplainer can retrieve them "
                                        "automatically by pressing the "
                                        "button below or from the "
                                        "Fuxplainer data upload page "
                                        "(Step 1)."
                                    ],

                                    color="info",
                                    className="mb-3"
                                ),

                                html.Div([

                                    dbc.Button(
                                        [
                                            html.I(
                                                className="fas fa-share-alt me-2"
                                            ),
                                            "Send results to Fuxplainer"
                                        ],
                                        id="send-to-fuxplainer-btn",
                                        color="info",
                                        outline=True,
                                        className="me-2"
                                    ),

                                    dbc.Button(
                                        [
                                            html.I(
                                                className="fas fa-check-circle me-2"
                                            ),
                                            "Check API availability"
                                        ],
                                        id="check-api-btn",
                                        color="secondary",
                                        outline=True,
                                    ),

                                ], className="d-flex justify-content-end mt-2"),

                                html.Div(
                                    id="fuxplainer-status",
                                    className="mt-2"
                                ),

                            ]),
                            className="mb-4"
                        ),

                    ]),

                    dbc.CardFooter(
                        html.Div([

                            dbc.Button(
                                [
                                    html.I(
                                        className="fas fa-arrow-left me-2"
                                    ),
                                    "Back"
                                ],

                                href="/run",
                                color="light",
                                className="nav-btn me-2"
                            ),

                            html.Div(
                                id="fuzzy-warning",
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