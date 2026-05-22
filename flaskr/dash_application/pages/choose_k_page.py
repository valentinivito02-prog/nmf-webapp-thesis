from dash import dcc, html
import dash_bootstrap_components as dbc


def layout() -> html.Div:
    return html.Div([
        dcc.Store(id="k-selection-results"),
        dcc.Store(id="dataset-store", storage_type="session"),
        dcc.Store(id="selected-k-store", storage_type="session"),
        dcc.Store(id="k-experiment-status", storage_type="session"),

        dcc.Download(id="download-k-results"),
        dcc.Download(id="download-k-config"),
        dcc.Download(id="download-k-graph"),
        dcc.Download(id="download-consensus-matrix"),

        dbc.Row(
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        html.H4(
                            "Step 2: Choose the Optimal Number of Clusters (k)",
                            className="card-title mb-0"
                        ),
                        className="card-header-gradient"
                    ),

                    dbc.CardBody([
                        html.P(
                            "Configure the evaluation settings to explore different values of k and compare the results.",
                            className="text-muted mb-4"
                        ),

                        html.Div(id="dataset-info", className="mb-4"),

                        html.H5("Evaluation Methods", className="mb-3"),
                        dbc.Card(
                            dbc.CardBody([
                                dcc.Checklist(
                                    id="method-selection",
                                    options=[
                                        {"label": " Elbow Method", "value": "elbow"},
                                        {"label": " Silhouette Score", "value": "silhouette"},
                                        {"label": " Cophenetic Index", "value": "cophenetic"},
                                    ],
                                    value=["elbow"],
                                    inline=True,
                                    labelStyle={
                                        "marginRight": "35px",
                                        "marginBottom": "10px"
                                    }
                                ),
                            ]),
                            className="mb-4"
                        ),

                        html.H5("Clustering Algorithms", className="mb-3"),
                        dbc.Card(
                            dbc.CardBody([
                                dcc.Checklist(
                                    id="clustering-selection",
                                    options=[
                                        {"label": " Argmax", "value": "argmax"},
                                        {"label": " K-Means", "value": "kmeans"},
                                        {"label": " Fuzzy C-Means", "value": "fcm"},
                                    ],
                                    value=["kmeans"],
                                    inline=True,
                                    labelStyle={
                                        "marginRight": "35px",
                                        "marginBottom": "10px"
                                    }
                                ),
                            ]),
                            className="mb-4"
                        ),

                        dbc.Row([
                            dbc.Col([
                                html.H5("Initialization Methods", className="mb-3"),
                                dcc.Dropdown(
                                    id="init-selection",
                                    options=[
                                        {"label": "Random", "value": "random"},
                                        {"label": "NNDSVD", "value": "nndsvd"},
                                        {"label": "Custom 1", "value": "custom1"},
                                        {"label": "Custom 2", "value": "custom2"},
                                    ],
                                    multi=True,
                                    placeholder="Select initialization methods"
                                ),
                            ], md=6),

                            dbc.Col([
                                html.H5("NMF Algorithm", className="mb-3"),
                                dcc.Dropdown(
                                    id="nmf-selection",
                                    options=[
                                        {"label": "Standard NMF", "value": "nmf_standard"},
                                    ],
                                    value="nmf_standard"
                                ),
                            ], md=6),
                        ], className="mb-4"),

                        html.H5("k Range", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Minimum k"),
                                dcc.Input(
                                    id="k-min",
                                    type="number",
                                    placeholder="Min (>=2)",
                                    min=2,
                                    value=2,
                                    style={"width": "100%"}
                                ),
                            ], md=6),

                            dbc.Col([
                                dbc.Label("Maximum k"),
                                dcc.Input(
                                    id="k-max",
                                    type="number",
                                    placeholder="Max (<= num_features)",
                                    style={"width": "100%"}
                                ),
                            ], md=6),
                        ], className="mb-4"),

                        html.Div(
                            dbc.Button(
                                "Run Experiment",
                                id="run-k-selection",
                                color="primary",
                                className="px-4"
                            ),
                            className="d-flex justify-content-center mb-2"
                        ),

                        html.Div(id="k-run-warning", className="mt-2"),

                        html.H5("Results", className="mb-3 mt-4"),
                        dbc.Card(
                            dbc.CardBody([
                                html.Div(
                                    dcc.Tabs(
                                        id="results-tabs",
                                        value="tab-graph",
                                        colors={
                                            "border": "#dee2e6",
                                            "primary": "#52b2cf",
                                            "background": "#f8f9fa"
                                        },
                                        children=[
                                            dcc.Tab(
                                                label="Graph",
                                                value="tab-graph",
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
                                                    html.Div([
                                                        html.H6(
                                                            "Select Metric to Visualize",
                                                            className="mb-2 mt-3"
                                                        ),

                                                        dcc.Dropdown(
                                                            id="result-method-selector",
                                                            options=[
                                                                {"label": "Elbow Method", "value": "elbow"},
                                                                {"label": "Silhouette Score", "value": "silhouette"},
                                                                {"label": "Cophenetic Index", "value": "cophenetic"},
                                                            ],
                                                            value="elbow",
                                                            clearable=False,
                                                            className="mb-3"
                                                        ),

                                                        dcc.Graph(
                                                            id="k-selection-graph",
                                                            figure={},
                                                            config={
                                                                "displaylogo": False,
                                                                "toImageButtonOptions": {
                                                                    "format": "png",
                                                                    "filename": "k_selection_plot",
                                                                    "height": 900,
                                                                    "width": 1400,
                                                                    "scale": 2
                                                                },
                                                                "modeBarButtonsToRemove": [
                                                                    "lasso2d",
                                                                    "select2d",
                                                                    "autoScale2d"
                                                                ]
                                                            },
                                                            style={
                                                                "borderRadius": "12px"
                                                            }
                                                        ),

                                                        html.Small(
                                                            "Use the camera icon in the graph toolbar to download the figure as PNG.",
                                                            className="text-muted"
                                                        )
                                                    ])
                                                ]
                                            ),

                                            dcc.Tab(
                                                label="Metrics Table",
                                                value="tab-metrics",
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
                                                        id="k-selection-metrics",
                                                        children=dbc.Alert(
                                                            "Numerical metrics will be displayed here.",
                                                            color="light",
                                                            className="mt-3"
                                                        ),
                                                        className="mt-3"
                                                    )
                                                ]
                                            ),

                                            dcc.Tab(
                                                label="Consensus Matrices",
                                                value="tab-consensus",
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
                                                    html.Div([
                                                        html.H6(
                                                            "Select Consensus Matrix View",
                                                            className="mb-4 mt-3"
                                                        ),

                                                        dbc.Row([
                                                            dbc.Col([
                                                                dbc.Label("Clustering Algorithm"),
                                                                dcc.Dropdown(
                                                                    id="consensus-method-selector",
                                                                    options=[
                                                                        {"label": "Argmax", "value": "argmax"},
                                                                        {"label": "K-Means", "value": "kmeans"},
                                                                        {"label": "Fuzzy C-Means", "value": "fcm"},
                                                                    ],
                                                                    value="kmeans",
                                                                    clearable=False
                                                                )
                                                            ], md=3),

                                                            dbc.Col([
                                                                dbc.Label("Initialization"),
                                                                dcc.Dropdown(
                                                                    id="consensus-init-selector",
                                                                    options=[
                                                                        {"label": "Random", "value": "random"},
                                                                        {"label": "NNDSVD", "value": "nndsvd"},
                                                                        {"label": "Custom 1", "value": "custom1"},
                                                                        {"label": "Custom 2", "value": "custom2"},
                                                                    ],
                                                                    value="nndsvd",
                                                                    clearable=False
                                                                )
                                                            ], md=3),

                                                            dbc.Col(
                                                                html.Div([
                                                                    dbc.Label("FCM Consensus Mode"),
                                                                    dcc.Dropdown(
                                                                        id="consensus-fcm-mode-selector",
                                                                        options=[
                                                                            {"label": "Hard", "value": "hard"},
                                                                            {"label": "Soft Dot", "value": "soft_dot"},
                                                                            {"label": "Soft Cosine", "value": "soft_cosine"},
                                                                        ],
                                                                        value="hard",
                                                                        clearable=False
                                                                    ),
                                                                ], id="consensus-fcm-mode-container"),
                                                                md=4
                                                            ),

                                                            dbc.Col([
                                                                dbc.Label("k"),
                                                                dcc.Input(
                                                                    id="consensus-k-selector",
                                                                    type="number",
                                                                    min=2,
                                                                    value=2,
                                                                    style={"width": "100%"}
                                                                )
                                                            ], md=2),
                                                        ], className="mb-4"),

                                                        dcc.Graph(
                                                            id="consensus-matrix-plot",
                                                            figure={},
                                                            config={
                                                                "displaylogo": False,
                                                                "toImageButtonOptions": {
                                                                    "format": "png",
                                                                    "filename": "consensus_matrix",
                                                                    "height": 1200,
                                                                    "width": 1200,
                                                                    "scale": 2
                                                                },
                                                                "modeBarButtonsToRemove": [
                                                                    "lasso2d",
                                                                    "select2d",
                                                                    "autoScale2d"
                                                                ]
                                                            },
                                                            style={
                                                                "borderRadius": "12px"
                                                            }
                                                        ),

                                                        html.Small(
                                                            "Use the camera icon in the graph toolbar to download the consensus matrix as PNG.",
                                                            className="text-muted"
                                                        ),

                                                        html.Div(
                                                            dbc.Button(
                                                                [
                                                                    html.I(className="fas fa-download me-2"),
                                                                    "Download Consensus Matrix"
                                                                ],
                                                                id="download-consensus-btn",
                                                                color="primary",
                                                                className="mt-3"
                                                            ),
                                                            className="d-flex justify-content-end"
                                                        ),

                                                        html.Div(
                                                            id="consensus-note",
                                                            children=dbc.Alert(
                                                                "Consensus matrices will be displayed here after running the experiment.",
                                                                color="light",
                                                                className="mt-3"
                                                            )
                                                        )
                                                    ])
                                                ]
                                            ),

                                            dcc.Tab(
                                                label="Configuration Summary",
                                                value="tab-summary",
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
                                                        id="k-selection-summary",
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
                                        [html.I(className="fas fa-download me-2"), "Download Metrics"],
                                        id="download-k-metrics-btn",
                                        color="success",
                                        className="me-2"
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-download me-2"), "Download Configuration"],
                                        id="download-k-config-btn",
                                        color="secondary",
                                        className="me-2"
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-download me-2"), "Download Graph"],
                                        id="download-k-graph-btn",
                                        color="primary"
                                    ),
                                ], className="d-flex justify-content-end mt-3")
                            ]),
                            className="mb-4"
                        ),

                        html.H5("Please Select the Best k", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dcc.Input(
                                    id="selected-k",
                                    type="number",
                                    min=2,
                                    placeholder="Insert selected k",
                                    style={"width": "100%"}
                                )
                            ], md=6),

                            dbc.Col([
                                dbc.Button(
                                    "Confirm k",
                                    id="confirm-k",
                                    color="success",
                                    className="w-100"
                                )
                            ], md=6),
                        ]),

                        html.Div(id="confirm-k-message", className="mt-3")
                    ]),

                    dbc.CardFooter(
                        html.Div([
                            html.Div([
                                dbc.Button(
                                    [html.I(className="fas fa-arrow-left me-2"), "Back"],
                                    href="/upload",
                                    color="light",
                                    className="nav-btn me-2"
                                ),
                                dbc.Button(
                                    [html.I(className="fas fa-arrow-right me-2"), "Next"],
                                    id="choose-k-next-btn",
                                    color="primary",
                                    className="nav-btn"
                                )
                            ]),
                            html.Div(
                                id="choose-k-warning",
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