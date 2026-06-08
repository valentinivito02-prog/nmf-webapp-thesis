from dash import dcc, html
import dash_bootstrap_components as dbc


def layout() -> html.Div:
    return html.Div([
        dcc.Store(id='dataset-store', storage_type='session'),
        dcc.Store(id='selected-k-store', storage_type='session'),
        dcc.Store(id='nmf-results-store', storage_type='session'),

        dcc.Download(id="download-w-matrix"),
        dcc.Download(id="download-h-matrix"),
        dcc.Download(id="download-clusters"),
        dcc.Download(id="download-centroids-representatives"),
        dcc.Download(id="download-final-nmf-configuration"),

        dbc.Row(
            dbc.Col(
                dbc.Card([

                    dbc.CardHeader(
                        html.H4(
                            "Step 3: Run Final NMF",
                            className="card-title mb-0"
                        ),
                        className="card-header-gradient"
                    ),

                    dbc.CardBody([

                        html.P(
                            "Run the final NMF analysis using the selected value of k and the desired configuration.",
                            className="text-muted mb-4"
                        ),

                        dbc.Alert(
                            "Selected k will be displayed here.",
                            id='selected-k-display',
                            color="light",
                            className="mb-4"
                        ),

                        dbc.Row([

                            dbc.Col([
                                html.H5("Clustering Algorithm", className="mb-3"),
                                dcc.Dropdown(
                                    id='final-clustering',
                                    options=[
                                        {'label': 'Argmax', 'value': 'argmax'},
                                        {'label': 'K-Means', 'value': 'kmeans'},
                                        {'label': 'Fuzzy C-Means', 'value': 'fcm'},
                                    ],
                                    value='kmeans'
                                ),
                            ], md=4),

                            dbc.Col([
                                html.H5("Initialization Method", className="mb-3"),
                                dcc.Dropdown(
                                    id='final-init',
                                    options=[
                                        {'label': 'Random', 'value': 'random'},
                                        {'label': 'NNDSVD', 'value': 'nndsvd'},
                                    ],
                                    value='random'
                                ),
                            ], md=4),

                            dbc.Col([
                                html.H5("NMF Algorithm", className="mb-3"),
                                dcc.Dropdown(
                                    id='final-nmf',
                                    options=[
                                        {'label': 'Standard NMF', 'value': 'nmf_standard'},
                                    ],
                                    value='nmf_standard'
                                ),
                            ], md=4),

                        ], className="mb-4"),

                        html.Div(
                            dbc.Button(
                                "Run Final NMF",
                                id='run-final-nmf',
                                color='primary',
                                className="px-4"
                            ),
                            className="d-flex justify-content-center mb-2"
                        ),

                        html.Div(
                            id="run-nmf-warning",
                            className="mt-2"
                        ),

                        html.H5("Results", className="mb-3 mt-4"),

                        dbc.Card(
                            dbc.CardBody([

                                dcc.Tabs(
                                    id="final-results-tabs",
                                    value="tab-w",
                                    colors={
                                        "border": "#dee2e6",
                                        "primary": "#52b2cf",
                                        "background": "#f8f9fa"
                                    },
                                    children=[

                                        dcc.Tab(
                                            label="Matrix W",
                                            value="tab-w",
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

                                                    dcc.Graph(
                                                        id='matrix-w-plot',
                                                        responsive=False,
                                                        config={
                                                            "displaylogo": False,
                                                            "toImageButtonOptions": {
                                                                "format": "png",
                                                                "filename": "matrix_W_heatmap",
                                                                "height": 1200,
                                                                "width": 1600,
                                                                "scale": 3
                                                            },
                                                            "modeBarButtonsToRemove": [
                                                                "lasso2d",
                                                                "select2d",
                                                                "autoScale2d"
                                                            ]
                                                        },
                                                        style={
                                                            "borderRadius": "12px",
                                                            "height": "700px"
                                                        }
                                                    ),

                                                    html.Div(
                                                        id="nmf-w-heatmap-note",
                                                        className="mt-3"
                                                    ),

                                                    html.Div(
                                                        dbc.Button(
                                                            [
                                                                html.I(className="fas fa-download me-2"),
                                                                "Download W"
                                                            ],
                                                            id="download-w-btn",
                                                            color="primary",
                                                            style={
                                                                "borderRadius": "10px",
                                                                "fontWeight": "600",
                                                                "padding": "10px 18px"
                                                            }
                                                        ),
                                                        className="d-flex justify-content-end mt-3"
                                                    )

                                                ], className="mt-3")
                                            ]
                                        ),

                                        dcc.Tab(
                                            label="Matrix H",
                                            value="tab-h",
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

                                                    dcc.Graph(
                                                        id='matrix-h-plot',
                                                        responsive=False,
                                                        config={
                                                            "displaylogo": False,
                                                            "toImageButtonOptions": {
                                                                "format": "png",
                                                                "filename": "matrix_H_heatmap",
                                                                "height": 1200,
                                                                "width": 1600,
                                                                "scale": 3
                                                            },
                                                            "modeBarButtonsToRemove": [
                                                                "lasso2d",
                                                                "select2d",
                                                                "autoScale2d"
                                                            ]
                                                        },
                                                        style={
                                                            "borderRadius": "12px",
                                                            "height": "700px"
                                                        }
                                                    ),

                                                    html.Div(
                                                        id="nmf-h-heatmap-note",
                                                        className="mt-3"
                                                    ),

                                                    html.Div(
                                                        dbc.Button(
                                                            [
                                                                html.I(className="fas fa-download me-2"),
                                                                "Download H"
                                                            ],
                                                            id="download-h-btn",
                                                            color="primary",
                                                            style={
                                                                "borderRadius": "10px",
                                                                "fontWeight": "600",
                                                                "padding": "10px 18px"
                                                            }
                                                        ),
                                                        className="d-flex justify-content-end mt-3"
                                                    )

                                                ], className="mt-3")
                                            ]
                                        ),

                                        dcc.Tab(
                                            label="Cluster Assignments",
                                            value="tab-clusters",
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

                                                    html.Div(
                                                        id='cluster-output',
                                                        children=dbc.Alert(
                                                            "Cluster assignments will be displayed here after execution.",
                                                            color="light"
                                                        ),
                                                        className="mt-3"
                                                    ),

                                                    html.Div(
                                                        dbc.Button(
                                                            [
                                                                html.I(className="fas fa-download me-2"),
                                                                "Download Clusters"
                                                            ],
                                                            id="download-clusters-btn",
                                                            color="primary",
                                                            style={
                                                                "borderRadius": "10px",
                                                                "fontWeight": "600",
                                                                "padding": "10px 18px"
                                                            }
                                                        ),
                                                        className="d-flex justify-content-end mt-3"
                                                    )

                                                ])
                                            ]
                                        ),

                                        dcc.Tab(
                                            label="Centroids & Representatives",
                                            value="tab-centroids",
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

                                                    html.Div(
                                                        id="centroids-representatives-output",
                                                        children=dbc.Alert(
                                                            "Centroids and representative vectors will be displayed here after execution.",
                                                            color="light"
                                                        ),
                                                        className="mt-3"
                                                    ),

                                                    html.Div(
                                                        dbc.Button(
                                                            [
                                                                html.I(className="fas fa-download me-2"),
                                                                "Download Centroids & Representatives"
                                                            ],
                                                            id="download-centroids-representatives-btn",
                                                            color="primary",
                                                            style={
                                                                "borderRadius": "10px",
                                                                "fontWeight": "600",
                                                                "padding": "10px 18px"
                                                            }
                                                        ),
                                                        className="d-flex justify-content-end mt-3"
                                                    )

                                                ])
                                            ]
                                        ),

                                        dcc.Tab(
                                        label="Configuration Summary",
                                        value="tab-final-summary",
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

                                                html.Div(
                                                    id="final-nmf-summary",
                                                    className="mt-3"
                                                ),

                                                html.Div(
                                                    dbc.Button(
                                                        [
                                                            html.I(className="fas fa-download me-2"),
                                                            "Download Final NMF Configuration"
                                                        ],
                                                        id="download-final-nmf-configuration-btn",
                                                        color="primary",
                                                        style={
                                                            "borderRadius": "10px",
                                                            "fontWeight": "600",
                                                            "padding": "10px 18px"
                                                        }
                                                    ),
                                                    className="d-flex justify-content-end mt-3"
                                                )

                                            ])
                                        ]
                                    ),

                                    ]
                                )

                            ]),
                            className="mb-4"
                        ),

                    ]),

                    dbc.CardFooter(
                        html.Div([
                            html.Div([
                                dbc.Button(
                                    [html.I(className="fas fa-arrow-left me-2"), "Back"],
                                    href="/choose-k",
                                    color="light",
                                    className="nav-btn me-2"
                                ),
                                dbc.Button(
                                    [html.I(className="fas fa-arrow-right me-2"), "Next"],
                                    id="run-next-btn",
                                    color="primary",
                                    className="nav-btn"
                                )
                            ]),
                            html.Div(
                                id="run-warning",
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