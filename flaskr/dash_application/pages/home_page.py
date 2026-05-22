from dash import html
import dash_bootstrap_components as dbc


def workflow_card(number, title, text):
    return dbc.Card(
        dbc.CardBody([

            html.Div(
                number,
                className="mb-2",
                style={
                    "width": "30px",
                    "height": "30px",
                    "borderRadius": "50%",
                    "backgroundColor": "#52b2cf",
                    "color": "white",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "fontWeight": "700",
                    "fontSize": "14px"
                }
            ),

            html.H6(
                title,
                className="mb-1",
                style={
                    "fontWeight": "700",
                    "color": "#2c3e50"
                }
            ),

            html.Small(
                text,
                className="text-muted"
            )

        ], style={"padding": "18px"}),

        className="shadow-sm border-0 h-100",

        style={
            "borderRadius": "12px",
            "backgroundColor": "white"
        }
    )


def layout() -> html.Div:

    return html.Div(

        className="home-page-container d-flex flex-column justify-content-center align-items-center",

        children=[

            dbc.Card(

                dbc.CardBody([

                    html.Div(
                        "NMF Web App",
                        className="mb-2 text-center",
                        style={
                            "color": "#52b2cf",
                            "fontWeight": "700",
                            "fontSize": "17px"
                        }
                    ),

                    html.H1(
                        "NMF Clustering & Fuzzy Analysis",
                        className="mb-3 text-center",
                        style={
                            "fontWeight": "700",
                            "color": "#2c3e50",
                            "fontSize": "42px"
                        }
                    ),

                    html.P([
                        "Upload a dataset, select the optimal number of clusters,",
                        html.Br(),
                        "run the final NMF model and generate fuzzy explanations."
                    ],
                        className="text-muted text-center mb-4",
                        style={
                            "fontSize": "15px",
                            "lineHeight": "1.5"
                        }
                    ),

                    dbc.Row([

                        dbc.Col(
                            workflow_card(
                                "1",
                                "Upload Dataset",
                                "Load a CSV dataset and inspect its structure."
                            ),
                            md=3
                        ),

                        dbc.Col(
                            workflow_card(
                                "2",
                                "Choose k",
                                "Compare metrics and select the optimal k."
                            ),
                            md=3
                        ),

                        dbc.Col(
                            workflow_card(
                                "3",
                                "Run NMF",
                                "Execute the final NMF configuration."
                            ),
                            md=3
                        ),

                        dbc.Col(
                            workflow_card(
                                "4",
                                "Fuzzy Explanation",
                                "Generate interpretable fuzzy descriptions."
                            ),
                            md=3
                        ),

                    ], className="g-3 mb-4"),

                    html.Div(

                        dbc.Button(
                            "Start Workflow",
                            href="/upload",
                            color="primary",
                            size="lg",
                            className="px-4",

                            style={
                                "borderRadius": "10px",
                                "fontWeight": "600",
                                "padding": "9px 26px"
                            }
                        ),

                        className="d-flex justify-content-center"
                    )

                ],

                style={
                    "padding": "24px"
                }),

                className="shadow-sm border-0",

                style={
                    "maxWidth": "950px",
                    "width": "100%",
                    "borderRadius": "16px",
                    "backgroundColor": "#f8fbfd",
                    "borderLeft": "5px solid #52b2cf"
                }
            )
        ]
    )