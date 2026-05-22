from dash import html
import dash_bootstrap_components as dbc


def section_header(title, subtitle=None):
    return html.Div([
        html.H4(
            title,
            className="mb-2",
            style={
                "fontWeight": "700",
                "color": "#2c3e50"
            }
        ),
        html.P(
            subtitle,
            className="text-muted mb-4",
            style={"fontSize": "15px"}
        ) if subtitle else None
    ])


def info_card(title, value):
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                title,
                className="text-muted mb-1",
                style={"fontSize": "14px"}
            ),
            html.H4(
                str(value),
                className="mb-0",
                style={
                    "fontWeight": "700",
                    "color": "#2c3e50"
                }
            )
        ]),
        className="text-center shadow-sm border-0",
        style={
            "borderRadius": "12px",
            "backgroundColor": "white"
        }
    )


def dataset_loaded_card(dataset_name, subtitle):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Strong("Dataset loaded: "),
                html.Span(dataset_name)
            ], className="mb-1"),
            html.Small(
                subtitle,
                className="text-muted"
            )
        ]),
        className="mt-3 mb-4 shadow-sm border-0",
        style={
            "backgroundColor": "#f8fbfd",
            "borderLeft": "5px solid #52b2cf",
            "borderRadius": "10px"
        }
    )


def light_info_card(title, text):
    return dbc.Card(
        dbc.CardBody([
            html.H6(
                title,
                className="mb-2",
                style={
                    "fontWeight": "700",
                    "color": "#2c3e50"
                }
            ),
            html.P(
                text,
                className="mb-0",
                style={
                    "fontSize": "14px",
                    "lineHeight": "1.6",
                    "color": "#2c3e50"
                }
            )
        ]),
        className="mt-3 shadow-sm border-0",
        style={
            "backgroundColor": "#f8fbfd",
            "borderLeft": "5px solid #52b2cf",
            "borderRadius": "10px"
        }
    )