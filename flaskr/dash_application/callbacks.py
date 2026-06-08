import dash
import base64
from dash import dcc, html, Input, dash_table, Output, State, ctx, ALL, MATCH, callback_context
import plotly.graph_objects as go
import requests
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import json
from flaskr import file_handler
from flaskr.file_handler import save_terms, save_explanations
from datetime import datetime
import re
import io
import os
import uuid
from dash.exceptions import PreventUpdate
from .services.giannico_service import run_k_experiments, run_final_nmf, run_fuzzy_from_nmf_results
from io import BytesIO
from .ui_components import (
dataset_loaded_card,
info_card,
light_info_card
)
def register_callbacks(dash_app):
    """Registra tutti i callback necessari all'app Dash per la gestione del workflow
    NMF."""
    @dash_app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("url", "pathname"),
        State("session-store", "data"),
        prevent_initial_call='initial_duplicate'
    )
    def init_sid(_pathname, session_data):
        if not isinstance(session_data, dict):
            session_data = {}
        if session_data.get("sid"):
            return dash.no_update
        session_data["sid"] = str(uuid.uuid4())
        return session_data
    
    def get_dataset_labels(dataset_data):
        if not dataset_data or "data" not in dataset_data:
            return [], []
        df = pd.DataFrame(dataset_data["data"])
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        non_numeric_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
        feature_labels = [
            make_readable_label(col)
            for col in numeric_columns
        ]
        if non_numeric_columns:
            sample_labels = df[non_numeric_columns[0]].astype(str).tolist()
        else:
            sample_labels = [f"Sample {i + 1}" for i in range(len(df))]
        return feature_labels, sample_labels    
    
    def make_readable_label(label):
        label = str(label)
        label = label.replace("_", " ")
        label = label.replace("-", " ")
        label = label.replace("/", " / ")
        label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
        label = re.sub(r"\s+", " ", label).strip()
        return label.title()
    
    def make_config_card(title, value):
        return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    title,
                    className="text-muted mb-1",
                    style={"fontSize": "13px"}
                ),
                html.H5(
                    str(value),
                    className="mb-0",
                    style={
                        "fontWeight": "700",
                        "color": "#2c3e50",
                        "fontSize": "17px"
                    }
                )
            ],
            style={"padding": "18px"}
        ),
        className="text-center shadow-sm border-0",
        style={
            "borderRadius": "12px",
            "backgroundColor": "white"
        }
    )
    
    def make_light_section_card(title, subtitle, children):
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5(
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
                        style={"fontSize": "14px"}
                    ),
                    children
                ],
                style={"padding": "22px"}
            ),
            className="shadow-sm border-0",
            style={
                "backgroundColor": "#f8fbfd",
                "borderLeft": "5px solid #52b2cf",
                "borderRadius": "10px"
            }
        )
    
    def make_interpretation_card(title, text):
        return dbc.Card(
            dbc.CardBody(
                [
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
                            "fontSize": "13px",
                            "lineHeight": "1.6",
                            "color": "#2c3e50"
                        }
                    )
                ],
                style={"padding": "16px"}
            ),
            className="mt-3 shadow-sm border-0",
            style={
                "backgroundColor": "#f8fbfd",
                "borderLeft": "5px solid #52b2cf",
                "borderRadius": "10px"
            }
        )
    
    def make_light_section_card(title, subtitle, children):
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H5(
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
                        style={"fontSize": "14px"}
                    ),
                    children
                ],
                style={"padding": "22px"}
            ),
            className="shadow-sm border-0",
            style={
                "backgroundColor": "#f8fbfd",
                "borderLeft": "5px solid #52b2cf",
                "borderRadius": "10px"
            }
        )
    
    def make_interpretation_card(title, text):
        return dbc.Card(
            dbc.CardBody(
                [
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
                            "fontSize": "13px",
                            "lineHeight": "1.6",
                            "color": "#2c3e50"
                        }
                    )
                ],
                style={"padding": "16px"}
            ),
            className="mt-3 shadow-sm border-0",
            style={
                "backgroundColor": "#f8fbfd",
                "borderLeft": "5px solid #52b2cf",
                "borderRadius": "10px"
            }
        )
    
    def make_dataset_title(filename):
        name = os.path.splitext(filename)[0]
        name = name.replace("_", " ")
        name = name.replace("-", " ")
        name = re.sub(r"\bfor\b.*", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\s+", " ", name).strip()
        return name.title()
    
    def send_excel_file(df, filename, sheet_name="Sheet1", index=False):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=index
            )
            worksheet = writer.sheets[sheet_name]
            for column_cells in worksheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter
                for cell in column_cells:
                    cell_value = str(cell.value) if cell.value is not None else ""
                    max_length = max(max_length, len(cell_value))
                worksheet.column_dimensions[column_letter].width = max_length + 3
        output.seek(0)
        return dcc.send_bytes(
            output.getvalue(),
            filename
        )
    
    @dash_app.callback(
        Output("upload-output", "children"),
        Output("dataset-store", "data"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        prevent_initial_call=True
    )
    def handle_dataset_upload(contents, filename):
        if contents is None:
            return (
                dbc.Alert(
                    "No dataset uploaded yet.",
                    color="light",
                    className="mt-3"
                ),
                dash.no_update
            )
        try:
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            if not filename or not filename.lower().endswith(".csv"):
                return (
                    dbc.Alert(
                        "Please upload a valid CSV file.",
                        color="danger",
                        className="mt-3"
                    ),
                    dash.no_update
                )
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            numeric_columns = df.select_dtypes(include="number").columns.tolist()
            non_numeric_columns = df.select_dtypes(exclude="number").columns.tolist()
            if non_numeric_columns:
                sample_names = df[non_numeric_columns[0]].astype(str).tolist()
            else:
                sample_names = [f"Sample {i + 1}" for i in range(len(df))]
            dataset_data = {
                "filename": filename,
                "display_name": make_dataset_title(filename),
                "columns": df.columns.tolist(),
                "shape": [df.shape[0], df.shape[1]],
                "sample_names": sample_names,
                "data": df.to_dict("records")
            }
            dataset_loaded_info = dbc.Card(
                dbc.CardBody([
                    html.Div([
                        html.Strong("Dataset loaded: "),
                        html.Span(make_dataset_title(filename))
                    ], className="mb-1"),
                    html.Small(
                        "Use the preview below to inspect the uploaded data before continuing.",
                        className="text-muted"
                    )
                ], style={"padding": "4px 2px"}),
                className="mt-3 mb-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#f8fbfd",
                    "borderLeft": "5px solid #52b2cf",
                    "borderRadius": "10px"
                }
            )
            dataset_cards = dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H6(
                                "Rows",
                                className="text-muted mb-1",
                                style={"fontSize": "13px"}
                            ),
                            html.H4(
                                str(df.shape[0]),
                                className="mb-0",
                                style={"fontSize": "22px", "fontWeight": "700"}
                            )
                        ], style={"padding": "12px"}),
                        className="text-center shadow-sm border-0"
                    ),
                    md=4
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H6(
                                "Columns",
                                className="text-muted mb-1",
                                style={"fontSize": "13px"}
                            ),
                            html.H4(
                                str(df.shape[1]),
                                className="mb-0",
                                style={"fontSize": "22px", "fontWeight": "700"}
                            )
                        ], style={"padding": "12px"}),
                        className="text-center shadow-sm border-0"
                    ),
                    md=4
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H6(
                                "Numeric Features",
                                className="text-muted mb-1",
                                style={"fontSize": "13px"}
                            ),
                            html.H4(
                                str(len(numeric_columns)),
                                className="mb-0",
                                style={"fontSize": "22px", "fontWeight": "700"}
                            )
                        ], style={"padding": "12px"}),
                        className="text-center shadow-sm border-0"
                    ),
                    md=4
                ),
            ], className="mb-3")
            preview_controls = dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label(
                                "Search in dataset preview",
                                className="mb-1",
                                style={
                                    "fontSize": "14px",
                                    "fontWeight": "600"
                                }
                            ),
                            dcc.Input(
                                id="dataset-preview-search",
                                type="text",
                                placeholder="Search values...",
                                debounce=True,
                                style={
                                    "width": "75%",
                                    "padding": "7px 10px",
                                    "borderRadius": "8px",
                                    "border": "1px solid #ced4da",
                                    "fontSize": "13px"
                                }
                            )
                        ], md=8),
                        dbc.Col(
                            html.Small(
                                "Search across all columns.",
                                className=(
                                    "text-muted d-flex align-items-center "
                                    "justify-content-center h-100"
                                ),
                                style={
                                    "fontSize": "12px",
                                    "textAlign": "center",
                                    "marginTop": "27px"
                                }
                            ),
                            md=4
                        )
                    ], align="center")
                ], style={"padding": "12px"}),
                className="mb-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#ffffff",
                    "borderRadius": "10px"
                }
            )
            preview_table = dash_table.DataTable(
                id="dataset-preview-table",
                data=df.head(10).to_dict("records"),
                columns=[
                    {
                        "name": make_readable_label(col),
                        "id": col
                    }
                    for col in df.columns
                ],
                page_size=10,
                page_action="native",
                sort_action="native",
                filter_action="none",
                style_table={
                    "overflowX": "auto",
                    "overflowY": "hidden",
                    "borderRadius": "12px",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                    "border": "1px solid #e9ecef"
                },
                style_header={
                    "backgroundColor": "#52b2cf",
                    "color": "white",
                    "fontWeight": "700",
                    "fontSize": "13px",
                    "textAlign": "center",
                    "padding": "6px",
                    "border": "1px solid #52b2cf",
                    "height": "40px"
                },
                style_cell={
                    "textAlign": "center",
                    "padding": "5px",
                    "fontSize": "12px",
                    "fontFamily": "Poppins, Arial, sans-serif",
                    "minWidth": "100px",
                    "maxWidth": "170px",
                    "whiteSpace": "normal",
                    "height": "auto"
                },
                style_data={
                    "backgroundColor": "white",
                    "color": "#2c3e50",
                    "border": "1px solid #f1f1f1"
                },
                style_data_conditional=[
                    {
                        "if": {
                            "row_index": "odd"
                        },
                        "backgroundColor": "#f8fbfd"
                    },
                    {
                        "if": {
                            "state": "active"
                        },
                        "backgroundColor": "#d9f0f7",
                        "border": "1px solid #52b2cf"
                    },
                    {
                        "if": {
                            "state": "selected"
                        },
                        "backgroundColor": "#d9f0f7",
                        "border": "1px solid #52b2cf"
                    }
                ],
            )
            preview = html.Div([
                dataset_loaded_info,
                dataset_cards,
                html.H5(
                    "Dataset Preview",
                    className="mb-3",
                    style={
                        "fontWeight": "700",
                        "color": "#2c3e50"
                    }
                ),
                preview_controls,
                preview_table
            ])
            return preview, dataset_data
        except Exception as e:
            return (
                dbc.Alert(
                    f"Error while reading the dataset: {str(e)}",
                    color="danger",
                    className="mt-3"
                ),
                dash.no_update
            )
    
    @dash_app.callback(
        Output("dataset-preview-table", "data"),
        Input("dataset-preview-search", "value"),
        State("dataset-store", "data"),
        prevent_initial_call=True
    )
    def filter_dataset_preview(search_value, dataset_data):
        if not dataset_data or "data" not in dataset_data:
            raise PreventUpdate
        df = pd.DataFrame(dataset_data["data"])
        if not search_value:
            return df.head(10).to_dict("records")
        search_value = str(search_value).lower().strip()
        filtered_df = df[
            df.astype(str)
            .apply(
                lambda row: row.str.lower().str.contains(search_value, na=False).any()
,
                axis=1
            )
        ]
        return filtered_df.head(50).to_dict("records")
    
    @dash_app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("upload-warning", "children"),
        Input("upload-next-btn", "n_clicks"),
        Input("dataset-store", "data"),
        prevent_initial_call=True
    )
    def handle_upload_next_click(n_clicks, dataset_data):
        triggered = ctx.triggered_id
        if triggered == "dataset-store":
            if dataset_data:
                return dash.no_update, ""
            return dash.no_update, dash.no_update
        if triggered == "upload-next-btn":
            if not dataset_data:
                return dash.no_update, dbc.Alert(
                "Please upload a dataset before continuing.",
                color="warning",
                className="mt-2 p-2 mb-0"
            )
            return "/choose-k", ""
        return dash.no_update, dash.no_update
    
    @dash_app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("choose-k-warning", "children"),
        Input("choose-k-next-btn", "n_clicks"),
        Input("selected-k-store", "data"),
        prevent_initial_call=True
    )
    def handle_choose_k_next_click(n_clicks, selected_k_data):
        triggered = ctx.triggered_id
        if triggered == "selected-k-store":
            if selected_k_data and "selected_k" in selected_k_data:
                return dash.no_update, ""
            return dash.no_update, dash.no_update
        if triggered == "choose-k-next-btn":
            if not selected_k_data or "selected_k" not in selected_k_data:
                return dash.no_update, dbc.Alert(
                "Please confirm the selected k before continuing.",
                color="warning",
                className="mt-2 p-2 mb-0"
            )
            return "/run", ""
        return dash.no_update, dash.no_update
    
    @dash_app.callback(
        Output("dataset-info", "children"),
        Input("dataset-store", "data")
    )
    def show_dataset_info(data):
        if not data:
            return dbc.Alert(
                "No dataset loaded. Please go back and upload a dataset.",
                color="danger"
            )
        rows, cols = data["shape"]
        dataset_name = (
            data.get("display_name")
            or data.get("filename")
            or "Uploaded Dataset"
        )
        numeric_features = len([
            col for col in data.get("columns", [])
            if str(col).lower() not in ["id", "label", "class"]
        ])
        return make_light_section_card(
            title="Dataset Overview",
            subtitle="Summary of the uploaded dataset used for k-selection analysis.",
            children=dbc.Row(
                [
                    dbc.Col(
                        make_config_card("Dataset", dataset_name),
                        md=6,
                        className="mb-3"
                    ),
                    dbc.Col(
                        make_config_card("Rows", rows),
                        md=6,
                        className="mb-3"
                    ),
                    dbc.Col(
                        make_config_card("Columns", cols),
                        md=6
                    ),
                    dbc.Col(
                        make_config_card("Numeric Features", numeric_features),
                        md=6
                    ),
                ],
                className="g-3"
            )
        )
    
    @dash_app.callback(
        Output("k-max", "max"),
        Input("dataset-store", "data")
    )
    def set_k_max(data):
        if not data:
            return 10
        return data["shape"][1]
    
    def filter_k_metrics_columns(results_df, methods, clustering):
        columns_to_show = ["init", "k"]
        if "elbow" in methods:
            columns_to_show.extend([
                "reconstruction_error_mean",
                "reconstruction_error_std"
            ])
        if "silhouette" in methods:
            if "argmax" in clustering:
                columns_to_show.extend([
                    "silhouette_argmax_mean",
                    "silhouette_argmax_std"
                ])
            if "kmeans" in clustering:
                columns_to_show.extend([
                    "silhouette_kmeans_mean",
                    "silhouette_kmeans_std"
                ])
            if "fcm" in clustering:
                columns_to_show.extend([
                    "silhouette_fcm_mean",
                    "silhouette_fcm_std"
                ])
        if "cophenetic" in methods:
            if "argmax" in clustering:
                columns_to_show.append("coph_argmax")
            if "kmeans" in clustering:
                columns_to_show.append("coph_kmeans")
            if "fcm" in clustering:
                columns_to_show.extend([
                    "coph_fcm_hard",
                    "coph_fcm_soft_dot",
                    "coph_fcm_soft_cosine"
                ])
        existing_columns = [
            col for col in columns_to_show
            if col in results_df.columns
        ]
        return results_df[existing_columns]
    
    @dash_app.callback(
        Output("k-selection-summary", "children"),
        Output("k-selection-metrics", "children"),
        Output("k-experiment-status", "data"),
        Input("run-k-selection", "n_clicks"),
        State("method-selection", "value"),
        State("clustering-selection", "value"),
        State("init-selection", "value"),
        State("nmf-selection", "value"),
        State("k-min", "value"),
        State("k-max", "value"),
        State("dataset-store", "data"),
        prevent_initial_call=True
    )
    def run_k_selection_summary(
        n_clicks,
        methods,
        clustering,
        init_methods,
        nmf_alg,
        k_min,
        k_max,
        dataset_data
    ):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update
        
        def warning_card(title, message):
            return dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            title,
                            className="mb-1",
                            style={
                                "fontWeight": "700",
                                "color": "#856404"
                            }
                        ),
                        html.Small(
                            message,
                            className="text-muted"
                        )
                    ],
                    style={"padding": "16px"}
                ),
                className="mt-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#fff8e1",
                    "borderLeft": "5px solid #ffc107",
                    "borderRadius": "10px"
                }
            )
        
        def error_card(title, message):
            return dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            title,
                            className="mb-1",
                            style={
                                "fontWeight": "700",
                                "color": "#842029"
                            }
                        ),
                        html.Small(
                            message,
                            className="text-muted"
                        )
                    ],
                    style={"padding": "16px"}
                ),
                className="mt-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#f8d7da",
                    "borderLeft": "5px solid #dc3545",
                    "borderRadius": "10px"
                }
            )
        
        def config_card(title, value):
            return dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            title,
                            className="text-muted mb-1",
                            style={"fontSize": "13px"}
                        ),
                        html.H5(
                            str(value),
                            className="mb-0",
                            style={
                                "fontWeight": "700",
                                "color": "#2c3e50",
                                "fontSize": "17px"
                            }
                        )
                    ],
                    style={"padding": "18px"}
                ),
                className="text-center shadow-sm border-0",
                style={
                    "borderRadius": "12px",
                    "backgroundColor": "white"
                }
            )
        
        if not dataset_data:
            alert = error_card(
                "Dataset not loaded",
                "Please upload a dataset before running the k-selection experiment."
            )
            return alert, alert, dash.no_update
        
        if not methods:
            alert = warning_card(
                "Missing evaluation method",
                "Please select at least one evaluation method."
            )
            return alert, alert, dash.no_update
        
        if not clustering:
            alert = warning_card(
                "Missing clustering algorithm",
                "Please select at least one clustering algorithm."
            )
            return alert, alert, dash.no_update
        
        if not init_methods:
            alert = warning_card(
                "Missing initialization method",
                "Please select at least one initialization method."
            )
            return alert, alert, dash.no_update
        
        if k_min is None or k_max is None:
            alert = warning_card(
                "Missing k range",
                "Please define both minimum and maximum k."
            )
            return alert, alert, dash.no_update
        
        if k_min < 2:
            alert = warning_card(
                "Invalid k range",
                "Minimum k must be at least 2."
            )
            return alert, alert, dash.no_update
        
        max_features = dataset_data["shape"][1]
        if k_max > max_features:
            alert = warning_card(
                "Invalid k range",
                f"Maximum k cannot exceed the number of features ({max_features})."
            )
            return alert, alert, dash.no_update
        
        if k_min > k_max:
            alert = warning_card(
                "Invalid k range",
                "Minimum k cannot be greater than maximum k."
            )
            return alert, alert, dash.no_update
        
        method_labels = {
            "elbow": "Elbow Method",
            "silhouette": "Silhouette Score",
            "cophenetic": "Cophenetic Index"
        }
        clustering_labels = {
            "argmax": "Argmax",
            "kmeans": "K-Means",
            "fcm": "Fuzzy C-Means"
        }
        init_labels = {
            "random": "Random",
            "nndsvd": "NNDSVD",
            "custom1": "Custom 1",
            "custom2": "Custom 2"
        }
        nmf_labels = {
            "nmf_standard": "Standard NMF"
        }
        methods_readable = [method_labels.get(m, m) for m in methods]
        clustering_readable = [clustering_labels.get(c, c) for c in clustering]
        init_readable = [init_labels.get(i, i) for i in init_methods]
        nmf_readable = nmf_labels.get(nmf_alg, nmf_alg)

        def prepare_metrics_display_dataframe(df, nmf_algorithm_label):
            display_df = df.copy()

            if "nmf_algorithm" not in display_df.columns:
                display_df.insert(
                1,
                "nmf_algorithm",
                nmf_algorithm_label
            )

            for col in display_df.columns:
                if col == "k":
                    continue

                if pd.api.types.is_numeric_dtype(display_df[col]):
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x:.3f}" if pd.notnull(x) else ""
                    )

            return display_df
        
        try:
            experiment_output = run_k_experiments(
                dataset_data=dataset_data,
                k_min=k_min,
                k_max=k_max,
                init_methods=init_methods
            )
            results_df = pd.DataFrame(experiment_output["metrics"])
            display_results_df = filter_k_metrics_columns(
                results_df=results_df,
                methods=methods,
                clustering=clustering
            )

            display_results_df = prepare_metrics_display_dataframe(
            display_results_df,
            nmf_readable
            )
        except Exception as e:
            alert = error_card(
                "Experiment error",
                f"Error during experiment: {str(e)}"
            )
            return alert, alert, dash.no_update
        
        if results_df.empty:
            alert = warning_card(
                "No results available",
                "The experiment completed but returned no results."
            )
            return alert, alert, dash.no_update
        
        k_col = "k" if "k" in results_df.columns else results_df.columns[0]
        reconstruction_col = (
            "reconstruction_error_mean"
            if "reconstruction_error_mean" in results_df.columns
            else None
        )
        silhouette_col = None
        if "silhouette_kmeans_mean" in results_df.columns:
            silhouette_col = "silhouette_kmeans_mean"
        elif "silhouette_argmax_mean" in results_df.columns:
            silhouette_col = "silhouette_argmax_mean"
        elif "silhouette_fcm_mean" in results_df.columns:
            silhouette_col = "silhouette_fcm_mean"
        cophenetic_col = None
        if "coph_kmeans" in results_df.columns:
            cophenetic_col = "coph_kmeans"
        elif "coph_argmax" in results_df.columns:
            cophenetic_col = "coph_argmax"
        elif "coph_fcm_hard" in results_df.columns:
            cophenetic_col = "coph_fcm_hard"
        elif "coph_fcm_soft_dot" in results_df.columns:
            cophenetic_col = "coph_fcm_soft_dot"
        elif "coph_fcm_soft_cosine" in results_df.columns:
            cophenetic_col = "coph_fcm_soft_cosine"
        
        try:
            if "silhouette" in methods and silhouette_col:
                best_k = results_df.loc[
                    results_df[silhouette_col].idxmax(),
                    k_col
                ]
                best_metric_label = "Silhouette Score"
            elif "cophenetic" in methods and cophenetic_col:
                best_k = results_df.loc[
                    results_df[cophenetic_col].idxmax(),
                    k_col
                ]
                best_metric_label = "Cophenetic Index"
            elif reconstruction_col:
                best_k = results_df.loc[
                    results_df[reconstruction_col].idxmin(),
                    k_col
                ]
                best_metric_label = "Reconstruction Error"
            else:
                best_k = results_df[k_col].iloc[0]
                best_metric_label = "First available k"
        except Exception:
            best_k = results_df[k_col].iloc[0]
            best_metric_label = "First available k"
        best_k = int(best_k)
        
        suggested_k_card = dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        "Suggested k",
                        className="text-muted mb-1",
                        style={"fontSize": "13px"}
                    ),
                    html.H4(
                        str(best_k),
                        className="mb-1",
                        style={
                            "fontWeight": "700",
                            "color": "#2c3e50"
                        }
                    ),
                    html.Small(
                        f"Based on {best_metric_label}.",
                        className="text-muted"
                    )
                ],
                style={"padding": "16px"}
            ),
            className="mb-3 shadow-sm border-0",
            style={
                "backgroundColor": "#f8fbfd",
                "borderLeft": "5px solid #52b2cf",
                "borderRadius": "10px"
            }
        )
        metrics_table = html.Div(
            [
                suggested_k_card,
                dash_table.DataTable(
                    data=display_results_df.to_dict("records"),
                    columns=[
                        {
                            "name": (
                                "NMF Algorithm"
                                if col == "nmf_algorithm"
                                else str(col).replace("_", " ").title()
                            ),
                            "id": col
                        }
                        for col in display_results_df.columns
                    ],
                    page_size=10,
                    page_action="native",
                    sort_action="native",
                    filter_action="none",
                    style_table={
                        "overflowX": "auto",
                        "overflowY": "hidden",
                        "borderRadius": "12px",
                        "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                        "border": "1px solid #e9ecef"
                    },
                    style_cell={
                        "textAlign": "center",
                        "padding": "7px",
                        "fontFamily": "Poppins, Arial, sans-serif",
                        "fontSize": "13px",
                        "minWidth": "120px",
                        "maxWidth": "220px",
                        "whiteSpace": "normal",
                        "height": "auto"
                    },
                    style_header={
                        "backgroundColor": "#52b2cf",
                        "color": "white",
                        "fontWeight": "700",
                        "fontSize": "13px",
                        "textAlign": "center",
                        "padding": "8px",
                        "border": "1px solid #52b2cf"
                    },
                    style_data={
                        "backgroundColor": "white",
                        "color": "#2c3e50",
                        "border": "1px solid #f1f1f1"
                    },
                    style_data_conditional=[
                        {
                            "if": {
                                "filter_query": f"{{{k_col}}} = {best_k}"
                            },
                            "backgroundColor": "#d9f0f7",
                            "fontWeight": "700",
                            "border": "1px solid #52b2cf"
                        },
                        {
                            "if": {
                                "row_index": "odd"
                            },
                            "backgroundColor": "#f8fbfd"
                        },
                        {
                            "if": {
                                "state": "active"
                            },
                            "backgroundColor": "#d9f0f7",
                            "border": "1px solid #52b2cf"
                        },
                        {
                            "if": {
                                "state": "selected"
                            },
                            "backgroundColor": "#d9f0f7",
                            "border": "1px solid #52b2cf"
                        }
                    ]
                )
            ]
        )
        
        dataset_name = (
            dataset_data.get("display_name")
            or dataset_data.get("filename")
            or "Uploaded Dataset"
        )
        summary_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H5(
                        "Selected Configuration",
                        className="mb-2",
                        style={
                            "fontWeight": "700",
                            "color": "#2c3e50"
                        }
                    ),
                    html.P(
                        "Summary of the settings used during the k-selection experiment.",
                        className="text-muted mb-4",
                        style={"fontSize": "14px"}
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                config_card("Dataset", dataset_name),
                                md=6,
                                className="mb-3"
                            ),
                            dbc.Col(
                                config_card("Calculated k", best_k),
                                md=6,
                                className="mb-3"
                            ),
                            dbc.Col(
                                config_card(
                                    "Evaluation Methods",
                                    ", ".join(methods_readable)
                                ),
                                md=6,
                                className="mb-3"
                            ),
                            dbc.Col(
                                config_card(
                                    "Clustering Algorithms",
                                    ", ".join(clustering_readable)
                                ),
                                md=6,
                                className="mb-3"
                            ),
                            dbc.Col(
                                config_card(
                                    "Initialization Methods",
                                    ", ".join(init_readable)
                                ),
                                md=6
                            ),
                            dbc.Col(
                                config_card(
                                    "NMF Algorithm",
                                    nmf_readable
                                ),
                                md=6
                            ),
                        ],
                        className="g-3"
                    )
                ],
                style={"padding": "22px"}
            ),
            className="shadow-sm border-0",
            style={
                "backgroundColor": "#f8fbfd",
                "borderLeft": "5px solid #52b2cf",
                "borderRadius": "10px"
            }
        )
        return summary_card, metrics_table, {
            "completed": True,
            "k_min": k_min,
            "k_max": k_max,
            "suggested_k": int(best_k),
            "metrics": results_df.to_dict("records"),
            "displayed_metrics": display_results_df.to_dict("records"),
            "columns": results_df.columns.tolist(),
            "displayed_columns": display_results_df.columns.tolist(),
            "artifacts": experiment_output.get("artifacts", {}),
            "selected_inits": experiment_output.get("selected_inits", []),
            "sample_names": dataset_data.get("sample_names", []),
            "feature_names": dataset_data.get("columns", []),
            "run_metrics": pd.DataFrame(
                experiment_output.get("run_metrics", [])
            ).to_dict("records")
        }
    
    @dash_app.callback(
        Output("k-run-warning", "children"),
        Input("run-k-selection", "n_clicks"),
        State("method-selection", "value"),
        State("clustering-selection", "value"),
        State("init-selection", "value"),
        State("nmf-selection", "value"),
        State("k-min", "value"),
        State("k-max", "value"),
        State("dataset-store", "data"),
        prevent_initial_call=True
    )
    def show_run_experiment_warning(n_clicks, methods, clustering, init_methods, nmf_alg, k_min, k_max, dataset_data):
        if not dataset_data:
            return dbc.Alert(
                    "No dataset loaded. Please upload a dataset first.",
                    color="danger",
                    className="mt-2 p-2 mb-0"
                )
        if not methods:
            return dbc.Alert(
            "Please select at least one evaluation method.",
            color="warning",
            className="mt-2 p-2 mb-0"
            )
        if not clustering:
            return dbc.Alert(
                "Please select at least one clustering algorithm.",
                color="warning",
                className="mt-2 p-2 mb-0"
            )
        if not init_methods:
            return dbc.Alert(
                "Please select at least one initialization method.",
                color="warning",
                className="mt-2 p-2 mb-0"
            )
        if not nmf_alg:
                return dbc.Alert(
                "Please select an NMF algorithm.",
                color="warning",
                className="mt-2 p-2 mb-0"
            )
        if k_min is None or k_max is None:
            return dbc.Alert(
            "Please define both minimum and maximum k.",
            color="warning",
            className="mt-2 p-2 mb-0"
            )
        if k_min < 2:
            return dbc.Alert(
            "Minimum k must be at least 2.",
            color="warning",
            className="mt-2 p-2 mb-0"
            )
        max_features = dataset_data["shape"][1]
        if k_max > max_features:
            return dbc.Alert(
                f"Maximum k cannot exceed the number of features ({max_features})."
,
                color="warning",
                className="mt-2 p-2 mb-0"
            )
        if k_min > k_max:
            return dbc.Alert(
                "Minimum k cannot be greater than maximum k.",
                color="warning",
                className="mt-2 p-2 mb-0"
                )
        return ""
    
    @dash_app.callback(
    Output("k-selection-graph", "figure"),
    Input("result-method-selector", "value"),
    Input("clustering-selection", "value"),
    Input("k-experiment-status", "data"),
    prevent_initial_call=False
    )
    def update_k_selection_graph(selected_method, clustering_selection, k_status):

        fig = go.Figure()

        fixed_layout = dict(
            template="plotly_white",
            autosize=False,
            height=750,
            margin=dict(l=50, r=50, t=80, b=130),
            font=dict(
                family="Poppins, Arial",
                size=13,
                color="#2c3e50"
            ),
            uirevision="k-selection-fixed-size"
        )

        if not k_status or not k_status.get("metrics"):
            fig.update_layout(
                title="k-selection graph",
                **fixed_layout
            )
            fig.add_annotation(
                text="Run the k-selection experiment to visualize the results.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="#6c757d")
            )
            return fig

        df = pd.DataFrame(k_status.get("metrics", []))

        if df.empty or "k" not in df.columns:
            fig.update_layout(
                title="k-selection graph",
                **fixed_layout
            )
            return fig

        if not clustering_selection:
            clustering_selection = []

        suggested_k = k_status.get("suggested_k")

        method_labels = {
            "elbow": "Elbow Method - Reconstruction Error by k",
            "silhouette": "Silhouette Score by k",
            "cophenetic": "Cophenetic Index by k"
        }

        yaxis_labels = {
            "elbow": "Reconstruction Error",
            "silhouette": "Silhouette Score",
            "cophenetic": "Cophenetic Index"
        }

        metric_columns = {}

        if selected_method == "elbow":
            metric_columns = {
                "Reconstruction Error": "reconstruction_error_mean"
            }

        elif selected_method == "silhouette":
            if "argmax" in clustering_selection:
                metric_columns["Argmax"] = "silhouette_argmax_mean"

            if "kmeans" in clustering_selection:
                metric_columns["K-Means"] = "silhouette_kmeans_mean"

            if "fcm" in clustering_selection:
                metric_columns["Fuzzy C-Means"] = "silhouette_fcm_mean"

        elif selected_method == "cophenetic":
            if "argmax" in clustering_selection:
                metric_columns["Argmax"] = "coph_argmax"

            if "kmeans" in clustering_selection:
                metric_columns["K-Means"] = "coph_kmeans"

            if "fcm" in clustering_selection:
                metric_columns["FCM Hard"] = "coph_fcm_hard"
                metric_columns["FCM Soft Dot"] = "coph_fcm_soft_dot"
                metric_columns["FCM Soft Cosine"] = "coph_fcm_soft_cosine"

        if not metric_columns:
            fig.update_layout(
                title=method_labels.get(selected_method, "k-selection graph"),
                **fixed_layout
            )
            fig.add_annotation(
                text="No compatible metric selected for the current clustering configuration.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=15, color="#6c757d")
            )
            return fig

        if "init" not in df.columns:
            df["init"] = "default"

        init_values = sorted(df["init"].dropna().unique())

        for init_value in init_values:
            init_df = df[df["init"] == init_value].sort_values("k")

            for metric_label, column_name in metric_columns.items():
                if column_name not in init_df.columns:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=init_df["k"],
                        y=init_df[column_name],
                        mode="lines+markers",
                        name=f"{metric_label} - {init_value}",
                        marker=dict(size=8),
                        line=dict(width=2),
                        customdata=[[init_value] for _ in range(len(init_df))],
                        hovertemplate=(
                            "Initialization: %{customdata[0]}<br>"
                            "k: %{x}<br>"
                            f"{metric_label}: "
                            "%{y:.4f}<extra></extra>"
                        )
                    )
                )

        if suggested_k is not None:
            fig.add_vline(
                x=suggested_k,
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Suggested k = {suggested_k}",
                annotation_position="top right"
            )

        fig.update_layout(
            title=method_labels.get(selected_method, "k-selection graph"),
            xaxis_title="k",
            yaxis_title=yaxis_labels.get(selected_method, "Metric value"),
            hovermode="x unified",
            legend=dict(
                title="Metric / Initialization",
                orientation="h",
                yanchor="bottom",
                y=-0.32,
                xanchor="center",
                x=0.5
            ),
            **fixed_layout
        )

        fig.update_xaxes(
            dtick=1,
            tickmode="linear",
            automargin=True
        )

        fig.update_yaxes(
            automargin=True
        )

        return fig

    
    @dash_app.callback(
        Output("confirm-k-message", "children"),
        Output("selected-k-store", "data"),
        Input("confirm-k", "n_clicks"),
        State("selected-k", "value"),
        State("dataset-store", "data"),
        State("k-experiment-status", "data"),
        prevent_initial_call=True
    )
    def confirm_selected_k(n_clicks, selected_k, dataset_data, experiment_status):
        if not n_clicks:
            return dash.no_update, dash.no_update
        
        def warning_card(title, message):
            return dbc.Card(
                dbc.CardBody([
                    html.Div(
                        title,
                        className="mb-1",
                        style={
                            "fontWeight": "700",
                            "color": "#856404"
                        }
                    ),
                    html.Small(
                        message,
                        className="text-muted"
                    )
                ]),
                className="mt-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#fff8e1",
                    "borderLeft": "5px solid #ffc107",
                    "borderRadius": "10px"
                }
            )
        
        def error_card(title, message):
            return dbc.Card(
                dbc.CardBody([
                    html.Div(
                        title,
                        className="mb-1",
                        style={
                            "fontWeight": "700",
                            "color": "#842029"
                        }
                    ),
                    html.Small(
                        message,
                        className="text-muted"
                    )
                ]),
                className="mt-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#f8d7da",
                    "borderLeft": "5px solid #dc3545",
                    "borderRadius": "10px"
                }
            )
        
        if not dataset_data:
            return (
                error_card(
                    "Dataset not loaded",
                    "Please upload a dataset before confirming the selected k."
                ),
                dash.no_update
            )
        if not experiment_status or not experiment_status.get("completed"):
            return (
                warning_card(
                    "Experiment required",
                    "Please run the k-selection experiment before confirming k."
                ),
                dash.no_update
            )
        if selected_k is None:
            return (
                warning_card(
                    "Missing k value",
                    "Please enter a valid k value before confirming."
                ),
                dash.no_update
            )
        selected_k = int(selected_k)
        max_features = dataset_data["shape"][1]
        if selected_k < 2:
            return (
                warning_card(
                    "Invalid k value",
                    "k must be at least 2."
                ),
                dash.no_update
            )
        if selected_k > max_features:
            return (
                warning_card(
                    "Invalid k value",
                    f"k cannot exceed the number of features ({max_features})."
                ),
                dash.no_update
            )
        confirmation_card = dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.Strong("Selected k confirmed: "),
                    html.Span(str(selected_k))
                ], className="mb-1"),
                html.Small(
                    "This value will be used in the final NMF computation.",
                    className="text-muted"
                )
            ]),
            className="mt-3 shadow-sm border-0",
            style={
                "backgroundColor": "#f8fbfd",
                "borderLeft": "5px solid #52b2cf",
                "borderRadius": "10px"
            }
        )
        return (
            confirmation_card,
            {"selected_k": selected_k}
        )
    
    @dash_app.callback(
        Output("selected-k-display", "children"),
        Input("selected-k-store", "data")
    )
    def show_selected_k(data):
        if not data or "selected_k" not in data:
            return "No k has been selected yet. Please go back to Step 2."
        return f"Selected k: {data['selected_k']}"
    
    @dash_app.callback(
        Output("consensus-fcm-mode-container", "style"),
        Input("consensus-method-selector", "value")
    )
    def toggle_fcm_consensus_mode(method):
        if method == "fcm":
            return {"display": "block"}
        return {"display": "none"}
    
    @dash_app.callback(
        Output("final-nmf-summary", "children"),
        Output("cluster-output", "children"),
        Output("nmf-results-store", "data"),
        Output("run-nmf-warning", "children"),
        Input("run-final-nmf", "n_clicks"),
        State("selected-k-store", "data"),
        State("final-clustering", "value"),
        State("final-init", "value"),
        State("final-nmf", "value"),
        State("dataset-store", "data"),
        prevent_initial_call=True
    )
    def run_final_nmf_summary(
        n_clicks,
        selected_k_data,
        final_clustering,
        final_init,
        final_nmf,
        dataset_data
    ):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        if not selected_k_data or "selected_k" not in selected_k_data:
            alert = dbc.Alert(
                "No k has been selected yet. Please go back to Step 2.",
                color="danger",
                className="mt-2 p-2 mb-0"
            )
            return dash.no_update, dash.no_update, dash.no_update, alert
        
        if not dataset_data:
            alert = dbc.Alert(
                "No dataset loaded. Please upload a dataset first.",
                color="danger",
                className="mt-2 p-2 mb-0"
            )
            return dash.no_update, dash.no_update, dash.no_update, alert
        
        if not final_clustering:
            alert = dbc.Alert(
                "Please select a clustering algorithm before running NMF.",
                color="warning",
                className="mt-2 p-2 mb-0"
            )
            return dash.no_update, dash.no_update, dash.no_update, alert
        
        if not final_init:
            alert = dbc.Alert(
                "Please select an initialization method before running NMF.",
                color="warning",
                className="mt-2 p-2 mb-0"
            )
            return dash.no_update, dash.no_update, dash.no_update, alert
        
        if not final_nmf:
            alert = dbc.Alert(
                "Please select an NMF algorithm before running NMF.",
                color="warning",
                className="mt-2 p-2 mb-0"
            )
            return dash.no_update, dash.no_update, dash.no_update, alert
        
        k_value = selected_k_data["selected_k"]
        
        try:
            nmf_results_data = run_final_nmf(
                dataset_data=dataset_data,
                selected_k=k_value,
                final_init=final_init
            )
            nmf_results_data["final_clustering"] = final_clustering
            nmf_results_data["final_init"] = final_init
            nmf_results_data["final_nmf"] = final_nmf
        except Exception as e:
            alert = dbc.Alert(
                f"Error while running final NMF: {str(e)}",
                color="danger",
                className="mt-2 p-2 mb-0"
            )
            return dash.no_update, dash.no_update, dash.no_update, alert
        
        clustering_labels = {
            "argmax": "Argmax",
            "kmeans": "K-Means",
            "fcm": "Fuzzy C-Means"
        }
        init_labels = {
            "random": "Random",
            "nndsvd": "NNDSVD"
        }
        nmf_labels = {
            "nmf_standard": "Standard NMF"
        }
        dataset_name = (
            dataset_data.get("display_name")
            or dataset_data.get("filename")
            or "Uploaded Dataset"
        )
        
        def config_card(title, value):
            return dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            title,
                            className="text-muted mb-1",
                            style={
                                "fontSize": "13px"
                            }
                        ),
                        html.H5(
                            str(value),
                            className="mb-0",
                            style={
                                "fontWeight": "700",
                                "color": "#2c3e50",
                                "fontSize": "17px"
                            }
                        )
                    ],
                    style={
                        "padding": "18px"
                    }
                ),
                className="text-center shadow-sm border-0",
                style={
                    "borderRadius": "12px",
                    "backgroundColor": "white"
                }
            )
        
        summary = dbc.Card(
            dbc.CardBody(
                [
                    html.H5(
                        "Final NMF Configuration",
                        className="mb-2",
                        style={
                            "fontWeight": "700",
                            "color": "#2c3e50"
                        }
                    ),
                    html.P(
                        "Summary of the selected configuration used for the final NMF computation.",
                        className="text-muted mb-4",
                        style={
                            "fontSize": "14px"
                        }
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                config_card("Dataset", dataset_name),
                                md=6,
                                className="mb-3"
                            ),
                            dbc.Col(
                                config_card("Selected k", k_value),
                                md=6,
                                className="mb-3"
                            ),
                            dbc.Col(
                                config_card(
                                    "Clustering Algorithm",
                                    clustering_labels.get(final_clustering, final_clustering)
                                ),
                                md=4
                            ),
                            dbc.Col(
                                config_card(
                                    "Initialization",
                                    init_labels.get(final_init, final_init)
                                ),
                                md=4
                            ),
                            dbc.Col(
                                config_card(
                                    "NMF Algorithm",
                                    nmf_labels.get(final_nmf, final_nmf)
                                ),
                                md=4
                            ),
                        ],
                        className="g-3"
                    )
                ],
                style={
                    "padding": "22px"
                }
            ),
            className="shadow-sm border-0",
            style={
                "backgroundColor": "#f8fbfd",
                "borderLeft": "5px solid #52b2cf",
                "borderRadius": "10px"
            }
        )
        
        clusters = nmf_results_data.get("clusters", {})
        selected_clusters = clusters.get(final_clustering)
        if selected_clusters is None or len(selected_clusters) == 0:
            cluster_output = dbc.Card(
                dbc.CardBody(
                    [
                        html.H6(
                            "Cluster Assignments",
                            className="mb-2",
                            style={
                                "fontWeight": "700",
                                "color": "#2c3e50"
                            }
                        ),
                        html.P(
                            "No cluster assignments were found in the NMF results.",
                            className="mb-0 text-muted"
                        )
                    ],
                    style={
                        "padding": "16px"
                    }
                ),
                className="mt-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#f8fbfd",
                    "borderLeft": "5px solid #ffc107",
                    "borderRadius": "10px"
                }
            )
        else:
            _, sample_labels = get_dataset_labels(dataset_data)
            cluster_rows = [
                {
                    "Sample": sample_labels[i] if i < len(sample_labels) else f"Sample {i+1}",
                    "Cluster": int(c) + 1
                }
                for i, c in enumerate(selected_clusters)
            ]
            cluster_interpretation = dbc.Card(
                dbc.CardBody(
                    [
                        html.H6(
                            "Cluster Assignments Interpretation",
                            className="mb-2",
                            style={
                                "fontWeight": "700",
                                "color": "#2c3e50"
                            }
                        ),
                        html.P(
                            (
                                "This table reports the final cluster assignment for each sample "
                                "after applying NMF and the selected clustering algorithm. "
                                "Cluster labels are shown using a user-friendly numbering system starting from 1."
                            ),
                            className="mb-0",
                            style={
                                "fontSize": "13px",
                                "lineHeight": "1.6",
                                "color": "#2c3e50"
                            }
                        )
                    ],
                    style={
                        "padding": "16px"
                    }
                ),
                className="mt-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#f8fbfd",
                    "borderLeft": "5px solid #52b2cf",
                    "borderRadius": "10px"
                }
            )
            cluster_output = html.Div(
                [
                    html.H5(
                        "Cluster Assignments",
                        className="mb-2",
                        style={
                            "fontWeight": "700",
                            "color": "#2c3e50"
                        }
                    ),
                    html.P(
                        "Final sample-to-cluster assignments produced by the selected clustering algorithm.",
                        className="text-muted mb-3",
                        style={
                            "fontSize": "14px"
                        }
                    ),
                    dash_table.DataTable(
                        data=cluster_rows,
                        columns=[
                            {
                                "name": "Sample",
                                "id": "Sample"
                            },
                            {
                                "name": "Cluster",
                                "id": "Cluster"
                            },
                        ],
                        page_size=10,
                        page_action="native",
                        sort_action="native",
                        filter_action="none",
                        style_table={
                            "overflowX": "auto",
                            "overflowY": "hidden",
                            "borderRadius": "12px",
                            "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                            "border": "1px solid #e9ecef"
                        },
                        style_header={
                            "backgroundColor": "#52b2cf",
                            "color": "white",
                            "fontWeight": "700",
                            "fontSize": "13px",
                            "textAlign": "center",
                            "padding": "8px",
                            "border": "1px solid #52b2cf"
                        },
                        style_cell={
                            "textAlign": "center",
                            "padding": "7px",
                            "fontSize": "13px",
                            "fontFamily": "Poppins, Arial, sans-serif",
                            "minWidth": "120px",
                            "maxWidth": "220px",
                            "whiteSpace": "normal",
                            "height": "auto"
                        },
                        style_data={
                            "backgroundColor": "white",
                            "color": "#2c3e50",
                            "border": "1px solid #f1f1f1"
                        },
                        style_data_conditional=[
                            {
                                "if": {
                                    "row_index": "odd"
                                },
                                "backgroundColor": "#f8fbfd"
                            },
                            {
                                "if": {
                                    "state": "active"
                                },
                                "backgroundColor": "#d9f0f7",
                                "border": "1px solid #52b2cf"
                            },
                            {
                                "if": {
                                    "state": "selected"
                                },
                                "backgroundColor": "#d9f0f7",
                                "border": "1px solid #52b2cf"
                            }
                        ],
                    ),
                    cluster_interpretation
                ]
            )
        return summary, cluster_output, nmf_results_data, ""

    @dash_app.callback(
    Output("centroids-representatives-output", "children"),
    Input("nmf-results-store", "data"),
    prevent_initial_call=False
    )
    def show_centroids_and_representatives(nmf_results):

        if not nmf_results or not nmf_results.get("nmf_completed"):
            return dbc.Alert(
                "Run the final NMF to display centroids and representative vectors.",
                color="light"
            )

        centroids = nmf_results.get("centroids", {})
        representatives = nmf_results.get("representatives", {})

        if not centroids and not representatives:
            return dbc.Alert(
                "No centroids or representative vectors found in the backend output.",
                color="warning"
            )

        def make_table(title, matrix, row_prefix="Cluster"):
            if matrix is None or len(matrix) == 0:
                return None

            df = pd.DataFrame(matrix)
            df.index = [f"{row_prefix} {i + 1}" for i in range(len(df))]
            df.columns = [f"LF{i + 1}" for i in range(df.shape[1])]
            df.insert(0, "Cluster", df.index)

            for col in df.columns:
                if col != "Cluster":
                    df[col] = df[col].apply(lambda x: f"{float(x):.3f}")

            return dbc.Card(
                dbc.CardBody([
                    html.H5(
                        title,
                        className="mb-2",
                        style={
                            "fontWeight": "700",
                            "color": "#2c3e50"
                        }
                    ),
                    html.P(
                        "Rows represent clusters and columns represent latent factors.",
                        className="text-muted mb-3",
                        style={"fontSize": "14px"}
                    ),
                    dash_table.DataTable(
                        data=df.to_dict("records"),
                        columns=[
                            {"name": col, "id": col}
                            for col in df.columns
                        ],
                        page_size=10,
                        style_table={
                            "overflowX": "auto",
                            "borderRadius": "12px",
                            "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                            "border": "1px solid #e9ecef"
                        },
                        style_cell={
                            "textAlign": "center",
                            "padding": "8px",
                            "fontFamily": "Poppins, Arial, sans-serif",
                            "fontSize": "13px"
                        },
                        style_header={
                            "backgroundColor": "#52b2cf",
                            "color": "white",
                            "fontWeight": "700"
                        },
                        style_data_conditional=[
                            {
                                "if": {"row_index": "odd"},
                                "backgroundColor": "#f8fbfd"
                            }
                        ]
                    )
                ]),
                className="mb-4 shadow-sm border-0",
                style={
                    "backgroundColor": "#f8fbfd",
                    "borderLeft": "5px solid #52b2cf",
                    "borderRadius": "10px"
                }
            )

        sections = []

        sections.append(
            html.H4(
                "Centroids & Representative Vectors",
                className="mb-3",
                style={
                    "fontWeight": "700",
                    "color": "#2c3e50"
                }
            )
        )

        sections.append(
            dbc.Alert(
                "K-Means and FCM centroids are computed in the latent space H. "
                "Representative vectors are computed as the mean latent-factor profile of samples assigned to each cluster.",
                color="info",
                className="mb-4"
            )
        )

        if representatives:
            method_labels = {
                "argmax": "Argmax Mean Representative Vectors",
                "kmeans": "K-Means Mean Representative Vectors",
                "fcm_hard": "FCM Hard Mean Representative Vectors"
            }

            for method, matrix in representatives.items():
                table = make_table(
                    method_labels.get(method, method),
                    matrix
                )
                if table:
                    sections.append(table)

        if centroids:
            centroid_labels = {
                "kmeans": "K-Means Centroids",
                "fcm": "Fuzzy C-Means Centroids"
            }

            for method, matrix in centroids.items():
                table = make_table(
                    centroid_labels.get(method, method),
                    matrix
                )
                if table:
                    sections.append(table)

        return html.Div(sections)

    
    @dash_app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("run-warning", "children"),
        Input("run-next-btn", "n_clicks"),
        Input("nmf-results-store", "data"),
        prevent_initial_call=True
    )
    def handle_run_next_click(n_clicks, nmf_results):
            triggered = ctx.triggered_id
            if triggered == "nmf-results-store":
                if nmf_results and nmf_results.get("nmf_completed"):
                    return dash.no_update, ""
                return dash.no_update, dash.no_update
            if triggered == "run-next-btn":
                if not nmf_results or not nmf_results.get("nmf_completed"):
                    return dash.no_update, dbc.Alert(
                    "Please run the final NMF before continuing.",
                    color="warning",
                    className="mt-2 p-2 mb-0"
                )
                return "/fuzzy", ""
            return dash.no_update, dash.no_update
    

    @dash_app.callback(
    Output("matrix-w-plot", "figure"),
    Output("matrix-h-plot", "figure"),
    Output("nmf-w-heatmap-note", "children"),
    Output("nmf-h-heatmap-note", "children"),
    Input("nmf-results-store", "data"),
    State("dataset-store", "data"),
    prevent_initial_call=False
    )
    def update_final_nmf_plots(nmf_results, dataset_data):

        fig_w = go.Figure()
        fig_h = go.Figure()

        binary_colorscale = [
            [0.0, "blue"],
            [0.499, "blue"],
            [0.5, "red"],
            [1.0, "red"]
        ]

        fixed_layout_w = dict(
            template="plotly_white",
            autosize=False,
            height=700,
            margin=dict(l=90, r=90, t=80, b=110),
            font=dict(
                family="Poppins, Arial",
                size=13,
                color="#2c3e50"
            ),
            uirevision="matrix-w-fixed-size"
        )

        fixed_layout_h = dict(
            template="plotly_white",
            autosize=False,
            height=700,
            margin=dict(l=90, r=90, t=80, b=130),
            font=dict(
                family="Poppins, Arial",
                size=13,
                color="#2c3e50"
            ),
            uirevision="matrix-h-fixed-size"
        )

        if not nmf_results or not nmf_results.get("nmf_completed"):

            fig_w.update_layout(
                title="Matrix W Binary Heatmap",
                **fixed_layout_w
            )
            fig_w.add_annotation(
                text="Run the final NMF to visualize Matrix W.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="#6c757d")
            )

            fig_h.update_layout(
                title="Matrix H Binary Heatmap",
                **fixed_layout_h
            )
            fig_h.add_annotation(
                text="Run the final NMF to visualize Matrix H.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="#6c757d")
            )

            return fig_w, fig_h, "", ""

        try:
            W = nmf_results.get("W_norm") or nmf_results.get("W")
            H = nmf_results.get("H_norm") or nmf_results.get("H")

            if W is None or H is None:
                artifacts = nmf_results.get("artifacts", {})
                W = artifacts.get("W_norm") or artifacts.get("W")
                H = artifacts.get("H_norm") or artifacts.get("H")

            if W is None or H is None:
                raise ValueError("W or H matrix not found in NMF results.")

            W = np.asarray(W, dtype=float)
            H = np.asarray(H, dtype=float)

            # ============================================================
            # BINARIZZAZIONE MATRIX W
            # Ogni feature viene associata al latent factor dominante.
            # ============================================================

            W_binary = np.zeros_like(W)
            dominant_lf_for_feature = np.argmax(W, axis=1)
            W_binary[np.arange(W.shape[0]), dominant_lf_for_feature] = 1

            # ============================================================
            # BINARIZZAZIONE MATRIX H
            # Ogni sample viene associato al latent factor dominante.
            # ============================================================

            H_binary = np.zeros_like(H)
            dominant_lf_for_sample = np.argmax(H, axis=0)
            H_binary[dominant_lf_for_sample, np.arange(H.shape[1])] = 1

            feature_labels, sample_labels = get_dataset_labels(dataset_data)

            if not feature_labels or len(feature_labels) != W.shape[0]:
                feature_labels = [f"Feature {i + 1}" for i in range(W.shape[0])]

            if not sample_labels or len(sample_labels) != H.shape[1]:
                sample_labels = [f"Sample {i + 1}" for i in range(H.shape[1])]

            latent_factor_labels = [f"LF{i + 1}" for i in range(W.shape[1])]

            fig_w = go.Figure(
                data=go.Heatmap(
                    z=W_binary,
                    x=latent_factor_labels,
                    y=feature_labels,
                    colorscale=binary_colorscale,
                    zmin=0,
                    zmax=1,
                    colorbar=dict(
                        title="Binary Activation",
                        thickness=14,
                        len=0.75,
                        tickvals=[0, 1],
                        ticktext=[
                            "0 = inactive",
                            "1 = active"
                        ]
                    ),
                    hovertemplate=(
                        "Feature: %{y}<br>"
                        "Latent Factor: %{x}<br>"
                        "Binary Value: %{z:.0f}<extra></extra>"
                    )
                )
            )

            fig_w.update_layout(
                title="Matrix W Binary Heatmap",
                xaxis_title="Latent Factors",
                yaxis_title="Features",
                **fixed_layout_w
            )

            fig_w.update_xaxes(
                tickangle=45,
                showgrid=False,
                automargin=True
            )

            fig_w.update_yaxes(
                autorange="reversed",
                showgrid=False,
                automargin=True
            )

            n_samples = len(sample_labels)
            tick_step_h = max(1, n_samples // 12)

            tick_indices_h = list(range(0, n_samples, tick_step_h))

            if (n_samples - 1) not in tick_indices_h:
                tick_indices_h.append(n_samples - 1)

            tick_values_h = [sample_labels[i] for i in tick_indices_h]
            tick_text_h = [sample_labels[i] for i in tick_indices_h]

            fig_h = go.Figure(
                data=go.Heatmap(
                    z=H_binary,
                    x=sample_labels,
                    y=[f"LF{i + 1}" for i in range(H.shape[0])],
                    colorscale=binary_colorscale,
                    zmin=0,
                    zmax=1,
                    colorbar=dict(
                        title="Binary Activation",
                        thickness=14,
                        len=0.75,
                        tickvals=[0, 1],
                        ticktext=[
                            "0 = inactive",
                            "1 = active"
                        ]
                    ),
                    hovertemplate=(
                        "Latent Factor: %{y}<br>"
                        "Sample: %{x}<br>"
                        "Binary Value: %{z:.0f}<extra></extra>"
                    )
                )
            )

            fig_h.update_layout(
                title="Matrix H Binary Heatmap",
                xaxis_title="Samples",
                yaxis_title="Latent Factors",
                **fixed_layout_h
            )

            fig_h.update_xaxes(
                tickmode="array",
                tickvals=tick_values_h,
                ticktext=tick_text_h,
                tickangle=35,
                tickfont=dict(size=10),
                showgrid=False,
                automargin=True
            )

            fig_h.update_yaxes(
                autorange="reversed",
                showgrid=False,
                tickfont=dict(size=12),
                automargin=True
            )

            w_download_note = html.P(
                "Use the camera icon in the graph toolbar to download the Matrix W binary heatmap as PNG.",
                className="text-muted mt-3 mb-0",
                style={
                    "fontSize": "13px"
                }
            )

            h_download_note = html.P(
                "Use the camera icon in the graph toolbar to download the Matrix H binary heatmap as PNG.",
                className="text-muted mt-3 mb-0",
                style={
                    "fontSize": "13px"
                }
            )

            w_note = dbc.Card(
                dbc.CardBody([

                    html.H6(
                        "Matrix W Binary Heatmap Interpretation",
                        className="mb-2",
                        style={
                            "fontWeight": "700",
                            "color": "#2c3e50"
                        }
                    ),

                    html.P(
                        (
                            "This binary heatmap represents the dominant latent factor for each dataset feature. "
                            "A value of 1 indicates that the feature is mainly associated with that latent factor, "
                            "whereas 0 indicates no dominant association for that latent factor."
                        ),
                        style={
                            "fontSize": "14px",
                            "color": "#2c3e50"
                        }
                    ),

                    dbc.Row([
                        dbc.Col(
                            dbc.Badge(
                                "Blue = inactive",
                                color="primary",
                                className="p-2 w-100"
                            ),
                            md=6
                        ),
                        dbc.Col(
                            dbc.Badge(
                                "Red = active",
                                color="danger",
                                className="p-2 w-100"
                            ),
                            md=6
                        ),
                    ], className="g-2 mt-2")

                ]),
                className="mt-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#f8fbfd",
                    "borderLeft": "5px solid #52b2cf",
                    "borderRadius": "10px"
                }
            )

            h_note = dbc.Card(
                dbc.CardBody([

                    html.H6(
                        "Matrix H Binary Heatmap Interpretation",
                        className="mb-2",
                        style={
                            "fontWeight": "700",
                            "color": "#2c3e50"
                        }
                    ),

                    html.P(
                        (
                            "This binary heatmap represents the dominant latent factor for each sample. "
                            "A value of 1 indicates that the sample is mainly associated with that latent factor, "
                            "whereas 0 indicates no dominant association for that latent factor."
                        ),
                        style={
                            "fontSize": "14px",
                            "color": "#2c3e50"
                        }
                    ),

                    dbc.Row([
                        dbc.Col(
                            dbc.Badge(
                                "Blue = inactive",
                                color="primary",
                                className="p-2 w-100"
                            ),
                            md=6
                        ),
                        dbc.Col(
                            dbc.Badge(
                                "Red = active",
                                color="danger",
                                className="p-2 w-100"
                            ),
                            md=6
                        ),
                    ], className="g-2 mt-2")

                ]),
                className="mt-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#f8fbfd",
                    "borderLeft": "5px solid #52b2cf",
                    "borderRadius": "10px"
                }
            )

            return (
                fig_w,
                fig_h,

                html.Div([
                    w_download_note,
                    w_note
                ]),

                html.Div([
                    h_download_note,
                    h_note
                ])
            )

        except Exception as e:

            fig_error = go.Figure()
            fig_error.update_layout(
                title="Matrix Binary Heatmap",
                template="plotly_white",
                autosize=False,
                height=700,
                uirevision="matrix-error-fixed-size"
            )
            fig_error.add_annotation(
                text=f"Error while generating heatmaps:<br>{str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=15, color="red")
            )

            return fig_error, fig_error, "", ""
    
    
    @dash_app.callback(
        Output("fuzzy-w-output", "children"),
        Output("fuzzy-h-output", "children"),
        Output("fuzzy-examples", "children"),
        Output("fuzzy-settings", "data"),
        Output("fuzzy-results", "data"),
        Output("fuzzy-run-warning", "children"),
        Input("run-fuzzy", "n_clicks"),
        State("num-fuzzy-sets", "value"),
        State("fuzzy-method", "value"),
        State("fuzzy-shape", "value"),
        State("fuzzy-target", "value"),
        State("nmf-results-store", "data"),
        State("dataset-store", "data"),
        prevent_initial_call=True
    )
    def generate_fuzzy_outputs(
        n_clicks,
        num_sets,
        fuzzy_method,
        fuzzy_shape,
        fuzzy_target,
        nmf_results,
        dataset_data
    ):
        if not n_clicks:
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update
            )
        
        def warning_card(title, message):
            return dbc.Card(
                dbc.CardBody([
                    html.Div(
                        title,
                        className="mb-1",
                        style={
                            "fontWeight": "700",
                            "color": "#856404"
                        }
                    ),
                    html.Small(
                        message,
                        className="text-muted"
                    )
                ], style={"padding": "16px"}),
                className="mt-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#fff8e1",
                    "borderLeft": "5px solid #ffc107",
                    "borderRadius": "10px"
                }
            )
        
        def error_card(title, message):
            return dbc.Card(
                dbc.CardBody([
                    html.Div(
                        title,
                        className="mb-1",
                        style={
                            "fontWeight": "700",
                            "color": "#842029"
                        }
                    ),
                    html.Small(
                        message,
                        className="text-muted"
                    )
                ], style={"padding": "16px"}),
                className="mt-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#f8d7da",
                    "borderLeft": "5px solid #dc3545",
                    "borderRadius": "10px"
                }
            )
        
        def info_card(title, value):
            return dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            title,
                            className="text-muted mb-1",
                            style={"fontSize": "13px"}
                        ),
                        html.H5(
                            str(value),
                            className="mb-0",
                            style={
                                "fontWeight": "700",
                                "color": "#2c3e50",
                                "fontSize": "17px"
                            }
                        )
                    ],
                    style={"padding": "18px"}
                ),
                className="text-center shadow-sm border-0",
                style={
                    "borderRadius": "12px",
                    "backgroundColor": "white"
                }
            )
        
        def interpretation_card(title, text):
            return dbc.Card(
                dbc.CardBody(
                    [
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
                                "fontSize": "13px",
                                "lineHeight": "1.6",
                                "color": "#2c3e50"
                            }
                        )
                    ],
                    style={"padding": "16px"}
                ),
                className="mb-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#f8fbfd",
                    "borderLeft": "5px solid #52b2cf",
                    "borderRadius": "10px"
                }
            )
        
        def empty_section_card(title, message):
            return dbc.Card(
                dbc.CardBody(
                    [
                        html.H6(
                            title,
                            className="mb-2",
                            style={
                                "fontWeight": "700",
                                "color": "#2c3e50"
                            }
                        ),
                        html.P(
                            message,
                            className="mb-0 text-muted",
                            style={"fontSize": "13px"}
                        )
                    ],
                    style={"padding": "16px"}
                ),
                className="mt-3 shadow-sm border-0",
                style={
                    "backgroundColor": "#f8fbfd",
                    "borderLeft": "5px solid #ffc107",
                    "borderRadius": "10px"
                }
            )
        
        def build_table_from_rows(rows, first_col_name="Element"):
            if not rows:
                return empty_section_card(
                    "No data available",
                    "No fuzzy table was generated for this section."
                )
            normalized_rows = []
            for row in rows:
                clean_row = {}
                for key, value in row.items():
                    clean_key = make_readable_label(key)
                    clean_value = make_readable_label(value) if isinstance(value, str) else value
                    clean_row[clean_key] = clean_value
                normalized_rows.append(clean_row)
            columns = [
                {
                    "name": col,
                    "id": col
                }
                for col in normalized_rows[0].keys()
            ]
            return dash_table.DataTable(
                data=normalized_rows,
                columns=columns,
                page_size=10,
                page_action="native",
                sort_action="native",
                filter_action="none",
                style_table={
                    "overflowX": "auto",
                    "overflowY": "hidden",
                    "borderRadius": "12px",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                    "border": "1px solid #e9ecef"
                },
                style_header={
                    "backgroundColor": "#52b2cf",
                    "color": "white",
                    "fontWeight": "700",
                    "fontSize": "13px",
                    "textAlign": "center",
                    "padding": "8px",
                    "border": "1px solid #52b2cf"
                },
                style_cell={
                    "textAlign": "center",
                    "padding": "7px",
                    "fontSize": "13px",
                    "fontFamily": "Poppins, Arial, sans-serif",
                    "minWidth": "120px",
                    "maxWidth": "220px",
                    "whiteSpace": "normal",
                    "height": "auto"
                },
                style_data={
                    "backgroundColor": "white",
                    "color": "#2c3e50",
                    "border": "1px solid #f1f1f1"
                },
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": "#f8fbfd"
                    },
                    {
                        "if": {"state": "active"},
                        "backgroundColor": "#d9f0f7",
                        "border": "1px solid #52b2cf"
                    },
                    {
                        "if": {"state": "selected"},
                        "backgroundColor": "#d9f0f7",
                        "border": "1px solid #52b2cf"
                    }
                ]
            )
        

        def description_card(title, text):
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
                            "fontSize": "13px",
                            "lineHeight": "1.7",
                            "color": "#2c3e50"
                        }
                    )
                ]),
                className="mb-3 shadow-sm border-0",
                style={
                    "backgroundColor": "white",
                    "borderRadius": "12px"
                }
            )


        def build_w_cards(w_rows, w_descriptions=None):
            if w_descriptions is None:
                w_descriptions = []

            if not w_rows:
                return empty_section_card(
                    "No latent factor explanations available",
                    "Select Matrix W and generate fuzzy explanations to display this section."
                )

            factors = {}

            for row in w_rows:
                factor = (
                    row.get("Latent Factor")
                    or row.get("latent_factor")
                    or row.get("LF")
                    or row.get("Factor")
                )

                if not factor:
                    factor = "Latent Factor"

                factors.setdefault(factor, []).append(row)

            children = [
                html.H5(
                    "Latent Factor Explanations (Matrix W)",
                    className="mb-2",
                    style={
                        "fontWeight": "700",
                        "color": "#2c3e50"
                    }
                ),

                html.P(
                    "Fuzzy linguistic labels describe how strongly each dataset feature contributes to each latent factor.",
                    className="text-muted mb-3",
                    style={"fontSize": "14px"}
                ),

                interpretation_card(
                    "Matrix W Interpretation",
                    (
                        "Matrix W represents the relationship between dataset features and latent factors. "
                        "Latent factor explanations are obtained by summarizing the fuzzy contribution "
                        "levels of the dataset features associated with each latent factor."
                    )
                )
            ]

            # ============================================================
            # TEXTUAL LATENT FACTOR EXPLANATIONS - CARD VERSION
            # ============================================================

            if w_descriptions:
                factor_cards = []

                for desc in w_descriptions:
                    factor_title = "Latent Factor Explanation"

                    if desc.startswith("The influence of ") and " is:" in desc:
                        try:
                            factor_title = desc.split("The influence of ")[1].split(" is:")[0]
                            factor_title = make_readable_label(factor_title)
                        except Exception:
                            factor_title = "Latent Factor Explanation"

                    factor_cards.append(
                        dbc.Card(
                            dbc.CardBody([
                                html.H6(
                                    factor_title,
                                    className="mb-2",
                                    style={
                                        "fontWeight": "700",
                                        "color": "#2c3e50"
                                    }
                                ),

                                html.P(
                                    desc,
                                    className="mb-0",
                                    style={
                                        "fontSize": "13px",
                                        "lineHeight": "1.7",
                                        "color": "#2c3e50"
                                    }
                                )
                            ]),
                            className="mb-3 shadow-sm border-0",
                            style={
                                "backgroundColor": "white",
                                "borderRadius": "12px"
                            }
                        )
                    )

                children.append(
                    dbc.Card(
                        dbc.CardBody([
                            html.H6(
                                "Latent Factor-Level Fuzzy Explanations",
                                className="mb-3",
                                style={
                                    "fontWeight": "700",
                                    "color": "#2c3e50"
                                }
                            ),

                            html.Div(factor_cards)
                        ]),
                        className="mb-4 shadow-sm border-0",
                        style={
                            "backgroundColor": "#f8fbfd",
                            "borderLeft": "5px solid #52b2cf",
                            "borderRadius": "10px"
                        }
                    )
                )

            # ============================================================
            # FUZZY TABLES - DO NOT REMOVE
            # ============================================================

            children.append(
                html.H5(
                    "Latent Factor Fuzzy Tables",
                    className="mb-3 mt-4",
                    style={
                        "fontWeight": "700",
                        "color": "#2c3e50"
                    }
                )
            )

            for factor_name, factor_rows in factors.items():
                clean_rows = []

                for row in factor_rows:
                    clean_row = {
                        key: value
                        for key, value in row.items()
                        if key not in ["Latent Factor", "latent_factor", "LF", "Factor"]
                    }
                    clean_rows.append(clean_row)

                children.append(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6(
                                    make_readable_label(factor_name),
                                    className="mb-3",
                                    style={
                                        "fontWeight": "700",
                                        "color": "#2c3e50"
                                    }
                                ),

                                build_table_from_rows(clean_rows)
                            ],
                            style={"padding": "18px"}
                        ),
                        className="mb-4 shadow-sm border-0",
                        style={
                            "backgroundColor": "white",
                            "borderRadius": "12px"
                        }
                    )
                )

            return html.Div(children)


        def build_h_cards(h_tables, h_descriptions=None):
            if h_descriptions is None:
                h_descriptions = []

            if not h_tables:
                return empty_section_card(
                    "No cluster explanations available",
                    "Select Matrix H and generate fuzzy explanations to display this section."
                )

            method_labels = {
                "argmax": "Argmax",
                "kmeans": "K-Means",
                "fcm_hard": "Fuzzy C-Means",
                "fcm": "Fuzzy C-Means"
            }

            children = [
                html.H5(
                    "Cluster Explanations (Matrix H)",
                    className="mb-2",
                    style={
                        "fontWeight": "700",
                        "color": "#2c3e50"
                    }
                ),

                html.P(
                    "Fuzzy linguistic labels summarize the latent-factor profile associated with each cluster.",
                    className="text-muted mb-3",
                    style={"fontSize": "14px"}
                ),

                interpretation_card(
                    "Matrix H Interpretation",
                    (
                        "Matrix H represents the activation of latent factors across samples. "
                        "Cluster explanations are obtained by summarizing the latent-factor profiles "
                        "of the samples assigned to each cluster. These fuzzy explanations help interpret "
                        "the structure discovered by the selected clustering algorithms."
                    )
                )
            ]

            # ============================================================
            # TEXTUAL CLUSTER EXPLANATIONS - CARD VERSION
            # ============================================================

            if h_descriptions:
                grouped_descriptions = {}

                for desc in h_descriptions:
                    method_name = "Method"

                    if desc.startswith("[") and "]" in desc:
                        method_name = desc.split("]")[0].replace("[", "")
                        clean_desc = desc.split("]", 1)[1].strip()
                    else:
                        clean_desc = desc

                    method_title = method_labels.get(
                        method_name,
                        make_readable_label(method_name)
                    )

                    grouped_descriptions.setdefault(method_title, []).append(clean_desc)

                method_sections = []

                for method_title, descriptions in grouped_descriptions.items():
                    cluster_cards = []

                    for desc in descriptions:
                        cluster_title = "Cluster Explanation"

                        if "Samples in " in desc and " are generally" in desc:
                            try:
                                cluster_title = desc.split("Samples in ")[1].split(" are generally")[0]
                                cluster_title = make_readable_label(cluster_title)
                            except Exception:
                                cluster_title = "Cluster Explanation"

                        cluster_cards.append(
                            dbc.Card(
                                dbc.CardBody([
                                    html.H6(
                                        cluster_title,
                                        className="mb-2",
                                        style={
                                            "fontWeight": "700",
                                            "color": "#2c3e50"
                                        }
                                    ),

                                    html.P(
                                        desc,
                                        className="mb-0",
                                        style={
                                            "fontSize": "13px",
                                            "lineHeight": "1.7",
                                            "color": "#2c3e50"
                                        }
                                    )
                                ]),
                                className="mb-3 shadow-sm border-0",
                                style={
                                    "backgroundColor": "white",
                                    "borderRadius": "12px"
                                }
                            )
                        )

                    method_sections.append(
                        dbc.Card(
                            dbc.CardBody([
                                html.H6(
                                    method_title,
                                    className="mb-3",
                                    style={
                                        "fontWeight": "700",
                                        "color": "#2c3e50"
                                    }
                                ),

                                html.Div(cluster_cards)
                            ]),
                            className="mb-3 shadow-sm border-0",
                            style={
                                "backgroundColor": "white",
                                "borderRadius": "12px"
                            }
                        )
                    )

                children.append(
                    dbc.Card(
                        dbc.CardBody([
                            html.H6(
                                "Cluster-Level Fuzzy Explanations",
                                className="mb-3",
                                style={
                                    "fontWeight": "700",
                                    "color": "#2c3e50"
                                }
                            ),

                            html.Div(method_sections)
                        ]),
                        className="mb-4 shadow-sm border-0",
                        style={
                            "backgroundColor": "#f8fbfd",
                            "borderLeft": "5px solid #52b2cf",
                            "borderRadius": "10px"
                        }
                    )
                )

            # ============================================================
            # FUZZY TABLES - DO NOT REMOVE
            # ============================================================

            children.append(
                html.H5(
                    "Cluster Fuzzy Tables",
                    className="mb-3 mt-4",
                    style={
                        "fontWeight": "700",
                        "color": "#2c3e50"
                    }
                )
            )

            for method_name, rows in h_tables.items():
                method_title = method_labels.get(
                    method_name,
                    make_readable_label(method_name)
                )

                children.append(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6(
                                    method_title,
                                    className="mb-3",
                                    style={
                                        "fontWeight": "700",
                                        "color": "#2c3e50"
                                    }
                                ),

                                build_table_from_rows(rows)
                            ],
                            style={"padding": "18px"}
                        ),
                        className="mb-4 shadow-sm border-0",
                        style={
                            "backgroundColor": "white",
                            "borderRadius": "12px"
                        }
                    )
                )

            return html.Div(children)


        def build_sample_cards(sample_rows, sample_descriptions=None):
            if sample_descriptions is None:
                sample_descriptions = []

            if not sample_rows:
                return empty_section_card(
                    "No example interpretations available",
                    "Generate fuzzy explanations to display sample-level interpretations."
                )

            samples = {}

            for row in sample_rows:
                sample = (
                    row.get("Sample")
                    or row.get("sample")
                    or row.get("Example")
                    or row.get("example")
                )

                if not sample:
                    sample = "Sample"

                samples.setdefault(sample, []).append(row)

            children = [
                html.H5(
                    "Example Interpretations",
                    className="mb-2",
                    style={
                        "fontWeight": "700",
                        "color": "#2c3e50"
                    }
                ),

                html.P(
                    "Sample-level fuzzy explanations describe the latent-factor activation pattern of individual observations.",
                    className="text-muted mb-3",
                    style={"fontSize": "14px"}
                ),

                interpretation_card(
                    "Sample Interpretation",
                    (
                        "Each example is described through fuzzy labels assigned to latent factors. "
                        "This helps understand how individual samples are represented in the NMF latent space."
                    )
                )
            ]

            # ============================================================
            # TEXTUAL SAMPLE EXPLANATIONS
            # ============================================================

            if sample_descriptions:
                sample_description_cards = []

                for desc in sample_descriptions:
                    sample_title = "Sample Explanation"

                    if " on " in desc and " is:" in desc:
                        try:
                            sample_title = desc.split(" on ")[1].split(" is:")[0]
                            sample_title = make_readable_label(sample_title)
                        except Exception:
                            sample_title = "Sample Explanation"

                    sample_description_cards.append(
                        dbc.Card(
                            dbc.CardBody([
                                html.H6(
                                    sample_title,
                                    className="mb-2",
                                    style={
                                        "fontWeight": "700",
                                        "color": "#2c3e50"
                                    }
                                ),

                                html.P(
                                    desc,
                                    className="mb-0",
                                    style={
                                        "fontSize": "13px",
                                        "lineHeight": "1.7",
                                        "color": "#2c3e50"
                                    }
                                )
                            ]),
                            className="mb-3 shadow-sm border-0",
                            style={
                                "backgroundColor": "white",
                                "borderRadius": "12px"
                            }
                        )
                    )

                children.append(
                    dbc.Card(
                        dbc.CardBody([
                            html.H6(
                                "Sample-Level Fuzzy Explanations",
                                className="mb-3",
                                style={
                                    "fontWeight": "700",
                                    "color": "#2c3e50"
                                }
                            ),

                            html.Div(sample_description_cards)
                        ]),
                        className="mb-4 shadow-sm border-0",
                        style={
                            "backgroundColor": "#f8fbfd",
                            "borderLeft": "5px solid #52b2cf",
                            "borderRadius": "10px"
                        }
                    )
                )

            # ============================================================
            # SAMPLE FUZZY TABLES
            # ============================================================

            children.append(
                html.H5(
                    "Sample Fuzzy Tables",
                    className="mb-3 mt-4",
                    style={
                        "fontWeight": "700",
                        "color": "#2c3e50"
                    }
                )
            )

            for sample_name, rows in samples.items():
                clean_rows = []

                for row in rows:
                    clean_row = {
                        key: value
                        for key, value in row.items()
                        if key not in ["Sample", "sample", "Example", "example"]
                    }
                    clean_rows.append(clean_row)

                children.append(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6(
                                    make_readable_label(sample_name),
                                    className="mb-3",
                                    style={
                                        "fontWeight": "700",
                                        "color": "#2c3e50"
                                    }
                                ),

                                build_table_from_rows(clean_rows)
                            ],
                            style={"padding": "18px"}
                        ),
                        className="mb-4 shadow-sm border-0",
                        style={
                            "backgroundColor": "white",
                            "borderRadius": "12px"
                        }
                    )
                )

            return html.Div(children)


        if not nmf_results or not nmf_results.get("nmf_completed"):
            alert = warning_card(
                "Final NMF required",
                "Please run the final NMF before generating fuzzy explanations."
            )
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                alert
            )

        if num_sets is None or num_sets < 2:
            alert = warning_card(
                "Invalid number of fuzzy sets",
                "Please select at least 2 fuzzy sets."
            )
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                alert
            )

        if fuzzy_method != "equidistant":
            alert = warning_card(
                "Method not available",
                "Only the Equidistant fuzzy creation method is currently connected to the backend."
            )
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                alert
            )

        if fuzzy_shape != "gaussian":
            alert = warning_card(
                "Shape not available",
                "Only Gaussian membership functions are currently implemented in the backend."
            )
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                alert
            )

        if not fuzzy_target:
            alert = warning_card(
                "Missing target matrix",
                "Please select at least one target matrix."
            )
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                alert
            )

        try:
            fuzzy_results = run_fuzzy_from_nmf_results(
                nmf_results=nmf_results,
                dataset_data=dataset_data,
                n_fuzzy_sets=num_sets,
                targets=fuzzy_target,
            )
        except Exception as e:
            alert = error_card(
                "Fuzzy generation error",
                f"Error while generating fuzzy explanations: {str(e)}"
            )
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                alert
            )

        method_labels = {
            "equidistant": "Equidistant",
            "quartile": "Quartile-Based",
            "manual": "Manual"
        }

        shape_labels = {
            "gaussian": "Gaussian",
            "triangular": "Triangular",
            "trapezoidal": "Trapezoidal"
        }

        target_labels = {
            "W": "Matrix W",
            "H": "Matrix H"
        }

        applied_to = ", ".join(
            target_labels.get(target, target)
            for target in fuzzy_target
        )

        w_rows = fuzzy_results.get("w_fuzzy_table", [])
        h_tables = fuzzy_results.get("h_fuzzy_tables", {})
        sample_rows = fuzzy_results.get("sample_fuzzy_table", [])

        w_descriptions = fuzzy_results.get("w_descriptions", [])
        h_descriptions = fuzzy_results.get("h_descriptions", [])
        sample_descriptions = fuzzy_results.get("sample_descriptions", [])

        if "W" in fuzzy_target:
            w_output = build_w_cards(
                w_rows,
                w_descriptions
            )
        else:
            w_output = empty_section_card(
                "No W fuzzy representation available",
                "Matrix W was not selected as a fuzzy explanation target."
            )

        if "H" in fuzzy_target:
            h_output = build_h_cards(
                h_tables,
                h_descriptions
            )
        else:
            h_output = empty_section_card(
                "No cluster explanations available",
                "Matrix H was not selected as a fuzzy explanation target."
            )

        examples_output = build_sample_cards(
            sample_rows,
            sample_descriptions
        )

        settings_data = {
            "fuzzy_completed": True,
            "num_fuzzy_sets": num_sets,
            "fuzzy_method": fuzzy_method,
            "fuzzy_shape": fuzzy_shape,
            "fuzzy_target": fuzzy_target,
            "results": fuzzy_results
        }

        save_explanations(fuzzy_results)

        return (
            w_output,
            h_output,
            examples_output,
            settings_data,
            fuzzy_results,
            ""
        )


    @dash_app.callback(
        Output("sample-explanation-selector", "options"),
        Output("sample-explanation-selector", "value"),
        Input("fuzzy-results", "data"),
        prevent_initial_call=False
    )
    def update_sample_explanation_selector(fuzzy_results):

        if not fuzzy_results:
            return [], None

        sample_rows = fuzzy_results.get("sample_fuzzy_table", [])

        if not sample_rows:
            return [], None

        sample_names = [
            row.get("Sample")
            for row in sample_rows
            if row.get("Sample")
        ]

        options = [
            {
                "label": sample_name,
                "value": sample_name
            }
            for sample_name in sample_names
        ]

        default_value = sample_names[0] if sample_names else None

        return options, default_value
    

    @dash_app.callback(
        Output("selected-sample-explanation", "children"),
        Input("sample-explanation-selector", "value"),
        State("fuzzy-results", "data"),
        prevent_initial_call=False
    )
    def show_selected_sample_explanation(selected_sample, fuzzy_results):

        if not fuzzy_results or not selected_sample:
            return dbc.Alert(
                "Generate fuzzy explanations and select a sample.",
                color="light"
            )

        sample_rows = fuzzy_results.get("sample_fuzzy_table", [])
        sample_descriptions = fuzzy_results.get("sample_descriptions", [])

        selected_row = None

        for row in sample_rows:
            if row.get("Sample") == selected_sample:
                selected_row = row
                break

        if selected_row is None:
            return dbc.Alert(
                "Selected sample not found.",
                color="warning"
            )

        selected_description = None

        for desc in sample_descriptions:
            if selected_sample in desc:
                selected_description = desc
                break

        rows = []

        for key, value in selected_row.items():
            if key == "Sample":
                continue

            rows.append(
                html.Tr([
                    html.Td(
                        html.Strong(str(key)),
                        style={
                            "width": "40%",
                            "verticalAlign": "middle"
                        }
                    ),
                    html.Td(
                        str(value),
                        style={
                            "verticalAlign": "middle"
                        }
                    )
                ])
            )

        children = [
            html.H6(
                f"Explanation for {selected_sample}",
                className="mb-3",
                style={
                    "fontWeight": "700",
                    "color": "#2c3e50"
                }
            )
        ]

        if selected_description:
            children.append(
                dbc.Card(
                    dbc.CardBody([
                        html.H6(
                            "Textual Explanation",
                            className="mb-2",
                            style={
                                "fontWeight": "700",
                                "color": "#2c3e50"
                            }
                        ),

                        html.P(
                            selected_description,
                            className="mb-0",
                            style={
                                "fontSize": "13px",
                                "lineHeight": "1.7",
                                "color": "#2c3e50"
                            }
                        )
                    ]),
                    className="mb-3 shadow-sm border-0",
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "12px"
                    }
                )
            )

        children.append(
            html.H6(
                "Sample Fuzzy Table",
                className="mb-3 mt-3",
                style={
                    "fontWeight": "700",
                    "color": "#2c3e50"
                }
            )
        )

        children.append(
            dbc.Table(
                [
                    html.Thead(
                        html.Tr([
                            html.Th("Latent Factor"),
                            html.Th("Fuzzy Label")
                        ])
                    ),
                    html.Tbody(rows)
                ],
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
                size="sm"
            )
        )

        return dbc.Card(
            dbc.CardBody(children),
            className="shadow-sm border-0",
            style={
                "backgroundColor": "#f8fbfd",
                "borderLeft": "5px solid #52b2cf",
                "borderRadius": "10px"
            }
        )


    @dash_app.callback(
        Output("methods-overview-output", "children"),
        Input("fuzzy-settings", "data"),
        State("nmf-results-store", "data"),
        prevent_initial_call=False
    )
    def show_methods_overview(fuzzy_settings, nmf_results):

        if not fuzzy_settings:
            return dbc.Alert(
                "Generate fuzzy explanations to display the methods overview.",
                color="light"
            )

        config = nmf_results.get("config", {}) if nmf_results else {}

        fuzzy_method = fuzzy_settings.get("fuzzy_method", "equidistant")
        fuzzy_shape = fuzzy_settings.get("fuzzy_shape", "gaussian")
        fuzzy_target = fuzzy_settings.get("fuzzy_target", [])
        num_sets = fuzzy_settings.get("num_fuzzy_sets", "-")

        selected_k = nmf_results.get("selected_k", "-") if nmf_results else "-"

        init_labels = {
            "random": "Random",
            "nndsvd": "NNDSVD"
        }

        target_labels = {
            "W": "Matrix W",
            "H": "Matrix H"
        }

        readable_init = init_labels.get(
            str(config.get("nmf_init", "Selected in Step 3")).lower(),
            str(config.get("nmf_init", "Selected in Step 3"))
        )

        applied_to = ", ".join(
            target_labels.get(target, target)
            for target in fuzzy_target
        ) if fuzzy_target else "-"


        def method_card(title, value, description):
            return dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.Div(
                            title,
                            className="text-muted mb-1",
                            style={
                                "fontSize": "13px",
                                "fontWeight": "600"
                            }
                        ),

                        html.H6(
                            value,
                            className="mb-2",
                            style={
                                "fontWeight": "700",
                                "color": "#2c3e50"
                            }
                        ),

                        html.P(
                            description,
                            className="mb-0",
                            style={
                                "fontSize": "13px",
                                "lineHeight": "1.6",
                                "color": "#2c3e50"
                            }
                        )
                    ]),
                    className="h-100 shadow-sm border-0",
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "12px"
                    }
                ),
                md=6,
                className="mb-3"
            )

        return dbc.Card(
            dbc.CardBody([

                html.H5(
                    "Methods Overview",
                    className="mb-2",
                    style={
                        "fontWeight": "700",
                        "color": "#2c3e50"
                    }
                ),

                html.P(
                    "This section summarizes the computational methods used to generate the final NMF results and fuzzy explanations.",
                    className="text-muted mb-4",
                    style={"fontSize": "14px"}
                ),

                dbc.Row([

                    method_card(
                        "NMF Algorithm",
                        "Standard NMF",
                        "Non-negative Matrix Factorization is used to decompose the dataset into latent factors."
                    ),

                    method_card(
                        "Selected k",
                        str(selected_k),
                        "The selected number of latent factors used to compute the final NMF decomposition."
                    ),

                    method_card(
                        "Initialization Method",
                        readable_init,
                        "The initialization strategy defines how the NMF factor matrices are initialized before optimization."
                    ),

                    method_card(
                        "Clustering Methods",
                        "Argmax, K-Means, Fuzzy C-Means",
                        "Different clustering strategies are applied in the latent-factor space to assign samples to groups."
                    ),

                    method_card(
                        "Cluster Explanations",
                        "Representative vectors",
                        "Each cluster is described using the mean latent-factor profile of the samples assigned to that cluster."
                    ),

                    method_card(
                        "Centroids",
                        "K-Means and Fuzzy C-Means",
                        "Centroids are computed in the latent-factor space and summarize the central profile of each cluster."
                    ),

                    method_card(
                        "Fuzzy Set Creation",
                        str(fuzzy_method).replace("_", " ").title(),
                        "The fuzzy sets define linguistic levels used to transform numerical values into interpretable labels."
                    ),

                    method_card(
                        "Membership Function",
                        str(fuzzy_shape).title(),
                        "The membership function shape controls how numerical values are mapped to fuzzy linguistic labels."
                    ),

                    method_card(
                        "Number of Fuzzy Sets",
                        str(num_sets),
                        "This parameter determines how many linguistic levels are used in the fuzzy representation."
                    ),

                    method_card(
                        "Applied To",
                        applied_to,
                        "The fuzzy explanation process is applied to the selected NMF matrices and derived representations."
                    ),

                ], className="g-3")

            ]),
            className="shadow-sm border-0",
            style={
                "backgroundColor": "#f8fbfd",
                "borderLeft": "5px solid #52b2cf",
                "borderRadius": "10px"
            }
        )

    
    @dash_app.callback(
        Output("dataset-summary", "children"),
        Input("dataset-store", "data")
    )
    def update_dataset_summary(dataset_data):
        if not dataset_data:
            return dbc.Alert("Dataset information is not available.", color="light")
        rows, cols = dataset_data["shape"]
        return dbc.Card(
            dbc.CardBody([
                html.H6("Dataset Information", className="mb-3"),
                html.Ul([
                    html.Li([html.Strong("Filename: "), dataset_data["filename"]]),
                    html.Li([html.Strong("Rows: "), str(rows)]),
                    html.Li([html.Strong("Features: "), str(cols)]),
                ])
            ]),
            className="border shadow-sm"
        )
    
    @dash_app.callback(
        Output("selected-k-report", "children"),
        Input("selected-k-store", "data"),
        Input("k-experiment-status", "data")
    )
    def update_selected_k_report(selected_k_data, k_status):
        selected_k = selected_k_data.get("selected_k") if selected_k_data else None
        suggested_k = k_status.get("suggested_k") if k_status else None
        return dbc.Card(
            dbc.CardBody([
                html.H6("Chosen k", className="mb-3"),
                html.P(f"Selected k: {selected_k if selected_k is not None else 'Not available'}"),
                html.P(f"Suggested k: {suggested_k if suggested_k is not None else 'Not available'}"),
            ]),
            className="border shadow-sm"
        )
    
    @dash_app.callback(
        Output("config-summary", "children"),
        Input("nmf-results-store", "data"),
        Input("fuzzy-settings", "data")
    )
    def update_config_summary(nmf_results, fuzzy_settings):
        return dbc.Card(
            dbc.CardBody([
                html.H6("Configuration Summary", className="mb-3"),
                html.Ul([
                    html.Li([html.Strong("Selected k: "), str(nmf_results.get("selected_k", "Not available")) if nmf_results else "Not available"]),
                    html.Li([html.Strong("Clustering algorithm: "), str(nmf_results.get("final_clustering", "Not available")) if nmf_results else "Not available"]),
                    html.Li([html.Strong("Initialization method: "), str(nmf_results.get("final_init", "Not available")) if nmf_results else "Not available"]),
                    html.Li([html.Strong("NMF algorithm: "), str(nmf_results.get("final_nmf", "Not available")) if nmf_results else "Not available"]),
                    html.Li([html.Strong("Fuzzy method: "), str(fuzzy_settings.get("fuzzy_method", "Not available")) if fuzzy_settings else "Not available"]),
                    html.Li([html.Strong("Fuzzy shape: "), str(fuzzy_settings.get("fuzzy_shape", "Not available")) if fuzzy_settings else "Not available"]),
                    html.Li([html.Strong("Fuzzy targets: "), ", ".join(fuzzy_settings.get("fuzzy_target", [])) if fuzzy_settings and fuzzy_settings.get("fuzzy_target") else "Not available"]),
                ])
            ]),
            className="border shadow-sm"
        )
    
    @dash_app.callback(
        Output("report-w", "figure"),
        Output("report-h", "figure"),
        Input("nmf-results-store", "data")
    )
    def update_final_report_plots(nmf_results):
        fig_w = go.Figure()
        fig_h = go.Figure()
        if not nmf_results:
            fig_w.add_annotation(text="Matrix W not available.", x=0.5, y=0.5, showarrow=False)
            fig_h.add_annotation(text="Matrix H not available.", x=0.5, y=0.5, showarrow=False)
            return fig_w, fig_h
        W = nmf_results.get("W_norm") or nmf_results.get("W")
        H = nmf_results.get("H_norm") or nmf_results.get("H")
        if W:
            fig_w.add_trace(go.Heatmap(z=W))
            fig_w.update_layout(title="Matrix W", template="plotly_white")
        if H:
            fig_h.add_trace(go.Heatmap(z=H))
            fig_h.update_layout(title="Matrix H", template="plotly_white")
        return fig_w, fig_h
    
    @dash_app.callback(
        Output("report-clusters", "children"),
        Input("nmf-results-store", "data")
    )
    def update_report_clusters(nmf_results):
        if not nmf_results:
            return dbc.Alert("Cluster information is not available.", color="light")
        final_clustering = nmf_results.get("final_clustering", "kmeans")
        clusters = nmf_results.get("clusters", {}).get(final_clustering)
        if clusters is None:
            return dbc.Alert("Cluster assignments are not available.", color="light")
        rows = [
            {
                "Sample": f"Sample {i + 1}",
                "Cluster": int(c) + 1
            }
            for i, c in enumerate(clusters)
        ]
        return dash_table.DataTable(
            data=rows,
            columns=[
                {"name": "Sample", "id": "Sample"},
                {"name": "Cluster", "id": "Cluster"},
            ],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "8px"},
            style_header={
                "backgroundColor": "#52b2cf",
                "color": "white",
                "fontWeight": "bold"
            }
        )
    
    @dash_app.callback(
        Output("fuzzy-report", "children"),
        Input("fuzzy-settings", "data")
    )
    def update_fuzzy_report(fuzzy_settings):
        if not fuzzy_settings or "results" not in fuzzy_settings:
            return dbc.Alert("Fuzzy explanations are not available.", color="light")
        results = fuzzy_settings["results"]
        return dbc.Card(
            dbc.CardBody([
                html.H6("Fuzzy Explanation Summary", className="mb-3"),
                html.H6("W Explanations"),
                html.Ul([html.Li(x) for x in results.get("w_descriptions", [])]),
                html.H6("H Explanations", className="mt-3"),
                html.Ul([html.Li(x) for x in results.get("h_descriptions", [])]),
                html.H6("Example Interpretations", className="mt-3"),
                html.Ul([html.Li(x) for x in results.get("sample_descriptions", [])]),
            ]),
            className="border shadow-sm"
        )
    
    @dash_app.callback(
        Output("download-all-results", "data"),
        Input("download-all-results-btn", "n_clicks"),
        State("dataset-store", "data"),
        State("selected-k-store", "data"),
        State("k-experiment-status", "data"),
        State("nmf-results-store", "data"),
        State("fuzzy-settings", "data"),
        prevent_initial_call=True
    )
    def download_all_results(
        n_clicks,
        dataset_data,
        selected_k_data,
        k_experiment_status,
        nmf_results,
        fuzzy_settings
    ):
        if not n_clicks:
            raise PreventUpdate
        
        export_data = {
            "dataset_info": {
                "filename": dataset_data.get("filename") if dataset_data else None,
                "shape": dataset_data.get("shape") if dataset_data else None,
                "columns": dataset_data.get("columns") if dataset_data else None,
            },
            "k_selection": {
                "selected_k": selected_k_data.get("selected_k") if selected_k_data else None,
                "experiment_status": k_experiment_status if k_experiment_status else{},
                "metrics": k_experiment_status.get("metrics", []) if k_experiment_status else [],
            },
            "final_nmf": {
                "completed": nmf_results.get("nmf_completed") if nmf_results else False,
                "selected_k": nmf_results.get("selected_k") if nmf_results else None,
                "configuration": {
                    "clustering_algorithm": nmf_results.get("final_clustering") if nmf_results else None,
                    "initialization_method": nmf_results.get("final_init") if nmf_results else None,
                    "nmf_algorithm": nmf_results.get("final_nmf") if nmf_results else None,
                },
                "metrics": nmf_results.get("metrics", {}) if nmf_results else {},
                "cluster_sizes": nmf_results.get("cluster_sizes", {}) if nmf_results else {},
                "clusters": nmf_results.get("clusters", {}) if nmf_results else {},
                "W": nmf_results.get("W") if nmf_results else None,
                "H": nmf_results.get("H") if nmf_results else None,
                "W_norm": nmf_results.get("W_norm") if nmf_results else None,
                "H_norm": nmf_results.get("H_norm") if nmf_results else None,
            },
            "fuzzy_explanations": {
                "completed": fuzzy_settings.get("fuzzy_completed") if fuzzy_settings else False,
                "configuration": {
                    "num_fuzzy_sets": fuzzy_settings.get("num_fuzzy_sets") if fuzzy_settings else None,
                    "fuzzy_method": fuzzy_settings.get("fuzzy_method") if fuzzy_settings else None,
                    "fuzzy_shape": fuzzy_settings.get("fuzzy_shape") if fuzzy_settings else None,
                    "fuzzy_target": fuzzy_settings.get("fuzzy_target") if fuzzy_settings else None,
                },
                "results": fuzzy_settings.get("results", {}) if fuzzy_settings else {},
            },
            "export_note": "This file contains the real outputs generated by the NMF web application workflow."
        }
        
        return {
            "content": json.dumps(export_data, indent=4, ensure_ascii=False),
            "filename": "all_results_real_outputs.json"
        }
    
    @dash_app.callback(
        Output('session-store', 'data'),
        Output('import-feedback', 'children'),
        Input('upload-fis', 'contents'),
        State('upload-fis', 'filename'),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def handle_json_import(contents, filename, session_data=None):
        if contents:
            try:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                uploaded_data = json.loads(decoded.decode('utf-8'))
                response = requests.post(
                    "http://127.0.0.1:5000/api/import_json",
                    headers={"X-Session-ID": (session_data or {}).get("sid")},
                    json=uploaded_data
                )
                if response.status_code == 200:
                    return dash.no_update, dbc.Alert("Importazione completata!", color="success", dismissable=True)
                else:
                    return dash.no_update, dbc.Alert(f"Errore: {response.json().get('error', 'Errore sconosciuto')}", color="danger", dismissable=True)
            except Exception as e:
                return dash.no_update, dbc.Alert(f"Errore durante l'import: {e}", color="danger", dismissable=True)
        return dash.no_update, dash.no_update
    
    @dash_app.callback(
            [Output("variable-modal", "is_open"),
            Output("main-content", "style"),
            Output("num-variables-store", "data")],
            [Input("modal-submit-button", "n_clicks")],
            [State("num-variables-input", "value")],
            State("session-store", "data")
        )
    def handle_modal_submit(n_clicks, num_variables, session_data=None):
        """Gestisce l'inserimento del numero di variabili da creare."""
        if not n_clicks:
            return [True, {"display": "none"}, None]
        try:
            num_vars = int(num_variables)
            if num_vars < 1:
                raise ValueError
            file_handler.save_data({"num_variables": num_vars})
            return [False, {"display": "block", "position": "relative"}, num_vars]
        except:
            return [True, {"display": "none"}, None]
    
    @dash_app.callback(
        Output("variable-title", "children"),
        Input("var-type-store", "data"),
        [Input("current-index", "data"),
        Input("num-variables-store", "data")],
        State("session-store", "data")
    )
    def update_title(var_type, current_index, num_vars, session_data=None):
        """Aggiorna il titolo della variabile in base all'indice corrente."""
        if num_vars is None or current_index is None:
            return ""
        try:
            current_index = int(current_index)
        except (ValueError, TypeError):
            return "Invalid variable index."
        return f"Creation of  {var_type} Variables {current_index + 1} of {num_vars}"
    
    @dash_app.callback(
        [Output("current-index", "data"),
        Output("back-button", "style"),
        Output("next-button", "style"),
        Output("next-button", "children"),
        Output("url", "pathname", allow_duplicate=True)],
        [Input("next-button", "n_clicks"),
        Input("back-button", "n_clicks"),
        Input("num-variables-store", "data")],
        [State("current-index", "data")],
        State("session-store", "data"),
        prevent_initial_call='initial_duplicate'
    )
    def navigate_variables(next_clicks, back_clicks, num_variables, current_index, session_data=None):
        ctx = dash.callback_context
        redirect = dash.no_update
        if not ctx.triggered:
            current_index = 0 if current_index is None else current_index
        else:
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if triggered_id == "next-button":
                if current_index < num_variables - 1:
                    current_index += 1
                else:
                    redirect = "/"
            elif triggered_id == "back-button" and current_index > 0:
                current_index -= 1
        num_variables = num_variables or 0
        is_last = (current_index == num_variables - 1)
        if num_variables <= 1:
            back_button_style = {'display': 'none'}
        else:
            back_button_style = {'display': 'none'} if current_index == 0 else {'display':'inline-block'}
        next_button_style = {'display': 'inline-block'}
        next_button_label = "Done" if is_last else "Next"
        return current_index, back_button_style, next_button_style, next_button_label, redirect
    
    @dash_app.callback(
        [
            Output('variable-name', 'value', allow_duplicate=True),
            Output('domain-min', 'value', allow_duplicate=True),
            Output('domain-max', 'value', allow_duplicate=True),
            Output('function-type', 'value', allow_duplicate=True),
            Output('term-name', 'value', allow_duplicate=True),
            Output('create-term-btn', 'children', allow_duplicate=True),
            Output('selected-term', 'data', allow_duplicate=True),
            Output('classification-term-count', 'data', allow_duplicate=True),
            Output('terms-list', 'children', allow_duplicate=True),
            Output('graph', 'figure', allow_duplicate=True)
        ],
        Input('current-index', 'data'),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def reset_static_fields(current_index, session_data=None):
        default_terms = [dbc.ListGroupItem("No Terms Present", style={"textAlign":"center"})]
        empty_graph = {
            'data': [],
            'layout': go.Layout(
                title='Fuzzy Set',
                xaxis={'title': 'Domain'},
                yaxis={'title': 'Degree of membership'}
            )
        }
        return '', 0, '', None, '', 'Create term', None, 0, default_terms, empty_graph
    
    @dash_app.callback(
        [
            Output('param-a', 'value', allow_duplicate=True),
            Output('param-b', 'value', allow_duplicate=True),
            Output('param-c', 'value', allow_duplicate=True),
            Output('param-d', 'value', allow_duplicate=True),
            Output('param-mean', 'value', allow_duplicate=True),
            Output('param-sigma', 'value', allow_duplicate=True),
        ],
        [Input('function-type', 'value')],
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def reset_fuzzy_parameters(function_type, session_data=None):
        return '', '', '', '', '', ''
    
    @dash_app.callback(
        Output('open-type', 'value'),  
        Input('open-type-radio', 'value'),
        State("session-store", "data")
    )
    def update_open_type(selected_value, session_data=None):
        return selected_value
    
    @dash_app.callback(
        Output('params-container', 'children'),
        [
            Input('var-type-store', 'data'),
            Input('function-type', 'value'),
            Input('num-variables-store', 'data'),
            Input('current-index', 'data'),
            Input('open-type', 'value')
        ],
        State("session-store", "data")
    )
    def update_params(var_type, function_type, num_variables, current_index, open_type, session_data=None):
        if not function_type or num_variables is None or current_index is None:
            return [] 
        params = []
        params.append(dbc.RadioItems(
            id='open-type-radio',
            options=[
                {'label': 'Left open', 'value': 'left'},
                {'label': 'Right open', 'value': 'right'}
            ],
            inline=True,
            value=open_type
        ))
        if function_type == 'Triangolare':
            params.append(dbc.Label("Parameter a:"))
            params.append(dbc.Input(id='param-a', type='number', value='', required=True))
            params.append(dbc.Label("Parameter b:"))
            if open_type == 'left':  
                params.append(dbc.Input(id='param-b', type='number', value='', disabled=True))
            elif open_type == 'right':
                params.append(dbc.Input(id='param-b', type='number', value='', disabled=True))
            elif open_type is None:
                params.append(dbc.Input(id='param-b', type='number', value='', required=True))
            params.append(dbc.Label("Parameter c:"))
            params.append(dbc.Input(id='param-c', type='number', value='', required=True))
            params.append(dbc.Input(id='param-d', style={'display': 'none'}))
            params.append(dbc.Input(id='param-mean', style={'display': 'none'}))
            params.append(dbc.Input(id='param-sigma', style={'display': 'none'}))
        elif function_type == 'Gaussian':
            params.append(dbc.Label("Parameter Mean:"))
            params.append(dbc.Input(id='param-mean', type='number', value='', required=True))
            params.append(dbc.Label("Parameter Sigma:"))
            params.append(dbc.Input(id='param-sigma', type='number', value='', required=True))
            params.append(dbc.Input(id='param-a', style={'display': 'none'}))
            params.append(dbc.Input(id='param-b', style={'display': 'none'}))
            params.append(dbc.Input(id='param-c', style={'display': 'none'}))
            params.append(dbc.Input(id='param-d', style={'display': 'none'}))
        elif function_type == 'Trapezoidale':
            params.append(dbc.Label("Parameter a:"))
            params.append(dbc.Input(id='param-a', type='number', value='', required=True))
            params.append(dbc.Label("Parameter b:"))
            if open_type == 'left':  
                params.append(dbc.Input(id='param-b', type='number', value='', disabled=True))
            else:
                params.append(dbc.Input(id='param-b', type='number', value='', required=True))
            params.append(dbc.Label("Parameter c:"))
            if open_type == 'right':  
                params.append(dbc.Input(id='param-c', type='number', value='', disabled=True))
            else:
                params.append(dbc.Input(id='param-c', type='number', value='', required=True))
            params.append(dbc.Label("Parameter d:"))
            params.append(dbc.Input(id='param-d', type='number', value='', required=True))
            params.append(dbc.Input(id='param-mean', style={'display': 'none'}))
            params.append(dbc.Input(id='param-sigma', style={'display': 'none'}))
        return params
    
    @dash_app.callback(
        Output('defuzzy-type', 'invalid'),
        Input('defuzzy-type', 'value'),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def validate_defuzzy_type(selected_value, session_data=None):
        if selected_value is None:
            return True  
        return False  
    
    @dash_app.callback(
        [
            Output('terms-list', 'children', allow_duplicate=True),
            Output("error-modal", "is_open"),
            Output("error-modal-body", "children"),
            Output('graph', 'figure', allow_duplicate=True),
            Output('term-name', 'value', allow_duplicate=True),  
            Output('param-a', 'value', allow_duplicate=True),     
            Output('param-b', 'value', allow_duplicate=True),     
            Output('param-c', 'value', allow_duplicate=True),     
            Output('param-d', 'value', allow_duplicate=True),     
            Output('param-mean', 'value', allow_duplicate=True),  
            Output('param-sigma', 'value', allow_duplicate=True),
            Output('classification-term-count', 'data', allow_duplicate=True), 
            Output('create-term-btn', 'children')
        ],
        [
            Input('create-term-btn', 'n_clicks'),
            Input('delete-term-btn', 'n_clicks'),
            Input('modify-term-btn', 'n_clicks'),
        ],
        [
            State('open-type', 'value'),
            State('var-type-store', 'data'), 
            State('variable-name', 'value'),
            State('domain-min', 'value'),
            State('domain-max', 'value'),
            State('function-type', 'value'),
            State('term-name', 'value'),
            State('param-a', 'value'),
            State('param-b', 'value'),
            State('param-c', 'value'),
            State('param-d', 'value'),
            State('param-mean', 'value'),
            State('param-sigma', 'value'),
            State('defuzzy-type', 'value'), 
            State('create-term-btn', 'children'),
            State('selected-term', 'data'),
        ],
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def handle_terms(create_clicks, delete_clicks, modify_clicks, open_type,
                    var_type, variable_name, domain_min, domain_max, function_type,
                    term_name, param_a, param_b, param_c, param_d, param_mean, param_sigma,
                    defuzzy_type, button_label, selected_term, session_data=None):
        """Gestisce la creazione, modifica ed eliminazione dei termini fuzzy.""" 
        ctx = dash.ctx
        if not ctx.triggered:
            return [dash.no_update] * 13
        triggered_id = ctx.triggered[0]['prop_id']
        if var_type == "input":
            defuzzy_type = None
        if open_type is None:
            open_type = []
        if isinstance(open_type, str) and ('left' in open_type or 'right' in open_type):
            function_type = f"{function_type}-open"
        # === CREAZIONE / MODIFICA ===
        if triggered_id == 'create-term-btn.n_clicks':             
            if button_label == 'Save change':
                # Verifica che il nuovo nome sia valido
                if not term_name:
                    return (
                        dash.no_update, True, "Please enter a valid term name.",
                        dash.no_update, dash.no_update, dash.no_update,
                        dash.no_update, dash.no_update, dash.no_update,
                        dash.no_update, dash.no_update, dash.no_update,
                        'Create term'
                    )
                terms_list, is_error, message, figure, count = modify_term(
                    open_type, var_type, variable_name, domain_min, domain_max,
                    function_type, term_name, param_a, param_b,
                    param_c, param_d, param_mean, param_sigma,
                    defuzzy_type, selected_term, session_data
                )
                terms_list, figure, count = update_terms_list_and_figure(variable_name, var_type, session_data)
                return (
                    terms_list, is_error, message, figure,
                    '', '', '', '', '', '', '',
                    count,
                    'Create term'
                )
            else:
                terms_list, is_error, message, figure, count = create_term(
                    open_type, var_type, variable_name, domain_min, domain_max,
                    function_type, term_name, param_a, param_b,
                    param_c, param_d, param_mean, param_sigma,
                    defuzzy_type, session_data=session_data
                )
                if message == "Term successfully created!":
                    terms_list, figure, count = update_terms_list_and_figure(variable_name, var_type, session_data)
                    return (
                        terms_list, False, "", figure,
                        '', '', '', '', '', '', '',
                        count,
                        'Create term'
                    )
                return (
                    terms_list, is_error, message, dash.no_update,
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update,
                    'Create term'
                )
        # === ELIMINAZIONE ===
        elif triggered_id == 'delete-term-btn.n_clicks':
            if not selected_term:
                return (
                    dash.no_update, True, "No term selected",
                    dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update,
                    'Create term'
                )
            delete_response = requests.post(f'http://127.0.0.1:5000/api/delete_term/{selected_term}',
                headers={"X-Session-ID": (session_data or {}).get("sid")}
            )
            if delete_response.status_code == 200:
                terms_list, figure, count = update_terms_list_and_figure(variable_name, var_type, session_data)
                return (
                    terms_list, False, f"Term '{selected_term}' successfully eliminated!",figure,
                    '', '', '', '', '', '', '',
                    count,
                    'Create term'
                )
            else:
                error_message = delete_response.json().get('error', 'Unknown Error')
                return (
                    dash.no_update, True, f"Error in term deletion: {error_message}", dash.no_update,
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update,
                    'Create term'
                )
        # === PREPARA LA MODIFICA ===
        elif triggered_id == 'modify-term-btn.n_clicks':
            if not selected_term:
                return (
                    dash.no_update, True, "No term selected",
                    dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update, dash.no_update, dash.no_update,
                    'Create term'
                )
            url = f'http://127.0.0.1:5000/api/get_term/{variable_name}/{selected_term}'
            headers = {'Content-Type': 'application/json'}
            response = requests.get(url, headers={"X-Session-ID": (session_data or {}).get("sid")})
            if response.status_code == 200:
                term_data = response.json()
                term_params = term_data.get('params', {})
                return (
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update, 
                    term_data.get('term_name', ''),
                    term_params.get('a', ''),
                    term_params.get('b', ''),
                    term_params.get('c', ''),
                    term_params.get('d', ''),
                    term_params.get('mean', ''),
                    term_params.get('sigma', ''),
                    dash.no_update,
                    'Save change'
                )
            else:
                return (
                    dash.no_update, True,
                    "Error loading term data.",
                    dash.no_update,
                    dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update, dash.no_update,
                    'Create term'
                )
        return [dash.no_update] * 13
    
    def validate_params(open_type, params, domain_min, domain_max, function_type):
        """Valida i parametri di un termine fuzzy rispetto al dominio e al tipo di funzione.""" 
        if function_type == 'Triangolare':
            a, b, c = params.get('a'), params.get('b'), params.get('c')
            if not (domain_min <= a <= domain_max and domain_min <= b <= domain_max and domain_min <= c <= domain_max):
                return False, "Parameters a, b, c shall be between the minimum and maximum domains."
            if not (a <= b <= c):
                return False, "Parameters shall respect the order a <= b <= c."
        elif function_type == 'Triangolare-open':
            if open_type == "left":
                a, b, c = params.get('a'), params.get('a'), params.get('c')
            if open_type == "right":
                a, b, c = params.get('a'), params.get('c'), params.get('c')
            if not (domain_min <= a <= domain_max and domain_min <= b <= domain_max and domain_min <= c <= domain_max):
                return False, "parameters a,b,c shall be between the maximum and minimum domains"
            if not (a <= b <= c):
                return False, "Parameters must respect the order a <= b <= c."
        elif function_type == 'Gaussian':
            mean, sigma = params.get('mean'), params.get('sigma')
            if not (domain_min <= mean <= domain_max):
                return False, "The mean parameter must be between the minimum andmaximum domains."
            if sigma <= 0:
                return False, "The sigma parameter must be greater than zero."
        elif function_type == 'Gaussian-open':
            mean, sigma = params.get('mean'), params.get('sigma')
            if not (domain_min <= mean <= domain_max):
                return False, "The mean parameter must be between the minimum andmaximum domains."
        elif function_type == 'Trapezoidale':
            a, b, c, d = params.get('a'), params.get('b'), params.get('c'), params.get('d')
            if not (domain_min <= a <= domain_max and domain_min <= b <= domain_max and domain_min <= c <= domain_max and domain_min <= d <= domain_max):
                return False, "Parameters a, b, c, d must be between the minimum and maximum domains."
            if not (a <= b <= c <= d):
                return False, "Parameters must respect the order a <= b <= c <= d."
        elif function_type == 'Trapezoidale-open':
            if open_type == "left":
                a, b, c, d = params.get('a'), params.get('a'), params.get('c'), params.get('d')
            if open_type == "right":
                a, b, c, d = params.get('a'), params.get('b'), params.get('d'), params.get('d')
            if not (domain_min <= a <= domain_max and domain_min <= b <= domain_max and domain_min <= c <= domain_max and domain_min <= d <= domain_max):
                return False, "Parameters a, b, c, d must be between the minimum and maximum domains."
            if not (a <= b <= c <= d):
                return False, "Parameters must respect the order a <= b <= c <= d."
        return True, ""
    
    def create_term(open_type, var_type, variable_name, domain_min, domain_max, function_type, term_name, param_a, param_b, param_c, param_d, param_mean, param_sigma, defuzzy_type=None, session_data=None):
        """Crea un nuovo termine fuzzy e aggiorna grafico e lista."""
        try:
            domain_min = int(domain_min)
            domain_max = int(domain_max)
        except (ValueError, TypeError):
            return dash.no_update, True, "The Domain values must be numbers.", dash.no_update, dash.no_update
        if not variable_name or not re.match(r"^[A-Za-z0-9_-]+$", variable_name):
            return dash.no_update, True, "The variable name is blank or contains invalid characters. Use only letters, numbers, hyphens, and underscores.", dash.no_update, dash.no_update
        if not term_name or not re.match(r"^[A-Za-z0-9_-]+$", term_name):
            return dash.no_update, True, "The term name is blank or contains invalidcharacters. Use only letters, numbers, hyphens, and underscores.", dash.no_update, dash.no_update
        if domain_min > domain_max:
            return dash.no_update, True, "The minimum domain cannot be greater than the maximum domain.", dash.no_update, dash.no_update
        params = {}
        if function_type == 'Triangolare':
            params = {'a': param_a, 'b': param_b, 'c': param_c}
        elif function_type == 'Triangolare-open':
            if open_type == 'left':
                params = {'a': param_a, 'b': param_a, 'c': param_c}
            elif open_type == 'right':
                params = {'a': param_a, 'b': param_c, 'c': param_c}
        elif function_type == 'Gaussian':
            params = {'mean': param_mean, 'sigma': param_sigma}
        elif function_type == 'Gaussian-open':
            params = {'mean': param_mean, 'sigma': param_sigma}
        elif function_type == 'Trapezoidale':
            params = {'a': param_a, 'b': param_b, 'c': param_c, 'd': param_d}
        elif function_type == 'Trapezoidale-open':
            if open_type == 'left':
                params = {'a': param_a, 'b': param_a, 'c': param_c, 'd': param_d}
            elif open_type == 'right':
                params = {'a': param_a, 'b': param_b, 'c': param_d, 'd': param_d}
        elif function_type == 'Classification':
            params = {}  
        is_valid, error_message = validate_params(open_type, params, domain_min, domain_max, function_type)
        if not is_valid:
            return dash.no_update, True, error_message, dash.no_update, dash.no_update
        payload = {
            'var_type': var_type,
            'term_name': term_name,
            'variable_name': variable_name,
            'domain_min': domain_min,
            'domain_max': domain_max,
            'function_type': function_type,
            'params': params
        }
        if function_type and 'open' in function_type and open_type:
            payload['open_type'] = open_type
        if var_type == "output" and defuzzy_type:
            payload['defuzzy_type'] = defuzzy_type
        response = requests.post('http://127.0.0.1:5000/api/create_term',
                    headers={"X-Session-ID": (session_data or {}).get("sid")},
                    json=payload)
        if response.status_code == 201:
            terms_list, figure, count = update_terms_list_and_figure(variable_name, var_type)
            return terms_list, True, "Term successfully created!", figure, count
        else:
            error_message = response.json().get('error', 'Unknown error')
            return dash.no_update, True, f"{error_message}", dash.no_update, dash.no_update
    
    @dash_app.callback(
        [
            Output('selected-term', 'data'),
            Output({'type': 'term-item', 'index': dash.ALL}, 'style')
        ],
        Input({'type': 'term-item', 'index': dash.ALL}, 'n_clicks'),
        State({'type': 'term-item', 'index': dash.ALL}, 'id'),
        State("session-store", "data")
    )
    def update_selected_term_and_styles(n_clicks_list, ids, session_data=None):
        """Aggiorna il termine selezionato e applica lo stile evidenziato.""" 
        default_style = {'cursor': 'pointer'}
        if not n_clicks_list or all(nc is None for nc in n_clicks_list):
            return dash.no_update, [default_style for _ in ids]
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, [default_style for _ in ids]
        triggered_prop = ctx.triggered[0]['prop_id']
        triggered_id_str = triggered_prop.split('.')[0]
        try:
            triggered_id = json.loads(triggered_id_str)
        except Exception:
            return dash.no_update, [default_style for _ in ids]
        selected_term = triggered_id.get('index')
        styles = []
        for item in ids:
            if item.get('index') == selected_term:
                styles.append({'cursor': 'pointer', 'backgroundColor': '#cce5ff'})
            else:
                styles.append(default_style)
        return selected_term, styles
    
    @dash_app.callback(
        [
            Output('modify-term-btn', 'disabled'),
            Output('delete-term-btn', 'disabled')
        ],
        Input('selected-term', 'data'),
        State("session-store", "data")
    )
    def update_buttons(selected_term, session_data=None):
        """Abilita o disabilita i pulsanti di modifica/eliminazione in base alla selezione.""" 
        if selected_term:
            return False, False
        return True, True
    
    def delete_term(variable_name, term_name, var_type, session_data=None):
        """Elimina un termine fuzzy esistente."""
        response = requests.post(f'http://127.0.0.1:5000/api/delete_term/{term_name}',
            headers={"X-Session-ID": (session_data or {}).get("sid")}
        )
        if response.status_code == 200:
            terms_list, figure, count = update_terms_list_and_figure(variable_name, var_type)
            return (
                terms_list, False, f"Term '{term_name}' successfully eliminated!", figure,
                dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update,
                count
            )
        else:
            return (
                dash.no_update, True,
                f"Error in term deletion: {response.json().get('error', 'Unknown Error')}",
                dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update
            )
    
    def modify_term(open_type, var_type, variable_name, domain_min, domain_max, function_type, term_name, param_a, param_b, param_c, param_d, param_mean, param_sigma, defuzzy_type=None, selected_term=None, session_data=None):
        """Modifica un termine fuzzy esistente e aggiorna grafico e lista.""" 
        try:
            domain_min = int(domain_min)
            domain_max = int(domain_max)
        except (ValueError, TypeError):
            return dash.no_update, True, "The Domain values must be numbers.", dash.no_update, dash.no_update
        if not term_name or not re.match(r"^[A-Za-z0-9_-]+$", term_name):
            return dash.no_update, True, "The term name is blank or contains invalidcharacters. Use only letters, numbers, hyphens, and underscores.", dash.no_update, dash.no_update
        if domain_min > domain_max:
            return dash.no_update, True, "The minimum domain cannot be greater than the maximum domain.", dash.no_update, dash.no_update
        params = {}
        if function_type == 'Triangolare':
            params = {'a': param_a, 'b': param_b, 'c': param_c}
        elif function_type == 'Triangolare-open':
            if open_type == 'left':
                params = {'a': param_a, 'b': param_a, 'c': param_c}
            elif open_type == 'right':
                params = {'a': param_a, 'b': param_c, 'c': param_c}
        elif function_type == 'Gaussian':
            params = {'mean': param_mean, 'sigma': param_sigma}
        elif function_type == 'Gaussian-open':
            params = {'mean': param_mean, 'sigma': param_sigma}
        elif function_type == 'Trapezoidale':
            params = {'a': param_a, 'b': param_b, 'c': param_c, 'd': param_d}
        elif function_type == 'Trapezoidale-open':
            if open_type == 'left':
                params = {'a': param_a, 'b': param_a, 'c': param_c, 'd': param_d}
            elif open_type == 'right':
                params = {'a': param_a, 'b': param_b, 'c': param_d, 'd': param_d}
        elif function_type == 'Classification':
            params = {}
        if function_type != 'Classification':
            is_valid, error_message = validate_params(open_type, params, domain_min, domain_max, function_type)
            if not is_valid:
                return dash.no_update, True, error_message, dash.no_update, dash.no_update
        payload = {
            'term_name': term_name,
            'variable_name': variable_name,
            'domain_min': domain_min,
            'domain_max': domain_max,
            'function_type': function_type,
            'params': params
        }
        if function_type and 'open' in function_type and open_type:
            payload['open_type'] = open_type
        if var_type == "output" and defuzzy_type:
            payload['defuzzy_type'] = defuzzy_type
        response = requests.put(f'http://127.0.0.1:5000/api/modify_term/{selected_term}',
            headers={"X-Session-ID": (session_data or {}).get("sid")},
            json=payload)
        if response.status_code == 201:
            terms_list, figure, count = update_terms_list_and_figure(variable_name, var_type)
            return terms_list, False, "Term successfully modified!", figure, count
        else:
            error_message = response.json().get('error', 'Unknown error')
            return dash.no_update, True, f"{error_message}", dash.no_update, dash.no_update
    
    def update_terms_list_and_figure(variable_name, var_type, session_data=None):
        """Recupera i termini fuzzy e costruisce il grafico corrispondente."""
        if variable_name and var_type:
            try:
                response = requests.get('http://127.0.0.1:5000/api/get_terms',
                    headers={"X-Session-ID": (session_data or {}).get("sid")})
                if response.status_code == 200:
                    terms_data = response.json()
                    terms_list = []
                    input_data = []
                    output_data = []
                    input_variables = terms_data.get('input', {})
                    output_variables = terms_data.get('output', {})
                    if var_type == 'input' and variable_name in input_variables:
                        variable_data = input_variables[variable_name]
                        for term in variable_data['terms']:
                            term_name = term.get('term_name', '')
                            x = term.get('x')
                            y = term.get('y')
                            terms_list.append(
                                dbc.ListGroupItem(
                                    term_name,
                                    id={'type': 'term-item', 'index': term_name},
                                    n_clicks=0,
                                    style={'cursor': 'pointer'}
                                )
                            )
                            if x is not None and y is not None:
                                input_data.append(go.Scatter(x=x, y=y, mode='lines', name=term_name))
                    elif var_type == 'output' and variable_name in output_variables:
                        variable_data = output_variables[variable_name]
                        for term in variable_data['terms']:
                            term_name = term.get('term_name', '')
                            x = term.get('x')
                            y = term.get('y')
                            terms_list.append(
                                dbc.ListGroupItem(
                                    term_name,
                                    id={'type': 'term-item', 'index': term_name},
                                    n_clicks=0,
                                    style={'cursor': 'pointer'}
                                )
                            )
                            if x is not None and y is not None:
                                output_data.append(go.Scatter(x=x, y=y, mode='lines', name
=term_name))
                    if var_type == 'input':
                        combined_figure = {
                            'data': input_data,
                            'layout': go.Layout(
                                title=f'Fuzzy set for {variable_name} (Input)',
                                xaxis={
                                    'title': 'Domain',
                                    'showgrid': True, 
                                    'gridwidth': 1,   
                                    'gridcolor': 'lightgray', 
                                    'dtick': 5  
                                },
                                yaxis={
                                    'title': 'Degree of membership',
                                    'showgrid': True,  
                                    'gridwidth': 1,    
                                    'gridcolor': 'lightgray', 
                                    'dtick': 0.1  
                                }
                            )
                        }
                    elif var_type == 'output':
                        combined_figure = {
                            'data': output_data,
                            'layout': go.Layout(
                                title=f'Fuzzy set for {variable_name} (Output)',
                                xaxis={
                                    'title': 'Domain',
                                    'showgrid': True,  
                                    'gridwidth': 1,    
                                    'gridcolor': 'lightgray',  
                                    'dtick': 5  
                                },
                                yaxis={
                                    'title': 'Degree of membership',
                                    'showgrid': True,  
                                    'gridwidth': 1,    
                                    'gridcolor': 'lightgray',  
                                    'dtick': 0.1  
                                }
                            )
                        }
                    else:
                        combined_figure = {
                            'data': [],
                            'layout': go.Layout(
                                title=f'No valid data for {variable_name}',
                                xaxis={'title': 'Domain'},
                                yaxis={'title': 'Degree of membership'}
                            )
                        }
                    return terms_list, combined_figure, len(variable_data['terms'])
                else:
                    return [html.Li("Error during the recovery of terms.")], dash.no_update, 0
            except Exception as e:
                return [html.Li(f"Error during data recovery: {str(e)}")], dash.no_update, 0
        else:
            return [], dash.no_update, 0
    
    #Classificazione
    @dash_app.callback(
        Output("classification-warning-modal", "is_open"),
        Input("classification-checkbox", "value"),
        State("classification-confirmed", "data"),
        State("session-store", "data")
    )
    def show_classification_modal(value, confirmed, session_data=None):
        if value and "Classification" in value:
            if not confirmed or confirmed is None:
                return True
        return False
    
    @dash_app.callback(
        Output("classification-confirmed", "data"),
        Input("url", "pathname"),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def reset_classification_confirmation(pathname, session_data=None):
        if pathname == "/choose-k":
            return False
        raise dash.exceptions.PreventUpdate
    
    @dash_app.callback(
        [
            Output("classification-checkbox", "value", allow_duplicate=True),
            Output("message", "children", allow_duplicate=True),
            Output("terms-list", "children", allow_duplicate=True),
            Output("graph", "figure", allow_duplicate=True),
            Output("classification-warning-modal", "is_open", allow_duplicate=True),
            Output("classification-confirmed", "data", allow_duplicate=True)  
        ],
        [
            Input("confirm-classification", "n_clicks"),
            Input("cancel-classification", "n_clicks")
        ],
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def handle_classification_change(confirm_click, cancel_click, session_data=
None):
        """Gestisce la conferma o l'annullamento della modalità Classification."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id == "confirm-classification":
            try:
                response = requests.post("http://127.0.0.1:5000/api/clear_output",
                    headers={"X-Session-ID": (session_data or {}).get("sid")})
                if response.status_code == 200:
                    return (
                        ["Classification"],  
                        "Output data has been cleared.",
                        [dbc.ListGroupItem("No Terms Present", style={"textAlign": "center"})],
                        {},  
                        False,  
                        True   
                    )
                else:
                    return (
                        ["Classification"],
                        f"{response.json().get('error', 'Unknown error')}",
                        dash.no_update,
                        dash.no_update,
                        False,
                        False
                    )
            except Exception as e:
                return (
                    ["Classification"],
                    f"Error connecting to the backend: {e}",
                    dash.no_update,
                    dash.no_update,
                    False,
                    False
                )
        elif triggered_id == "cancel-classification":
            return (
                [], "", dash.no_update, dash.no_update, False, False
            )
        raise dash.exceptions.PreventUpdate
    
    @dash_app.callback(
        Output("classification-counter", "style"),
        Output("classification-counter", "children"),
        Input("classification-checkbox", "value"),
        Input("classification-term-count", "data"),
        State("session-store", "data")
    )
    def update_classification_counter(classification_value, term_count, session_data=None):
        """Aggiorna e mostra il contatore dei termini in modalità Classification.""" 
        if classification_value and "Classification" in classification_value:
            return {"display": "block"}, f"Classification Class Created: {term_count}"
        return {"display": "none"}, ""
    
    @dash_app.callback(
        Output('url', 'pathname'),
        Input("classification-checkbox", "value"),
        State("classification-confirmed", "data"),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def handle_classification_redirect(checkbox_value, confirmed, session_data=
None):
        if checkbox_value and "Classification" in checkbox_value:
            if confirmed:
                return "/classification"
            else:
                raise dash.exceptions.PreventUpdate 
        return "/choose-k"
    
    @dash_app.callback(
        Output("terms-list", "children"),
        Output("classification-term-count", "data"),
        Output("variable-name", "value"),
        Input("url", "pathname"),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def load_terms_on_classification(pathname, session_data=None):
        if pathname != "/classification":
            raise dash.exceptions.PreventUpdate
        try:
            response = requests.get("http://127.0.0.1:5000/api/get_terms",
                headers={"X-Session-ID": (session_data or {}).get("sid")})
            if response.status_code != 200:
                return [dbc.ListGroupItem("Error loading terms", style={"textAlign": "center"})], 0, ""
            data = response.json()
            output_data = data.get("output", {})
            if not output_data:
                return [dbc.ListGroupItem("No Terms Present", style={"textAlign": "center"})], 0, ""
            # Prendi il nome della variabile output (ce n’è solo una in Classification)
            variable_name = next(iter(output_data))
            terms = output_data[variable_name].get("terms", [])
            count = len(terms)
            if not terms:
                return [dbc.ListGroupItem("No Terms Present", style={"textAlign": "center"})], 0, variable_name
            term_items = [
                dbc.ListGroupItem(
                    term["term_name"],
                    id={'type': 'term-item', 'index': term["term_name"]},
                    n_clicks=0,
                    style={'cursor': 'pointer'}
                )
                for term in terms
            ]
            return term_items, count, variable_name
        except Exception as e:
            return [dbc.ListGroupItem(f"Error: {e}", style={"textAlign": "center"})], 0, ""
    
    #Rules
    @dash_app.callback(
        Output("rules-container", "children"),
        Input("create-rule", "n_clicks"),
        [State({"type": "if-dropdown", "index": ALL}, "value"),  
        State({"type": "if-term-dropdown", "index": ALL}, "value"),  
        State("then-dropdown", "value"),  
        State("then-term-dropdown", "value"),  
        State("rules-container", "children")],
        State("session-store", "data"),  
        prevent_initial_call=True
    )
    def update_rules(n_clicks, all_input_vars, all_input_terms, output_var, output_term, existing_rules, session_data=None):
        """Crea una nuova regola fuzzy e la aggiunge al contenitore.""" 
        if n_clicks is None:
            return existing_rules
        if not all(all_input_vars) or not all(all_input_terms) or not output_var or not output_term:
            return existing_rules  
        if_part = " AND ".join([f"({var} IS {term})" for var, term in zip(all_input_vars, all_input_terms)])
        new_rule_text = f"IF {if_part} THEN ({output_var} IS {output_term})"
        new_rule = html.Div(
            new_rule_text,
            className="rule-item",
            style={"marginBottom": "10px", "padding": "5px", "border": "1px solid #ccc", "borderRadius": "5px"}
        )
        return existing_rules + [new_rule]
    
    @dash_app.callback(
        [Output({"type": "if-dropdown", "index": ALL}, "options"),
        Output({"type": "if-term-dropdown", "index": ALL}, "options"),
        Output("then-dropdown", "options"),
        Output("then-dropdown", "value"),
        Output("then-term-dropdown", "options")],
        [Input({"type": "if-dropdown", "index": ALL}, "value")],
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def update_dropdowns(all_input_values, session_data=None):
        """Aggiorna le opzioni delle dropdown IF/THEN in base alle variabili disponibili."""
        try:
            response = requests.get("http://127.0.0.1:5000/api/get_variables_and_terms",
                headers={"X-Session-ID": (session_data or {}).get("sid")})
            if response.status_code != 200:
                return [[]] * len(all_input_values), [[]] * len(all_input_values), [], None, []
            data = response.json()
            input_vars = list(data.get("input", {}).keys())
            output_data = data.get("output", {})
            output_var_name = next(iter(output_data), None)
            input_options_list = []
            for i, selected in enumerate(all_input_values):
                used = [v for j, v in enumerate(all_input_values) if j != i and v]
                available = [v for v in input_vars if v not in used]
                input_options_list.append([{"label": v, "value": v} for v in available])
            if_term_options = []
            for selected in all_input_values:
                if selected and selected in data["input"]:
                    terms = data["input"][selected]
                    if_term_options.append([{"label": t["label"], "value": t["value"]} for t in terms])
                else:
                    if_term_options.append([])
            then_dropdown_options = [{"label": output_var_name, "value": output_var_name}] if output_var_name else []
            then_dropdown_value = output_var_name
            then_terms = []
            if output_var_name and output_var_name in output_data:
                then_terms = [{"label": t["label"], "value": t["value"]} for t in output_data[output_var_name]]
            return input_options_list, if_term_options, then_dropdown_options, then_dropdown_value, then_terms
        except Exception as e:
            print(f"Error in update_dropdowns: {e}")
            return [[]] * len(all_input_values), [[]] * len(all_input_values), [], None, []
    
    @dash_app.callback(
        [
                Output('rules-list', 'children', allow_duplicate=True),
                Output('rules-store', 'data', allow_duplicate=True),
                Output("error-modal", "is_open", allow_duplicate=True),
                Output("error-modal-body", "children", allow_duplicate=True)
        ],
        Input('create-rule', 'n_clicks'),
        [
            State({'type': 'if-dropdown', 'index': ALL}, 'value'),
            State({'type': 'if-term-dropdown', 'index': ALL}, 'value'),
            State('then-dropdown', 'value'),
            State('then-term-dropdown', 'value'),
            State('rules-list', 'children'),
            State('rules-store', 'data')
        ],
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def create_rule(n_clicks, input_vars, input_terms, output_variable, output_term, current_rules, rules_data, session_data=None):
        """Crea una regola fuzzy e la salva nel backend.""" 
        if n_clicks is None:
            return current_rules, rules_data, False, ''
        if not all(input_vars) or not all(input_terms) or not output_variable or not output_term:
            missing = []
            if not all(v for v in input_vars): missing.append("input variable")
            if not all(t for t in input_terms): missing.append("input term")
            if not output_variable: missing.append("output variable")
            if not output_term: missing.append("output term")
            return current_rules, rules_data, True, f'Please fill in: {", ".join(missing)}'
        inputs = [{"input_variable": var, "input_term": term} for var, term in zip(input_vars, input_terms)]
        rule_text = " AND ".join([f"({i['input_variable']} IS {i['input_term']})" for i in inputs])
        rule_text = f"IF {rule_text} THEN ({output_variable} IS {output_term})"
        existing_rules_texts = [rule['props']['children'] if isinstance(rule, dict) else rule.children for rule in current_rules]
        if rule_text in existing_rules_texts:
            return current_rules, rules_data, True, 'Error this rule already exists!'
        response = requests.post(
            "http://127.0.0.1:5000/api/create_rule",
            headers={"X-Session-ID": (session_data or {}).get("sid")},
            json={
                "inputs": inputs,
                "output_variable": output_variable,
                "output_term": output_term
            }
        )
        if response.status_code == 201:
            rule_id = response.json().get("rule_id")
            rules_data.append({
                "id": rule_id,
                "inputs": inputs,
                "output_variable": output_variable,
                "output_term": output_term
            })
            new_rule = dbc.ListGroupItem(
                rule_text,
                id={'type': 'rule-item', 'index': rule_id},
                n_clicks=0,
                style={"cursor": "pointer"}
            )
            return current_rules + [new_rule], rules_data, False, ''
        return current_rules, rules_data, True, 'Error while saving the rule'
    
    @dash_app.callback(
        [Output("rules-list", "children", allow_duplicate=True),
        Output("delete-rule", "disabled"),
        Output("selected-rule-id", "data")],
        Input({"type": "rule-item", "index": ALL}, "n_clicks"),
        State({"type": "rule-item", "index": ALL}, "id"),
        State("rules-store", "data"),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def select_rule(n_clicks, all_ids, rules_data, session_data=None):
        """Evidenzia la regola selezionata e salva il relativo ID.""" 
        selected_id = None
        styles = []
        if not n_clicks:
            return dash.no_update, True, None
        for idx, click in enumerate(n_clicks):
            if click and click > 0:
                selected_id = all_ids[idx]['index']
                break
        rule_items = []
        for rule in rules_data:
            inputs = rule.get("inputs", [])
            inputs_text = " AND ".join(
                f"({inp['input_variable']} IS {inp['input_term']})" for inp in inputs
            )
            output_text = f"({rule['output_variable']} IS {rule['output_term']})"
            rule_text = f"IF {inputs_text} THEN {output_text}"
            style = {"cursor": "pointer"}
            if rule["id"] == selected_id:
                style["backgroundColor"] = "#d1ecf1"
            rule_items.append(
                dbc.ListGroupItem(
                    rule_text,
                    id={'type': 'rule-item', 'index': rule['id']},
                    n_clicks=0,
                    style=style
                )
            )
        return rule_items, selected_id is None, selected_id
    
    @dash_app.callback(
        Output("rules-store", "data", allow_duplicate=True),
        Input("delete-rule", "n_clicks"),
        State("selected-rule-id", "data"),
        State("rules-store", "data"),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def delete_selected_rule(n_clicks, selected_rule_id, rules_data, session_data
=None):
        """Elimina la regola selezionata e aggiorna la lista."""
        if not selected_rule_id:
            raise dash.exceptions.PreventUpdate
        response = requests.delete(f"http://127.0.0.1:5000/api/delete_rule/{selected_rule_id}",
            headers={"X-Session-ID": (session_data or {}).get("sid")})
        if response.status_code != 200:
            return dash.no_update
        updated_rules = [r for r in rules_data if r["id"] != selected_rule_id]
        return updated_rules
    
    @dash_app.callback(
        [Output("rules-store", "data"),
        Output("input-variables", "data")],
        Input("url_rules", "pathname"),
        State("session-store", "data")
    )
    def load_rules_on_page_load(pathname, session_data=None):
        try:
            sid = (session_data or {}).get("sid")
            response_rules = requests.get("http://127.0.0.1:5000/api/get_rules",
                headers={"X-Session-ID": sid})
            rules = response_rules.json() if response_rules.status_code == 200 else []
            response_vars = requests.get("http://127.0.0.1:5000/api/get_variables_and_terms",
                headers={"X-Session-ID": sid})
            input_vars = []
            if response_vars.status_code == 200:
                data = response_vars.json()
                input_vars = list(data.get("input", {}).keys())
            return rules, input_vars
        except Exception as e:
            print(f"Error loading rules or variables: {e}")
            return [], []
    
    @dash_app.callback(
        Output("rules-list", "children"),
        Input("rules-store", "data"),
        State("session-store", "data")
    )
    def display_existing_rules(rules_data, session_data=None):
        """Mostra tutte le regole presenti nella lista.""" 
        rules_display = []
        for rule in rules_data:
            inputs = rule.get("inputs", [])
            inputs_text = " AND ".join(
                f"({inp['input_variable']} IS {inp['input_term']})" for inp in inputs
            )
            output_text = f"({rule['output_variable']} IS {rule['output_term']})"
            rule_text = f"IF {inputs_text} THEN {output_text}"
            rules_display.append(
                dbc.ListGroupItem(
                    rule_text,
                    id={'type': 'rule-item', 'index': rule['id']},
                    n_clicks=0,
                    style={"cursor": "pointer"}
                )
            )
        return rules_display
    
    @dash_app.callback(
        [Output('input-container', 'children', allow_duplicate=True),
        Output('input-count', 'data', allow_duplicate=True)],
        Input('input-variables', 'data'),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def init_input_blocks(input_variables, session_data=None):
        """Inizializza il primo blocco IF-Term per la creazione di una regola.""" 
        if not input_variables:
            return [], 0
        first_input = html.Div([
            dbc.Label("IF", className="w-100 text-center mb-0"),
            dcc.Dropdown(id={"type": "if-dropdown", "index": 0}, placeholder="Select Input Variable", style={"width": "200px"}),
            dbc.Label("Term", className="w-100 text-center mb-0"),
            dcc.Dropdown(id={"type": "if-term-dropdown", "index": 0}, placeholder="Select Term", style={"width": "200px"}),
        ], className="d-flex flex-column align-items-center border rounded p-2", style={"minWidth": "220px"})
        return [first_input], 1
    
    @dash_app.callback(
        [Output('input-container', 'children'),
        Output('input-count', 'data')],
        Input('add-input', 'n_clicks'),
        State('input-container', 'children'),
        State('input-variables', 'data'),
        State('input-count', 'data'),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def manage_inputs(add_clicks, current_inputs, input_variables, input_count, session_data=None):
        """Aggiunge dinamicamente nuovi blocchi IF-Term per la creazione delle regole.""" 
        if not input_variables or input_count >= len(input_variables):
            return current_inputs, input_count
        new_label = "IF" if input_count == 0 else "AND"
        new_input = html.Div([
            dbc.Label(new_label, className="w-100 text-center mb-0"),
            dcc.Dropdown(id={"type": "if-dropdown", "index": input_count}, placeholder="Select Input Variable", style={"width": "200px"}),
            dbc.Label("TERM", className="w-100 text-center mb-0"),
            dcc.Dropdown(id={"type": "if-term-dropdown", "index": input_count}, placeholder="Select Term", style={"width": "200px"}),
        ], className="d-flex flex-column align-items-center border rounded p-2", style={"minWidth": "220px"})
        current_inputs.append(new_input)
        return current_inputs, input_count + 1
    
    #Regole
    @dash_app.callback(
        Output("inference-data", "data"),
        Output("rules-list-membership", "children"),
        Output("membership-values", "children"),
        Output("winner-term-store", "data"),
        Input("start-inference", "n_clicks"),
        State({"type": "inference-input", "variable": ALL}, "value"),
        State("inference-input-ids", "data"),
        State("is-classification", "data"),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def run_inference(n_clicks, input_values, input_ids, is_classification, session_data=None):
        """Esegue l'inferenza fuzzy sui valori inseriti e mostra attivazioni e output."""
        no_output = dash.no_update, [], [], {}
        try:
            inputs_dict = {}
            for var_name, val in zip(input_ids or [], input_values or []):
                if val is not None:
                    try:
                        inputs_dict[var_name] = float(val)
                    except (ValueError, TypeError):
                        continue
            if not inputs_dict:
                msg = html.Div(
                    "Inserire un valore per le variabili di input prima di eseguire l'inferenza.",
                    className="text-warning text-center"
                )
                return dash.no_update, [msg], [], {}
            
            response = requests.post("http://127.0.0.1:5000/api/infer",
                headers={"X-Session-ID": (session_data or {}).get("sid")},
                json={"inputs": inputs_dict})
            if response.status_code != 200:
                err = html.Div("Errore nell'inferenza lato server.", className="text-danger text-center")
                return dash.no_update, [err], [], {}
            
            result = response.json()
            rule_outputs = result.get("rule_outputs", [])
            outputs = result.get("results", {})
            if not rule_outputs:
                msg = html.Div(
                    "Nessuna regola attivata. Il valore inserito potrebbe essere fuori daldominio definito.",
                    className="text-warning text-center"
                )
                return dash.no_update, [msg], [], {}
            
            # --- Sezione regole attivate --
            rules_display = []
            for rule in rule_outputs:
                if rule.get("inputs"):
                    inputs_text = " AND ".join(
                        f"({inp['input_variable']} IS {inp['input_term']})"
                        for inp in rule["inputs"]
                    )
                else:
                    inputs_text = "?"
                output_text = f"({rule['output_variable']} IS {rule['output_term']})"
                activation = rule['activation']
                rule_text = f"IF {inputs_text} THEN {output_text} →{round(activation, 3)}"
                color = "#28a745" if activation > 0 else "#6c757d"
                rules_display.append(
                    html.P(rule_text, className="rule-inference text-center",
                           style={"fontSize": "0.9em", "color": color, "fontWeight": "600" if activation > 0 else "normal"})
                )
            
            # --- Classificazione: aggiorna winner-term-store --
            winner_term_store = {}
            if is_classification:
                for var_name, value in outputs.items():
                    winner_term_store[var_name] = value
            
            return result, rules_display, [], winner_term_store
        except Exception as e:
            print(f"Inference error: {e}")
            err = html.Div(f"Errore durante l'inferenza: {e}", className="text-danger text-center")
            return dash.no_update, [err], [], {}
    
    @dash_app.callback(
        Output({"type": "classification-output", "variable": ALL}, "children"),
        Input("winner-term-store", "data"),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def update_classification_results(winner_term_store, session_data=None):
        outputs = []
        ctx = callback_context
        for output in ctx.outputs_list:
            var_name = output["id"]["variable"]
            winner_class = winner_term_store.get(var_name, "N/A")
            outputs.append(winner_class)
        return outputs
    
    @dash_app.callback(
        Output({"type": "output", "variable": ALL}, "children"),
        Input("inference-data", "data"),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def update_numeric_outputs(inference_data, session_data=None):
        if not inference_data:
            raise dash.exceptions.PreventUpdate
        outputs = inference_data.get("results", {})
        result = []
        for var_id in ctx.outputs_list:
            try:
                variable_name = var_id["id"]["variable"]
                value = outputs.get(variable_name, 0)
                result.append(f"{value:.2f}")
            except Exception as e:
                result.append("0.00")
        return result
    
    #Plot Inferenza
    @dash_app.callback(
        Output("inference-plot-modal", "is_open"),
        Output("inference-plot", "figure"),
        Input("visualize-plot", "n_clicks"),
        Input("close-inference-plot", "n_clicks"),
        State("inference-plot-modal", "is_open"),
        State({"type": "inference-input", "variable": ALL}, "value"),
        State("inference-input-ids", "data"),
        State("session-store", "data"),
        prevent_initial_call=True
    )
    def toggle_inference_modal(open_click, close_click, is_open, input_values, input_ids, session_data=None):
        if ctx.triggered_id == "close-inference-plot":
            return False, ctx.no_update
        
        inputs_dict = {}
        for var_name, val in zip(input_ids or [], input_values or []):
            if val is not None:
                try:
                    inputs_dict[var_name] = float(val)
                except (ValueError, TypeError):
                    continue
        if not inputs_dict:
            fig = go.Figure()
            fig.add_annotation(
                text="No input provided.",
                xref="paper", yref="paper", showarrow=False,
                font=dict(size=18, color="red")
            )
            return True, fig
        
        response = requests.post("http://127.0.0.1:5000/api/infer",
            headers={"X-Session-ID": (session_data or {}).get("sid")},
            json={"inputs": inputs_dict})
        if response.status_code != 200:
            fig = go.Figure()
            fig.add_annotation(
                text="Error during inference.",
                xref="paper", yref="paper", showarrow=False,
                font=dict(size=16, color="red")
            )
            return True, fig
        
        data = response.json()
        rule_outputs = data.get("rule_outputs", [])
        if not rule_outputs:
            fig = go.Figure()
            fig.add_annotation(
                text="No inference results available.",
                xref="paper", yref="paper", showarrow=False,
                font=dict(size=16)
            )
            return True, fig
        
        # Organizza per output_variable
        from collections import defaultdict
        grouped = defaultdict(list)
        for item in rule_outputs:
            key = item["output_variable"]
            grouped[key].append(item)
        
        fig = go.Figure()
        for output_var, terms in grouped.items():
            x = [t["output_term"] for t in terms]
            y = [t["activation"] for t in terms]
            fig.add_trace(go.Bar(x=x, y=y, name=output_var))
        fig.update_layout(
            title="Activation of Output Terms",
            xaxis_title="Output Terms",
            yaxis_title="Activation Level",
            yaxis=dict(range=[0, 1]),
            barmode='group',
            template="plotly_white"
        )
        return True, fig
    
    @dash_app.callback(
        Output("test-page-content", "children"),
        Output("is-classification", "data"),
        Output("inference-input-ids", "data"),
        Input("url", "pathname"),
        State("session-store", "data"),
        prevent_initial_call=False
    )
    def render_test_content(pathname, session_data):
        if pathname != "/test":
            raise PreventUpdate
        data = fetch_data(session_data)
        if not data or "terms" not in data:
            return html.Div(
                "Nessun dato disponibile. Aggiungere variabili input e output prima di testare.",
                className="text-danger"
            ), False, []
        terms = data["terms"]
        input_controls = []
        for var_name, var_data in terms.get("input", {}).items():
            domain = var_data.get("domain")
            if not domain:
                continue
            domain_min, domain_max = domain
            input_controls.append(
                dbc.Col([
                    dbc.Label(
                        f"{var_name} ({domain_min}-{domain_max})",
                        html_for=f"{var_name}-input",
                        className="mb-2 text-center",
                        style={"width": "100%"}
                    ),
                    dbc.Input(
                        id={"type": "inference-input", "variable": var_name},
                        type="number",
                        min=domain_min,
                        max=domain_max,
                        step=0.1 if (domain_max - domain_min) < 10 else 1,
                        className="input-field",
                        placeholder=f"{domain_min} - {domain_max}",
                        style={"width": "100%"}
                    )
                ], md=4, className="pe-2")
            )
        output_controls = []
        for idx, (var_name, var_data) in enumerate(terms.get("output", {}).items()):
            terms_list = var_data.get("terms", [])
            is_classification = terms_list and terms_list[0].get("function_type") == "Classification"
            output_controls.append(
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            f"{'Classification Result' if is_classification else 'Result'}: {var_name}",
                            className=f"{'bg-info text-white' if is_classification else 'bg-primary text-white'} fw-medium py-2 text-center"
                        ),
                        dbc.CardBody([
                            html.H2(
                                "Classe" if is_classification else "0",
                                id={"type": "classification-output", "variable": var_name} if is_classification else {"type": "output", "variable": var_name},
                                className=f"card-text text-center {'text-info' if is_classification else 'text-primary'} mb-0",
                                style={"fontSize": "2.5rem"}
                            ),
                            html.H4(
                                id=f"winner-term-{var_name}",
                                className="text-center text-dark mt-3"
                            ) if is_classification else None
                        ])
                    ], className="variable-card h-100 mx-auto"),
                    md=6,
                    className="pe-2" if idx % 2 == 0 else "ps-2"
                )
            )
        is_classification_global = any(
            terms_list and terms_list[0].get("function_type") == "Classification"
            for terms_list in (v.get("terms", []) for v in terms.get("output", {}).values())
        )
        inference_input_ids = list(terms.get("input", {}).keys())
        # Controlla se esistono regole
        rules = data.get("rules", [])
        no_rules_warning = None
        if not rules:
            no_rules_warning = dbc.Alert(
                "Nessuna regola definita. Aggiungere regole fuzzy prima di eseguire il test.",
                color="warning",
                className="mb-3"
            )
        form_children = []
        if no_rules_warning:
            form_children.append(no_rules_warning)
        form_children += [
            html.Div(
                id="inference-inputs",
                children=[dbc.Row(input_controls, className="mb-4 g-3")]
            ),
            html.Div(
                dbc.Button([
                    html.I(className="fas fa-calculator mr-2"), " Calculate Inference"
                ],
                    id="start-inference",
                    color="primary",
                    className="action-btn",
                    style={"width": "290px"}),
                className="d-flex justify-content-center mb-4"
            ),
            html.Div(
                dbc.Button([
                    html.I(className="fas fa-chart-line mr-2"), " Visualize Plot"
                ],
                    id="visualize-plot",
                    color="success",
                    className="action-btn",
                    style={"width": "250px"}),
                className="d-flex justify-content-center mb-4"
            ),
            dbc.Modal(
                id="inference-plot-modal",
                is_open=False,
                size="xl",
                centered=True,
                children=[
                    dbc.ModalHeader("Inference Result"),
                    dbc.ModalBody(dcc.Graph(id="inference-plot")),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close-inference-plot", color="secondary")
                    )
                ]
            ),
            html.Div(
                id="rule-membership-section",
                children=[
                    html.H5(
                        "Activation of Rules",
                        className="mb-3 text-center",
                        style={"color": "#2c3e50"}
                    ),
                    dbc.Row([
                        dbc.Col(
                            html.Div(id="rules-list-membership", className="rule-membership-container"),
                            md=8,
                            className="mx-auto"
                        )
                    ], className="justify-content-center mb-4", style={"minHeight": "150px"}),
                    dbc.Row([
                        dbc.Col(
                            html.Div(id="membership-values", className="membership-values-container"),
                            md=8,
                            className="mx-auto"
                        )
                    ])
                ],
                style={
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "10px",
                    "padding": "1.5rem",
                    "marginBottom": "2rem"
                }
            ),
            dbc.Row(output_controls, className="g-4 justify-content-center mb-3")
        ]
        content = html.Div(form_children, style={"padding": "1rem 2rem"})
        return content, is_classification_global, inference_input_ids
    
    @dash_app.callback(
        Output("report-content", "children"),
        Input("url", "pathname"),
        State("session-store", "data"),
        prevent_initial_call=False
    )
    def render_report_content(pathname, session_data):
        if pathname != "/results":
            raise PreventUpdate
        data = fetch_data(session_data)
        if not data:
            return html.Div(
                "Nessun dato disponibile. Aggiungere variabili input e output.",
                className="text-danger"
            )
        terms = data.get("terms", {})
        rules = data.get("rules", [])
        input_children = generate_variable_section(terms.get("input", {}), "input")
        output_children = generate_variable_section(terms.get("output", {}), "output")
        rules_children = generate_rules_section(rules)
        return [
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Input Variables", className="gradient-header"),
                        dbc.CardBody(input_children)
                    ], className="shadow-sm")
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Output Variables", className="gradient-header"),
                        dbc.CardBody(output_children)
                    ], className="shadow-sm")
                ], md=6),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Fuzzy Rules", className="gradient-header"),
                        dbc.CardBody([
                            html.Ul(rules_children, className="list-unstyled")
                        ])
                    ], className="shadow-sm")
                ]),
            ], className="mb-4"),
        ]
    
    @dash_app.callback(
    Output("consensus-matrix-plot", "figure"),
    Output("consensus-note", "children"),
    Input("consensus-method-selector", "value"),
    Input("consensus-init-selector", "value"),
    Input("consensus-fcm-mode-selector", "value"),
    Input("consensus-k-selector", "value"),
    Input("k-experiment-status", "data"),
    State("dataset-store", "data"),
    )
    def update_consensus_matrix(method, init_method, fcm_mode, selected_k, k_status, dataset_data):

        method_labels = {
            "argmax": "Argmax",
            "kmeans": "K-Means",
            "fcm": "Fuzzy C-Means"
        }

        fcm_mode_labels = {
            "hard": "Hard",
            "soft_dot": "Soft Dot",
            "soft_cosine": "Soft Cosine"
        }

        init_labels = {
            "random": "Random",
            "nndsvd": "NNDSVD",
            "custom1": "Custom 1",
            "custom2": "Custom 2"
        }

        def empty_fig(message):
            fig_empty = go.Figure()

            fig_empty.update_layout(
                title="Consensus Matrix",
                template="plotly_white",
                autosize=False,
                width=900,
                height=750,
                uirevision="consensus-fixed-size",
                margin=dict(
                    l=80,
                    r=20,
                    t=90,
                    b=120
                ),
                xaxis={"visible": False},
                yaxis={"visible": False}
            )

            fig_empty.add_annotation(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={
                    "size": 15,
                    "color": "#6c757d"
                }
            )

            return fig_empty

        if not k_status or "artifacts" not in k_status:
            return empty_fig(
                "Run the k-selection experiment to generate consensus matrices."
            ), dbc.Alert(
                "Run the k-selection experiment first to generate real consensus matrices.",
                color="warning",
                className="mt-3"
            )

        if selected_k is None:
            return empty_fig(
                "Please select a valid k value."
            ), dbc.Alert(
                "Please select a valid k value.",
                color="warning",
                className="mt-3"
            )

        artifacts = k_status["artifacts"]

        if not init_method or init_method not in artifacts:
            return empty_fig(
                "No consensus matrix is available for the selected initialization."
            ), dbc.Alert(
                f"No consensus artifacts found for initialization {init_method}.",
                color="warning",
                className="mt-3"
            )

        init_artifacts = artifacts[init_method]
        k_key = str(selected_k)

        if k_key not in init_artifacts:
            return empty_fig(
                f"No consensus matrix is available for k={selected_k}."
            ), dbc.Alert(
                f"No consensus matrix found for k={selected_k}.",
                color="warning",
                className="mt-3"
            )

        consensus_block = init_artifacts[k_key].get("consensus", {})

        if method == "argmax":
            matrix = consensus_block.get("argmax")
            method_label = method_labels["argmax"]
            mode_label = None

        elif method == "kmeans":
            matrix = consensus_block.get("kmeans")
            method_label = method_labels["kmeans"]
            mode_label = None

        elif method == "fcm":
            matrix = consensus_block.get("fcm", {}).get(fcm_mode)
            method_label = method_labels["fcm"]
            mode_label = fcm_mode_labels.get(fcm_mode, fcm_mode)

        else:
            matrix = None
            method_label = method
            mode_label = None

        if matrix is None:
            return empty_fig(
                f"No consensus matrix is available for {method_label}."
            ), dbc.Alert(
                f"No real consensus matrix available for {method_label}.",
                color="warning",
                className="mt-3"
            )

        matrix = np.asarray(matrix, dtype=float)

        _, sample_labels = get_dataset_labels(dataset_data)

        if not sample_labels or len(sample_labels) != matrix.shape[0]:
            sample_labels = [f"Sample {i + 1}" for i in range(matrix.shape[0])]

        if method == "fcm":
            title = f"Consensus Matrix - {method_label} ({mode_label}), k={selected_k}"
        else:
            title = f"Consensus Matrix - {method_label}, k={selected_k}"

        n_samples = len(sample_labels)
        tick_step = max(1, n_samples // 12)

        tick_indices = list(range(0, n_samples, tick_step))

        if (n_samples - 1) not in tick_indices:
            tick_indices.append(n_samples - 1)

        tick_values = [sample_labels[i] for i in tick_indices]
        tick_text = [sample_labels[i] for i in tick_indices]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=sample_labels,
                y=sample_labels,
                colorscale="RdBu_r",
                zmin=0,
                zmax=1,
                zmid=0.5,
                colorbar=dict(
                    title="Consensus Strength",
                    thickness=14,
                    len=0.75,
                    x=1.03,
                    tickvals=[0, 0.5, 1],
                    ticktext=[
                        "Low (0)",
                        "Medium (0.5)",
                        "High (1)"
                    ]
                ),
                hovertemplate=(
                    "Sample X: %{x}<br>"
                    "Sample Y: %{y}<br>"
                    "Consensus: %{z:.3f}<extra></extra>"
                )
            )
        )

        fig.update_layout(
            title={
                "text": title,
                "x": 0.02,
                "xanchor": "left"
            },
            template="plotly_white",
            autosize=False,
            width=900,
            height=750,
            uirevision="consensus-fixed-size",
            margin=dict(
                l=80,
                r=20,
                t=90,
                b=120
            ),
            xaxis_title="Samples",
            yaxis_title="Samples",
            font=dict(
                family="Poppins, Arial",
                size=13,
                color="#2c3e50"
            )
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_text,
            tickangle=35,
            showgrid=False,
            tickfont=dict(size=10),
            automargin=True
        )

        fig.update_yaxes(
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_text,
            autorange="reversed",
            showgrid=False,
            tickfont=dict(size=10),
            automargin=True
        )

        readable_init = init_labels.get(init_method, init_method)

        if method == "fcm":
            note_text = (
                f"This consensus matrix represents pairwise sample co-clustering stability "
                f"across repeated NMF runs using {method_label} ({mode_label}), "
                f"initialization {readable_init}, and k={selected_k}. "
                f"Values range from 0 to 1: values close to 1 indicate that two samples "
                f"are consistently assigned to the same cluster across repeated runs, "
                f"whereas values close to 0 indicate that they are rarely assigned to "
                f"the same cluster."
            )
        else:
            note_text = (
                f"This consensus matrix represents pairwise sample co-clustering stability "
                f"across repeated NMF runs using {method_label}, initialization {readable_init}, "
                f"and k={selected_k}. Values range from 0 to 1: values close to 1 indicate "
                f"that two samples are consistently assigned to the same cluster across "
                f"repeated runs, whereas values close to 0 indicate that they are rarely "
                f"assigned to the same cluster."
            )

        note = dbc.Card(
            dbc.CardBody([
                html.H6(
                    "Consensus Matrix Interpretation",
                    className="mb-2",
                    style={
                        "fontWeight": "700",
                        "color": "#2c3e50"
                    }
                ),

                html.P(
                    note_text,
                    className="mb-3",
                    style={
                        "fontSize": "14px",
                        "lineHeight": "1.6"
                    }
                ),

                dbc.Row([
                    dbc.Col(
                        dbc.Badge(
                            "Blue = low consensus",
                            color="primary",
                            className="p-2 w-100"
                        ),
                        md=4
                    ),
                    dbc.Col(
                        dbc.Badge(
                            "White = medium consensus",
                            color="light",
                            text_color="dark",
                            className="p-2 w-100"
                        ),
                        md=4
                    ),
                    dbc.Col(
                        dbc.Badge(
                            "Red = high consensus",
                            color="danger",
                            className="p-2 w-100"
                        ),
                        md=4
                    ),
                ], className="g-2")
            ]),
            className="mt-3 shadow-sm border-0",
            style={
                "backgroundColor": "#f8fbfd",
                "borderLeft": "5px solid #52b2cf",
                "borderRadius": "10px"
            }
        )

        return fig, note

    
    @dash_app.callback(
        Output("download-k-graph", "data"),
        Input("download-k-graph-btn", "n_clicks"),
        State("k-selection-graph", "figure"),
        prevent_initial_call=True
    )
    def download_k_graph(n_clicks, figure):
        if not n_clicks or not figure:
            raise PreventUpdate
        fig = go.Figure(figure)
        image_bytes = fig.to_image(
            format="png",
            width=1200,
            height=700,
            scale=2
        )
        return dcc.send_bytes(
            lambda buffer: buffer.write(image_bytes),
            "k_selection_graph.png"
        )


    @dash_app.callback(
    Output("download-k-metrics", "data"),
    Input("download-k-metrics-btn", "n_clicks"),
    State("k-experiment-status", "data"),
    prevent_initial_call=True
    )
    def download_k_metrics(n_clicks, k_status):

        if not n_clicks:
            raise PreventUpdate

        if not k_status or "displayed_metrics" not in k_status:
            raise PreventUpdate

        df = pd.DataFrame(k_status.get("displayed_metrics", []))

        if df.empty:
            raise PreventUpdate

        df.columns = [str(col).replace("_", " ").title() for col in df.columns]

        return send_excel_file(
            df,
            filename="nmf_k_selection_metrics.xlsx",
            sheet_name="K Selection Metrics",
            index=False
        )
    
    @dash_app.callback(
        Output("download-consensus-matrix", "data"),
        Input("download-consensus-btn", "n_clicks"),
        State("k-experiment-status", "data"),
        State("consensus-method-selector", "value"),
        State("consensus-init-selector", "value"),
        State("consensus-fcm-mode-selector", "value"),
        State("consensus-k-selector", "value"),
        State("dataset-store", "data"),
        prevent_initial_call=True
    )
    def download_consensus_matrix(
        n_clicks,
        k_status,
        method,
        init_method,
        fcm_mode,
        selected_k,
        dataset_data
    ):
        if not n_clicks:
            raise PreventUpdate
        if not k_status or "artifacts" not in k_status:
            raise PreventUpdate
        artifacts = k_status["artifacts"]
        if not init_method or init_method not in artifacts:
            raise PreventUpdate
        init_artifacts = artifacts[init_method]
        k_key = str(selected_k)
        if k_key not in init_artifacts:
            raise PreventUpdate
        consensus_block = init_artifacts[k_key].get("consensus", {})
        if method == "argmax":
            matrix = consensus_block.get("argmax")
            filename = f"consensus_argmax_k{selected_k}.csv"
        elif method == "kmeans":
            matrix = consensus_block.get("kmeans")
            filename = f"consensus_kmeans_k{selected_k}.csv"
        elif method == "fcm":
            matrix = consensus_block.get("fcm", {}).get(fcm_mode)
            filename = f"consensus_fcm_{fcm_mode}_k{selected_k}.csv"
        else:
            matrix = None
            filename = "consensus_matrix.csv"
        if matrix is None:
            raise PreventUpdate
        _, sample_labels = get_dataset_labels(dataset_data)
        if not sample_labels or len(sample_labels) != len(matrix):
            sample_labels = [f"Sample {i+1}" for i in range(len(matrix))]
        df = pd.DataFrame(
            matrix,
            index=sample_labels,
            columns=sample_labels
        )
        df.index.name = "Sample"
        return send_excel_file(
            df,
            filename.replace(".csv", ".xlsx"),
            sheet_name="Consensus Matrix",
            index=True
        )
    
    @dash_app.callback(
        Output("download-w-matrix", "data"),
        Input("download-w-btn", "n_clicks"),
        State("nmf-results-store", "data"),
        State("dataset-store", "data"),
        prevent_initial_call=True
    )
    def download_w_matrix(n_clicks, nmf_results, dataset_data):
        if not n_clicks:
            raise PreventUpdate
        if not nmf_results:
            raise PreventUpdate
        W = nmf_results.get("W_norm") or nmf_results.get("W")
        if W is None:
            raise PreventUpdate
        W = np.asarray(W, dtype=float)
        feature_labels, _ = get_dataset_labels(dataset_data)
        if not feature_labels or len(feature_labels) != W.shape[0]:
            feature_labels = [f"Feature {i+1}" for i in range(W.shape[0])]
        latent_factor_labels = [f"LF{i+1}" for i in range(W.shape[1])]
        df_w = pd.DataFrame(
            W,
            index=feature_labels,
            columns=latent_factor_labels
        )
        df_w.index.name = "Feature"
        return send_excel_file(
            df_w,
            "matrix_W.xlsx",
            sheet_name="Matrix W",
            index=True
        )
    

    @dash_app.callback(
        Output("download-h-matrix", "data"),
        Input("download-h-btn", "n_clicks"),
        State("nmf-results-store", "data"),
        State("dataset-store", "data"),
        prevent_initial_call=True
    )
    def download_h_matrix(n_clicks, nmf_results, dataset_data):
        if not n_clicks:
            raise PreventUpdate
        if not nmf_results:
            raise PreventUpdate
        H = nmf_results.get("H_norm") or nmf_results.get("H")
        if H is None:
            raise PreventUpdate
        H = np.asarray(H, dtype=float)
        _, sample_labels = get_dataset_labels(dataset_data)
        if not sample_labels or len(sample_labels) != H.shape[1]:
            sample_labels = [f"Sample {i+1}" for i in range(H.shape[1])]
        latent_factor_labels = [f"LF{i+1}" for i in range(H.shape[0])]
        df_h = pd.DataFrame(
            H,
            index=latent_factor_labels,
            columns=sample_labels
        )
        df_h.index.name = "Latent Factor"
        return send_excel_file(
            df_h,
            "matrix_H.xlsx",
            sheet_name="Matrix H",
            index=True
        )
    

    @dash_app.callback(
        Output("download-clusters", "data"),
        Input("download-clusters-btn", "n_clicks"),
        State("nmf-results-store", "data"),
        State("dataset-store", "data"),
        prevent_initial_call=True
    )
    def download_clusters(n_clicks, nmf_results, dataset_data):
        if not n_clicks:
            raise PreventUpdate
        if not nmf_results:
            raise PreventUpdate
        final_clustering = nmf_results.get("final_clustering", "kmeans")
        clusters = nmf_results.get("clusters", {}).get(final_clustering)
        if clusters is None:
            raise PreventUpdate
        _, sample_labels = get_dataset_labels(dataset_data)
        if not sample_labels or len(sample_labels) != len(clusters):
            sample_labels = [f"Sample {i+1}" for i in range(len(clusters))]
        clustering_labels = {
            "argmax": "Argmax",
            "kmeans": "K-Means",
            "fcm_hard": "Fuzzy C-Means"
        }
        df_clusters = pd.DataFrame({
            "Sample": sample_labels,
            "Cluster": [int(c) + 1 for c in clusters],
            "Clustering Algorithm": clustering_labels.get(
                final_clustering,
                final_clustering
            )
        })
        return send_excel_file(
            df_clusters,
            "cluster_assignments.xlsx",
            sheet_name="Cluster Assignments",
            index=False
        )


    @dash_app.callback(
    Output("download-centroids-representatives", "data"),
    Input("download-centroids-representatives-btn", "n_clicks"),
    State("nmf-results-store", "data"),
    prevent_initial_call=True
    )
    def download_centroids_representatives(n_clicks, nmf_results):

        if not n_clicks:
            raise PreventUpdate

        if not nmf_results or not nmf_results.get("nmf_completed"):
            raise PreventUpdate

        centroids = nmf_results.get("centroids", {})
        representatives = nmf_results.get("representatives", {})

        if not centroids and not representatives:
            raise PreventUpdate

        method_labels = {
            "argmax": "Argmax Mean",
            "kmeans": "K-Means Mean",
            "fcm_hard": "Fuzzy C-Means Mean",
            "fcm": "Fuzzy C-Means",
        }

        centroid_labels = {
            "kmeans": "K-Means Centroid",
            "fcm": "Fuzzy C-Means Centroid",
        }

        rows = []

        def add_matrix_rows(matrix, method_name, representation_type):
            if matrix is None or len(matrix) == 0:
                return

            matrix = np.asarray(matrix, dtype=float)

            for cluster_idx in range(matrix.shape[0]):
                row = {
                    "Representation Type": representation_type,
                    "Method": method_name,
                    "Cluster": f"Cluster {cluster_idx + 1}"
                }

                for lf_idx in range(matrix.shape[1]):
                    row[f"LF{lf_idx + 1}"] = round(float(matrix[cluster_idx, lf_idx]), 3)

                rows.append(row)

        for method_name, matrix in representatives.items():
            add_matrix_rows(
                matrix=matrix,
                method_name=method_labels.get(method_name, method_name),
                representation_type="Representative Vector"
            )

        for method_name, matrix in centroids.items():
            add_matrix_rows(
                matrix=matrix,
                method_name=centroid_labels.get(method_name, method_name),
                representation_type="Centroid"
            )

        if not rows:
            raise PreventUpdate

        df_centroids_representatives = pd.DataFrame(rows)

        return send_excel_file(
            df_centroids_representatives,
            "centroids_and_representatives.xlsx",
            sheet_name="Centroids Representatives",
            index=False
        )
    

    @dash_app.callback(
    Output("download-final-nmf-configuration", "data"),
    Input("download-final-nmf-configuration-btn", "n_clicks"),
    State("nmf-results-store", "data"),
    State("dataset-store", "data"),
    prevent_initial_call=True
    )
    def download_final_nmf_configuration(n_clicks, nmf_results, dataset_data):

        if not n_clicks:
            raise PreventUpdate

        if not nmf_results or not nmf_results.get("nmf_completed"):
            raise PreventUpdate

        clustering_labels = {
            "argmax": "Argmax",
            "kmeans": "K-Means",
            "fcm": "Fuzzy C-Means",
            "fcm_hard": "Fuzzy C-Means"
        }

        init_labels = {
            "random": "Random",
            "nndsvd": "NNDSVD"
        }

        nmf_labels = {
            "nmf_standard": "Standard NMF"
        }

        dataset_name = "Uploaded Dataset"

        if dataset_data:
            dataset_name = (
                dataset_data.get("display_name")
                or dataset_data.get("filename")
                or "Uploaded Dataset"
            )

        final_clustering = nmf_results.get("final_clustering", "-")
        final_init = nmf_results.get("final_init", nmf_results.get("selected_init", "-"))
        final_nmf = nmf_results.get("final_nmf", "nmf_standard")

        configuration = {
            "final_nmf_configuration": {
                "dataset": dataset_name,
                "selected_k": nmf_results.get("selected_k", "-"),
                "clustering_algorithm": clustering_labels.get(
                    final_clustering,
                    final_clustering
                ),
                "initialization": init_labels.get(
                    final_init,
                    final_init
                ),
                "nmf_algorithm": nmf_labels.get(
                    final_nmf,
                    final_nmf
                )
            }
        }

        return dict(
            content=json.dumps(
                configuration,
                indent=4,
                ensure_ascii=False
            ),
            filename="final_nmf_configuration.json",
            type="application/json"
        )


    @dash_app.callback(
        Output("download-w-explanations", "data"),
        Input("download-w-explanations-btn", "n_clicks"),
        State("fuzzy-settings", "data"),
        prevent_initial_call=True
    )
    def download_w_explanations(n_clicks, fuzzy_settings):
        if not n_clicks:
            raise PreventUpdate
        if not fuzzy_settings or "results" not in fuzzy_settings:
            raise PreventUpdate
        results = fuzzy_settings["results"]
        descriptions = results.get("w_descriptions", [])
        table = results.get("w_fuzzy_table", [])
        export_data = {
            "descriptions": descriptions,
            "fuzzy_table": table
        }
        return {
            "content": json.dumps(export_data, indent=4, ensure_ascii=False),
            "filename": "w_fuzzy_explanations.json"
        }
    

    @dash_app.callback(
        Output("download-h-explanations", "data"),
        Input("download-h-explanations-btn", "n_clicks"),
        State("fuzzy-settings", "data"),
        prevent_initial_call=True
    )
    def download_h_explanations(n_clicks, fuzzy_settings):
        if not n_clicks:
            raise PreventUpdate
        if not fuzzy_settings or "results" not in fuzzy_settings:
            raise PreventUpdate
        results = fuzzy_settings["results"]
        descriptions = results.get("h_descriptions", [])
        tables = results.get("h_fuzzy_tables", {})
        export_data = {
            "descriptions": descriptions,
            "fuzzy_tables": tables
        }
        return {
            "content": json.dumps(export_data, indent=4, ensure_ascii=False),
            "filename": "h_fuzzy_explanations.json"
        }
    

    @dash_app.callback(
        Output("download-example-explanations", "data"),
        Input("download-example-explanations-btn", "n_clicks"),
        State("fuzzy-settings", "data"),
        prevent_initial_call=True
    )
    def download_example_explanations(n_clicks, fuzzy_settings):
        if not n_clicks:
            raise PreventUpdate
        if not fuzzy_settings or "results" not in fuzzy_settings:
            raise PreventUpdate
        results = fuzzy_settings["results"]
        descriptions = results.get("sample_descriptions", [])
        table = results.get("sample_fuzzy_table", [])
        export_data = {
            "descriptions": descriptions,
            "fuzzy_table": table
        }
        return {
            "content": json.dumps(export_data, indent=4, ensure_ascii=False),
            "filename": "example_fuzzy_explanations.json"
        }
    
    @dash_app.callback(
        Output("download-methods-configuration", "data"),
        Input("download-methods-configuration-btn", "n_clicks"),
        State("fuzzy-settings", "data"),
        State("nmf-results-store", "data"),
        prevent_initial_call=True
    )
    def download_methods_configuration(n_clicks, fuzzy_settings, nmf_results):

        if not n_clicks:
            raise PreventUpdate

        if not fuzzy_settings:
            raise PreventUpdate

        config = nmf_results.get("config", {}) if nmf_results else {}

        fuzzy_method = fuzzy_settings.get("fuzzy_method", "equidistant")
        fuzzy_shape = fuzzy_settings.get("fuzzy_shape", "gaussian")
        fuzzy_target = fuzzy_settings.get("fuzzy_target", [])
        num_sets = fuzzy_settings.get("num_fuzzy_sets", "-")

        selected_k = nmf_results.get("selected_k", "-") if nmf_results else "-"

        init_labels = {
            "random": "Random",
            "nndsvd": "NNDSVD"
        }

        target_labels = {
            "W": "Matrix W",
            "H": "Matrix H"
        }

        readable_init = init_labels.get(
            str(config.get("nmf_init", "Selected in Step 3")).lower(),
            str(config.get("nmf_init", "Selected in Step 3"))
        )

        applied_to = [
            target_labels.get(target, target)
            for target in fuzzy_target
        ] if fuzzy_target else []

        methods_configuration = {
            "nmf": {
                "algorithm": "Standard NMF",
                "selected_k": selected_k,
                "initialization_method": readable_init,
                "description": (
                    "Non-negative Matrix Factorization is used to decompose "
                    "the dataset into latent factors."
                )
            },

            "clustering": {
                "methods": [
                    "Argmax",
                    "K-Means",
                    "Fuzzy C-Means"
                ],
                "cluster_explanations": "Representative vectors",
                "centroids": [
                    "K-Means centroids",
                    "Fuzzy C-Means centroids"
                ],
                "description": (
                    "Clustering strategies are applied in the latent-factor space. "
                    "Clusters are described using representative vectors computed "
                    "as the mean latent-factor profile of assigned samples."
                )
            },

            "fuzzy_explanations": {
                "fuzzy_set_creation": str(fuzzy_method).replace("_", " ").title(),
                "membership_function": str(fuzzy_shape).title(),
                "number_of_fuzzy_sets": num_sets,
                "applied_to": applied_to,
                "description": (
                    "Fuzzy linguistic labels are used to transform numerical "
                    "matrix values into interpretable qualitative descriptions."
                )
            }
        }

        return dict(
            content=json.dumps(
                methods_configuration,
                indent=4,
                ensure_ascii=False
            ),
            filename="methods_configuration.json",
            type="application/json"
        )
    
    # ──Callback: Verifica disponibilità API per Fuxplainer ───────────────
    @dash_app.callback(
        Output("fuxplainer-status", "children"),
        Input("check-api-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def check_api_availability(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        try:
            resp = requests.get(
                "http://localhost:5000/api/explanations",
                timeout=3
            )
            data = resp.json()
            if data.get("available"):
                return dbc.Alert(
                    [
                        html.I(className="fas fa-check-circle me-2"),
                        "API available."
                    ],
                    color="success",
                    dismissable=True
                )
            else:
                return dbc.Alert(
                    [
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "API available but no rules have been generated. "
                        "Press 'Generate Fuzzy Explanations'."
                    ],
                    color="warning",
                    dismissable=True
                )
        except Exception as e:
            return dbc.Alert(
                [
                    html.I(className="fas fa-times-circle me-2"),
                    f"Unable to reach API: {str(e)}"
                ],
                color="danger",
                dismissable=True
            )
    
    # ──Callback: Invia a Fuxplainer (apre localhost:5001/fetch-from-nmf) ─
    @dash_app.callback(
        Output("fuxplainer-status", "children", allow_duplicate=True),
        Input("send-to-fuxplainer-btn", "n_clicks"),
        State("fuzzy-settings", "data"),
        prevent_initial_call=True
    )
    def send_to_fuxplainer(n_clicks, fuzzy_settings):
        if not n_clicks:
            raise PreventUpdate
        # Verifica che le explanations siano state generate
        if not fuzzy_settings or not fuzzy_settings.get("fuzzy_completed"):
            return dbc.Alert(
                [
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "No explanation have been generated. "
                    "Press 'Generate Fuzzy Explanations'."
                ],
                color="warning",
                dismissable=True
            )
        try:
            # Verifica che Fuxplainer sia in ascolto su :5001
            ping = requests.get("http://localhost:5001/", timeout=3)
            fuxplainer_up = ping.status_code < 500
        except Exception:
            fuxplainer_up = False
        
        if not fuxplainer_up:
            return dbc.Alert(
                [
                    html.I(className="fas fa-times-circle me-2"),
                    "Fuxplainer not avaibe on",
                    html.Code("http://localhost:5001"),
                ],
                color="danger",
                dismissable=True
            )
        
        # Tutto ok: restituisce un link cliccabile che apre Fuxplainer
        # sul percorso /fetch-from-nmf (che recupera automaticamente i dati)
        return dbc.Alert(
            [
                html.I(className="fas fa-check-circle me-2"),
                "API and Fuxpleiner are ready! ",
                html.A(
                    "Open Fuxplainer → Importa da NMF",
                    href="http://localhost:5001/fetch-from-nmf",
                    target="_blank",
                    className="alert-link"
                ),
                " to automatically load fuzzy rules."
            ],
            color="success",
            dismissable=True
        )

#Report_page callbakcs
def fetch_data(session_data=None):
    """Recupera dati da backend per la pagina report."""
    try:
        sid = (session_data or {}).get("sid")
        response_terms = requests.get("http://127.0.0.1:5000/api/get_terms",
            headers={"X-Session-ID": sid})
        response_rules = requests.get("http://127.0.0.1:5000/api/get_rules",
            headers={"X-Session-ID": sid})
        if response_terms.status_code == 200 and response_rules.status_code == 200:
            terms_data = response_terms.json()
            rules_data = response_rules.json()
            return {"terms": terms_data, "rules": rules_data}
        else:
            return None
    except Exception as e:
        print(f"Error while loading data: {e}")
        return None

def generate_variable_section(variables, var_type):
    """Genera le card per visualizzare le variabili (input/output) nel report.""" 
    children = []
    for var_name, var_data in variables.items():
        children.append(
            dbc.Card([
                dbc.CardHeader(f"{var_type.capitalize()} Variable: {var_name}"),
                dbc.CardBody([
                    html.Div([
                        html.H5(var_name, className="text-primary mb-2" if var_type == "input" else "text-success mb-2"),
                        dbc.Row([
                            dbc.Col(f"Domain: {var_data['domain'][0]}-{var_data['domain'][1]}", width=6),
                            dbc.Col(f"Type: {var_type.capitalize()}", width=6),
                        ]),
                        html.Div(
                            className="mt-2",
                            children=[
                                html.Small("Membership Functions:", className="text-muted"),
                                html.Div([
                                    dbc.Badge(term["term_name"], color="info" if var_type == "input" else "secondary", className="me-1")
                                    for term in var_data["terms"]
                                ], className="mt-1")
                            ]
                        )
                    ], className="variable-card mb-3 p-3")
                ])
            ], className="shadow-sm")
        )
    return children

def generate_rules_section(rules):
    """Genera la sezione di visualizzazione delle regole fuzzy nel report.""" 
    children = []
    for rule in rules:
        inputs = rule.get("inputs", [])
        inputs_text = " AND ".join(
            f"({inp['input_variable']} IS {inp['input_term']})" for inp in inputs
        )
        output_text = f"({rule['output_variable']} IS {rule['output_term']})"
        rule_text = f"IF {inputs_text} THEN {output_text}"
        children.append(
            html.Li(
                rule_text,
                className="rule-item mb-2 p-2"
            )
        )
    return children