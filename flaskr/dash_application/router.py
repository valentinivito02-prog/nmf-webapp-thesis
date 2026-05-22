import logging
from typing import Any, Dict, Callable
from dash import Output, Input, html, dcc

from .pages import (
    home_page,
    upload_page,
    choose_k_page,
    run_page,
    fuzzy_page,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def error_404_layout() -> html.Div:
    return html.Div(
        className="error-container",
        children=[
            html.H1("404 - Page not found", className="error-title"),
            html.P("The page you are looking for does not exist."),
            dcc.Link(
                "Back to Home",
                href="/",
                style={"color": "white", "textDecoration": "underline"}
            )
        ],
        style={
            "textAlign": "center",
            "marginTop": "50px",
            "padding": "20px",
            "backgroundColor": "#f8d7da",
            "borderRadius": "5px"
        }
    )


def register_routing(dash_app: Any) -> None:
    routes: Dict[str, Callable[[], html.Div]] = {
        "/": home_page.layout,
        "/upload": upload_page.layout,
        "/choose-k": choose_k_page.layout,
        "/run": run_page.layout,
        "/fuzzy": fuzzy_page.layout,
    }

    missing_layouts = [
        path for path, layout in routes.items()
        if not callable(layout)
    ]

    if missing_layouts:
        error_msg = f"Missing layouts for routes: {', '.join(missing_layouts)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    @dash_app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
        prevent_initial_call=False
    )
    def render_page_content(pathname: str) -> html.Div:
        logger.info(f"Requested route: {pathname}")

        try:
            normalized_path = pathname.strip().rstrip("/")
            if not normalized_path:
                normalized_path = "/"

            layout_func = routes.get(normalized_path, error_404_layout)
            return layout_func()

        except Exception as e:
            logger.error(
                f"Critical error during rendering: {str(e)}",
                exc_info=True
            )
            return error_404_layout()