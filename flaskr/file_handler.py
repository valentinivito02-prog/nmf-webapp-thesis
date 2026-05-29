from flask import request, has_request_context
from collections import defaultdict

# Dizionario in-memory: _STORE[session_id][chiave] = dati
# Viene azzerato ad ogni riavvio del server (comportamento intenzionale).
# L'isolamento per utente avviene tramite il session ID (uuid4) generato
# dal browser e trasmesso nell'header X-Session-ID di ogni richiesta API.
_STORE = defaultdict(dict)


def _current_sid() -> str:
    """Restituisce il session ID dall'header X-Session-ID, o 'default' se assente."""
    if has_request_context():
        sid = request.headers.get("X-Session-ID")
        if isinstance(sid, str) and sid.strip():
            return sid.strip()

    return "default"


def save_data(data):
    sid = _current_sid()
    _STORE[sid]["data"] = data if isinstance(data, dict) else {}


def load_data():
    sid = _current_sid()
    return _STORE[sid].get("data", {})


def load_terms():
    sid = _current_sid()
    return _STORE[sid].get("terms", {})


def save_terms(data):
    sid = _current_sid()
    _STORE[sid]["terms"] = data if isinstance(data, dict) else {}


def load_rule():
    sid = _current_sid()
    return _STORE[sid].get(
        "rules",
        {
            "rules": [],
            "dropdown_options": []
        }
    )


def save_rule(data):
    sid = _current_sid()
    _STORE[sid]["rules"] = (
        data
        if isinstance(data, dict)
        else {
            "rules": [],
            "dropdown_options": []
        }
    )


def save_explanations(data):
    """Salva le fuzzy explanations generate da nmf-webapp (w, h, examples)."""
    sid = _current_sid()
    _STORE[sid]["explanations"] = data if isinstance(data, dict) else {}


def load_explanations():
    """Carica le fuzzy explanations salvate per la sessione corrente."""
    sid = _current_sid()
    return _STORE[sid].get("explanations", None)