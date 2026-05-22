from __future__ import annotations
import numpy as np
import pandas as pd

from src.settings import LABEL_MAP


def z_membership(x: np.ndarray, a: float, b: float) -> np.ndarray:
    '''
    Z-shaped membership function.
    High membership on the left side, low on the right side.
    '''
    x = np.asarray(x, dtype=float)

    if b <= a:
        raise ValueError(f"Richiesto b > a, ricevuti a={a}, b={b}.")

    y = np.ones_like(x, dtype=float)
    mid = (a + b) / 2.0

    mask1 = (x > a) & (x <= mid)
    mask2 = (x > mid) & (x < b)

    y[x >= b] = 0.0
    y[mask1] = 1.0 - 2.0 * ((x[mask1] - a) / (b - a)) ** 2
    y[mask2] = 2.0 * ((x[mask2] - b) / (b - a)) ** 2

    return np.clip(y, 0.0, 1.0)


def s_membership(x: np.ndarray, a: float, b: float) -> np.ndarray:
    '''
    S-shaped membership function.
    Low membership on the left side, high on the right side.
    '''
    x = np.asarray(x, dtype=float)

    if b <= a:
        raise ValueError(f"Richiesto b > a, ricevuti a={a}, b={b}.")

    y = np.zeros_like(x, dtype=float)
    mid = (a + b) / 2.0

    mask1 = (x > a) & (x <= mid)
    mask2 = (x > mid) & (x < b)

    y[x >= b] = 1.0
    y[mask1] = 2.0 * ((x[mask1] - a) / (b - a)) ** 2
    y[mask2] = 1.0 - 2.0 * ((x[mask2] - b) / (b - a)) ** 2

    return np.clip(y, 0.0, 1.0)


def gaussian_membership(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    # Gaussian membership function.
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-12)
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def generate_equidistant_fuzzy_sets(
    universe: np.ndarray,
    n_sets: int = 5,
) -> list[np.ndarray]:
    '''
    Costruisce fuzzy sets equidistanti sull'universo dato.

    Convenzione:
    - primo set: Z-shaped
    - ultimi set interni: Gaussian
    - ultimo set: S-shaped
    '''
    universe = np.asarray(universe, dtype=float)

    if universe.ndim != 1:
        raise ValueError("universe deve essere un array 1D.")
    if n_sets < 2:
        raise ValueError("n_sets deve essere almeno 2.")

    centers = np.linspace(float(np.min(universe)), float(np.max(universe)), n_sets)
    step = centers[1] - centers[0]
    sigma = step / 2.5

    fuzzy_sets: list[np.ndarray] = []
    fuzzy_sets.append(z_membership(universe, centers[0], centers[0] + step))

    for center in centers[1:-1]:
        fuzzy_sets.append(gaussian_membership(universe, center, sigma))

    fuzzy_sets.append(s_membership(universe, centers[-1] - step, centers[-1]))
    return fuzzy_sets


def get_max_membership_fuzzy_set_ids(
    x_universe: np.ndarray,
    fuzzy_sets: list[np.ndarray],
    input_vector: np.ndarray,
) -> list[int]:
    # Per ogni valore del vettore restituisce l'id (1-based) del fuzzy set con massima membership.
    x_universe = np.asarray(x_universe, dtype=float)
    input_vector = np.asarray(input_vector, dtype=float)

    max_membership_ids: list[int] = []

    for value in input_vector:
        memberships = [float(np.interp(value, x_universe, fs)) for fs in fuzzy_sets]
        memberships = np.asarray(memberships, dtype=float)
        max_membership_ids.append(int(np.argmax(memberships)) + 1)

    return max_membership_ids


def fuzzify_vector(
    vector: np.ndarray,
    x_universe: np.ndarray,
    fuzzy_sets: list[np.ndarray],
    label_map: dict[int, str] | None = None,
) -> list[str]:
    # Fuzzifica un vettore numerico in una lista di etichette linguistiche.
    if label_map is None:
        label_map = LABEL_MAP

    vector = np.asarray(vector, dtype=float)
    ids = get_max_membership_fuzzy_set_ids(x_universe, fuzzy_sets, vector)
    return [label_map.get(i, "Unknown") for i in ids]


def fuzzify_matrix_rows(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    x_universe: np.ndarray,
    fuzzy_sets: list[np.ndarray],
    empty_label: str = "Empty cluster",
    label_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    '''
    Fuzzifica una matrice per righe.
    Ogni riga è trasformata in una descrizione linguistica per colonna.
    '''
    if label_map is None:
        label_map = LABEL_MAP

    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError(f"matrix deve essere 2D, ricevuto shape={matrix.shape}.")
    if len(row_labels) != matrix.shape[0]:
        raise ValueError("row_labels non coerente con il numero di righe.")
    if len(col_labels) != matrix.shape[1]:
        raise ValueError("col_labels non coerente con il numero di colonne.")

    rows = []
    for row in matrix:
        if np.isnan(row).all():
            rows.append([empty_label] * len(col_labels))
        else:
            rows.append(
                fuzzify_vector(
                    vector=row,
                    x_universe=x_universe,
                    fuzzy_sets=fuzzy_sets,
                    label_map=label_map,
                )
            )

    return pd.DataFrame(rows, index=row_labels, columns=col_labels)