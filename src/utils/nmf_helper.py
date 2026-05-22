from __future__ import annotations
from typing import Tuple

import numpy as np
import nimfa
from nimfa.methods import seeding as seeding_mod

from src.settings import Nmf as NmfConfig


def resolve_nimfa_seed(requested: str, *, fallback: str = "nndsvd") -> str:
    # Converte un seed 'requested' in un seed supportato dalla nimfa installata.
    from src.settings import SEED_ALIASES

    req = (requested or "").strip().lower()
    mapped = SEED_ALIASES.get(req, req)

    available = {s.lower() for s in dir(seeding_mod) if not s.startswith("_")}
    if mapped in available:
        return mapped

    fb = fallback.lower()
    if fb in available:
        print(f"[WARN] seed '{requested}' -> '{mapped}' non supportato; uso fallback '{fb}'.")
        return fb

    # ultima spiaggia: primo seed disponibile (stabile)
    first = sorted(available)[0]
    print(f"[WARN] seed '{requested}' non supportato e fallback '{fallback}' non disponibile; uso '{first}'.")
    return first


def nimfa_factorize(    
    V: np.ndarray,
    k: int,
    max_iter: int,
    seed: int,
    init: str = "nndsvd",
    update: str = "euclidean",
    track_error: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    This function was built to make it easier to execute many NMF runs.

    Factorize V ≈ W H using nimfa.
    V shape: (features, samples) and must be non-negative.

    Returns:
      W: (features, k)
      H: (k, samples)
    '''
    if np.any(V < 0):
        raise ValueError("NMF requires non-negative V. Found negative values in V.")

    np.random.seed(int(seed))       # Nimfa initialization often uses numpy's global RNG

    V_m = np.asmatrix(V)

    # In this function we assume that the Nimfa init has already been resolved before call
    nmf = nimfa.Nmf(
        V_m,
        rank=int(k),
        seed=str(init),
        max_iter=int(max_iter),
        update=str(update),
        track_error=bool(track_error),
    )
    fit = nmf()

    W = np.asarray(fit.basis())     # (features, k)
    H = np.asarray(fit.coef())      # (k, samples)
    return W, H


def frobenius_reconstruction_error(X: np.ndarray, X_hat: np.ndarray) -> float:
    return float(np.linalg.norm(X - X_hat, ord="fro"))


def row_normalize_sum_to_one(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    '''
    Normalizza le righe di X in modo che sommino a 1.

    Nel progetto ci si aspetta che:
    - W ha shape (n_features, k)

    Quindi:
    - W_norm[i, :] = distribuzione del contributo della feature i
      sui k latent factors
    '''
    X = np.asarray(X, dtype=np.float64)
    denom = X.sum(axis=1, keepdims=True)
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    return X / denom


def column_normalize_sum_to_one(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    '''
    Normalizza le colonne di X in modo che sommino a 1.

    Nel progetto ci si aspetta che:
    - H ha shape (k, n_samples)

    Quindi:
    - H_norm[:, j] = distribuzione dei latent factors nel sample j
    '''
    X = np.asarray(X, dtype=np.float64)
    denom = X.sum(axis=0, keepdims=True)
    denom = np.where(np.abs(denom) < eps, 1.0, denom)
    return X / denom
