from __future__ import annotations
import numpy as np

from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform


def cophenetic_from_consensus(consensus: np.ndarray, linkage_method: str = "average") -> float:
    # Cophenetic correlation computed on D = 1 - consensus.
    C = np.asarray(consensus, dtype=np.float64)
    C = np.clip(C, 0.0, 1.0)

    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0  # enforce perfect symmetry

    y = squareform(D, checks=False)
    Z = linkage(y, method=str(linkage_method))
    coph, _ = cophenet(Z, y)
    return float(coph)


def connectivity_from_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return (labels[:, None] == labels[None, :]).astype(np.float32)


def sharpen_membership(U: np.ndarray, alpha: float = 2.0, eps: float = 1e-12) -> np.ndarray:
    '''
    Row-wise sharpening of fuzzy memberships.

    U: (n_samples, n_clusters), each row should sum ~1.
    alpha > 1 sharpens dominant memberships.
    alpha = 1 leaves U unchanged.
    '''
    U = np.asarray(U, dtype=np.float64)

    if U.ndim != 2:
        raise ValueError(f"U must be 2D, got shape={U.shape}")

    if np.any(U < 0):
        raise ValueError("Membership matrix U contains negative values.")

    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    if np.isclose(alpha, 1.0):
        U_out = U.copy()
    else:
        U_pow = np.power(U, alpha, dtype=np.float64)
        row_sums = U_pow.sum(axis=1, keepdims=True)
        U_out = U_pow / np.maximum(row_sums, eps)

    U_out = np.nan_to_num(U_out, nan=0.0, posinf=0.0, neginf=0.0)
    return U_out


def connectivity_from_membership_dot(
    U: np.ndarray,
    sharpen: bool = False,
    alpha: float = 2.0,
) -> np.ndarray:
    '''
    Fuzzy connectivity via dot product:
        S_ij = sum_c u_ic * u_jc
    '''
    U = np.asarray(U, dtype=np.float64)

    if sharpen:
        U = sharpen_membership(U, alpha=alpha)

    S = U @ U.T
    S = np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=0.0)

    # numerical safety
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    S = np.clip(S, 0.0, 1.0)
    return S.astype(np.float32)


def connectivity_from_membership_cosine(
    U: np.ndarray,
    sharpen: bool = False,
    alpha: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    # Fuzzy connectivity via cosine similarity between membership rows.
    U = np.asarray(U, dtype=np.float64)

    if sharpen:
        U = sharpen_membership(U, alpha=alpha)

    norms = np.linalg.norm(U, axis=1, keepdims=True)
    U_norm = U / np.maximum(norms, eps)

    S = U_norm @ U_norm.T
    S = np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=0.0)

    # numerical safety | ensures diag=1.
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    S = np.clip(S, 0.0, 1.0)
    return S.astype(np.float32)


def consensus_from_connectivities(conns: list[np.ndarray]) -> np.ndarray:
    if not conns:
        raise ValueError("No runs provided to build consensus.")
    C = np.mean(conns, axis=0, dtype=np.float32)
    np.fill_diagonal(C, 1.0)
    return C


def connectivity_from_fcm_membership(
    U: np.ndarray,
    mode: str,
    sharpen: bool = False,
    alpha: float = 2.0,
) -> np.ndarray:
    mode = str(mode).lower()

    if mode == "soft_dot":
        return connectivity_from_membership_dot(U, sharpen=sharpen, alpha=alpha)
    if mode == "soft_cosine":
        return connectivity_from_membership_cosine(U, sharpen=sharpen, alpha=alpha)

    raise ValueError(f"Unsupported fuzzy connectivity mode: {mode!r}")


def fcm_mode_tag(mode: str, use_sharpening: bool, alpha: float) -> str:
    mode = str(mode).lower()
    if mode == "hard":
        return "hard"
    if use_sharpening:
        return f"{mode}_sharp{alpha:g}"
    return mode