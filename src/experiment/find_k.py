from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.nmf_helper import (
    nimfa_factorize, 
    resolve_nimfa_seed, 
    frobenius_reconstruction_error
)
from src.clustering import KMeans
from src.utils.fcm_helper import fit_fcm
from src.nmf.cophenetic import (
    connectivity_from_labels,
    connectivity_from_fcm_membership,
    consensus_from_connectivities,
    cophenetic_from_consensus,
)
from src.settings import (
    Nmf as NmfConfig, 
    KMeans as KMeansConfig, 
    FCMeans as FCMeansConfig
)


def experimental_block_cluster_on_H(
    X: np.ndarray,
    k_values: Sequence[int] = range(2, 15),
    max_rep: int = 20,
    nmf_init: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Dict[str, Any]]]:
    '''
    Blocco sperimentale NMF + clustering su H.

    Assunzioni:
    - X originale ha shape (n_samples, n_features)
    - per seguire la convenzione classica della letteratura NMF
      e clusterizzare i campioni su H, si fattorizza X.T

    Quindi:
        X_nmf = X.T  -> (n_features, n_samples)
        X_nmf ≈ W @ H
    con:
        W : (n_features, k)
        H : (k, n_samples)
    e i campioni si clusterizzano su:
        Z = H.T -> (n_samples, k)
    '''
    print("\nInizio sperimentazione...")

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X deve essere 2D, ricevuto shape={X.shape}.")

    if np.any(X < 0):
        raise ValueError("NMF richiede una matrice non negativa.")

    # Preparazione matrice per NMF
    print("\ncreando NMF...")
    n_samples, n_features = X.shape
    X_nmf = X.T     # convenzione

    risultati_per_run: List[Dict[str, Any]] = []
    risultati_per_k: List[Dict[str, Any]] = []
    artifacts: Dict[int, Dict[str, Any]] = {}

    init = resolve_nimfa_seed(nmf_init if nmf_init is not None else NmfConfig.INIT)       # resolve seed before calling nimfa

    # Ciclo sui valori di k
    for k in k_values:
        print(f"\nciclando su k = {k}...")
        reconstruction_errors: List[float] = []
        silhouettes_argmax: List[float] = []
        silhouettes_kmeans: List[float] = []
        silhouettes_fcm: List[float] = []

        labels_argmax_all: List[np.ndarray] = []
        labels_kmeans_all: List[np.ndarray] = []
        fcm_connectivities_all: Dict[str, List[np.ndarray]] = {
            mode: [] for mode in FCMeansConfig.CONSENSUS_MODES
        }

        run_artifacts: List[Dict[str, Any]] = []

        # Repliche della NMF
        if max_rep < 1:
            raise ValueError(f"Il numero di ripetizioni della NMF dato ({max_rep}) è minore di 1.")

        for rep in range(1, max_rep + 1):
            print(f"  rep = {rep}/{max_rep}", end="\r")
            # NMF
            W, H = nimfa_factorize(
                V=X_nmf,
                k=int(k),
                max_iter=NmfConfig.MAX_ITER,
                seed=int(rep),
                init=init,
                update=NmfConfig.UPDATE,
                track_error=False,
            )

            W = np.asarray(W, dtype=np.float64)
            H = np.asarray(H, dtype=np.float64)

            if W.shape != (n_features, k):
                raise ValueError(f"W ha shape {W.shape}, attesa {(n_features, k)}.")

            if H.shape != (k, n_samples):
                raise ValueError(f"H ha shape {H.shape}, attesa {(k, n_samples)}.")

            # Errore di ricostruzione
            X_hat = W @ H
            reconstruction_error = frobenius_reconstruction_error(X_nmf, X_hat)

            Z = H.T     # Embedding dei campioni

            # argmax
            labels_argmax = np.asarray(np.argmax(H, axis=0), dtype=np.int64)
            silhouette_argmax = safe_silhouette_score(Z, labels_argmax)

            # K-Means
            kmeans = KMeans(
                n_clusters=int(k),
                init=KMeansConfig.INIT,
                n_init=KMeansConfig.N_INIT,
                max_iter=KMeansConfig.MAX_ITER,
                tol=KMeansConfig.TOL,
                random_state=int(rep),
            ).fit(Z)

            labels_kmeans = np.asarray(kmeans.labels_, dtype=np.int64)
            centroids_kmeans = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
            silhouette_kmeans = safe_silhouette_score(Z, labels_kmeans)

            # Fuzzy-C-Means
            membership_fcm, centroids_fcm, labels_fcm_hard = fit_fcm(
                Z=Z,
                n_clusters=int(k),
                random_state=int(rep),
            )
            silhouette_fcm = safe_silhouette_score(Z, labels_fcm_hard)

            connectivity_fcm_by_mode: Dict[str, np.ndarray] = {
                "hard": connectivity_from_labels(labels_fcm_hard),
                "soft_dot": connectivity_from_fcm_membership(
                    membership_fcm,
                    mode="soft_dot",
                    sharpen=bool(FCMeansConfig.USE_SHARPENING),
                    alpha=float(FCMeansConfig.SHARPEN_ALPHA),
                ),
                "soft_cosine": connectivity_from_fcm_membership(
                    membership_fcm,
                    mode="soft_cosine",
                    sharpen=bool(FCMeansConfig.USE_SHARPENING),
                    alpha=float(FCMeansConfig.SHARPEN_ALPHA),
                ),
            }

            # Salvataggio artefatti della run
            run_artifacts.append(
                {
                    "k": k,
                    "rep": rep,
                    "X_nmf_shape": X_nmf.shape,
                    "W": W,
                    "H": H,
                    "Z": Z,
                    "labels_argmax": labels_argmax,
                    "labels_kmeans": labels_kmeans,
                    "labels_fcm_hard": labels_fcm_hard,
                    "membership_fcm": membership_fcm,
                    "centroids_kmeans": centroids_kmeans,
                    "centroids_fcm": centroids_fcm,
                    "reconstruction_error": reconstruction_error,
                    "silhouette_argmax": silhouette_argmax,
                    "silhouette_kmeans": silhouette_kmeans,
                    "silhouette_fcm": silhouette_fcm,
                    "fcm_connectivities": connectivity_fcm_by_mode,
                    "fcm_sharpening_enabled": bool(FCMeansConfig.USE_SHARPENING),
                    "fcm_sharpen_alpha": float(FCMeansConfig.SHARPEN_ALPHA),
                }
            )

            risultati_per_run.append(
                {
                    "k": k,
                    "rep": rep,
                    "reconstruction_error": reconstruction_error,
                    "silhouette_argmax": silhouette_argmax,
                    "silhouette_kmeans": silhouette_kmeans,
                    "silhouette_fcm": silhouette_fcm,
                }
            )

            reconstruction_errors.append(reconstruction_error)
            silhouettes_argmax.append(silhouette_argmax)
            silhouettes_kmeans.append(silhouette_kmeans)
            silhouettes_fcm.append(silhouette_fcm)

            labels_argmax_all.append(labels_argmax)
            labels_kmeans_all.append(labels_kmeans)
            for mode, connectivity in connectivity_fcm_by_mode.items():
                fcm_connectivities_all[mode].append(connectivity)

        # Matrici di Consenso
        consensus_argmax = build_hard_consensus_matrix(labels_argmax_all)
        consensus_kmeans = build_hard_consensus_matrix(labels_kmeans_all)
        consensus_fcm_by_mode = {
            mode: consensus_from_connectivities(connectivities)
            for mode, connectivities in fcm_connectivities_all.items()
        }

        # Correlazione Cophenetica
        coph_argmax = float(cophenetic_from_consensus(consensus_argmax, linkage_method="average"))
        coph_kmeans = float(cophenetic_from_consensus(consensus_kmeans, linkage_method="average"))
        coph_fcm_by_mode = {
            mode: float(cophenetic_from_consensus(consensus, linkage_method="average"))
            for mode, consensus in consensus_fcm_by_mode.items()
        }

        artifacts[k] = {
            "consensus": {
                "argmax": consensus_argmax,
                "kmeans": consensus_kmeans,
                "fcm": consensus_fcm_by_mode,
            },
            "cophenetic": {
                "argmax": coph_argmax,
                "kmeans": coph_kmeans,
                "fcm": coph_fcm_by_mode,
            },
            "config": {
                "fcm_modes": FCMeansConfig.CONSENSUS_MODES,
                "fcm_use_sharpening": bool(FCMeansConfig.USE_SHARPENING),
                "fcm_sharpen_alpha": float(FCMeansConfig.SHARPEN_ALPHA),
            },
        }

        # Aggregazione risultati per k
        risultati_per_k.append(
            {
                "k": k,
                "reconstruction_error_mean": mean_or_nan(reconstruction_errors),
                "reconstruction_error_std": std_or_nan(reconstruction_errors),
                "silhouette_argmax_mean": mean_or_nan(silhouettes_argmax),
                "silhouette_argmax_std": std_or_nan(silhouettes_argmax),
                "silhouette_kmeans_mean": mean_or_nan(silhouettes_kmeans),
                "silhouette_kmeans_std": std_or_nan(silhouettes_kmeans),
                "silhouette_fcm_mean": mean_or_nan(silhouettes_fcm),
                "silhouette_fcm_std": std_or_nan(silhouettes_fcm),
                "coph_argmax": coph_argmax,
                "coph_kmeans": coph_kmeans,
                "coph_fcm_hard": coph_fcm_by_mode["hard"],
                "coph_fcm_soft_dot": coph_fcm_by_mode["soft_dot"],
                "coph_fcm_soft_cosine": coph_fcm_by_mode["soft_cosine"],
                "fcm_use_sharpening": bool(FCMeansConfig.USE_SHARPENING),
                "fcm_sharpen_alpha": float(FCMeansConfig.SHARPEN_ALPHA),
            }
        )

    risultati_per_run_df = pd.DataFrame(risultati_per_run)
    risultati_per_k_df = pd.DataFrame(risultati_per_k)

    print("\nBlocco sperimentale completato.")
    return risultati_per_run_df, risultati_per_k_df, artifacts



def safe_silhouette_score(Z: np.ndarray, labels: np.ndarray) -> float:
    from src.nmf.silhouette import safe_silhouette          # appoggio a silhouette.py

    labels = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(labels)

    if unique_labels.size < 2:
        return float("nan")

    if unique_labels.size >= Z.shape[0]:
        return float("nan")

    return float(safe_silhouette(Z, labels))


def build_hard_consensus_matrix(labels_list: Sequence[np.ndarray]) -> np.ndarray:
    connectivities = [
        connectivity_from_labels(np.asarray(labels, dtype=np.int64))
        for labels in labels_list
    ]
    return np.asarray(consensus_from_connectivities(connectivities), dtype=np.float64)


def mean_or_nan(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if arr.size > 0 else float("nan")


def std_or_nan(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanstd(arr, ddof=1)) if arr.size > 1 else float("nan")


def save_consensus_matrices(
    artifacts: dict,
    output_dir: str | Path,
) -> None:
    '''
    Salva tutte le matrici di consenso in formato .npy.

    Struttura attesa di artifacts:
        artifacts[k]["consensus"]["argmax"]
        artifacts[k]["consensus"]["kmeans"]
        artifacts[k]["consensus"]["fcm"]["hard" | "soft_dot" | "soft_cosine"]
    '''
    print("\nSalvando le matrici di consenso (.npy)...")
    from src.settings import relative_to_project
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for k in sorted(artifacts.keys()):
        consensus_block = artifacts[k]["consensus"]

        k_dir = out_path / f"k_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        matrices = {
            "argmax": consensus_block["argmax"],
            "kmeans": consensus_block["kmeans"],
            "fcm_hard": consensus_block["fcm"]["hard"],
            "fcm_soft_dot": consensus_block["fcm"]["soft_dot"],
            "fcm_soft_cosine": consensus_block["fcm"]["soft_cosine"],
        }

        for name, matrix in matrices.items():
            matrix = np.asarray(matrix, dtype=np.float32)       # float 32 perché tanto le matrici di consenso sono min 0 e max 1
            np.save(k_dir / f"consensus_{name}.npy", matrix)     

    print(f"\nMatrici di consenso salvate in:\n{relative_to_project(out_path)}")
