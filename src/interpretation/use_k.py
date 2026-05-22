from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.clustering import KMeans
from src.utils.nmf_helper import (
    nimfa_factorize, 
    resolve_nimfa_seed, 
    frobenius_reconstruction_error,
    row_normalize_sum_to_one,
    column_normalize_sum_to_one
)
from src.utils.fcm_helper import fit_fcm
from src.settings import (
    FEATURES,
    Nmf as NmfConfig,
    KMeans as KMeansConfig,
    FCMeans as FCMeansConfig,    
)


def run_final_nmf_clustering_on_h(
    X: np.ndarray,
    *,
    k: int = 3,
    seed: int = 42,
    nmf_init: str | None = None,
) -> Dict[str, Any]:
    '''
    Esegue una singola NMF finale coerente col blocco sperimentale e poi:
    - clusterizza su H grezza
    - normalizza W e H solo dopo il clustering
    - costruisce i representative vectors su H normalizzata

    Convenzione del progetto:
        X         : (n_samples, n_features)
        X_nmf     : X.T -> (n_features, n_samples)
        W         : (n_features, k)
        H         : (k, n_samples)
        Z = H.T   : (n_samples, k)
    '''
    print("\nInizio del blocco interpretativo...")

    X = np.asarray(X, dtype=np.float64)

    # Controlli di sicurezza
    if X.ndim != 2:
        raise ValueError(f"X deve essere 2D, ricevuto shape={X.shape}.")
    if np.any(X < 0):
        raise ValueError("NMF richiede una matrice non negativa.")

    n_samples, n_features = X.shape
    X_nmf = X.T

    init = resolve_nimfa_seed(nmf_init if nmf_init is not None else NmfConfig.INIT)   # resolve seed before calling nimfa

    # NMF finale, una sola volta
    W, H = nimfa_factorize(
        V=X_nmf,
        k=int(k),
        max_iter=NmfConfig.MAX_ITER,
        seed=int(seed),
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

    # Metriche NMF (errore di ricostruzione)
    X_hat = W @ H
    reconstruction_error = frobenius_reconstruction_error(X_nmf, X_hat)

    # Embedding dei sample
    Z = H.T  # (n_samples, k)

    # Clustering su H grezza / Z
    # argmax
    labels_argmax = np.asarray(np.argmax(H, axis=0), dtype=np.int64)
    silhouette_argmax = safe_silhouette_score(Z, labels_argmax)

    # kmeans
    kmeans = KMeans(
        n_clusters=int(k),
        init=KMeansConfig.INIT,
        n_init=KMeansConfig.N_INIT,
        max_iter=KMeansConfig.MAX_ITER,
        tol=KMeansConfig.TOL,
        random_state=int(seed),
    ).fit(Z)
    
    labels_kmeans = np.asarray(kmeans.labels_, dtype=np.int64)
    centroids_kmeans = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    silhouette_kmeans = safe_silhouette_score(Z, labels_kmeans)

    # fcmeans
    membership_fcm, centroids_fcm, labels_fcm_hard = fit_fcm(
        Z=Z,
        n_clusters=int(k),
        random_state=int(seed),
    )
    silhouette_fcm = safe_silhouette_score(Z, labels_fcm_hard)

    # Normalizzazione solo dopo il clustering
    W_norm = row_normalize_sum_to_one(W)
    H_norm = column_normalize_sum_to_one(H)

    samples_by_lf_norm = H_norm.T   # (samples, latent_factors)

    rep_argmax, sizes_argmax = build_cluster_representative_vectors(
        sample_embeddings=samples_by_lf_norm,
        labels=labels_argmax,
        n_clusters=int(k),
    )
    rep_kmeans, sizes_kmeans = build_cluster_representative_vectors(
        sample_embeddings=samples_by_lf_norm,
        labels=labels_kmeans,
        n_clusters=int(k),
    )
    rep_fcm_hard, sizes_fcm_hard = build_cluster_representative_vectors(
        sample_embeddings=samples_by_lf_norm,
        labels=labels_fcm_hard,
        n_clusters=int(k),
    )

    print("\nFine computazione del blocco interpretativo iniziale.")

    return {
        "config": {
            "k": int(k),
            "seed": int(seed),
            "nmf_init": init,
            "nmf_max_iter": int(NmfConfig.MAX_ITER),
            "nmf_update": str(NmfConfig.UPDATE),
            "kmeans_init": str(KMeansConfig.INIT),
            "kmeans_n_init": int(KMeansConfig.N_INIT),
            "kmeans_max_iter": int(KMeansConfig.MAX_ITER),
            "kmeans_tol": float(KMeansConfig.TOL),
            "fcm_m": float(FCMeansConfig.M),
            "fcm_max_iter": int(FCMeansConfig.MAX_ITER),
            "fcm_tol": float(FCMeansConfig.TOL),
        },
        "X_nmf_shape": X_nmf.shape,
        "W": W,
        "H": H,
        "W_norm": W_norm,
        "H_norm": H_norm,
        "Z": Z,
        "labels_argmax": labels_argmax,
        "labels_kmeans": labels_kmeans,
        "labels_fcm_hard": labels_fcm_hard,
        "membership_fcm": membership_fcm,
        "centroids_kmeans": centroids_kmeans,
        "centroids_fcm": centroids_fcm,
        "representatives": {
            "argmax": rep_argmax,
            "kmeans": rep_kmeans,
            "fcm_hard": rep_fcm_hard,
        },
        "cluster_sizes": {
            "argmax": sizes_argmax,
            "kmeans": sizes_kmeans,
            "fcm_hard": sizes_fcm_hard,
        },
        "metrics": {
            "reconstruction_error": reconstruction_error,
            "silhouette_argmax": silhouette_argmax,
            "silhouette_kmeans": silhouette_kmeans,
            "silhouette_fcm": silhouette_fcm,
        },
    }



def safe_silhouette_score(Z: np.ndarray, labels: np.ndarray) -> float:
    from src.nmf.silhouette import safe_silhouette          # appoggio a silhouette.py

    labels = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(labels)

    if unique_labels.size < 2:
        return float("nan")

    if unique_labels.size >= Z.shape[0]:
        return float("nan")

    return float(safe_silhouette(Z, labels))


def build_cluster_representative_vectors(
    sample_embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Representative vectors = media dei sample embeddings del cluster.
    (Ogni riga è il profilo medio del cluster sui fattori latenti)

    sample_embeddings:
    - shape (n_samples, n_latent_factors)
    - nel nostro caso useremo H_norm.T (supponendo di essere all'inizio del blocco interpretativo)
    '''
    sample_embeddings = np.asarray(sample_embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if sample_embeddings.ndim != 2:
        raise ValueError(
            f"sample_embeddings deve essere 2D, ricevuto shape={sample_embeddings.shape}."
        )
    if labels.ndim != 1:
        raise ValueError(f"labels deve essere 1D, ricevuto shape={labels.shape}.")
    if sample_embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            "Numero di sample in sample_embeddings e labels non coerente: "
            f"{sample_embeddings.shape[0]} vs {labels.shape[0]}."
        )

    n_latent_factors = sample_embeddings.shape[1]
    representative_vectors = np.full((n_clusters, n_latent_factors), np.nan, dtype=np.float64)
    cluster_sizes = np.zeros(n_clusters, dtype=np.int64)

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_sizes[cluster_id] = int(mask.sum())

        if np.any(mask):
            representative_vectors[cluster_id] = sample_embeddings[mask].mean(axis=0)

    return representative_vectors, cluster_sizes


def save_final_nmf_clustering_outputs(
    artifacts: Dict[str, Any],
    output_dir: str | Path,
) -> None:
    # Salva gli artefatti del blocco finale in CSV/NPY.
    from src.settings import get_csv_dir, get_numpy_dir

    output_dir = Path(output_dir)
    
    csv_dir = get_csv_dir(output_dir)
    npy_dir = get_numpy_dir(output_dir)
    
    # Definizioni legacy (funzionanti ma stilisticamente diverse dagli altri .py vicini)
    #output_dir = Path(output_dir)
    #csv_dir = output_dir / "csv"
    #npy_dir = output_dir / "numpy"

    csv_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    config = artifacts["config"]
    k = int(config["k"])

    feature_names = list(FEATURES)
    lf_names = [f"LF{i+1}" for i in range(k)]

    W = np.asarray(artifacts["W"], dtype=np.float64)
    H = np.asarray(artifacts["H"], dtype=np.float64)
    W_norm = np.asarray(artifacts["W_norm"], dtype=np.float64)
    H_norm = np.asarray(artifacts["H_norm"], dtype=np.float64)
    Z = np.asarray(artifacts["Z"], dtype=np.float64)

    n_samples = Z.shape[0]
    sample_names = [f"sample_{i}" for i in range(n_samples)]
    cluster_names = [f"Cluster {i+1}" for i in range(k)]

    pd.DataFrame(W, index=feature_names, columns=lf_names).to_csv(csv_dir / "W.csv")
    pd.DataFrame(H, index=lf_names, columns=sample_names).to_csv(csv_dir / "H.csv")
    pd.DataFrame(W_norm, index=feature_names, columns=lf_names).to_csv(csv_dir / "W_normalized.csv")
    pd.DataFrame(H_norm, index=lf_names, columns=sample_names).to_csv(csv_dir / "H_normalized.csv")
    pd.DataFrame(Z, index=sample_names, columns=lf_names).to_csv(csv_dir / "H_transposed_samples_by_lf.csv")

    pd.DataFrame([artifacts["metrics"]]).to_csv(csv_dir / "metrics.csv", index=False)
    pd.DataFrame([config]).to_csv(csv_dir / "config.csv", index=False)

    pd.DataFrame({"sample": sample_names, "label": artifacts["labels_argmax"]}).to_csv(
        csv_dir / "labels_argmax.csv", index=False
    )
    pd.DataFrame({"sample": sample_names, "label": artifacts["labels_kmeans"]}).to_csv(
        csv_dir / "labels_kmeans.csv", index=False
    )
    pd.DataFrame({"sample": sample_names, "label": artifacts["labels_fcm_hard"]}).to_csv(
        csv_dir / "labels_fcm_hard.csv", index=False
    )

    pd.DataFrame(
        np.asarray(artifacts["membership_fcm"], dtype=np.float64),
        index=sample_names,
        columns=cluster_names,
    ).to_csv(csv_dir / "membership_fcm.csv")

    pd.DataFrame(
        np.asarray(artifacts["centroids_kmeans"], dtype=np.float64),
        index=cluster_names,
        columns=lf_names,
    ).to_csv(csv_dir / "centroids_kmeans.csv")

    pd.DataFrame(
        np.asarray(artifacts["centroids_fcm"], dtype=np.float64),
        index=cluster_names,
        columns=lf_names,
    ).to_csv(csv_dir / "centroids_fcm.csv")

    for method in ("argmax", "kmeans", "fcm_hard"):
        pd.DataFrame(
            np.asarray(artifacts["representatives"][method], dtype=np.float64),
            index=cluster_names,
            columns=lf_names,
        ).to_csv(csv_dir / f"representative_vectors_{method}.csv")

        pd.DataFrame(
            {
                "cluster": cluster_names,
                "size": np.asarray(artifacts["cluster_sizes"][method], dtype=np.int64),
            }
        ).to_csv(csv_dir / f"cluster_sizes_{method}.csv", index=False)

    np.save(npy_dir / "W.npy", W)
    np.save(npy_dir / "H.npy", H)
    np.save(npy_dir / "W_normalized.npy", W_norm)
    np.save(npy_dir / "H_normalized.npy", H_norm)
    np.save(npy_dir / "Z.npy", Z)
