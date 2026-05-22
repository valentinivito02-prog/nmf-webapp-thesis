from typing import Tuple

import numpy as np
from fcmeans import FCM

from src.settings import FCMeans as FCMeansConfig

def fit_fcm(
    Z: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Adapter per FCM: ritorna membership, centroidi e hard labels nel formato atteso.
    fcm = FCM(
        n_clusters=int(n_clusters),
        m=FCMeansConfig.M,
        max_iter=FCMeansConfig.MAX_ITER,
        error=FCMeansConfig.TOL,
        random_state=int(random_state),
    )
    fcm.fit(Z)

    membership = np.asarray(fcm.u, dtype=np.float64)
    centroids = np.asarray(fcm.centers, dtype=np.float64)
    labels_hard = np.asarray(fcm.predict(Z), dtype=np.int64)

    # Controlli di shape
    if membership.ndim != 2:
        raise ValueError(f"FCM membership must be 2D, got shape={membership.shape}")
    if membership.shape != (Z.shape[0], n_clusters):
        raise ValueError(
            f"FCM membership has shape {membership.shape}, expected {(Z.shape[0], n_clusters)}"
        )
    if centroids.shape != (n_clusters, Z.shape[1]):
        raise ValueError(
            f"FCM centroids have shape {centroids.shape}, expected {(n_clusters, Z.shape[1])}"
        )
    if labels_hard.shape != (Z.shape[0],):
        raise ValueError(
            f"FCM hard labels have shape {labels_hard.shape}, expected {(Z.shape[0],)}"
        )
    
    row_sums = membership.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("FCM membership rows do not sum to 1 within tolerance.")

    return membership, centroids, labels_hard