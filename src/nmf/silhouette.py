import numpy as np


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    # Controllo di sicurezza (devono esserci almeno 2 cluster)
    uniq, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    k = uniq.size
    if k < 2:
        return float("nan")

    n = X.shape[0]
    if n == 0:
        return float("nan")

    # Distanze euclidee (n x n)
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))    # D[i,i]=0

    # Matrice indicatrice cluster: M (n x k), M[i,c]=1 se punto i nel cluster c
    M = np.eye(k, dtype=float)[inv]         # one-hot
    cluster_sizes = counts.astype(float)    # (k,)

    S = D @ M       # Somma distanze da ogni punto a ciascun cluster: S = D @ M  -> (n x k)

    # Media distanze da ogni punto a ciascun cluster: mean_dist (n x k)
    mean_dist = S / cluster_sizes[None, :]

    # a(i): media distanze nel proprio cluster escludendo sé stesso
    # S[i, own] include D[i,i]=0, quindi basta dividere per (size-1)
    own = inv
    denom_a = (cluster_sizes[own] - 1.0)
    a = np.empty(n, dtype=float)
    a[:] = np.nan
    mask_not_singleton = denom_a > 0
    a[mask_not_singleton] = S[np.arange(n)[mask_not_singleton], own[mask_not_singleton]] / denom_a[mask_not_singleton]

    # b(i): minimo tra le medie verso cluster diversi dal proprio
    tmp = mean_dist.copy()
    tmp[np.arange(n), own] = np.inf
    b = np.min(tmp, axis=1)

    s = (b - a) / np.maximum(a, b)      # silhouette per punto

    # cluster singoletti -> silhouette = 0 (convenzione tipo sklearn)
    s[~mask_not_singleton] = 0.0

    return float(np.nanmean(s))
