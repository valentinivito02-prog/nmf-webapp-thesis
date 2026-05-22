from __future__ import annotations
import numpy as np


class KMeans:

    def __init__(
        self,
        n_clusters: int,
        *,
        init: str = "kmeans++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        # Controlli di sicurezza
        if n_clusters <= 0:
            raise ValueError("n_clusters must be > 0")
        if n_init <= 0:
            raise ValueError("n_init must be > 0")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if tol < 0:
            raise ValueError("tol must be >= 0")

        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None


    def fit(self, X: np.ndarray):
        X = self._validate_X(X)
        rng_master = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_n_iter = None

        seeds = rng_master.integers(0, 2**32 - 1, size=self.n_init)

        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            centers = self._init_centroids(X, rng)
            labels, inertia, n_iter, centers = self._run_lloyd(X, centers, rng)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = int(best_n_iter)

        return self


    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf")
        return X


    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples, n_features = X.shape
        k = self.n_clusters

        # Controlli di sicurezza
        if k <= 0:
            raise ValueError("n_clusters deve essere positivo.")
        if n_samples == 0:
            raise ValueError("X non può essere vuota.")
        if k > n_samples:
            raise ValueError(
                f"n_clusters={k} non può essere maggiore del numero di campioni ({n_samples})."
            )

        if self.init == "random":
            idx = rng.choice(n_samples, size=k, replace=False)
            return X[idx].copy()

        if self.init == "kmeans++":
            centers = np.empty((k, n_features), dtype=X.dtype)

            # Primo centro scelto uniformemente a caso
            first_idx = rng.integers(0, n_samples)
            centers[0] = X[first_idx]

            # Distanza quadratica minima di ogni punto dal centro più vicino già scelto
            closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)

            for i in range(1, k):
                dist_sum = closest_dist_sq.sum()

                # Caso degenerato: tutti i punti hanno distanza zero dai centroidi già scelti
                # oppure la somma non è numericamente valida
                if dist_sum <= 0 or not np.isfinite(dist_sum):
                    idx = rng.integers(0, n_samples)
                else:
                    probs = closest_dist_sq / dist_sum
                    idx = rng.choice(n_samples, p=probs)

                centers[i] = X[idx]

                dist_sq_new = np.sum((X - centers[i]) ** 2, axis=1)
                closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_new)

            return centers

        raise ValueError(f"Metodo di inizializzazione non supportato: {self.init}")


    @staticmethod
    def _assign_labels(X, centers):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1)

    @staticmethod
    def _compute_inertia(X, centers, labels):
        diffs = X - centers[labels]
        return float(np.sum(diffs * diffs))

    def _recompute_centers(self, X, labels, rng):
        k = self.n_clusters
        centers = np.empty((k, X.shape[1]), dtype=np.float64)

        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = X[mask].mean(axis=0)
            else:
                # Quando un cluster è vuoto, reinizializza casualmente
                centers[j] = X[rng.integers(0, X.shape[0])]

        return centers


    def _run_lloyd(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        rng: np.random.Generator
    ) -> tuple[np.ndarray, float, int, np.ndarray]:
        prev_inertia = None
        # Assegna ogni punto al centro più vicino
        labels = self._assign_labels(X, centers)

        for it in range(1, self.max_iter + 1):  
            # Aggiorna i centroidi
            centers_new = self._recompute_centers(X, labels, rng) 
            # Ricalcola l'assegnamento dei punti (dopo che abbiamo aggiornato i centroidi)
            labels_new = self._assign_labels(X, centers_new)  
            # Inerzia della partizione corrente rispetto ai centroidi aggiornati
            inertia = self._compute_inertia(X, centers_new, labels_new)

            if prev_inertia is not None:
                if abs(prev_inertia - inertia) <= self.tol * max(prev_inertia, 1.0):
                    return labels_new, inertia, it, centers_new

            centers = centers_new
            labels = labels_new
            prev_inertia = inertia

        # Se non converge entro max_iter, restituiamo comunque uno stato finale coerente
        final_inertia = self._compute_inertia(X, centers, labels)
        return labels, final_inertia, self.max_iter, centers