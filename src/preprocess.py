import numpy as np

def run_preprocess(X_clean) :
    from src.settings import ScalingMethods

    if ScalingMethods.MIN_MAX == True :
        # Preprocessing: Min-Max per feature su X_clean (campioni x feature)
        X_scaled, feat_min, feat_max = minmax_scale_features(X_clean)
        xmin, xmax = check_minmax_scaled(X_scaled)

        print('\nPreprocessing completato (Min-Max per feature)')
        print('   Dimensioni di X dopo scaling:', X_scaled.shape)
        print(f'   Min globale dopo scaling: {xmin:.6g}')
        print(f'   Max globale dopo scaling: {xmax:.6g}')
        return X_scaled
    #elif ScalingMethod.XXX == True :



def minmax_scale_features(X: np.ndarray, eps: float = 1e-12):
    '''
    Min-Max scaling per feature (colonne):
      X_scaled[:, j] = (X[:, j] - min_j) / (max_j - min_j)

    Ritorna:
      X_scaled: array float, stessa shape di X
      min_: min per feature (shape: n_features,)
      max_: max per feature (shape: n_features,)
    '''
    X = np.asarray(X, dtype=float)

    min_ = X.min(axis=0)
    max_ = X.max(axis=0)

    denom = (max_ - min_)
    denom = np.where(denom < eps, 1.0, denom)  # evita divisione per 0 (feature costanti)

    X_scaled = (X - min_) / denom
    return X_scaled, min_, max_


def check_minmax_scaled(X_scaled: np.ndarray, tol: float = 1e-9):
    '''
    Controlli rapidi:
    - non negatività
    - massimo <= 1 (entro tolleranza numerica)
    '''
    X_scaled = np.asarray(X_scaled, dtype=float)
    xmin = X_scaled.min()
    xmax = X_scaled.max()
    if xmin < -tol:
        raise ValueError(f"X_scaled contiene valori negativi: min={xmin}")
    if xmax > 1.0 + tol:
        raise ValueError(f"X_scaled contiene valori > 1: max={xmax}")
    return xmin, xmax