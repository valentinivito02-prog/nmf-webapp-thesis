from __future__ import annotations
from typing import Iterable, Optional, List, Tuple, Union
from pathlib import Path

import pandas as pd
import numpy as np

from src.settings import FEATURES, DATA_DIR
from src.utils.load_helper import list_chunk_paths


def run_load() :
    # Carica i dati (csv)
    chunk_paths = list_chunk_paths(DATA_DIR / 'radiomics' / 'chunk_*.csv')   # se DATA_DIR / 'radiomics' / '1_Tabular_Dataset' / '1_XY_train.csv' è TEMPORANEAMENTE così! | dev'essere: DATA_DIR / 'radiomics' / 'chunk_*.csv') | caricare un dataset completo ma più piccolo ha senso per sperimentazione
    X_clean, y_class = load_radiomics_dataset(chunk_paths, keep_features=FEATURES)
    
    # Controlli di sicurezza
    if X_clean.ndim != 2:
        raise ValueError(
            f"X_clean non è 2D: shape={X_clean.shape}, ndim={X_clean.ndim}"
        )
    if y_class.ndim != 1:
        raise ValueError(
            f"y_class non è 1D: shape={y_class.shape}, ndim={y_class.ndim}"
        )
    if X_clean.shape[0] != len(y_class):
        raise ValueError(
            f"Incoerenza tra campioni e label: X_clean ha {X_clean.shape[0]} righe, y_class ha {len(y_class)} elementi"
        )

    return X_clean, y_class



def load_radiomics_dataset(
    csv_paths: Union[str, Path, Iterable[Union[str, Path]]],
    keep_features: Optional[List[str]] = None,
    label_col: str = 'Classe',
    dedup_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Carica uno o più file CSV radiomics e restituisce:
    - X_clean: matrice feature (n_samples, n_features)
    - y_class: classi (n_samples,) usate solo ex-post

    Parametri:
    - csv_paths: path singolo o lista/generator di path
    - keep_features: whitelist di feature da tenere (se None tiene tutte le colonne feature)
    - label_col: nome colonna label (default 'Classe')
    - dedup_cols: colonne su cui deduplicare (es. ['ID_Paziente','ID']) se presenti
    '''

    if isinstance(csv_paths, (str, Path)):
        paths = [Path(csv_paths)]
    else:
        paths = [Path(p) for p in csv_paths]

    if not paths:
        raise ValueError('csv_paths è vuoto.')

    frames = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f'File non trovato: {p}')
        frames.append(pd.read_csv(p))

    df = pd.concat(frames, ignore_index=True)

    if label_col not in df.columns:
        raise ValueError(f"Colonna label '{label_col}' non trovata nel dataset.")

    # Rimuoviamo colonne non-feature (ID, diagnostics, augmented, label)
    non_feature_cols = {label_col, 'ID', 'ID_Paziente', 'augmented'}
    columns_to_drop = [c for c in df.columns if c.lower().startswith('diagnostics')]
    columns_to_drop += [c for c in df.columns if c in non_feature_cols]

    # Se dedup_cols è None, proviamo a deduplicare automaticamente se ci sono colonne ID note.
    if dedup_cols is None:
        auto = []
        if 'ID_Paziente' in df.columns:
            auto.append('ID_Paziente')
        if 'ID' in df.columns:
            auto.append('ID')
        dedup_cols = auto if auto else None

    if dedup_cols:
        present = [c for c in dedup_cols if c in df.columns]
        if len(present) == len(dedup_cols):
            df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    # Costruiamo y_class e df_features dopo l’eventuale deduplicazione
    y_class = df[label_col].to_numpy()
    df_features = df.drop(columns=list(set(columns_to_drop)), errors='ignore')

    # Se fornita una whitelist, teniamo solo quelle colonne.
    if keep_features is not None:
        missing = [c for c in keep_features if c not in df_features.columns]
        if missing:
            raise ValueError(f'Mancano nel CSV le seguenti feature richieste: {missing}')
        df_features = df_features[keep_features]

    # Gestione NaN: rimuoviamo i campioni incompleti
    if df_features.isnull().any().any():
        print('Attenzione: presenti valori NaN. Verranno rimossi i campioni incompleti.')
        good_idx = ~df_features.isnull().any(axis=1)
        df_features = df_features.loc[good_idx].reset_index(drop=True)
        y_class = y_class[good_idx.to_numpy()]

    # Controlla se X_clean riporta solo valori numerici
    if not all(pd.api.types.is_numeric_dtype(df_features[c]) for c in df_features.columns):
        raise ValueError("Sono presenti colonne non numeriche tra le feature.")

    X_clean = df_features.to_numpy()

    print('\nDataset caricato correttamente')
    print(f'   Campioni: {X_clean.shape[0]}')
    print(f'   Feature:  {X_clean.shape[1]}')

    return X_clean, y_class
