from pathlib import Path
from datetime import datetime
from typing import Optional, List, Union

import pandas as pd

from src.settings import DATA_DIR


def list_chunk_paths(chunk_glob: Optional[Union[str, Path]] = None) -> List[Path]:
    # Ritorna la lista dei file chunk ordinati.
    # crea caso default se la funzione viene chiamata senza parametri (invece di forzare la firma della funzione)
    if chunk_glob is None:
        chunk_glob = DATA_DIR / "radiomics" / "chunk_*.csv"

    chunk_glob = Path(chunk_glob)

    directory = chunk_glob.parent
    pattern = chunk_glob.name

    def chunk_index(p: Path) -> int:
        try:
            return int(p.stem.split('_')[-1])
        except ValueError:
            return 10**9

    paths = sorted(directory.glob(pattern), key=chunk_index)

    if not paths:
        raise FileNotFoundError(f"Nessun file trovato con glob: {chunk_glob}")

    return paths


def save_original_classes(y_class, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "sample": [f"sample_{i}" for i in range(len(y_class))],
        "original_class": y_class,
    })
    df.to_csv(csv_dir / "original_classes.csv", index=False)