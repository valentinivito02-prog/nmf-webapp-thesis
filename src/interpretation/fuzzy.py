from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.fuzzy_helper import (
    generate_equidistant_fuzzy_sets,
    fuzzify_matrix_rows,
)


def run_fuzzy_interpretation(
    results_dir: str | Path,
    *,
    output_dir: str | Path,
    n_fuzzy_sets: int = 5,
    n_random_samples: int = 10,
) -> Dict[str, Any]:
    '''
    Post-processing fuzzy indipendente sugli output di use_k.py.

    Regole fuzzy su:
    - fattori latenti da W_normalized.T
    - representative vectors dei cluster nello spazio di H
    - sample casuali e riproducibili da H_transposed_samples_by_lf
    '''
    from src.settings import relative_to_project

    loaded = load_use_k_outputs(results_dir)

    config = loaded["config"]
    metrics_df = loaded["metrics_df"]
    w_norm_df = loaded["W_normalized_df"]
    h_samples_df = loaded["H_samples_df"]
    representatives = loaded["representatives"]
    cluster_sizes = loaded["cluster_sizes"]

    seed = int(config["seed"])
    k = int(config["k"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    universe = np.arange(0.0, 1.01, 0.01)
    fuzzy_sets = generate_equidistant_fuzzy_sets(universe=universe, n_sets=n_fuzzy_sets)

    '''
    Latent factors from W
    W_normalized is features x k
    Transpose is used to inspect each latent factor across features.
    '''
    w_by_lf_df = w_norm_df.T.copy()
    w_by_lf_df.index = [f"LF{i+1}" for i in range(w_by_lf_df.shape[0])]

    df_fuzzy_w = fuzzify_matrix_rows(
        matrix=w_by_lf_df.to_numpy(dtype=float),
        row_labels=list(w_by_lf_df.index),
        col_labels=list(w_by_lf_df.columns),
        x_universe=universe,
        fuzzy_sets=fuzzy_sets,
    )
    latent_factor_descriptions = describe_latent_factors(df_fuzzy_w)


    # Cluster representatives from H space
    representative_fuzzy_tables: dict[str, pd.DataFrame] = {}
    representative_descriptions: dict[str, list[str]] = {}

    for method_name, rep_df in representatives.items():
        rep_numeric_df = rep_df.copy()
        rep_numeric_df.index = [f"Cluster {i+1}" for i in range(rep_numeric_df.shape[0])]
        rep_numeric_df.columns = [f"LF{i+1}" for i in range(rep_numeric_df.shape[1])]

        df_fuzzy_rep = fuzzify_matrix_rows(
            matrix=rep_numeric_df.to_numpy(dtype=float),
            row_labels=list(rep_numeric_df.index),
            col_labels=list(rep_numeric_df.columns),
            x_universe=universe,
            fuzzy_sets=fuzzy_sets,
        )

        representative_fuzzy_tables[method_name] = df_fuzzy_rep
        representative_descriptions[method_name] = describe_cluster_representatives(df_fuzzy_rep)


    # Random reproducible sample examples from H space
    selected_samples_df = select_random_samples(
        h_samples_df=h_samples_df,
        n_samples_to_select=n_random_samples,
        seed=seed,
    )

    selected_samples_numeric_df = selected_samples_df.copy()
    selected_samples_numeric_df.columns = [f"LF{i+1}" for i in range(selected_samples_numeric_df.shape[1])]

    df_fuzzy_samples = fuzzify_matrix_rows(
        matrix=selected_samples_numeric_df.to_numpy(dtype=float),
        row_labels=list(selected_samples_numeric_df.index),
        col_labels=list(selected_samples_numeric_df.columns),
        x_universe=universe,
        fuzzy_sets=fuzzy_sets,
    )
    sample_descriptions = describe_samples(df_fuzzy_samples)


    # Comparison across cluster methods
    comparison_df = build_cluster_methods_comparison(
        metrics_df=metrics_df,
        cluster_sizes=cluster_sizes,
    )


    # Save outputs
    df_fuzzy_w.to_csv(output_dir / "latent_factors_fuzzy.csv")
    _write_lines(output_dir / "latent_factors_descriptions.txt", latent_factor_descriptions)

    for method_name, df_rep in representative_fuzzy_tables.items():
        df_rep.to_csv(output_dir / f"representatives_{method_name}_fuzzy.csv")
        _write_lines(
            output_dir / f"representatives_{method_name}_descriptions.txt",
            representative_descriptions[method_name],
        )

    selected_samples_numeric_df.to_csv(output_dir / "selected_random_samples_numeric.csv")
    df_fuzzy_samples.to_csv(output_dir / "selected_random_samples_fuzzy.csv")
    _write_lines(output_dir / "selected_random_samples_descriptions.txt", sample_descriptions)

    comparison_df.to_csv(output_dir / "cluster_methods_comparison.csv", index=False)

    results_dir_path = Path(results_dir)
    results_dir_str = relative_to_project(results_dir_path)

    summary_lines = [
        "Fuzzy interpretation summary",
        f"results_dir = {results_dir_str}",
        f"k = {k}",
        f"seed = {seed}",
        f"n_fuzzy_sets = {n_fuzzy_sets}",
        f"n_random_samples = {len(selected_samples_numeric_df)}",
        "",
        "Generated files:",
        "- latent_factors_fuzzy.csv",
        "- latent_factors_descriptions.txt",
        "- representatives_argmax_fuzzy.csv",
        "- representatives_argmax_descriptions.txt",
        "- representatives_kmeans_fuzzy.csv",
        "- representatives_kmeans_descriptions.txt",
        "- representatives_fcm_hard_fuzzy.csv",
        "- representatives_fcm_hard_descriptions.txt",
        "- selected_random_samples_numeric.csv",
        "- selected_random_samples_fuzzy.csv",
        "- selected_random_samples_descriptions.txt",
        "- cluster_methods_comparison.csv",
    ]
    _write_lines(output_dir / "summary.txt", summary_lines)

    return {
        "config": config,
        "latent_factors_fuzzy": df_fuzzy_w,
        "latent_factor_descriptions": latent_factor_descriptions,
        "representatives_fuzzy": representative_fuzzy_tables,
        "representatives_descriptions": representative_descriptions,
        "selected_samples_numeric": selected_samples_numeric_df,
        "selected_samples_fuzzy": df_fuzzy_samples,
        "selected_samples_descriptions": sample_descriptions,
        "cluster_methods_comparison": comparison_df,
        "output_dir": output_dir,
    }



def _read_csv_with_index(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    return pd.read_csv(path, index_col=0)


def _read_csv_no_index(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    return pd.read_csv(path)


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def _safe_float_from_metrics(metrics_df: pd.DataFrame, column: str) -> float:
    if metrics_df.empty or column not in metrics_df.columns:
        return float("nan")
    return float(metrics_df.iloc[0][column])


def _load_cluster_sizes(csv_dir: Path, method_name: str) -> pd.DataFrame:
    return _read_csv_no_index(csv_dir / f"cluster_sizes_{method_name}.csv")


def load_use_k_outputs(results_dir: str | Path) -> Dict[str, Any]:
    # Legge gli output già salvati da use_k.py.
    results_dir = Path(results_dir)
    csv_dir = results_dir / "csv"

    if not csv_dir.exists():
        raise FileNotFoundError(f"Cartella CSV non trovata: {csv_dir}")

    config_df = _read_csv_no_index(csv_dir / "config.csv")
    metrics_df = _read_csv_no_index(csv_dir / "metrics.csv")

    w_norm_df = _read_csv_with_index(csv_dir / "W_normalized.csv")
    h_samples_df = _read_csv_with_index(csv_dir / "H_transposed_samples_by_lf.csv")

    rep_argmax_df = _read_csv_with_index(csv_dir / "representative_vectors_argmax.csv")
    rep_kmeans_df = _read_csv_with_index(csv_dir / "representative_vectors_kmeans.csv")
    rep_fcm_hard_df = _read_csv_with_index(csv_dir / "representative_vectors_fcm_hard.csv")

    cluster_sizes_argmax_df = _load_cluster_sizes(csv_dir, "argmax")
    cluster_sizes_kmeans_df = _load_cluster_sizes(csv_dir, "kmeans")
    cluster_sizes_fcm_hard_df = _load_cluster_sizes(csv_dir, "fcm_hard")

    return {
        "results_dir": results_dir,
        "csv_dir": csv_dir,
        "config": config_df.iloc[0].to_dict(),
        "metrics_df": metrics_df,
        "W_normalized_df": w_norm_df,
        "H_samples_df": h_samples_df,
        "representatives": {
            "argmax": rep_argmax_df,
            "kmeans": rep_kmeans_df,
            "fcm_hard": rep_fcm_hard_df,
        },
        "cluster_sizes": {
            "argmax": cluster_sizes_argmax_df,
            "kmeans": cluster_sizes_kmeans_df,
            "fcm_hard": cluster_sizes_fcm_hard_df,
        },
    }


def select_random_samples(
    h_samples_df: pd.DataFrame,
    n_samples_to_select: int,
    seed: int,
) -> pd.DataFrame:
    # Estrae sample casuali ma riproducibili da H_transposed_samples_by_lf.csv.
    if n_samples_to_select <= 0:
        raise ValueError("n_samples_to_select deve essere > 0.")

    n_total = len(h_samples_df)
    n_select = min(n_samples_to_select, n_total)

    rng = np.random.default_rng(int(seed))
    selected_positions = rng.choice(n_total, size=n_select, replace=False)

    return h_samples_df.iloc[selected_positions].copy()


def describe_latent_factors(df_fuzzy_w: pd.DataFrame) -> list[str]:
    descriptions: list[str] = []

    for lf_name, row in df_fuzzy_w.iterrows():
        parts = [f"{value} for {feature_name}" for feature_name, value in row.items()]
        descriptions.append(f"The influence of {lf_name} is: " + ", ".join(parts) + ".")

    return descriptions


def describe_cluster_representatives(df_fuzzy_rep: pd.DataFrame) -> list[str]:
    descriptions: list[str] = []

    for cluster_name, row in df_fuzzy_rep.iterrows():
        if (row == "Empty cluster").all():
            descriptions.append(
                f"{cluster_name} is empty and cannot be characterized linguistically."
            )
            continue

        parts = [f"{fuzzy_value} values of {lf_name}" for lf_name, fuzzy_value in row.items()]
        descriptions.append(
            f"Samples in {cluster_name} are generally characterized by "
            + ", and ".join(parts)
            + "."
        )

    return descriptions


def describe_samples(df_fuzzy_samples: pd.DataFrame) -> list[str]:
    descriptions: list[str] = []

    for sample_name, row in df_fuzzy_samples.iterrows():
        parts = [f"{fuzzy_value} for {lf_name}" for lf_name, fuzzy_value in row.items()]
        descriptions.append(f"The influence on {sample_name} is: " + ", ".join(parts) + ".")

    return descriptions


def build_cluster_methods_comparison(
    metrics_df: pd.DataFrame,
    cluster_sizes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    sil_argmax = _safe_float_from_metrics(metrics_df, "silhouette_argmax")
    sil_kmeans = _safe_float_from_metrics(metrics_df, "silhouette_kmeans")
    sil_fcm = _safe_float_from_metrics(metrics_df, "silhouette_fcm")

    rows = []

    for method_name, size_df in cluster_sizes.items():
        sizes = size_df["size"].to_numpy(dtype=int)

        if method_name == "argmax":
            silhouette = sil_argmax
        elif method_name == "kmeans":
            silhouette = sil_kmeans
        elif method_name == "fcm_hard":
            silhouette = sil_fcm
        else:
            silhouette = float("nan")

        rows.append(
            {
                "method": method_name,
                "silhouette": silhouette,
                "empty_clusters": int(np.sum(sizes == 0)),
                "min_cluster_size": int(np.min(sizes)),
                "max_cluster_size": int(np.max(sizes)),
                "cluster_sizes": ", ".join(map(str, sizes.tolist())),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["silhouette", "empty_clusters"], ascending=[False, True]).reset_index(drop=True)
