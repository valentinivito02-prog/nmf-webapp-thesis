import pandas as pd
import numpy as np
import re

from src.preprocess import run_preprocess
from src.experiment.find_k import experimental_block_cluster_on_H
from src.interpretation.use_k import run_final_nmf_clustering_on_h
from src.utils.fuzzy_helper import generate_equidistant_fuzzy_sets, fuzzify_matrix_rows
from src.interpretation.fuzzy import (
    describe_latent_factors,
    describe_cluster_representatives,
    describe_samples,
)


def make_readable_label(label):
    label = str(label)

    label = label.replace("_", " ")
    label = label.replace("-", " ")
    label = label.replace("/", " / ")

    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    label = re.sub(r"\s+", " ", label).strip()

    return label.title()

def dataframe_from_store(dataset_data):
    if not dataset_data or "data" not in dataset_data:
        raise ValueError("Dataset not available.")

    df = pd.DataFrame(dataset_data["data"])
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        raise ValueError("The dataset must contain numeric columns.")

    if (numeric_df < 0).any().any():
        raise ValueError("NMF requires non-negative values.")

    return numeric_df


def make_json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [make_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def run_k_experiments(dataset_data, k_min, k_max, init_methods=None):
    df = dataframe_from_store(dataset_data)

    X = df.to_numpy()
    X_scaled = run_preprocess(X)

    k_values = list(range(k_min, k_max + 1))

    if not init_methods:
        init_methods = ["random"]

    all_metrics = []
    all_run_metrics = []
    all_artifacts = {}

    for init_method in init_methods:
        risultati_per_run, risultati_per_k, artifacts = experimental_block_cluster_on_H(
            X_scaled,
            k_values,
            max_rep=5,
            nmf_init=init_method
        )

        risultati_per_k = risultati_per_k.copy()
        risultati_per_run = risultati_per_run.copy()

        risultati_per_k["init"] = init_method
        risultati_per_run["init"] = init_method

        all_metrics.append(risultati_per_k)
        all_run_metrics.append(risultati_per_run)
        all_artifacts[init_method] = make_json_safe(artifacts)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    run_metrics_df = pd.concat(all_run_metrics, ignore_index=True)

    return {
        "metrics": metrics_df,
        "run_metrics": run_metrics_df,
        "artifacts": all_artifacts,
        "selected_inits": init_methods
    }


def run_final_nmf(dataset_data, selected_k, final_init=None):
    df = dataframe_from_store(dataset_data)

    X = df.to_numpy()
    X_scaled = run_preprocess(X)

    artifacts = run_final_nmf_clustering_on_h(
        X_scaled,
        k=int(selected_k),
        seed=42,
        nmf_init=final_init
    )

    if artifacts is None:
        raise ValueError("run_final_nmf_clustering_on_h returned None.")

    if not isinstance(artifacts, dict):
        raise ValueError(f"Expected dict from final NMF, got {type(artifacts)}.")

    safe_artifacts = make_json_safe(artifacts)

    return {
        "nmf_completed": True,
        "selected_k": int(selected_k),
        "selected_init": final_init,
        "artifacts": safe_artifacts,
        "artifacts_keys": list(safe_artifacts.keys()),

        "W": safe_artifacts.get("W"),
        "H": safe_artifacts.get("H"),
        "W_norm": safe_artifacts.get("W_norm"),
        "H_norm": safe_artifacts.get("H_norm"),

        "clusters": {
            "argmax": safe_artifacts.get("labels_argmax", []),
            "kmeans": safe_artifacts.get("labels_kmeans", []),
            "fcm": safe_artifacts.get("labels_fcm_hard", []),
        },

        "metrics": safe_artifacts.get("metrics", {}),
        "cluster_sizes": safe_artifacts.get("cluster_sizes", {}),
        "config": safe_artifacts.get("config", {}),
    }

def run_fuzzy_from_nmf_results(nmf_results, dataset_data=None, n_fuzzy_sets=3, targets=None):
    if not nmf_results or not nmf_results.get("nmf_completed"):
        raise ValueError("Final NMF results are not available.")

    if targets is None:
        targets = ["W", "H"]

    artifacts = nmf_results.get("artifacts", {})
    config = nmf_results.get("config", {})
    selected_k = int(nmf_results.get("selected_k", config.get("k", 2)))

    # =========================
    # DATASET LABELS
    # =========================

    feature_names = None
    sample_names_all = None

    if dataset_data and "data" in dataset_data:
        df = pd.DataFrame(dataset_data["data"])

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        non_numeric_columns = df.select_dtypes(exclude=["number"]).columns.tolist()

        feature_names = [
            make_readable_label(col)
            for col in numeric_columns
        ]

        if non_numeric_columns:
            sample_names_all = df[non_numeric_columns[0]].astype(str).tolist()
        else:
            sample_names_all = [f"Sample {i + 1}" for i in range(len(df))]

    universe = np.arange(0.0, 1.01, 0.01)
    fuzzy_sets = generate_equidistant_fuzzy_sets(
        universe=universe,
        n_sets=int(n_fuzzy_sets)
    )

    results = {
        "fuzzy_completed": True,
        "n_fuzzy_sets": int(n_fuzzy_sets),
        "targets": targets,
        "w_descriptions": [],
        "h_descriptions": [],
        "sample_descriptions": [],
        "w_fuzzy_table": [],
        "h_fuzzy_tables": {},
        "sample_fuzzy_table": [],
    }

    # =========================
    # MATRIX W
    # =========================

    if "W" in targets:
        W_norm = nmf_results.get("W_norm") or artifacts.get("W_norm")

        if W_norm is None:
            raise ValueError("W_norm not found in NMF results.")

        W_norm = np.asarray(W_norm, dtype=float)

        if not feature_names or len(feature_names) != W_norm.shape[0]:
            feature_names = [f"Feature {i + 1}" for i in range(W_norm.shape[0])]

        lf_names = [f"LF{i + 1}" for i in range(W_norm.shape[1])]

        w_by_lf = W_norm.T

        df_fuzzy_w = fuzzify_matrix_rows(
            matrix=w_by_lf,
            row_labels=lf_names,
            col_labels=feature_names,
            x_universe=universe,
            fuzzy_sets=fuzzy_sets,
        )

        results["w_fuzzy_table"] = (
            df_fuzzy_w.reset_index()
            .rename(columns={"index": "Latent Factor"})
            .to_dict("records")
        )

        results["w_descriptions"] = describe_latent_factors(df_fuzzy_w)

    # =========================
    # MATRIX H
    # =========================

    if "H" in targets:
        representatives = artifacts.get("representatives", {})

        if not representatives:
            raise ValueError("Representative vectors not found in NMF results.")

        for method_name, rep_matrix in representatives.items():
            rep_matrix = np.asarray(rep_matrix, dtype=float)

            cluster_names = [f"Cluster {i + 1}" for i in range(rep_matrix.shape[0])]
            lf_names = [f"LF{i + 1}" for i in range(rep_matrix.shape[1])]

            df_fuzzy_rep = fuzzify_matrix_rows(
                matrix=rep_matrix,
                row_labels=cluster_names,
                col_labels=lf_names,
                x_universe=universe,
                fuzzy_sets=fuzzy_sets,
            )

            results["h_fuzzy_tables"][method_name] = (
                df_fuzzy_rep.reset_index()
                .rename(columns={"index": "Cluster"})
                .to_dict("records")
            )

            results["h_descriptions"].extend(
                [f"[{method_name}] {desc}" for desc in describe_cluster_representatives(df_fuzzy_rep)]
            )

        H_norm = nmf_results.get("H_norm") or artifacts.get("H_norm")

        if H_norm is not None:
            H_norm = np.asarray(H_norm, dtype=float)
            h_samples = H_norm.T

            n_samples = min(10, h_samples.shape[0])
            sample_matrix = h_samples[:n_samples]

            if not sample_names_all or len(sample_names_all) < n_samples:
                sample_names = [f"Sample {i + 1}" for i in range(n_samples)]
            else:
                sample_names = sample_names_all[:n_samples]

            lf_names = [f"LF{i + 1}" for i in range(sample_matrix.shape[1])]

            df_fuzzy_samples = fuzzify_matrix_rows(
                matrix=sample_matrix,
                row_labels=sample_names,
                col_labels=lf_names,
                x_universe=universe,
                fuzzy_sets=fuzzy_sets,
            )

            results["sample_fuzzy_table"] = (
                df_fuzzy_samples.reset_index()
                .rename(columns={"index": "Sample"})
                .to_dict("records")
            )

            results["sample_descriptions"] = describe_samples(df_fuzzy_samples)

    return results