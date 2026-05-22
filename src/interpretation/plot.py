from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.settings import PROJECT_DIR


# Basic I/O helpers
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


def _relative_path_str(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_DIR))
    except Exception:
        return str(path)


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()



# Loading inputs
def load_plot_inputs(
    results_dir: str | Path,
    fuzzy_dir: str | Path,
) -> Dict[str, Any]:
    results_dir = Path(results_dir)
    csv_dir = results_dir / "csv"
    fuzzy_dir = Path(fuzzy_dir)

    if not csv_dir.exists():
        raise FileNotFoundError(f"Cartella CSV non trovata: {csv_dir}")

    config_df = _read_csv_no_index(csv_dir / "config.csv")
    metrics_df = _read_csv_no_index(csv_dir / "metrics.csv")

    w_norm_df = _read_csv_with_index(csv_dir / "W_normalized.csv")
    h_samples_df = _read_csv_with_index(csv_dir / "H_transposed_samples_by_lf.csv")
    membership_fcm_df = _read_csv_with_index(csv_dir / "membership_fcm.csv")

    rep_argmax_df = _read_csv_with_index(csv_dir / "representative_vectors_argmax.csv")
    rep_kmeans_df = _read_csv_with_index(csv_dir / "representative_vectors_kmeans.csv")
    rep_fcm_hard_df = _read_csv_with_index(csv_dir / "representative_vectors_fcm_hard.csv")

    cluster_sizes_argmax_df = _read_csv_no_index(csv_dir / "cluster_sizes_argmax.csv")
    cluster_sizes_kmeans_df = _read_csv_no_index(csv_dir / "cluster_sizes_kmeans.csv")
    cluster_sizes_fcm_hard_df = _read_csv_no_index(csv_dir / "cluster_sizes_fcm_hard.csv")

    labels_argmax_df = _read_csv_no_index(csv_dir / "labels_argmax.csv")
    labels_kmeans_df = _read_csv_no_index(csv_dir / "labels_kmeans.csv")
    labels_fcm_hard_df = _read_csv_no_index(csv_dir / "labels_fcm_hard.csv")

    original_classes_path = csv_dir / "original_classes.csv"
    original_classes_df = (
        _read_csv_no_index(original_classes_path)
        if original_classes_path.exists()
        else None
    )

    selected_random_samples_path = fuzzy_dir / "selected_random_samples_numeric.csv"
    selected_random_samples_df = (
        _read_csv_with_index(selected_random_samples_path)
        if selected_random_samples_path.exists()
        else None
    )

    return {
        "results_dir": results_dir,
        "config": config_df.iloc[0].to_dict(),
        "metrics_df": metrics_df,
        "W_normalized_df": w_norm_df,
        "H_samples_df": h_samples_df,
        "membership_fcm_df": membership_fcm_df,
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
        "labels": {
            "argmax": labels_argmax_df,
            "kmeans": labels_kmeans_df,
            "fcm_hard": labels_fcm_hard_df,
        },
        "original_classes_df": original_classes_df,
        "selected_random_samples_df": selected_random_samples_df,
    }



# Alignment helpers
def align_grouping_labels_to_samples(
    h_samples_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    sample_col: str,
    group_col: str,
) -> pd.Series:
    # Allinea una colonna di gruppi ai sample di h_samples_df usando il nome sample.
    if sample_col not in labels_df.columns:
        raise ValueError(f"labels_df deve contenere la colonna '{sample_col}'.")
    if group_col not in labels_df.columns:
        raise ValueError(f"labels_df deve contenere la colonna '{group_col}'.")

    mapping_df = labels_df[[sample_col, group_col]].copy()
    mapping_df[sample_col] = mapping_df[sample_col].astype(str)
    mapping_df = mapping_df.drop_duplicates(subset=[sample_col]).set_index(sample_col)

    sample_index = h_samples_df.index.astype(str)
    missing = [sample for sample in sample_index if sample not in mapping_df.index]

    if missing:
        raise ValueError(
            "Alcuni sample di H_transposed_samples_by_lf.csv non hanno una label associata. "
            f"Esempi mancanti: {missing[:10]}"
        )

    aligned = mapping_df.loc[sample_index, group_col]
    aligned.index = h_samples_df.index
    return aligned.astype(str)


def split_dataframe_into_parts(
    df: pd.DataFrame,
    n_parts: int,
) -> list[pd.DataFrame]:
    if n_parts <= 0:
        raise ValueError("n_parts deve essere > 0.")

    n_rows = len(df)
    if n_rows == 0:
        return [df.copy()]

    indices = np.array_split(np.arange(n_rows), n_parts)
    return [df.iloc[idx].copy() for idx in indices if len(idx) > 0]



# Generic plots
def plot_heatmap(
    data: pd.DataFrame,
    title: str,
    output_path: Path,
    xlabel: str,
    ylabel: str,
    annotate: bool = False,
) -> None:
    fig, ax = plt.subplots(
        figsize=(max(8, data.shape[1] * 0.8), max(4, data.shape[0] * 0.6))
    )

    im = ax.imshow(data.to_numpy(dtype=float), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=0, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value")

    if annotate:
        values = data.to_numpy(dtype=float)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    _save_current_figure(output_path)


def plot_cluster_sizes_bar(
    cluster_sizes_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    if "cluster" not in cluster_sizes_df.columns or "size" not in cluster_sizes_df.columns:
        raise ValueError("cluster_sizes_df deve contenere le colonne 'cluster' e 'size'.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cluster_sizes_df["cluster"].astype(str), cluster_sizes_df["size"].astype(int))
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of samples")

    _save_current_figure(output_path)


def plot_selected_samples_heatmap(
    selected_samples_df: pd.DataFrame,
    output_path: Path,
) -> None:
    data = selected_samples_df.copy()
    data.columns = [f"LF{i+1}" for i in range(data.shape[1])]

    plot_heatmap(
        data=data,
        title="Selected random samples in latent space",
        output_path=output_path,
        xlabel="Latent factors",
        ylabel="Selected samples",
        annotate=True,
    )


def plot_fcm_max_membership_histogram(
    membership_fcm_df: pd.DataFrame,
    output_path: Path,
) -> None:
    max_membership = membership_fcm_df.to_numpy(dtype=float).max(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(max_membership, bins=20)
    ax.set_title("FCM maximum membership distribution")
    ax.set_xlabel("Maximum membership per sample")
    ax.set_ylabel("Frequency")

    _save_current_figure(output_path)



# H stacked plots
def plot_h_stacked_grouped(
    h_samples_df: pd.DataFrame,
    grouping_labels: pd.Series,
    output_path: Path,
    title: str,
    group_name: str,
) -> None:
    '''
    Stacked bar plot delle righe di H_transposed_samples_by_lf,
    ordinate per gruppo e sample.
    '''
    if not h_samples_df.index.equals(grouping_labels.index):
        raise ValueError("grouping_labels deve essere allineato all'index di h_samples_df.")

    latent_cols = list(h_samples_df.columns)

    df = h_samples_df.copy()
    df["__group__"] = grouping_labels.astype(str).values
    df["__sample_name__"] = df.index.astype(str)
    df["__dominant_lf__"] = np.argmax(h_samples_df.to_numpy(dtype=float), axis=1)

    df = df.reset_index(drop=True)
    df = df.sort_values(["__group__", "__dominant_lf__", "__sample_name__"]).reset_index(drop=True)

    values = df[latent_cols].to_numpy(dtype=float)
    groups = df["__group__"].tolist()
    sample_names = df["__sample_name__"].tolist()

    fig_width = max(16, len(sample_names) * 0.12)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    x = np.arange(len(sample_names))
    bottoms = np.zeros(len(sample_names), dtype=float)

    colors = plt.get_cmap("tab10").colors

    for col_idx, col_name in enumerate(latent_cols):
        y = values[:, col_idx]
        ax.bar(
            x,
            y,
            bottom=bottoms,
            width=1.0,
            label=col_name,
            color=colors[col_idx % len(colors)],
            linewidth=0.0,
        )
        bottoms += y

    if groups:
        change_positions = []
        prev = groups[0]
        for i, g in enumerate(groups[1:], start=1):
            if g != prev:
                change_positions.append(i)
                prev = g

        for pos in change_positions:
            ax.axvline(pos - 0.5, linestyle="--", linewidth=1.0)

        group_centers = []
        start = 0
        current = groups[0]

        for i in range(1, len(groups) + 1):
            if i == len(groups) or groups[i] != current:
                center = (start + i - 1) / 2.0
                group_centers.append((center, current))
                if i < len(groups):
                    start = i
                    current = groups[i]

        for center, grp in group_centers:
            ax.text(
                center,
                1.02,
                str(grp),
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                fontsize=10,
            )

    ax.set_title(title)
    ax.set_xlabel(group_name)
    ax.set_ylabel("Contribution")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.5, len(sample_names) - 0.5)
    ax.set_xticks([])
    ax.legend(title="Latent factors", bbox_to_anchor=(1.02, 1), loc="upper left")

    _save_current_figure(output_path)


def plot_h_stacked_grouped_in_parts(
    h_samples_df: pd.DataFrame,
    grouping_labels: pd.Series,
    output_dir: Path,
    file_prefix: str,
    title_prefix: str,
    group_name: str,
    n_parts: int = 3,
) -> list[str]:
    # Divide il dataframe ordinato per gruppi in n_parts e salva n_parts plot SVG.
    if not h_samples_df.index.equals(grouping_labels.index):
        raise ValueError("grouping_labels deve essere allineato all'index di h_samples_df.")

    latent_cols = list(h_samples_df.columns)

    df = h_samples_df.copy()
    df["__group__"] = grouping_labels.astype(str).values
    df["__sample_name__"] = df.index.astype(str)
    df["__dominant_lf__"] = np.argmax(h_samples_df.to_numpy(dtype=float), axis=1)

    df = df.reset_index(drop=True)
    df = df.sort_values(["__group__", "__dominant_lf__", "__sample_name__"]).reset_index(drop=True)

    parts = split_dataframe_into_parts(df, n_parts=n_parts)
    generated_files: list[str] = []

    for part_idx, part_df in enumerate(parts, start=1):
        numeric_df = part_df[latent_cols].copy()
        numeric_df.index = part_df["__sample_name__"].astype(str)

        part_grouping = pd.Series(
            part_df["__group__"].astype(str).to_numpy(),
            index=numeric_df.index,
        )

        out_path = output_dir / f"{file_prefix}_part_{part_idx}.svg"

        plot_h_stacked_grouped(
            h_samples_df=numeric_df,
            grouping_labels=part_grouping,
            output_path=out_path,
            title=f"{title_prefix} - part {part_idx}",
            group_name=group_name,
        )

        generated_files.append(out_path.name)

    return generated_files



# H heatmaps and mean profiles
def plot_h_heatmap_sorted(
    h_samples_df: pd.DataFrame,
    grouping_labels: pd.Series,
    output_path: Path,
    title: str,
) -> None:
    # Heatmap di H ordinata per gruppo.
    if not h_samples_df.index.equals(grouping_labels.index):
        raise ValueError("grouping_labels deve essere allineato all'index di h_samples_df.")

    df = h_samples_df.copy()
    df["__group__"] = grouping_labels.astype(str).values
    df["__dominant_lf__"] = np.argmax(h_samples_df.to_numpy(dtype=float), axis=1)
    df["__sample_name__"] = df.index.astype(str)

    df = df.reset_index(drop=True)
    df = df.sort_values(["__group__", "__dominant_lf__", "__sample_name__"]).reset_index(drop=True)

    latent_cols = list(h_samples_df.columns)
    values = df[latent_cols].to_numpy(dtype=float)
    groups = df["__group__"].tolist()

    fig_height = max(6, len(values) / 120)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    im = ax.imshow(values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Latent factors")
    ax.set_ylabel("Samples")

    ax.set_xticks(np.arange(len(latent_cols)))
    ax.set_xticklabels([f"LF{i+1}" for i in range(len(latent_cols))])
    ax.set_yticks([])

    if groups:
        prev = groups[0]
        for i, g in enumerate(groups[1:], start=1):
            if g != prev:
                ax.axhline(i - 0.5, linestyle="--", linewidth=1.0)
                prev = g

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value")

    _save_current_figure(output_path)


def plot_mean_profiles(
    h_samples_df: pd.DataFrame,
    grouping_labels: pd.Series,
    output_path: Path,
    title: str,
    group_name: str,
) -> None:
    # Profili medi dei latent factors per gruppo.
    if not h_samples_df.index.equals(grouping_labels.index):
        raise ValueError("grouping_labels deve essere allineato all'index di h_samples_df.")

    df = h_samples_df.copy()
    df["group"] = grouping_labels.astype(str).values

    mean_profiles = df.groupby("group").mean().sort_index()

    fig, ax = plt.subplots(figsize=(max(8, len(mean_profiles) * 1.2), 5))

    bottoms = np.zeros(len(mean_profiles), dtype=float)
    colors = plt.get_cmap("tab10").colors

    for i, col in enumerate(mean_profiles.columns):
        values = mean_profiles[col].to_numpy(dtype=float)
        ax.bar(
            mean_profiles.index.astype(str),
            values,
            bottom=bottoms,
            label=col,
            color=colors[i % len(colors)],
        )
        bottoms += values

    ax.set_title(title)
    ax.set_ylabel("Mean contribution")
    ax.set_xlabel(group_name)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Latent factors", bbox_to_anchor=(1.02, 1), loc="upper left")

    _save_current_figure(output_path)



# Summary
def build_plot_summary_lines(
    results_dir: Path,
    output_dir: Path,
    generated_files: list[str],
) -> list[str]:
    return [
        "Interpretation plots summary",
        f"results_dir = {_relative_path_str(results_dir)}",
        f"output_dir = {_relative_path_str(output_dir)}",
        "",
        "Generated files:",
        *[f"- {name}" for name in generated_files],
    ]



# Main plotting pipeline
def run_interpretation_plots(
    results_dir: str | Path,
    *,
    output_dir: str | Path,
    fuzzy_dir: str | Path,
):
    print("\nCreando plot(s) interpretativi...")

    loaded = load_plot_inputs(results_dir=results_dir, fuzzy_dir=fuzzy_dir)

    results_dir = Path(results_dir)
    
    fuzzy_dir = Path(fuzzy_dir)
    fuzzy_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    w_norm_df = loaded["W_normalized_df"]
    h_samples_df = loaded["H_samples_df"]
    membership_fcm_df = loaded["membership_fcm_df"]
    representatives = loaded["representatives"]
    cluster_sizes = loaded["cluster_sizes"]
    labels = loaded["labels"]
    original_classes_df = loaded["original_classes_df"]
    selected_random_samples_df = loaded["selected_random_samples_df"]

    generated_files: list[str] = []


    # W heatmap
    w_heatmap_path = output_dir / "W_normalized_heatmap.png"
    plot_heatmap(
        data=w_norm_df,
        title="W normalized heatmap",
        output_path=w_heatmap_path,
        xlabel="Latent factors",
        ylabel="Features",
        annotate=False,
    )
    generated_files.append(_relative_path_str(w_heatmap_path))


    # Representative vectors heatmaps
    for method_name, rep_df in representatives.items():
        rep_data = rep_df.copy()
        rep_data.index = [f"Cluster {i+1}" for i in range(rep_data.shape[0])]
        rep_data.columns = [f"LF{i+1}" for i in range(rep_data.shape[1])]

        out_path = output_dir / f"representatives_{method_name}_heatmap.png"
        plot_heatmap(
            data=rep_data,
            title=f"Representative vectors heatmap - {method_name}",
            output_path=out_path,
            xlabel="Latent factors",
            ylabel="Clusters",
            annotate=True,
        )
        generated_files.append(_relative_path_str(out_path))


    # Cluster sizes bar plots
    for method_name, size_df in cluster_sizes.items():
        out_path = output_dir / f"cluster_sizes_{method_name}.png"
        plot_cluster_sizes_bar(
            cluster_sizes_df=size_df,
            title=f"Cluster sizes - {method_name}",
            output_path=out_path,
        )
        generated_files.append(_relative_path_str(out_path))


    # Selected random samples heatmap
    if selected_random_samples_df is not None:
        out_path = output_dir / "selected_random_samples_heatmap.png"
        plot_selected_samples_heatmap(
            selected_samples_df=selected_random_samples_df,
            output_path=out_path,
        )
        generated_files.append(_relative_path_str(out_path))


    # FCM max membership histogram
    out_path = output_dir / "fcm_max_membership_histogram.png"
    plot_fcm_max_membership_histogram(
        membership_fcm_df=membership_fcm_df,
        output_path=out_path,
    )
    generated_files.append(_relative_path_str(out_path))


    # Stacked H by cluster methods (3 parts each, svg)
    for method_name, labels_df in labels.items():
        grouping_labels = align_grouping_labels_to_samples(
            h_samples_df=h_samples_df,
            labels_df=labels_df,
            sample_col="sample",
            group_col="label",
        )

        method_dir = output_dir / f"stacked_{method_name}"
        method_dir.mkdir(parents=True, exist_ok=True)

        generated = plot_h_stacked_grouped_in_parts(
            h_samples_df=h_samples_df,
            grouping_labels=grouping_labels,
            output_dir=method_dir,
            file_prefix=f"H_stacked_by_{method_name}",
            title_prefix=f"H stacked bar plot grouped by {method_name}",
            group_name=f"{method_name} clusters",
            n_parts=3,
        )
        generated_files.extend([_relative_path_str(method_dir / name) for name in generated])


    # Stacked H by original classes (3 parts, svg)
    if original_classes_df is not None:
        grouping_labels = align_grouping_labels_to_samples(
            h_samples_df=h_samples_df,
            labels_df=original_classes_df,
            sample_col="sample",
            group_col="original_class",
        )

        class_dir = output_dir / "stacked_original_classes"
        class_dir.mkdir(parents=True, exist_ok=True)

        generated = plot_h_stacked_grouped_in_parts(
            h_samples_df=h_samples_df,
            grouping_labels=grouping_labels,
            output_dir=class_dir,
            file_prefix="H_stacked_by_original_class",
            title_prefix="H stacked bar plot grouped by original classes",
            group_name="Original classes",
            n_parts=3,
        )
        generated_files.extend([_relative_path_str(class_dir / name) for name in generated])


    # H heatmaps by cluster methods
    for method_name, labels_df in labels.items():
        grouping_labels = align_grouping_labels_to_samples(
            h_samples_df=h_samples_df,
            labels_df=labels_df,
            sample_col="sample",
            group_col="label",
        )

        out_path = heatmap_dir / f"H_heatmap_by_{method_name}.svg"
        plot_h_heatmap_sorted(
            h_samples_df=h_samples_df,
            grouping_labels=grouping_labels,
            output_path=out_path,
            title=f"H matrix sorted by {method_name}",
        )
        generated_files.append(_relative_path_str(out_path))


    # H heatmap by original classes
    if original_classes_df is not None:
        grouping_labels = align_grouping_labels_to_samples(
            h_samples_df=h_samples_df,
            labels_df=original_classes_df,
            sample_col="sample",
            group_col="original_class",
        )

        out_path = heatmap_dir / "H_heatmap_by_original_class.svg"
        plot_h_heatmap_sorted(
            h_samples_df=h_samples_df,
            grouping_labels=grouping_labels,
            output_path=out_path,
            title="H matrix sorted by original classes",
        )
        generated_files.append(_relative_path_str(out_path))


    # Mean profiles by cluster methods
    for method_name, labels_df in labels.items():
        grouping_labels = align_grouping_labels_to_samples(
            h_samples_df=h_samples_df,
            labels_df=labels_df,
            sample_col="sample",
            group_col="label",
        )

        out_path = output_dir / f"mean_profiles_{method_name}.svg"
        plot_mean_profiles(
            h_samples_df=h_samples_df,
            grouping_labels=grouping_labels,
            output_path=out_path,
            title=f"Mean latent factor profiles by {method_name}",
            group_name="Cluster",
        )
        generated_files.append(_relative_path_str(out_path))


    # Mean profiles by original classes
    if original_classes_df is not None:
        grouping_labels = align_grouping_labels_to_samples(
            h_samples_df=h_samples_df,
            labels_df=original_classes_df,
            sample_col="sample",
            group_col="original_class",
        )

        out_path = output_dir / "mean_profiles_original_classes.svg"
        plot_mean_profiles(
            h_samples_df=h_samples_df,
            grouping_labels=grouping_labels,
            output_path=out_path,
            title="Mean latent factor profiles by original class",
            group_name="Class",
        )
        generated_files.append(_relative_path_str(out_path))


    # Summary
    summary_lines = build_plot_summary_lines(
        results_dir=results_dir,
        output_dir=output_dir,
        generated_files=generated_files,
    )
    _write_lines(output_dir / "summary.txt", summary_lines)

    return {
        "results_dir": results_dir,
        "output_dir": output_dir,
        "generated_files": generated_files,
    }