from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.settings import relative_to_project


def plot_metrics(risultati_per_k_df: pd.DataFrame, output_dir: str = "plots", dpi: int = 300) -> None:
    print("\nCreando plot(s)...")

    df = risultati_per_k_df.copy().sort_values("k").reset_index(drop=True)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    x = df["k"]

    # Reconstruction Error
    fig, ax = plt.subplots(figsize=(8, 5))

    y = df["reconstruction_error_mean"]
    yerr = df["reconstruction_error_std"]

    ax.plot(x, y, marker="o", label="Reconstruction error mean")
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, label="±1 std")

    ax.set_title("Reconstruction Error vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Reconstruction Error")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path / "reconstruction_error.png", dpi=dpi)
    plt.close(fig)

    # Silhouette
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, mean_col, std_col in [
        ("NMF (Argmax)", "silhouette_argmax_mean", "silhouette_argmax_std"),
        ("NMF + KMeans", "silhouette_kmeans_mean", "silhouette_kmeans_std"),
        ("NMF + FCM", "silhouette_fcm_mean", "silhouette_fcm_std"),
    ]:

        y = df[mean_col]
        yerr = df[std_col]

        ax.plot(x, y, marker="o", label=label)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15)

    ax.set_title("Silhouette vs k")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path / "silhouette.png", dpi=dpi)
    plt.close(fig)

    # Cophenetic (no consensus)
    from src.settings import FCMeans as FCMeansConfig

    if FCMeansConfig.USE_SHARPENING:
        subtitle = f"FCM sharpening for soft modes α = {FCMeansConfig.SHARPEN_ALPHA}"
    else:
        subtitle = "FCM sharpening for soft modes disabled"

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, col in [
        ("NMF (Argmax)", "coph_argmax"),
        ("NMF + KMeans", "coph_kmeans"),
        ("NMF + FCM (hard)", "coph_fcm_hard"),
        ("NMF + FCM (soft dot)", "coph_fcm_soft_dot"),
        ("NMF + FCM (soft cosine)", "coph_fcm_soft_cosine"),
    ]:
        ax.plot(x, df[col], marker="o", label=label)

    ax.set_title(f"Cophenetic coefficient vs k\n{subtitle}")
    ax.set_xlabel("k")
    ax.set_ylabel("Cophenetic Coefficient")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path / "cophenetic.png", dpi=dpi)
    plt.close(fig)

    print(f"\nSalvati in: {relative_to_project(out_path)}")


def plot_consensus_matrices(
    artifacts: dict,
    output_dir: str | Path,
    dpi: int = 300,
) -> None:
    # Crea una heatmap per ogni matrice di consenso e la salva in PNG.
    print("\nCreando plot(s) matrici di consenso...")
    if not artifacts :
        raise ValueError('Gli artefatti passati sono invalidi.')

    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for k in sorted(artifacts.keys()):
        consensus_block = artifacts[k]["consensus"]

        k_dir = output_dir / f"k_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        matrices = {
            "argmax": consensus_block["argmax"],
            "kmeans": consensus_block["kmeans"],
            "fcm_hard": consensus_block["fcm"]["hard"],
            "fcm_soft_dot": consensus_block["fcm"]["soft_dot"],
            "fcm_soft_cosine": consensus_block["fcm"]["soft_cosine"],
        }

        for name, matrix in matrices.items():
            matrix = np.asarray(matrix, dtype=np.float32)  # se è commentata è perché arrivano già in questo formato

            # Riordinamento dei campioni per similarità di consenso
            dist = 1.0 - matrix
            np.fill_diagonal(dist, 0.0)

            Z = linkage(squareform(dist, checks=False), method="average")
            order = leaves_list(Z)

            matrix = matrix[order][:, order]

            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="equal")
            fig.colorbar(im, ax=ax)

            ax.set_title(f"Consensus matrix - k={k} - {name}")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Samples")

            fig.tight_layout()
            fig.savefig(k_dir / f"{name}.png", dpi=dpi)
            plt.close(fig)

    print(f"\nConsensus plots salvati in:\n{relative_to_project(output_dir)}")


# Vecchia funzione di quando il blocco cophenetic faceva una sola mode per il FCM
'''def fcm_plot_label(tag: str) -> str:
    # Convert a technical FCM mode tag (e.g. 'soft_dot_sharp2') into a human-readable plot label.
    tag = str(tag).lower()

    if tag == "hard":
        return "NMF + FCM (hard)"

    parts = tag.split("_")

    # soft_dot_sharp2
    if len(parts) >= 2:
        method = parts[1]
    else:
        method = "?"

    if "sharp" in tag:
        alpha = tag.split("sharp")[-1]
        return f"NMF + FCM (soft {method}, α={alpha})"

    return f"NMF + FCM (soft {method})"'''