#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import umap
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score

from scripts.cluster_shape_metrics import METRICS, kmeans, load_rows, standardize


NEIGHBORS = (5, 10, 15, 30)
SEEDS = (42, 1028, 2026, 10001, 26028)
CONDITION_COLORS = {"H2O2 0": "#2878B5", "H2O2 10": "#D95319"}
CLUSTER_COLORS = {0: "#4DAF4A", 1: "#984EA3"}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_umap(z: np.ndarray, n_neighbors: int, seed: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, n_components=2, min_dist=0.1,
        metric="euclidean", random_state=seed, transform_seed=seed, n_jobs=1,
    )
    return reducer.fit_transform(z)


def cluster_neighbor_fraction(embedding: np.ndarray, labels: np.ndarray, target: int = 1, k: int = 5) -> float:
    distances = np.linalg.norm(embedding[:, None, :] - embedding[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    fractions = []
    for index in np.where(labels == target)[0]:
        nearest = np.argsort(distances[index])[:k]
        fractions.append(np.mean(labels[nearest] == target))
    return float(np.mean(fractions))


def parameter_figure(rows: list[dict[str, str]], embeddings: dict[int, np.ndarray], labels: np.ndarray, output: Path) -> None:
    conditions = np.asarray([row["condition"] for row in rows])
    fig, axes = plt.subplots(2, len(NEIGHBORS), figsize=(20, 9))
    for column, n_neighbors in enumerate(NEIGHBORS):
        embedding = embeddings[n_neighbors]
        for condition, color in CONDITION_COLORS.items():
            mask = conditions == condition
            axes[0, column].scatter(embedding[mask, 0], embedding[mask, 1], s=36, color=color, alpha=0.8, label=condition)
        for cluster, color in CLUSTER_COLORS.items():
            mask = labels == cluster
            axes[1, column].scatter(embedding[mask, 0], embedding[mask, 1], s=36, color=color, alpha=0.8, label=f"Cluster {cluster + 1}")
        axes[0, column].set_title(f"n_neighbors = {n_neighbors}", fontweight="bold")
        for row_index in range(2):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            axes[row_index, column].set_xlabel("UMAP 1")
            axes[row_index, column].set_ylabel("UMAP 2")
    axes[0, -1].legend(frameon=False, loc="best")
    axes[1, -1].legend(frameon=False, loc="best")
    fig.text(0.01, 0.73, "Condition", rotation=90, fontsize=14, fontweight="bold", va="center")
    fig.text(0.01, 0.28, "Blind k-means", rotation=90, fontsize=14, fontweight="bold", va="center")
    fig.suptitle("UMAP parameter comparison of 10 shape metrics", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0.025, 0, 1, 0.96))
    fig.savefig(output.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def labeled_figure(rows: list[dict[str, str]], embedding: np.ndarray, labels: np.ndarray, output: Path) -> None:
    conditions = np.asarray([row["condition"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for condition, color in CONDITION_COLORS.items():
        mask = conditions == condition
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1], s=55, color=color, alpha=0.8, label=condition)
    for cluster, color in CLUSTER_COLORS.items():
        mask = labels == cluster
        axes[1].scatter(embedding[mask, 0], embedding[mask, 1], s=55, color=color, alpha=0.8, label=f"Cluster {cluster + 1}")
    for index in np.where(labels == 1)[0]:
        short_id = rows[index]["cell_id"].rsplit("_", 1)[-1]
        for ax in axes:
            ax.annotate(short_id, embedding[index], xytext=(3, 3), textcoords="offset points", fontsize=7, fontweight="bold")
    axes[0].set_title("Colored by condition")
    axes[1].set_title("Colored by k-means; Cluster 2 IDs labeled")
    for ax in axes:
        ax.legend(frameon=False)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("UMAP of standardized shape metrics (n_neighbors=15)", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP mapping and stability analysis of RBC shape metrics.")
    parser.add_argument("--input", type=Path, default=Path("outputs/shape_quantification/cell_shapes_all.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/shape_umap"))
    parser.add_argument("--include-border", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, values = load_rows(args.input, args.include_border)
    z, _, _ = standardize(values)
    labels, _, _ = kmeans(z, 2)
    conditions = np.asarray([row["condition"] for row in rows])
    condition_binary = (conditions == "H2O2 10").astype(int)

    primary_embeddings: dict[int, np.ndarray] = {}
    coordinate_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    for n_neighbors in NEIGHBORS:
        embeddings = [fit_umap(z, n_neighbors, seed) for seed in SEEDS]
        primary_embeddings[n_neighbors] = embeddings[0]
        distance_correlations = []
        for left in range(len(embeddings)):
            for right in range(left + 1, len(embeddings)):
                distance_correlations.append(float(spearmanr(pdist(embeddings[left]), pdist(embeddings[right])).statistic))
        cluster_silhouettes = [float(silhouette_score(embedding, labels)) for embedding in embeddings]
        condition_silhouettes = [float(silhouette_score(embedding, condition_binary)) for embedding in embeddings]
        neighbor_fractions = [cluster_neighbor_fraction(embedding, labels) for embedding in embeddings]
        trust_scores = [float(trustworthiness(z, embedding, n_neighbors=min(10, n_neighbors))) for embedding in embeddings]
        stability_rows.append({
            "n_neighbors": n_neighbors, "min_dist": 0.1, "seeds": ";".join(map(str, SEEDS)),
            "mean_pairwise_distance_rank_correlation": np.mean(distance_correlations),
            "min_pairwise_distance_rank_correlation": np.min(distance_correlations),
            "mean_trustworthiness": np.mean(trust_scores),
            "mean_cluster2_neighbor_fraction_k5": np.mean(neighbor_fractions),
            "min_cluster2_neighbor_fraction_k5": np.min(neighbor_fractions),
            "mean_kmeans_cluster_silhouette_in_umap": np.mean(cluster_silhouettes),
            "mean_condition_silhouette_in_umap": np.mean(condition_silhouettes),
        })
        for index, row in enumerate(rows):
            coordinate_rows.append({
                "cell_id": row["cell_id"], "condition": row["condition"],
                "kmeans_cluster": int(labels[index] + 1), "n_neighbors": n_neighbors,
                "min_dist": 0.1, "random_seed": SEEDS[0],
                "UMAP1": embeddings[0][index, 0], "UMAP2": embeddings[0][index, 1],
            })

    write_csv(args.output / "umap_coordinates.csv", coordinate_rows)
    write_csv(args.output / "umap_stability.csv", stability_rows)
    parameter_figure(rows, primary_embeddings, labels, args.output / "umap_parameter_comparison")
    labeled_figure(rows, primary_embeddings[15], labels, args.output / "umap_n15_labeled")
    print(f"Wrote UMAP results for {len(rows)} cells to {args.output}")


if __name__ == "__main__":
    main()
