#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import cdist


METRICS = [
    "area_px", "equivalent_diameter_px", "perimeter_px", "circularity",
    "aspect_ratio", "eccentricity", "solidity", "convexity",
    "feret_diameter_max_px", "extent",
]

METRIC_LABELS = [
    "Area", "Eq. diameter", "Perimeter", "Circularity", "Aspect ratio",
    "Eccentricity", "Solidity", "Convexity", "Max Feret", "Extent",
]


def load_rows(path: Path, include_border: bool) -> tuple[list[dict[str, str]], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not include_border:
        rows = [row for row in rows if row["touches_border"].lower() != "true"]
    values = np.asarray([[float(row[key]) for key in METRICS] for row in rows], dtype=float)
    return rows, values


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    return (values - mean) / std, mean, std


def pca(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    scores = z @ vt.T
    variance = singular ** 2 / (z.shape[0] - 1)
    explained = variance / variance.sum()
    return scores, explained, vt.T


def kmeans(z: np.ndarray, k: int, seed: int = 20261028, starts: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed + k)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(starts):
        centers = z[rng.choice(z.shape[0], size=k, replace=False)].copy()
        labels = np.zeros(z.shape[0], dtype=int)
        for _ in range(300):
            distances = cdist(z, centers)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _ > 0:
                break
            labels = new_labels
            for cluster in range(k):
                members = z[labels == cluster]
                if members.size:
                    centers[cluster] = members.mean(axis=0)
                else:
                    centers[cluster] = z[np.argmax(distances.min(axis=1))]
        inertia = float(np.sum((z - centers[labels]) ** 2))
        if best is None or inertia < best[2]:
            best = labels.copy(), centers.copy(), inertia
    if best is None:
        raise RuntimeError("k-means failed")
    return best


def silhouette(z: np.ndarray, labels: np.ndarray) -> float:
    distances = cdist(z, z)
    scores = []
    for index, cluster in enumerate(labels):
        same = labels == cluster
        same[index] = False
        a = distances[index, same].mean() if same.any() else 0.0
        other_means = [distances[index, labels == other].mean() for other in np.unique(labels) if other != cluster]
        b = min(other_means)
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(scores))


def adjusted_rand_index(labels: np.ndarray, conditions: np.ndarray) -> float:
    clusters = np.unique(labels)
    groups = np.unique(conditions)
    table = np.asarray([[np.sum((labels == c) & (conditions == g)) for g in groups] for c in clusters], dtype=int)
    comb2 = lambda n: n * (n - 1) / 2
    sum_cells = sum(comb2(int(n)) for n in table.flat)
    sum_rows = sum(comb2(int(n)) for n in table.sum(axis=1))
    sum_cols = sum(comb2(int(n)) for n in table.sum(axis=0))
    total = comb2(table.sum())
    expected = sum_rows * sum_cols / total
    maximum = 0.5 * (sum_rows + sum_cols)
    return float((sum_cells - expected) / (maximum - expected)) if maximum != expected else 1.0


def binary_purity(labels: np.ndarray, conditions: np.ndarray) -> float:
    unique_clusters = np.unique(labels)
    unique_conditions = np.unique(conditions)
    if len(unique_clusters) != 2 or len(unique_conditions) != 2:
        return float("nan")
    direct = np.sum((labels == unique_clusters[0]) & (conditions == unique_conditions[0])) + np.sum((labels == unique_clusters[1]) & (conditions == unique_conditions[1]))
    reverse = np.sum((labels == unique_clusters[0]) & (conditions == unique_conditions[1])) + np.sum((labels == unique_clusters[1]) & (conditions == unique_conditions[0]))
    return float(max(direct, reverse) / labels.size)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pca(rows: list[dict[str, str]], scores: np.ndarray, explained: np.ndarray, labels: np.ndarray, silhouettes: dict[int, float], path: Path) -> None:
    conditions = np.asarray([row["condition"] for row in rows])
    condition_colors = {"H2O2 0": "#2878B5", "H2O2 10": "#D95319"}
    cluster_colors = ["#4DAF4A", "#984EA3"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for condition, color in condition_colors.items():
        mask = conditions == condition
        axes[0, 0].scatter(scores[mask, 0], scores[mask, 1], s=45, color=color, alpha=0.8, label=condition)
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("PCA colored by condition")
    for cluster in np.unique(labels):
        mask = labels == cluster
        axes[0, 1].scatter(scores[mask, 0], scores[mask, 1], s=45, color=cluster_colors[cluster], alpha=0.8, label=f"Cluster {cluster + 1}")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].set_title("PCA colored by blind k-means (k=2)")
    for ax in axes[0]:
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        ax.axhline(0, color="0.85", linewidth=0.8)
        ax.axvline(0, color="0.85", linewidth=0.8)
    ks = sorted(silhouettes)
    axes[1, 0].plot(ks, [silhouettes[k] for k in ks], marker="o", color="#333333")
    axes[1, 0].set_xticks(ks)
    axes[1, 0].set_xlabel("Number of k-means clusters")
    axes[1, 0].set_ylabel("Silhouette score")
    axes[1, 0].set_title("Cluster-number scan")
    cumulative = np.cumsum(explained) * 100
    axes[1, 1].bar(np.arange(1, len(explained) + 1), explained * 100, color="#7AA6C2", label="Individual")
    axes[1, 1].plot(np.arange(1, len(explained) + 1), cumulative, marker="o", color="#D95319", label="Cumulative")
    axes[1, 1].set_xlabel("Principal component")
    axes[1, 1].set_ylabel("Explained variance (%)")
    axes[1, 1].set_title("PCA explained variance")
    axes[1, 1].legend(frameon=False)
    fig.suptitle("Unsupervised mapping of 10 RBC shape metrics", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def plot_heatmap(rows: list[dict[str, str]], z: np.ndarray, tree: np.ndarray, labels: np.ndarray, path: Path) -> None:
    order = leaves_list(tree)
    conditions = np.asarray([0 if row["condition"] == "H2O2 0" else 1 for row in rows])
    fig = plt.figure(figsize=(13, 13))
    grid = fig.add_gridspec(3, 1, height_ratios=[0.18, 0.18, 8], hspace=0.04)
    ax_condition = fig.add_subplot(grid[0])
    ax_cluster = fig.add_subplot(grid[1])
    ax_heat = fig.add_subplot(grid[2])
    ax_condition.imshow(conditions[order][None, :], aspect="auto", cmap=ListedColormap(["#2878B5", "#D95319"]), vmin=0, vmax=1)
    ax_condition.set_ylabel("Condition", rotation=0, ha="right", va="center")
    ax_cluster.imshow(labels[order][None, :], aspect="auto", cmap=ListedColormap(["#4DAF4A", "#984EA3"]), vmin=0, vmax=1)
    ax_cluster.set_ylabel("k-means", rotation=0, ha="right", va="center")
    for ax in (ax_condition, ax_cluster):
        ax.set_xticks([])
        ax.set_yticks([])
    image = ax_heat.imshow(z[order].T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax_heat.set_yticks(range(len(METRIC_LABELS)), METRIC_LABELS)
    ax_heat.set_xlabel("Cells ordered by Ward hierarchical clustering")
    fig.colorbar(image, ax=ax_heat, label="Standardized value (z-score)", shrink=0.6)
    fig.suptitle("Shape-feature heatmap", fontsize=17, fontweight="bold")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_dendrogram(rows: list[dict[str, str]], tree: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(24, 8))
    dendrogram(tree, labels=[row["cell_id"] for row in rows], leaf_rotation=90, leaf_font_size=5, ax=ax, color_threshold=None)
    ax.set_ylabel("Ward distance")
    ax.set_title("Hierarchical clustering of standardized shape metrics")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def plot_cluster_montage(rows: list[dict[str, str]], labels: np.ndarray, cluster: int, raw_dir: Path, path: Path) -> None:
    selected = [(row, label) for row, label in zip(rows, labels) if label == cluster]
    columns = 4
    row_count = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(row_count, columns, figsize=(columns * 3, row_count * 3))
    axes = np.atleast_1d(axes).ravel()
    image_cache: dict[str, np.ndarray] = {}
    radius = 55
    for ax, (row, _) in zip(axes, selected):
        filename = row["source_image"]
        if filename not in image_cache:
            image_cache[filename] = iio.imread(raw_dir / filename)
        image = image_cache[filename]
        x, y = int(round(float(row["centroid_x_px"]))), int(round(float(row["centroid_y_px"])))
        y0, y1 = max(0, y - radius), min(image.shape[0], y + radius)
        x0, x1 = max(0, x - radius), min(image.shape[1], x + radius)
        ax.imshow(image[y0:y1, x0:x1])
        ax.set_title(row["cell_id"], fontsize=10, fontweight="bold")
        ax.axis("off")
    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.suptitle(f"Cells assigned to k-means cluster {cluster + 1}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map and cluster RBC shape metrics without using condition labels.")
    parser.add_argument("--input", type=Path, default=Path("outputs/shape_quantification/cell_shapes_all.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/shape_clustering"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--include-border", action="store_true", help="Include truncated cells touching image borders.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, values = load_rows(args.input, args.include_border)
    z, means, stds = standardize(values)
    scores, explained, loadings = pca(z)
    conditions = np.asarray([row["condition"] for row in rows])
    cluster_results = {k: kmeans(z, k) for k in range(2, 7)}
    silhouettes = {k: silhouette(z, result[0]) for k, result in cluster_results.items()}
    labels, centers, inertia = cluster_results[2]
    tree = linkage(z, method="ward")
    ari = adjusted_rand_index(labels, conditions)
    purity = binary_purity(labels, conditions)

    coordinate_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        record: dict[str, object] = {
            "cell_id": row["cell_id"], "cell_number": row["cell_number"],
            "condition": row["condition"], "source_image": row["source_image"],
            "centroid_x_px": row["centroid_x_px"], "centroid_y_px": row["centroid_y_px"],
            "kmeans_cluster": int(labels[index] + 1),
        }
        record.update({f"PC{component + 1}": scores[index, component] for component in range(scores.shape[1])})
        record.update({f"z_{metric}": z[index, metric_index] for metric_index, metric in enumerate(METRICS)})
        coordinate_rows.append(record)
    write_csv(args.output / "cell_coordinates_clusters.csv", coordinate_rows)

    contingency_rows = []
    for cluster in np.unique(labels):
        contingency_rows.append({
            "kmeans_cluster": int(cluster + 1),
            "H2O2_0": int(np.sum((labels == cluster) & (conditions == "H2O2 0"))),
            "H2O2_10": int(np.sum((labels == cluster) & (conditions == "H2O2 10"))),
            "total": int(np.sum(labels == cluster)),
        })
    write_csv(args.output / "cluster_condition_contingency.csv", contingency_rows)

    cluster_summary_rows = []
    for cluster in np.unique(labels):
        for metric_index, metric in enumerate(METRICS):
            cluster_values = values[labels == cluster, metric_index]
            cluster_summary_rows.append({
                "kmeans_cluster": int(cluster + 1), "metric": metric,
                "n": cluster_values.size, "mean": np.mean(cluster_values),
                "median": np.median(cluster_values), "standard_deviation": np.std(cluster_values, ddof=1),
                "q1": np.percentile(cluster_values, 25), "q3": np.percentile(cluster_values, 75),
            })
    write_csv(args.output / "cluster_feature_summary.csv", cluster_summary_rows)

    quality_rows = [{
        "k": k, "silhouette_score": silhouettes[k],
        "inertia": cluster_results[k][2],
        "adjusted_rand_vs_condition": adjusted_rand_index(cluster_results[k][0], conditions),
        "binary_purity_vs_condition": binary_purity(cluster_results[k][0], conditions) if k == 2 else "",
    } for k in sorted(cluster_results)]
    write_csv(args.output / "clustering_quality.csv", quality_rows)
    write_csv(args.output / "feature_standardization.csv", [
        {"metric": metric, "mean": means[i], "standard_deviation": stds[i]}
        for i, metric in enumerate(METRICS)
    ])
    write_csv(args.output / "pca_loadings.csv", [
        {"metric": metric, **{f"PC{i + 1}_loading": loadings[j, i] for i in range(loadings.shape[1])}}
        for j, metric in enumerate(METRICS)
    ])
    write_csv(args.output / "pca_explained_variance.csv", [
        {"component": f"PC{i + 1}", "explained_variance_ratio": value, "cumulative_ratio": np.sum(explained[:i + 1])}
        for i, value in enumerate(explained)
    ])
    plot_pca(rows, scores, explained, labels, silhouettes, args.output / "pca_and_clustering")
    plot_heatmap(rows, z, tree, labels, args.output / "clustered_feature_heatmap")
    plot_dendrogram(rows, tree, args.output / "hierarchical_dendrogram")
    plot_cluster_montage(rows, labels, 1, args.raw, args.output / "cluster_2_cell_montage")
    print(f"Analyzed {len(rows)} cells; k=2 silhouette={silhouettes[2]:.4f}, ARI={ari:.4f}, purity={purity:.4f}, inertia={inertia:.3f}")


if __name__ == "__main__":
    main()
