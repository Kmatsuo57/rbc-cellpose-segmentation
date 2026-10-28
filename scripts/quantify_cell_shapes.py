#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.stats import mannwhitneyu
from skimage import measure, segmentation

from rbc_segmentation.io_utils import preview_uint8, read_image


METRICS = [
    ("area_px", "Area (px²)"),
    ("equivalent_diameter_px", "Equivalent diameter (px)"),
    ("perimeter_px", "Perimeter (px)"),
    ("circularity", "Circularity"),
    ("aspect_ratio", "Aspect ratio"),
    ("eccentricity", "Eccentricity"),
    ("solidity", "Solidity"),
    ("convexity", "Convexity"),
    ("feret_diameter_max_px", "Maximum Feret diameter (px)"),
    ("extent", "Extent"),
]

CONDITIONS = [("0", "H2O2 0"), ("10", "H2O2 10")]


def shape_rows(labels: np.ndarray, condition_code: str, condition: str, source: Path) -> list[dict[str, object]]:
    regions = sorted(measure.regionprops(labels), key=lambda r: (r.centroid[0], r.centroid[1]))
    rows: list[dict[str, object]] = []
    height, width = labels.shape
    for number, region in enumerate(regions, start=1):
        y0, x0, y1, x1 = region.bbox
        perimeter = float(region.perimeter)
        convex_image = region.image_convex if hasattr(region, "image_convex") else region.convex_image
        convex_perimeter = float(measure.perimeter(convex_image))
        minor = float(region.axis_minor_length)
        row = {
            "cell_id": f"H2O2_{condition_code}_{number:03d}",
            "cell_number": number,
            "condition": condition,
            "source_image": source.name,
            "instance_label": int(region.label),
            "centroid_x_px": round(float(region.centroid[1]), 3),
            "centroid_y_px": round(float(region.centroid[0]), 3),
            "touches_border": bool(y0 == 0 or x0 == 0 or y1 == height or x1 == width),
            "area_px": round(float(region.area), 3),
            "equivalent_diameter_px": round(float(region.equivalent_diameter_area), 6),
            "perimeter_px": round(perimeter, 6),
            "circularity": round(float(4 * math.pi * region.area / (perimeter * perimeter)), 6) if perimeter else "",
            "major_axis_length_px": round(float(region.axis_major_length), 6),
            "minor_axis_length_px": round(minor, 6),
            "aspect_ratio": round(float(region.axis_major_length / minor), 6) if minor else "",
            "eccentricity": round(float(region.eccentricity), 6),
            "solidity": round(float(region.solidity), 6),
            "convexity": round(float(convex_perimeter / perimeter), 6) if perimeter else "",
            "feret_diameter_max_px": round(float(region.feret_diameter_max), 6),
            "extent": round(float(region.extent), 6),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numbered_overlay(original: np.ndarray, labels: np.ndarray, rows: list[dict[str, object]], path: Path) -> None:
    image = preview_uint8(original)
    boundary = segmentation.find_boundaries(labels, mode="outer")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    display = image.copy()
    display[boundary] = (255, 40, 40)
    height, width = labels.shape
    fig, ax = plt.subplots(figsize=(width / 150, height / 150), dpi=150)
    ax.imshow(display)
    for row in rows:
        ax.text(
            float(row["centroid_x_px"]), float(row["centroid_y_px"]), f'{int(row["cell_number"]):03d}',
            ha="center", va="center", fontsize=5.5, fontweight="bold", color="yellow",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def cliff_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (a.size * b.size))


def summaries_and_stats(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary: list[dict[str, object]] = []
    stats: list[dict[str, object]] = []
    valid = [row for row in rows if not row["touches_border"]]
    for key, label in METRICS:
        groups: dict[str, np.ndarray] = {}
        for _, condition in CONDITIONS:
            values = np.asarray([float(row[key]) for row in valid if row["condition"] == condition], dtype=float)
            groups[condition] = values
            summary.append({
                "metric": key, "metric_label": label, "condition": condition,
                "n_non_border": values.size, "mean": np.mean(values), "std": np.std(values, ddof=1),
                "median": np.median(values), "q1": np.percentile(values, 25), "q3": np.percentile(values, 75),
                "min": np.min(values), "max": np.max(values),
            })
        a, b = (groups[condition] for _, condition in CONDITIONS)
        test = mannwhitneyu(a, b, alternative="two-sided")
        stats.append({
            "metric": key, "metric_label": label,
            "n_H2O2_0_non_border": a.size, "n_H2O2_10_non_border": b.size,
            "median_H2O2_0": np.median(a), "median_H2O2_10": np.median(b),
            "median_change_percent_10_vs_0": (np.median(b) / np.median(a) - 1) * 100 if np.median(a) else "",
            "mann_whitney_u": test.statistic, "p_value_exploratory": test.pvalue,
            "cliffs_delta_0_vs_10": cliff_delta(a, b),
        })
    return summary, stats


def distribution_figure(rows: list[dict[str, object]], path_base: Path) -> None:
    valid = [row for row in rows if not row["touches_border"]]
    rng = np.random.default_rng(20261028)
    fig, axes = plt.subplots(3, 4, figsize=(16, 11.5))
    colors = ["#2878B5", "#D95319"]
    for ax, (key, label) in zip(axes.flat, METRICS):
        groups = [np.asarray([float(row[key]) for row in valid if row["condition"] == condition]) for _, condition in CONDITIONS]
        violin = ax.violinplot(groups, positions=[1, 2], widths=0.72, showextrema=False)
        for body, color in zip(violin["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.25)
        box = ax.boxplot(groups, positions=[1, 2], widths=0.3, patch_artist=True, showfliers=False, medianprops={"color": "black"})
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for position, values, color in zip([1, 2], groups, colors):
            x = position + rng.uniform(-0.09, 0.09, size=values.size)
            ax.scatter(x, values, s=13, color=color, alpha=0.65, edgecolors="none")
        ax.set_xticks([1, 2], ["H2O2 0", "H2O2 10"])
        ax.set_title(label, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
    for ax in axes.flat[len(METRICS):]:
        ax.axis("off")
    fig.suptitle("RBC shape comparison (non-border cells)", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.012, "Dots are individual cells; box = IQR and median; violin = distribution", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(path_base.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(path_base.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify and compare RBC shape metrics from Cellpose labels.")
    parser.add_argument("--predictions", type=Path, default=Path("outputs/cellpose_shape_comparison/cpsam_v2/rgb"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/shape_quantification"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for code, condition in CONDITIONS:
        label_path = args.predictions / code / "labels.tif"
        source_path = args.raw / f"{code}.jpg"
        if not label_path.exists() or not source_path.exists():
            raise SystemExit(f"Missing {label_path} or {source_path}")
        labels = tifffile.imread(label_path)
        original = read_image(source_path)
        rows = shape_rows(labels, code, condition, source_path)
        all_rows.extend(rows)
        write_csv(args.output / f"cell_shapes_H2O2_{code}.csv", rows)
        numbered_overlay(original, labels, rows, args.output / f"numbered_original_H2O2_{code}.png")
    write_csv(args.output / "cell_shapes_all.csv", all_rows)
    summary, stats = summaries_and_stats(all_rows)
    write_csv(args.output / "shape_summary.csv", summary)
    write_csv(args.output / "shape_statistics_exploratory.csv", stats)
    distribution_figure(all_rows, args.output / "shape_metrics_comparison")
    print(f"Wrote {len(all_rows)} cell records to {args.output}")


if __name__ == "__main__":
    main()
