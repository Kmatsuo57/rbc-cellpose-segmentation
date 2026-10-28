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
from skimage import color, measure, transform

from rbc_segmentation.io_utils import preview_uint8, read_image


CONDITIONS = (("0", "H2O2 0"), ("10", "H2O2 10"))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))


def probability_images(original: np.ndarray, raw_cellprob: np.ndarray, condition: str, output: Path) -> None:
    height, width = original.shape[:2]
    resized = transform.resize(
        raw_cellprob, (height, width), order=1,
        preserve_range=True, anti_aliasing=True,
    ).astype(np.float32)
    probability = sigmoid(resized)
    cmap = matplotlib.colormaps["turbo"]

    fig, ax = plt.subplots(figsize=(width / 150, height / 150), dpi=150)
    image = ax.imshow(probability, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(f"Cellpose cell probability — {condition}", fontsize=15, fontweight="bold")
    ax.axis("off")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.027, pad=0.02)
    colorbar.set_label("Cell probability (sigmoid of Cellpose score)")
    fig.savefig(output / f"cell_probability_{condition.replace(' ', '_')}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    base = preview_uint8(original)
    if base.ndim == 2:
        base = color.gray2rgb(base)
    heat_rgb = (cmap(probability)[..., :3] * 255).astype(np.uint8)
    alpha = (0.15 + 0.55 * probability)[..., None]
    overlay = ((1 - alpha) * base[..., :3] + alpha * heat_rgb).astype(np.uint8)
    iio.imwrite(output / f"cell_probability_overlay_{condition.replace(' ', '_')}.png", overlay)
    tifffile.imwrite(output / f"cell_probability_0to1_{condition.replace(' ', '_')}.tif", probability, compression="zlib")


def square_crop(original: np.ndarray, centroid_x: float, centroid_y: float, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if size <= 0 or size % 2:
        raise ValueError("crop size must be a positive even number")
    rgb = preview_uint8(original)
    if rgb.ndim == 2:
        rgb = color.gray2rgb(rgb)
    center_x, center_y = int(round(centroid_x)), int(round(centroid_y))
    half = size // 2
    desired_x0, desired_y0 = center_x - half, center_y - half
    desired_x1, desired_y1 = desired_x0 + size, desired_y0 + size
    x0, y0 = max(0, desired_x0), max(0, desired_y0)
    x1, y1 = min(rgb.shape[1], desired_x1), min(rgb.shape[0], desired_y1)
    crop = rgb[y0:y1, x0:x1, :3]
    pad_left, pad_top = x0 - desired_x0, y0 - desired_y0
    pad_right, pad_bottom = desired_x1 - x1, desired_y1 - y1
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
    if crop.shape != (size, size, 3):
        raise RuntimeError(f"Unexpected crop shape {crop.shape}")
    return crop, (int(x0), int(y0), int(x1), int(y1))


def contact_sheet(crops: list[tuple[str, np.ndarray]], condition: str, output: Path) -> None:
    columns = 8
    rows = math.ceil(len(crops) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.7, rows * 1.9))
    axes = np.atleast_1d(axes).ravel()
    for ax, (cell_id, rgba) in zip(axes, crops):
        ax.imshow(rgba)
        ax.set_title(cell_id.rsplit("_", 1)[-1], fontsize=8, fontweight="bold")
        ax.axis("off")
    for ax in axes[len(crops):]:
        ax.axis("off")
    fig.suptitle(f"Isolated Cellpose instances — {condition}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export colorful Cellpose probability maps and per-cell transparent crops.")
    parser.add_argument("--predictions", type=Path, default=Path("outputs/cellpose_shape_comparison/cpsam_v2/rgb"))
    parser.add_argument("--measurements", type=Path, default=Path("outputs/shape_quantification/cell_shapes_all.csv"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cell_visuals"))
    parser.add_argument("--crop-size", type=int, default=96, help="Even square crop size in source-image pixels.")
    args = parser.parse_args()
    probability_dir = args.output / "cell_probability"
    crops_dir = args.output / "cells"
    probability_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    with args.measurements.open(newline="", encoding="utf-8") as handle:
        measurements = list(csv.DictReader(handle))
    by_condition = {condition: [row for row in measurements if row["condition"] == condition] for _, condition in CONDITIONS}
    index_rows: list[dict[str, object]] = []

    for code, condition in CONDITIONS:
        source_path = args.raw / f"{code}.jpg"
        result_dir = args.predictions / code
        original = read_image(source_path)
        labels = tifffile.imread(result_dir / "labels.tif")
        raw_cellprob = tifffile.imread(result_dir / "cell_probability.tif")
        probability_images(original, raw_cellprob, condition, probability_dir)
        crops: list[tuple[str, np.ndarray]] = []
        for row in by_condition[condition]:
            instance_label = int(row["instance_label"])
            crop, (x0, y0, x1, y1) = square_crop(
                original, float(row["centroid_x_px"]), float(row["centroid_y_px"]), args.crop_size,
            )
            filename = f'{row["cell_id"]}.png'
            iio.imwrite(crops_dir / filename, crop)
            crops.append((row["cell_id"], crop))
            index_rows.append({
                "cell_id": row["cell_id"], "condition": condition,
                "source_image": source_path.name, "instance_label": instance_label,
                "crop_file": str(crops_dir / filename),
                "crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1,
                "crop_size_px": args.crop_size,
                "touches_border": row["touches_border"],
            })
        contact_sheet(crops, condition, args.output / f"cell_crops_contact_sheet_H2O2_{code}.png")

    with (args.output / "cell_crops_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Wrote {len(index_rows)} cell crops and probability maps to {args.output}")


if __name__ == "__main__":
    main()
