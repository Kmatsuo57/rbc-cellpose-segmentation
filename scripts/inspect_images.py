#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile
from skimage import filters, measure, morphology

from rbc_segmentation.io_utils import discover_images, read_image, squeeze_to_2d_or_channels, to_gray


def estimate_diameter(gray: np.ndarray) -> tuple[float | None, int, str]:
    """Conservative rough estimate from thresholded, near-circular non-border objects."""
    x = np.asarray(gray, dtype=np.float32)
    lo, hi = np.percentile(x, (1, 99))
    if hi <= lo:
        return None, 0, "constant image"
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    threshold = filters.threshold_otsu(x)
    best: tuple[list[float], str] = ([], "")
    for mask, polarity in ((x > threshold, "bright"), (x < threshold, "dark")):
        mask = morphology.remove_small_objects(mask, max_size=15)
        labels = measure.label(mask)
        diameters: list[float] = []
        for r in measure.regionprops(labels):
            if r.area < 16 or r.area > 0.05 * x.size:
                continue
            y0, x0, y1, x1 = r.bbox
            if y0 == 0 or x0 == 0 or y1 == x.shape[0] or x1 == x.shape[1]:
                continue
            circularity = 4 * np.pi * r.area / max(r.perimeter ** 2, 1)
            if circularity >= 0.45 and r.solidity >= 0.75:
                diameters.append(float(r.equivalent_diameter_area))
        if len(diameters) > len(best[0]):
            best = (diameters, polarity)
    if len(best[0]) < 3:
        return None, len(best[0]), "insufficient reliable candidates"
    return float(np.median(best[0])), len(best[0]), f"rough {best[1]}-object estimate"


def metadata(path: Path) -> dict[str, object]:
    arr = read_image(path)
    axes = "unknown"
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            axes = tif.series[0].axes
    row: dict[str, object] = {
        "file": str(path), "format": path.suffix.lower(), "shape": "x".join(map(str, arr.shape)),
        "axes": axes, "dtype": str(arr.dtype), "min": float(np.min(arr)), "max": float(np.max(arr)),
        "channels": arr.shape[-1] if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4) else 1,
        "diameter_px_estimate": "", "diameter_candidates": 0, "diameter_note": "",
    }
    try:
        plane = squeeze_to_2d_or_channels(arr)
        diameter, count, note = estimate_diameter(to_gray(plane))
        row.update(diameter_px_estimate="" if diameter is None else round(diameter, 2), diameter_candidates=count, diameter_note=note)
    except ValueError as exc:
        row["diameter_note"] = str(exc)
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Inventory RBC image metadata and estimate cell diameter.")
    p.add_argument("--input", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("outputs/image_inventory.csv"))
    args = p.parse_args()
    images = discover_images(args.input)
    if not images:
        raise SystemExit(f"No supported images found in {args.input}")
    rows = [metadata(path) for path in images]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} image records to {args.output}")


if __name__ == "__main__":
    main()
