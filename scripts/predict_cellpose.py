#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tifffile
from cellpose import models
from skimage import color, measure, segmentation, transform

from rbc_segmentation.io_utils import (
    discover_images, normalize_percentile, preview_uint8, read_image,
    squeeze_to_2d_or_channels, to_gray, unique_stem, write_json,
)


def prepare_input(original: np.ndarray, mode: str) -> np.ndarray:
    original = squeeze_to_2d_or_channels(original)
    if mode == "rgb":
        if original.ndim == 2:
            return np.repeat(original[..., None], 3, axis=-1)
        if original.shape[-1] == 1:
            return np.repeat(original, 3, axis=-1)
        return original[..., :3]
    gray = to_gray(original).astype(np.float32)
    lo, hi = float(gray.min()), float(gray.max())
    gray = (gray - lo) / max(hi - lo, np.finfo(np.float32).eps)
    if mode == "invert_gray":
        gray = 1.0 - gray
    return np.repeat(gray[..., None], 3, axis=-1)


def object_rows(labels: np.ndarray) -> list[dict[str, object]]:
    regions = list(measure.regionprops(labels))
    rows = []
    for r in regions:
        y0, x0, y1, x1 = r.bbox
        rows.append({
            "label": int(r.label), "area_px": int(r.area),
            "equivalent_diameter_px": round(float(r.equivalent_diameter_area), 3),
            "solidity": round(float(r.solidity), 4),
            "centroid_y": round(float(r.centroid[0]), 3),
            "centroid_x": round(float(r.centroid[1]), 3),
            "touches_border": y0 == 0 or x0 == 0 or y1 == labels.shape[0] or x1 == labels.shape[1],
        })
    return rows


def save_overlay(original: np.ndarray, labels: np.ndarray, path: Path) -> None:
    base = preview_uint8(original)
    if base.ndim == 2:
        base = color.gray2rgb(base)
    overlay = segmentation.mark_boundaries(base, labels, color=(1, 0, 0), mode="thick")
    iio.imwrite(path, (np.clip(overlay, 0, 1) * 255).astype(np.uint8))


def main() -> None:
    p = argparse.ArgumentParser(description="Compare Cellpose-SAM modes on RBC images.")
    p.add_argument("--input", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("outputs/cellpose"))
    p.add_argument("--filename", action="append", help="Only process this basename; may be repeated.")
    p.add_argument("--model", default="cpsam_v2")
    p.add_argument("--mode", action="append", choices=("rgb", "gray", "invert_gray"))
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--diameter", type=float, help="Optional expected cell diameter in pixels.")
    p.add_argument("--min-size", type=int, default=100)
    p.add_argument("--gpu", action="store_true", help="Use MPS/CUDA when available.")
    p.add_argument("--tile-norm", type=int, default=0, help="Cellpose tile normalization block size; 0 disables it.")
    p.add_argument("--scale", type=float, default=1.0, help="Inference scale; labels are restored to source resolution.")
    args = p.parse_args()
    images = discover_images(args.input)
    if args.filename:
        selected = set(args.filename)
        images = [path for path in images if path.name in selected]
    if not images:
        raise SystemExit(f"No supported images found in {args.input}")
    if not 0 < args.scale <= 1:
        p.error("--scale must be > 0 and <= 1")
    modes = args.mode or ["rgb", "gray", "invert_gray"]
    model = models.CellposeModel(gpu=args.gpu, pretrained_model=args.model)
    for mode in modes:
        for path in images:
            original = squeeze_to_2d_or_channels(read_image(path))
            model_input = prepare_input(original, mode)
            inference_input = model_input
            if args.scale != 1:
                target = (round(model_input.shape[0] * args.scale), round(model_input.shape[1] * args.scale), model_input.shape[2])
                inference_input = transform.resize(model_input, target, order=1, preserve_range=True, anti_aliasing=True)
            labels_small, flows, _ = model.eval(
                inference_input, channel_axis=-1,
                normalize={"normalize": True, "percentile": [1, 99], "tile_norm_blocksize": args.tile_norm},
                diameter=args.diameter, flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
                min_size=max(15, round(args.min_size * args.scale * args.scale)),
            )
            labels = labels_small
            if args.scale != 1:
                labels = transform.resize(labels_small, original.shape[:2], order=0, preserve_range=True, anti_aliasing=False).astype(labels_small.dtype)
            stem = unique_stem(path, args.input)
            out = args.output / args.model / mode / stem
            out.mkdir(parents=True, exist_ok=True)
            iio.imwrite(out / "original_preview.png", preview_uint8(original))
            normalized = normalize_percentile(inference_input, 1, 99)
            iio.imwrite(out / "normalized_preview.png", (normalized * 255).astype(np.uint8))
            tifffile.imwrite(out / "labels.tif", labels.astype(np.uint32), compression="zlib")
            if isinstance(flows, (list, tuple)) and len(flows) > 2 and np.asarray(flows[2]).ndim == 2:
                tifffile.imwrite(out / "cell_probability.tif", np.asarray(flows[2], dtype=np.float32), compression="zlib")
            save_overlay(original, labels, out / "overlay.png")
            rows = object_rows(labels)
            fields = list(rows[0]) if rows else ["label", "area_px", "equivalent_diameter_px"]
            with (out / "objects.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            diameters = [float(r["equivalent_diameter_px"]) for r in rows]
            write_json(out / "summary.json", {
                "source": str(path), "model": args.model, "mode": mode,
                "source_shape": original.shape, "detected_instances": int(labels.max()),
                "median_equivalent_diameter_px": float(np.median(diameters)) if diameters else None,
                "cellprob_threshold": args.cellprob_threshold, "flow_threshold": args.flow_threshold,
                "requested_diameter": args.diameter, "min_size": args.min_size,
                "tile_norm_blocksize": args.tile_norm, "gpu_requested": args.gpu,
                "inference_scale": args.scale,
            })
            print(f"{args.model}/{mode}: {path} -> {labels.max()} objects")


if __name__ == "__main__":
    main()
