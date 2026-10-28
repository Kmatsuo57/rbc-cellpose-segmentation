#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rbc_segmentation.io_utils import read_image, squeeze_to_2d_or_channels


def main() -> None:
    p = argparse.ArgumentParser(description="Validate image/instance-label pairs before custom training.")
    p.add_argument("--images", type=Path, default=Path("data/annotations/images"))
    p.add_argument("--labels", type=Path, default=Path("data/annotations/labels"))
    args = p.parse_args()
    label_paths = sorted([*args.labels.glob("*.tif"), *args.labels.glob("*.tiff")])
    if not label_paths:
        raise SystemExit(f"No TIFF labels found in {args.labels}")
    errors: list[str] = []
    for label_path in label_paths:
        candidates = [p for p in args.images.iterdir() if p.is_file() and p.stem == label_path.stem]
        if len(candidates) != 1:
            errors.append(f"{label_path.name}: expected exactly one same-stem image, found {len(candidates)}")
            continue
        image = squeeze_to_2d_or_channels(read_image(candidates[0]))
        label = np.squeeze(read_image(label_path))
        if label.ndim != 2 or label.shape != image.shape[:2]:
            errors.append(f"{label_path.name}: label {label.shape} vs image {image.shape}")
        if not np.issubdtype(label.dtype, np.integer) or np.min(label) < 0:
            errors.append(f"{label_path.name}: labels must be non-negative integers")
        ids = np.unique(label)
        positive = ids[ids > 0]
        if len(positive) and not np.array_equal(positive, np.arange(1, int(positive.max()) + 1)):
            errors.append(f"{label_path.name}: positive IDs are not contiguous (relabel before training)")
    if errors:
        raise SystemExit("Annotation validation failed:\n- " + "\n- ".join(errors))
    print(f"Validated {len(label_paths)} image/label pairs")


if __name__ == "__main__":
    main()
