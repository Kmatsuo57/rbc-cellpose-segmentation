from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile
from imageio.v3 import imread
from skimage import color, exposure


SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def discover_images(root: Path) -> list[Path]:
    """Return supported image files recursively, in deterministic order."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def read_image(path: Path) -> np.ndarray:
    arr = tifffile.imread(path) if path.suffix.lower() in {".tif", ".tiff"} else imread(path)
    return np.asarray(arr)


def squeeze_to_2d_or_channels(arr: np.ndarray) -> np.ndarray:
    """Remove singleton axes while rejecting stacks/ambiguous high-dimensional data."""
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        return arr[..., :3] if arr.shape[-1] == 4 else arr
    raise ValueError(
        f"Expected one 2D plane or channels-last RGB(A), got shape {arr.shape}. "
        "Export a representative 2D plane before running 2D instance segmentation."
    )


def to_gray(arr: np.ndarray, channel: int | None = None) -> np.ndarray:
    arr = squeeze_to_2d_or_channels(arr)
    if arr.ndim == 2:
        return arr
    if channel is not None:
        if not 0 <= channel < arr.shape[-1]:
            raise ValueError(f"Channel {channel} is outside 0..{arr.shape[-1] - 1}")
        return arr[..., channel]
    return color.rgb2gray(arr[..., :3])


def normalize_percentile(arr: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    """Percentile-normalize spatial axes independently for each channel."""
    arr = np.asarray(arr, dtype=np.float32)
    axes = (0, 1)
    lo = np.percentile(arr, pmin, axis=axes, keepdims=True)
    hi = np.percentile(arr, pmax, axis=axes, keepdims=True)
    return np.clip((arr - lo) / np.maximum(hi - lo, np.finfo(np.float32).eps), 0, 1)


def preview_uint8(arr: np.ndarray) -> np.ndarray:
    arr = squeeze_to_2d_or_channels(np.asarray(arr))
    if arr.dtype == np.uint8:
        return arr
    if arr.ndim == 2:
        return exposure.rescale_intensity(arr, in_range="image", out_range=np.uint8).astype(np.uint8)
    channels = [
        exposure.rescale_intensity(arr[..., c], in_range="image", out_range=np.uint8)
        for c in range(arr.shape[-1])
    ]
    return np.stack(channels, axis=-1).astype(np.uint8)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unique_stem(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return "__".join(rel.parts)
