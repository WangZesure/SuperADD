"""Debug utilities: save intermediate pipeline results as images for inspection.

When config.debug=True, the pipeline calls save_debug_images() during predict,
writing raw distance maps, fused map, and binary mask to disk.
"""

from pathlib import Path

import cv2
import numpy as np
import torch


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a float array to 0-255 uint8 for visualization."""
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def save_debug_images(
    debug_dir: Path,
    basename: str,
    raw_maps: dict,
    fused: torch.Tensor,
    anomaly_map: np.ndarray,
    binary_result: np.ndarray,
):
    """Save intermediate results for a single test image.

    Layout:
        {debug_dir}/{basename}/
        ├── raw_s{scale}_l{layer}.png   — per-scale per-layer distance maps
        ├── fused.png                    — fused continuous anomaly map
        └── binary.png                   — final binary mask
    """
    out_dir = Path(debug_dir) / basename
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw maps
    for (scale, layer), dists in raw_maps.items():
        arr = dists.cpu().numpy() if isinstance(dists, torch.Tensor) else dists
        cv2.imwrite(
            str(out_dir / f'raw_s{scale}_l{layer}.png'),
            _normalize_to_uint8(arr),
        )

    # Fused continuous map
    fused_np = fused.cpu().numpy() if isinstance(fused, torch.Tensor) else fused
    cv2.imwrite(str(out_dir / 'fused.png'), _normalize_to_uint8(fused_np))

    # Binary mask
    cv2.imwrite(str(out_dir / 'binary.png'), binary_result)