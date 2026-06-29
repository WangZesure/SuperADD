"""Normal-only Component Calibration (NCC).

Replaces fixed percentile + morphology with:
  1. Threshold calibrated from normal validation score statistics (no anomaly labels)
  2. Connected-component confidence filtering: remove low-confidence components
     that are likely false positives (small area, low peak, single-scale support)

The threshold is the same percentile-based one from training (passed via
normal_stats), but the post-threshold component filtering is the new
contribution. Component features are computed without any anomaly labels.
"""

import cv2
import numpy as np
from scipy import ndimage

from industrial.post_process import (
    multi_oriented_closing,
    erosion_on_binary_maps,
    fill_closed_regions,
)


def calibrate(
    anomaly_map: np.ndarray,
    normal_stats: dict,
    config: dict,
    **kwargs,
) -> np.ndarray:
    """Binarize anomaly map with calibrated threshold + component filtering.

    Args:
        anomaly_map:  (H, W) float32 continuous anomaly scores
        normal_stats: {'threshold': float, optionally 'normal_component_stats': {...}}
        config:       {closing_radius, closing_angles, closing_lower_threshold,
                      binary_erosion, min_component_area, min_peak_ratio, ...}
        **kwargs:     optional 'scale_support': (H, W) int — # scales supporting each pixel

    Returns:
        (H, W) uint8 binary mask with values {0, 255}
    """
    threshold = normal_stats['threshold']

    # --- Step 1: Morphology (same as baseline) ---
    binary = multi_oriented_closing(
        anomaly_map,
        threshold,
        config['closing_radius'],
        config['closing_angles'],
        config['closing_lower_threshold'],
    )
    binary = fill_closed_regions(binary)
    binary = erosion_on_binary_maps(binary, config['binary_erosion'])

    # --- Step 2: Component confidence filtering ---
    min_area = config.get('min_component_area', 5)
    min_peak_ratio = config.get('min_peak_ratio', 1.0)
    scale_support = kwargs.get('scale_support', None)

    binary = _filter_components(
        binary,
        anomaly_map,
        min_area=min_area,
        min_peak_ratio=min_peak_ratio,
        threshold=threshold,
        scale_support=scale_support,
    )

    return binary


def _filter_components(
    binary: np.ndarray,
    anomaly_map: np.ndarray,
    min_area: int,
    min_peak_ratio: float,
    threshold: float,
    scale_support: np.ndarray | None,
) -> np.ndarray:
    """Filter connected components by confidence.

    A component is kept if:
      - area >= min_area
      - peak score >= min_peak_ratio * threshold (peak must exceed threshold)
      - (optional) mean scale support >= 1 (at least one scale supports it)

    Args:
        binary:        (H, W) uint8 {0, 255}
        anomaly_map:    (H, W) float continuous scores
        min_area:       minimum pixel area to keep
        min_peak_ratio: peak / threshold must exceed this
        threshold:      calibrated threshold value
        scale_support:  (H, W) int, # scales supporting each pixel (optional)

    Returns:
        (H, W) uint8 filtered binary mask
    """
    # Label connected components
    binary_bool = binary > 0
    labeled, num_components = ndimage.label(binary_bool)

    if num_components == 0:
        return binary

    keep = np.zeros_like(binary, dtype=np.uint8)

    for comp_id in range(1, num_components + 1):
        mask = labeled == comp_id
        area = int(mask.sum())

        # Filter 1: area
        if area < min_area:
            continue

        # Filter 2: peak score
        peak = float(anomaly_map[mask].max())
        if peak < min_peak_ratio * threshold:
            continue

        # Filter 3: scale support (optional)
        if scale_support is not None:
            mean_support = float(scale_support[mask].mean())
            if mean_support < 1.0:
                continue

        keep[mask] = 255

    return keep