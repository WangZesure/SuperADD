"""Percentile threshold + morphology (SuperADD baseline).

Wraps SuperADD's PostProcessing: multi-oriented closing, hole filling, erosion.
The threshold is pre-computed from normal validation maps during training.
"""

import numpy as np
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
    """Binarize anomaly map with fixed threshold + morphology.

    Args:
        anomaly_map:  (H, W) continuous anomaly scores
        normal_stats: {'threshold': float}
        config:       {closing_radius, closing_angles, closing_lower_threshold, binary_erosion}
        **kwargs:     unused

    Returns:
        (H, W) uint8 binary mask with values {0, 255}
    """
    threshold = normal_stats['threshold']

    binary = multi_oriented_closing(
        anomaly_map,
        threshold,
        config['closing_radius'],
        config['closing_angles'],
        config['closing_lower_threshold'],
    )
    binary = fill_closed_regions(binary)
    binary = erosion_on_binary_maps(binary, config['binary_erosion'])
    return binary