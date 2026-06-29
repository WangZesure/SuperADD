"""kNN deduplication subsampling (SuperADD baseline).

Wraps SuperADD's subsampling_distance_based_fast which iteratively removes
near-duplicate features based on kNN distance thresholds.
"""

import numpy as np
from industrial.nearest_neighbor import subsampling_distance_based_fast


def subsample(
    features: np.ndarray,
    target_size: int,
    device: str = 'cuda',
    **kwargs,
) -> np.ndarray:
    """Subsample features by kNN-based deduplication.

    Args:
        features:     (N, D) float32 feature vectors
        target_size:  desired memory bank size
        device:       torch device for distance computation
        **kwargs:    iterations (default 100), knn_neighbors (default 100)

    Returns:
        (M, D) float32 subsampled features
    """
    return subsampling_distance_based_fast(
        features,
        target_size,
        device,
        iterations=kwargs.get('iterations', 100),
        normalize=kwargs.get('normalize', False),
        knn_neighbors=kwargs.get('knn_neighbors', 100),
    )