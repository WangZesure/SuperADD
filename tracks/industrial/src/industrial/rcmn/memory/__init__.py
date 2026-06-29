"""Memory construction strategies.

A memory strategy subsamples normal feature vectors into a compact memory bank.
Interface (method-agnostic):

    subsample(features: np.ndarray, target_size: int, device: str, **kwargs) -> np.ndarray

Args:
    features:     (N, D) float32 normal feature vectors
    target_size:  desired memory bank size M (M <= N)
    device:       'cuda' | 'cpu'
    **kwargs:     strategy-specific params (iterations, knn_neighbors, ...)

Returns:
    subsampled features: (M, D) float32

To add a new strategy:
    1. Create a new module in this directory (e.g. coverage_aware.py)
    2. Implement the `subsample` function with the above signature
    3. Register it here in MEMORY_STRATEGIES
"""

from . import knn_dedup, coverage_aware

MEMORY_STRATEGIES = {
    "knn_dedup": knn_dedup.subsample,
    "coverage_aware": coverage_aware.subsample,
}