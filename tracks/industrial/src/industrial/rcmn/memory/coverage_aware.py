"""Coverage-aware memory construction (CMC).

Unlike kNN dedup (which only removes near-duplicates), CMC actively ensures
the memory bank covers the full diversity of normal patterns, including
rare but legitimate variations (lighting, position, texture).

Selection score per candidate feature:
    score(f) = alpha * rarity(f) + beta * diversity(f)

- rarity(f):   inverse kNN density — sparse regions should be preserved
- diversity(f): distance to nearest already-selected feature — encourages
                spread across the feature manifold

Greedy selection: iteratively pick the highest-scoring candidate until
target_size is reached. This is equivalent to a density-weighted
farthest-point sampling.

Stability cue (augmentation-invariance) is optional and requires paired
features; left for future extension via the `stability_weights` kwarg.
"""

import numpy as np
import torch
from industrial.nearest_neighbor import nearest_neighbors


def subsample(
    features: np.ndarray,
    target_size: int,
    device: str = 'cuda',
    **kwargs,
) -> np.ndarray:
    """Coverage-aware subsampling of normal feature vectors.

    Args:
        features:     (N, D) float32 feature vectors
        target_size:  desired memory bank size M
        device:       torch device
        **kwargs:
            alpha (float): rarity weight, default 1.0
            beta (float):  diversity weight, default 1.0
            knn_neighbors (int): k for density estimation, default 50
            batch_size (int): batch for distance computation, default 50000
            stability_weights (np.ndarray): optional (N,) per-feature stability
                weights in [0,1]. If provided, multiplies rarity score.

    Returns:
        (M, D) float32 subsampled features
    """
    alpha = kwargs.get('alpha', 1.0)
    beta = kwargs.get('beta', 1.0)
    knn_k = kwargs.get('knn_neighbors', 50)
    batch_size = kwargs.get('batch_size', 50000)
    stability_weights = kwargs.get('stability_weights', None)

    N = len(features)
    if N <= target_size:
        return features

    feats_gpu = torch.from_numpy(features).to(device)

    # --- Step 1: Rarity via kNN density ---
    # Mean distance to k nearest neighbors = density estimate.
    # High mean distance = low density = rare = should be preserved.
    rarity = _compute_knn_density(feats_gpu, knn_k, batch_size, device)

    # Normalize rarity to [0, 1]
    rarity = rarity / (rarity.max() + 1e-8)

    # Apply optional stability weighting
    if stability_weights is not None:
        rarity = rarity * torch.from_numpy(stability_weights).to(device)

    # --- Step 2: Greedy selection with diversity ---
    selected_indices = _greedy_coverage_select(
        feats_gpu, rarity, target_size, alpha, beta, device
    )

    selected = features[selected_indices]
    return selected


def _compute_knn_density(
    feats: torch.Tensor, k: int, batch_size: int, device: str
) -> torch.Tensor:
    """Compute mean kNN distance for each feature (rarity proxy).

    Returns:
        (N,) tensor, higher value = rarer (more isolated)
    """
    N = len(feats)
    densities = torch.zeros(N, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        query = feats[start:end]
        # kNN distances (k+1 because self is always nearest)
        dists, _ = nearest_neighbors(query, feats, knn_neighbors=k + 1, normalize=False)
        # Exclude self-distance (0), take mean of remaining k
        mean_dist = dists[:, 1:].mean(dim=-1)
        densities[start:end] = mean_dist

    return densities


def _greedy_coverage_select(
    feats: torch.Tensor,
    rarity: torch.Tensor,
    target_size: int,
    alpha: float,
    beta: float,
    device: str,
) -> np.ndarray:
    """Greedy selection balancing rarity and diversity.

    Iteratively selects the feature with highest:
        score = alpha * rarity + beta * min_dist_to_selected

    First pick: highest rarity (no diversity signal yet).
    Subsequent: highest combined score.

    Uses lazy greedy with distance updates for efficiency.
    """
    N = len(feats)
    selected = []
    selected_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # Track min distance from each unselected point to the selected set
    min_dist = torch.full((N,), float('inf'), device=device)

    for _ in range(target_size):
        if len(selected) == 0:
            # First pick: highest rarity
            idx = torch.argmax(rarity).item()
        else:
            # score = alpha * rarity + beta * normalized_diversity
            # Normalize min_dist to [0,1] range for stable weighting
            max_d = min_dist.max().item()
            norm_dist = min_dist / (max_d + 1e-8)
            score = alpha * rarity + beta * norm_dist
            # Mask out already selected
            score = score.masked_fill(selected_mask, -float('inf'))
            idx = torch.argmax(score).item()

        selected.append(idx)
        selected_mask[idx] = True

        # Update min distances: dist from all points to the newly selected
        new_dists = torch.cdist(
            feats, feats[idx:idx + 1], compute_mode='use_mm_for_euclid_dist'
        ).squeeze(-1)
        min_dist = torch.minimum(min_dist, new_dists)

    return np.array(selected)