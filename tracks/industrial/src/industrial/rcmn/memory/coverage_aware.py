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
            query_batch (int): query batch size for kNN, default 2048
            key_chunk (int): key chunk size for chunked distance, default 8192
            stability_weights (np.ndarray): optional (N,) per-feature stability
                weights in [0,1]. If provided, multiplies rarity score.

    Returns:
        (M, D) float32 subsampled features
    """
    alpha = kwargs.get('alpha', 1.0)
    beta = kwargs.get('beta', 1.0)
    knn_k = kwargs.get('knn_neighbors', 50)
    query_batch = kwargs.get('query_batch', 2048)
    key_chunk = kwargs.get('key_chunk', 8192)
    stability_weights = kwargs.get('stability_weights', None)

    N = len(features)
    if N <= target_size:
        return features

    feats_gpu = torch.from_numpy(features).to(device)

    # --- Step 1: Rarity via kNN density ---
    # Mean distance to k nearest neighbors = density estimate.
    # High mean distance = low density = rare = should be preserved.
    rarity = _compute_knn_density_chunked(
        feats_gpu, knn_k, query_batch, key_chunk, device
    )

    # Normalize rarity to [0, 1]
    rarity = rarity / (rarity.max() + 1e-8)

    # Apply optional stability weighting
    if stability_weights is not None:
        rarity = rarity * torch.from_numpy(stability_weights).to(device)

    # --- Step 2: Greedy selection with diversity ---
    selected_indices = _greedy_coverage_select(
        feats_gpu, rarity, target_size, alpha, beta, device, key_chunk
    )

    selected = features[selected_indices]
    return selected


def _compute_knn_density_chunked(
    feats: torch.Tensor,
    k: int,
    query_batch: int,
    key_chunk: int,
    device: str,
) -> torch.Tensor:
    """Compute mean kNN distance for each feature (rarity proxy).

    Uses double chunking (query + key) to avoid OOM on large feature sets.
    For each query batch, computes distances to key chunks and tracks
    the k smallest distances seen so far.

    Returns:
        (N,) tensor, higher value = rarer (more isolated)
    """
    N, D = feats.shape
    densities = torch.zeros(N, device=device)

    for q_start in range(0, N, query_batch):
        q_end = min(q_start + query_batch, N)
        query = feats[q_start:q_end]  # (B, D)

        # Track top-k smallest distances for this query batch
        # Initialize with infinity
        topk_dists = torch.full((q_end - q_start, k + 1), float('inf'), device=device)

        for k_start in range(0, N, key_chunk):
            k_end = min(k_start + key_chunk, N)
            key = feats[k_start:k_end]  # (C, D)

            # Compute pairwise distances: (B, C)
            dists = torch.cdist(query, key, compute_mode='use_mm_for_euclid_dist')

            # Merge with current top-k: concatenate and re-select top k+1
            merged = torch.cat([topk_dists, dists], dim=1)  # (B, k+1+C)
            topk_dists, _ = torch.topk(merged, k=k + 1, dim=1, largest=False)

        # Exclude self-distance (0), take mean of remaining k
        # Sort to ensure self (distance 0) is at position 0
        topk_dists, _ = torch.sort(topk_dists, dim=1)
        mean_dist = topk_dists[:, 1:].mean(dim=-1)
        densities[q_start:q_end] = mean_dist

    return densities


def _greedy_coverage_select(
    feats: torch.Tensor,
    rarity: torch.Tensor,
    target_size: int,
    alpha: float,
    beta: float,
    device: str,
    key_chunk: int = 8192,
) -> np.ndarray:
    """Greedy selection balancing rarity and diversity.

    Iteratively selects the feature with highest:
        score = alpha * rarity + beta * min_dist_to_selected

    First pick: highest rarity (no diversity signal yet).
    Subsequent: highest combined score.

    Distance updates are chunked to avoid OOM on large feature sets.
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
            max_d = min_dist.max().item()
            norm_dist = min_dist / (max_d + 1e-8)
            score = alpha * rarity + beta * norm_dist
            score = score.masked_fill(selected_mask, -float('inf'))
            idx = torch.argmax(score).item()

        selected.append(idx)
        selected_mask[idx] = True

        # Update min distances: dist from all points to the newly selected
        # Chunk over all N points to avoid OOM
        selected_feat = feats[idx:idx + 1]  # (1, D)
        for k_start in range(0, N, key_chunk):
            k_end = min(k_start + key_chunk, N)
            chunk = feats[k_start:k_end]
            new_dists = torch.cdist(
                chunk, selected_feat, compute_mode='use_mm_for_euclid_dist'
            ).squeeze(-1)
            min_dist[k_start:k_end] = torch.minimum(
                min_dist[k_start:k_end], new_dists
            )

    return np.array(selected)